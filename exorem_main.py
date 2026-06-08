"""
ExoREM — 1-D radiative-convective equilibrium model.

Entry point: :func:`run_exorem`.

References
----------
- Baudino et al. 2015  https://doi.org/10.1051/0004-6361/201526332
- Baudino et al. 2017  https://doi.org/10.3847/1538-4357/aa95be
- Charnay et al. 2018  https://doi.org/10.3847/1538-4357/aaac7d
- Blain et al. 2020    https://doi.org/10.1051/0004-6361/202039072

Mirrors the Fortran main program and its internal subroutines (exorem.f90).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .chemistry import (
    calculate_chemistry, GASES_NAMES, CONDENSATE_NAMES,
    N_GASES, N_CONDENSATES,
    calculate_gases_molar_mass, gas_id,
    h2o_saturation_pressure, nh3_saturation_pressure,
)
from .cloud_mixing import calculate_cloud_mixing, calculate_cloud_mixing2
from .interface import (
    read_exorem_input_parameters, build_stellar_irradiance,
    write_spectrum, write_temperature_profile, write_vmr_profile,
    write_hdf5_output, write_hdf5_output_fortran,
)
from .math_utils import interp_ex_0d, matinv, chi2_reduced
from .objects import (
    Atmosphere, Cloud, ExoremRetrieval, LightSource,
    Species, Spectrometrics, Target,
)
from .optics import get_refractive_index, rayleigh_scattering_coefficient
from .physics import (
    CST_R, CST_SIGMA, PI,
    planck_function, spherical_black_body_spectral_radiance,
)
from .radiative_transfer import calculate_radiative_transfer
from . import radiative_transfer as _rt_module
from .transit_spectrum import calculate_transit_spectrum

# ---------------------------------------------------------------------------
# Internal log(1 bar)  – reference altitude anchor
# ---------------------------------------------------------------------------
_A_1BAR = math.log(1e3)   # 1 bar = 1000 mbar


# ===========================================================================
# Profiling helper
# ===========================================================================

class _PhaseTimer:
    """
    Lightweight wall-clock profiler for the main iteration loop.

    Usage::

        t = _PhaseTimer()
        t.tick("adiabat")            # opens phase 'adiabat'
        ... work ...
        t.tick("rt_clear")           # closes 'adiabat', opens 'rt_clear'
        ... work ...
        t.tick("retrieval")          # closes 'rt_clear', opens 'retrieval'
        ...
        t.end()                      # closes the last open phase

    Then:
        t.write_csv(path)            # per-iteration timings
        t.print_summary()            # totals sorted by time

    Uses :func:`time.perf_counter` for high resolution (better than
    ``time.time()`` for sub-second intervals).  Records as many calls per
    phase as it sees, so a phase that runs every iteration accumulates
    len(self.phases[name]) == n_iterations entries.
    """

    __slots__ = ("phases", "_t_last", "_name_last", "_iter_marks")

    def __init__(self) -> None:
        self.phases: dict[str, list[float]] = {}
        self._t_last: float | None = None
        self._name_last: str | None = None
        # Records the iteration boundary so write_csv() can produce
        # one-row-per-iteration output.
        self._iter_marks: list[int] = []

    def tick(self, name: str | None) -> None:
        """Close the current open phase (if any) and open *name*.

        Pass ``name=None`` (or call :meth:`end`) to close the current
        phase without opening another.
        """
        now = time.perf_counter()
        if self._t_last is not None and self._name_last is not None:
            self.phases.setdefault(self._name_last, []).append(
                now - self._t_last)
        self._t_last = now
        self._name_last = name

    def end(self) -> None:
        """Close the current phase without opening a new one."""
        self.tick(None)

    def mark_iteration(self) -> None:
        """Record that an iteration boundary has been crossed (used by
        write_csv to align rows with iterations)."""
        # The mark is the *count* of recorded ticks so far for the
        # most-populous phase, which equals the iteration index.
        if self.phases:
            self._iter_marks.append(max(len(t) for t in self.phases.values()))
        else:
            self._iter_marks.append(0)

    def summary(self) -> dict[str, tuple[float, float, int]]:
        """Return ``{phase: (total_s, mean_s, n_calls)}``."""
        return {name: (sum(t), sum(t) / len(t) if t else 0.0, len(t))
                for name, t in self.phases.items()}

    def print_summary(self) -> None:
        """Print phase totals sorted by total time."""
        if not self.phases:
            return
        summary = self.summary()
        items = sorted(summary.items(), key=lambda kv: -kv[1][0])
        total = sum(s[0] for s in summary.values())
        print(f"\n==================== Phase timing summary "
              f"({total:.1f}s total) ====================")
        print(f"  {'phase':<32s} {'total_s':>10s} {'mean_s':>10s} "
              f"{'n_calls':>8s} {'%':>6s}")
        print(f"  {'-' * 32} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 6}")
        for name, (tot, mean, n) in items:
            pct = 100.0 * tot / total if total > 0 else 0.0
            print(f"  {name:<32s} {tot:>10.3f} {mean:>10.4f} "
                  f"{n:>8d} {pct:>5.1f}%")
        print(f"  {'-' * 32} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 6}")
        print(f"  {'TOTAL':<32s} {total:>10.3f}\n")

    def write_csv(self, path: str | Path) -> None:
        """Write per-iteration timings as CSV.  One row per iteration,
        one column per phase.  Phases that didn't run on a given
        iteration are blank in that row."""
        if not self.phases:
            return
        all_phases = list(self.phases.keys())
        max_iters = max(len(t) for t in self.phases.values())
        with open(path, "w") as fh:
            fh.write("iteration," + ",".join(all_phases) + "\n")
            for it in range(max_iters):
                row = [str(it)]
                for p in all_phases:
                    if it < len(self.phases[p]):
                        row.append(f"{self.phases[p][it]:.6f}")
                    else:
                        row.append("")
                fh.write(",".join(row) + "\n")


# ===========================================================================
# Public entry point
# ===========================================================================


def run_exorem(input_file: str | Path) -> dict:
    """
    Run the full ExoREM radiative-convective equilibrium model.

    Parameters
    ----------
    input_file : path to the Exorem namelist input file

    Returns
    -------
    dict with keys:
        - wavenumbers          (n_wavenumbers,)
        - spectral_radiosity   (n_levels, n_wavenumbers)
        - pressures            (n_levels,)
        - temperatures         (n_levels,)
        - flux                 (n_levels, n_wavenumbers)
        - spectral_radius      (n_wavenumbers,)  if transmission spectrum computed
        - species_vmr_layers   (n_layers, n_species)
        - (and more)
    """
    t0 = time.time()
    timer = _PhaseTimer()
    timer.tick("init")          # ← starts measuring _init_exorem

    # ------------------------------------------------------------------
    # 1. Initialise
    # ------------------------------------------------------------------
    state = _init_exorem(input_file)

    # Unpack working variables
    atm         = state["atm"]
    target      = state["target"]
    light       = state["light"]
    spec        = state["spec"]
    spectrometrics = state["spectrometrics"]
    cloud_obj   = state["cloud_obj"]
    retrieval   = state["retrieval"]
    opts        = state["opts"]

    n_levels    = atm.n_levels
    n_layers    = atm.n_layers
    n_wavenumbers = spectrometrics.n_wavenumbers
    wavenumbers = spectrometrics.wavenumbers
    wavenumber_step = spectrometrics.wavenumber_step
    n_species   = spec.n_species
    n_clouds    = cloud_obj.n_clouds

    # Working arrays (all lowercase = local loop variables)
    gases_vmr         = state["gases_vmr"]           # (N_GASES, n_layers)
    species_vmr_layers= state["species_vmr_layers"]  # (n_layers, n_species)
    cloud_vmr         = state["cloud_vmr"]           # (n_clouds, n_layers)
    gases_molar_mass  = state["gases_molar_mass"]
    elements_in_gases = state["elements_in_gases"]   # (N_GASES, N_ELEMENTS)
    gases_c_p         = state["gases_c_p"]
    temperatures_thermo = state["temperatures_thermo"]
    gases_delta_g     = state["gases_delta_g"]
    condensates_delta_g = state["condensates_delta_g"]
    h2_h2_cia         = state["h2_h2_cia"]
    h2_he_cia         = state["h2_he_cia"]
    h2o_n2_cia        = state["h2o_n2_cia"]
    h2o_h2o_cia       = state["h2o_h2o_cia"]
    kcoeff_tables     = state["kcoeff_tables"]

    # Retrieval / Jacobian matrices
    matrix_s  = _init_s_matrix(atm.pressures, retrieval)
    matrix_t  = np.zeros((n_levels, n_levels))
    matrix_t_cloud = np.zeros((n_levels, n_levels))

    # Make the outputs directory available to the retrieval function for
    # writing diagnostic CSVs (retrieval_summary.csv, retrieval_per_layer.csv).
    retrieval._debug_path = opts.get("path_outputs", None)

    n_retrieved = retrieval.retrieval_level_bottom - retrieval.retrieval_level_top + 1
    rad_noise   = state["rad_noise"]
    rad_diff    = np.zeros(n_retrieved)

    # Convergence bookkeeping
    retrieval_converged = False
    chi2_0 = np.finfo(float).max
    chi2_1 = np.finfo(float).max

    # Internal radiosity target  (sigma * T_int^4 * 1e3  erg s⁻¹ cm⁻²)
    radiosity_internal_target = np.full(
        n_levels,
        CST_SIGMA * target.target_internal_temperature**4 * 1e3,
    )

    # Fluxes / optical depths
    tau           = np.zeros((n_levels, n_wavenumbers, state["ng_max"]))
    tau_rayleigh  = np.zeros((n_levels, n_wavenumbers))
    tau_clear     = np.zeros_like(tau)
    tau_rayleigh_clear = np.zeros_like(tau_rayleigh)
    flux          = np.zeros((n_levels, n_wavenumbers))
    flux_clear    = np.zeros_like(flux)
    flux_cloud    = np.zeros_like(flux)
    flux_conv     = np.zeros(n_levels)
    spectral_radiosity       = np.zeros_like(flux)
    spectral_radiosity_clear = np.zeros_like(flux)
    spectral_radiosity_cloud = np.zeros_like(flux)
    radiosity_internal       = np.zeros(n_levels)
    radiosity_internal_cloud = np.zeros(n_levels)

    # Condensation / cloud bookkeeping
    p_c_condensates    = np.zeros(N_CONDENSATES)
    vmr_sat_condensates= np.zeros((N_CONDENSATES, n_layers))
    vmr_c_condensates  = np.zeros(N_CONDENSATES)
    layer_condensates  = np.zeros(N_CONDENSATES, dtype=int)
    tau_cloud_out      = np.zeros((n_clouds, n_layers)) if n_clouds > 0 else np.zeros((1, n_layers))

    # dt / temperature correction
    dt       = np.zeros(n_levels)
    dt_conv  = np.zeros(n_levels)

    print(f"\nExoREM {state.get('version', '')}\n____\n")

    # r31: intra-RT profiling collection.  Each iteration we reset the dict
    # in the radiative_transfer module, call RT, then append the per-phase
    # totals here as one row.
    rt_phase_history: list[dict] = []

    # ------------------------------------------------------------------
    # 2. Main iteration loop
    # ------------------------------------------------------------------
    from pathlib import Path as _Path
    _init_out = _Path(opts.get("path_outputs", ".")) / "python_init.csv"
    with open(_init_out, "w") as _fh:
        _fh.write("iteration,level,pressure_Pa,temperature_K\n")
        for _k in range(atm.n_levels):
            _fh.write(f"0,{_k},{atm.pressures[_k]:.6e},"
                      f"{atm.temperatures[_k]:.4f}\n")

    for iteration in range(retrieval.n_iterations + 1):
        print(f"\nIteration {iteration}\n____")
        timer.tick("adiabat_pre")

        # --- Adiabatic projection (Fortran-faithful: ONCE at iter 15) ---
        # Matches Fortran exorem.f90 line 161:
        #     if (iter == n_non_adiabatic_iterations) call init_adiabat()
        # The first n_non_adiabatic_iterations steps run pure-radiative so the
        # photosphere/radiative solution can establish itself; then, once, the
        # deep super-adiabatic zone is projected onto grad_ad + DGRAD anchored
        # at the radiative-convective boundary (see _init_adiabat).  From then
        # on the deep zone is governed by the retrieval + the convective term
        # (_add_convective_term), whose conv_add closure lets it settle a few
        # percent super-adiabatic and carry the internal flux.
        #
        # Running this EVERY iteration (the previous broken behaviour) forces
        # gr = grad_ad exactly each step, which zeroes the convective excess,
        # which zeroes conv_add and the convective matrix_t coupling — the deep
        # energy balance can then never close and T_eff collapses.
        if retrieval.n_iterations > 0 and iteration == retrieval.n_non_adiabatic_iterations:
            _init_adiabat(atm, target, spec, gases_vmr, gases_c_p,
                          temperatures_thermo, radiosity_internal_target,
                          verbose=True)

        # --- Check which clouds have actually condensed ---
        any_cloud_condensed = _check_cloud_condensation(
            cloud_vmr, n_clouds)

        timer.tick("rt_clear")
        # r31: initialise the RT-internal profile dict for THIS iteration
        _rt_module.RT_PROFILE = {}
        # --- Clear-sky radiative transfer ---
        if n_clouds > 0 and cloud_obj.cloud_fraction >= 1.0 - 1e-12 and any_cloud_condensed:
            if cloud_obj.cloud_fraction > 1.0:
                raise RuntimeError(f"Cloud fraction {cloud_obj.cloud_fraction:.2f} > 1")
            radiosity_internal[:] = 0.0
            matrix_t[:]           = 0.0
            flux[:]               = 0.0
            spectral_radiosity[:] = 0.0
            tau_clear[:]          = 0.0
            tau_rayleigh_clear[:] = 0.0
        else:
            (tau, tau_rayleigh, tau_cloud_out,
             radiosity_internal, matrix_t, flux,
             spectral_radiosity) = _do_radiative_transfer(
                state, gases_vmr, cloud_vmr, 0, atm, spec, cloud_obj,
                spectrometrics, light, kcoeff_tables,
                h2_h2_cia, h2_he_cia, h2o_n2_cia, h2o_h2o_cia,
                tau, tau_rayleigh)
            tau_clear[:]          = tau.copy()
            tau_rayleigh_clear[:] = tau_rayleigh.copy()

        # r31: snapshot the per-phase RT timings (will overwrite if cloudy
        # RT also runs below, then sum back at end of iter).
        rt_phase_history.append(dict(_rt_module.RT_PROFILE or {}))

        flux_clear[:] = flux.copy()
        flux_cloud[:] = 0.0
        spectral_radiosity_clear[:] = spectral_radiosity.copy()
        spectral_radiosity_cloud[:] = 0.0

        timer.tick("rt_cloud")
        # --- Cloudy radiative transfer ---
        if n_clouds > 0 and any_cloud_condensed:
            (tau, tau_rayleigh, tau_cloud_out,
             radiosity_internal_cloud, matrix_t_cloud, flux_cloud,
             spectral_radiosity_cloud) = _do_radiative_transfer(
                state, gases_vmr, cloud_vmr, n_clouds, atm, spec, cloud_obj,
                spectrometrics, light, kcoeff_tables,
                h2_h2_cia, h2_he_cia, h2o_n2_cia, h2o_h2o_cia,
                tau, tau_rayleigh)

            cf = cloud_obj.cloud_fraction
            spectral_radiosity[:] = (
                (1.0 - cf) * spectral_radiosity_clear
                + cf * spectral_radiosity_cloud
            )
            radiosity_internal[:] = (
                (1.0 - cf) * radiosity_internal
                + cf * radiosity_internal_cloud
            )
            matrix_t[:] = (
                (1.0 - cf) * matrix_t
                + cf * matrix_t_cloud
            )
            flux[:] = (
                (1.0 - cf) * flux_clear
                + cf * flux_cloud
            )

        timer.tick("convergence_check")
        # --- Convergence check ---
        solution_deviation = abs(
            1.0 - radiosity_internal[n_levels - 1]
            / radiosity_internal_target[0])

        if iteration == retrieval.n_iterations or retrieval_converged:
            _print_convergence_info(
                atm, spectrometrics, light, target,
                radiosity_internal, radiosity_internal_target,
                spectral_radiosity, retrieval_converged, retrieval)

            timer.tick("final_write")
            # Transmission spectrum (optional)
            spectral_radius    = np.zeros(n_wavenumbers)
            d_spectral_radius  = np.zeros((n_levels, n_wavenumbers))
            if opts.get("output_transmission_spectra", False):
                spectral_radius, d_spectral_radius = calculate_transit_spectrum(
                    tau, tau_rayleigh, kcoeff_tables["weights_k"],
                    atm.z, target.target_radius,
                    calculate_contribution=True)

            # Write outputs
            _write_outputs(
                opts, atm, spec, spectrometrics,
                spectral_radiosity, spectral_radius, d_spectral_radius,
                species_vmr_layers, cloud_vmr, cloud_obj,
                tau, tau_cloud_out, flux, flux_clear, flux_cloud,
                target=target, gases_vmr=gases_vmr,
                radiosity_internal=radiosity_internal,
                radiosity_internal_target=radiosity_internal_target,
                matrix_t=matrix_t, chi2_0=chi2_0,
            )
            timer.mark_iteration()
            timer.end()
            break

        timer.tick("convective_term")
        # --- Convective term (Fortran-faithful: gated to iter >= 15) ---
        # Matches Fortran exorem.f90 line 280:
        #     if (iter >= n_non_adiabatic_iterations) call add_convective_term()
        # The convective term does two things on every super-adiabatic
        # interface: (1) adds matrix_t coupling that links T-changes along the
        # adiabat, and (2) adds the convective-flux closure
        #     conv_add = 1e3 * total_flux * (gr/grad_ad - 1)^2
        # to radiosity_internal, which supplies the ~99.7% of the deep internal
        # flux that radiation cannot carry.  It is gated to start at the same
        # iteration as the adiabat projection so the radiative solution is in
        # place first.
        # Stash the iteration index so _add_convective_term can label
        # its diagnostic dump (no behavioural effect).
        retrieval._current_iteration = iteration
        if iteration >= retrieval.n_non_adiabatic_iterations:
            flux_conv, dt_conv, matrix_t = _add_convective_term(
                atm, target, spec, gases_vmr, gases_c_p,
                temperatures_thermo, radiosity_internal_target,
                radiosity_internal, matrix_t, light, spectrometrics,
                retrieval=retrieval,
            )
        else:
            flux_conv = np.zeros(n_levels)
            dt_conv   = np.zeros(n_levels)

        # --- Flux residual ---
        i0 = retrieval.retrieval_level_top - 1
        for i in range(n_retrieved):
            rad_diff[i] = (
                radiosity_internal_target[i0 + i]
                - radiosity_internal[i0 + i]
            )

        timer.tick("retrieval")
        # --- Temperature profile retrieval ---
        # Grey-Rosseland optical depth (tau = kappa_IR P / g) -> used to fade the
        # OE step out of the optically-thin, data-empty upper atmosphere above the
        # photosphere (tau=2/3).  See UPPER_ATM_INFO_WEIGHTING.
        tau_rep = (_representative_optical_depth(
                       atm.pressures,
                       target.target_internal_temperature,
                       target.target_gravity)
                   if UPPER_ATM_INFO_WEIGHTING else None)
        temperatures_before = atm.temperatures.copy()
        # Per-level Jacobian sensitivity (L1-norm of each matrix_t column) on the
        # finalized matrix the OE step solves — captured BEFORE the retrieval call
        # since the step consumes/weights matrix_t.  sens→0 = data-empty layer.
        _jac_sens = (np.abs(matrix_t).sum(axis=0)
                     if DUMP_RETRIEVAL_TRACE else None)
        retrieval_converged = _temperature_profile_retrieval(
            atm, retrieval, matrix_s, matrix_t,
            rad_diff, rad_noise, dt,
            solution_deviation, iteration,
            tau_rep=tau_rep,
        )

        timer.tick("adiabat_post")
        # --- (r31) post-retrieval adiabat re-clamp REMOVED ---
        # The previous r24/r30 code called _init_adiabat here too, after every
        # Newton step.  That was the root cause of the runaway stratospheric
        # hot bubble: forcing gr = grad_ad exactly each step killed the
        # convective Jacobian coupling in matrix_t, and the retrieval pushed
        # all unresolved residual into the upper atmosphere.  The Fortran
        # reference never does this; _add_convective_term alone handles
        # convective stability via matrix_t coupling.
        #
        # r45: enthalpy-conserving convective adjustment (NOT the old re-clamp).
        # Unlike _init_adiabat, this conserves enthalpy within each convective
        # region and targets grad_ad + DGRAD_ADJ (slightly super-adiabatic) so
        # the convective Jacobian coupling SURVIVES. It removes super-adiabatic
        # cold dips at the RCB and lets the convective zone find its own top
        # boundary each iteration. Toggle with CONVECTIVE_ADJUSTMENT.
        if CONVECTIVE_ADJUSTMENT and retrieval.n_iterations > 0:
            _adj_info = _convective_adjustment(
                atm, gases_vmr, gases_c_p, temperatures_thermo,
                atm.n_layers,
                path_outputs=getattr(retrieval, "_debug_path", None),
                iteration=iteration,
            )
            if iteration == 0:
                print(f"  convective adjustment: {_adj_info['n_regions']} region(s), "
                      f"RCB at L={_adj_info['rcb_level']} "
                      f"(P={_adj_info['rcb_pressure']:.2e} Pa), "
                      f"max ΔT={_adj_info['max_dT']:.1f} K, "
                      f"enthalpy Δ={_adj_info['enthalpy_rel_change']*100:+.2f}%")

        # r46: periodic radiative-zone jump smoother. Every SMOOTH_INTERVAL
        # iterations, dissolve big cold-trough/hot-bump discontinuities above
        # the RCB, then let the retrieval re-converge. Complements (does not
        # replace) the convective adjustment. Toggle RADIATIVE_SMOOTHING.
        if (RADIATIVE_SMOOTHING and retrieval.n_iterations > 0
                and iteration > 0 and iteration % SMOOTH_INTERVAL == 0):
            _sm_info = _smooth_radiative_jumps(
                atm, gases_vmr, gases_c_p, temperatures_thermo,
                atm.n_layers,
                path_outputs=getattr(retrieval, "_debug_path", None),
                iteration=iteration,
            )
            if _sm_info["n_smoothed"] > 0:
                print(f"  [iter {iteration}] radiative smoothing: "
                      f"{_sm_info['n_smoothed']} jump(s) above L={_sm_info['rcb_level']}, "
                      f"max jump {_sm_info['max_jump_before']:.0f}→"
                      f"{_sm_info['max_jump_after']:.0f} K"
                      + (f", apriori blend {_sm_info['apriori_blend']:.1f}"
                         if _sm_info['apriori_blend'] > 0 else ""))

        # r47: data-empty upper-atmosphere held at radiative-equilibrium T_skin.
        # The physics-correct version of what r46 (apriori-blend) was
        # approximating: layers with negligible radiative Jacobian (the OE has
        # no information about them) are relaxed toward the skin temperature
        # T_skin = T_int · 2^(-1/4) every iteration, with under-relaxation.
        # See _apply_skin_temperature notes for the assumptions.
        if RADIATIVE_EQUILIBRIUM_CAP and retrieval.n_iterations > 0:
            _rcb_for_skin = (_adj_info["rcb_level"]
                             if CONVECTIVE_ADJUSTMENT else 0)
            _skin_info = _apply_skin_temperature(
                atm, atm.n_layers, matrix_t, radiosity_internal_target,
                convective_top=_rcb_for_skin,
                path_outputs=getattr(retrieval, "_debug_path", None),
                iteration=iteration,
            )
            if iteration == 0:
                print(f"  skin temperature cap: T_skin={_skin_info['T_skin']:.1f} K, "
                      f"{_skin_info['n_affected']} data-empty layer(s) "
                      f"(L={_skin_info['first_empty_level']}-"
                      f"{_skin_info['last_empty_level']}), "
                      f"sens threshold={_skin_info['threshold_sens']:.2e}, "
                      f"max ΔT={_skin_info['max_dT']:.1f} K, α={_skin_info['alpha']:.2f}")

        timer.tick("altitude_update")
        # --- Altitude / gravity update ---
        _calculate_altitude(atm, target, gases_molar_mass, gases_vmr)

        timer.tick("kzz_update")
        # --- Kzz update ---
        atm.flux_conv = flux_conv      # persist for the HDF5 dump (diagnostics)
        if not opts.get("load_kzz_profile", False):
            _calculate_eddy_diffusion_coefficient(
                atm, target, spec, retrieval, gases_vmr,
                gases_c_p, temperatures_thermo,
                radiosity_internal_target, flux_conv,
            )

        timer.tick("chemistry")
        # --- Chemistry update ---
        use_chem = opts.get("use_chemistry", True)
        do_chem  = (use_chem
                    and (iteration % retrieval.chemistry_iteration_interval == 0
                         or iteration >= retrieval.n_iterations - retrieval.n_burn_iterations))
        if do_chem:
            _calculate_thermochemical_equilibrium(
                atm, spec, gases_vmr, species_vmr_layers,
                p_c_condensates, vmr_sat_condensates,
                vmr_c_condensates, layer_condensates,
                gases_molar_mass, gases_delta_g, condensates_delta_g,
                temperatures_thermo, gases_c_p,
                elements_in_gases,
            )
        else:
            print("No thermochemical calculations")

        timer.tick("cloud_mixing")
        # --- Cloud vertical mixing ---
        do_cloud = (n_clouds > 0
                    and (iteration % retrieval.cloud_iteration_interval == 0
                         or iteration >= retrieval.n_iterations - retrieval.n_burn_iterations))
        if do_cloud:
            _calculate_cloud_vmr(
                atm, cloud_obj, gases_molar_mass, gases_vmr,
                vmr_sat_condensates, vmr_c_condensates,
                layer_condensates, p_c_condensates,
                cloud_vmr,
            )
        else:
            print("No cloud mixing calculations")

        timer.tick("diagnostics")
        # --- Diagnostics ---
        chi2_1 = chi2_0
        chi2_0 = chi2_reduced(
            radiosity_internal[i0: i0 + n_retrieved],
            radiosity_internal_target[i0: i0 + n_retrieved],
            rad_noise,
        )

        _print_iteration_summary(
            iteration, atm, target, spectrometrics, light,
            radiosity_internal, radiosity_internal_target,
            spectral_radiosity, tau_cloud_out, cloud_vmr, cloud_obj,
            chi2_0, chi2_1, n_retrieved,
        )

        # --- Diagnostic: dump T-profile + mu per iteration to a CSV ---
        # Writes one row per level to outputs/iteration_profiles.csv so the
        # full convergence trajectory can be plotted afterwards.
        _dump_iteration_profile(opts.get("path_outputs", "."), iteration,
                                atm, gases_vmr, gases_molar_mass)

        # --- Diagnostic: dump per-species VMRs (the chemistry is the
        # current suspect, so we need the full mixing ratio table). ---
        _dump_iteration_vmrs(opts.get("path_outputs", "."), iteration,
                              atm, gases_vmr, gases_molar_mass, spec)

        # --- Diagnostic: per-iteration T + Jacobian sensitivity + net flux ---
        # (retrieval_trace.csv) — the upper-atmosphere over-cooling diagnostic.
        _dump_retrieval_trace(opts.get("path_outputs", "."), iteration,
                              atm, _jac_sens, radiosity_internal,
                              float(radiosity_internal_target[0]))

        timer.mark_iteration()

    tf = time.time()
    print(f"\nDone in {tf - t0:.3f} s")

    # ------------------------------------------------------------------
    # 3. Phase-timing summary + CSV log
    # ------------------------------------------------------------------
    timer.end()   # idempotent if already ended
    timer.print_summary()
    if DUMP_DIAGNOSTICS:
        timing_csv = Path(opts.get("path_outputs", ".")) / "iteration_timings.csv"
        try:
            timer.write_csv(timing_csv)
            print(f"  Per-iteration timings written to: {timing_csv}")
        except Exception as err:
            print(f"  WARNING: failed to write timing CSV ({err})")

    # r31: RT-internal per-phase breakdown
    if rt_phase_history:
        rt_phases = sorted({k for d in rt_phase_history for k in d.keys()})
        if DUMP_DIAGNOSTICS:
            rt_csv = Path(opts.get("path_outputs", ".")) / "rt_internal_timings.csv"
            try:
                with open(rt_csv, "w") as fh:
                    fh.write("iteration," + ",".join(rt_phases) + "\n")
                    for it, d in enumerate(rt_phase_history):
                        row = [str(it)] + [f"{d.get(p, 0.0):.6f}" for p in rt_phases]
                        fh.write(",".join(row) + "\n")
                print(f"  RT-internal phase timings: {rt_csv}")
            except Exception as err:
                print(f"  WARNING: failed to write RT-internal CSV ({err})")

        totals = {p: sum(d.get(p, 0.0) for d in rt_phase_history)
                  for p in rt_phases}
        rt_total = sum(totals.values())
        if rt_total > 0:
            print(f"\n=========== RT-internal breakdown "
                  f"({rt_total:.1f} s across {len(rt_phase_history)} calls) ===========")
            print(f"  {'phase':<20s} {'total_s':>10s} {'mean_s':>10s} {'%':>6s}")
            print(f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 6}")
            for p in sorted(rt_phases, key=lambda k: -totals[k]):
                pct = 100.0 * totals[p] / rt_total
                mean = totals[p] / len(rt_phase_history)
                print(f"  {p:<20s} {totals[p]:>10.3f} {mean:>10.4f} "
                      f"{pct:>5.1f}%")
            print()

    return {
        "wavenumbers":         spectrometrics.wavenumbers,
        "spectral_radiosity":  spectral_radiosity,
        "pressures":           atm.pressures,
        "temperatures":        atm.temperatures,
        "pressures_layers":    atm.pressures_layers,
        "temperatures_layers": atm.temperatures_layers,
        "flux":                flux,
        "flux_clear":          flux_clear,
        "flux_cloud":          flux_cloud,
        "tau":                 tau,
        "tau_rayleigh":        tau_rayleigh,
        "tau_cloud":           tau_cloud_out,
        "species_vmr_layers":  species_vmr_layers,
        "cloud_vmr":           cloud_vmr,
    }


# ===========================================================================
# Initialisation
# ===========================================================================


def _dump_iteration_profile(path_outputs: str, iteration: int,
                             atm: Atmosphere, gases_vmr: np.ndarray,
                             gases_molar_mass: np.ndarray) -> None:
    """
    Append the current T-profile to ``<path_outputs>/iteration_profiles.csv``.

    File format (one row per (iteration, level)):

        iteration,level,pressure_Pa,temperature_K,mu_g_per_mol

    On iteration 0 the file is created (header included); subsequent
    iterations append.  Useful for plotting the convergence trajectory
    after the run completes.
    """
    if not DUMP_DIAGNOSTICS:
        return
    from pathlib import Path
    out = Path(path_outputs) / "iteration_profiles.csv"
    mode = "w" if iteration == 0 else "a"
    n_layers = atm.n_layers

    # Mean molar mass per layer (g/mol) — matches _calculate_altitude formula
    mu = np.zeros(atm.n_levels)
    for i in range(N_GASES):
        mu[:n_layers] += gases_vmr[i, :] * gases_molar_mass[i] * 1e3
    mu[n_layers] = mu[n_layers - 1]

    try:
        with open(out, mode) as fh:
            if iteration == 0:
                fh.write("iteration,level,pressure_Pa,temperature_K,mu_g_per_mol\n")
            for k in range(atm.n_levels):
                fh.write(f"{iteration},{k},{atm.pressures[k]:.6e},"
                         f"{atm.temperatures[k]:.4f},{mu[k]:.4f}\n")
    except Exception as e:
        print(f"  Warning: could not dump iteration profile: {e}")


def _dump_iteration_vmrs(path_outputs: str, iteration: int,
                          atm: Atmosphere, gases_vmr: np.ndarray,
                          gases_molar_mass: np.ndarray, spec) -> None:
    """
    Append per-species VMR rows to ``<path_outputs>/iteration_vmrs.csv``.

    Long format:  iteration, level, pressure_Pa, temperature_K,
                   vmr_sum, mu_g_per_mol, <species_1>, <species_2>, …

    Only the species that ExoREM actually carries (via ``spec.species_names``)
    plus the always-present H2, He, H are dumped.  ``vmr_sum`` should be
    1.0 ± 1e-6 if the chemistry normalisation is healthy; large deviations
    are immediate evidence that the chemistry update is broken.

    r32: mu_g_per_mol is now computed directly from gases_vmr * gases_molar_mass
    at dump time, not from atm.molar_masses_layers.  The previous approach read
    molar_masses_layers which is set in _calculate_altitude (before chemistry) and
    therefore lagged one iteration at iter=0, producing a spurious 6.7777 g/mol
    entry that made every iter-0 trace look like the wrong atmospheric composition.
    """
    if not DUMP_DIAGNOSTICS:
        return
    from pathlib import Path
    out = Path(path_outputs) / "iteration_vmrs.csv"
    n_layers = atm.n_layers

    from .chemistry import GASES_NAMES

    try:
        mode = "w" if iteration == 0 else "a"
        with open(out, mode) as fh:
            if iteration == 0:
                header = ["iteration", "level", "pressure_Pa",
                          "temperature_K", "vmr_sum", "mu_g_per_mol"]
                header += [str(n) for n in GASES_NAMES]
                fh.write(",".join(header) + "\n")
            for k in range(n_layers):       # layers, not levels
                vmrs = gases_vmr[:, k]
                s = float(vmrs.sum())
                # r32: compute mu directly from current gases_vmr so the
                # logged value is always consistent with the species listed in
                # the same row.  gases_molar_mass is in kg/mol; convert to g/mol.
                mu = float(np.dot(vmrs, gases_molar_mass) * 1e3) if gases_molar_mass is not None else 0.0
                row = [str(iteration), str(k),
                       f"{atm.pressures_layers[k]:.6e}",
                       f"{atm.temperatures_layers[k]:.4f}",
                       f"{s:.6e}", f"{mu:.4f}"]
                row += [f"{float(v):.6e}" for v in vmrs]
                fh.write(",".join(row) + "\n")
    except Exception as e:
        print(f"  Warning: could not dump iteration VMRs: {e}")


def _dump_retrieval_trace(path_outputs, iteration: int, atm: Atmosphere,
                          jac_sens: np.ndarray | None,
                          radiosity_internal: np.ndarray | None = None,
                          radiosity_target: float = 0.0) -> None:
    """Append per-iteration retrieval diagnostics to ``retrieval_trace.csv``.

    Long format, one row per level:

        iteration,level,pressure_Pa,temperature_K,jacobian_sens,net_flux_frac

    ``jacobian_sens`` is the L1-norm of column ``level`` of the temperature
    Jacobian ``matrix_t`` (computed in the main loop, after the convective
    coupling is added, on the matrix the optimal-estimation step actually
    solves).  It measures how much information the OE step has about that
    level's temperature: ``jacobian_sens → 0`` marks the data-empty,
    near-singular layers in the optically-thin upper atmosphere where the
    step is unconstrained.

    ``net_flux_frac`` = radiosity_internal[level] / radiosity_internal_target
    (target = σT_int⁴, constant).  In equilibrium this is ~1 at every level;
    a departure is the flux residual the retrieval is trying to null.  Pairing
    it with ``jacobian_sens`` shows whether a mismatched flux in the thin zone
    actually has any leverage on temperature — the crux of why the Python's
    non-grey radiative zone diverges from the Fortran's.

    Iteration 0 creates the file (with header); later iterations append.
    """
    if not DUMP_RETRIEVAL_TRACE:
        return
    from pathlib import Path
    out = Path(path_outputs) / "retrieval_trace.csv"
    mode = "w" if iteration == 0 else "a"
    n_levels = atm.n_levels
    sens = jac_sens if jac_sens is not None else np.zeros(n_levels)
    rad = radiosity_internal if radiosity_internal is not None else np.zeros(n_levels)
    rtgt = radiosity_target if radiosity_target else 0.0
    try:
        with open(out, mode) as fh:
            if iteration == 0:
                fh.write("iteration,level,pressure_Pa,temperature_K,"
                         "jacobian_sens,net_flux_frac\n")
            for k in range(n_levels):
                s = float(sens[k]) if k < len(sens) else 0.0
                nf = (float(rad[k]) / rtgt) if (rtgt and k < len(rad)) else 0.0
                fh.write(f"{iteration},{k},{atm.pressures[k]:.6e},"
                         f"{atm.temperatures[k]:.4f},{s:.6e},{nf:.6e}\n")
    except Exception as e:
        print(f"  Warning: could not dump retrieval trace: {e}")


def _dump_retrieval_debug(path_outputs, iteration: int, atm: Atmosphere,
                           T_old: np.ndarray, dt_proposed: np.ndarray,
                           dt_applied: np.ndarray,
                           rad_diff: np.ndarray, rad_noise: np.ndarray,
                           trace_KSK: float, cond_M: float,
                           max_relative_step: float,
                           alpha: float, beta: float,
                           i0: int, n_retrieved: int) -> None:
    """
    Append rich per-iteration diagnostics to two CSVs in *path_outputs*:

    1. ``retrieval_summary.csv`` — one row per iteration::
         iteration, trace_KSK, cond_M, max_rel_step, alpha, beta,
         rad_diff_max, rad_diff_norm, rad_noise_max

    2. ``retrieval_per_layer.csv`` — one row per (iteration, level)::
         iteration, level, pressure_Pa, T_old, dt_proposed, dt_applied,
         T_new, rad_diff (only at retrieved levels, else NaN)

    Diagnostic value of each field:
        trace_KSK    : magnitude of K Sₐ Kᵀ (large → strong information)
        cond_M       : condition number of M = KSKᵀ + Sₑ
                       (>1e10 → ill-posed, retrieval direction may be wrong)
        max_rel_step : max |dt_proposed| / T_old (sanity check for Newton)
        alpha, beta  : final damping factors applied
        rad_diff     : observation residual driving the retrieval

    Skip silently when path_outputs is None (allows internal calls without
    plumbing the path everywhere).
    """
    if not DUMP_DIAGNOSTICS or path_outputs is None:
        return
    from pathlib import Path
    p = Path(path_outputs)

    # 1. Summary
    summary = p / "retrieval_summary.csv"
    try:
        mode = "w" if iteration == 0 else "a"
        with open(summary, mode) as fh:
            if iteration == 0:
                fh.write("iteration,trace_KSK,cond_M,max_rel_step,alpha,beta,"
                         "rad_diff_max,rad_diff_norm,rad_noise_max\n")
            rd_max  = float(np.max(np.abs(rad_diff))) if rad_diff.size else 0.0
            rd_norm = float(np.linalg.norm(rad_diff)) if rad_diff.size else 0.0
            rn_max  = float(np.max(rad_noise)) if rad_noise.size else 0.0
            fh.write(f"{iteration},{trace_KSK:.6e},{cond_M:.6e},"
                     f"{max_relative_step:.6e},{alpha:.4f},{beta:.4f},"
                     f"{rd_max:.6e},{rd_norm:.6e},{rn_max:.6e}\n")
    except Exception as e:
        print(f"  Warning: could not dump retrieval_summary: {e}")

    # 2. Per-layer detail
    detail = p / "retrieval_per_layer.csv"
    try:
        mode = "w" if iteration == 0 else "a"
        with open(detail, mode) as fh:
            if iteration == 0:
                fh.write("iteration,level,pressure_Pa,T_old,"
                         "dt_proposed,dt_applied,T_new,rad_diff\n")
            for k in range(atm.n_levels):
                # rad_diff is indexed over retrieved levels (i0..i0+n_retrieved)
                if i0 <= k < i0 + n_retrieved:
                    rd = float(rad_diff[k - i0])
                    rd_str = f"{rd:.6e}"
                else:
                    rd_str = "nan"
                fh.write(f"{iteration},{k},{atm.pressures[k]:.6e},"
                         f"{T_old[k]:.4f},{dt_proposed[k]:.6e},"
                         f"{dt_applied[k]:.6e},{atm.temperatures[k]:.4f},"
                         f"{rd_str}\n")
    except Exception as e:
        print(f"  Warning: could not dump retrieval_per_layer: {e}")


# ---------------------------------------------------------------------------
# Master switch for all Python-only diagnostic output.
# When False (the default) a run writes ONLY the Fortran-layout HDF5 output,
# matching the original code's output directory exactly.  When True it also
# emits the offline debugging aids that the Fortran does not produce:
#   * per-iteration .npz dumps (matrix state, convective masks, smoothing, skin)
#   * per-iteration CSVs  (iteration_profiles.csv, iteration_vmrs.csv,
#                          retrieval_per_layer.csv)
#   * phase-timing CSVs   (iteration_timings.csv, rt_internal_timings.csv)
# Flip this to True to restore every diagnostic file in one go.
# ---------------------------------------------------------------------------
DUMP_DIAGNOSTICS = False

# Per-iteration retrieval trace (T + Jacobian sensitivity per level) written to
# retrieval_trace.csv.  Lightweight (one CSV, ~n_levels rows/iteration) and
# independent of DUMP_DIAGNOSTICS, so it can stay on during grid runs to
# diagnose the optically-thin upper-atmosphere over-cooling (the near-singular
# OE step: jacobian_sens → 0 marks the data-empty layers).
DUMP_RETRIEVAL_TRACE = True


def _dump_retrieval_matrices(
    path_outputs, iteration: int,
    matrix_t: np.ndarray, matrix_s: np.ndarray,
    K: np.ndarray, SK: np.ndarray, KSK: np.ndarray,
    M_bare: np.ndarray, M_reg: np.ndarray, M_inv: np.ndarray,
    R: np.ndarray,
    dt_proposed: np.ndarray, dt_applied: np.ndarray,
    rad_diff: np.ndarray, rad_noise: np.ndarray,
    T_old: np.ndarray, T_new: np.ndarray,
    lambda_tikhonov: float,
) -> None:
    """Dump the full per-iteration matrix state for offline diagnosis."""
    if not DUMP_DIAGNOSTICS or path_outputs is None:
        return
    from pathlib import Path
    p = Path(path_outputs)
    p.mkdir(parents=True, exist_ok=True)
    fn = p / f"retrieval_matrices_iter{iteration:03d}.npz"
    try:
        try:
            sv_M_reg = np.linalg.svd(M_reg, compute_uv=False)
        except Exception:
            sv_M_reg = np.full(M_reg.shape[0], np.nan)
        try:
            sv_R = np.linalg.svd(R, compute_uv=False)
        except Exception:
            sv_R = np.full(min(R.shape), np.nan)
        np.savez_compressed(
            fn,
            matrix_t=matrix_t, matrix_s=matrix_s,
            K=K, SK=SK, KSK=KSK,
            M_bare=M_bare, M_reg=M_reg, M_inv=M_inv, R=R,
            dt_proposed=dt_proposed, dt_applied=dt_applied,
            rad_diff=rad_diff, rad_noise=rad_noise,
            T_old=T_old, T_new=T_new,
            sv_M_reg=sv_M_reg, sv_R=sv_R,
            lambda_tikhonov=float(lambda_tikhonov),
        )
    except Exception as e:
        print(f"  Warning: could not dump retrieval matrices: {e}")


def _dump_conv_diagnostic(
    path_outputs, iteration: int,
    gr_arr: np.ndarray, grad_ad_arr: np.ndarray,
    is_conv_python: np.ndarray,
    pressures: np.ndarray, temperatures: np.ndarray,
    super_adiab_tol: float,
) -> None:
    """Save lapse rate, adiabatic gradient, and the two convective-zone
    masks (Python contiguous filter vs Fortran-like 'any super-adiabatic')."""
    if not DUMP_DIAGNOSTICS or path_outputs is None or iteration < 0:
        return
    from pathlib import Path
    p = Path(path_outputs)
    p.mkdir(parents=True, exist_ok=True)
    fn = p / f"retrieval_conv_iter{iteration:03d}.npz"
    try:
        is_conv_fortranlike = np.zeros_like(is_conv_python, dtype=bool)
        for j in range(1, len(gr_arr)):
            if grad_ad_arr[j] > 0 and gr_arr[j] > grad_ad_arr[j]:
                is_conv_fortranlike[j] = True
        np.savez_compressed(
            fn,
            gr_arr=gr_arr, grad_ad_arr=grad_ad_arr,
            is_conv_python=is_conv_python,
            is_conv_fortranlike=is_conv_fortranlike,
            pressures=pressures, temperatures=temperatures,
            super_adiab_tol=float(super_adiab_tol),
        )
    except Exception as e:
        print(f"  Warning: could not dump conv diagnostic: {e}")


def _init_exorem(input_file: str | Path) -> dict:
    """Read parameters, allocate arrays, load k-tables and CIA data."""
    from .interface import read_exorem_input_parameters, build_stellar_irradiance

    atm         = Atmosphere()
    target      = Target()
    light       = LightSource()
    spec        = Species()
    spectrometrics = Spectrometrics()
    cloud_obj   = Cloud()
    retrieval   = ExoremRetrieval()

    opts = read_exorem_input_parameters(
        input_file, atm, target, light, spec, spectrometrics, cloud_obj, retrieval)

    # Number of species / clouds from opts
    n_species = len(opts.get("species_names", []))
    n_clouds  = len(opts.get("cloud_names",   []))
    spec.n_species  = n_species
    cloud_obj.n_clouds = n_clouds
    spec.species_names = opts["species_names"]
    spec.species_at_equilibrium = np.array(
        opts.get("species_at_equilibrium", [True] * n_species), dtype=bool)

    n_levels = atm.n_levels
    n_layers = n_levels - 1
    atm.n_layers = n_layers

    # --- Pressure / temperature grid ---
    _init_atmosphere(atm, target, opts)

    # --- Wavenumber grid (auto-clamps wn_max to k-table coverage) ---
    _init_wavenumbers(
        spectrometrics,
        path_k_coefficients=opts["path_k_coefficients"],
        species_names=spec.species_names,
    )

    # --- k-coefficient tables (HDF5) ---
    kcoeff_tables = _load_k_coefficients(
        opts["path_k_coefficients"], spec.species_names, spectrometrics)

    # --- CIA ---
    h2_h2_cia, h2_he_cia, h2o_n2_cia, h2o_h2o_cia = _load_cia(
        opts, spectrometrics.wavenumbers, atm)

    # --- Thermochemical tables ---
    thermo = _load_thermochemical_tables(opts["path_thermochemical_tables"])
    gases_molar_mass, elements_in_gases = calculate_gases_molar_mass()

    # --- Rayleigh scattering coefficients ---
    rayleigh_coeffs = _init_rayleigh_scattering(
        spec.species_names, spectrometrics.wavenumbers)

    # --- Stellar irradiance ---
    # Master switch: if add_light_source is False, force the irradiance to
    # zero everywhere.  All downstream usages (RT, total-flux integration,
    # T_eq calculation) propagate zeros correctly, so this single guard is
    # all that's needed to truly disable irradiation.
    if opts.get("add_light_source", False):
        light.irradiance = build_stellar_irradiance(
            spectrometrics.wavenumbers, spectrometrics.wavenumber_step, light,
            opts["light_source_spectrum_file"])
    else:
        print("  add_light_source = False → stellar irradiance set to zero "
              "(no irradiation)")
        light.irradiance = np.zeros(spectrometrics.n_wavenumbers)

    # --- Allocation of working arrays ---
    gases_vmr          = np.zeros((N_GASES, n_layers))
    species_vmr_layers = np.zeros((n_layers, n_species))
    cloud_vmr          = np.zeros((max(n_clouds, 1), n_layers))

    # --- Seed gases_vmr with the bulk H2 / He / Z mixture from the .nml ---
    # The first ``_calculate_altitude`` call needs a sensible mean molar mass;
    # without this seed mu = 0 and every downstream scale-height divides by
    # zero on iteration 0.  Chemistry overwrites these on the first do_chem step.
    _i_h2 = gas_id("H2")
    _i_he = gas_id("He")
    if 0 <= _i_h2:
        gases_vmr[_i_h2, :] = atm.h2_vmr
    if 0 <= _i_he:
        gases_vmr[_i_he, :] = atm.he_vmr
    if atm.z_vmr > 0:
        # Distribute the metallicity-equivalent across a few heavy species
        for _name, _frac in (("H2O", 0.5), ("CH4", 0.3), ("NH3", 0.2)):
            _i = gas_id(_name)
            if 0 <= _i:
                gases_vmr[_i, :] = atm.z_vmr * _frac

    # --- Copy cloud and species opts onto the dataclass objects ---
    # ``read_exorem_input_parameters`` reads these into ``opts`` but doesn't
    # always copy them onto the corresponding dataclass attributes, leaving
    # downstream code with empty arrays.  Do it here once.
    if n_clouds > 0:
        cloud_obj.cloud_names                = list(opts.get("cloud_names", []))
        cloud_obj.sedimentation_parameter    = np.asarray(opts.get("sedimentation_parameter", []), dtype=float)
        cloud_obj.supersaturation_parameter  = np.asarray(opts.get("supersaturation_parameter", []), dtype=float)
        cloud_obj.sticking_efficiency        = np.asarray(opts.get("sticking_efficiency", []), dtype=float)
        cloud_obj.cloud_particle_density     = np.asarray(opts.get("cloud_particle_density", []), dtype=float)
        cloud_obj.reference_wavenumber       = np.asarray(opts.get("reference_wavenumber", []), dtype=float)
        # ``cloud_particle_radius`` from the .nml is a list of scalars (one
        # per cloud).  Broadcast to (n_clouds, n_layers) so per-layer code works.
        cpr = np.asarray(opts.get("cloud_particle_radius", [1e-6] * n_clouds), dtype=float)
        cloud_obj.cloud_particle_radius = np.broadcast_to(
            cpr[:, None], (n_clouds, n_layers)).copy()
        # Cloud molar mass — derive from the cloud name's matching condensate
        # in CONDENSATE_NAMES (chemistry module).
        from .chemistry import condensate_id
        from .physics import ELEMENTS_MOLAR_MASS
        cloud_obj.cloud_molar_mass = np.full(n_clouds, 18e-3)  # default H2O molar mass
        # q_ext_ref placeholder — will be overwritten if a cloud-optics file is loaded
        cloud_obj.q_ext_ref = np.zeros((n_clouds, n_layers))

    spec.cia_names = list(opts.get("cia_names", []))

    # ----------------------------------------------------------------------
    # Build elemental_h_ratio (atomic-number-indexed X/H atomic ratios).
    #
    # The .nml uses a hierarchy of flags:
    #   atmosphere.use_metallicity        — overall switch: if True, scale solar
    #                                       abundances by atmospheric metallicity
    #   species.use_atmospheric_metallicity — when True, apply the atmospheric
    #                                          scaling to LISTED elements too
    #                                          (suppresses per-element override)
    #   species.use_elements_metallicity   — when True, listed elements use
    #                                          elements_metallicity × solar
    #                                          (else they use explicit
    #                                          elements_h_ratio)
    #
    # Logic for every element X (atomic_number = Z, index i = Z-1):
    #     - X == H (i=0):  always 1.0 (it's the reference)
    #     - X in elements_names AND NOT use_atm_met:
    #         - if use_elem_met:  elements_metallicity[X] × solar[X]
    #         - else:             elements_h_ratio[X]            (explicit)
    #     - otherwise:
    #         - if use_atm_met:   atmospheric_metallicity × solar[X]
    #         - else:             atmospheric_metallicity × solar[X]   (same)
    # ----------------------------------------------------------------------
    from .physics import ELEMENTS_SYMBOL
    from .loaders import load_solar_abundances

    n_elem = len(ELEMENTS_SYMBOL)
    solar_xh = load_solar_abundances(opts.get("path_data", "./data"))
    spec.solar_h_ratio = solar_xh   # used by chemistry quench timescales (metallicity_C/N/O)

    use_metallicity   = bool(opts.get("use_metallicity", True))
    use_atm_met       = bool(opts.get("use_atmospheric_metallicity", False))
    use_elem_met      = bool(opts.get("use_elements_metallicity", True))
    atm_met           = float(getattr(atm, "metallicity", 1.0))

    listed_names         = list(opts.get("elements_names", []))
    listed_h_ratios      = list(opts.get("elements_h_ratio", []))
    listed_metallicities = list(opts.get("elements_metallicity", []))

    listed_index_to_pos: dict[int, int] = {}
    for j, name in enumerate(listed_names):
        try:
            listed_index_to_pos[ELEMENTS_SYMBOL.index(name)] = j
        except ValueError:
            print(f"  Warning: element '{name}' in elements_names not in periodic table — skipping")

    spec.elemental_h_ratio = np.zeros(n_elem)
    spec.elemental_h_ratio[0] = 1.0      # H/H ≡ 1 by definition; never scaled

    if use_metallicity:
        for i in range(1, n_elem):
            if i in listed_index_to_pos and not use_atm_met:
                j = listed_index_to_pos[i]
                if use_elem_met:
                    m = float(listed_metallicities[j]) if j < len(listed_metallicities) else 1.0
                    spec.elemental_h_ratio[i] = m * solar_xh[i]
                else:
                    spec.elemental_h_ratio[i] = (
                        float(listed_h_ratios[j]) if j < len(listed_h_ratios) else 0.0)
            else:
                spec.elemental_h_ratio[i] = atm_met * solar_xh[i]
    else:
        # use_metallicity=False: fall back to explicit elements_h_ratio for
        # listed elements only.  This branch is uncommon; warn and proceed.
        print("  Note: use_metallicity=False — using only explicit elements_h_ratio")
        for i, j in listed_index_to_pos.items():
            spec.elemental_h_ratio[i] = (
                float(listed_h_ratios[j]) if j < len(listed_h_ratios) else 0.0)

    # Loud diagnostic so the user can see what chemistry will actually use
    _show = ["H", "He", "C", "N", "O", "Na", "S", "K", "Fe", "Ti", "V"]
    print("  elemental_h_ratio (X/H by number, after metallicity logic):")
    for n in _show:
        i = ELEMENTS_SYMBOL.index(n)
        print(f"    {n:3s}  Z={i+1:2d}  X/H = {spec.elemental_h_ratio[i]:.3e}")

    ng_max = int(kcoeff_tables.get("ng_max", 1))

    # --- Retrieval noise ---
    i0 = retrieval.retrieval_level_top - 1
    i1 = retrieval.retrieval_level_bottom
    n_retrieved = i1 - i0
    rad_target = CST_SIGMA * target.target_internal_temperature**4 * 1e3

    rad_noise = np.exp(
        np.log(retrieval.retrieval_flux_error_top)
        * np.arange(n_retrieved) / max(n_retrieved - 1, 1)
        + np.log(retrieval.retrieval_flux_error_bottom)
        * (1.0 - np.arange(n_retrieved) / max(n_retrieved - 1, 1))
    ) * rad_target

    return {
        "version":       "Python-port 1.0",
        "atm":           atm,
        "target":        target,
        "light":         light,
        "spec":          spec,
        "spectrometrics": spectrometrics,
        "cloud_obj":     cloud_obj,
        "retrieval":     retrieval,
        "opts":          opts,
        "gases_vmr":     gases_vmr,
        "species_vmr_layers": species_vmr_layers,
        "cloud_vmr":     cloud_vmr,
        "gases_molar_mass": gases_molar_mass,
        "elements_in_gases": elements_in_gases,
        "gases_delta_g": thermo["gases_delta_g"],
        "condensates_delta_g": thermo["condensates_delta_g"],
        "gases_c_p":     thermo["gases_c_p"],
        "temperatures_thermo": thermo["temperatures"],
        "kcoeff_tables": kcoeff_tables,
        "rayleigh_coeffs": rayleigh_coeffs,
        "h2_h2_cia":     h2_h2_cia,
        "h2_he_cia":     h2_he_cia,
        "h2o_n2_cia":    h2o_n2_cia,
        "h2o_h2o_cia":   h2o_h2o_cia,
        "rad_noise":     rad_noise,
        "ng_max":        ng_max,
    }


def _kappa_ir_freedman_approx(T_int: float) -> float:
    """
    Approximate Rosseland-mean infrared opacity at the photosphere of an
    H₂/He-dominated atmosphere with solar composition, as a function of the
    *internal* temperature.

    This is a simplified single-power-law fit to the Freedman et al. (2008,
    `ApJS 174 504`_) Rosseland tables evaluated near the radiative-equilibrium
    photosphere (T = T_int, τ = 2/3), valid in the 200-3000 K range to roughly
    a factor of 2:

    .. math:: \\kappa_{IR}(T_{int}) \\approx 5\\times10^{-4}
              \\left(\\frac{T_{int}}{500\\,\\mathrm{K}}\\right)^{1.8}
              \\;\\mathrm{m^2\\,kg^{-1}}

    Values across the brown dwarf regime:

    ===========  ==================
    T_int (K)    κ_IR (m²/kg)
    ===========  ==================
    200          1.0 × 10⁻⁴
    500          5.0 × 10⁻⁴
    1000         1.7 × 10⁻³
    1500         3.7 × 10⁻³
    2000         6.3 × 10⁻³
    3000         1.4 × 10⁻²
    ===========  ==================

    The cooler the atmosphere, the LESS opaque it is per unit mass at the
    photosphere — counter-intuitive, but it's because the dominant
    absorbers (H₂O, CH₄, NH₃ bands) shift redward and broaden with rising
    T, increasing the integrated Rosseland mean.  Combined with
    ``P_photo = 2g/(3 κ_IR)``, this gives a cooler atmosphere a DEEPER
    photosphere — consistent with observed Y-dwarf vs. L-dwarf
    photospheric pressures.

    Parameters
    ----------
    T_int
        Internal temperature in K (the BD's σT_int⁴ flux temperature).

    Returns
    -------
    kappa_ir : float
        Estimated Rosseland-mean IR opacity in m²/kg.

    Notes
    -----
    This is a CRUDE default intended only to produce a Guillot-style
    apriori with the photosphere in approximately the right place.  If
    the spectrum or T-profile demands a different photospheric
    pressure, override via ``guillot_p_photo_bar`` (preferred) or
    ``guillot_kappa_ir`` in the namelist.

    .. _ApJS 174 504: https://ui.adsabs.harvard.edu/abs/2008ApJS..174..504F
    """
    if T_int <= 0.0:
        raise ValueError("T_int must be > 0 to estimate κ_IR")
    return 5.0e-4 * (T_int / 500.0) ** 1.8


# =====================================================================
# H2-dissociation correction to the adiabatic gradient (Fortran corr_adia)
# =====================================================================
# gas_id() is 0-indexed in the Python port (H=10, H2=11).
_I_H  = gas_id("H")
_I_H2 = gas_id("H2")
_DH0_H2_DISSOC = 230.65e3   # J/mol, H2 dissociation enthalpy (Fortran `dh0`)


def _corr_adia(t: float, h2_vmr: float, h_vmr: float) -> tuple[float, float]:
    """Faithful port of the Fortran ``corr_adia`` (exorem.f90:712-728).

    Returns ``(corr, dcpr)``, the H₂-dissociation correction to the adiabatic
    gradient.  The Fortran adiabatic gradient is

        gradiant = corr / (c_p/R + dcpr)

    versus the dissociation-free ``1 / (c_p/R)``.  Both ``corr`` (>1) and
    ``dcpr`` (>0) grow with the atomic-H abundance, so they switch on only in
    the deep, hot (T >~ 1500 K) zone where H₂ begins to dissociate.  Below that
    H is essentially all molecular, ``h_vmr -> 0`` and ``(corr, dcpr) -> (1, 0)``
    so the gradient reduces to ``1/(c_p/R)``.  In the hot deep the net effect is
    to *shallow* the adiabat (e.g. 0.245 -> 0.199 at 3300 K): ``dcpr`` dominates
    over ``corr-1`` in the ratio.

    ``h2_vmr`` is the VMR of molecular H₂ and ``h_vmr`` the VMR of atomic H,
    matching the Fortran call order ``corr_adia(t, gases_vmr(H2), gases_vmr(H))``.
    """
    h_abd = max(2.0 * h2_vmr + h_vmr, 1e-300)   # total H nuclei (Fortran max(.,tiny))
    tdqhdt = 2.0 * h_vmr * h2_vmr * _DH0_H2_DISSOC / (CST_R * t) / (h_abd + h2_vmr)
    corr = 1.0 + tdqhdt / (2.0 - h_vmr)
    dcpr = tdqhdt * _DH0_H2_DISSOC / (CST_R * t) / (1.0 - 0.5 * h_vmr)
    return corr, dcpr


def _grad_ad_h2he_approx(T: float) -> float:
    """Closed-form approximation of ``∇_ad = (∂lnT/∂lnP)_S`` for an
    H₂-He mixture as a function of temperature.

    Used by ``_make_guillot_apriori`` to integrate the deep adiabat with
    the *same* T-dependence that ``_init_adiabat`` sees through the
    proper ``gases_c_p`` tables.  If the two used different gradients,
    cumulative mismatch between the apriori adiabat and the projected
    adiabat would create a "dip" at the RCB on the first iteration.

    Values:
        * T < 200 K   : translational only, ``∇_ad = 2/5``.
        * 200-800 K   : translation + rotation excited, ``∇_ad = 2/7``.
        * 800-2000 K  : vibrational modes start activating, ``∇_ad``
          falls smoothly from 0.286 to 0.256.
        * 2000-5000 K : full vib excitation, ``∇_ad`` falls from
          0.256 to 0.216.
        * T > 5000 K  : dissociation regime, fixed at 0.216 (a stand-in;
          the real value drops sharply but our atmospheres rarely reach
          that depth).

    Matches Saumon-Chabrier interior models to ≲2 % over 500-3000 K.
    """
    if T < 200.0:
        return 2.0 / 5.0
    if T < 800.0:
        return 2.0 / 7.0
    if T < 2000.0:
        return 0.286 - 0.030 * (T - 800.0) / 1200.0
    if T < 5000.0:
        return 0.256 - 0.040 * (T - 2000.0) / 3000.0
    return 0.216


def _make_guillot_apriori(
    pressures: np.ndarray,
    T_int: float,
    T_irr: float = 0.0,
    gravity: float = 25.0,
    kappa_ir: float = 1.0e-2,
    gamma_v: float = 0.4,
    grad_ad: float = 0.286,
    mu_star: float = 1.0 / math.sqrt(3.0),
) -> np.ndarray:
    """
    Build a self-consistent apriori T-profile as ``max(T_Guillot, T_adiabat)``.

    *Why* — exorem's retrieval has no autonomous mechanism to find the
    radiative-convective boundary.  It can adjust T along each layer's
    radiative gradient (via the radiative Jacobian) and along the adiabat
    in regions where ``_add_convective_term`` has matrix coupling (which
    requires interfaces to be super-adiabatic), but it cannot decide
    "the RCB sits at 0.3 bar, move it."  It can only fine-tune around
    whatever RCB the apriori already implies.  So the apriori must
    contain the RCB approximately in the right place.

    The Guillot (2010) eq. 49 grey analytic profile gives a smooth
    radiative-equilibrium T(p) for a given internal temperature and
    optional irradiation.  Its asymptotic deep-zone gradient is
    ``∇ → 1/4`` (slightly sub-adiabatic).  A *T-dependent* adiabat
    anchored at the photosphere (τ = 2/3, where T_Guillot = T_int by
    construction) crosses the Guillot curve once.  Taking the pointwise
    maximum gives:

    * **Above the crossing** (low τ): Guillot dominates — radiative.
    * **Below the crossing** (high τ): the adiabat dominates — convective.

    The crossing IS the radiative-convective boundary by construction.

    Parameters
    ----------
    pressures
        Level pressures in Pa.  Must be either monotonically increasing
        or decreasing; the function handles both.
    T_int
        Internal temperature in K — the BD/giant-planet "effective"
        temperature that carries the interior heat flux σT_int⁴.
    T_irr
        Irradiation temperature in K.  Set to 0 for non-irradiated
        objects (the second term of Guillot's eq. 49 drops out).
    gravity
        Surface gravity in m/s².
    kappa_ir
        Rosseland-mean thermal-IR opacity in m²/kg.  Sets the optical-
        depth/pressure relation via ``τ = κ_IR P/g``.  Default 1e-2 is
        reasonable for H₂/He giant-planet atmospheres around 1000 K;
        adjust if the resulting photospheric pressure is implausible.
    gamma_v
        Ratio ``κ_v / κ_IR`` of visible to thermal opacity.  Only matters
        when T_irr > 0.  Default 0.4 (typical of warm-Jupiter
        atmospheres).
    grad_ad
        *Fallback* adiabatic gradient ``dlnT/dlnP`` used only for the
        T-independent Guillot radiative part (and as a sanity check).
        Default 0.286 (rotation-excited H₂/He, the value relevant near
        a brown dwarf's photosphere).  The deep adiabat integration
        uses ``_grad_ad_h2he_approx(T)`` instead — see notes below.
    mu_star
        Cosine of stellar zenith angle (only used when T_irr > 0).
        Default ``1/√3`` for global averaging (Guillot 2010, Eq. 50).

    Returns
    -------
    T : ndarray, shape ``pressures.shape``
        ``max(T_Guillot(p), T_adiabat(p))`` in K, with the adiabat
        integrated with a T-dependent ``∇_ad``.

    Notes
    -----
    The photospheric pressure ``P_photo = 2g/(3 κ_IR)`` falls out of the
    τ = κ_IR P / g relation evaluated at τ = 2/3.  Adjusting ``kappa_ir``
    or ``gravity`` shifts P_photo and therefore the RCB depth.

    The adiabat is integrated layer-by-layer with
    ``_grad_ad_h2he_approx(T_local)`` to match the T-dependent gradient
    that ``_init_adiabat`` sees through the proper ``gases_c_p`` tables.
    Without this, a constant ∇_ad in the apriori would disagree with the
    chemistry-aware ∇_ad used by the projection at the (cooler)
    photospheric region, creating a cumulative discontinuity at the RCB.
    """
    tau = kappa_ir * pressures / gravity

    # --- Guillot 2010, Eq. 49 -----------------------------------------
    T_guillot4 = 0.75 * T_int**4 * (2.0 / 3.0 + tau)

    if T_irr > 0.0:
        sqrt3 = math.sqrt(3.0)
        xi = (2.0 / 3.0
              + 1.0 / (gamma_v * sqrt3)
              + (gamma_v / sqrt3 - 1.0 / (gamma_v * sqrt3))
                * np.exp(-gamma_v * tau * sqrt3))
        T_guillot4 = T_guillot4 + 0.75 * mu_star * T_irr**4 * xi

    T_guillot = T_guillot4 ** 0.25

    # --- T-dependent adiabat anchored at the photosphere --------------
    # τ = 2/3 ⇒ T_guillot = T_int (exactly, by construction with T_irr=0).
    # We integrate the adiabat outward from the level closest to
    # P_photo using a T-dependent ``∇_ad`` so the result matches the
    # gradient that ``_init_adiabat``'s projection will use later.
    P_photo  = (2.0 / 3.0) * gravity / kappa_ir
    n_levels = len(pressures)
    T_adi    = np.zeros_like(pressures)

    # Anchor index = level whose pressure is closest to P_photo.
    idx_anchor = int(np.argmin(np.abs(pressures - P_photo)))
    T_adi[idx_anchor] = T_int

    # Integrate to the layer at position idx_anchor+1, idx_anchor+2, ...
    # using each step's local T to pick the right ∇_ad.  This works
    # for either array ordering — we just need adjacent pairs.
    for i in range(idx_anchor + 1, n_levels):
        gad = _grad_ad_h2he_approx(T_adi[i - 1])
        T_adi[i] = T_adi[i - 1] * (pressures[i] / pressures[i - 1]) ** gad
    for i in range(idx_anchor - 1, -1, -1):
        gad = _grad_ad_h2he_approx(T_adi[i + 1])
        T_adi[i] = T_adi[i + 1] * (pressures[i] / pressures[i + 1]) ** gad

    return np.maximum(T_guillot, T_adi)


def _init_atmosphere(atm: Atmosphere, target: Target, opts: dict) -> None:
    """Build the pressure–temperature grid."""
    n_levels = atm.n_levels
    n_layers = n_levels - 1

    # Log-uniform pressure grid.
    # ``pressure_min`` / ``pressure_max`` come from the .nml in Pa, matching
    # the convention of the apriori ``temperature_profile_*.dat`` file.
    # Level 0 sits at p = pressure_max (deepest), level n_levels-1 at
    # p = pressure_min (top of atmosphere).
    log_pmin = math.log(atm.pressure_min)
    log_pmax = math.log(atm.pressure_max)
    atm.pressures = np.exp(np.linspace(log_pmax, log_pmin, n_levels))

    # Initial temperature profile — three sources, in order of preference:
    #
    #   (1) ``temperature_profile_file = "guillot"`` (case-insensitive) →
    #       generate a self-consistent ``max(T_Guillot, T_adiabat)`` profile
    #       at runtime.  This is the recommended setting because exorem's
    #       retrieval cannot autonomously move the radiative-convective
    #       boundary — it only fine-tunes around the apriori's RCB.  A
    #       Guillot-2010 + adiabat construction puts the RCB approximately
    #       at the right depth (where the two curves cross) by definition.
    #       Optional overrides via ``opts``:
    #           guillot_T_irr        (K)         default 0
    #           guillot_kappa_ir     (m²/kg)     default 1e-2
    #           guillot_gamma_v                  default 0.4
    #           guillot_grad_ad                  default 0.26
    #
    #   (2) Tabulated apriori file (the original behaviour).  Path is
    #       resolved relative to ``path_temperature_profile``.
    #
    #   (3) Isothermal column at T_int as a last-ditch fallback.
    tp_file = opts.get("temperature_profile_file", "None")
    tp_path: Optional[Path] = None
    if tp_file and tp_file.lower() not in ("none", "guillot"):
        tp_path = Path(tp_file)
        if not tp_path.is_absolute():
            tp_path = Path(opts.get("path_temperature_profile", ".")) / tp_path

    if tp_file and tp_file.lower() == "guillot":
        T_int    = target.target_internal_temperature
        gravity  = target.target_gravity
        T_irr    = float(opts.get("guillot_T_irr",   0.0))
        gamma_v  = float(opts.get("guillot_gamma_v", 0.4))
        grad_ad  = float(opts.get("guillot_grad_ad", 0.286))
        if T_int <= 0:
            raise ValueError(
                "temperature_profile_file = 'guillot' requires a positive "
                "target_internal_temperature in the namelist.")
        if gravity <= 0:
            raise ValueError(
                "temperature_profile_file = 'guillot' requires a positive "
                "target_gravity in the namelist.")

        # Resolve P_photo / κ_IR using a three-tier priority:
        #
        #   1. ``guillot_p_photo_bar`` set → use it directly.  κ_IR is
        #      derived as  κ_IR = 2g/(3 P_photo).  This is the most
        #      physically intuitive control (you state where the
        #      photosphere lives) and κ_IR adapts automatically to
        #      gravity.
        #
        #   2. ``guillot_kappa_ir`` set → use it directly (legacy
        #      override; you might know an atmosphere-specific value
        #      from a tabulated opacity calculation).
        #
        #   3. Neither set → derive κ_IR from a Freedman-style
        #      scaling at T_int, and back out P_photo from g.
        #      *This is the default* and gives a reasonable Guillot
        #      profile for any gravity without further tuning.
        p_photo_bar_user = opts.get("guillot_p_photo_bar", None)
        kappa_ir_user    = opts.get("guillot_kappa_ir",    None)

        if p_photo_bar_user is not None:
            P_photo  = float(p_photo_bar_user) * 1.0e5     # bar → Pa
            kappa_ir = (2.0 / 3.0) * gravity / P_photo
            kappa_src = f"derived from guillot_p_photo_bar={p_photo_bar_user:.3g}"
        elif kappa_ir_user is not None:
            kappa_ir = float(kappa_ir_user)
            P_photo  = (2.0 / 3.0) * gravity / kappa_ir
            kappa_src = f"user-set guillot_kappa_ir={kappa_ir:.2e}"
        else:
            kappa_ir = _kappa_ir_freedman_approx(T_int)
            P_photo  = (2.0 / 3.0) * gravity / kappa_ir
            kappa_src = (f"Freedman-style scaling κ(T={T_int:.0f}K)="
                         f"{kappa_ir:.2e} m²/kg")

        atm.temperatures = _make_guillot_apriori(
            atm.pressures,
            T_int=T_int, T_irr=T_irr, gravity=gravity,
            kappa_ir=kappa_ir, gamma_v=gamma_v, grad_ad=grad_ad)
        # Stash the photospheric pressure on the Atmosphere object so
        # ``_init_adiabat`` knows exactly where the radiative-convective
        # boundary sits and can stop its projection there.  This is the
        # one piece of information exorem's retrieval cannot recover on
        # its own — the apriori is the only source of truth for the RCB.
        atm.pressure_rcb = P_photo
        # r39e: stash a snapshot of the *original* apriori T-profile so
        # ``_init_adiabat`` can use the apriori's native lapse rate (rather
        # than the running, retrieval-modified T-profile) to detect the
        # RCB.  This makes the break location stable across iterations.
        atm.temperatures_apriori_original = atm.temperatures.copy()
        print(
            f"  Generated Guillot+adiabat apriori "
            f"(T_int={T_int:.1f} K, T_irr={T_irr:.1f} K, g={gravity:.2f} m/s²,\n"
            f"    κ_IR={kappa_ir:.2e} m²/kg [{kappa_src}],\n"
            f"    P_photo={P_photo:.2e} Pa = {P_photo/1e5:.3f} bar, "
            f"∇_ad={grad_ad:.3f}):\n"
            f"    T range {atm.temperatures.min():.1f}–"
            f"{atm.temperatures.max():.1f} K  "
            f"over p range {atm.pressures.min():.2e}–"
            f"{atm.pressures.max():.2e} Pa")
    elif tp_path is not None and tp_path.exists():
        from .interface import load_temperature_profile
        p_file, t_file = load_temperature_profile(tp_path)
        # p_file is the apriori's native ordering (low → high pressure).
        # np.interp requires the xp array to be monotonically increasing —
        # do NOT reverse it.  atm.pressures itself is decreasing (deep → top)
        # but that's fine for the x argument.
        atm.temperatures = np.interp(
            np.log(atm.pressures), np.log(p_file), t_file)
        # The file-input path must set the SAME bookkeeping the guillot branch
        # sets, or the convective closure (_init_adiabat needs pressure_rcb) and
        # the a-priori term (need temperatures_apriori_original) silently break.
        atm.temperatures_apriori_original = atm.temperatures.copy()
        _T_int   = target.target_internal_temperature
        _gravity = target.target_gravity
        _kappa_ir = float(opts.get("guillot_kappa_ir",
                                   _kappa_ir_freedman_approx(_T_int)
                                   if _T_int > 0 else 5.0e-4))
        _p_photo_bar = opts.get("guillot_p_photo_bar", None)
        if _p_photo_bar is not None:
            atm.pressure_rcb = float(_p_photo_bar) * 1.0e5
        elif _gravity > 0:
            atm.pressure_rcb = (2.0 / 3.0) * _gravity / _kappa_ir
        print(f"  Loaded apriori T-profile from {tp_path.name} "
              f"(T range {t_file.min():.1f}–{t_file.max():.1f} K, "
              f"p range {p_file.min():.2e}–{p_file.max():.2e} Pa; "
              f"P_photo/RCB set to {getattr(atm,'pressure_rcb',float('nan')):.2e} Pa)")
    else:
        if tp_file and tp_file.lower() != "none":
            print(f"  Warning: T-profile file not found at {tp_path}; "
                  f"falling back to isothermal column.")
        atm.temperatures = np.full(n_levels, target.target_internal_temperature
                                    if target.target_internal_temperature > 0
                                    else 1000.0)

    atm.pressures_layers    = np.sqrt(atm.pressures[:-1] * atm.pressures[1:])
    atm.temperatures_layers = np.sqrt(atm.temperatures[:-1] * atm.temperatures[1:])
    atm.gravities_layers    = np.full(n_layers, target.target_gravity)
    atm.molar_masses_layers = np.full(n_levels, 2.3e-3)   # kg mol⁻¹, H2-dominated
    atm.scale_height        = (CST_R * atm.temperatures_layers
                                / (atm.molar_masses_layers[:n_layers] * atm.gravities_layers))  # m → km
    atm.z                   = np.zeros(n_levels)
    atm.eddy_diffusion_coefficient = np.full(n_layers, 1e6)


def _peek_ktable_wavenumber_max(path: str, species_names: list[str]) -> float | None:
    """
    Peek at every k-table .h5 file and return the smallest of the maximum
    tabulated wavenumbers across all loaded species (cm⁻¹).

    Returns ``None`` if no k-tables can be read.  Used by
    :func:`_init_wavenumbers` to clamp the user's ``wavenumber_max`` to a
    range actually covered by the opacity data, preventing the spectrum
    from extending into regions where the k-tables have zero opacity (and
    therefore would expose unphysical thermal radiation from deep, hot
    layers).
    """
    from pathlib import Path

    try:
        import h5py
    except ImportError:
        return None

    wn_max_per_species: list[float] = []
    for s in species_names:
        candidates: list[Path] = []
        for pat in (f"{s}.h5", f"{s}.ktable.exorem.h5",
                    f"{s}.ktable.*.h5", f"{s}.*.h5"):
            candidates.extend(Path(path).glob(pat))
        if not candidates:
            continue
        fpath = sorted(candidates)[0]
        try:
            with h5py.File(fpath, "r") as fh:
                wn_k = None
                for key in ("wavenumbers", "wavenumber", "wno", "wn", "nu",
                            "bin_centers"):
                    if key in fh:
                        wn_k = np.asarray(fh[key][...]).ravel()
                        break
                if wn_k is None and "bin_edges" in fh:
                    edges = np.asarray(fh["bin_edges"][...]).ravel()
                    wn_k = 0.5 * (edges[:-1] + edges[1:])
                if wn_k is not None and wn_k.size:
                    wn_max_per_species.append(float(wn_k.max()))
        except (OSError, KeyError):
            continue

    if not wn_max_per_species:
        return None
    return min(wn_max_per_species)


def _init_wavenumbers(
    spectrometrics: Spectrometrics,
    path_k_coefficients: str | None = None,
    species_names: list[str] | None = None,
) -> None:
    """Build the wavenumber grid.

    r35: the spectrum now extends to the user's requested ``wavenumber_max``
    even when that exceeds the k-table coverage.  The earlier r28 behaviour
    was to silently clamp ``wavenumber_max`` down to ``min(max(wn_k))``
    across the k-tables to prevent "unphysical thermal emission from deep
    hot layers" in regions without molecular opacity.  In practice that
    threw away the high-wavenumber escape window that real brown-dwarf
    atmospheres use to radiate ~42% of a 3000 K blackbody's flux above
    8130 cm⁻¹.  With the window suppressed, the retrieval was forced to
    deliver σT_int⁴ entirely through the low-wn region by building a
    stratospheric inversion at ~20 Pa — the persistent "hot area at top"
    that survived every other r30-r34 fix.

    Above the k-table coverage:
      * absorbing-species opacity is zero (handled by ``n_wn_active_per_species``
        in radiative_transfer.py).
      * CIA is zero (``np.interp`` with ``left=0, right=0`` in load_cia).
      * Rayleigh scattering IS computed from refractive indices (which
        scales as ν⁴, so it dominates at short wavelengths and provides
        the right physical limit on deep-zone escape).
      * Planck thermal emission contributes normally at every layer.

    This makes the high-wn region a Rayleigh-scattering + thermal emission
    medium with no line opacity — a reasonable approximation for what a
    real H/He atmosphere does in the optical/UV between absorber bands.
    """
    wn_min  = spectrometrics.wavenumber_min
    wn_max  = spectrometrics.wavenumber_max
    wn_step = spectrometrics.wavenumber_step

    # Diagnostic only: report where k-table coverage ends, but don't clamp.
    if path_k_coefficients is not None and species_names:
        wn_max_avail = _peek_ktable_wavenumber_max(
            path_k_coefficients, species_names)
        if wn_max_avail is not None and wn_max_avail < wn_max:
            print(f"  Note: requested wavenumber_max = {wn_max:.1f} cm⁻¹ "
                  f"exceeds k-table coverage ({wn_max_avail:.1f} cm⁻¹).")
            print(f"  Extension region [{wn_max_avail:.0f}, {wn_max:.0f}] cm⁻¹ "
                  f"will use Rayleigh + Planck only (no line opacity).")

    n_wn = int(round((wn_max - wn_min) / wn_step)) + 1
    spectrometrics.n_wavenumbers = n_wn
    spectrometrics.wavenumbers   = wn_min + np.arange(n_wn) * wn_step
    spectrometrics.spectral_radius = np.zeros(n_wn)

    print(f"  Spectrum: {wn_min:.2f} to {spectrometrics.wavenumbers[-1]:.2f} "
          f"cm⁻¹ (step {wn_step:.2f} cm⁻¹, {n_wn} points)")


def _load_k_coefficients(path: str, species_names: list[str],
                          spectrometrics: Spectrometrics) -> dict:
    """
    Load k-coefficient tables from HDF5 files.

    Delegates to :func:`exorem.loaders.load_k_coefficients`, which handles
    the petitRADTRANS-style storage (``bin_centers, p, t, kcoeff, weights,
    samples``) used by the example Exorem data tree and transposes the
    arrays into the (n_g, n_wn, n_t, n_p) order this module expects.
    """
    from .loaders import load_k_coefficients
    return load_k_coefficients(path, species_names, spectrometrics)


def _load_thermochemical_tables(path: str) -> dict:
    """
    Load JANAF-derived Gibbs free energy and heat capacity tables.

    Delegates to :func:`exorem.loaders.load_thermochemical_tables`, which
    reads the per-species ``.tct.dat`` columnar files (``gases/`` and
    ``condensates/`` sub-directories).
    """
    from .loaders import load_thermochemical_tables
    return load_thermochemical_tables(path)


def _load_cia(opts: dict, wavenumbers: np.ndarray, atm: Atmosphere
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load collision-induced absorption cross-sections.

    Delegates to :func:`exorem.loaders.load_cia`, which parses the
    Exorem-native CIA format (``n_T n_wn`` header, then a temperature list,
    then ``wn σ(T1) … σ(T_nT)`` rows).
    """
    from .loaders import load_cia
    return load_cia(opts, wavenumbers, atm)


def _init_rayleigh_scattering(
    species_names: list[str],
    wavenumbers: np.ndarray,
) -> np.ndarray:
    """
    Pre-compute Rayleigh scattering cross-sections (cm²) for each gas.

    Returns a (N_GASES, n_wavenumbers) matrix indexed by ``gas_id``.

    Includes EVERY gas that has a refractive-index formula defined in
    ``optics._REFRACTIVE_INDEX_TABLE`` — most importantly H2 and He, which
    are the bulk constituents of an H/He atmosphere and the DOMINANT
    Rayleigh scatterers, but which are typically NOT in the .nml
    ``species_names`` list (which only contains *absorbing* species).

    Previously this routine looped only over ``species_names`` (13 minor
    absorbers like CH4, CO, H2O, ...), so H2 and He contributed zero
    Rayleigh scattering — the dominant ~90 % of the column.  At short
    wavelengths where the molecular k-tables have no opacity, the
    atmosphere then appeared completely transparent and the spectrum
    showed unphysical thermal emission from deep, hot layers.
    """
    from .optics import _REFRACTIVE_INDEX_TABLE

    n_wn = len(wavenumbers)
    coeffs = np.zeros((N_GASES, n_wn))

    # Union: gases with refractive-index formulas, plus the user-listed
    # absorbing species (some of which may lack a formula and will use the
    # default n=1.0003 → small but non-zero Rayleigh).
    rayleigh_species = set(_REFRACTIVE_INDEX_TABLE.keys()) | set(species_names)

    for name in rayleigh_species:
        i = gas_id(name)
        if i < 0:
            continue
        for j, wn in enumerate(wavenumbers):
            n_ri = get_refractive_index(name, wn)
            coeffs[i, j] = rayleigh_scattering_coefficient(n_ri, wn)
    return coeffs


# ===========================================================================
# Radiative-transfer wrapper
# ===========================================================================


def _do_radiative_transfer(
    state, gases_vmr, cloud_vmr, n_clouds_active, atm, spec, cloud_obj,
    spectrometrics, light, kcoeff_tables,
    h2_h2_cia, h2_he_cia, h2o_n2_cia, h2o_h2o_cia,
    tau_in, tau_rayleigh_in,
):
    """Thin wrapper around calculate_radiative_transfer."""
    tables = kcoeff_tables
    target = state["target"]
    n_species = spec.n_species

    # Pad tables to uniform shapes
    ng_max  = int(tables["ng_max"])
    n_wn_max = max(tables["n_k_wavenumbers"])
    n_p_max  = max(tables["n_k_pressures"])
    n_t_max  = max(tables["n_k_temperatures"])
    n_sp     = n_species

    wavenumbers_k_pad = np.zeros((n_wn_max, n_sp))
    p_k_pad           = np.zeros((n_p_max, n_sp))
    t_k_pad           = np.zeros((n_t_max, n_p_max, n_sp))
    kcoeff_pad        = np.zeros((ng_max, n_wn_max, n_t_max, n_p_max, n_sp))

    for i in range(n_sp):
        nwk = tables["n_k_wavenumbers"][i]
        npk = tables["n_k_pressures"][i]
        ntk = tables["n_k_temperatures"][i]
        ng_i = tables["ng"][i]
        wavenumbers_k_pad[:nwk, i] = tables["wavenumbers_k"][i]
        p_k_pad[:npk, i]           = tables["p_k_species"][i]
        t_k_pad[:ntk, :npk, i]    = tables["t_k_species"][i][:ntk, :npk] if tables["t_k_species"][i].ndim >= 2 else tables["t_k_species"][i][:ntk, np.newaxis]
        kcoeff_pad[:ng_i, :nwk, :ntk, :npk, i] = tables["kcoeff_species"][i][:ng_i, :nwk, :ntk, :npk]

    # Cloud arrays
    if n_clouds_active > 0:
        q_ext   = cloud_obj.q_ext
        q_scat  = cloud_obj.q_scat
        ssa     = cloud_obj.single_scattering_albedo
        asym    = cloud_obj.asymetry_factor
        q_ext_ref = cloud_obj.q_ext_ref
        cp_density = cloud_obj.cloud_particle_density
        cp_radius  = cloud_obj.cloud_particle_radius
    else:
        nw = spectrometrics.n_wavenumbers
        nl = atm.n_layers
        q_ext = np.zeros((1, nw, nl))
        q_scat= np.zeros((1, nw, nl))
        ssa   = np.zeros((1, nw, nl))
        asym  = np.zeros((1, nw, nl))
        q_ext_ref = np.zeros((1, nl))
        cp_density = np.zeros(1)
        cp_radius  = np.zeros((1, nl))

    # Build species_vmr_layers from gases_vmr using gas_id mapping
    # (was previously passed as zeros, killing all per-species absorption)
    species_vmr_layers = np.zeros((atm.n_layers, n_species))
    for k, name in enumerate(spec.species_names):
        i_gas = gas_id(name)
        if 0 <= i_gas < gases_vmr.shape[0]:
            species_vmr_layers[:, k] = gases_vmr[i_gas, :]

    return calculate_radiative_transfer(
        gases_vmr=gases_vmr,
        pressures_layers=atm.pressures_layers,
        temperatures_layers=atm.temperatures_layers,
        gravities_layers=atm.gravities_layers,
        species_vmr_layers=species_vmr_layers,
        pressures=atm.pressures,
        temperatures=atm.temperatures,
        light_source_irradiance=light.irradiance,
        n_species=n_species,
        i_single_species=-1,
        wavenumber_min=spectrometrics.wavenumber_min,
        wavenumber_step=spectrometrics.wavenumber_step,
        rayleigh_scattering_coefficients=state["rayleigh_coeffs"],
        n_clouds=n_clouds_active,
        cloud_vmr=cloud_vmr,
        cloud_particle_density=cp_density,
        cloud_particle_radius=cp_radius,
        cloud_q_ext=q_ext,
        cloud_q_scat=q_scat,
        cloud_single_scattering_albedo=ssa,
        cloud_asymetry_factor=asym,
        cloud_q_ext_ref=q_ext_ref,
        n_k_pressures=tables["n_k_pressures"],
        n_k_temperatures=tables["n_k_temperatures"],
        n_k_wavenumbers=tables["n_k_wavenumbers"],
        wavenumbers_k=wavenumbers_k_pad,
        ng=tables["ng"],
        p_k_species=p_k_pad,
        t_k_species=t_k_pad,
        weights_k=tables["weights_k"],
        samples_k=tables["samples_k"],
        kcoeff_species=kcoeff_pad,
        h2_h2_cia=h2_h2_cia,
        h2_he_cia=h2_he_cia,
        h2o_n2_cia=h2o_n2_cia,
        h2o_h2o_cia=h2o_h2o_cia,
        n_levels=atm.n_levels,
        n_layers=atm.n_layers,
        n_wavenumbers=spectrometrics.n_wavenumbers,
        wavenumbers=spectrometrics.wavenumbers,
        scale_height=atm.scale_height,
        cos_average_angle=target.cos_average_angle,
        idx_h2=gas_id("H2"),
        idx_he=gas_id("He"),
        idx_h2o=gas_id("H2O"),
    )


# ===========================================================================
# Atmospheric physics sub-routines
# ===========================================================================


def _check_cloud_condensation(cloud_vmr: np.ndarray, n_clouds: int) -> bool:
    if n_clouds == 0:
        return False
    return bool(np.any(cloud_vmr > 0.0))


def _init_s_matrix(pressures: np.ndarray, retrieval: ExoremRetrieval) -> np.ndarray:
    """Build the a-priori covariance matrix S."""
    n_levels = len(pressures)
    matrix_s = np.zeros((n_levels, n_levels))
    log_p = np.log(pressures)
    log_range = log_p[0] - log_p[-1]

    for i in range(n_levels):
        frac_i = (log_p[i] - log_p[-1]) / log_range if log_range != 0 else 0.5
        corr_i = (retrieval.smoothing_bottom * frac_i
                  + retrieval.smoothing_top * (1.0 - frac_i))
        for k in range(n_levels):
            frac_k = (log_p[k] - log_p[-1]) / log_range if log_range != 0 else 0.5
            corr_k = (retrieval.smoothing_bottom * frac_k
                      + retrieval.smoothing_top * (1.0 - frac_k))
            arg = -0.5 * (log_p[i] - log_p[k])**2 / (corr_i * corr_k) if corr_i * corr_k > 0 else 0.0
            matrix_s[i, k] = retrieval.weight_apriori * math.exp(arg)

    return matrix_s


# ===========================================================================
# Faithfulness flags for the retrieval's non-Fortran safety nets
# ===========================================================================
# These guards were added to mask instabilities that we now attribute to the
# missing corr_adia H2-dissociation term in the deep adiabat.  With CORR_ADIA
# = True the deep is stable grid-wide, so the band-aids may no longer be
# needed — and the *unconditional* inversion clamp in particular appears to
# CAUSE the residual upper-atmosphere cold-top: once a dip forms at the
# (creeping) radiative–convective boundary, T[i] = min(T[i], 1.05*T[i-1])
# caps every layer above it at 1.05x the over-cooled layer below, ratcheting
# the column down to the T_MIN floor instead of letting it recover toward the
# Fortran's cold-but-finite (~170–260 K) radiative equilibrium.
#
# Fortran (exorem.f90 L2670-2681) applies this clamp ONLY for iter <= 10.
#   True  -> faithful: clamp active only during the iter<=10 burn-in.
#   False -> legacy port behaviour: clamp every iteration.
INVERSION_CLAMP_BURNIN_ONLY = True

# Fortran has NO per-iteration rate limit (it errors out on T<0 instead).  The
# port caps |dT| <= RETRIEVAL_MAX_DT_FRAC * T_old every iteration.
#   float -> cap each layer's step at that fraction of T_old.
#   None  -> faithful: no rate limit.
RETRIEVAL_MAX_DT_FRAC = 0.30


def _temperature_profile_retrieval(
    atm: Atmosphere,
    retrieval: ExoremRetrieval,
    matrix_s: np.ndarray,
    matrix_t: np.ndarray,
    rad_diff: np.ndarray,
    rad_noise: np.ndarray,
    dt: np.ndarray,
    solution_deviation: float,
    iteration: int,
    tau_rep: np.ndarray = None,
) -> bool:
    """
    Constrained linear inversion to update the temperature profile.

    Returns True if the retrieval has converged.

    ``tau_rep`` is the per-level bulk cumulative optical depth (index 0 =
    deepest).  When provided and ``UPPER_ATM_INFO_WEIGHTING`` is on, it is used
    to fade the optimal-estimation step out of the optically-thin, data-empty
    upper atmosphere (which the radiative inversion cannot constrain) and relax
    those levels toward the a-priori profile instead.  See the comment block on
    ``UPPER_ATM_INFO_WEIGHTING`` for the full rationale.
    """
    n_levels    = atm.n_levels
    i0          = retrieval.retrieval_level_top - 1
    n_retrieved = retrieval.retrieval_level_bottom - i0

    # S * Kᵀ
    K = matrix_t[:, i0: i0 + n_retrieved]
    SK = matrix_s @ K                           # (n_levels, n_retrieved)

    # K * S * Kᵀ
    KSK = K.T @ SK                              # (n_retrieved, n_retrieved)

    print(f"  Trace of matrix KSKt: {np.trace(KSK):.4e}")

    # (KSKᵀ + Se) — observation-space covariance.
    # The bare matrix (KSK + Se) has cond_M ≳ 1e6 throughout this problem
    # because many vertical modes of the temperature profile have very weak
    # Jacobian signature (deep optically-thick layers contribute little to
    # the spectrum; very tenuous layers contribute nothing).  Inverting such
    # a near-singular matrix amplifies noise in those modes, producing
    # nonsensical proposed dT (we measured up to 11,000× T_old).
    #
    # Tikhonov regularisation: M_reg = M + λ·(trace(M)/n)·I.
    # This adds a small isotropic ridge proportional to the mean diagonal
    # eigenvalue.  Well-constrained modes (large eigenvalues) are barely
    # affected; near-singular modes (small eigenvalues) are damped, so the
    # gain matrix never explodes there.  With λ = 1e-3 the condition number
    # is capped at ≲ 1/λ = 10³, restoring reliable Newton directions.
    Se = np.diag(rad_noise**2)
    M_bare = KSK + Se
    n_obs = M_bare.shape[0]
    trace_M = float(np.trace(M_bare))
    # Fortran inverts M_bare = KSKᵀ + diag(rad_noise²) directly (no ridge).
    # The ridge is now opt-in (USE_TIKHONOV) and OFF by default so the gain
    # matches the reference.  The near-singular thin-layer modes that the ridge
    # was meant to tame are instead handled exactly as Fortran handles them:
    # the per-layer rate limit caps the magnitude of the raw Newton step and the
    # re-enabled 1.05·T_below inter-layer clamp prevents the residual from
    # accumulating as an upper-atmosphere hot bubble.
    if USE_TIKHONOV:
        lambda_tikhonov = 1.0e-3
        M_reg = M_bare + lambda_tikhonov * (trace_M / n_obs) * np.eye(n_obs)
    else:
        lambda_tikhonov = 0.0
        M_reg = M_bare
    M_inv = matinv(M_reg)
    # legacy name M kept so the cond(M) printout below sees the same matrix
    M = M_reg

    # Gain matrix: R = SK * M⁻¹
    R = SK @ M_inv                              # (n_levels, n_retrieved)

    # --- DIAGNOSTICS: capture state BEFORE any clamping ---
    dt_proposed = (R @ rad_diff).copy()                       # raw Newton step

    # Condition number of M (large → ill-posed retrieval).
    # We compute this AFTER Tikhonov regularisation; with λ = 1e-3 it
    # should sit around 10³, vs >10⁶ for the bare matrix.
    try:
        sv = np.linalg.svd(M, compute_uv=False)
        cond_M = float(sv[0] / max(sv[-1], 1e-300))
    except Exception:
        cond_M = float('nan')
    print(f"  Tikhonov λ={lambda_tikhonov:.0e}, cond(M_reg)={cond_M:.2e}")

    # Temperature update with three layers of protection:
    #   (a) NaN/Inf guard — reject the step if any element is non-finite.
    #   (b) Blow-up detection — only when SOME layer's proposed change is
    #       extreme (>5× its current T), clip dt per-layer to ±30%·T_old.
    #   (c) Line-search — final safety net so no layer drops below T_MIN.
    dt[:] = dt_proposed.copy()

    # (a) NaN guard
    if not np.all(np.isfinite(dt)):
        print("  Warning: non-finite dt — rejecting this retrieval step.")
        dt[:] = 0.0

    T_MIN = 100.0         # K — physical floor; chemistry equilibrium becomes
                          # numerically singular below ~100 K (deep Wien tail
                          # of Planck function + huge exp(-ΔG/RT) factors)
    T_MAX = 10000.0       # K — physical ceiling; above this H2 dissociates
                          # heavily, opacity tables don't extend reliably,
                          # and chemistry blows up.  A BD interior should
                          # never reach this temperature in radiative-
                          # convective equilibrium for T_int < 2000 K.
    T_old = atm.temperatures.copy()

    # ------------------------------------------------------------------
    # Upper-atmosphere well-posedness: fade the OE step out of the
    # optically-thin, data-empty upper atmosphere and relax those levels
    # toward the a-priori (monotonic) radiative-equilibrium profile.
    # See the UPPER_ATM_INFO_WEIGHTING comment block for the full rationale.
    # This runs BEFORE the rate-limit / line-search so the blended step is
    # still subject to all the existing safety nets.
    # ------------------------------------------------------------------
    if UPPER_ATM_INFO_WEIGHTING and tau_rep is not None:
        T_apri = getattr(atm, "temperatures_apriori_original", None)
        if T_apri is not None:
            w = tau_rep / (tau_rep + TAU_INFO)        # ->1 thick, ->0 thin
            dt_apri = APRIORI_RELAX * (T_apri - T_old)
            dt_blended = w * dt + (1.0 - w) * dt_apri
            n_thin = int(np.sum(w < 0.5))
            if n_thin > 0:
                print(f"  Upper-atm info weighting: {n_thin} optically-thin "
                      f"levels (tau<{TAU_INFO:g}) relaxed toward a-priori "
                      f"(max OE |dT| there was "
                      f"{np.max(np.abs(dt[w < 0.5])):.0f} K, "
                      f"blended to {np.max(np.abs(dt_blended[w < 0.5])):.0f} K).")
            dt[:] = dt_blended
            dt_proposed = dt.copy()   # diagnostics reflect the blended proposal

    #
    # Previous behaviour (r38 and earlier): only clipped when the *global*
    # max(|dt_i|/T_i) exceeded 2.0.  This failure mode appeared once a
    # convective-term-correction fix in r39a let the retrieval push
    # mid-atmosphere layers up to 10^4 K:
    #
    #   iter 19: max |dt|/T_old = 2.06 → clamped (β=0.15)  → max applied 603 K
    #   iter 20: max |dt|/T_old = 1.97 → NOT clamped (β=1.0) → max applied 10,494 K
    #
    # The runaway grew in absolute terms (T at 1 bar: 2.9 kK → 4.7 kK → 7.6 kK
    # → … → 31 kK over the next 30 iterations) while the *relative* step
    # remained below the 2.0 threshold, so the clamp never re-engaged.
    #
    # Fix: always cap each layer's step at MAX_DT_FRAC × T_old per iteration.
    # This is a hard Newton-damping rate limit that prevents runaway when
    # the linearized retrieval produces wildly overconfident proposals
    # (which happens routinely when the Jacobian is poorly conditioned
    # around any one layer, e.g. when the apriori is far from the truth
    # in the deep zone).  30 % per iter is loose enough to allow rapid
    # burn-in convergence (50 iters × 30 % stacks well past order-of-
    # magnitude T changes) while preventing chemistry blow-up.
    #
    # Diagnostic `beta_clamp` in retrieval_summary.csv now reports the
    # *effective* max ratio applied → 1.0 when no layer hit the cap,
    # MAX_DT_FRAC when at least one layer was clipped.
    # ------------------------------------------------------------------
    MAX_DT_FRAC = RETRIEVAL_MAX_DT_FRAC
    max_relative_step = float(np.max(np.abs(dt) / np.maximum(T_old, 1.0)))

    if MAX_DT_FRAC is not None:
        max_dt = MAX_DT_FRAC * T_old
        clipped_any = bool(np.any(np.abs(dt) > max_dt))
        dt[:] = np.clip(dt, -max_dt, max_dt)
        beta_clamp = MAX_DT_FRAC if clipped_any else 1.0

        if max_relative_step > MAX_DT_FRAC:
            n_layers_clipped = int(np.sum(np.abs(dt) >= max_dt - 1e-12))
            print(f"  Note: per-layer rate limit engaged "
                  f"(proposed max |ΔT|/T = {max_relative_step:.2f} "
                  f"→ {n_layers_clipped} layers clipped to ±{MAX_DT_FRAC*100:.0f}% T_old).")
    else:
        beta_clamp = 1.0   # rate limit disabled (faithful) — no clipping

    T_proposed = T_old + dt

    # (c) Line-search: scale dt by α so no layer drops below T_MIN
    alpha = 1.0
    if np.any(T_proposed < T_MIN):
        cooling = dt < 0
        if cooling.any():
            safe = (T_old[cooling] - T_MIN) / (-dt[cooling])
            safe = safe[safe > 0]
            alpha = max(0.0, min(1.0, 0.95 * float(safe.min()))) if safe.size else 0.0
        else:
            alpha = 0.0
        print(f"  Warning: T-update damped α={alpha:.3f} to keep T ≥ {T_MIN:.0f}K.")
        dt[:] *= alpha

    atm.temperatures[:] = np.clip(T_old + dt, T_MIN, T_MAX)

    # --- DIAGNOSTICS: dump retrieval state ---
    _dump_retrieval_debug(
        getattr(retrieval, "_debug_path", None), iteration, atm,
        T_old, dt_proposed, dt, rad_diff, rad_noise,
        trace_KSK=float(np.trace(KSK)),
        cond_M=cond_M,
        max_relative_step=max_relative_step,
        alpha=alpha,
        beta=beta_clamp,
        i0=i0, n_retrieved=n_retrieved,
    )

    # --- Full matrix dump (no behavioural effect) ---
    _dump_retrieval_matrices(
        getattr(retrieval, "_debug_path", None), iteration,
        matrix_t=matrix_t, matrix_s=matrix_s,
        K=K, SK=SK, KSK=KSK,
        M_bare=M_bare, M_reg=M_reg, M_inv=M_inv, R=R,
        dt_proposed=dt_proposed, dt_applied=dt,
        rad_diff=rad_diff, rad_noise=rad_noise,
        T_old=T_old, T_new=atm.temperatures.copy(),
        lambda_tikhonov=lambda_tikhonov,
    )

    # Early convergence
    temperature_variation = float(np.max(np.abs(dt) / atm.temperatures))
    converged = False
    # Faithful port of the Fortran's two-branch test (exorem.f90 2662-2667):
    #   * converged DURING the pure-radiative burn-in  -> shorten it so the
    #     adiabat engages on the very next iteration;
    #   * converged AFTER the burn-in                   -> declare convergence.
    # The Python previously had only the second branch, so it always ran the
    # full fixed burn-in.  For some (T_int, g) the pure-radiative phase then
    # over-runs and diverges (deep collapses to the opacity-table floor ~100 K,
    # top runs away to thousands of K) before convection can stabilise it —
    # the grid's catastrophic cells.  Engaging the adiabat as soon as the
    # radiative phase settles prevents that.
    if (solution_deviation <= retrieval.retrieval_tolerance
            and iteration < retrieval.n_non_adiabatic_iterations
            and temperature_variation <= retrieval.retrieval_tolerance):
        retrieval.n_non_adiabatic_iterations = iteration + 1
    elif (solution_deviation <= retrieval.retrieval_tolerance
            and iteration > retrieval.n_non_adiabatic_iterations
            and temperature_variation <= retrieval.retrieval_tolerance):
        converged = True

    # r33: RESTORE the 1.05*T[i-1] inter-layer inversion clamp during burn-in.
    #
    # r32 removed this clamp based on a misdiagnosis (see HANDOFF_r32.md).  The
    # claim was that the clamp prevented the deep zone from becoming super-
    # adiabatic.  In fact the clamp T[i] = min(T[i], 1.05*T[i-1]) can ONLY bind
    # when T_above > T_below (an inversion).  In the convective deep zone we
    # have T_above < T_below by construction, so T_above/T_below < 1, well below
    # 1.05 — the clamp never fires there.  The deep super-adiabat is unaffected.
    #
    # What the clamp actually does: it limits the GROWTH of inversions in the
    # RADIATIVE upper atmosphere.  During the pure-radiative burn-in there is
    # no convective Jacobian coupling to redistribute heat, so the retrieval is
    # free to pile flux residual into the upper layers as a hot bubble (the
    # r28-r30 pathology).  Removing the clamp in r32 reproduced that pathology
    # at a different altitude: the r32 test run shows a 2635 K bubble centred
    # at L34 (~4 mbar) with TOA OLR 25% above σT_int^4.
    #
    # The α line-search (above) limits the MAGNITUDE of the global step, but
    # has no notion of inter-layer ordering and cannot prevent inversions.  The
    # per-level β trust-region clamp (also above, |dt|/T ≤ 0.15) is per-layer
    # only; it has the same blindness.  This 1.05*T inter-layer constraint is
    # the only inversion-specific guard.  Keep it on during burn-in, drop it
    # afterwards (when the convective coupling provides the right physics).
    # r48: RE-ENABLED — this is the fix for the upper-atmosphere hot-bubble
    # runaway.  r39d disabled this clamp on the assumption that the per-layer
    # 30 % rate limit + T_MIN/T_MAX clip + _init_adiabat's deep-zone adiabat
    # enforcement were "sufficient".  They are not: none of them constrains
    # INTER-LAYER ordering.  The optimal-estimation step is near-singular in the
    # optically-thin upper atmosphere (dF_net/dT → 0), so it proposes enormous
    # ΔT there (10³–10⁴ K); the rate limit caps the magnitude but the applied
    # step is still positive every iteration, so without an ordering constraint
    # the thin layers heat ~30 %/iter and compound into a hot bubble (the
    # r28–r30 / r32 pathology).  The α line-search bounds only the GLOBAL step
    # magnitude and the β/30 % clamp is PER-LAYER — both are blind to whether a
    # layer has become hotter than the one beneath it.  This 1.05·T_below
    # constraint is the ONLY inversion-specific guard, and it is exactly what the
    # Fortran reference applies (exorem.f90 L2670-2681).
    #
    # It only binds while T_above > 1.05·T_below (a developing inversion); in the
    # convective deep zone T_above < T_below by construction, so it never touches
    # the super-adiabat.  The true converged profile's gentle upper inversion
    # (~1.8 %/level up to ~271 K at the top) is well under 5 %/level, so the
    # clamp is also inactive at the fixed point and does not distort the answer.
    # It runs only during the radiative burn-in (before the convective coupling
    # exists to redistribute heat physically), then switches off.
    #
    # Fortran hard-codes the window as `iter <= 10`.  We gate it on the full
    # GRID EVIDENCE (full T_int×g×Met sweep): the hot-bubble runaway this clamp
    # guards against happens AFTER the burn-in, not during it.  The r33/r48
    # rationale gated the clamp to the burn-in on the assumption that "the
    # convective coupling provides the right physics afterwards" — but that
    # coupling only exists in the deep CONVECTIVE zone.  The optically-thin
    # upper atmosphere is radiative with no convective coupling, so once the
    # clamp switches off the near-singular optimal-estimation step (dF/dT→0,
    # ΔT~10³–10⁴ K, positive every iteration) heats the top unchecked; in the
    # worst (low-g, low-T_int) cells the inversion propagates down and collapses
    # the deep onto the opacity-table floor (~100 K) — the grid's catastrophic
    # cells.  The clamp is INACTIVE at the converged fixed point (the true
    # gentle upper inversion is ~1.8 %/level, well under 5 %), and inactive in
    # the deep convective zone (T_above < T_below by construction), so leaving
    # it on for every iteration does not distort healthy profiles — it only
    # caps the pathological inter-layer inversion.  (The Fortran's retrieval is
    # better-conditioned in the thin upper layers and does not need the clamp
    # past iter≤10; this port's does, so we keep the guard on throughout.)
    #
    # FAITHFULNESS GATE (added with the corr_adia deep-adiabat fix): the
    # Fortran applies this clamp only for iter <= 10.  With the deep now stable
    # the unconditional form appears to drive the upper-atmosphere cold-top, so
    # INVERSION_CLAMP_BURNIN_ONLY=True restores the Fortran's iter<=10 window.
    apply_clamp = (iteration <= 10) if INVERSION_CLAMP_BURNIN_ONLY else True
    if apply_clamp:
        for i in range(1, n_levels):
            atm.temperatures[i] = min(atm.temperatures[i],
                                      1.05 * atm.temperatures[i - 1])

    # Update layer temperatures
    n_layers = atm.n_layers
    atm.temperatures_layers[:] = np.sqrt(
        atm.temperatures[:n_layers] * atm.temperatures[1:n_levels])

    return converged


def _calculate_altitude(
    atm: Atmosphere, target: Target,
    gases_molar_mass: np.ndarray, gases_vmr: np.ndarray,
) -> None:
    """Recompute scale heights, altitudes, and layer gravities."""
    n_levels = atm.n_levels
    n_layers = atm.n_layers

    # Mean molar mass
    mu = np.zeros(n_levels)
    for i in range(N_GASES):
        mu[:n_layers] += gases_vmr[i, :] * gases_molar_mass[i] * 1e3   # kg → g mol⁻¹
    mu[n_layers] = mu[n_layers - 1]
    atm.molar_masses_layers = mu * 1e-3     # back to kg mol⁻¹
    print(f"  Mean molar mass: {mu.mean():.3e} g mol⁻¹")

    log_p = np.log(atm.pressures)

    # Reference level (1 bar = 1000 mbar = 1e5 Pa)
    level_1bar = int(np.argmin(np.abs(atm.pressures - 1e5)))

    if level_1bar > 0:
        t_1bar = np.interp(math.log(1e5), log_p[::-1], atm.temperatures[::-1])
        sh = (CST_R * 0.5 * (t_1bar + atm.temperatures_layers[level_1bar])
              / (mu[level_1bar] * 1e-3 * target.target_gravity))
        atm.z[level_1bar] = sh * (_A_1BAR - log_p[level_1bar])

        for j in range(level_1bar, 0, -1):
            gg = target.target_gravity * (target.target_radius / (target.target_radius + atm.z[j]))**2
            sh = CST_R * atm.temperatures_layers[j - 1] / (mu[j - 1] * 1e-3 * gg)
            atm.z[j - 1] = atm.z[j] + sh * (log_p[j] - log_p[j - 1])
            atm.gravities_layers[j - 1] = target.target_gravity * (
                target.target_radius / (target.target_radius + 0.5 * (atm.z[j] + atm.z[j - 1])))**2
            atm.scale_height[j - 1] = (CST_R * atm.temperatures_layers[j - 1]
                                        / (mu[j - 1] * 1e-3 * atm.gravities_layers[j - 1]))

    for j in range(level_1bar, n_layers):
        gg = target.target_gravity * (target.target_radius / (target.target_radius + atm.z[j]))**2
        sh = CST_R * atm.temperatures_layers[j] / (mu[j] * 1e-3 * gg)
        atm.z[j + 1] = atm.z[j] + sh * (log_p[j] - log_p[j + 1])
        atm.gravities_layers[j] = target.target_gravity * (
            target.target_radius / (target.target_radius + 0.5 * (atm.z[j] + atm.z[j + 1])))**2
        atm.scale_height[j] = (CST_R * atm.temperatures_layers[j]
                                / (mu[j] * 1e-3 * atm.gravities_layers[j]))


def _calculate_eddy_diffusion_coefficient(
    atm: Atmosphere, target: Target, spec: Species,
    retrieval: ExoremRetrieval,
    gases_vmr: np.ndarray,
    gases_c_p: np.ndarray,
    temperatures_thermo: np.ndarray,
    radiosity_internal_target: np.ndarray,
    flux_conv: np.ndarray,
) -> None:
    """Ackerman & Marley Kzz parameterisation."""
    n_layers = atm.n_layers
    T_int    = target.target_internal_temperature
    alpha    = max(min(1.0 + 2.0 * (1500.0 - T_int) / 1200.0, 3.0), 1.0)
    iconv = 0
    # diagnostics: store cpr and gr per layer so the harness can compare the
    # convective-flux inputs (the residual Kzz traces to flux_conv/cpr, not the
    # formula).  Filled in the loop below; last layer mirrors the previous one.
    atm.cpr_layers = np.zeros(n_layers)
    atm.gr_layers  = np.zeros(n_layers)

    for i in range(n_layers - 1):
        t_layer = math.sqrt(atm.temperatures_layers[i] * atm.temperatures_layers[i + 1])

        # c_p at this level
        cpr = sum(
            gases_vmr[k, i]
            * interp_ex_0d(t_layer, temperatures_thermo, gases_c_p[k, :])
            for k in range(N_GASES)
        ) / CST_R

        gr = (cpr * math.log(atm.temperatures_layers[i + 1] / atm.temperatures_layers[i])
              / math.log(atm.pressures_layers[i + 1] / atm.pressures_layers[i])) \
            if atm.pressures_layers[i + 1] != atm.pressures_layers[i] else 0.0
        atm.cpr_layers[i] = cpr
        atm.gr_layers[i]  = gr

        # scale_height is stored in METRES (= R·T/(μ·g) ≈ 1.4e5 m).  The Fortran
        # Kzz formula (exorem.f90 calculate_eddy_diffusion_coefficient) uses the
        # scale height in METRES throughout — every occurrence there is written
        # `scale_height(km)*1d3`.  The earlier code converted to km (H_km) and used
        # that, which left the prefactor and the flux term each 1e3× too small
        # (⇒ Kzz 1e4× too small) and the overshoot ratio 1e12× too small.  That
        # collapsed the radiative-zone Kzz and forced the CO/CH4 quench ~6 layers
        # too shallow (the dominant mid-atmosphere T error).  Use metres.
        H_m  = atm.scale_height[i]             # metres
        g_i  = atm.gravities_layers[i]
        p_i  = atm.pressures_layers[i]
        J_i  = radiosity_internal_target[i]

        # Flux-term units: the Fortran writes this term with gravity in CGS
        # (cm s⁻², = g_SI·100) and pressure as `1d2·pressures_layers` with
        # pressures_layers in mbar (= P in Pa).  In SI (g in m s⁻², p in Pa) the
        # Fortran's `*1d-2 ... /(1d2·p_mbar)` reduces exactly to `*g/p`, i.e. the
        # `1e-2` and `1e2` factors cancel the 100× CGS-gravity and the mbar→Pa
        # conversions.  Writing them literally (as before) left the flux argument
        # 1e4× too small ⇒ Kzz 21.5× too small (the residual after the H_km fix).
        if atm.eddy_mode == "Ackerman":
            ml = H_m * max(0.1, gr)
            kzz = (1e4 * H_m / 3.0
                   * (ml / H_m) ** (4.0 / 3.0)
                   * (1e-3 * J_i / cpr * H_m * g_i / p_i) ** (1.0 / 3.0))
            atm.eddy_diffusion_coefficient[i] = kzz

        elif atm.eddy_mode in ("AckermanConvective", "infinity"):
            ml  = H_m
            kzz = (1e4 * H_m / 3.0
                   * (ml / H_m) ** (4.0 / 3.0)
                   * (1e-3 * max(flux_conv[i], 1e-6 * radiosity_internal_target[0])
                      / cpr * H_m * g_i / p_i) ** (1.0 / 3.0))
            if flux_conv[i] > 1.0:
                iconv = i
                eddy_overshoot = 0.0
            else:
                kzz_conv = atm.eddy_diffusion_coefficient[iconv]
                # Fortran overshoot uses sqrt(gravities_layers/1d5) with gravity
                # in CGS (cm s⁻²); in SI (g in m s⁻²) that is sqrt(g/1e3), not /1e5.
                eddy_overshoot = (kzz_conv
                                  * (atm.scale_height[i] / atm.scale_height[iconv]) ** 2
                                  * (p_i / atm.pressures_layers[iconv])
                                  ** (alpha * math.sqrt(g_i / 1e3)))
            atm.eddy_diffusion_coefficient[i] = max(kzz, eddy_overshoot)

    atm.eddy_diffusion_coefficient[n_layers - 1] = atm.eddy_diffusion_coefficient[n_layers - 2]
    atm.cpr_layers[n_layers - 1] = atm.cpr_layers[n_layers - 2]
    atm.gr_layers[n_layers - 1]  = atm.gr_layers[n_layers - 2]


def _calculate_thermochemical_equilibrium(
    atm: Atmosphere, spec: Species,
    gases_vmr: np.ndarray,
    species_vmr_layers: np.ndarray,
    p_c_condensates, vmr_sat_condensates, vmr_c_condensates, layer_condensates,
    gases_molar_mass, gases_delta_g, condensates_delta_g,
    temperatures_thermo, gases_c_p,
    elements_in_gases: np.ndarray,
) -> None:
    """Run chemistry and update species_vmr_layers."""
    result = calculate_chemistry(
        at_equilibrium=bool(np.all(spec.species_at_equilibrium)),
        pressures_layers=atm.pressures_layers,
        temperatures_layers=atm.temperatures_layers,
        gravities_layers=atm.gravities_layers,
        eddy_diffusion_coefficient=atm.eddy_diffusion_coefficient,
        pressures=atm.pressures,
        scale_height=atm.scale_height,
        elemental_h_ratio=spec.elemental_h_ratio,
        temperatures_thermochemistry=temperatures_thermo,
        gases_delta_g=gases_delta_g,
        condensates_delta_g=condensates_delta_g,
        gases_c_p=gases_c_p,
        gases_molar_mass=gases_molar_mass,
        elements_in_gases=elements_in_gases,
        solar_h_ratio=getattr(spec, "solar_h_ratio", None),
    )

    gases_vmr[:] = result["gases_vmr"]

    for i, name in enumerate(spec.species_names):
        try:
            gas_idx = gas_id(name)
            species_vmr_layers[:, i] = gases_vmr[gas_idx, :]
        except ValueError:
            pass

    p_c_condensates[:]    = result["p_c_condensates"]
    vmr_sat_condensates[:] = result["vmr_sat_condensates"]
    vmr_c_condensates[:]  = result["vmr_c_condensates"]
    layer_condensates[:]  = result["layer_condensates"]


def _calculate_cloud_vmr(
    atm: Atmosphere, cloud_obj: Cloud,
    gases_molar_mass: np.ndarray,
    gases_vmr: np.ndarray,
    vmr_sat_condensates: np.ndarray,
    vmr_c_condensates: np.ndarray,
    layer_condensates: np.ndarray,
    p_c_condensates: np.ndarray,
    cloud_vmr: np.ndarray,
) -> None:
    """Compute cloud vertical mixing for each cloud species."""
    from .chemistry import CONDENSATE_NAMES, condensate_id
    n_layers = atm.n_layers

    # Mean molar mass per layer (g mol⁻¹)
    mu = np.zeros(n_layers)
    for k in range(N_GASES):
        mu += gases_vmr[k, :] * gases_molar_mass[k] * 1e3
    mu = np.maximum(mu, 1e-30)

    for ic in range(cloud_obj.n_clouds):
        name = cloud_obj.cloud_names[ic]
        pbot = 500.0   # mbar default

        # Match cloud to condensate
        ic2_match = None
        for ic2, cname in enumerate(CONDENSATE_NAMES):
            if cname.strip() == name.strip():
                ic2_match = ic2
                pbot = float(p_c_condensates[ic2]) * 1e3   # bar → mbar
                break

        layer_clouds_ic = layer_condensates[ic2_match] if ic2_match is not None else n_layers

        # Saturation VMR for this cloud
        cloud_vmr_sat = np.zeros(n_layers)
        q0 = 0.0
        if ic2_match is not None:
            for j in range(n_layers):
                cloud_vmr_sat[j] = max(
                    vmr_sat_condensates[ic2_match, j]
                    * cloud_obj.cloud_molar_mass[ic] / mu[j] * 1e3,
                    0.0)
            if layer_condensates[ic2_match] >= n_layers:
                q0 = cloud_vmr_sat[n_layers - 1]
            else:
                lc = layer_condensates[ic2_match]
                q0 = vmr_c_condensates[ic2_match] * cloud_obj.cloud_molar_mass[ic] / mu[lc] * 1e3

        radius_tmp = cloud_obj.cloud_particle_radius[ic, :].copy()

        if cloud_obj.cloud_mode == "fixedRadiusTime":
            q_cloud, radius_tmp, _, _ = calculate_cloud_mixing2(
                n_layers, atm.pressures, atm.temperatures,
                atm.pressures_layers, atm.temperatures_layers,
                atm.gravities_layers, atm.molar_masses_layers,
                int(layer_clouds_ic), float(pbot), cloud_vmr_sat,
                q0, atm.eddy_diffusion_coefficient,
                cloud_obj.cloud_mode, atm.eddy_mode,
                float(cloud_obj.sedimentation_parameter[ic]),
                radius_tmp,
                float(cloud_obj.cloud_particle_density[ic]),
                float(cloud_obj.supersaturation_parameter[ic]),
            )
        else:
            q_cloud, radius_tmp, _, _ = calculate_cloud_mixing(
                n_layers, atm.pressures, atm.temperatures,
                atm.pressures_layers, atm.temperatures_layers,
                atm.gravities_layers, atm.molar_masses_layers,
                int(layer_clouds_ic), float(pbot), cloud_vmr_sat,
                q0, atm.eddy_diffusion_coefficient,
                cloud_obj.cloud_mode, atm.eddy_mode,
                float(cloud_obj.sedimentation_parameter[ic]),
                radius_tmp,
                float(cloud_obj.cloud_particle_density[ic]),
            )

        cloud_vmr[ic, :] = np.maximum(q_cloud, 0.0)
        cloud_obj.cloud_particle_radius[ic, :] = radius_tmp


## DGRAD: the deep adiabat is set slightly super-adiabatic so that the
## convective-flux closure works.  This MUST match the Fortran reference
## (exorem.f90 line 992: `dgrad = 5d-3`).
##
## Why a NON-ZERO value is essential (root-cause fix):
##   Deep in a brown-dwarf interior, convection carries ~100 % of the
##   internal flux and radiation only ~0.3 %.  The energy balance there is
##   closed NOT by radiation but by the convective term in
##   _add_convective_term, which adds
##         conv_add = 1e3 * total_flux * (gr/grad_ad - 1)^2
##   to radiosity_internal.  A deep zone that sits a few percent
##   super-adiabatic (gr/grad_ad - 1 ~ 0.03) generates conv_add ~ target,
##   exactly closing the deep flux residual.
##
##   If DGRAD = 0 AND _init_adiabat is run every iteration (the previous
##   broken configuration), the deep zone is forced gr = grad_ad EXACTLY,
##   so (gr/grad_ad - 1) = 0, so conv_add = 0 and the convective matrix_t
##   coupling = 0.  The interior can then shed its internal heat by neither
##   channel; the unclosed residual is absorbed by the retrieval as a cold
##   bias in the photosphere and the emergent flux collapses
##   (J_int/(sigma T_int^4) ~ 0.26, T_eff ~ 356 K instead of 500 K).
##
##   The fix is therefore (a) DGRAD = 5e-3 to seed the super-adiabaticity,
##   and (b) call _init_adiabat ONCE at iter == n_non_adiabatic_iterations
##   (see the main loop) so the retrieval + convective term can let the
##   deep zone settle to the few-percent super-adiabaticity the closure
##   needs, exactly as the Fortran does.
DGRAD = 5.0e-3

# ----------------------------------------------------------------------------
# Upper-atmosphere well-posedness (optical-depth-aware retrieval).
#
# DIAGNOSIS (200K/500K/1000K runs): the converged profiles develop a large,
# spurious temperature inversion — a hot bump — in the upper atmosphere at a
# FIXED pressure (~100 Pa) regardless of T_int, reaching ~3x the expected
# temperature (and the T_MAX clip at 1000 K).  Tracing it iteration by
# iteration: the a-priori is fine (monotonic, isothermal skin), but during the
# early iterations the WHOLE upper atmosphere has rad_diff = +target (the net
# radiative flux is ~0 there because the internal flux is not yet transported
# up through the optically-thick interior) while the radiative Jacobian carries
# essentially NO information about those temperatures (a tenuous, optically-thin
# layer barely changes the net flux — verified by finite differences and by the
# row-norm of matrix_t: <1% leverage over tens of levels).  The optimal-
# estimation step is therefore ill-conditioned there and, chasing an
# unsatisfiable residual it has no leverage to reduce, proposes absurd ~10^4 K
# heating steps that the per-iteration rate limit only slows.  A hot bump grows,
# its extreme gradients corrupt the local Jacobian further, and the lack of
# leverage then FREEZES it.  A non-irradiated atmosphere has no stratosphere, so
# the inversion is unphysical.
#
# This is NOT cured by regularisation (we tested Levenberg-Marquardt, leverage-
# aware noise inflation, and the Rodgers a-priori-departure term on the real
# dumped matrices — the data-empty direction is near-singular and any residual
# component there blows up), so the data-empty levels must be taken OUT of the
# radiative inversion and set by the physical prior instead.
#
# FIX: weight each level's temperature update by its radiative information
# content, measured by the bulk (band-median) cumulative optical depth tau.
#   * Optically THICK levels (tau >> 1: the photosphere and the whole deep
#     convective interior) keep the full OE step — UNCHANGED.  This is the key
#     discriminator: the deep zone is ALSO low-leverage radiatively, but it is
#     optically thick and is closed by convection, so it must NOT be touched.
#   * Optically THIN levels (tau << 1: the data-empty upper atmosphere) have the
#     OE step faded out and relax toward the a-priori radiative-equilibrium
#     (Guillot) profile, which is monotonic and asymptotes to the skin
#     temperature.  "No information -> follow the physical prior" is the correct
#     treatment of a data-empty region; this is the principled version of the
#     disabled r47 skin-temperature pin (which pinned to a single T_skin with a
#     crude leverage threshold that over-grabbed sub-photospheric layers).
# The crossover is smooth (w = tau/(tau + TAU_INFO)), so there is no hard
# boundary / discontinuity to propagate.  Because the bump only ever seeds where
# tau is tiny, applying this from iteration 0 prevents it from forming at all,
# which is why it works where in-place regularisation of the already-diverged
# matrices does not.
# RETIRED (see Fortran reference output.csv for the 500 K case): relaxing the
# optically-thin zone toward the *grey* Guillot a-priori is wrong-headed.  The
# grey a-priori is isothermal at the grey skin temperature T_int*2^-1/4 = 420 K
# in the optically-thin region, but the real (non-grey) radiative zone is COLD
# with a steep gradient (the Fortran cools to ~173 K at ~20 Pa, rising to ~271 K
# at the top).  Relaxing toward 420 K flattens the Python's radiative zone into an
# isothermal slab, which carries NO net flux -> the internal flux is blocked and
# T_eff collapses to exactly T_skin (J_int/sigmaT^4 = 0.5).  Set False; the real
# fix must let the radiative zone reach its cold non-grey equilibrium.
UPPER_ATM_INFO_WEIGHTING = False
TAU_INFO = 0.3        # grey-Rosseland optical depth at the OE/prior crossover.
                      # w = tau/(tau+TAU_INFO): the photosphere is at tau=2/3, so
                      # TAU_INFO=0.3 places the crossover just above it — levels
                      # above the photosphere (the data-empty upper atmosphere)
                      # relax to the prior, the photosphere and deep zone keep OE.
APRIORI_RELAX = 0.5   # relaxation rate of the optically-thin zone toward the
                      # a-priori (per iteration); 0.5 ≈ a 2-iteration timescale.

# --- Convection optical-depth gate -----------------------------------------
# The Fortran's converged upper atmosphere is comfortably sub-adiabatic, so its
# add_convective_term (which fires on EVERY super-adiabatic interface, no margin,
# no contiguity) produces flux_conv = 0 above the photosphere.  The Python's
# retrieval, by contrast, transiently drives the optically-thin upper atmosphere
# super-adiabatic; the UNCAPPED conv_add = 1e3·total_flux·(gr/grad_ad − 1)² then
# self-reinforces (conv_add inflates radiosity_internal → retrieval cools to null
# it → steeper lapse → larger excess → larger conv_add), collapsing the upper
# atmosphere to the T_MIN floor (verified on T500/g26/M1: spurious flux_conv up
# to 1.57e4 W/m² at ~1e-3 bar where the Fortran is exactly 0).
#
# This is the runaway the r39 contiguity filter suppressed and the r40 reversion
# re-admitted.  Neither setting is right: r39 (contiguous-from-bottom) ALSO kills
# the *real* detached convective zone the Fortran has at ~2.8–0.07 bar (that is
# why it excluded L28); r40 restores that zone but re-admits the spurious one.
# The correct discriminator is not contiguity but OPTICAL DEPTH: real convective
# zones (deep + detached) sit at grey-Rosseland tau ≳ 0.1; the spurious runaway
# lives at tau ≪ 0.1, where the medium is transparent and convection is
# unphysical (Marley & McKay 1999; Hubeny & Mihalas 2014 §17.4 place the top of
# the convective zone at the photosphere region).  We therefore fire the
# convective coupling only where the (monotonic, photosphere-calibrated) grey
# tau at the interface's shallower (upper) boundary exceeds CONV_TAU_MIN.
#
# CAVEAT: CONV_TAU_MIN was calibrated on T500/g26/M1, where the real zone bottoms
# at tau(L30)=0.17 and the spurious zone tops at tau(L33)=0.086 — a clean gap.
# The grey tau adapts to T_int (κ∝T^1.8) and g, so this *should* generalise, but
# the threshold MUST be validated across the grid.  This gate matches the
# Fortran's *result* (flux_conv=0 in the thin zone); it does not address *why*
# the Python's thin zone goes super-adiabatic (the upstream seed — likely the
# iter-1 flux overshoot / the Tikhonov ridge perturbing the first step — remains
# open and is the truly faithful fix).
CONV_TAU_GATE = True   # gate convection on grey optical depth (kills thin-zone runaway)
CONV_TAU_MIN  = 0.1    # min grey-Rosseland tau (lower interface boundary) for convection

# --- Adiabatic-gradient H2-dissociation correction (Fortran corr_adia) -----
# The Fortran adiabatic gradient is gradiant = corr/(c_p/R + dcpr) with corr,
# dcpr from corr_adia() — the H2-dissociation correction (exorem.f90:712).  The
# Python port historically used the dissociation-free 1/(c_p/R), which is up to
# ~17% too steep in the deep hot zone (T >~ 2500 K, worst at low g / high
# T_int where the deep is hottest).  A too-steep deep adiabat both over-steepens
# the projected adiabat in _init_adiabat and raises the Schwarzschild trigger
# threshold (gr > grad_ad) in _add_convective_term, fragmenting/destabilising
# the deep convective zone.  True = faithful Fortran gradient (default).
# False = old dissociation-free gradient, for A/B testing.
CORR_ADIA = True

# --- Retrieval matrix conditioning -----------------------------------------
# The Fortran reference (exorem.f90 L2601-2618) inverts the bare observation
# covariance  M = K·S·Kᵀ + diag(rad_noise²)  with NO Tikhonov ridge.  The ridge
# (M += λ·trace(M)/n·I) was added here to damp the near-singular vertical modes
# of the optically-thin upper atmosphere, but it changes the gain away from the
# reference and was never what caused the upper-atmosphere runaway — that was
# the disabled 1.05·T_below inter-layer clamp (re-enabled in
# _temperature_profile_retrieval).  Default False to match Fortran exactly; flip
# to True only as a fallback if `matinv` reports ill-conditioning / non-finite
# gains on a particular case.
USE_TIKHONOV = False


def _kappa_ir_for_tau(T_int: float) -> float:
    """Rosseland-mean IR opacity (m^2/kg) at the photosphere; the same fit the
    Guillot a-priori uses (Freedman-like, kappa ~ 5e-4 (T_int/500)^1.8)."""
    if T_int <= 0.0:
        return 5.0e-4
    return 5.0e-4 * (T_int / 500.0) ** 1.8


def _representative_optical_depth(pressures: np.ndarray,
                                  T_int: float,
                                  gravity: float) -> np.ndarray:
    """Per-level GREY-ROSSELAND cumulative optical depth, tau = kappa_IR * P / g.

    This is deliberately NOT the band-median of the spectral tau: that statistic
    is dominated by strong molecular line cores and saturates within a couple of
    levels of the top, so it cannot locate the (flux-weighted) photosphere.  The
    grey-Rosseland tau used here is the same monotonic, flux-weighted opacity the
    Guillot a-priori is built on; it crosses 2/3 at the photosphere
    (P_photo = 2g/3kappa_IR) and falls smoothly to ~0 in the optically-thin upper
    atmosphere, which is exactly the discriminator the information weighting
    needs.  It is monotonic in pressure and immune to the bump's corruption of
    the spectral Jacobian.
    """
    kappa_ir = _kappa_ir_for_tau(T_int)
    g = gravity if gravity and gravity > 0.0 else 10.0
    return kappa_ir * pressures / g


def _check_adiabat_precondition(
    atm: Atmosphere,
    gases_vmr: np.ndarray,
    gases_c_p: np.ndarray,
    temperatures_thermo: np.ndarray,
) -> None:
    """Verify that the deep zone is super-adiabatic before _init_adiabat runs.

    r32 guard: _init_adiabat walks levels bottom-to-top and sets
    ``in_convective = False`` the first time it finds a sub-adiabatic layer,
    leaving everything above untouched.  If the *very first* comparison (j=1)
    is sub-adiabatic, the function exits immediately having done nothing.
    This function checks the first N_CHECK deep layers and logs a prominent
    warning if none is super-adiabatic, so the operator can diagnose the
    burn-in phase rather than spending 50 iterations puzzling over a
    persisting stratospheric bubble.

    It is diagnostic-only — it does not modify the atmosphere.
    """
    N_CHECK = 20  # inspect the bottom 20 levels (deepest ~3 decades in pressure)
    n_levels = atm.n_levels
    n_layers = atm.n_layers
    n_super = 0

    for j in range(1, min(N_CHECK + 1, n_levels)):
        T_below = atm.temperatures[j - 1]
        T_here  = atm.temperatures[j]

        if atm.pressures[j] == atm.pressures[j - 1] or T_here <= 0 or T_below <= 0:
            continue

        gr_rad = (math.log(T_below / T_here)
                  / math.log(atm.pressures[j - 1] / atm.pressures[j]))

        t_lay  = math.sqrt(T_below * T_here)
        lay_j  = min(j - 1, n_layers - 1)
        cpr    = sum(gases_vmr[k, lay_j]
                     * interp_ex_0d(t_lay, temperatures_thermo, gases_c_p[k, :])
                     for k in range(N_GASES)) / CST_R
        grad_ad = 1.0 / cpr if cpr > 0 else 0.0

        if grad_ad > 0 and gr_rad > grad_ad + DGRAD:
            n_super += 1

    if n_super == 0:
        print(
            f"\n  *** r32 WARNING: _init_adiabat precondition FAILED ***\n"
            f"  None of the bottom {N_CHECK} deep levels is super-adiabatic\n"
            f"  (gr_rad > grad_ad + DGRAD = {DGRAD:.3e} anywhere).\n"
            f"  _init_adiabat will exit at j=1 doing NOTHING and\n"
            f"  _add_convective_term will contribute zero matrix_t coupling.\n"
            f"  This reproduces the r30 hot-bubble pathology delayed to\n"
            f"  iter n_non_adiabatic_iterations.  Check that the 1.05*T\n"
            f"  burn-in clamp (removed in r32) has not been re-introduced,\n"
            f"  and that n_non_adiabatic_iterations is large enough for the\n"
            f"  radiative burn-in to build super-adiabatic excess in the\n"
            f"  deep zone before convection activates.\n"
        )
    else:
        print(
            f"  _init_adiabat precondition OK: {n_super}/{N_CHECK} deep levels"
            f" are super-adiabatic (gr > grad_ad + {DGRAD:.3e})."
        )


def _init_adiabat(
    atm: Atmosphere, target: Target,
    spec: Species,
    gases_vmr: np.ndarray,
    gases_c_p: np.ndarray,
    temperatures_thermo: np.ndarray,
    radiosity_internal_target: np.ndarray,
    verbose: bool = True,
) -> None:
    """Project the deep convective zone onto the adiabat, top-down from the RCB.

    Faithful port of the Fortran reference ``init_adiabat`` (exorem.f90
    lines 988-1057).  The essential features the previous Python version
    got wrong are restored here:

    1. **Direction / anchor.**  Walk the interfaces from the TOP downward.
       Leave radiative (sub-adiabatic) interfaces untouched.  At the first
       super-adiabatic interface (the radiative-convective boundary, RCB),
       and at every deeper interface, reset the *deeper* level to follow
       ``grad_ad + DGRAD`` from the *shallower* level above it.  The whole
       deep adiabat is therefore anchored to the radiative solution at the
       RCB and grown downward.  (The old version anchored at the deepest
       level ``T[0]`` and grew the adiabat *upward*, letting a wrong deep
       anchor dictate the entire column.)

    2. **Frequency.**  This is called exactly ONCE, at
       ``iteration == n_non_adiabatic_iterations`` (see the main loop),
       after the pure-radiative burn-in has established the photosphere.
       Thereafter the deep zone is governed by the retrieval and
       ``_add_convective_term``.

    3. **Slope.**  The projected gradient is ``grad_ad + DGRAD`` with
       ``DGRAD = 5e-3`` (not 0), seeding the few-percent super-adiabaticity
       that the convective-flux closure (conv_add) needs to carry the
       internal flux.

    Levels are ordered index 0 = deepest (highest P), index n_levels-1 =
    top (lowest P).  The top two levels are always left to the radiative
    solution (the top of the atmosphere is optically thin and never
    convective).
    """
    n_levels = atm.n_levels
    n_layers = atm.n_layers
    if verbose:
        print("  Setting adiabatic gradient below upper adiabatic level")

    def _grad_ad_at(idx_deep: int, idx_shallow: int,
                    corr_layer: Optional[int] = None) -> float:
        t_lay = math.sqrt(atm.temperatures[idx_deep] * atm.temperatures[idx_shallow])
        lay_j = min(idx_deep, n_layers - 1)
        cpr = sum(gases_vmr[k, lay_j]
                  * interp_ex_0d(t_lay, temperatures_thermo, gases_c_p[k, :])
                  for k in range(N_GASES)) / CST_R
        if CORR_ADIA:
            # Fortran init_adiabat (exorem.f90:1000-1004): gradiant = corr/(c_p/R
            # + dcpr).  In the main scan corr_adia uses the H2/H VMR at layer i
            # (1-indexed) = idx_shallow (0-indexed); at the deepest interface
            # (exorem.f90:1036-1038) it uses layer 1 (1-indexed) = idx_deep=0.
            lc = idx_shallow if corr_layer is None else corr_layer
            lc = min(lc, n_layers - 1)
            corr, dcpr = _corr_adia(t_lay,
                                    gases_vmr[_I_H2, lc],
                                    gases_vmr[_I_H,  lc])
            denom = cpr + dcpr
            return (corr / denom) if denom > 0 else 0.0
        return (1.0 / cpr) if cpr > 0 else 0.0

    level_max_adiabat = -1     # index of the RCB (top of the convective zone)
    n_forced = 0

    # iu = shallower level of the interface; iu-1 = deeper level being set.
    # Fortran:  do i = n_levels-1, 3, -1   ->   Python iu = n_levels-2 .. 2
    for iu in range(n_levels - 2, 1, -1):
        if atm.pressures[iu - 1] == atm.pressures[iu]:
            continue
        log_dp = math.log(atm.pressures[iu - 1] / atm.pressures[iu])
        gr = math.log(atm.temperatures[iu - 1] / atm.temperatures[iu]) / log_dp
        grad_ad = _grad_ad_at(iu - 1, iu)
        grad_target = grad_ad + DGRAD

        if grad_ad > 0.0 and gr > grad_target:
            if level_max_adiabat < 0:
                level_max_adiabat = iu          # first (shallowest) convective interface
            # project the DEEPER level onto the adiabat from the level above
            atm.temperatures[iu - 1] = atm.temperatures[iu] * (
                atm.pressures[iu - 1] / atm.pressures[iu]) ** grad_target
            n_forced += 1
        # else: radiative interface -> leave the deeper level unchanged

    # Deepest level (index 0) from index 1 (Fortran lines 1031-1057).
    if atm.pressures[0] != atm.pressures[1]:
        log_dp0 = math.log(atm.pressures[0] / atm.pressures[1])
        gr0 = math.log(atm.temperatures[0] / atm.temperatures[1]) / log_dp0
        grad_ad0 = _grad_ad_at(0, 1, corr_layer=0)
        grad_target0 = grad_ad0 + DGRAD
        if grad_ad0 > 0.0 and gr0 > grad_target0:
            if level_max_adiabat < 0:
                level_max_adiabat = 1
            atm.temperatures[0] = atm.temperatures[1] * (
                atm.pressures[0] / atm.pressures[1]) ** grad_target0
            n_forced += 1

    atm.temperatures_layers[:] = np.sqrt(
        atm.temperatures[:n_layers] * atm.temperatures[1:n_levels])

    if verbose:
        if n_forced > 0:
            print(f"    Adiabat at levels <= {level_max_adiabat}: forced "
                  f"{n_forced} deep level(s) onto grad_ad + DGRAD "
                  f"(DGRAD={DGRAD:.4f}), anchored at the RCB and built downward")
        else:
            print("    No super-adiabatic deep zone found; profile left to RT.")


# =====================================================================
# r45: convective adjustment (post-retrieval, enthalpy-conserving)
# =====================================================================
#
# Master toggle. Set False to disable and recover exact post-r42/r44
# behaviour for A/B testing.
#
# FORTRAN-FAITHFUL FIX: turned OFF.  This enthalpy-conserving adjustment
# does not exist in the published Fortran reference.  It is an extra hard
# adiabatic mixing step layered on top of the retrieval that competes with
# it.  With the convective-flux closure restored (DGRAD=5e-3, init_adiabat
# once, convective term gated), the retrieval + convective term handle the
# convective zone exactly as the Fortran does, so this is not needed and
# only adds a second, conflicting controller.  Re-enable only for A/B tests.
CONVECTIVE_ADJUSTMENT = False
# Target super-adiabaticity for the mixed convective zone. The zone is
# put on gr = grad_ad + DGRAD_ADJ so _add_convective_term (which fires on
# gr > grad_ad) still sees it as convective and contributes matrix_t
# coupling + conv flux. Keep small.
DGRAD_ADJ = 1.0e-3
# Trigger tolerance: only mix an interface whose gr exceeds the target by
# this relative margin. Prevents marginal re-triggering / non-convergence.
ADJ_TOL = 5.0e-3


def _convective_adjustment(
    atm: Atmosphere,
    gases_vmr: np.ndarray,
    gases_c_p: np.ndarray,
    temperatures_thermo: np.ndarray,
    n_layers: int,
    path_outputs=None,
    iteration: int = -1,
) -> dict:
    """Enthalpy-conserving convective adjustment, applied after the Newton
    step.

    Standard radiative-convective-equilibrium technique (Manabe & Strickler
    1964): wherever the post-retrieval profile is super-adiabatic
    (gr > grad_ad), real convection would mix the layers back onto the
    adiabat.  This routine does exactly that, conserving the enthalpy
    integral ∫ cp T dp within each convective region (the bottom boundary
    L=0 is treated as a fixed reservoir, so a region touching it exchanges
    heat with the interior — this is the physical convective heat flux).

    What it DOES:
      * removes super-adiabatic cold dips at the RCB (e.g. the L=26 dip
        that triggers the radiative hot-bump cascade)
      * lets the convective zone find its own top boundary each iteration
        (the RCB is no longer frozen at the apriori location)
      * keeps the mixed zone at grad_ad + DGRAD_ADJ so the convective
        Jacobian term still carries flux

    What it does NOT do (by design — these are convectively stable):
      * touch temperature inversions (T rising with altitude). The radiative
        hot bump is an inversion; convection cannot remove it directly.
        The hypothesis under test is that keeping the RCB continuous and
        the convective flux flowing will, over iterations, remove the
        retrieval's *incentive* to build the bump.

    Returns a diagnostics dict (also dumped to npz if path_outputs given).
    """
    n_levels = atm.n_levels
    P = atm.pressures
    T = atm.temperatures
    T_before = T.copy()

    # Pressure weight per level (half-layer above + half-layer below).
    dp = np.empty(n_levels)
    dp[0]  = 0.5 * (P[0] - P[1])
    dp[-1] = 0.5 * (P[-2] - P[-1])
    dp[1:-1] = 0.5 * (P[:-2] - P[2:])

    def _grad_ad_iface(j: int) -> float:
        """Chemistry-dependent adiabatic gradient at interface j-1|j."""
        t_lay = math.sqrt(T[j - 1] * T[j])
        lay_j = min(j - 1, n_layers - 1)
        cpr = sum(gases_vmr[k, lay_j]
                  * interp_ex_0d(t_lay, temperatures_thermo, gases_c_p[k, :])
                  for k in range(N_GASES)) / CST_R
        return 1.0 / cpr if cpr > 0 else 0.0

    def _cp_level(j: int) -> float:
        lay = min(j, n_layers - 1)
        return sum(gases_vmr[k, lay]
                   * interp_ex_0d(T[j], temperatures_thermo, gases_c_p[k, :])
                   for k in range(N_GASES))

    # Pre-compute level heat capacities (for enthalpy weights). Cheap.
    cp_lev = np.array([_cp_level(j) for j in range(n_levels)])
    w = cp_lev * dp                                   # enthalpy weight / level

    def _enthalpy() -> float:
        return float(np.sum(w * T))

    E_before = _enthalpy()

    regions = []           # list of (j0, j1, anchored_to_reservoir)
    n_passes = 0
    MAX_PASSES = 30
    for _pass in range(MAX_PASSES):
        n_passes = _pass + 1
        # local adiabatic gradients and target at every interface
        grad_ad = np.zeros(n_levels)
        target  = np.zeros(n_levels)
        gr      = np.zeros(n_levels)
        for j in range(1, n_levels):
            grad_ad[j] = _grad_ad_iface(j)
            target[j]  = grad_ad[j] + DGRAD_ADJ
            gr[j]      = (math.log(T[j - 1] / T[j])
                          / math.log(P[j - 1] / P[j]))
        # super-adiabatic interfaces (above target by tolerance)
        sup = np.zeros(n_levels, dtype=bool)
        for j in range(1, n_levels):
            if grad_ad[j] > 0 and gr[j] > target[j] * (1.0 + ADJ_TOL):
                sup[j] = True
        if not sup.any():
            break

        regions = []
        j = 1
        while j < n_levels:
            if sup[j]:
                # extend up through contiguous super-adiabatic interfaces
                j1 = j
                while j1 + 1 < n_levels and sup[j1 + 1]:
                    j1 += 1
                # extend down through adiabatic-or-steeper interfaces to a
                # stable anchor (level 0 or a clearly sub-adiabatic iface)
                j0 = j - 1
                while j0 > 0 and grad_ad[j0] > 0 and \
                        gr[j0] >= target[j0] * (1.0 - ADJ_TOL):
                    j0 -= 1
                # build adiabat shape a[k] = T[k]/T[j0] integrating local
                # target gradient across the region
                a = np.ones(j1 - j0 + 1)
                for idx, k in enumerate(range(j0 + 1, j1 + 1), start=1):
                    a[idx] = a[idx - 1] * (P[k] / P[k - 1]) ** target[k]
                seg = slice(j0, j1 + 1)
                wseg = w[seg]
                if j0 == 0:
                    # reservoir-anchored: pin T[0], exchange heat with interior
                    Tj0 = T[0]
                    anchored = True
                else:
                    # enthalpy-conserving within the region
                    Tj0 = float(np.sum(wseg * T[seg]) / np.sum(wseg * a))
                    anchored = False
                T[seg] = Tj0 * a
                regions.append((int(j0), int(j1), anchored))
                j = j1 + 1
            else:
                j += 1

    atm.temperatures_layers[:] = np.sqrt(
        atm.temperatures[:n_layers] * atm.temperatures[1:n_levels])

    E_after = _enthalpy()

    # RCB diagnostic: top of the contiguous convective region from the bottom
    rcb_level = 0
    for j in range(1, n_levels):
        ga = _grad_ad_iface(j)
        if ga <= 0:
            break
        grj = math.log(T[j - 1] / T[j]) / math.log(P[j - 1] / P[j])
        if grj >= ga * (1.0 - ADJ_TOL):
            rcb_level = j
        else:
            break

    info = {
        "n_passes": n_passes,
        "n_regions": len(regions),
        "regions": regions,
        "rcb_level": rcb_level,
        "rcb_pressure": float(P[rcb_level]) if rcb_level < n_levels else 0.0,
        "enthalpy_before": E_before,
        "enthalpy_after": E_after,
        "enthalpy_rel_change": (E_after - E_before) / E_before
                               if E_before != 0 else 0.0,
        "max_dT": float(np.max(np.abs(T - T_before))),
        "level_max_dT": int(np.argmax(np.abs(T - T_before))),
    }

    _dump_convective_adjustment(path_outputs, iteration, T_before, T.copy(),
                                atm.pressures, info)
    return info


def _dump_convective_adjustment(path_outputs, iteration, T_before, T_after,
                                 pressures, info) -> None:
    """Dump per-iteration convective-adjustment state for offline analysis."""
    if not DUMP_DIAGNOSTICS or path_outputs is None or iteration < 0:
        return
    from pathlib import Path
    p = Path(path_outputs)
    p.mkdir(parents=True, exist_ok=True)
    fn = p / f"retrieval_adjust_iter{iteration:03d}.npz"
    try:
        regions = np.array(info["regions"], dtype=float) if info["regions"] \
            else np.zeros((0, 3))
        np.savez_compressed(
            fn,
            T_before=T_before, T_after=T_after, pressures=pressures,
            dT=T_after - T_before,
            regions=regions,
            rcb_level=info["rcb_level"],
            rcb_pressure=info["rcb_pressure"],
            enthalpy_before=info["enthalpy_before"],
            enthalpy_after=info["enthalpy_after"],
            enthalpy_rel_change=info["enthalpy_rel_change"],
            n_passes=info["n_passes"],
            n_regions=info["n_regions"],
        )
    except Exception as e:
        print(f"  Warning: could not dump convective adjustment: {e}")


# =====================================================================
# r46: periodic radiative-zone jump smoother
# =====================================================================
#
# Motivation (user diagnosis, confirmed numerically): in the radiative zone
# above the RCB the OE retrieval has near-zero gain, so it parks heat in a
# cold-trough / hot-bump dipole instead of a smooth ~T_eff profile. The
# dipole is a stable local minimum of the broken gain structure;
# weight_apriori (tested at 0.1 and 10) does not dislodge it.
#
# This routine implements the "every now and again, smooth out the big jumps
# and let it reconverge" idea: every SMOOTH_INTERVAL iterations, find adjacent
# levels ABOVE the convective zone whose |ΔT| exceeds SMOOTH_DT_THRESHOLD and
# remove the discontinuity by enthalpy-conserving pairwise mixing. The
# retrieval then re-converges over the next interval.
#
# IMPORTANT physical caveat: the dipole region is genuinely heat-DEFICIENT
# (~37% below the radiative-equilibrium value — the cold trough's mass
# dominates), so pure enthalpy-conserving smoothing lands around ~300 K and
# relies on the retrieval re-adding heat. If the retrieval just rebuilds the
# dipole, set SMOOTH_APRIORI_BLEND > 0: the smoothed region is then blended
# toward the original Guillot apriori (which sits at the physically-correct
# ~420-530 K radiative-equilibrium value), restoring the missing heat.
# blend=0.0 is the pure-smoothing idea; blend=1.0 fully re-seeds broken
# regions with the radiative-equilibrium prior.
#
RADIATIVE_SMOOTHING = False     # SUPERSEDED by r47 (RADIATIVE_EQUILIBRIUM_CAP).
                                # Set True only for A/B testing the older heuristic.
SMOOTH_INTERVAL = 20            # apply every N iterations (used only if SMOOTHING True)
SMOOTH_DT_THRESHOLD = 150.0     # K; adjacent jump above this is "big"
SMOOTH_APRIORI_BLEND = 0.0      # 0 = pure smoothing; >0 restores heat from prior


def _smooth_radiative_jumps(
    atm: Atmosphere,
    gases_vmr: np.ndarray,
    gases_c_p: np.ndarray,
    temperatures_thermo: np.ndarray,
    n_layers: int,
    path_outputs=None,
    iteration: int = -1,
) -> dict:
    """Periodic enthalpy-conserving smoothing of large jumps in the radiative
    zone (above the convective zone). See r46 notes above."""
    n_levels = atm.n_levels
    P = atm.pressures
    T = atm.temperatures
    T_before = T.copy()

    dp = np.empty(n_levels)
    dp[0] = 0.5 * (P[0] - P[1])
    dp[-1] = 0.5 * (P[-2] - P[-1])
    dp[1:-1] = 0.5 * (P[:-2] - P[2:])

    # Identify the top of the contiguous convective zone (= RCB) so we only
    # smooth strictly above it (never disturb the adiabat/photosphere).
    def _grad_ad_iface(j):
        t_lay = math.sqrt(T[j - 1] * T[j])
        lay_j = min(j - 1, n_layers - 1)
        cpr = sum(gases_vmr[k, lay_j]
                  * interp_ex_0d(t_lay, temperatures_thermo, gases_c_p[k, :])
                  for k in range(N_GASES)) / CST_R
        return 1.0 / cpr if cpr > 0 else 0.0

    rcb_level = 0
    for j in range(1, n_levels):
        ga = _grad_ad_iface(j)
        if ga <= 0:
            break
        grj = math.log(T[j - 1] / T[j]) / math.log(P[j - 1] / P[j])
        if grj >= ga * (1.0 - ADJ_TOL):
            rcb_level = j
        else:
            break

    lo = rcb_level + 1
    n_smoothed = 0
    max_jump_before = 0.0
    if lo < n_levels - 1:
        jumps0 = np.abs(np.diff(T[lo:]))
        max_jump_before = float(jumps0.max()) if jumps0.size else 0.0
        # enthalpy-conserving removal of the largest jump, repeated
        for _ in range(2000):
            jumps = np.abs(np.diff(T[lo:]))
            if jumps.size == 0 or jumps.max() < SMOOTH_DT_THRESHOLD:
                break
            k = int(np.argmax(jumps)) + lo + 1     # interface between k-1|k
            wm = ((dp[k - 1] * T[k - 1] + dp[k] * T[k])
                  / (dp[k - 1] + dp[k]))
            T[k - 1] = wm
            T[k] = wm
            n_smoothed += 1

        # optional blend toward the radiative-equilibrium prior to restore the
        # heat deficit (the cold trough cannot be warmed by redistribution
        # alone).
        if SMOOTH_APRIORI_BLEND > 0.0:
            Tap = getattr(atm, "temperatures_apriori_original", None)
            if Tap is not None and len(Tap) == n_levels:
                b = SMOOTH_APRIORI_BLEND
                T[lo:] = (1.0 - b) * T[lo:] + b * Tap[lo:]

    atm.temperatures_layers[:] = np.sqrt(
        atm.temperatures[:n_layers] * atm.temperatures[1:n_levels])

    jumps_after = np.abs(np.diff(T[lo:])) if lo < n_levels - 1 else np.array([0.0])
    info = {
        "rcb_level": rcb_level,
        "lo": lo,
        "n_smoothed": n_smoothed,
        "max_jump_before": max_jump_before,
        "max_jump_after": float(jumps_after.max()) if jumps_after.size else 0.0,
        "apriori_blend": SMOOTH_APRIORI_BLEND,
        "max_dT": float(np.max(np.abs(T - T_before))),
    }

    if DUMP_DIAGNOSTICS and path_outputs is not None and iteration >= 0:
        try:
            from pathlib import Path
            pth = Path(path_outputs)
            pth.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pth / f"retrieval_smooth_iter{iteration:03d}.npz",
                T_before=T_before, T_after=T.copy(), pressures=P,
                dT=T - T_before, rcb_level=rcb_level, lo=lo,
                n_smoothed=n_smoothed,
                max_jump_before=info["max_jump_before"],
                max_jump_after=info["max_jump_after"],
                apriori_blend=SMOOTH_APRIORI_BLEND,
            )
        except Exception as e:
            print(f"  Warning: could not dump radiative smoothing: {e}")

    return info


# =====================================================================
# r47: data-empty upper atmosphere held at radiative-equilibrium T_skin
# =====================================================================
#
# Three-region treatment of the atmosphere, made explicit:
#
#   1. Deep convective zone (τ ≫ 1, ∇_rad > ∇_ad):
#         T follows the adiabat anchored by T_int (entropy fixed by interior).
#         Handled by _init_adiabat + _convective_adjustment (r45).
#
#   2. Radiative photosphere (τ ~ 0.1 to 10):
#         Radiative Jacobian dF_outgoing/dT_layer is large — the OE
#         retrieval has real information here.  Let it do its job.
#
#   3. Optically thin upper atmosphere (τ ≪ 1):
#         Jacobian → 0; the OE cannot find T from the flux.  T is instead
#         set by *local radiative equilibrium*:  emission B(T) must equal
#         the mean intensity J of the upcoming radiation.
#
#         For non-irradiated atmospheres (F_down at TOA ≈ 0):
#             J ≈ F_up / 2 ≈ σ T_int⁴ / 2
#             ⇒ T_skin = T_int · 2^(-1/4)  ≈ 0.841 T_int
#
#         For T_int = 500 K this gives T_skin ≈ 420 K — exactly where
#         the Guillot apriori sits, and exactly where blend=1.0 was
#         pulling things via the back door.
#
# This routine implements region 3 properly:
#   - Identify "data-empty" layers ABOVE the convective zone (Jacobian
#     column L1 norm below REQ_SENSITIVITY_FRAC · max).
#   - Relax their T toward T_skin with under-relaxation (REQ_RELAX) so
#     the retrieval can still compete in the transition zone.
#
# This is the physics-correct version of what r46 (apriori-blend > 0)
# was approximating.  With r47 enabled, r46 is redundant and is now off
# by default — they can be tested independently.
#
# Limitations / assumptions (made explicit so we don't pretend more than
# the math gives us):
#   - Assumes F_down ≈ 0 in the upper atmosphere (true for non-irradiated;
#     for irradiated atmospheres the formula needs J = (F_up + F_down)/2
#     with the actual TOA downwelling included).
#   - Assumes a single skin T applies (i.e. negligible attenuation between
#     data-empty layers, so they all see roughly the same upcoming J).
#     For non-gray molecular bands a per-band/per-layer treatment would
#     be more accurate but requires per-band fluxes from the RT solver.
#
#
# FORTRAN-FAITHFUL FIX: turned OFF.  This skin-temperature pin does not
# exist in the published Fortran reference.  Its 5% Jacobian-leverage
# threshold over-grabs sub-photospheric layers (it cuts INTO the
# photospheric leverage peak) and drags 450-500 K layers toward
# T_skin ~ 420 K, directly suppressing the emergent flux.  The data-empty
# upper atmosphere is handled instead by the loose deep / tight top
# rad_noise weighting, exactly as in the Fortran.  Re-enable only for
# A/B tests.
RADIATIVE_EQUILIBRIUM_CAP = False
REQ_SENSITIVITY_FRAC = 0.05    # layer is "data-empty" if sens(j) < this · max(sens)
REQ_RELAX = 0.3                # under-relaxation: T_new = (1−α)·T + α·T_skin


def _apply_skin_temperature(
    atm: Atmosphere,
    n_layers: int,
    matrix_t: np.ndarray,
    radiosity_internal_target: np.ndarray,
    convective_top: int = 0,
    path_outputs=None,
    iteration: int = -1,
) -> dict:
    """Hold data-empty upper-atmosphere layers at the radiative-equilibrium
    skin temperature.  See r47 notes above."""
    n_levels = atm.n_levels
    T = atm.temperatures
    T_before = T.copy()
    # NB: exorem stores fluxes in cgs (erg s^-1 cm^-2), so the matching
    # Stefan-Boltzmann constant is sigma_cgs = 5.6704e-5 (NOT sigma_SI = 5.6704e-8).
    # Using SI here gave T_skin = 2364 K instead of 420 K — a 1000x units
    # bug that effectively cooked the upper atmosphere.
    sigma_SB = 5.670374419e-5

    # --- 1) Per-layer Jacobian sensitivity (L1 norm of matrix_t column) ---
    sens = np.zeros(n_levels)
    nT = matrix_t.shape[1]
    for j in range(n_levels):
        if j < nT:
            sens[j] = float(np.sum(np.abs(matrix_t[:, j])))
    max_sens = float(sens.max()) if sens.size and sens.max() > 0 else 1.0
    threshold = REQ_SENSITIVITY_FRAC * max_sens

    # --- 2) Skin temperature from the internal-flux target ---
    # σT_int⁴ = radiosity_internal_target[0]  (the model's energy-flux setpoint)
    F_int = float(radiosity_internal_target[0])
    T_int4 = F_int / sigma_SB if F_int > 0 else 0.0
    T_skin = (T_int4 / 2.0) ** 0.25 if T_int4 > 0 else 0.0

    # --- 3) Identify data-empty layers ABOVE the convective zone ---
    is_empty = np.zeros(n_levels, dtype=bool)
    for j in range(max(convective_top + 1, 0), n_levels):
        if sens[j] < threshold:
            is_empty[j] = True

    # --- 4) Relax those layers toward T_skin ---
    alpha = REQ_RELAX
    n_affected = 0
    for j in range(n_levels):
        if is_empty[j]:
            T[j] = (1.0 - alpha) * T[j] + alpha * T_skin
            n_affected += 1

    atm.temperatures_layers[:] = np.sqrt(
        atm.temperatures[:n_layers] * atm.temperatures[1:n_levels])

    first_empty = int(np.argmax(is_empty)) if n_affected > 0 else -1
    last_empty = (n_levels - 1 - int(np.argmax(is_empty[::-1]))) if n_affected > 0 else -1
    info = {
        "n_affected": n_affected,
        "T_skin": T_skin,
        "threshold_sens": float(threshold),
        "max_sens": float(max_sens),
        "max_dT": float(np.max(np.abs(T - T_before))),
        "first_empty_level": first_empty,
        "last_empty_level": last_empty,
        "convective_top": int(convective_top),
        "alpha": float(alpha),
    }

    if DUMP_DIAGNOSTICS and path_outputs is not None and iteration >= 0:
        try:
            from pathlib import Path
            pth = Path(path_outputs)
            pth.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pth / f"retrieval_skin_iter{iteration:03d}.npz",
                T_before=T_before, T_after=T.copy(),
                pressures=atm.pressures,
                sensitivity=sens, threshold=threshold,
                is_empty=is_empty, T_skin=T_skin, alpha=alpha,
                convective_top=convective_top,
                n_affected=n_affected,
            )
        except Exception as e:
            print(f"  Warning: could not dump skin temperature: {e}")

    return info


def _add_convective_term(
    atm: Atmosphere, target: Target, spec: Species,
    gases_vmr: np.ndarray,
    gases_c_p: np.ndarray,
    temperatures_thermo: np.ndarray,
    radiosity_internal_target: np.ndarray,
    radiosity_internal: np.ndarray,
    matrix_t: np.ndarray,
    light: LightSource,
    spectrometrics: Spectrometrics,
    retrieval=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add the convective term to the Jacobian matrix."""
    n_levels = atm.n_levels
    n_layers = atm.n_layers

    total_flux = (radiosity_internal_target[n_levels - 1]
                  + float(np.sum(light.irradiance) * PI * spectrometrics.wavenumber_step))

    flux_conv = np.zeros(n_levels)
    dt_conv   = np.zeros(n_levels)

    # ------------------------------------------------------------------
    # r39: identify the contiguous *bottom-up* convective region first.
    #
    # The previous loop fired the convective adjustment on *every* interface
    # where gr > grad_ad, anywhere in the column.  In the well-converged
    # Fortran reference, the radiative upper atmosphere is comfortably
    # sub-adiabatic, so this never triggers there.  In the current Python
    # state the upper atm settles at a lapse rate within ~5% of grad_ad
    # (T(10 Pa)=921 K, T(0.1 Pa)=320 K → gr ≈ 0.30, grad_ad ≈ 0.306) and
    # ~12 interfaces above the photosphere are marginally super-adiabatic.
    # Each contributes 1e3·total_flux·excess² to radiosity_internal[j] and
    # radiosity_internal[j-1].  At the top level (n_levels-1) the
    # contribution is ~0.7×σT_int_th⁴, inflating the retrieval's
    # rad_diff target and creating a self-reinforcing fixed point where:
    #   * radiosity_internal[-1]  ≈ 1.04·σT_int_th⁴   ← reported  (=T_int)
    #   * Σ spectral_radiosity[-1,:]·dwn ≈ 0.29·σT_int_th⁴ (=T_eff)
    # The retrieval, seeing T_int ≈ target, makes no further corrections
    # and never escapes the bad profile.  The "jolt" at the final iter
    # (T_int → 368 K) is just the post-loop convergence printer reading
    # the fresh, un-inflated radiosity_internal.
    #
    # Physical fix: convection is by definition a *contiguous* region
    # starting from the deepest super-adiabatic interface and extending
    # upward until the first sub-adiabatic interface is encountered.
    # Above that point the atmosphere is radiative and conv_add must not
    # fire — even if local noise pushes gr slightly above grad_ad.  This
    # matches Schwarzschild's criterion as applied in standard 1-D RC
    # codes (e.g. Marley & McKay 1999; Hubeny & Mihalas 2014, §17.4).
    #
    # We also add a small margin on grad_ad (SUPER_ADIAB_TOL) so that
    # near-adiabatic radiative layers are not flipped into convection by
    # numerical noise (the natural lapse rate of a grey atmosphere at the
    # top of the convective zone is exactly grad_ad).
    # ------------------------------------------------------------------
    # r40: REVERT the contiguous-zone filter and the 1.005 tolerance.
    #
    # Matrix dumps from a 50-iter instrumented run (see FINDINGS.md)
    # showed the filter consistently excludes the *same five interfaces*
    # from the convective Jacobian:
    #     L=27 excluded 27/50 iters
    #     L=28 excluded 38/50 iters   (cold pocket centre)
    #     L=29 excluded 33/50 iters
    #     L=77 excluded 47/50 iters
    #     L=79 excluded 48/50 iters   (TOA sawtooth)
    # These are exactly the levels where the failures sit. Fortran
    # exorem.f90's add_convective_term fires on every super-adiabatic
    # interface with no margin and no contiguity requirement, and does
    # not exhibit these failures.
    #
    # The original justification for the filter (suppress noise-driven
    # convection in the r28-r30 stratospheric hot bubble pathology) no
    # longer applies: r34+ added per-iter _init_adiabat enforcement of
    # the deep zone, and the per-layer rate limit (MAX_DT_FRAC) plus
    # T_MIN/T_MAX clip handle the runaway cases the filter was
    # protecting against. With the Guillot apriori the iter-0 upper-atm
    # state is no longer at the threshold of spurious super-adiabaticity.
    #
    # Restore the Fortran-equivalent: fire conv coupling at every
    # interface with gr > grad_ad. No SUPER_ADIAB_TOL, no contiguity
    # requirement.
    # ------------------------------------------------------------------
    SUPER_ADIAB_TOL = 1.0       # match Fortran: gr > grad_ad, no margin
    is_conv_interface = np.zeros(n_levels, dtype=bool)
    gr_arr      = np.zeros(n_levels)
    grad_ad_arr = np.zeros(n_levels)

    for j in range(1, n_levels):
        if atm.pressures[j] == atm.pressures[j - 1]:
            continue
        gr_arr[j] = (math.log(atm.temperatures[j - 1] / atm.temperatures[j])
                     / math.log(atm.pressures[j - 1] / atm.pressures[j]))
        t_lay = math.sqrt(atm.temperatures[j - 1] * atm.temperatures[j])
        lay_j = min(j - 1, n_layers - 1)
        cpr_j = sum(gases_vmr[k, lay_j]
                    * interp_ex_0d(t_lay, temperatures_thermo, gases_c_p[k, :])
                    for k in range(N_GASES)) / CST_R
        if CORR_ADIA:
            # Fortran exorem.f90:374-378: gradiant = corr/(c_p/R + dcpr), with
            # corr_adia taking the H2/H VMR at layer min(j, n_layers) (1-indexed)
            # = min(j, n_layers-1) (0-indexed) — one layer shallower than the
            # c_p layer (j-1).  Replicates the Fortran's layer-index offset.
            lay_c = min(j, n_layers - 1)
            corr, dcpr = _corr_adia(t_lay,
                                    gases_vmr[_I_H2, lay_c],
                                    gases_vmr[_I_H,  lay_c])
            denom = cpr_j + dcpr
            grad_ad_arr[j] = corr / denom if denom > 0 else 0.0
        else:
            grad_ad_arr[j] = 1.0 / cpr_j if cpr_j > 0 else 0.0

    # Fortran-equivalent: every super-adiabatic interface gets conv coupling.
    for j in range(1, n_levels):
        if grad_ad_arr[j] > 0 and gr_arr[j] > grad_ad_arr[j]:
            is_conv_interface[j] = True

    # --- Optical-depth gate (see CONV_TAU_GATE block at module level) ---------
    # Suppress convection where the medium is optically thin: the spurious
    # thin-zone runaway lives at grey tau << 0.1, while the real deep + detached
    # convective zones sit at tau >~ 0.1.  An interface reaches into the thin
    # zone when its SHALLOWER (upper, level-j) boundary is optically thin, so we
    # gate on tau[j] — the smaller of the two boundary taus (monotonic tau ⇒
    # tau[j-1] ≥ tau[j]).  Gating on the deeper boundary tau[j-1] instead is too
    # lenient: it keeps interfaces that straddle the photosphere (deep boundary
    # thick, top thin), which on the full grid is exactly where the catastrophic
    # spurious spikes sit (e.g. T1000/g50/M0: 6e6 W/m²; T700/g25/M0: 2.5e5 W/m²,
    # both at interfaces whose upper boundary is thin but whose base is not).
    # Validated against the Fortran convective zones across the noirr grid: tau[j]
    # < 0.1 suppresses only Fortran-radiative interfaces in every cell except a
    # single small (≤90 W/m²) interface at the top of T400/g50/M1's real zone,
    # which extends anomalously to tau≈0.06 — acceptable collateral.
    if CONV_TAU_GATE and is_conv_interface.any():
        tau_grey = _representative_optical_depth(
            atm.pressures,
            target.target_internal_temperature,
            target.target_gravity)
        n_gated = 0
        for j in range(1, n_levels):
            if is_conv_interface[j] and tau_grey[j] < CONV_TAU_MIN:
                is_conv_interface[j] = False
                n_gated += 1
        if n_gated > 0:
            _it = getattr(retrieval, "_current_iteration", -1) if retrieval is not None else -1
            print(f"  Conv tau-gate: suppressed {n_gated} optically-thin "
                  f"super-adiabatic interface(s) (tau < {CONV_TAU_MIN:g}).")


    # Track the deepest convective interface for the bottom-boundary special case.
    gr      = 0.0
    grad_ad = 0.0

    # FORTRAN-FAITHFUL: no cap on the super-adiabatic excess.
    # The Fortran reference (exorem.f90 lines 381-399) uses the raw
    # (gr/grad_ad - 1) in both the matrix_t coupling (linear) and the
    # conv_add radiosity contribution (quadratic).  The conv_add term is
    # what closes the deep energy balance: a deep zone sitting ~3%
    # super-adiabatic gives conv_add ~ target, supplying the internal flux
    # radiation cannot carry.  Capping the excess (the previous r42 patch)
    # throttles this closure and was only introduced to mask the runaway
    # that the DGRAD=0 / init-adiabat-every-iteration bug produced; with
    # that bug fixed the excess stays small on its own and no cap is needed.
    # Transient excursions are bounded by the retrieval's MAX_DT_FRAC rate
    # limit and the T_MIN/T_MAX clips, exactly as in the Fortran.

    for j in range(n_levels - 1, 0, -1):
        if not is_conv_interface[j]:
            continue

        gr      = gr_arr[j]
        grad_ad = grad_ad_arr[j]

        log_dp = math.log(atm.pressures[j - 1] / atm.pressures[j])
        excess = gr / grad_ad - 1.0                       # raw excess (no cap)
        coeff  = 2e3 * total_flux * excess / (grad_ad * log_dp)

        matrix_t[j - 1, j]     += coeff / atm.temperatures[j - 1]
        matrix_t[j, j]         -= coeff / atm.temperatures[j]
        matrix_t[j, j - 1]     -= coeff / atm.temperatures[j]
        matrix_t[j - 1, j - 1] += coeff / atm.temperatures[j - 1]

        conv_add = 1e3 * total_flux * excess**2
        radiosity_internal[j]     += conv_add
        radiosity_internal[j - 1] += conv_add
        flux_conv[j]     += conv_add
        flux_conv[j - 1] += conv_add

    # Bottom-boundary special case (Fortran exorem.f90 lines 406-411): if
    # the deepest interface (j = 1) is convective, add the same conv_add to
    # level 0.  Skip entirely when no convective zone exists (gr/grad_ad = 0).
    if grad_ad > 0 and gr > grad_ad and is_conv_interface[1]:
        excess_b     = gr / grad_ad - 1.0                 # raw excess (no cap)
        conv_add_b = 1e3 * total_flux * excess_b * excess_b
        radiosity_internal[0] += conv_add_b
        flux_conv[0]          += conv_add_b
        log_dp_b   = math.log(atm.pressures[0] / atm.pressures[1])
        coeff_b    = 2e3 * total_flux * excess_b / (grad_ad * log_dp_b)
        matrix_t[0, 0] += coeff_b / atm.temperatures[0]

    # --- Conv-zone diagnostic dump (no behavioural effect) ---
    if retrieval is not None:
        _iter = getattr(retrieval, "_current_iteration", -1)
        _dump_conv_diagnostic(
            getattr(retrieval, "_debug_path", None), _iter,
            gr_arr, grad_ad_arr, is_conv_interface,
            atm.pressures, atm.temperatures, SUPER_ADIAB_TOL,
        )

    return flux_conv, dt_conv, matrix_t


# ===========================================================================
# Output helpers
# ===========================================================================


def _print_convergence_info(
    atm, spectrometrics, light, target,
    radiosity_internal, radiosity_internal_target,
    spectral_radiosity, retrieval_converged, retrieval,
) -> None:
    print()
    if retrieval_converged:
        print(f"Info: tolerance reached ({retrieval.retrieval_tolerance:.2e})")
    else:
        print("Info: max number of iterations reached")

    J_int = radiosity_internal[-1]
    J_tgt = radiosity_internal_target[0]
    print(f"  J_int / (sigma * T_int_th^4) = {J_int / J_tgt:.6g}")

    T_int = math.copysign(
        (abs(J_int) / (CST_SIGMA * 1e3)) ** 0.25, J_int)
    print(f"  T_int = {T_int:.2f} K  (T_int_th = {target.target_internal_temperature:.2f} K)")

    T_eq = (float(np.sum(light.irradiance) * PI * spectrometrics.wavenumber_step)
            / (CST_SIGMA * 1e3)) ** 0.25
    print(f"  T_eq  = {T_eq:.2f} K")

    T_eff = (float(np.sum(spectral_radiosity[-1, :]) * spectrometrics.wavenumber_step)
             / (CST_SIGMA * 1e3)) ** 0.25
    print(f"  T_eff = {T_eff:.2f} K")


def _print_iteration_summary(
    iteration, atm, target, spectrometrics, light,
    radiosity_internal, radiosity_internal_target,
    spectral_radiosity, tau_cloud, cloud_vmr, cloud_obj,
    chi2_0, chi2_1, n_retrieved,
) -> None:
    for ic in range(cloud_obj.n_clouds):
        name = cloud_obj.cloud_names[ic]
        tau_max = tau_cloud[ic].max() if ic < tau_cloud.shape[0] else 0.0
        vmr_max = cloud_vmr[ic].max() if ic < cloud_vmr.shape[0] else 0.0
        print(f"  {name} cloud: tau_max = {tau_max:.3e}  "
              f"VMR_max = {vmr_max:.3e}")

    J_int = radiosity_internal[-1]
    J_tgt = radiosity_internal_target[0]
    print(f"\n  J_int / (sigma * T_int_th^4) = {J_int / J_tgt:.6g}")

    T_int = math.copysign((abs(J_int) / (CST_SIGMA * 1e3)) ** 0.25, J_int)
    print(f"  T_int = {T_int:.2f} K  (T_int_th = {target.target_internal_temperature:.2f} K)")

    T_eq = (float(np.sum(light.irradiance) * PI * spectrometrics.wavenumber_step)
            / (CST_SIGMA * 1e3)) ** 0.25
    print(f"  T_eq  = {T_eq:.2f} K")

    T_eff = (float(np.sum(spectral_radiosity[-1, :]) * spectrometrics.wavenumber_step)
             / (CST_SIGMA * 1e3)) ** 0.25
    print(f"  T_eff = {T_eff:.2f} K")

    chi2_var = (chi2_0 - chi2_1) / chi2_1 if chi2_1 != 0 else 0.0
    print(f"  Chi2 = {chi2_0:.4e}  Chi2_var = {chi2_var:.4e}  ({n_retrieved} pts)")


def _write_outputs(
    opts, atm, spec, spectrometrics,
    spectral_radiosity, spectral_radius, d_spectral_radius,
    species_vmr_layers, cloud_vmr, cloud_obj,
    tau, tau_cloud, flux, flux_clear, flux_cloud,
    *,
    target=None, gases_vmr=None,
    radiosity_internal=None, radiosity_internal_target=None,
    matrix_t=None, chi2_0=None,
) -> None:
    """
    Write the converged model results.

    Mirrors the Fortran ``if (output_hdf5) call write_hdf5_output else call
    write_outputs`` logic: by default (``output_hdf5`` true) a single HDF5 file
    is produced in the original Fortran group layout; set ``output_hdf5`` false
    in the namelist to fall back to the flat ``.dat`` text files instead.
    """
    out_dir = Path(opts.get("path_outputs", "./outputs"))
    suffix  = opts.get("output_files_suffix", "")
    wn = spectrometrics.wavenumbers

    # Fortran writes the HDF5 when output_hdf5 is set, and the text files
    # otherwise.  Default to the HDF5 so the deliverables match the reference.
    if not opts.get("output_hdf5", True):
        sp_file = out_dir / (opts.get("spectrum_file_prefix", "spectrum") + suffix + ".dat")
        write_spectrum(sp_file, wn, spectral_radiosity[-1, :])
        tp_file = out_dir / (opts.get("temperature_profile_file_prefix", "tp") + suffix + ".dat")
        write_temperature_profile(tp_file, atm.pressures, atm.temperatures)
        vmr_file = out_dir / (opts.get("vmr_file_prefix", "vmr") + suffix + ".dat")
        write_vmr_profile(vmr_file, atm.pressures_layers,
                          spec.species_names, species_vmr_layers)
        return

    # ------------------------------------------------------------------
    # HDF5 in the original Fortran layout.
    # The Python model state is already SI, so arrays are passed unscaled;
    # only the radiosity is CGS internally and is converted to W m-2 (x1e-3),
    # exactly as the Fortran does.
    # ------------------------------------------------------------------
    from .physics import ELEMENTS_SYMBOL, CST_SIGMA
    from .chemistry import GASES_NAMES

    n_lay = atm.n_layers

    # absorber VMRs (layer grid), keyed by absorber name
    absorbers_vmr = {}
    abs_names = set()
    for i, nm in enumerate(spec.species_names):
        nm = str(nm)
        if nm:
            abs_names.add(nm)
            absorbers_vmr[nm] = species_vmr_layers[:, i]

    # non-absorber gas VMRs (layer grid), keyed by gas name
    gases_vmr_map = {}
    if gases_vmr is not None:
        for j, nm in enumerate(GASES_NAMES):
            if j >= gases_vmr.shape[0] or nm in abs_names:
                continue
            gases_vmr_map[nm] = gases_vmr[j, :n_lay]

    # elemental abundances (X/H) and the solar reference, for elements present
    elemental_abundances, solar_abundances = {}, {}
    ehr = getattr(spec, "elemental_h_ratio", None)
    shr = getattr(spec, "solar_h_ratio", None)
    if ehr is not None:
        for i, sym in enumerate(ELEMENTS_SYMBOL):
            if i < len(ehr) and ehr[i] > 0.0:
                elemental_abundances[sym] = float(ehr[i])
                if shr is not None and i < len(shr) and shr[i] > 0.0:
                    solar_abundances[sym] = float(shr[i])

    # run-quality scalars
    ratio = tint = None
    if radiosity_internal is not None and radiosity_internal_target is not None:
        ratio = float(radiosity_internal[-1] / radiosity_internal_target[-1])
    if radiosity_internal is not None:
        ri = float(radiosity_internal[-1])
        tint = float(np.sign(ri) * (abs(ri) / (CST_SIGMA * 1e3)) ** 0.25)
    chi2_val = float(chi2_0) if (chi2_0 is not None and np.isfinite(chi2_0)) else None

    mmm = (float(np.mean(atm.molar_masses_layers))
           if getattr(atm, "molar_masses_layers", None) is not None else None)
    mol_lay = (atm.molar_masses_layers[:n_lay]
               if getattr(atm, "molar_masses_layers", None) is not None else None)

    payload = {
        # run quality
        "radiosity_target_ratio":        ratio,
        "actual_internal_temperature":   tint,
        "chi2_retrieval":                chi2_val,
        # levels (SI as-is; radiosity CGS -> W m-2)
        "levels_pressure_Pa":            atm.pressures,
        "levels_temperature_K":          atm.temperatures,
        "levels_radiosity_W_m2":         (radiosity_internal * 1e-3
                                          if radiosity_internal is not None else None),
        "levels_altitude_m":             getattr(atm, "z", None),
        "kernel_temperature":            matrix_t,
        # layers
        "layers_pressure_Pa":            atm.pressures_layers,
        "layers_temperature_K":          atm.temperatures_layers,
        "layers_gravity_m_s2":           getattr(atm, "gravities_layers", None),
        "layers_molar_mass_kg_mol":      mol_lay,
        "mean_molar_mass_kg_mol":        mmm,
        "layers_eddy_diffusion_coefficient_cm2_s": getattr(atm, "eddy_diffusion_coefficient", None),
        "layers_scale_height_m":         getattr(atm, "scale_height", None),
        # convective-flux diagnostics (matching the Fortran HDF5 dataset names):
        # radiosity_convective (levels, W m-2) and c_p (layers, J K-1 mol-1)
        "levels_radiosity_convective_W_m2": (getattr(atm, "flux_conv", None) * 1e-3
                                             if getattr(atm, "flux_conv", None) is not None else None),
        "layers_isobaric_molar_heat_capacity_J_K_mol": (getattr(atm, "cpr_layers", None) * CST_R
                                             if getattr(atm, "cpr_layers", None) is not None else None),
        "absorbers_vmr":                 absorbers_vmr,
        "gases_vmr":                     gases_vmr_map,
        # spectra (emergent radiosity CGS -> W m-2 / cm-1)
        "wavenumber_cm1":                wn,
        "emission_spectral_radiosity_W_m2_cm1": spectral_radiosity[-1, :] * 1e-3,
        "transmission_spectral_radius_m": (spectral_radius
                                           if opts.get("output_transmission_spectra", False)
                                           else None),
        # model parameters
        "target_mass_kg":                getattr(target, "target_mass", None),
        "target_internal_temperature_K": getattr(target, "target_internal_temperature", None),
        "target_radius_m":               getattr(target, "target_radius", None),
        "elemental_abundances":          elemental_abundances,
        "solar_elemental_abundances":    solar_abundances,
    }

    h5_file = out_dir / ("output" + suffix + ".h5")
    write_hdf5_output_fortran(h5_file, payload)
