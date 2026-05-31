"""
Radiative transfer for giant-planet / exoplanet atmospheres.

Implements:
  - k-distribution opacity interpolation
  - CIA (collision-induced absorption)
  - Cloud optical depth
  - Rayleigh scattering
  - Two-stream flux solver (adding method + hemispheric closure)

Mirrors the Fortran ``radiative_transfer`` module (radiative_transfer.f90).
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .physics import CST_P0, CST_T0, CST_N0
from .math_utils import interp_ex_0d

# Numba JIT — graceful fallback if numba isn't installed
try:
    from numba import njit, prange, get_num_threads, get_thread_id
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def njit(*args, **kwargs):
        """No-op fallback when numba is unavailable."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def _wrap(fn):
            return fn
        return _wrap
    prange = range  # type: ignore
    def get_num_threads(): return 1  # type: ignore
    def get_thread_id(): return 0    # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PI: float = math.pi
_PREC_TS: float = 1e-10   # two-stream precision guard
_EMAX1:   float = 8.0     # semi-infinite threshold
_EMAX2:   float = 24.0    # infinite threshold

# ---------------------------------------------------------------------------
# r31: intra-RT profiling.  Set RT_PROFILE = {} from the caller before the
# call and read it back afterwards.  Per-phase wall-clock totals (seconds)
# accumulate into the dict.  No-op if the dict is empty / None.
# ---------------------------------------------------------------------------
RT_PROFILE: Optional[dict] = None


class _RTTimer:
    """Tiny stopwatch that writes to RT_PROFILE if it's a dict."""
    __slots__ = ("t0", "active")
    def __init__(self):
        self.active = isinstance(RT_PROFILE, dict)
        self.t0 = time.perf_counter() if self.active else 0.0
    def tick(self, phase: str) -> None:
        if not self.active:
            return
        now = time.perf_counter()
        RT_PROFILE[phase] = RT_PROFILE.get(phase, 0.0) + (now - self.t0)
        self.t0 = now


# ===========================================================================
# Main entry point
# ===========================================================================

def calculate_radiative_transfer(
    # --- Atmospheric state ---
    gases_vmr:            np.ndarray,   # (n_gases, n_layers)
    pressures_layers:     np.ndarray,   # (n_layers,)  Pa → converted internally to mbar
    temperatures_layers:  np.ndarray,   # (n_layers,)
    gravities_layers:     np.ndarray,   # (n_layers,)  m s⁻²
    species_vmr_layers:   np.ndarray,   # (n_layers, n_species)
    pressures:            np.ndarray,   # (n_levels,)  Pa
    temperatures:         np.ndarray,   # (n_levels,)
    # --- Stellar irradiance ---
    light_source_irradiance: np.ndarray,  # (n_wavenumbers,)
    # --- Species & k-coefficients ---
    n_species:            int,
    i_single_species:     int,          # -1 = all species
    wavenumber_min:       float,
    wavenumber_step:      float,
    rayleigh_scattering_coefficients: np.ndarray,  # (n_species, n_wavenumbers) cm²
    # --- Cloud parameters ---
    n_clouds:             int,
    cloud_vmr:            np.ndarray,   # (n_clouds, n_layers)
    cloud_particle_density: np.ndarray, # (n_clouds,)
    cloud_particle_radius:  np.ndarray, # (n_clouds, n_layers)
    cloud_q_ext:          np.ndarray,   # (n_clouds, n_wavenumbers, n_layers)
    cloud_q_scat:         np.ndarray,
    cloud_single_scattering_albedo: np.ndarray,
    cloud_asymetry_factor: np.ndarray,
    cloud_q_ext_ref:      np.ndarray,  # (n_clouds, n_layers)
    # --- k-coefficient tables ---
    n_k_pressures:        np.ndarray,   # (n_species,)
    n_k_temperatures:     np.ndarray,   # (n_species,)
    n_k_wavenumbers:      np.ndarray,   # (n_species,)
    wavenumbers_k:        np.ndarray,   # (n_k_wavenumbers_max, n_species)
    ng:                   np.ndarray,   # (n_species,) number of g-samples
    p_k_species:          np.ndarray,   # (n_k_pressures_max, n_species)
    t_k_species:          np.ndarray,   # (n_k_temperatures_max, n_k_pressures_max, n_species)
    weights_k:            np.ndarray,   # (n_k_samples_max,)
    samples_k:            np.ndarray,   # (n_k_samples_max,)
    kcoeff_species:       np.ndarray,   # (ng_max, n_wavn_max, n_temp_max, n_pres_max, n_species)
    # --- CIA ---
    h2_h2_cia:   np.ndarray,   # (n_layers, n_wavenumbers)
    h2_he_cia:   np.ndarray,
    h2o_n2_cia:  np.ndarray,
    h2o_h2o_cia: np.ndarray,
    # --- Grid dimensions ---
    n_levels:   int,
    n_layers:   int,
    n_wavenumbers: int,
    wavenumbers: np.ndarray,   # (n_wavenumbers,)
    scale_height: np.ndarray,  # (n_layers,)  km
    cos_average_angle: float,
    # --- Derived gas indices ---
    idx_h2:  int,
    idx_he:  int,
    idx_h2o: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute radiative fluxes using the two-stream / k-distribution method.

    Returns
    -------
    tau               : (n_levels, n_wavenumbers, ng_max)  total optical depths
    tau_rayleigh      : (n_levels, n_wavenumbers)          Rayleigh optical depths
    tau_cloud         : (n_clouds, n_layers)               cloud optical depths
    radiosity_internal: (n_levels,)  broadband net flux (W m⁻²)
    matrix_t          : (n_levels, n_levels)  Jacobian matrix
    flux              : (n_levels, n_wavenumbers)  spectral net flux
    spectral_radiosity: (n_levels, n_wavenumbers)
    """
    if n_clouds <= 0:
        print("Calculating radiative transfer without clouds...")
    else:
        print("Calculating radiative transfer with clouds...")

    n_k_samples_max = int(weights_k.shape[0])

    # -----------------------------------------------------------------------
    # Pre-compute g-index for k-combination
    # -----------------------------------------------------------------------
    if n_k_samples_max > 1:
        indg = 0
        for ig in range(n_k_samples_max):
            if samples_k[ig] > 0.5:
                break
            indg = ig
        fracg = (samples_k[indg + 1] - 0.5) / (samples_k[indg + 1] - samples_k[indg])
    else:
        indg = 0
        fracg = 0.0

    # -----------------------------------------------------------------------
    # Allocate output arrays
    # -----------------------------------------------------------------------
    tau          = np.zeros((n_levels, n_wavenumbers, n_k_samples_max))
    tau_rayleigh = np.zeros((n_levels, n_wavenumbers))
    tau_cloud_out = np.zeros((n_clouds, n_layers))

    # r31: intra-RT timer.  Phases:
    #   setup            -- log-precompute, wavenumber lookups, scratch alloc
    #   layer_opacity    -- the (j) layer loop: CIA + k-table interp + combine
    #   rayleigh_cloud   -- vectorised Rayleigh + cloud setup (out of i,ig loops)
    #   planck           -- Planck array calls
    #   twostream        -- the (i, ig, j) two-stream solver + matrix_t accum
    rt_timer = _RTTimer()

    # -----------------------------------------------------------------------
    # Pre-compute logs of k-table grids (constant across the j-loop).
    # The species loop calls np.log(p_k_species[...]) and np.log(t_k_species[...])
    # ~3 000 times per RT call, each on a 10–25-element slice, which is
    # dominated by Python overhead.  Compute them once here.
    # -----------------------------------------------------------------------
    log_pk_all = np.log(np.maximum(p_k_species, 1e-300))
    log_tk_all = np.log(np.maximum(t_k_species, 1e-300))
    # Also precompute the model wavenumber grid (constant per RT call) and the
    # nearest-k-wavenumber lookup per species (constant per RT call) so we
    # avoid rebuilding them 80 × 13 = 1 040 times.
    wn_model = wavenumber_min + wavenumber_step * np.arange(n_wavenumbers)
    iw_arr_per_species: list[np.ndarray] = []
    n_wn_active_per_species: list[int] = []
    for ik in range(n_species):
        wk = wavenumbers_k[:n_k_wavenumbers[ik], ik]
        pos = np.clip(np.searchsorted(wk, wn_model), 1, wk.size - 1)
        left_better = np.abs(wk[pos - 1] - wn_model) <= np.abs(wk[pos] - wn_model)
        iw_arr_per_species.append(np.where(left_better, pos - 1, pos).astype(np.intp))
        nwa = int(np.searchsorted(wn_model, wk[-1] * (1.0 + 1e-6)))
        n_wn_active_per_species.append(max(0, min(n_wavenumbers, nwa)))

    # Hoisted scratch arrays for the species loop (re-used per layer).
    dtau  = np.empty((n_k_samples_max, n_wavenumbers))
    dtauk = np.empty((n_k_samples_max, n_wavenumbers))

    rt_timer.tick("setup")

    # -----------------------------------------------------------------------
    # Layer optical depth calculation
    # -----------------------------------------------------------------------
    # h0 (cm), cmam (Loschmidt column, cm⁻²)
    h0   = np.zeros(n_layers)
    cmam = np.zeros(n_layers)

    for j in range(n_layers - 1, -1, -1):    # j from n_layers-1 down to 0
        tj = temperatures_layers[j]
        pj = pressures_layers[j]          # Pa (code uses mbar internally)

        # scale height column (cm)
        # BUGFIX: scale_height is computed in SI metres (exorem_main.py L1532
        # and _calculate_altitude: R*T/(mu*g) with mu in kg/mol, g in m/s²),
        # NOT in km.  The Fortran reference stores scale_height in km (it uses
        # the 1d2 prefactor with mu in g/mol and g in cm/s², see exorem.f90
        # L485 + the "gravity ... (cm sec-2)" note at L1257), hence its h0 uses
        # 1d5 (km→cm).  Here scale_height is metres, so the correct conversion
        # is m→cm = 1e2.  The previous 1e5 made every optical depth ~1000×
        # too large, pushing the photosphere to the top of the atmosphere so
        # the emergent flux collapsed to σ·T_top⁴ (T_eff≈270 K instead of the
        # interior σ·T_int⁴ ≈ 500 K).
        h0[j]   = 1e2 * scale_height[j] * CST_T0 / tj          # m → cm
        # BUGFIX (Pa vs mbar): the Fortran wrote this prefactor with pressures
        # in mbar, so the denominator is 1 atm expressed in mbar
        # (CST_P0*1e-2 = 1013.25).  The Python port feeds ΔP in Pa and never
        # converted it, so cmam·h0 was 100.22× the true hydrostatic column
        # density (verified: 1.919e29 vs 1.915e27 cm⁻²).  Dividing by CST_P0 in
        # Pa makes ΔP/CST_P0 a proper dimensionless pressure fraction and brings
        # the column density to 1.002× physical.  This 100× over-count was
        # making the LINE optical depth ~100× too large and clamping the
        # emergent flux to ~190 K regardless of composition.
        cmam[j] = (CST_N0 * 1e-6
                   * (pressures[j] - pressures[j + 1])
                   / CST_P0)  # Loschmidt cm⁻³  (Pa/Pa, dimensionless fraction)

        # -- CIA contribution ---
        # BUGFIX (Pa vs mbar, squared): same root cause as cmam but squared,
        # because CIA opacity ∝ density²·path ∝ pj·ΔP.  With pj and ΔP in Pa and
        # the denominator (CST_P0*1e-2)² in mbar², fac_cont was 10022× (=100.2²)
        # the physical (n/n_L)²·path (verified: 5.76e12 vs 5.75e8 cm).  Using
        # CST_P0² in Pa² corrects it.  This must be fixed together with cmam:
        # otherwise the 1e4×-too-strong CIA continuum becomes the new clamp once
        # the line opacity is corrected.
        fac_cont = (CST_T0 / tj) * h0[j] * pj * (pressures[j] - pressures[j + 1]) / CST_P0 ** 2

        qh2  = gases_vmr[idx_h2,  j]
        qhe  = gases_vmr[idx_he,  j]
        qh2o = gases_vmr[idx_h2o, j]

        dtauc = fac_cont * (
            qh2 * (qh2 * h2_h2_cia[j, :]  + qhe  * h2_he_cia[j, :]) +
            qh2o * (qh2o * h2o_h2o_cia[j, :] +
                    (1.0 - qh2o) * h2o_n2_cia[j, :])
        )  # (n_wavenumbers,)

        # -- Cloud opacity ---
        if n_clouds > 0:
            for ik in range(n_wavenumbers):
                cloudj = 0.0
                for ic in range(n_clouds):
                    r_ic = cloud_particle_radius[ic, j]
                    if r_ic > 0:
                        cloudj += (
                            cloud_vmr[ic, j]
                            / (4.0 / 3.0 * cloud_particle_density[ic] * r_ic
                               * gravities_layers[j] * 1e-2)
                            * (pressures[j] - pressures[j + 1]) * 1e2
                            * cloud_q_ext[ic, ik, j]
                        )
                    cloudj = max(0.0, cloudj)
                dtauc[ik] += cloudj

        # -- k-coefficient species loop (uses hoisted scratch arrays) ---
        # NB: dtau is fully overwritten on the ik == 0 branch and accumulated
        # afterwards, so no fill needed at the top of each layer.

        # BUGFIX (Pa vs bar): the k-table pressure axis (p_k_species) is stored
        # in BAR (petitRADTRANS convention, 1e-6 .. 1e2 bar), but pj is in Pa.
        # Using log(pj_Pa) directly put almost every layer above the table's top
        # pressure, so tab_pk clipped to the 100-bar entry and every line was
        # evaluated at maximum pressure broadening throughout the column —
        # over-opaque in the molecular bands while leaving the CIA far-IR alone.
        # Convert Pa->bar (x1e-5) here ONLY; cmam and fac_cont correctly use Pa.
        apj = math.log(pj * 1e-5)

        for ik in range(n_species):
            if i_single_species > 0 and (ik + 1) != i_single_species:
                continue

            qlj = species_vmr_layers[j, ik]

            # Pressure interpolation (uses precomputed log_pk_all)
            pkd = log_pk_all[:n_k_pressures[ik], ik]
            tab_pk = interp_ex_0d(apj, pkd, np.arange(1, n_k_pressures[ik] + 1, dtype=float))

            if tab_pk <= 1.0:
                ipk1, fipk1, ipk2 = 0, 0.0, 0
            elif tab_pk < n_k_pressures[ik]:
                ipk1 = int(tab_pk) - 1
                fipk1 = 1.0 - (tab_pk - int(tab_pk))
                ipk2 = ipk1 + 1
            else:
                ipk1 = n_k_pressures[ik] - 1
                fipk1 = 1.0
                ipk2 = ipk1

            # Temperature interpolation at ipk1/ipk2 (uses precomputed log_tk_all)
            tkd1 = log_tk_all[:n_k_temperatures[ik], ipk1, ik]
            tab_tk1 = interp_ex_0d(
                math.log(tj), tkd1,
                np.arange(1, n_k_temperatures[ik] + 1, dtype=float))
            tkd2 = log_tk_all[:n_k_temperatures[ik], ipk2, ik]
            tab_tk2 = interp_ex_0d(
                math.log(tj), tkd2,
                np.arange(1, n_k_temperatures[ik] + 1, dtype=float))
            tab_tk = fipk1 * tab_tk1 + (1.0 - fipk1) * tab_tk2

            if tab_tk <= 1.0:
                itk1, fitk1, itk2 = 0, 0.0, 0
            elif tab_tk < n_k_temperatures[ik]:
                itk1 = int(tab_tk) - 1
                fitk1 = 1.0 - (tab_tk - int(tab_tk))
                itk2 = itk1 + 1
            else:
                itk1 = n_k_temperatures[ik] - 1
                fitk1 = 1.0
                itk2 = itk1

            # ---- VECTORISED inner block (replaces 35 lines of Python loops) ----
            ng_ik = ng[ik]
            scalar = qlj * cmam[j] * h0[j]
            iw_arr = iw_arr_per_species[ik]
            n_wn_active = n_wn_active_per_species[ik]

            if n_wn_active > 0 and ng_ik > 0:
                # Bilinear interp in (T, P) — fancy-indexed in one shot
                iw_slice = iw_arr[:n_wn_active]
                ig_rows = np.arange(ng_ik)[:, None]
                k11 = kcoeff_species[ig_rows, iw_slice[None, :], itk1, ipk1, ik]
                k21 = kcoeff_species[ig_rows, iw_slice[None, :], itk2, ipk1, ik]
                k12 = kcoeff_species[ig_rows, iw_slice[None, :], itk1, ipk2, ik]
                k22 = kcoeff_species[ig_rows, iw_slice[None, :], itk2, ipk2, ik]

                # log-space when all 4 corners > 1e-40; linear otherwise.
                # Subnormal-stall avoidance: floor at 1e-100 instead of 1e-300
                # so np.log()/np.exp() never produce subnormals.  log(1e-100)
                # = -230, well below any value used in subsequent arithmetic.
                use_log = np.minimum(np.minimum(k11, k21), np.minimum(k12, k22)) > 1e-40
                k_lin = (fipk1 * (fitk1 * k11 + (1.0 - fitk1) * k21)
                         + (1.0 - fipk1) * (fitk1 * k12 + (1.0 - fitk1) * k22))
                safe11 = np.maximum(k11, 1e-100)
                safe21 = np.maximum(k21, 1e-100)
                safe12 = np.maximum(k12, 1e-100)
                safe22 = np.maximum(k22, 1e-100)
                k_log = np.exp(
                    fipk1       * (fitk1 * np.log(safe11) + (1.0 - fitk1) * np.log(safe21))
                    + (1.0 - fipk1) * (fitk1 * np.log(safe12) + (1.0 - fitk1) * np.log(safe22)))
                k_interp = np.where(use_log, k_log, k_lin)

                dtauk[:ng_ik, :n_wn_active] = k_interp * scalar
                if n_wn_active < n_wavenumbers:
                    dtauk[:ng_ik, n_wn_active:] = 0.0

                # Combine species (random-overlap assumption); still per-wn
                # because the sort is intrinsically sequential.
                if ik == 0:
                    dtau[:ng[0], :n_wn_active] = dtauk[:ng[0], :n_wn_active]
                    if n_wn_active < n_wavenumbers:
                        dtau[:ng[0], n_wn_active:] = 0.0
                elif n_k_samples_max > 1:
                    for i in range(n_wn_active):
                        _combine_k_distributions(
                            dtau, dtauk, ng_ik, n_k_samples_max,
                            indg, fracg, samples_k, weights_k, i)
                else:
                    dtau[0, :n_wn_active] += dtauk[0, :n_wn_active]

        # -- Accumulate total optical depth (vectorised) ---
        # Was a double Python loop over (i, ig) = 2 416 ops per layer.
        # tau[j] is (n_wn, n_g); dtau is (n_g, n_wn); dtauc is (n_wn,).
        tau[j] = tau[j + 1] + dtau.T + dtauc[:, None]

    rt_timer.tick("layer_opacity")

    # -----------------------------------------------------------------------
    # r31: Rayleigh + cloud hoist (Gemini proposal #2, adapted)
    #
    # The per-layer Rayleigh and cloud quantities depend only on (j, i),
    # never on (ig).  The previous code recomputed them inside the (i, ig, j)
    # triple loop -- 16x redundant work AND a bug: `tau_rayleigh[j, i] +=`
    # accumulated 16x instead of assigning once (the Fortran reference uses
    # `=`, not `+=`, on this very line).  That over-fattened the transit
    # spectrum's Rayleigh tau by 16x.  The clear-sky RT physics happens to
    # be unaffected because `dtau_ray` is overwritten and consumed locally
    # inside the ig loop, not accumulated.
    #
    # Here we pre-compute everything once at the right resolution.
    # dtau_ray_all is indexed by physical j (NOT reversed jj).
    # -----------------------------------------------------------------------
    # Per-layer Rayleigh: (n_layers, n_wavenumbers).
    #   gases_vmr.T:                       (n_layers, n_gases)
    #   rayleigh_scattering_coefficients:  (n_gases, n_wavenumbers)
    dtau_ray_all = (gases_vmr.T @ rayleigh_scattering_coefficients
                   ) * (cmam * h0)[:, None]

    # Cumulative Rayleigh from TOA downward, matching Fortran's
    # tau_rayleigh(j, i) = tau_rayleigh(j + 1, i) + dtau_ray  recursion.
    # tau_rayleigh[n_levels-1] = 0 (top), tau_rayleigh[0] = full column.
    tau_rayleigh[:n_layers] = np.cumsum(dtau_ray_all[::-1, :], axis=0)[::-1, :]

    # Cloud quantities, vectorised across (j, i).  Zero arrays if n_clouds == 0.
    if n_clouds > 0:
        # Per-cloud per-layer geometric optical depth (before q_ext)
        # cloud_particle_radius: (n_clouds, n_layers); pressures: (n_levels,)
        dp = (pressures[:-1] - pressures[1:]) * 1e2   # (n_layers,)
        # tc[ic, j]; only where r_ic > 0
        tc_all = np.zeros((n_clouds, n_layers))
        for ic in range(n_clouds):
            r_ic = cloud_particle_radius[ic, :]                     # (n_layers,)
            mask = r_ic > 0
            if np.any(mask):
                denom = (4.0 / 3.0 * cloud_particle_density[ic]
                         * r_ic[mask] * gravities_layers[mask] * 1e-2)
                tc_all[ic, mask] = (cloud_vmr[ic, mask] / denom) * dp[mask]
                tau_cloud_out[ic, :] = np.maximum(tc_all[ic, :], 0.0) \
                                       * cloud_q_ext_ref[ic, :]
        tc_pos = np.maximum(tc_all, 0.0)                            # (n_clouds, n_layers)
        # Cloud contributions, summed across cloud species; (n_wavenumbers, n_layers)
        taucl_scat_all = np.einsum('cj,cij->ij', tc_pos, cloud_q_scat)
        omeg_cl_all    = np.einsum('cj,cij->ij', tc_pos,
                                   cloud_single_scattering_albedo * cloud_q_ext)
        gfac_cl_all    = np.einsum('cj,cij->ij', tc_pos,
                                   cloud_asymetry_factor * cloud_q_scat)
    else:
        taucl_scat_all = np.zeros((n_wavenumbers, n_layers))
        omeg_cl_all    = np.zeros((n_wavenumbers, n_layers))
        gfac_cl_all    = np.zeros((n_wavenumbers, n_layers))

    rt_timer.tick("rayleigh_cloud")

    # -----------------------------------------------------------------------
    # Planck function at levels and layers
    # -----------------------------------------------------------------------
    pl_lev, dpl_lev = _planck_array(wavenumbers, temperatures[:n_levels],
                                    n_levels, n_wavenumbers)
    tl2 = np.empty(n_levels)
    tl2[0] = temperatures[0]
    tl2[1:] = temperatures_layers[:n_layers]
    pl_lay, _ = _planck_array(wavenumbers, tl2, n_levels, n_wavenumbers)

    rt_timer.tick("planck")

    # -----------------------------------------------------------------------
    # Two-stream flux calculation
    # -----------------------------------------------------------------------
    flux               = np.zeros((n_levels, n_wavenumbers))
    spectral_radiosity = np.zeros((n_levels, n_wavenumbers))

    # r37: replaced the Python (i, ig) loop with a single call to a
    # parallel Numba inner-loop wrapper.  See _twostream_loop_parallel for
    # the algorithmic notes.  Bit-exact within floating-point reduction
    # order (matrix_t and radiosity_internal accumulate with a different
    # thread-by-thread sum order, but the totals match to < 1e-12 relative).
    matrix_t, radiosity_internal = _twostream_loop_dispatch(
        n_wavenumbers, n_k_samples_max, n_layers,
        tau, pl_lev, dpl_lev, dtau_ray_all,
        weights_k, wavenumber_step, cos_average_angle,
        light_source_irradiance,
        n_clouds, taucl_scat_all, omeg_cl_all, gfac_cl_all,
        flux, spectral_radiosity,
    )

    rt_timer.tick("twostream")

    return tau, tau_rayleigh, tau_cloud_out, radiosity_internal, matrix_t, flux, spectral_radiosity


# ===========================================================================
# Two-stream solver
# ===========================================================================

@njit(cache=True, fastmath=True, parallel=True)
def _twostream_loop_parallel(
    n_wavenumbers: int,
    n_k_samples_max: int,
    n_layers: int,
    tau: np.ndarray,                # (n_levels, n_wavenumbers, n_k_samples_max)
    pl_lev: np.ndarray,             # (n_levels, n_wavenumbers)
    dpl_lev: np.ndarray,            # (n_levels, n_wavenumbers)
    dtau_ray_all: np.ndarray,       # (n_layers, n_wavenumbers)
    weights_k: np.ndarray,          # (n_k_samples_max,)
    wavenumber_step: float,
    cos_average_angle: float,
    light_source_irradiance: np.ndarray,   # (n_wavenumbers,)
    has_clouds: int,                # 0 or 1
    taucl_scat_rev: np.ndarray,     # (n_wavenumbers, n_layers) – dummy if no clouds
    omeg_cl_rev: np.ndarray,        # (n_wavenumbers, n_layers)
    gfac_cl_rev: np.ndarray,        # (n_wavenumbers, n_layers)
    flux_out: np.ndarray,           # (n_levels, n_wavenumbers) – per-i write, safe
    spectral_radiosity_out: np.ndarray,    # (n_levels, n_wavenumbers)
) -> tuple:
    """r37: combined ig-vectorisation + prange wavenumber parallelisation.

    Lowers the entire (i, ig) double-loop into Numba so that the only Python
    boundary crossing is one call per RT step (was n_wavenumbers × n_g per RT
    step — typically 1600+ crossings).  The wavenumber loop uses ``prange``
    for thread-level parallelism; matrix_t and radiosity_internal are
    accumulated into per-thread arrays then reduced to avoid races.
    flux_out[:,i] and spectral_radiosity_out[:,i] are per-wavenumber column
    writes and so are race-free across threads.

    Combined effect on the r36 baseline of 39 s/iter in `twostream`:
      * eliminating ~2400 Numba boundary crossings → 3-5× speedup
      * 4-8 thread parallelism over n_wavenumbers → another 3-6× on typical
        4-8 core laptops/workstations
    Total expected: 10-20× over r36 for the same algorithm (bit-exact within
    parallel-reduction numerics).

    Returns
    -------
    matrix_t           : (n_levels, n_levels)
    radiosity_internal : (n_levels,)
    """
    n_levels = n_layers + 1
    PI = 3.141592653589793
    TINY = 2.2250738585072014e-308

    n_threads = get_num_threads()
    matrix_t_thread = np.zeros((n_threads, n_levels, n_levels))
    radiosity_thread = np.zeros((n_threads, n_levels))

    # r39e: pre-allocate per-thread workspaces ONCE outside prange.
    # Previously each prange iteration allocated 7 small numpy arrays.  With
    # 151 wavenumbers × ~16 g-samples per RT call that's ~17 000 allocations
    # per call.  Even when numba's parallelisation works correctly, every
    # thread hits the same malloc lock, serialising the work and eating
    # much of the parallelism win.  Lifting the buffers out and indexing by
    # tid removes the contention and typically gives 1.5-2× on top of
    # whatever speed-up parallelism provides.
    pl_lev2_t   = np.empty((n_threads, n_levels))
    dtau_scat_t = np.empty((n_threads, n_layers))
    omeg_scat_t = np.empty((n_threads, n_layers))
    gfac_scat_t = np.empty((n_threads, n_layers))
    flux_up_t   = np.empty((n_threads, n_levels))
    flux_down_t = np.empty((n_threads, n_levels))
    d_kernel_t  = np.empty((n_threads, n_levels, n_levels))

    for i in prange(n_wavenumbers):
        tid = get_thread_id()

        # ---- per-thread workspace views (no allocation) ------------------
        pl_lev2   = pl_lev2_t[tid]
        dtau_scat = dtau_scat_t[tid]
        omeg_scat = omeg_scat_t[tid]
        gfac_scat = gfac_scat_t[tid]
        flux_up   = flux_up_t[tid]
        flux_down = flux_down_t[tid]
        d_kernel  = d_kernel_t[tid]

        # ---- build pl_lev2 (depends only on i) ---------------------------
        # Equivalent to:
        #   pl_lev2[1:n_levels] = pl_lev[0:n_layers, i][::-1]
        #   pl_lev2[0]          = pl_lev[n_levels - 1, i]
        pl_lev2[0] = pl_lev[n_levels - 1, i]
        for k in range(n_layers):
            pl_lev2[k + 1] = pl_lev[n_layers - 1 - k, i]

        for ig in range(n_k_samples_max):
            # ---- layer prep (jj-reversed indexing) -----------------------
            for jj in range(n_layers):
                j = n_layers - 1 - jj
                d_ray = dtau_ray_all[j, i]
                dtau_scat[jj] = tau[j, i, ig] - tau[j + 1, i, ig] + d_ray
                omeg_scat[jj] = d_ray
                gfac_scat[jj] = 0.0

                if has_clouds == 1:
                    taucl_tot = TINY + d_ray + taucl_scat_rev[i, jj]
                    omeg_scat[jj] += omeg_cl_rev[i, jj]
                    gfac_scat[jj] = gfac_cl_rev[i, jj] / taucl_tot

                if dtau_scat[jj] > 0.0:
                    omeg_scat[jj] = omeg_scat[jj] / dtau_scat[jj]
                else:
                    omeg_scat[jj] = 0.0

            # ---- two-stream call ----------------------------------------
            flux_down[0] = light_source_irradiance[i]

            calculate_two_stream_fluxes(
                n_layers, pl_lev2, dtau_scat, omeg_scat, gfac_scat,
                flux_up, flux_down, d_kernel, cos_average_angle)

            # ---- matrix_t accumulator (per-thread) ----------------------
            # matrix_t[jj, lv] += c_acc * d_kernel[nl-1-lv, nl-1-jj] * dpl_lev[jj, i]
            c_acc = weights_k[ig] * PI * wavenumber_step
            for jj in range(n_levels):
                dpl_ji = dpl_lev[jj, i]
                jj_rev = n_levels - 1 - jj
                for lv in range(n_levels):
                    lv_rev = n_levels - 1 - lv
                    matrix_t_thread[tid, jj, lv] += (
                        c_acc * d_kernel[lv_rev, jj_rev] * dpl_ji
                    )

            # ---- flux / spectral_radiosity / radiosity_internal --------
            w_pi = weights_k[ig] * PI
            l_irr_i_w_pi = light_source_irradiance[i] * w_pi
            for lv in range(n_levels):
                lv_rev = n_levels - 1 - lv
                net = (flux_up[lv_rev] - flux_down[lv_rev]) * w_pi
                flux_out[lv, i] += net
                spectral_radiosity_out[lv, i] += net + l_irr_i_w_pi
                radiosity_thread[tid, lv] += net * wavenumber_step

    # ---- reduce thread-local accumulators ---------------------------------
    matrix_t = np.zeros((n_levels, n_levels))
    radiosity_internal = np.zeros(n_levels)
    for tid in range(n_threads):
        for jj in range(n_levels):
            radiosity_internal[jj] += radiosity_thread[tid, jj]
            for lv in range(n_levels):
                matrix_t[jj, lv] += matrix_t_thread[tid, jj, lv]

    return matrix_t, radiosity_internal


def _twostream_loop_dispatch(
    n_wavenumbers, n_k_samples_max, n_layers,
    tau, pl_lev, dpl_lev, dtau_ray_all,
    weights_k, wavenumber_step, cos_average_angle,
    light_source_irradiance,
    n_clouds, taucl_scat_all, omeg_cl_all, gfac_cl_all,
    flux_out, spectral_radiosity_out,
):
    """Build pre-reversed cloud arrays then call the parallel inner loop.

    Cloud arrays are pre-reversed (axis=1 = layer dim) so that the inner
    Numba loop only does scalar indexing.  When n_clouds == 0 we still
    pass length-correct dummy arrays since Numba's type inference needs
    consistent shapes regardless of the branch taken.
    """
    if n_clouds > 0:
        taucl_scat_rev = np.ascontiguousarray(taucl_scat_all[:, ::-1])
        omeg_cl_rev    = np.ascontiguousarray(omeg_cl_all[:, ::-1])
        gfac_cl_rev    = np.ascontiguousarray(gfac_cl_all[:, ::-1])
        has_clouds_int = 1
    else:
        taucl_scat_rev = np.zeros((n_wavenumbers, n_layers))
        omeg_cl_rev    = np.zeros((n_wavenumbers, n_layers))
        gfac_cl_rev    = np.zeros((n_wavenumbers, n_layers))
        has_clouds_int = 0

    return _twostream_loop_parallel(
        n_wavenumbers, n_k_samples_max, n_layers,
        tau, pl_lev, dpl_lev, dtau_ray_all,
        weights_k, wavenumber_step, cos_average_angle,
        light_source_irradiance,
        has_clouds_int, taucl_scat_rev, omeg_cl_rev, gfac_cl_rev,
        flux_out, spectral_radiosity_out,
    )



@njit(cache=True, fastmath=True)
def calculate_two_stream_fluxes(
    n_layers: int,
    planck_func: np.ndarray,       # (n_levels,)
    dtau: np.ndarray,              # (n_layers,)
    single_scattering_albedos: np.ndarray,  # (n_layers,)  – modified in-place
    asymmetry_parameters: np.ndarray,       # (n_layers,)
    flux_upward: np.ndarray,       # (n_levels,)  – modified in-place
    flux_downward: np.ndarray,     # (n_levels,)  – modified in-place
    d_kernel: np.ndarray,          # (n_levels, n_levels)  – modified in-place
    cos_avg: float,
) -> None:
    """
    Two-stream radiative transfer solver (adding method + hemispheric closure).

    Level indexing follows the Fortran convention: level 1 is the top of the
    atmosphere. Python arrays are 0-based but semantically identical.

    Parameters
    ----------
    n_layers   : number of model layers
    planck_func: Planck function at each of the n_levels level boundaries
    dtau       : layer extinction optical depths
    single_scattering_albedos : layer SSA (clamped internally)
    asymmetry_parameters      : layer asymmetry factors
    flux_upward, flux_downward: on entry ``flux_downward[0]`` is the downward
                                irradiance at the TOA; on exit both arrays
                                contain the full flux profiles.
    d_kernel   : flux kernel (Jacobian w.r.t. Planck function)
    cos_avg    : cosine of the average propagation angle (hemispheric closure)
    """
    n_levels = n_layers + 1
    np2 = n_layers + 2

    downward_flux_top = float(flux_downward[0])

    # -- Delta-Eddington scaling ---
    dtau_p  = np.empty(n_layers)
    om_p    = np.empty(n_layers)
    g1      = np.empty(n_layers)
    g2      = np.empty(n_layers)
    sk      = np.empty(n_layers)

    for l in range(n_layers):
        ssa = min(single_scattering_albedos[l], 1.0 - _PREC_TS)
        single_scattering_albedos[l] = ssa

        g = asymmetry_parameters[l]
        dtau_p[l] = (1.0 - ssa * g) * dtau[l]
        om_p[l]   = ((1.0 - g) * ssa
                     / (1.0 - ssa * g))
        g1[l] = (1.0 - 0.5 * om_p[l]) / cos_avg
        g2[l] = 0.5 * om_p[l] / cos_avg
        sk[l] = math.sqrt(max(g1[l]**2 - g2[l]**2, 0.0))

    # -- Layer reflectance (rl) and transmittance (Tl) ---
    rl = np.empty(n_levels)
    Tl = np.empty(n_levels)   # called temperatures_layers in Fortran

    for l in range(n_layers):
        skt = sk[l] * dtau_p[l]
        denom_frac = g2[l] / (g1[l] + sk[l]) if (g1[l] + sk[l]) > 0 else 0.0

        if skt > _EMAX2:
            rl[l] = denom_frac
            Tl[l] = 0.0
        else:
            ekt = math.exp(skt)
            if skt > _EMAX1:
                rl[l] = denom_frac
                Tl[l] = 2.0 * sk[l] / (ekt * (g1[l] + sk[l]))
            else:
                e2ktm = ekt**2 - 1.0
                denom = g1[l] * e2ktm + sk[l] * (e2ktm + 2.0)
                rl[l] = g2[l] * e2ktm / denom
                Tl[l] = 2.0 * sk[l] * ekt / denom

    # -- Surface: semi-infinite with same optical properties as bottom layer ---
    alb = g2[n_layers - 1] / (g1[n_layers - 1] + sk[n_layers - 1]) if (g1[n_layers - 1] + sk[n_layers - 1]) > 0 else 0.0
    rl[n_levels - 1] = 1.0 - alb
    Tl[n_levels - 1] = 0.0
    p_n   = planck_func[n_levels - 2]
    p_np1 = planck_func[n_levels - 1]
    emis = (1.0 - alb
            + (1.0 + alb) * cos_avg * (1.0 - p_n / p_np1) / dtau_p[n_layers - 1]
            if (abs(p_np1) > 0 and dtau_p[n_layers - 1] > 0) else 1.0 - alb)

    # -- Adding method: combined reflectances from top and bottom ---
    rs = np.empty(n_levels)
    ru = np.empty(n_levels)
    dd = np.empty(n_layers)
    du = np.empty(n_levels)

    rs[0] = rl[0]
    ru[n_levels - 1] = alb

    for l in range(n_layers):
        dd[l] = 1.0 / max(1.0 - rs[l] * rl[l + 1], 1e-300)
        rs[l + 1] = rl[l + 1] + Tl[l + 1]**2 * rs[l] * dd[l]

        idx_bot = n_layers - 1 - l  # [PATCHED]
        du[idx_bot] = 1.0 / max(1.0 - rl[idx_bot] * ru[n_layers - l], 1e-300)
        ru[idx_bot] = rl[idx_bot] + Tl[idx_bot]**2 * ru[n_layers - l] * du[idx_bot]

    # -- Flux loop: iterate once for source=Planck, once per level for kernel ---
    ufl = np.empty(n_levels)
    dfl = np.empty(n_levels)
    dflux = np.empty(n_layers)

    planck_func_tmp = np.empty(n_levels)

    for j in range(n_levels + 1):
        dfl[n_levels - 1] = 0.0
        flux_downward[0] = 0.0
        planck_func_tmp[:] = 0.0

        if j < n_levels:
            planck_func_tmp[j] = 1.0
        else:
            flux_downward[0] = downward_flux_top
            planck_func_tmp[:] = planck_func

        ufl[n_levels - 1] = emis * planck_func_tmp[n_levels - 1]

        for l in range(n_layers):
            if planck_func_tmp[l] < _PREC_TS and planck_func_tmp[l + 1] < _PREC_TS:
                ufl[l] = 0.0
                dfl[l] = 0.0
            else:
                ufl[l], dfl[l] = _up_down_fluxes(
                    planck_func_tmp[l], planck_func_tmp[l + 1],
                    single_scattering_albedos[l],
                    asymmetry_parameters[l],
                    dtau[l], cos_avg)

        # Adding: downward pass
        dflux[0] = dfl[0] + Tl[0] * flux_downward[0]
        for l in range(n_layers - 1):
            dflux[l + 1] = (Tl[l + 1]
                            * (rs[l] * (rl[l + 1] * dflux[l] + ufl[l + 1]) * dd[l] + dflux[l])
                            + dfl[l + 1])

        flux_downward[n_levels - 1] = ((dflux[n_layers - 1]
                                        + rs[n_layers - 1] * emis * planck_func_tmp[n_levels - 1])
                                       / max(1.0 - rs[n_layers - 1] * alb, 1e-300))

        # Adding: upward pass
        uflux = np.empty(n_levels)
        uflux[n_levels - 1] = ufl[n_levels - 1]

        # r37 BUG FIX: was du[idx + 1] which is off-by-one and reads
        # du[n_layers] (i.e. du[n_levels-1]) on the first iteration —
        # that index is never written by the recursion above, so the
        # original code was reading uninitialized memory.  The Fortran
        # reference (radiative_transfer.f90 line 653) reads du(n_levels - l),
        # which is the SAME index it writes at line 595.  The Python
        # equivalent is therefore du[idx], not du[idx + 1].  Below this
        # fix is now deterministic and matches Fortran semantics.
        for l in range(n_layers):
            idx = n_layers - 1 - l
            uflux[idx] = (Tl[idx]
                          * (uflux[idx + 1] + ru[idx + 1] * dfl[idx])
                          * du[idx]
                          + ufl[idx])

        flux_upward[0] = uflux[0] + ru[0] * flux_downward[0]
        flux_upward[n_levels - 1] = alb * flux_downward[n_levels - 1] + emis * planck_func_tmp[n_levels - 1]

        for l in range(1, n_layers):
            idx = n_levels - 1 - l
            flux_upward[idx] = ((uflux[idx] + ru[idx] * dflux[n_layers - 1 - l])
                                / max(1.0 - ru[idx] * rs[n_layers - 1 - l], 1e-300))

        for l in range(1, n_layers):
            flux_downward[l] = dflux[l - 1] + rs[l - 1] * flux_upward[l]

        if j < n_levels:
            for l in range(n_levels):
                d_kernel[l, j] = flux_upward[l] - flux_downward[l]


# ===========================================================================
# Layer flux helper
# ===========================================================================

@njit(cache=True, fastmath=True)
def _up_down_fluxes(
    emission_1: float,
    emission_2: float,
    ssa: float,
    g: float,
    optical_depth: float,
    cos_avg: float,
) -> tuple[float, float]:
    """
    Two-stream (Paige & Crisp) upward and downward thermal fluxes for a
    single homogeneous layer.
    """
    dair = ssa * (1.0 - g) / max(1.0 - g * ssa, 1e-300)
    cap  = math.sqrt(max(1.0 - dair, 0.0)) / cos_avg

    sq_term = math.sqrt(max(1.0 - dair, 0.0))
    opttp = math.sqrt(max(1.0 - 0.5 * dair + sq_term, 0.0))
    omttp = math.sqrt(max(1.0 - 0.5 * dair - sq_term, 0.0))

    dtauir = (1.0 - g * ssa) * optical_depth
    db = ((emission_2 - emission_1) * cos_avg / dtauir
          if dtauir > 0 else 0.0)

    if cap * dtauir < 24.0:
        epkt = math.exp(cap * dtauir)
        emkt = 1.0 / epkt

        v1 = opttp
        v2 = omttp
        v3 = -(emission_1 - db)
        v4 = emkt * omttp
        v5 = epkt * opttp
        v6 = -(emission_2 + db)

        denom = v1 * v5 - v4 * v2
        if abs(denom) < 1e-300:
            denom = 1e-300
        c1 = (v3 * v5 - v6 * v2) / denom
        c2 = (v1 * v6 - v3 * v4) / denom

        upward_flux   = c1 * omttp + c2 * opttp + emission_1 + db
        downward_flux = c1 * emkt * opttp + c2 * epkt * omttp + emission_2 - db
    else:
        ratio = omttp / opttp if opttp > 0 else 0.0
        upward_flux   = emission_1 * (1.0 - ratio) + db * (1.0 + ratio)
        downward_flux = emission_2 * (1.0 - ratio) - db * (1.0 + ratio)

    return upward_flux, downward_flux


# ===========================================================================
# Planck function on a grid
# ===========================================================================

@njit(cache=True, fastmath=True)
def _planck_array(
    wavenumbers: np.ndarray,
    temperatures: np.ndarray,
    n_levels: int,
    n_wavenumbers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Planck function and its temperature derivative on a grid.

    Returns
    -------
    pl  : (n_levels, n_wavenumbers)  in erg s⁻¹ cm⁻² sr⁻¹ / cm⁻¹
    dpl : (n_levels, n_wavenumbers)  derivative w.r.t. temperature
    """
    # Inlined constants (Numba can't see the .physics module).
    # CST_H = 6.62607015e-34, CST_C = 2.99792458e8, CST_K = 1.380649e-23
    hck = 6.62607015e-34 * 2.99792458e8 * 1e2 / 1.380649e-23    # (cm)
    hc2 = 2.0 * 6.62607015e-34 * (2.99792458e8 * 1e2) ** 2      # (J cm²)

    pl  = np.zeros((n_levels, n_wavenumbers))
    dpl = np.zeros((n_levels, n_wavenumbers))

    # The dpl formula divides by denom**2.  If we let exp_val grow all the
    # way to the float64 ceiling (~1e308), then denom**2 = (1e308)**2 = 1e616
    # which overflows.  Cap hckt_nu at log(sqrt(max_float)) ≈ 354 so that
    # exp_val**2 still fits comfortably; pl/dpl are essentially zero at that
    # point anyway (Wien tail of the Planck function).
    log_safe = 0.5 * math.log(np.finfo(np.float64).max)   # ≈ 354.9
    tiny     = np.finfo(np.float64).tiny

    for j in range(n_levels):
        t = temperatures[j]
        for i in range(n_wavenumbers):
            nu = wavenumbers[i]
            hckt_nu = hck * nu / t
            if hckt_nu > log_safe:
                pl[j, i]  = tiny
                dpl[j, i] = tiny
            else:
                exp_val = math.exp(hckt_nu)
                denom = exp_val - 1.0
                if denom <= 0:
                    denom = 1e-300
                pl[j, i]  = hc2 * nu**3 / denom * 1e7   # J → erg
                dpl[j, i] = (hc2 * nu**3 * (hckt_nu / t) * exp_val
                             / denom**2 * 1e7)
    return pl, dpl


# ===========================================================================
# k-distribution combination (random overlap)
# ===========================================================================

@njit(cache=True, fastmath=True)
def _combine_k_distributions(
    dtau: np.ndarray,      # (n_k_samples_max, n_wavenumbers) — updated in-place
    dtauk: np.ndarray,     # (n_k_samples_max, n_wavenumbers) — new species
    ng_ik: int,
    n_k_samples_max: int,
    indg: int,
    fracg: float,
    samples_k: np.ndarray,
    weights_k: np.ndarray,
    i_wn: int,             # wavenumber index
) -> None:
    """
    Combine two k-distributions at wavenumber index *i_wn* using the
    random overlap assumption (Fortran ``amedk / amedg`` logic).
    """
    # amedg = log-interp of dtau at indg, fracg
    a, b = dtau[indg, i_wn], dtau[indg + 1, i_wn]
    if a <= 1e-40 or b <= 1e-40:
        amedg = fracg * a + (1.0 - fracg) * b
    else:
        amedg = math.exp(fracg * math.log(a) + (1.0 - fracg) * math.log(b))

    a, b = dtauk[indg, i_wn], dtauk[indg + 1, i_wn]
    if a <= 1e-40 or b <= 1e-40:
        amedk = fracg * a + (1.0 - fracg) * b
    else:
        amedk = math.exp(fracg * math.log(a) + (1.0 - fracg) * math.log(b))

    if (amedg <= amedk * 1e-2
            and dtau[ng_ik - 1, i_wn] <= 1e-2 * dtauk[ng_ik - 1, i_wn]):
        for ig in range(ng_ik):
            dtau[ig, i_wn] = dtauk[ig, i_wn] + amedg
        return

    if (amedk <= amedg * 1e-2
            and dtauk[ng_ik - 1, i_wn] <= 1e-2 * dtau[ng_ik - 1, i_wn]):
        for ig in range(ng_ik):
            dtau[ig, i_wn] += amedk
        return

    # Full random-overlap resampling
    n_combined = ng_ik * ng_ik
    wmix    = np.empty(n_combined)
    dtausum = np.empty(n_combined)
    k = 0
    for ig1 in range(ng_ik):
        for ig2 in range(ng_ik):
            wmix[k]    = weights_k[ig1] * weights_k[ig2]
            val = dtau[ig1, i_wn] + dtauk[ig2, i_wn]
            dtausum[k] = math.log(val) if val > 1e-40 else math.log(1e-40)
            k += 1

    # Sort by dtausum (numba-compatible argsort)
    order = np.argsort(dtausum)
    dtausum_s = dtausum[order]
    wmix_s    = wmix[order]

    # Cumulative weights
    wmix_cum = np.empty(n_combined)
    s = 0.0
    for k in range(n_combined):
        s += wmix_s[k]
        wmix_cum[k] = s

    # Per-quadrature-point: 1D linear interpolation in log-tau space, with
    # *linear extrapolation* outside the support (matches math_utils.interp_ex).
    for ig in range(ng_ik):
        x = samples_k[ig]
        idx = np.searchsorted(wmix_cum, x)
        if idx <= 0:
            # extrapolate left
            x0, x1 = wmix_cum[0], wmix_cum[1] if n_combined > 1 else wmix_cum[0]
            y0, y1 = dtausum_s[0], dtausum_s[1] if n_combined > 1 else dtausum_s[0]
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
            log_tau = y0 + slope * (x - x0)
        elif idx >= n_combined:
            # extrapolate right
            x0, x1 = wmix_cum[-2], wmix_cum[-1]
            y0, y1 = dtausum_s[-2], dtausum_s[-1]
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
            log_tau = y1 + slope * (x - x1)
        else:
            # interpolate
            x0, x1 = wmix_cum[idx - 1], wmix_cum[idx]
            y0, y1 = dtausum_s[idx - 1], dtausum_s[idx]
            if x1 > x0:
                log_tau = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            else:
                log_tau = y0
        dtau[ig, i_wn] = math.exp(log_tau)
