#!/usr/bin/env python3
"""
verify_forward_model.py  -- single-shot forward-model check (now with chemistry).

Pipeline (matches one iteration of the real solver up to the RT):
    init -> install Fortran T-profile -> altitude
         -> thermochemical equilibrium  (so VMRs/mu are physical, not the
            vmr_example_ref.dat placeholder)
         -> altitude again (mu now from equilibrium VMRs)
         -> ONE radiative transfer
and reports J_int, T_int and T_eff at the top of the atmosphere.

The earlier version skipped chemistry and therefore ran the RT with the
a-priori placeholder composition (H2O=0.15, CH4=0.09, NH3=0.06; mu=6.78
g/mol), which jammed the IR photosphere into the cold upper layers and gave
T_eff~=224 K.  With chemistry on, mu should fall to ~2.3-2.5 g/mol and -- if
the forward model and chemistry match Fortran -- T_eff should approach T_int.

Run from the project root:
    python verify_forward_model.py \
        --nml inputs/example_no_irr.nml \
        --profile inputs/atmospheres/temperature_profiles/fortran_profile_500K.dat

Flags:
    --no-chemistry   reproduce the old placeholder-VMR behaviour (mu=6.78)
    --dump-tau PATH  save the (level, wn, g) optical-depth array to PATH
"""
from __future__ import annotations
import argparse, sys
import numpy as np


def mean_mu(gases_vmr, gases_molar_mass, layer):
    return float(np.sum(gases_vmr[:, layer] * gases_molar_mass) * 1e3)  # g/mol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nml", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--no-chemistry", action="store_true")
    ap.add_argument("--dump-tau", default=None)
    args = ap.parse_args()

    from exorem import exorem_main as em
    from exorem.physics import CST_SIGMA

    print("[1/6] initialising model state ...")
    state = em._init_exorem(args.nml)
    atm            = state["atm"]
    target         = state["target"]
    light          = state["light"]
    spec           = state["spec"]
    spectrometrics = state["spectrometrics"]
    cloud_obj      = state["cloud_obj"]
    gases_vmr          = state["gases_vmr"]
    species_vmr_layers = state["species_vmr_layers"]
    gases_molar_mass   = state["gases_molar_mass"]
    gases_delta_g      = state["gases_delta_g"]
    condensates_delta_g= state["condensates_delta_g"]
    temperatures_thermo= state["temperatures_thermo"]
    gases_c_p          = state["gases_c_p"]
    elements_in_gases  = state["elements_in_gases"]
    cloud_vmr          = state["cloud_vmr"]
    kcoeff_tables      = state["kcoeff_tables"]
    h2_h2_cia  = state["h2_h2_cia"]
    h2_he_cia  = state["h2_he_cia"]
    h2o_n2_cia = state["h2o_n2_cia"]
    h2o_h2o_cia= state["h2o_h2o_cia"]

    n_levels      = atm.n_levels
    n_layers      = atm.n_layers
    n_wavenumbers = spectrometrics.n_wavenumbers
    T_int         = float(target.target_internal_temperature)
    L = 31 if n_layers > 31 else n_layers // 2

    print(f"[2/6] installing reference profile {args.profile!r} ...")
    data = np.genfromtxt(args.profile, comments="#")
    p_ref, t_ref = data[:, 0], data[:, 1]
    order = np.argsort(p_ref)
    T_on_grid = np.interp(np.log(atm.pressures), np.log(p_ref[order]), t_ref[order])
    if len(p_ref) == n_levels:
        dp = np.abs(np.sort(p_ref)[::-1] - atm.pressures) / np.maximum(atm.pressures, 1e-30)
        print(f"      {len(p_ref)} levels; max relative P mismatch = {dp.max():.2e}")
    atm.temperatures[:] = T_on_grid
    atm.temperatures_layers[:] = np.sqrt(
        atm.temperatures[:n_layers] * atm.temperatures[1:n_levels])

    print("[3/6] altitude (pre-chemistry) ...")
    em._calculate_altitude(atm, target, gases_molar_mass, gases_vmr)
    print(f"      mu(layer {L}) = {mean_mu(gases_vmr, gases_molar_mass, L):.3f} g/mol "
          f"(placeholder a-priori composition)")

    if not args.no_chemistry:
        print("[4/6] thermochemical equilibrium on the installed profile ...")
        N_COND = em.N_CONDENSATES
        p_c_condensates     = np.zeros(N_COND)
        vmr_sat_condensates = np.zeros((N_COND, n_layers))
        vmr_c_condensates   = np.zeros(N_COND)
        layer_condensates   = np.zeros(N_COND, dtype=int)
        em._calculate_thermochemical_equilibrium(
            atm, spec, gases_vmr, species_vmr_layers,
            p_c_condensates, vmr_sat_condensates,
            vmr_c_condensates, layer_condensates,
            gases_molar_mass, gases_delta_g, condensates_delta_g,
            temperatures_thermo, gases_c_p, elements_in_gases,
        )
        em._calculate_altitude(atm, target, gases_molar_mass, gases_vmr)
        print(f"      mu(layer {L}) = {mean_mu(gases_vmr, gases_molar_mass, L):.3f} g/mol "
              f"(equilibrium); scale_height "
              f"{atm.scale_height.min():.3e}-{atm.scale_height.max():.3e} m")
    else:
        print("[4/6] chemistry SKIPPED (--no-chemistry): using placeholder VMRs")

    print("[5/6] one radiative-transfer evaluation ...")
    tau          = np.zeros((n_levels, n_wavenumbers, state["ng_max"]))
    tau_rayleigh = np.zeros((n_levels, n_wavenumbers))
    (tau, tau_rayleigh, tau_cloud_out,
     radiosity_internal, matrix_t, flux,
     spectral_radiosity) = em._do_radiative_transfer(
        state, gases_vmr, cloud_vmr, 0, atm, spec, cloud_obj,
        spectrometrics, light, kcoeff_tables,
        h2_h2_cia, h2_he_cia, h2o_n2_cia, h2o_h2o_cia,
        tau, tau_rayleigh)
    if args.dump_tau:
        np.save(args.dump_tau, tau)
        print(f"      saved tau -> {args.dump_tau}")

    print("[6/6] diagnostics\n")
    sig_cgs = CST_SIGMA * 1e3
    J_int = float(radiosity_internal[-1])
    J_tgt = sig_cgs * T_int**4
    T_int_meas = np.sign(J_int) * (abs(J_int) / sig_cgs) ** 0.25
    T_eff = (float(np.sum(spectral_radiosity[-1, :]) * spectrometrics.wavenumber_step)
             / sig_cgs) ** 0.25

    tau_med = np.median(tau[:, :, 0], axis=1)
    photo = None
    for lv in range(n_levels - 1, -1, -1):
        if tau_med[lv] >= 2.0 / 3.0:
            photo = lv; break

    print(f"  J_int / (sigma*T_int_th^4) = {J_int / J_tgt:.6g}")
    print(f"  T_int = {T_int_meas:.2f} K   (T_int_th = {T_int:.2f} K)")
    print(f"  T_eff = {T_eff:.2f} K")
    print(f"  total column tau (median over wn, g=0) = {tau_med[0]:.3e}")
    if photo is not None:
        print(f"  photosphere (median tau=2/3) at level {photo} "
              f"(P={atm.pressures[photo]:.3e} Pa, T={atm.temperatures[photo]:.1f} K)")

    rel = abs(T_eff - T_int) / T_int
    print()
    if rel < 0.10:
        print(f"  PASS: T_eff within {rel*100:.1f}% of T_int_th.")
        return 0
    print(f"  FAIL: T_eff off by {rel*100:.1f}%. If mu is now ~2.3-2.5 but T_eff is "
          f"still low,\n        the residual is in the opacity/chemistry match to "
          f"Fortran (next audit stage),\n        not the scale_height units.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
