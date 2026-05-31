#!/usr/bin/env python3
"""
dump_chem_opacity.py  -- read-only.  Runs ONE RT *after* thermochemical
equilibrium (like the corrected harness) and dumps:
  (A) the equilibrium composition at a cold-upper, a mid, and a deep layer,
  (B) the refractory/alkali gas VMRs in the cold upper atmosphere (tests the
      "no condensation/rainout" hypothesis),
  (C) where the optical depth lives in wavelength (thermal IR vs optical),
  (D) cumulative tau vs level at the Planck peak and at the Na/K optical band.

Run from project root:
    python dump_chem_opacity.py \
        --nml inputs/example_no_irr.nml \
        --profile inputs/atmospheres/temperature_profiles/fortran_profile_500K.dat

Writes nothing except chem_opacity_dump.npz.  Paste stdout back.
"""
from __future__ import annotations
import argparse, sys
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nml", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--no-chemistry", action="store_true")
    args = ap.parse_args()

    from exorem import exorem_main as em
    from exorem.physics import CST_SIGMA
    try:
        from exorem.chemistry import GASES_NAMES
    except Exception:
        GASES_NAMES = None

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
    h2_h2_cia  = state["h2_h2_cia"]; h2_he_cia = state["h2_he_cia"]
    h2o_n2_cia = state["h2o_n2_cia"]; h2o_h2o_cia = state["h2o_h2o_cia"]

    n_levels      = atm.n_levels
    n_layers      = atm.n_layers
    n_wavenumbers = spectrometrics.n_wavenumbers
    wn            = np.asarray(spectrometrics.wavenumbers, float)
    T_int         = float(target.target_internal_temperature)
    Mg            = gases_molar_mass * 1e3

    def gname(i):
        if GASES_NAMES and i < len(GASES_NAMES):
            return GASES_NAMES[i]
        return f"gas{i}(M={Mg[i]:.1f})"

    # install profile
    data = np.genfromtxt(args.profile, comments="#")
    p_ref, t_ref = data[:, 0], data[:, 1]
    order = np.argsort(p_ref)
    atm.temperatures[:] = np.interp(np.log(atm.pressures),
                                    np.log(p_ref[order]), t_ref[order])
    atm.temperatures_layers[:] = np.sqrt(
        atm.temperatures[:n_layers] * atm.temperatures[1:n_levels])
    em._calculate_altitude(atm, target, gases_molar_mass, gases_vmr)

    if not args.no_chemistry:
        N_COND = em.N_CONDENSATES
        em._calculate_thermochemical_equilibrium(
            atm, spec, gases_vmr, species_vmr_layers,
            np.zeros(N_COND), np.zeros((N_COND, n_layers)),
            np.zeros(N_COND), np.zeros(N_COND, dtype=int),
            gases_molar_mass, gases_delta_g, condensates_delta_g,
            temperatures_thermo, gases_c_p, elements_in_gases)
        em._calculate_altitude(atm, target, gases_molar_mass, gases_vmr)

    # choose 3 layers: cold-upper, mid, deep  (layer index = level index here)
    def nearest_layer(p_target):
        return int(np.argmin(np.abs(atm.pressures_layers - p_target)))
    L_cold = nearest_layer(1.0e2)     # ~1e-3 bar
    L_mid  = nearest_layer(1.0e4)     # ~0.1 bar
    L_deep = nearest_layer(1.0e6)     # ~10 bar

    print("=== (A) equilibrium composition (top 14 gases by VMR) ===")
    for lab, L in [("COLD-UPPER", L_cold), ("MID", L_mid), ("DEEP", L_deep)]:
        v = gases_vmr[:, L]
        mu = float(np.sum(v * Mg))
        top = np.argsort(v)[::-1][:14]
        print(f"\n  {lab}: layer {L}  P={atm.pressures_layers[L]:.3e} Pa  "
              f"T={atm.temperatures_layers[L]:.1f} K  mu={mu:.3f} g/mol  "
              f"sumVMR={v.sum():.4f}")
        for i in top:
            if v[i] <= 0:
                continue
            print(f"      {gname(i):6s}  VMR={v[i]:.4e}")

    print("\n=== (B) refractory / alkali gas VMRs in the COLD upper atmosphere ===")
    print("    (if these are ~solar X/H rather than ~0, condensation/rainout is NOT applied)")
    refr = ["Fe", "FeH", "Na", "NaCl", "K", "KCl", "Ti", "TiO", "TiO2",
            "V", "VO", "VO2", "Al", "Ca", "Cr", "Mg", "Mn", "Ni", "SiH4", "SiO", "Zn"]
    if GASES_NAMES:
        name2idx = {n: k for k, n in enumerate(GASES_NAMES)}
        for nm in refr:
            if nm in name2idx:
                i = name2idx[nm]
                print(f"      {nm:6s}  VMR(cold L{L_cold})={gases_vmr[i, L_cold]:.4e}   "
                      f"VMR(deep L{L_deep})={gases_vmr[i, L_deep]:.4e}")

    # one RT
    tau          = np.zeros((n_levels, n_wavenumbers, state["ng_max"]))
    tau_rayleigh = np.zeros((n_levels, n_wavenumbers))
    (tau, tau_rayleigh, tau_cloud_out,
     radiosity_internal, matrix_t, flux,
     spectral_radiosity) = em._do_radiative_transfer(
        state, gases_vmr, cloud_vmr, 0, atm, spec, cloud_obj,
        spectrometrics, light, kcoeff_tables,
        h2_h2_cia, h2_he_cia, h2o_n2_cia, h2o_h2o_cia,
        tau, tau_rayleigh)

    print("\n=== (C) where does the optical depth live? (column = level 0) ===")
    print("    wn[cm^-1]  band      colTau(min_g)  colTau(max_g)  cumTau->coldL%d(g0)  tau_ray_col"
          % L_cold)
    sel = list(range(0, n_wavenumbers, max(1, n_wavenumbers // 16)))
    for j in sel:
        band = "thermal" if wn[j] <= 2500 else ("near-IR" if wn[j] <= 8130 else "OPTICAL/ext")
        cmin = tau[0, j, :].min(); cmax = tau[0, j, :].max()
        ccold = tau[L_cold, j, 0]
        print(f"   {wn[j]:8.1f}  {band:11s} {cmin:.3e}    {cmax:.3e}    "
              f"{ccold:.3e}        {tau_rayleigh[0, j]:.3e}")

    print("\n=== (D) cumulative tau vs level (g=0) at two diagnostic wavenumbers ===")
    j_ir  = int(np.argmin(np.abs(wn - 1.95 * T_int)))   # ~Planck peak for 500 K
    j_opt = int(np.argmin(np.abs(wn - 16900.0)))        # Na D / optical
    print(f"   IR wn={wn[j_ir]:.0f} cm^-1 ; OPT wn={wn[j_opt]:.0f} cm^-1")
    print("   level   P[Pa]      T[K]     tau_IR(g0)    tau_OPT(g0)")
    for lv in range(n_levels - 1, -1, max(1, n_levels // 16)):
        print(f"    {lv:3d}  {atm.pressures[lv]:.3e}  {atm.temperatures[lv]:7.1f}  "
              f"{tau[lv, j_ir, 0]:.4e}   {tau[lv, j_opt, 0]:.4e}")

    sig_cgs = CST_SIGMA * 1e3
    emergent = np.asarray(spectral_radiosity[-1, :], float)
    bol = float(np.sum(emergent) * spectrometrics.wavenumber_step)
    print("\n=== recap ===")
    print(f"  T_eff = {(bol/sig_cgs)**0.25:.2f} K   target T_int = {T_int:.1f} K")
    print(f"  column tau median over wn (g0) = {np.median(tau[0,:,0]):.3e}")

    np.savez_compressed("chem_opacity_dump.npz",
        wavenumbers=wn, pressures=atm.pressures, temperatures=atm.temperatures,
        gases_vmr=gases_vmr, gases_molar_mass=gases_molar_mass,
        tau=tau, tau_rayleigh=tau_rayleigh,
        radiosity_internal=radiosity_internal, spectral_radiosity=spectral_radiosity,
        T_int=T_int)
    print("\nSaved chem_opacity_dump.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
