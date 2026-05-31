#!/usr/bin/env python3
"""
dump_rt_state.py  -- read-only forward-model diagnostic.

Same one-shot RT as verify_forward_model.py (init -> install Fortran profile ->
recompute altitude -> ONE radiative transfer), but instead of a pass/fail it
dumps the intermediate quantities needed to find why the emergent flux is ~25x
too low and why the mean molar mass is ~6.8 g/mol.

Run from the project root:
    python dump_rt_state.py \
        --nml inputs/example_no_irr.nml \
        --profile inputs/atmospheres/temperature_profiles/fortran_profile_500K.dat

Nothing is written except rt_dump.npz in the CWD.  Paste the stdout back.
"""
from __future__ import annotations
import argparse, sys
import numpy as np

# CODATA CGS constants for an independent Planck / brightness-temperature check
H_CGS = 6.62607015e-27     # erg s
C_CGS = 2.99792458e10      # cm s^-1
K_CGS = 1.380649e-16       # erg K^-1


def planck_cgs(nu_cm, T):
    """B_nu(T) per wavenumber, erg s^-1 cm^-2 sr^-1 / cm^-1.  nu_cm in cm^-1."""
    nu_cm = np.asarray(nu_cm, float)
    x = H_CGS * C_CGS * nu_cm / (K_CGS * max(T, 1e-6))
    return 2.0 * H_CGS * C_CGS**2 * nu_cm**3 / np.expm1(np.clip(x, 1e-30, 700.0))


def brightness_T(nu_cm, B):
    """Invert planck_cgs for T given spectral radiance B (same units)."""
    nu_cm = np.asarray(nu_cm, float); B = np.asarray(B, float)
    out = np.full_like(B, np.nan)
    good = B > 0
    arg = 1.0 + 2.0 * H_CGS * C_CGS**2 * nu_cm[good]**3 / B[good]
    out[good] = H_CGS * C_CGS * nu_cm[good] / (K_CGS * np.log(arg))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nml", required=True)
    ap.add_argument("--profile", required=True)
    args = ap.parse_args()

    from exorem import exorem_main as em
    from exorem.physics import CST_SIGMA

    print("=== setup ===")
    state = em._init_exorem(args.nml)
    atm            = state["atm"]
    target         = state["target"]
    light          = state["light"]
    spec           = state["spec"]
    spectrometrics = state["spectrometrics"]
    cloud_obj      = state["cloud_obj"]
    gases_vmr        = state["gases_vmr"]            # (N_GASES, n_layers)
    gases_molar_mass = state["gases_molar_mass"]     # kg/mol
    cloud_vmr        = state["cloud_vmr"]
    kcoeff_tables    = state["kcoeff_tables"]
    h2_h2_cia        = state["h2_h2_cia"]
    h2_he_cia        = state["h2_he_cia"]
    h2o_n2_cia       = state["h2o_n2_cia"]
    h2o_h2o_cia      = state["h2o_h2o_cia"]

    n_levels      = atm.n_levels
    n_layers      = atm.n_layers
    n_wavenumbers = spectrometrics.n_wavenumbers
    wn            = np.asarray(spectrometrics.wavenumbers, float)
    dwn           = float(spectrometrics.wavenumber_step)
    T_int         = float(target.target_internal_temperature)
    sig_cgs       = CST_SIGMA * 1e3      # erg s^-1 cm^-2 K^-4

    # install Fortran profile (same as the harness)
    data = np.genfromtxt(args.profile, comments="#")
    p_ref, t_ref = data[:, 0], data[:, 1]
    order = np.argsort(p_ref)
    atm.temperatures[:] = np.interp(np.log(atm.pressures),
                                    np.log(p_ref[order]), t_ref[order])
    atm.temperatures_layers[:] = np.sqrt(
        atm.temperatures[:n_layers] * atm.temperatures[1:n_levels])
    em._calculate_altitude(atm, target, gases_molar_mass, gases_vmr)

    # ---------------------------------------------------------------
    # (A) composition / molar mass
    # ---------------------------------------------------------------
    print("\n=== (A) VMR / mean molar mass ===")
    Mg = gases_molar_mass * 1e3                      # g/mol
    vmr_sum = gases_vmr.sum(axis=0)                  # per layer
    print(f"  N_GASES rows = {gases_vmr.shape[0]}, n_layers = {n_layers}")
    print(f"  sum_i VMR_i  per layer:  min={vmr_sum.min():.4f} "
          f"max={vmr_sum.max():.4f} mean={vmr_sum.mean():.4f}  (should be ~1)")
    # try to find H2 / He by molar mass
    def find_idx(target_M):
        cand = np.where(np.abs(Mg - target_M) < 0.2)[0]
        return cand[0] if len(cand) else None
    iH2, iHe = find_idx(2.016), find_idx(4.003)
    L = 31 if n_layers > 31 else n_layers // 2
    print(f"  H2 index = {iH2}, He index = {iHe}")
    if iH2 is not None:
        print(f"    VMR_H2 at layer {L} = {gases_vmr[iH2, L]:.4e}")
    if iHe is not None:
        print(f"    VMR_He at layer {L} = {gases_vmr[iHe, L]:.4e}")
    # top contributors to mu at the photosphere layer
    contrib = gases_vmr[:, L] * Mg                   # g/mol contribution
    topc = np.argsort(contrib)[::-1][:10]
    print(f"  top molar-mass contributors at layer {L} "
          f"(P={atm.pressures_layers[L]:.3e} Pa, T={atm.temperatures_layers[L]:.1f} K):")
    print("     idx     M[g/mol]      VMR        VMR*M")
    for i in topc:
        print(f"    {i:4d}   {Mg[i]:9.3f}   {gases_vmr[i, L]:.4e}   {contrib[i]:.4e}")
    print(f"  sum VMR*M at layer {L} = {contrib.sum():.4f} g/mol  "
          f"(this is mu); sum VMR = {vmr_sum[L]:.4f}")
    print(f"  atm.molar_masses_layers: min={atm.molar_masses_layers.min():.4e} "
          f"max={atm.molar_masses_layers.max():.4e} kg/mol")
    print(f"  atm.scale_height:        min={atm.scale_height.min():.4e} "
          f"max={atm.scale_height.max():.4e} m")
    print(f"  target_gravity = {target.target_gravity:.4f} m/s^2")

    # ---------------------------------------------------------------
    # one RT evaluation
    # ---------------------------------------------------------------
    print("\n=== running one RT ===")
    tau          = np.zeros((n_levels, n_wavenumbers, state["ng_max"]))
    tau_rayleigh = np.zeros((n_levels, n_wavenumbers))
    (tau, tau_rayleigh, tau_cloud_out,
     radiosity_internal, matrix_t, flux,
     spectral_radiosity) = em._do_radiative_transfer(
        state, gases_vmr, cloud_vmr, 0, atm, spec, cloud_obj,
        spectrometrics, light, kcoeff_tables,
        h2_h2_cia, h2_he_cia, h2o_n2_cia, h2o_h2o_cia,
        tau, tau_rayleigh)

    # ---------------------------------------------------------------
    # (B) optical depth structure
    # ---------------------------------------------------------------
    print("\n=== (B) optical depth (g=0 quadrature point) ===")
    # column tau = cumulative value at the deepest level (index 0)
    col_tau = tau[0, :, 0]
    print(f"  tau shape = {tau.shape}  (levels, wavenumbers, ng)")
    print(f"  column tau over wavenumber: min={col_tau.min():.3e} "
          f"max={col_tau.max():.3e} median={np.median(col_tau):.3e}")
    print(f"  tau_rayleigh column (level 0): min={tau_rayleigh[0].min():.3e} "
          f"max={tau_rayleigh[0].max():.3e}")
    jpk = int(np.argmin(np.abs(wn - 1.95 * T_int)))   # near Planck peak
    print(f"  near Planck-peak wavenumber wn[{jpk}] = {wn[jpk]:.1f} cm^-1; "
          f"cumulative tau vs level (top->deep), g=0:")
    lev_list = list(range(n_levels - 1, -1, max(1, n_levels // 12)))
    for lv in lev_list:
        print(f"     level {lv:3d}  P={atm.pressures[lv]:.3e} Pa  "
              f"T={atm.temperatures[lv]:7.1f} K  tau={tau[lv, jpk, 0]:.4e}")

    # ---------------------------------------------------------------
    # (C) ENERGY CONSERVATION: net flux vs level (the key diagnostic)
    # ---------------------------------------------------------------
    print("\n=== (C) net flux (radiosity_internal) vs level ===")
    target_flux = sig_cgs * T_int**4
    print(f"  target sigma*T_int^4   = {target_flux:.4e} erg/s/cm^2  (T_int={T_int:.1f})")
    print(f"  sigma*T_bottom^4       = {sig_cgs*atm.temperatures[0]**4:.4e} "
          f"(T_bot={atm.temperatures[0]:.1f})")
    print(f"  sigma*T_top^4          = {sig_cgs*atm.temperatures[-1]**4:.4e} "
          f"(T_top={atm.temperatures[-1]:.1f})")
    ri = np.asarray(radiosity_internal, float)
    print(f"  radiosity_internal: min={ri.min():.4e} max={ri.max():.4e} "
          f"max/min={ri.max()/max(abs(ri.min()),1e-30):.3e}")
    print("  level    P[Pa]      T[K]     J_net[erg/s/cm2]   J_net/target")
    for lv in range(n_levels - 1, -1, -1):
        print(f"   {lv:3d}  {atm.pressures[lv]:.3e}  {atm.temperatures[lv]:7.1f}  "
              f"{ri[lv]:+.4e}      {ri[lv]/target_flux:+.4f}")

    # ---------------------------------------------------------------
    # (D) emergent spectrum + per-channel brightness temperature
    # ---------------------------------------------------------------
    print("\n=== (D) emergent spectrum (top level) ===")
    emergent = np.asarray(spectral_radiosity[-1, :], float)   # erg/s/cm2/cm^-1
    bol = float(np.sum(emergent) * dwn)
    Teff = (bol / sig_cgs) ** 0.25 if bol > 0 else float("nan")
    print(f"  sum(emergent)*dwn = {bol:.4e} erg/s/cm^2  -> T_eff = {Teff:.2f} K")
    print(f"  emergent spectral flux: min={emergent.min():.3e} "
          f"max={emergent.max():.3e}")
    # compare to a 500 K and a 427 K blackbody flux (pi*B) channel by channel
    print("   wn[cm^-1]  emergent_flux   T_bright[K]   pi*B(500)flux   ratio_em/B500")
    sel = list(range(0, n_wavenumbers, max(1, n_wavenumbers // 14)))
    BbT = brightness_T(wn, emergent / np.pi)        # invert pi*B = emergent
    piB500 = np.pi * planck_cgs(wn, 500.0)
    for j in sel:
        r = emergent[j] / piB500[j] if piB500[j] > 0 else float("nan")
        print(f"   {wn[j]:8.1f}  {emergent[j]:.4e}   {BbT[j]:8.1f}    "
              f"{piB500[j]:.4e}    {r:.3f}")

    # ---------------------------------------------------------------
    # (E) the 'flux' array (net spectral flux per level), top & bottom
    # ---------------------------------------------------------------
    print("\n=== (E) 'flux' array sanity ===")
    fl = np.asarray(flux, float)
    print(f"  flux shape = {fl.shape}")
    print(f"  sum(flux[top,:])*dwn    = {np.sum(fl[-1,:])*dwn:+.4e}")
    print(f"  sum(flux[bottom,:])*dwn = {np.sum(fl[0,:])*dwn:+.4e}")
    print(f"  sum(flux[mid,:])*dwn    = {np.sum(fl[n_levels//2,:])*dwn:+.4e}")

    np.savez_compressed(
        "rt_dump.npz",
        wavenumbers=wn, dwn=dwn, pressures=atm.pressures,
        temperatures=atm.temperatures, scale_height=atm.scale_height,
        molar_masses_layers=atm.molar_masses_layers,
        gases_vmr=gases_vmr, gases_molar_mass=gases_molar_mass,
        tau=tau, tau_rayleigh=tau_rayleigh,
        radiosity_internal=ri, flux=fl, spectral_radiosity=spectral_radiosity,
        T_int=T_int)
    print("\nSaved full arrays to rt_dump.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
