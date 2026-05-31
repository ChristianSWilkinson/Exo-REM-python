#!/usr/bin/env python3
"""
verify_with_fortran_vmrs.py  -- isolates the RT + opacity from Python's chemistry.

Loads Fortran's converged composition (the 13 absorber VMRs + H2/He), its
mean molar mass, and its T-profile from the Fortran HDF5-dump CSV, injects
them into Python's state, runs ONE radiative transfer, and compares Python's
emergent spectrum to Fortran's.

If T_eff ~= 500 K here, the forward model / opacity / scale_height-units are
correct and the only remaining work is chemistry (condensation + Kzz quench).
If it is still ~190 K, the problem is in the RT/opacity and needs more work.

Run from project root (output.csv in CWD or pass --fortran):
    python verify_with_fortran_vmrs.py \
        --nml inputs/example_no_irr.nml \
        --fortran output.csv
"""
from __future__ import annotations
import argparse, csv, sys
import numpy as np


def load_fortran_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    D = dict(zip(rows[0], rows[1]))
    def arr(key):
        s = D[key].strip().replace('[', ' ').replace(']', ' ')
        return np.fromstring(s, sep=' ')
    def sc(key):
        return float(D[key])
    return D, arr, sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nml", required=True)
    ap.add_argument("--fortran", default="output.csv")
    args = ap.parse_args()

    from exorem import exorem_main as em
    from exorem.physics import CST_SIGMA, CST_R
    from exorem.chemistry import gas_id

    D, arr, sc = load_fortran_csv(args.fortran)

    ABS = ["CH4", "CO", "CO2", "FeH", "H2O", "H2S", "HCN", "K", "NH3", "Na", "PH3", "TiO", "VO"]
    base = "/outputs/layers/volume_mixing_ratios/"
    vmr_abs = {sp: arr(base + "absorbers/" + sp) for sp in ABS}
    vmr_H2  = arr(base + "gases/H2")
    vmr_He  = arr(base + "gases/He")
    P_lay_f = arr("/outputs/layers/pressure")
    T_lay_f = arr("/outputs/layers/temperature")
    mu_f    = arr("/outputs/layers/mean_molar_mass")      # g/mol
    T_lev_f = arr("/outputs/levels/temperature")
    P_lev_f = arr("/outputs/levels/pressure")
    wn_f    = arr("/outputs/spectra/wavenumber")
    sr_f    = arr("/outputs/spectra/emission/spectral_radiosity")
    Tint    = sc("/model_parameters/target/internal_temperature")

    print("[1/5] init ...")
    state = em._init_exorem(args.nml)
    atm            = state["atm"]; target = state["target"]; light = state["light"]
    spec           = state["spec"]; spectrometrics = state["spectrometrics"]
    cloud_obj      = state["cloud_obj"]
    gases_vmr      = state["gases_vmr"]; cloud_vmr = state["cloud_vmr"]
    gases_molar_mass = state["gases_molar_mass"]
    kcoeff_tables  = state["kcoeff_tables"]
    h2_h2_cia = state["h2_h2_cia"]; h2_he_cia = state["h2_he_cia"]
    h2o_n2_cia = state["h2o_n2_cia"]; h2o_h2o_cia = state["h2o_h2o_cia"]
    n_levels = atm.n_levels; n_layers = atm.n_layers
    nwn = spectrometrics.n_wavenumbers

    # ---- align layer ordering (both should be the same 80-layer grid) ----
    same_dir = (P_lay_f[0] > P_lay_f[-1]) == (atm.pressures_layers[0] > atm.pressures_layers[-1])
    def lay(a):
        return a if same_dir else a[::-1]
    dP = np.abs(lay(P_lay_f) - atm.pressures_layers) / np.maximum(atm.pressures_layers, 1e-30)
    print(f"      layer-grid max relative P mismatch = {dP.max():.2e} "
          f"({'direct' if same_dir else 'flipped'})")

    print("[2/5] installing Fortran profile + composition ...")
    # profile (levels) -- interpolate Fortran level T onto Python's level grid in log-P
    o = np.argsort(P_lev_f)
    atm.temperatures[:] = np.interp(np.log(atm.pressures), np.log(P_lev_f[o]), T_lev_f[o])
    atm.temperatures_layers[:] = lay(T_lay_f)

    em._calculate_altitude(atm, target, gases_molar_mass, gases_vmr)  # sets z, gravities

    # inject Fortran VMRs into gases_vmr (absorbers + H2/He); zero the other
    # carbon/metal gases that have no k-table anyway is unnecessary -- only the
    # 13 absorbers and H2/He/H2O matter for opacity + CIA.
    for sp in ABS:
        gases_vmr[gas_id(sp), :] = lay(vmr_abs[sp])
    gases_vmr[gas_id("H2"), :] = lay(vmr_H2)
    gases_vmr[gas_id("He"), :] = lay(vmr_He)

    # use Fortran's mean molar mass for scale height (SI metres).
    # Fortran stores mean_molar_mass in kg/mol (SI); auto-detect to be safe
    # (kg/mol ~ 0.002, g/mol ~ 2) so a format change can't double-convert.
    mu_raw = lay(mu_f)
    mu_kg = mu_raw if np.median(mu_raw) < 0.1 else mu_raw * 1e-3
    print(f"      (Fortran mu raw median = {np.median(mu_raw):.4e} -> "
          f"{'kg/mol' if np.median(mu_raw) < 0.1 else 'g/mol'})")
    atm.molar_masses_layers[:] = mu_kg
    atm.scale_height[:] = CST_R * atm.temperatures_layers / (mu_kg * atm.gravities_layers)
    print(f"      mu range {lay(mu_f).min():.3f}-{lay(mu_f).max():.3f} g/mol; "
          f"scale_height {atm.scale_height.min():.3e}-{atm.scale_height.max():.3e} m")

    print("[3/5] one RT ...")
    tau          = np.zeros((n_levels, nwn, state["ng_max"]))
    tau_rayleigh = np.zeros((n_levels, nwn))
    (tau, tau_rayleigh, tau_cloud_out,
     radiosity_internal, matrix_t, flux,
     spectral_radiosity) = em._do_radiative_transfer(
        state, gases_vmr, cloud_vmr, 0, atm, spec, cloud_obj,
        spectrometrics, light, kcoeff_tables,
        h2_h2_cia, h2_he_cia, h2o_n2_cia, h2o_h2o_cia,
        tau, tau_rayleigh)

    print("[4/5] diagnostics\n")
    sig_cgs = CST_SIGMA * 1e3
    dwn = spectrometrics.wavenumber_step
    em_py = np.asarray(spectral_radiosity[-1, :], float)
    bol_py = float(np.sum(em_py) * dwn)
    Teff_py_cgs = (bol_py / sig_cgs) ** 0.25
    Teff_py_si  = (bol_py / CST_SIGMA) ** 0.25
    print(f"  Python T_eff (treating radiosity as CGS) = {Teff_py_cgs:.2f} K")
    print(f"  Python T_eff (treating radiosity as SI)  = {Teff_py_si:.2f} K")
    print(f"  target T_int = {Tint:.1f} K")
    tau_med = np.median(tau[:, :, 0], axis=1)
    for lv in range(n_levels - 1, -1, -1):
        if tau_med[lv] >= 2.0/3.0:
            print(f"  photosphere (median tau=2/3) at level {lv} "
                  f"(P={atm.pressures[lv]:.3e} Pa, T={atm.temperatures[lv]:.1f} K)")
            break

    print("\n[5/5] Python vs Fortran emergent spectrum (Fortran in W/m^2, "
          "Python/1000 to compare):")
    print("   wn[cm^-1]   Fortran_sr     Python_sr/1000   ratio")
    sel = list(range(0, nwn, max(1, nwn // 16)))
    srf = sr_f if len(sr_f) == nwn else np.interp(spectrometrics.wavenumbers, wn_f, sr_f)
    for j in sel:
        pj = em_py[j] / 1e3
        r = pj / srf[j] if srf[j] > 0 else float('nan')
        print(f"   {spectrometrics.wavenumbers[j]:8.1f}  {srf[j]:.4e}    {pj:.4e}    {r:.3f}")

    bol_f = float(np.sum(srf) * dwn)
    print(f"\n  bolometric: Fortran = {bol_f:.4e} W/m^2  (T_eff={(bol_f/CST_SIGMA)**0.25:.1f} K)")
    print(f"              Python  = {bol_py/1e3:.4e} W/m^2  (Python/1000)")
    print(f"  ratio Python/Fortran (bolometric) = {(bol_py/1e3)/bol_f:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
