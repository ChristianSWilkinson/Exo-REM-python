#!/usr/bin/env python3
"""
dump_planck_cia.py  -- read-only.  With Fortran's VMRs/profile injected:
  (1) checks Python's _planck_array against an analytic CGS blackbody and
      against sigma*T^4 (reveals whether the source function is mis-normalised),
  (2) isolates the emergent flux into full / CIA-only / lines-only to see what
      is clamping the thermal IR.

Run from project root with output.csv present:
    python dump_planck_cia.py --nml inputs/example_no_irr.nml --fortran output.csv
"""
from __future__ import annotations
import argparse, csv, sys
import numpy as np

H_CGS, C_CGS, K_CGS = 6.62607015e-27, 2.99792458e10, 1.380649e-16
def planck_cgs(nu, T):                      # erg s^-1 cm^-2 sr^-1 / cm^-1
    x = H_CGS*C_CGS*np.asarray(nu)/(K_CGS*T)
    return 2*H_CGS*C_CGS**2*np.asarray(nu)**3/np.expm1(np.clip(x,1e-30,700))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nml", required=True)
    ap.add_argument("--fortran", default="output.csv")
    a = ap.parse_args()
    from exorem import exorem_main as em
    from exorem.physics import CST_SIGMA, CST_R
    from exorem.chemistry import gas_id
    from exorem.radiative_transfer import _planck_array

    with open(a.fortran) as f: rows=list(csv.reader(f))
    D=dict(zip(rows[0],rows[1]))
    arr=lambda k: np.fromstring(D[k].strip().replace('[',' ').replace(']',' '),sep=' ')
    sc =lambda k: float(D[k])

    state=em._init_exorem(a.nml)
    atm=state["atm"]; target=state["target"]; light=state["light"]; spec=state["spec"]
    sm=state["spectrometrics"]; cloud_obj=state["cloud_obj"]
    gv=state["gases_vmr"]; cloud_vmr=state["cloud_vmr"]; gmm=state["gases_molar_mass"]
    kct=state["kcoeff_tables"]
    cia=(state["h2_h2_cia"],state["h2_he_cia"],state["h2o_n2_cia"],state["h2o_h2o_cia"])
    nlev=atm.n_levels; nlay=atm.n_layers; nwn=sm.n_wavenumbers
    wn=np.asarray(sm.wavenumbers,float); dwn=sm.wavenumber_step
    sig_cgs=CST_SIGMA*1e3

    ABS=["CH4","CO","CO2","FeH","H2O","H2S","HCN","K","NH3","Na","PH3","TiO","VO"]
    P_lay_f=arr("/outputs/layers/pressure")
    same=(P_lay_f[0]>P_lay_f[-1])==(atm.pressures_layers[0]>atm.pressures_layers[-1])
    lay=lambda x: x if same else x[::-1]
    o=np.argsort(arr("/outputs/levels/pressure"))
    atm.temperatures[:]=np.interp(np.log(atm.pressures),
        np.log(arr("/outputs/levels/pressure")[o]),arr("/outputs/levels/temperature")[o])
    atm.temperatures_layers[:]=lay(arr("/outputs/layers/temperature"))
    em._calculate_altitude(atm,target,gmm,gv)
    for sp in ABS: gv[gas_id(sp),:]=lay(arr("/outputs/layers/volume_mixing_ratios/absorbers/"+sp))
    gv[gas_id("H2"),:]=lay(arr("/outputs/layers/volume_mixing_ratios/gases/H2"))
    gv[gas_id("He"),:]=lay(arr("/outputs/layers/volume_mixing_ratios/gases/He"))
    mu=lay(arr("/outputs/layers/mean_molar_mass")); mu=mu if np.median(mu)<0.1 else mu*1e-3
    atm.molar_masses_layers[:]=mu
    atm.scale_height[:]=CST_R*atm.temperatures_layers/(mu*atm.gravities_layers)

    # ---------- (1) Planck check ----------
    print("=== (1) Planck source-function check ===")
    for T_test in (500.0, 738.0):
        pl,_=_planck_array(wn, np.array([T_test]), 1, nwn)
        Bpy=pl[0]; Bcg=planck_cgs(wn,T_test)
        jpk=int(np.argmin(np.abs(wn-1.95*T_test)))
        intpiB=np.pi*np.sum(Bpy)*dwn
        print(f"\n  T={T_test:.0f} K:")
        print(f"    _planck_array @ {wn[jpk]:.0f} cm^-1 = {Bpy[jpk]:.4e}")
        print(f"    analytic CGS  @ {wn[jpk]:.0f} cm^-1 = {Bcg[jpk]:.4e}")
        print(f"    ratio python/CGS = {Bpy[jpk]/Bcg[jpk]:.4e}")
        print(f"    pi*integral(B_python)dnu = {intpiB:.4e}")
        print(f"      vs sigma_cgs*T^4 = {sig_cgs*T_test**4:.4e}  (ratio {intpiB/(sig_cgs*T_test**4):.4e})")
        print(f"      vs sigma_SI *T^4 = {CST_SIGMA*T_test**4:.4e}  (ratio {intpiB/(CST_SIGMA*T_test**4):.4e})")

    # ---------- (2) line / CIA isolation ----------
    def run(gv_in):
        tau=np.zeros((nlev,nwn,state["ng_max"])); tr=np.zeros((nlev,nwn))
        out=em._do_radiative_transfer(state,gv_in,cloud_vmr,0,atm,spec,cloud_obj,
            sm,light,kct,*cia,tau,tr)
        sr=out[6]; em_top=np.asarray(sr[-1,:],float)
        bol=float(np.sum(em_top)*dwn)
        return bol,(bol/sig_cgs)**0.25,em_top,out[0],out[1]

    print("\n=== (2) opacity isolation (Fortran VMRs) ===")
    bolF,TF,emF,tauF,trF=run(gv.copy())
    gv_cia=gv.copy()
    for sp in ABS: gv_cia[gas_id(sp),:]=0.0          # CIA + Rayleigh only
    bolC,TC,emC,_,_=run(gv_cia)
    gv_lin=gv.copy()
    gv_lin[gas_id("H2"),:]=0.0; gv_lin[gas_id("He"),:]=0.0   # lines + Rayleigh, no H2/He CIA
    bolL,TL,emL,_,_=run(gv_lin)
    print(f"  FULL       : bol={bolF:.4e}  T_eff(cgs)={TF:.2f} K")
    print(f"  CIA-only   : bol={bolC:.4e}  T_eff(cgs)={TC:.2f} K   (absorbers zeroed)")
    print(f"  LINES-only : bol={bolL:.4e}  T_eff(cgs)={TL:.2f} K   (H2/He zeroed -> no CIA)")
    print(f"  -> CIA/full = {bolC/bolF:.3f} ; lines/full = {bolL/bolF:.3f}")

    # column optical depth split (total vs rayleigh) for FULL
    print("\n=== (3) FULL column optical depth vs wavenumber ===")
    print("   wn[cm^-1]   tau_tot(min_g)  tau_tot(max_g)   tau_rayleigh")
    for j in range(0,nwn,max(1,nwn//14)):
        print(f"   {wn[j]:8.1f}   {tauF[0,j,:].min():.3e}    {tauF[0,j,:].max():.3e}    {trF[0,j]:.3e}")
    np.savez_compressed("planck_cia_dump.npz", wn=wn, emF=emF, emC=emC, emL=emL,
                        tauF=tauF, trF=trF)
    print("\nSaved planck_cia_dump.npz")

if __name__=="__main__":
    sys.exit(main())
