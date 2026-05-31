"""
Offline synthetic verification of the silicate O-sink (_calculate_mg_si_o).

Uses realistic synthetic Delta-G (SiO the dominant Si carrier, H2O the dominant
O carrier, as in the real silicate-forming regime) and backs out condensate
Delta-G to hit a chosen supersaturation ratio R = product/k_eq.  Invariants
checked (must hold for any numbers):

  (C1) O conservation:   O_in_gas(VMR) == gas_element_abd[O] after  (osi ties them)
  (C2) Si conservation:  Si_in_gas(VMR) == gas_element_abd[Si] after
  (C3) Closed element balance:  dO == 2*dSi + dMg  exactly
  (C4) Finiteness & non-negativity of all VMRs and abundances
  (C5) Net O & Si removal >= 0 when supersaturated
  (C6) Equilibrium reached:  final product ~= k_eq  for the condensing phase
  (C7) Exact no-op when subsaturated (R<1)
  (C8) Fallback flag reproduces the simple stub
"""
import math
import numpy as np
import exorem.chemistry as C

NEL = C.N_ELEMENTS; NG = C.N_GASES
EIG = np.zeros((NG, NEL))
for i, n in enumerate(C.GASES_NAMES):
    EIG[i] = C.count_all_elements(n)

iH2=C.gas_id("H2"); iHe=C.gas_id("He"); iH2O=C.gas_id("H2O"); iCO=C.gas_id("CO")
iCO2=C.gas_id("CO2"); iCH4=C.gas_id("CH4"); iCH3=C.gas_id("CH3"); iSiO=C.gas_id("SiO")
iSiH4=C.gas_id("SiH4"); iMg=C.gas_id("Mg")
iO,iC,iSi,iMgE = 7,5,13,11
cFo=C.condensate_id("Mg2SiO4"); cEn=C.condensate_id("MgSiO3"); cSi=C.condensate_id("SiO2")

GAS_DG = {"H2O":-250., "SiO":-400., "SiH4":400., "CO":-100., "CH4":0., "CO2":0., "CH3":200.}
T=1800.0; p=5000.0; p_bar=p*1e-2          # p_bar = 50 bar
DUMMY = np.linspace(1e7, 1e3, 12)

def gdg_array():
    g = np.zeros(NG)
    for k,v in GAS_DG.items():
        g[C.gas_id(k)] = v
    return g

def base_vmr():
    v = np.full(NG, 1e-300)
    v[iH2]=0.85; v[iHe]=0.145; v[iH2O]=3e-3; v[iCO]=4e-4; v[iCO2]=1e-5
    v[iCH4]=1e-4; v[iCH3]=1e-9; v[iSiO]=5e-5; v[iSiH4]=1e-9; v[iMg]=4e-5
    return v

def keqs(vmr):
    q = vmr[iH2] if vmr[iH2]>0 else 1e-300
    g = gdg_array()
    kc  = C._keq(["H2O","CH4","CO","H2"],[-1,-1,1,3],g,p_bar,T,q)
    kc3 = math.sqrt(max(C._keq(["CH4","CH3","H2"],[-2,2,1],g,p_bar,T,q),0.0))
    kc2 = C._keq(["CO","H2O","CO2","H2"],[-1,-1,1,1],g,p_bar,T,q)
    ks  = C._keq(["SiH4","H2O","SiO","H2"],[-1,-1,1,3],g,p_bar,T,q)
    return kc,kc3,kc2,ks

def osi_consistent():
    """Hand-built vmr passed through osi once so the gas is self-consistent."""
    v = base_vmr(); gae = v@EIG
    kc,kc3,kc2,ks = keqs(v)
    C._osi(v, gae, EIG, gdg_array(), kc,kc3,kc2,ks, p_bar, T, False)
    return v

def k_target(phase, vmr, R):
    """Condensate Delta-G giving product/k_eq = R for the chosen phase."""
    q = vmr[iH2]
    if phase=="en":
        prod = vmr[iMg]*vmr[iSiO]*vmr[iH2O]**2
        k = prod/R
        dg = (C.CST_R*T/1e3)*math.log(k*p_bar**2/q**2) + 0.0 + GAS_DG["SiO"] + 2*GAS_DG["H2O"]
        return ("MgSiO3", dg, prod, k)
    if phase=="fo":
        prod = vmr[iMg]**2*vmr[iSiO]*vmr[iH2O]**3
        k = prod/R
        dg = (C.CST_R*T/1e3)*math.log(k*p_bar**3/q**3) + 0.0 + GAS_DG["SiO"] + 3*GAS_DG["H2O"]
        return ("Mg2SiO4", dg, prod, k)
    if phase=="si":
        prod = vmr[iSiO]*vmr[iH2O]
        k = prod/R
        dg = (C.CST_R*T/1e3)*math.log(k*p_bar/q) + GAS_DG["SiO"] + GAS_DG["H2O"]
        return ("SiO2", dg, prod, k)

def build_cdg(specs):
    cdg = np.full(C.N_CONDENSATES, +1e4)   # everything subsaturated by default
    for cname, dg in specs.items():
        cdg[C.condensate_id(cname)] = dg
    return cdg

def run(label, phase_ratios, expect_cond, check_equil_phase=None):
    vmr = osi_consistent()
    vmr0 = vmr.copy()
    gae0 = vmr@EIG
    specs = {}
    kmap = {}
    for ph, R in phase_ratios.items():
        cname, dg, prod, k = k_target(ph, vmr, R)
        specs[cname] = dg; kmap[ph] = k
    cdg = build_cdg(specs)

    is_cond = np.zeros(C.N_CONDENSATES, dtype=bool)
    vmr_sat = np.zeros(C.N_CONDENSATES)
    layer_cond = np.zeros(C.N_CONDENSATES, dtype=int)
    GAE = np.tile(gae0.reshape(-1,1), (1,12))

    out = C._calculate_mg_si_o(
        T, p, 5, vmr, gdg_array(), cdg, is_cond, vmr_sat, layer_cond,
        DUMMY, DUMMY, gas_element_abd=GAE, elements_in_gases=EIG,
        co_ch4_quench=False, co_co2_quench=False, qcoco2=0.0)

    gae_after = GAE[:,5]
    eg = out@EIG
    fails = []
    if not np.all(np.isfinite(out)): fails.append("non-finite VMR")
    if np.any(out < 0.0): fails.append("negative VMR")
    if not np.all(np.isfinite(gae_after)): fails.append("non-finite abundance")
    if np.any(gae_after[[iO,iSi,iMgE]] < 0.0): fails.append("negative O/Si/Mg abundance")

    triggered = bool(is_cond[cFo] or is_cond[cEn] or is_cond[cSi])
    if expect_cond:
        if not triggered: fails.append("expected condensation, none triggered")
        if abs(eg[iO]-gae_after[iO]) > 1e-9*max(abs(gae_after[iO]),1e-30):
            fails.append("C1 O gas/abd mismatch")
        if abs(eg[iSi]-gae_after[iSi]) > 1e-9*max(abs(gae_after[iSi]),1e-30):
            fails.append("C2 Si gas/abd mismatch")
        dO = gae0[iO]-gae_after[iO]; dSi = gae0[iSi]-gae_after[iSi]; dMg = gae0[iMgE]-eg[iMgE]
        if abs(dO-(2*dSi+dMg)) > 1e-9*max(abs(dO),1e-30):
            fails.append("C3 balance dO != 2dSi+dMg")
        if dO < -1e-18: fails.append(f"C5 O increased ({dO:.3e})")
        if dSi < -1e-18: fails.append(f"C5 Si increased ({dSi:.3e})")
        if check_equil_phase:
            if check_equil_phase=="en": prod=out[iMg]*out[iSiO]*out[iH2O]**2
            elif check_equil_phase=="fo": prod=out[iMg]**2*out[iSiO]*out[iH2O]**3
            else: prod=out[iSiO]*out[iH2O]
            k = kmap[check_equil_phase]
            rel = abs(prod-k)/max(k,1e-300)
            if rel > 1e-5:
                fails.append(f"C6 product/k_eq={prod/k:.6f} (not at equilibrium)")
            print(f"    equilibrium check[{check_equil_phase}]: product/k_eq = {prod/k:.8f}")
        print(f"    dO={dO:.4e} dSi={dSi:.4e} dMg={dMg:.4e}  Fo/En/Si={int(is_cond[cFo])}{int(is_cond[cEn])}{int(is_cond[cSi])}")
    else:
        if triggered: fails.append("subsaturated but condensation triggered")
        if not np.array_equal(out, vmr0): fails.append(f"VMR changed (max|d|={np.max(np.abs(out-vmr0)):.3e})")
        if not np.array_equal(gae_after, gae0): fails.append(f"abundance changed (max|d|={np.max(np.abs(gae_after-gae0)):.3e})")

    print(f"[{'PASS' if not fails else 'FAIL'}] {label}")
    for f in fails: print("        -", f)
    return not fails

print("="*72); print("Silicate O-sink synthetic verification"); print("="*72)
ok = True
ok &= run("enstatite supersaturated R=20",  {"en":20.0}, True, check_equil_phase="en")
ok &= run("enstatite strongly super R=500", {"en":500.0}, True, check_equil_phase="en")
ok &= run("forsterite supersaturated R=20", {"fo":20.0}, True, check_equil_phase="fo")
ok &= run("silica supersaturated R=20",     {"si":20.0}, True, check_equil_phase="si")
ok &= run("forsterite+enstatite both super",{"fo":15.0,"en":30.0}, True)
ok &= run("all three supersaturated",       {"fo":15.0,"en":30.0,"si":8.0}, True)
ok &= run("subsaturated en R=0.1 (no-op)",  {"en":0.1}, False)
ok &= run("all subsaturated R=0.05 (no-op)",{"fo":0.05,"en":0.05,"si":0.05}, False)

# (C8) fallback flag reproduces the simple stub
def fallback_test():
    C._USE_MG_SI_O_SINK = False
    try:
        vmr = osi_consistent(); v1 = vmr.copy()
        cdg = build_cdg({"MgSiO3": k_target("en", vmr, 20.0)[1]})
        is_cond=np.zeros(C.N_CONDENSATES,bool); vsat=np.zeros(C.N_CONDENSATES); lc=np.zeros(C.N_CONDENSATES,int)
        GAE=np.tile((vmr@EIG).reshape(-1,1),(1,12))
        # full-signature call but flag off -> must delegate to _simple (ignores gae)
        out_full = C._calculate_mg_si_o(T,p,5,v1.copy(),gdg_array(),cdg,is_cond.copy(),vsat.copy(),lc.copy(),
                                        DUMMY,DUMMY,gas_element_abd=GAE.copy(),elements_in_gases=EIG,
                                        co_ch4_quench=False,co_co2_quench=False,qcoco2=0.0)
        out_simple = C._calculate_mg_si_o_simple(T,p,5,v1.copy(),gdg_array(),cdg,is_cond.copy(),vsat.copy(),lc.copy(),DUMMY,DUMMY)
        same = np.allclose(out_full,out_simple,rtol=0,atol=0)
        print(f"[{'PASS' if same else 'FAIL'}] fallback flag reproduces simple stub (identical={same})")
        return same
    finally:
        C._USE_MG_SI_O_SINK = True
ok &= fallback_test()

print("="*72); print("OVERALL:", "ALL PASS" if ok else "FAILURES PRESENT"); print("="*72)
