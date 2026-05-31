#!/usr/bin/env python3
"""inspect_ktable_grid.py -- read-only.  Prints the (P,T,wavenumber) grids of
the loaded k-tables next to the model's own pressure/temperature range, to
check unit consistency of the pressure axis used for k interpolation.

    python inspect_ktable_grid.py --nml inputs/example_no_irr.nml
"""
import argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--nml", required=True)
a = ap.parse_args()
from exorem import exorem_main as em
state = em._init_exorem(a.nml)
kct = state["kcoeff_tables"]; atm = state["atm"]

print("\n================ MODEL ranges ================")
print(f"  pressures_layers : {atm.pressures_layers.min():.4e} .. "
      f"{atm.pressures_layers.max():.4e}   (Pa, used as log(pj) for k interp)")
print(f"  temperatures_lay : {atm.temperatures_layers.min():.2f} .. "
      f"{atm.temperatures_layers.max():.2f}  K")

# find species-name list under whatever key exists
names = None
for k in ("species_names","absorbers_names","absorbers","gas_names","names"):
    if k in kct and len(kct[k]) == len(kct["p_k_species"]):
        names = list(kct[k]); break
if names is None:
    names = [f"sp{i}" for i in range(len(kct["p_k_species"]))]

print("\n================ K-TABLE grids ================")
print(f"{'sp':7s} {'n_p':>4s}  {'p_k min':>11s} {'p_k max':>11s}   "
      f"{'t_k min':>8s} {'t_k max':>8s}   {'wn min':>8s} {'wn max':>8s}")
for i, pk in enumerate(kct["p_k_species"]):
    pk = np.asarray(pk, float).ravel()
    tk = np.asarray(kct["t_k_species"][i], float).ravel()
    wk = np.asarray(kct["wavenumbers_k"][i], float).ravel()
    print(f"{names[i]:7s} {pk.size:4d}  {pk.min():11.3e} {pk.max():11.3e}   "
          f"{tk.min():8.1f} {tk.max():8.1f}   {wk.min():8.1f} {wk.max():8.1f}")

# verdict on the pressure axis
allp = np.concatenate([np.asarray(p,float).ravel() for p in kct["p_k_species"]])
pmin, pmax = allp.min(), allp.max()
print("\n================ VERDICT ================")
print(f"  k-table pressure axis spans {pmin:.3e} .. {pmax:.3e}")
if pmax < 1e3:
    unit = "BAR (petitRADTRANS-style)"; mism = "Pa vs bar -> ~1e5x; log(pj_Pa) clips to top of table"
elif pmax < 1e6:
    unit = "mbar (or bar reaching ~1e2-1e5)"; mism = "likely mbar -> 100x mismatch"
else:
    unit = "Pa"; mism = "consistent with model Pa -> no pressure-unit bug here"
print(f"  => looks like {unit}")
print(f"  => {mism}")
print(f"  model pj range is {atm.pressures_layers.min():.2e}..{atm.pressures_layers.max():.2e} Pa; "
      f"k interp uses log(pj) directly against this axis.")
