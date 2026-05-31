"""
Robust replacement loaders matching the real Exorem data-directory layout.

Why this module exists
----------------------
The placeholder loaders inside ``exorem_main.py`` look for filenames that
do *not* match what the real Exorem data distribution ships:

    they expect              real filename
    --------------           --------------------------------
    {sp}.h5                  {sp}.ktable.exorem.h5
    {sp}.cia                 {sp}.cia.txt
    thermochemistry.h5       gases/<sp>.tct.dat + condensates/<sp>.tct.dat

These functions handle the actual files, falling back gracefully when
something is missing.  :func:`patch_loaders` rewires the three functions
inside ``exorem_main`` at runtime — call it once before :func:`run_exorem`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .chemistry import (
    GASES_NAMES, CONDENSATE_NAMES, N_GASES, N_CONDENSATES, gas_id, condensate_id,
)
from .interface import read_data_file
from .physics import CST_R


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_HDF5_KEY_CANDIDATES = {
    "wavenumbers":   ("wavenumbers", "wavenumber", "wno", "wn", "nu",
                      "bin_centers", "bin_edges"),
    "pressures":     ("pressures", "pressure", "p", "p_grid"),
    "temperatures":  ("temperatures", "temperature", "t", "t_grid"),
    "kcoeff":        ("kcoeff", "k_coefficients", "kcoefficients",
                      "k", "xsec", "cross_section", "kdata"),
    "weights":       ("weights", "gauss_weights", "w_gauss"),
    "samples":       ("samples", "g_points", "g"),
}


def _read_h5_with_fallback(h5file, key_set: str):
    """Read the first dataset present from a tuple of candidate names."""
    for name in _HDF5_KEY_CANDIDATES[key_set]:
        if name in h5file:
            return np.asarray(h5file[name])
        # Some Exorem tables nest under groups — recurse one level
        for grp_name in h5file:
            grp = h5file[grp_name]
            if hasattr(grp, "keys") and name in grp:
                return np.asarray(grp[name])
    return None


def _glob_first(directory: Path, patterns: Iterable[str]) -> Path | None:
    """Return the first file in *directory* matching any of *patterns*."""
    for pat in patterns:
        matches = sorted(directory.glob(pat))
        if matches:
            return matches[0]
    return None


def _infer_kcoeff_axes(shape: tuple[int, ...], n_p: int, n_t: int,
                        n_wn: int, n_g: int) -> tuple[int, int, int, int] | None:
    """
    Given a 4-D kcoeff shape and the sizes of (p, t, wn, g) grids, return the
    permutation ``(pos_p, pos_t, pos_wn, pos_g)`` such that
    ``shape[pos_p] == n_p`` etc.  Returns ``None`` if ambiguous.

    Handles the common cases:
    - petitRADTRANS  : (n_p, n_t, n_wn, n_g)
    - Exo-Mol native : (n_wn, n_p, n_t, n_g)
    - Other          : try by elimination
    """
    targets = {"p": n_p, "t": n_t, "wn": n_wn, "g": n_g}
    # Build all permutations consistent with the shape
    assignment: dict[str, int] = {}
    remaining_axes = list(range(4))
    for name in ("g", "p", "t", "wn"):  # try unique sizes first
        v = targets[name]
        candidates = [ax for ax in remaining_axes if shape[ax] == v]
        if len(candidates) == 1:
            assignment[name] = candidates[0]
            remaining_axes.remove(candidates[0])
    if len(assignment) == 4:
        return (assignment["p"], assignment["t"], assignment["wn"], assignment["g"])

    # Still ambiguous (multiple axes have the same size).
    # Fall back to petitRADTRANS convention: (n_p, n_t, n_wn, n_g)
    if shape == (n_p, n_t, n_wn, n_g):
        return (0, 1, 2, 3)
    return None


# ===========================================================================
# K-coefficient loader
# ===========================================================================

def load_k_coefficients(path: str | os.PathLike,
                        species_names: list[str],
                        spectrometrics) -> dict:
    """Read one HDF5 k-table per species into the format ``exorem_main`` expects."""
    try:
        import h5py
    except ImportError:
        print("Warning: h5py not installed; using zero k-tables.")
        return _placeholder_k_tables(species_names, spectrometrics)

    directory = Path(path)
    if not directory.exists():
        print(f"Warning: k-coefficient directory '{directory}' does not exist.")
        return _placeholder_k_tables(species_names, spectrometrics)

    tables: dict = {
        "species": list(species_names),
        "n_k_wavenumbers":  [], "n_k_pressures": [], "n_k_temperatures": [],
        "p_k_species":      [], "t_k_species":   [], "wavenumbers_k":   [],
        "weights_k":        None,   "samples_k":  None,
        "kcoeff_species":   [],     "ng":         [],
    }

    for sp in species_names:
        fpath = _glob_first(directory, (
            f"{sp}.h5",
            f"{sp}.ktable.exorem.h5",
            f"{sp}.ktable.*.h5",
            f"{sp}.*.h5",
            f"{sp}*.h5",
        ))
        if fpath is None:
            print(f"  ! no k-table file found for species '{sp}' in {directory}")
            tables["n_k_wavenumbers"].append(spectrometrics.n_wavenumbers)
            tables["n_k_pressures"].append(1)
            tables["n_k_temperatures"].append(1)
            tables["p_k_species"].append(np.array([1e5]))
            tables["t_k_species"].append(np.array([[1000.0]]))
            tables["wavenumbers_k"].append(spectrometrics.wavenumbers)
            tables["kcoeff_species"].append(
                np.zeros((1, spectrometrics.n_wavenumbers, 1, 1)))
            tables["ng"].append(1)
            continue

        print(f"  + loading k-table for {sp:6s}: {fpath.name}")
        with h5py.File(fpath, "r") as fh:
            wn_k = _read_h5_with_fallback(fh, "wavenumbers")
            # If we got bin_edges instead of centers, take midpoints
            if wn_k is not None and "bin_centers" not in fh and "bin_edges" in fh:
                edges = np.asarray(fh["bin_edges"])
                wn_k = 0.5 * (edges[:-1] + edges[1:])
            p_k  = _read_h5_with_fallback(fh, "pressures")
            t_k  = _read_h5_with_fallback(fh, "temperatures")
            k_k  = _read_h5_with_fallback(fh, "kcoeff")
            wt_k = _read_h5_with_fallback(fh, "weights")
            s_k  = _read_h5_with_fallback(fh, "samples")

        if wn_k is None or k_k is None:
            print(f"    ! could not find wavenumbers/kcoeff datasets in {fpath.name}; "
                  "available keys: " + ", ".join(list_h5_keys(fpath)))
            continue

        if p_k is None: p_k = np.array([1e5])
        if t_k is None: t_k = np.array([1000.0])
        if wt_k is None: wt_k = np.array([1.0])
        if s_k  is None: s_k  = np.array([0.5])

        # --- Normalise axis order ---------------------------------------
        # petitRADTRANS k-tables are stored as (n_p, n_t, n_wn, n_g).
        # exorem_main expects     kcoeff[ng, n_wn, n_t, n_p].
        # Detect the file ordering by matching sizes, then transpose.
        n_p, n_t, n_wn, n_g = p_k.size, t_k.size, wn_k.size, wt_k.size
        if k_k.ndim == 4:
            axis_map = _infer_kcoeff_axes(k_k.shape, n_p, n_t, n_wn, n_g)
            if axis_map is not None:
                # axis_map gives (pos_of_n_p, pos_of_n_t, pos_of_n_wn, pos_of_n_g)
                pos_p, pos_t, pos_wn, pos_g = axis_map
                k_k = np.transpose(k_k, (pos_g, pos_wn, pos_t, pos_p))
            else:
                print(f"    ! cannot infer axis order of kcoeff{tuple(k_k.shape)} "
                      f"given n_p={n_p} n_t={n_t} n_wn={n_wn} n_g={n_g}")

        # Make sure t_k is (n_t, n_p): broadcast a 1-D vector if needed
        if t_k.ndim == 1:
            t_k = np.broadcast_to(t_k[:, None], (t_k.size, n_p)).copy()
        elif t_k.shape == (n_p, n_t):       # stored as (n_p, n_t)
            t_k = t_k.T.copy()

        tables["n_k_wavenumbers"].append(len(wn_k))
        tables["n_k_pressures"].append(len(p_k))
        tables["n_k_temperatures"].append(
            t_k.shape[0] if t_k.ndim > 1 else len(t_k))
        tables["p_k_species"].append(p_k)
        tables["t_k_species"].append(t_k)
        tables["wavenumbers_k"].append(wn_k)
        tables["kcoeff_species"].append(k_k)
        tables["ng"].append(len(wt_k))

        if tables["weights_k"] is None:
            tables["weights_k"] = wt_k
            tables["samples_k"] = s_k

    if tables["weights_k"] is None:
        tables["weights_k"] = np.array([1.0])
        tables["samples_k"] = np.array([0.5])

    tables["n_k_wavenumbers"]  = np.array(tables["n_k_wavenumbers"],  dtype=int)
    tables["n_k_pressures"]    = np.array(tables["n_k_pressures"],    dtype=int)
    tables["n_k_temperatures"] = np.array(tables["n_k_temperatures"], dtype=int)
    tables["ng"]               = np.array(tables["ng"],               dtype=int)
    tables["ng_max"]           = int(tables["ng"].max()) if tables["ng"].size else 1
    return tables


def list_h5_keys(path: Path) -> list[str]:
    """Return all dataset paths inside an HDF5 file (1 level deep)."""
    import h5py
    keys: list[str] = []
    with h5py.File(path, "r") as fh:
        def visit(name, obj):
            if hasattr(obj, "dtype"):
                keys.append(name)
        fh.visititems(visit)
    return keys


def _placeholder_k_tables(species_names, spectrometrics):
    n_wn, n_sp = spectrometrics.n_wavenumbers, len(species_names)
    return {
        "wavenumbers_k":  [spectrometrics.wavenumbers] * n_sp,
        "n_k_wavenumbers": np.full(n_sp, n_wn, dtype=int),
        "n_k_pressures":   np.ones(n_sp, dtype=int),
        "n_k_temperatures":np.ones(n_sp, dtype=int),
        "p_k_species":     [np.array([1e5])] * n_sp,
        "t_k_species":     [np.array([[1000.0]])] * n_sp,
        "weights_k":       np.array([1.0]),
        "samples_k":       np.array([0.5]),
        "kcoeff_species":  [np.zeros((1, n_wn, 1, 1))] * n_sp,
        "ng":              np.ones(n_sp, dtype=int),
        "ng_max":          1,
    }


# ===========================================================================
# CIA loader
# ===========================================================================

def load_cia(opts: dict, wavenumbers: np.ndarray, atm
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load collision-induced absorption tables.

    The Exorem-native CIA text format is:
       line 1                : ``<n_temperatures> <n_wavenumbers>``
       lines 2 .. n_T+1      : one temperature per line  (K)
       lines n_T+2 ..        : ``<wavenumber> <σ(T1)> <σ(T2)> … <σ(T_nT)>``

    Returns four ``(n_layers, n_wavenumbers)`` arrays interpolated onto the
    model temperature profile and wavenumber grid.
    """
    n_layers, n_wn = atm.n_layers, len(wavenumbers)
    h2h2   = np.zeros((n_layers, n_wn))
    h2he   = np.zeros((n_layers, n_wn))
    h2on2  = np.zeros((n_layers, n_wn))
    h2oh2o = np.zeros((n_layers, n_wn))
    targets = {
        "H2-H2":   h2h2,
        "H2-He":   h2he,
        "H2O-N2":  h2on2,
        "H2O-H2O": h2oh2o,
    }

    cia_dir = Path(opts.get("path_cia", "./data/cia"))
    if not cia_dir.exists():
        print(f"Warning: CIA directory '{cia_dir}' not found.")
        return h2h2, h2he, h2on2, h2oh2o

    for name in opts.get("cia_names", []):
        fpath = _glob_first(cia_dir, (
            f"{name}.cia.txt", f"{name}.cia",
            f"{name}.txt", f"{name}.*",
        ))
        if fpath is None:
            print(f"  ! CIA file not found for {name} in {cia_dir}")
            continue
        print(f"  + loading CIA {name:9s}: {fpath.name}")

        try:
            temps_file, wn_file, sigma_file = _read_exorem_cia(fpath)
        except Exception as err:
            print(f"    ! could not parse {fpath.name} as Exorem CIA: {err}")
            continue

        if name not in targets:
            print(f"    ! {name} is not one of {list(targets)} — skipping")
            continue

        # Interpolate σ(wn, T) onto the model (wavenumbers, layer-T) grid
        target_arr = targets[name]
        # First interpolate in wavenumber for every file temperature
        sigma_on_grid = np.zeros((temps_file.size, n_wn))
        for it in range(temps_file.size):
            sigma_on_grid[it, :] = np.interp(
                wavenumbers, wn_file, sigma_file[:, it], left=0.0, right=0.0)
        # Then interpolate in temperature per wavenumber
        T_layer = np.asarray(atm.temperatures_layers)
        for iw in range(n_wn):
            target_arr[:, iw] = np.interp(
                T_layer, temps_file, sigma_on_grid[:, iw],
                left=sigma_on_grid[0, iw], right=sigma_on_grid[-1, iw])

    return h2h2, h2he, h2on2, h2oh2o


def _read_exorem_cia(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse an Exorem-native CIA text file.

    Returns
    -------
    temperatures : (n_T,)            in K
    wavenumbers  : (n_wn,)           in cm⁻¹
    sigma        : (n_wn, n_T)       cross-section per molecule pair
    """
    tokens: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.extend(line.split())

    if len(tokens) < 2:
        raise ValueError(f"file is empty or malformed: {path}")
    n_T  = int(tokens[0])
    n_wn = int(tokens[1])
    expected = 2 + n_T + n_wn * (1 + n_T)
    if len(tokens) < expected:
        raise ValueError(
            f"expected {expected} tokens (header + {n_T} temps + {n_wn}×{1+n_T} rows), "
            f"got {len(tokens)}")

    temperatures = np.array([float(t) for t in tokens[2:2 + n_T]])
    rest = np.array([float(t) for t in tokens[2 + n_T:expected]],
                    dtype=float).reshape(n_wn, 1 + n_T)
    wavenumbers = rest[:, 0]
    sigma       = rest[:, 1:]
    return temperatures, wavenumbers, sigma


# ===========================================================================
# Thermochemical table loader
# ===========================================================================

def load_thermochemical_tables(path: str | os.PathLike) -> dict:
    """
    Walk ``path/gases/*.tct.dat`` and ``path/condensates/*.tct.dat`` and
    assemble (N_GASES, n_T) / (N_CONDENSATES, n_T) tables of Gibbs free
    energy of formation and isobaric molar heat capacity.

    Returns a dict with keys ``temperatures``, ``gases_delta_g``,
    ``condensates_delta_g``, ``gases_c_p`` — matching what
    ``exorem_main`` consumes.
    """
    base = Path(path)
    gases_dir       = base / "gases"
    condensates_dir = base / "condensates"

    if not gases_dir.exists() or not condensates_dir.exists():
        print(f"Warning: thermochemical directories missing under {base}.")
        n_T = 100
        T   = np.linspace(100.0, 5000.0, n_T)
        return {
            "temperatures":        T,
            "gases_delta_g":       np.zeros((N_GASES, n_T)),
            "condensates_delta_g": np.zeros((N_CONDENSATES, n_T)),
            "gases_c_p":           np.full((N_GASES, n_T), 2.5 * CST_R),
        }

    # First pass: read every available file, remember temperature grids
    raw_gas: dict[str, dict[str, np.ndarray]] = {}
    raw_con: dict[str, dict[str, np.ndarray]] = {}

    for sp in GASES_NAMES:
        f = gases_dir / f"{sp}.tct.dat"
        if f.exists():
            raw_gas[sp] = _read_tct(f)

    for sp in CONDENSATE_NAMES:
        f = condensates_dir / f"{sp}.tct.dat"
        if f.exists():
            raw_con[sp] = _read_tct(f)

    if not raw_gas and not raw_con:
        print(f"Warning: no .tct.dat files found under {base}.")
        n_T = 100
        T   = np.linspace(100.0, 5000.0, n_T)
        return {
            "temperatures":        T,
            "gases_delta_g":       np.zeros((N_GASES, n_T)),
            "condensates_delta_g": np.zeros((N_CONDENSATES, n_T)),
            "gases_c_p":           np.full((N_GASES, n_T), 2.5 * CST_R),
        }

    # Use the union of all temperature grids (ascending, unique)
    all_T: list[float] = []
    for rec in list(raw_gas.values()) + list(raw_con.values()):
        all_T.extend(rec["T"].tolist())
    if not all_T:
        print(f"Warning: every .tct.dat file in {base} is empty; "
              "falling back to a default T grid.")
        n_T = 100
        T_common = np.linspace(100.0, 5000.0, n_T)
    else:
        T_common = np.array(sorted(set(np.round(all_T, 6))), dtype=float)
        n_T = T_common.size

    gases_delta_g       = np.zeros((N_GASES,       n_T))
    condensates_delta_g = np.zeros((N_CONDENSATES, n_T))
    gases_c_p           = np.full((N_GASES, n_T), 2.5 * CST_R)

    for sp, rec in raw_gas.items():
        i = gas_id(sp)
        if i < 0 or rec["T"].size == 0:
            continue
        gases_delta_g[i, :] = np.interp(T_common, rec["T"], rec["dG"])
        if rec["cp"] is not None and rec["cp"].size > 0:
            gases_c_p[i, :] = np.interp(T_common, rec["T"], rec["cp"])

    for sp, rec in raw_con.items():
        i = condensate_id(sp)
        if i < 0 or rec["T"].size == 0:
            continue
        condensates_delta_g[i, :] = np.interp(T_common, rec["T"], rec["dG"])

    print(f"  loaded {len(raw_gas)} gas tables and "
          f"{len(raw_con)} condensate tables ({n_T} T-points)")

    return {
        "temperatures":        T_common,
        "gases_delta_g":       gases_delta_g,
        "condensates_delta_g": condensates_delta_g,
        "gases_c_p":           gases_c_p,
    }


def _read_tct(file_path: Path) -> dict[str, np.ndarray | None]:
    """Read a single .tct.dat file and return {'T', 'dG', 'cp'} arrays."""
    data, labels, _units = read_data_file(file_path)
    if data.size == 0:
        return {"T": np.array([]), "dG": np.array([]), "cp": None}

    # Locate columns by label, case-insensitive prefix match
    lower = [l.lower() for l in labels]

    def col(prefixes):
        for pre in prefixes:
            for i, l in enumerate(lower):
                if l.startswith(pre):
                    return data[i]
        return None

    T  = col(("temperature",))
    dG = col(("gibbs_free_energy", "delta_g", "gibbs"))
    cp = col(("isobaric_molar_heat", "c_p", "cp"))

    if T is None or dG is None:
        # File doesn't follow the expected layout; assume first two columns
        if data.shape[0] >= 2:
            T = data[0]
            dG = data[-1]   # gibbs is usually last column
            cp = data[1] if data.shape[0] >= 3 else None

    return {"T": T, "dG": dG, "cp": cp}


# ===========================================================================
# Solar abundance table
# ===========================================================================

def load_solar_abundances(path: str | os.PathLike) -> np.ndarray:
    """
    Read ``solar_abundances.dat`` and return a length-N_ELEMENTS array of
    *linear* X/H atomic ratios.

    File format (Lodders 2020, also used by the Fortran Exorem code)::

        # atomic_number elemental_abundance elemental_abundance_uncertainty  ! ...
        # None None None
        1 12.00 0.00
        2 10.924 0.02
        3 3.27 0.03
        ...

    The middle column is the *log₁₀ A(X)* value in the standard
    astronomical convention with ``A(H) = 12``.  We convert to a linear
    ratio relative to hydrogen::

        X/H  =  10^(log_A(X) - 12)

    so that, e.g., ``solar_xh[0] == 1.0`` (H/H) and
    ``solar_xh[7] ≈ 5.37e-4`` (O/H, Lodders 2020).

    Parameters
    ----------
    path : path to ``solar_abundances.dat`` (or the directory containing it).

    Returns
    -------
    np.ndarray of shape (N_ELEMENTS,) — zeros for atomic numbers not present
    in the file (e.g. Z=43 Tc is intentionally absent from the Lodders 2020
    table).
    """
    from .physics import N_ELEMENTS

    p = Path(path)
    if p.is_dir():
        p = p / "solar_abundances.dat"
    if not p.exists():
        print(f"  Warning: solar_abundances.dat not found at {p} "
              "— elemental_h_ratio will be 0 for unlisted elements.")
        return np.zeros(N_ELEMENTS)

    ratios = np.zeros(N_ELEMENTS)
    with open(p) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            # Strip any inline '!' comment
            line = line.split("!", 1)[0].strip()
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                Z = int(parts[0])
                log_A = float(parts[1])
            except ValueError:
                continue
            if 1 <= Z <= N_ELEMENTS:
                ratios[Z - 1] = 10.0 ** (log_A - 12.0)
    return ratios


# ===========================================================================
# Monkey-patch entry point
# ===========================================================================

def patch_loaders() -> None:
    """
    Rewire the three placeholder loaders inside :mod:`exorem.exorem_main`
    to use the robust versions defined here.

    Call this once before :func:`exorem.run_exorem`.
    """
    from . import exorem_main as _em

    _em._load_k_coefficients      = load_k_coefficients
    _em._load_cia                 = load_cia
    _em._load_thermochemical_tables = load_thermochemical_tables
    print("Loaders patched: K-tables, CIA, thermochemistry now read the real data files.")


__all__ = [
    "load_k_coefficients",
    "load_cia",
    "load_thermochemical_tables",
    "load_solar_abundances",
    "patch_loaders",
    "list_h5_keys",
]
