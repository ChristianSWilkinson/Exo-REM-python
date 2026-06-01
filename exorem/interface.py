"""
I/O interface for Exorem.

Reads configuration files (Fortran-namelist style), loads k-coefficient HDF5
tables, stellar spectra, CIA tables, and writes output files.

Mirrors the Fortran ``interface`` and ``exorem_interface`` modules
(interface.f90, exorem_interface.f90).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .physics import PI
from .math_utils import interp, convolve
from .objects import (
    Atmosphere, LightSource, Species, Spectrometrics,
    Cloud, Target, ExoremRetrieval, Thermodynamics,
    FILE_NAME_SIZE,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
EXOREM_VERSION = "Python-port 1.0"

# ---------------------------------------------------------------------------
# Input-file parser (Fortran namelist-like)
# ---------------------------------------------------------------------------


def parse_input_file(path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Parse a Fortran-style namelist input file.

    Delegates to :mod:`exorem.nml_parser`, which correctly handles
    comma-separated arrays, slashes inside path strings, and Fortran
    comments — the failure modes of the original regex-based parser.
    """
    from .nml_parser import parse_input_file as _nml_parse
    return _nml_parse(path)


# ---------------------------------------------------------------------------
# ExoREM input parameter reader
# ---------------------------------------------------------------------------


def read_exorem_input_parameters(
    input_file: str | Path,
    atm:        Atmosphere,
    target:     Target,
    light:      LightSource,
    spec:       Species,
    spectrometrics: Spectrometrics,
    cloud_obj:  Cloud,
    retrieval:  ExoremRetrieval,
) -> dict[str, Any]:
    """
    Read all Exorem parameters from *input_file* and populate the data objects.

    Returns a dict of miscellaneous scalar options that do not map cleanly to
    a single object (paths, flags, etc.).
    """
    print(f"Reading parameters in file '{input_file}'")
    cfg = parse_input_file(input_file)

    # ---- output files ----
    out = cfg.get("output_files", {})
    opts = {
        "spectrum_file_prefix":           out.get("spectrum_file_prefix", "spectrum"),
        "temperature_profile_file_prefix":out.get("temperature_profile_file_prefix", "tp"),
        "vmr_file_prefix":                out.get("vmr_file_prefix", "vmr"),
        "output_files_suffix":            out.get("output_files_suffix", ""),
    }

    # ---- target ----
    tp = cfg.get("target_parameters", {})
    target.target_mass               = float(tp.get("target_mass", 0.0))
    target.target_equatorial_gravity = float(tp.get("target_equatorial_gravity", 0.0))
    target.target_equatorial_radius  = float(tp.get("target_equatorial_radius", 0.0))
    target.target_polar_radius       = float(tp.get("target_polar_radius", 0.0))
    target.target_flattening         = float(tp.get("target_flattening", 0.0))
    target.target_radius             = float(tp.get("target_equatorial_radius", 0.0))
    target.latitude                  = float(tp.get("latitude", 0.0))
    target.target_internal_temperature = float(tp.get("target_internal_temperature", 0.0))
    target.emission_angle            = float(tp.get("emission_angle", 0.0))
    target.cos_average_angle         = float(tp.get("cos_average_angle", 2.0/3.0))

    # ---- gravity resolution -------------------------------------------------
    # The namelist exposes three overlapping ways to specify gravity:
    #   1. ``target_gravity``               — a plain user-friendly value
    #   2. ``target_equatorial_gravity``    — legacy name
    #   3. ``target_mass`` + ``target_equatorial_radius`` — derived via G·M/R²
    # and a switch ``use_gravity`` that's supposed to pick between (1)/(2) and
    # (3) but used to be silently ignored.  We now honour it explicitly:
    #
    #   use_gravity = True  → use whichever of target_gravity or
    #                         target_equatorial_gravity is set (priority to
    #                         target_gravity if both are set and positive).
    #   use_gravity = False → derive from G·M/R², ignoring the explicit values.
    #
    # If ``use_gravity`` is absent we infer the intent: a positive explicit
    # gravity field means "use it", otherwise fall back to G·M/R².
    g_explicit = float(tp.get("target_gravity", 0.0)) \
                 or target.target_equatorial_gravity
    g_derived  = 0.0
    if target.target_mass > 0.0 and target.target_radius > 0.0:
        from .physics import CST_G
        g_derived = CST_G * target.target_mass / target.target_radius ** 2

    if "use_gravity" in tp:
        if bool(tp["use_gravity"]):
            target.target_gravity = g_explicit if g_explicit > 0 else g_derived
        else:
            target.target_gravity = g_derived if g_derived > 0 else g_explicit
    else:
        target.target_gravity = g_explicit if g_explicit > 0 else g_derived

    if target.target_gravity <= 0.0:
        raise ValueError(
            "Could not determine target_gravity from the namelist.  Set "
            "either (target_gravity OR target_equatorial_gravity) > 0, OR "
            "both target_mass > 0 and target_equatorial_radius > 0.")

    # Diagnostic line so the user can confirm which gravity actually got used.
    src = []
    if g_explicit > 0: src.append(f"explicit={g_explicit:.3f}")
    if g_derived  > 0: src.append(f"G·M/R²={g_derived:.3f}")
    print(f"  target_gravity resolved to {target.target_gravity:.3f} m/s²  "
          f"(candidates: {', '.join(src) if src else 'none'})")

    # ---- light source ----
    lp = cfg.get("light_source_parameters", {})
    light.radius                 = float(lp.get("light_source_radius", 0.0))
    light.range                  = float(lp.get("light_source_range", 0.0))
    light.effective_temperature  = float(lp.get("light_source_effective_temperature", 0.0))
    light.irradiation            = float(lp.get("light_source_irradiation", 0.0))
    light.incidence_angle        = float(lp.get("incidence_angle", 0.0))
    opts["add_light_source"]     = bool(lp.get("add_light_source", False))
    opts["use_irradiation"]      = bool(lp.get("use_irradiation", False))
    opts["use_light_source_spectrum"]  = bool(lp.get("use_light_source_spectrum", False))
    opts["light_source_spectrum_file"] = str(lp.get("light_source_spectrum_file", "None"))
    # When the user opts out of a real stellar spectrum, force the filename
    # to "None" so build_stellar_irradiance falls back to the Planck branch.
    if not opts["use_light_source_spectrum"]:
        opts["light_source_spectrum_file"] = "None"

    # ---- atmosphere ----
    ap = cfg.get("atmosphere_parameters", {})
    atm.n_levels           = int(ap.get("n_levels", 50))
    atm.n_layers           = atm.n_levels - 1
    atm.h2_vmr             = float(ap.get("h2_vmr", 0.85))
    atm.he_vmr             = float(ap.get("he_vmr", 0.15))
    atm.z_vmr              = float(ap.get("z_vmr", 0.0))
    atm.metallicity        = float(ap.get("metallicity", 1.0))
    opts["use_metallicity"]= bool(ap.get("use_metallicity", True))
    atm.pressure_min       = float(ap.get("pressure_min", 1e-6))
    atm.pressure_max       = float(ap.get("pressure_max", 1e3))
    atm.eddy_mode          = str(ap.get("eddy_mode", "Ackerman"))
    opts["n_species"]      = int(ap.get("n_species", 0))
    opts["n_clouds"]       = int(ap.get("n_clouds", 0))
    opts["n_cia"]          = int(ap.get("n_cia", 0))
    opts["eddy_diffusion_coefficient"] = ap.get("eddy_diffusion_coefficient", [1e6])
    opts["load_kzz_profile"]           = bool(ap.get("load_kzz_profile", False))
    opts["use_pressure_grid"]          = bool(ap.get("use_pressure_grid", False))

    # ---- species ----
    sp = cfg.get("species_parameters", {})
    opts["species_names"]       = _to_list(sp.get("species_names", []))
    opts["species_at_equilibrium"] = _to_list(sp.get("species_at_equilibrium", []))
    opts["cia_names"]           = _to_list(sp.get("cia_names", []))
    opts["elements_names"]      = _to_list(sp.get("elements_names", []))
    opts["elements_h_ratio"]    = _to_list(sp.get("elements_h_ratio", []))
    opts["elements_metallicity"]= _to_list(sp.get("elements_metallicity", []))
    opts["use_atmospheric_metallicity"] = bool(sp.get("use_atmospheric_metallicity", False))
    opts["use_elements_metallicity"]    = bool(sp.get("use_elements_metallicity", True))
    opts["load_vmr_profiles"]   = bool(sp.get("load_vmr_profiles", False))
    opts["vmr_profiles_file"]   = str(sp.get("vmr_profiles_file", "None"))
    opts["use_chemistry"]       = bool(sp.get("use_chemistry", True))
    opts["use_rayleigh"]        = bool(sp.get("use_rayleigh", True))

    # ---- spectrum ----
    sm = cfg.get("spectrum_parameters", {})
    spectrometrics.wavenumber_min  = float(sm.get("wavenumber_min", 1000.0))
    spectrometrics.wavenumber_max  = float(sm.get("wavenumber_max", 10000.0))
    spectrometrics.wavenumber_step = float(sm.get("wavenumber_step", 1.0))

    # ---- clouds ----
    cp = cfg.get("clouds_parameters", {})
    cloud_obj.cloud_mode     = str(cp.get("cloud_mode", "fixedRadius"))
    cloud_obj.cloud_fraction = float(cp.get("cloud_fraction", 1.0))
    opts["cloud_names"]      = _to_list(cp.get("cloud_names", []))
    opts["cloud_particle_radius"]     = _to_list(cp.get("cloud_particle_radius", []))
    opts["sedimentation_parameter"]   = _to_list(cp.get("sedimentation_parameter", []))
    opts["supersaturation_parameter"] = _to_list(cp.get("supersaturation_parameter", []))
    opts["sticking_efficiency"]       = _to_list(cp.get("sticking_efficiency", []))
    opts["cloud_particle_density"]    = _to_list(cp.get("cloud_particle_density", []))
    opts["reference_wavenumber"]      = _to_list(cp.get("reference_wavenumber", []))
    opts["load_cloud_profiles"]       = bool(cp.get("load_cloud_profiles", False))

    # ---- retrieval ----
    rp = cfg.get("retrieval_parameters", {})
    retrieval.n_iterations               = int(rp.get("n_iterations", 20))
    retrieval.n_non_adiabatic_iterations = int(rp.get("n_non_adiabatic_iterations", 5))
    retrieval.n_burn_iterations          = int(rp.get("n_burn_iterations", 0))
    retrieval.chemistry_iteration_interval = int(rp.get("chemistry_iteration_interval", 1))
    retrieval.cloud_iteration_interval   = int(rp.get("cloud_iteration_interval", 1))
    retrieval.retrieval_level_top        = int(rp.get("retrieval_level_top", 1))
    retrieval.retrieval_level_bottom     = int(rp.get("retrieval_level_bottom", atm.n_levels))
    # The example.nml writes bottom=2, top=81 (numerically the opposite of
    # what exorem_main expects: n_retrieved = bottom - top + 1).  Swap when
    # needed so the downstream math comes out positive.
    if retrieval.retrieval_level_bottom < retrieval.retrieval_level_top:
        retrieval.retrieval_level_top, retrieval.retrieval_level_bottom = (
            retrieval.retrieval_level_bottom, retrieval.retrieval_level_top)
    retrieval.retrieval_flux_error_top   = float(rp.get("retrieval_flux_error_top", 1e-2))
    retrieval.retrieval_flux_error_bottom= float(rp.get("retrieval_flux_error_bottom", 1e-2))
    retrieval.retrieval_tolerance        = float(rp.get("retrieval_tolerance", 1e-3))
    retrieval.smoothing_top              = float(rp.get("smoothing_top", 1.0))
    retrieval.smoothing_bottom           = float(rp.get("smoothing_bottom", 1.0))
    retrieval.weight_apriori             = float(rp.get("weight_apriori", 1.0))
    opts["temperature_profile_file"]     = str(rp.get("temperature_profile_file", "None"))
    # Optional Guillot-apriori controls (only used if
    # temperature_profile_file = "guillot").  Reasonable defaults for an
    # H₂/He giant-planet / brown-dwarf atmosphere; override in the .nml
    # if a different opacity/composition is needed.
    if "guillot_T_irr"        in rp: opts["guillot_T_irr"]        = float(rp["guillot_T_irr"])
    if "guillot_kappa_ir"     in rp: opts["guillot_kappa_ir"]     = float(rp["guillot_kappa_ir"])
    if "guillot_p_photo_bar"  in rp: opts["guillot_p_photo_bar"]  = float(rp["guillot_p_photo_bar"])
    if "guillot_gamma_v"      in rp: opts["guillot_gamma_v"]      = float(rp["guillot_gamma_v"])
    if "guillot_grad_ad"      in rp: opts["guillot_grad_ad"]      = float(rp["guillot_grad_ad"])
    if "guillot_gamma_v"  in rp: opts["guillot_gamma_v"]  = float(rp["guillot_gamma_v"])
    if "guillot_grad_ad"  in rp: opts["guillot_grad_ad"]  = float(rp["guillot_grad_ad"])

    # ---- options ----
    op = cfg.get("options", {})
    opts["output_transmission_spectra"]           = bool(op.get("output_transmission_spectra", False))
    opts["output_species_spectral_contributions"] = bool(op.get("output_species_spectral_contributions", False))
    opts["output_cia_spectral_contribution"]      = bool(op.get("output_cia_spectral_contribution", False))
    opts["output_thermal_spectral_contribution"]  = bool(op.get("output_thermal_spectral_contribution", False))
    opts["output_fluxes"]                         = bool(op.get("output_fluxes", True))
    opts["output_hdf5"]                           = bool(op.get("output_hdf5", False))
    opts["output_full"]                           = bool(op.get("output_full", False))

    # ---- paths ----
    pp = cfg.get("paths", {})
    opts["path_data"]                  = str(pp.get("path_data", "./data"))
    opts["path_cia"]                   = str(pp.get("path_cia", "./data/cia"))
    opts["path_clouds"]                = str(pp.get("path_clouds", "./data/clouds"))
    opts["path_k_coefficients"]        = str(pp.get("path_k_coefficients", "./data/k_coefficients"))
    opts["path_temperature_profile"]   = str(pp.get("path_temperature_profile", "./data"))
    opts["path_thermochemical_tables"] = str(pp.get("path_thermochemical_tables", "./data/thermochemistry"))
    opts["path_vmr_profiles"]          = str(pp.get("path_vmr_profiles", "./data"))
    opts["path_light_source_spectra"]  = str(pp.get("path_light_source_spectra", "./data"))
    opts["path_outputs"]               = str(pp.get("path_outputs", "./outputs"))

    # Resolve relative paths against the directory of the namelist file
    # (so 'path_data = ../data/' works no matter where Python is launched).
    nml_dir = Path(input_file).resolve().parent
    for k in ("path_data", "path_cia", "path_clouds", "path_k_coefficients",
              "path_temperature_profile", "path_thermochemical_tables",
              "path_vmr_profiles", "path_light_source_spectra", "path_outputs"):
        p = Path(opts[k])
        if not p.is_absolute():
            opts[k] = str((nml_dir / p).resolve())

    # Resolve the stellar-spectrum file too (if we actually need it)
    sp_file = opts.get("light_source_spectrum_file", "None")
    if (opts.get("use_light_source_spectrum")
            and sp_file and sp_file.lower() != "none"
            and not Path(sp_file).is_absolute()):
        opts["light_source_spectrum_file"] = str(
            Path(opts["path_light_source_spectra"]) / sp_file)

    # Ensure the output directory exists
    Path(opts["path_outputs"]).mkdir(parents=True, exist_ok=True)

    return opts


def _to_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if v is None or v == "":
        return []
    return [v]


# ---------------------------------------------------------------------------
# CIA table loading
# ---------------------------------------------------------------------------


def read_cia_file(
    file_spec: str,
    wavenumbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a HITRAN-format CIA file.

    Parameters
    ----------
    file_spec   : path to the CIA data file
    wavenumbers : target wavenumber grid (cm⁻¹)

    Returns
    -------
    cia           : (n_wavenumbers, n_temperatures) CIA cross-section cm⁵ molecule⁻²
    temperatures  : (n_temperatures,)  K
    wavenumbers_k : (n_wavenumbers_file,) original wavenumber grid
    """
    data_blocks: list[tuple[float, np.ndarray, np.ndarray]] = []

    with open(file_spec) as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Header line: "Pair  T  N_pts  wn_lo  wn_hi  ..."
        parts = line.split()
        if len(parts) < 3:
            i += 1
            continue
        try:
            t_ref = float(parts[1])
            n_pts = int(parts[2])
        except (ValueError, IndexError):
            i += 1
            continue

        wn  = np.empty(n_pts)
        cia = np.empty(n_pts)
        for k in range(n_pts):
            i += 1
            if i >= len(lines):
                break
            row = lines[i].split()
            wn[k]  = float(row[0])
            cia[k] = float(row[1])

        data_blocks.append((t_ref, wn, cia))
        i += 1

    if not data_blocks:
        raise IOError(f"No CIA data found in {file_spec}")

    temperatures = np.array([b[0] for b in data_blocks])
    wn0, cia0 = data_blocks[0][1], data_blocks[0][2]

    cia_interp = np.zeros((len(wavenumbers), len(temperatures)))
    for j, (_, wn, c) in enumerate(data_blocks):
        cia_interp[:, j] = np.interp(wavenumbers, wn, c, left=0.0, right=0.0)

    return cia_interp, temperatures, wn0


def read_data_file(
    file_path: str | Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Read a two-column ASCII data file with a commented header.

    The header format expected is::

        # col1_label  col2_label ... ! optional comment
        # unit1       unit2 ...

    Returns
    -------
    columns : (n_columns, n_rows)  numerical data
    labels  : column labels
    units   : column units
    """
    path = Path(file_path)
    labels: list[str] = []
    units:  list[str] = []
    rows: list[list[float]] = []

    with open(path) as fh:
        header_lines = 0
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                header_lines += 1
                content = line.lstrip("#").split("!")[0].strip()
                if header_lines == 1:
                    labels = content.split()
                elif header_lines == 2:
                    units  = content.split()
                continue
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                continue

    if not rows:
        return np.empty((0, 0)), labels, units

    data = np.array(rows).T    # (n_columns, n_rows)
    return data, labels, units


# ---------------------------------------------------------------------------
# Temperature profile reader (HDF5)
# ---------------------------------------------------------------------------


def load_temperature_profile(
    file_path: str | Path,
    key_pressure:    str = "pressure",
    key_temperature: str = "temperature",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a pressure–temperature profile.

    Supports two file layouts:

    1. **ASCII** (e.g. ``temperature_profile_example_ref.dat``):
       Two header lines starting with ``#`` (column names then units),
       then one row per level with whitespace-separated values.  Column 0
       is pressure (Pa) and column 1 is temperature (K) by convention.

    2. **HDF5**:  Datasets named ``pressure`` and ``temperature`` (or the
       names supplied via *key_pressure* / *key_temperature*).

    The format is auto-detected from the file extension.

    Returns
    -------
    pressure    : (n_levels,)  Pa, monotonically increasing (top → bottom)
    temperature : (n_levels,)  K
    """
    p = Path(file_path)
    suffix = p.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        try:
            import h5py  # type: ignore
        except ImportError:
            raise ImportError("h5py is required to read HDF5 temperature profiles.")
        with h5py.File(p, "r") as fh:
            pressure    = np.asarray(fh[key_pressure])
            temperature = np.asarray(fh[key_temperature])
    else:
        # ASCII / whitespace-separated columns
        rows = []
        with open(p) as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("!"):
                    continue
                # Replace Fortran double-precision exponent before parsing
                s_e = s.replace("D", "E").replace("d", "e")
                cols = s_e.split()
                try:
                    rows.append((float(cols[0]), float(cols[1])))
                except (IndexError, ValueError):
                    continue
        if not rows:
            raise ValueError(f"No numeric rows found in {p}")
        arr = np.array(rows)
        pressure, temperature = arr[:, 0], arr[:, 1]

    # Ensure ascending pressure (some files may be ordered bottom → top)
    if pressure[0] > pressure[-1]:
        pressure = pressure[::-1]
        temperature = temperature[::-1]

    return pressure, temperature


# ---------------------------------------------------------------------------
# Stellar spectrum builder
# ---------------------------------------------------------------------------


def build_stellar_irradiance(
    wavenumbers: np.ndarray,
    wavenumber_step: float,
    light: LightSource,
    spectrum_file: str = "None",
) -> np.ndarray:
    """
    Build the stellar irradiance array on the model wavenumber grid.

    If *spectrum_file* is ``'None'`` (default), a scaled Planck function is
    used.  Otherwise the file is read and re-normalised to match the Planck
    integral.

    Returns
    -------
    irradiance : (n_wavenumbers,)  erg s⁻¹ cm⁻² sr⁻¹ / cm⁻¹
    """
    from .physics import planck_function, spherical_black_body_spectral_radiance

    n_wn = len(wavenumbers)

    # --- Black-body irradiance ---
    irradiance = np.array([
        spherical_black_body_spectral_radiance(
            wavenumbers[i], light.effective_temperature,
            light.radius, light.range, 0.25)
        * 1e3   # W m⁻² sr⁻¹/cm⁻¹ → erg s⁻¹ cm⁻² sr⁻¹/cm⁻¹
        for i in range(n_wn)
    ])

    if spectrum_file.strip().lower() == "none":
        return irradiance

    # --- Read stellar spectrum file ---
    print(f"Reading stellar spectrum in file '{spectrum_file}'")
    data, labels, units = read_data_file(spectrum_file)

    if "wavelength" not in labels:
        raise ValueError("Label 'wavelength' not found in stellar spectrum file.")
    if "spectral_radiosity" not in labels:
        raise ValueError("Label 'spectral_radiosity' not found in stellar spectrum file.")

    i_wvl = labels.index("wavelength")
    i_rad = labels.index("spectral_radiosity")

    if units[i_wvl] != "angstrom":
        raise ValueError("Wavelengths must be in angstrom.")
    if units[i_rad] != "erg.s-1.cm-2.a-1":
        raise ValueError("Radiosities must be in erg.s-1.cm-2.a-1.")

    wl_ang = data[i_wvl, ::-1].copy()      # ascending wavelength (Å)
    rad    = data[i_rad, ::-1].copy()

    # Convert Å → cm⁻¹
    wl_ang = np.maximum(wl_ang, np.finfo(float).tiny)
    wn_file = 1e8 / wl_ang                  # Å → cm⁻¹  (1 Å = 1e-8 cm)

    # Convert erg s⁻¹ cm⁻² Å⁻¹ → erg s⁻¹ cm⁻² sr⁻¹ / cm⁻¹
    # F_nu = F_lambda * lambda² / c  (in consistent units)
    rad_wavenumber = rad[::-1] * 1e8 * (wl_ang[::-1] * 1e-8)**2 / PI

    # Fine grid for convolution
    interp_step = wavenumber_step / 20.0
    n_fine = int((wavenumbers[-1] - wavenumbers[0]) / interp_step) + 40 + 1
    wn_fine = wavenumbers[0] + np.arange(n_fine) * interp_step

    rad_fine = np.interp(wn_fine, wn_file[::-1], rad_wavenumber[::-1],
                         left=0.0, right=0.0)
    rad_fine *= (light.radius / light.range)**2 * 0.25

    # Box-car average down to model grid
    n_filter = max(1, int(round(wavenumber_step / interp_step)))
    box = np.ones(n_filter) / n_filter
    rad_conv = convolve(rad_fine, box)
    irradiance_file = rad_conv[:len(wn_fine):n_filter][:n_wn]

    # Renormalise so total power matches the Planck integral
    sum_bb    = irradiance.sum()
    sum_file  = irradiance_file.sum()
    ratio = sum_file / sum_bb if sum_bb > 0 else 1.0
    print(f"  Stellar / black body irradiance = {ratio:.3f}")
    print("  Compensating...")
    if sum_file > 0:
        irradiance_file = irradiance_file * sum_bb / sum_file

    return irradiance_file


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_spectrum(
    path: str | Path,
    wavenumbers: np.ndarray,
    spectral_radiosity: np.ndarray,
    header: str = "",
) -> None:
    """
    Write the emergent spectrum to a plain-text file.

    Columns: wavenumber (cm⁻¹), spectral radiosity (erg s⁻¹ cm⁻² sr⁻¹ / cm⁻¹).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as fh:
        if header:
            for line in header.splitlines():
                fh.write(f"# {line}\n")
        fh.write("# wavenumber  spectral_radiosity\n")
        fh.write("# cm-1        erg.s-1.cm-2.sr-1/cm-1\n")
        for wn, sr in zip(wavenumbers, spectral_radiosity):
            fh.write(f"{wn:.6f}  {sr:.6e}\n")

    print(f"Spectrum written to '{path}'")


def write_temperature_profile(
    path: str | Path,
    pressures: np.ndarray,
    temperatures: np.ndarray,
) -> None:
    """Write a pressure–temperature profile to a plain-text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as fh:
        fh.write("# pressure   temperature\n")
        fh.write("# Pa         K\n")
        for p, t in zip(pressures, temperatures):
            fh.write(f"{p:.6e}  {t:.4f}\n")

    print(f"Temperature profile written to '{path}'")


def write_vmr_profile(
    path: str | Path,
    pressures_layers: np.ndarray,
    species_names: list[str],
    species_vmr_layers: np.ndarray,  # (n_layers, n_species)
) -> None:
    """Write VMR profiles to a plain-text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as fh:
        header = "# pressure  " + "  ".join(species_names) + "\n"
        fh.write(header)
        fh.write("# Pa\n")
        for j, p in enumerate(pressures_layers):
            row = f"{p:.6e}  " + "  ".join(f"{species_vmr_layers[j, i]:.6e}"
                                            for i in range(len(species_names)))
            fh.write(row + "\n")

    print(f"VMR profiles written to '{path}'")


def write_hdf5_output(
    path: str | Path,
    data: dict,
) -> None:
    """
    Write all outputs to a single HDF5 file.

    Parameters
    ----------
    path : output file path (will be created / overwritten)
    data : dict mapping dataset names to numpy arrays or scalars
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        raise ImportError("h5py is required for HDF5 output.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as fh:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                fh.create_dataset(key, data=value, compression="gzip")
            elif isinstance(value, str):
                fh.attrs[key] = value
            else:
                fh.attrs[key] = value

    print(f"HDF5 output written to '{path}'")


def write_hdf5_output_fortran(path, payload: dict) -> None:
    """
    Write the model results in the *original Fortran ExoREM* HDF5 layout.

    Group tree (mirrors ``exorem.f90 :: write_hdf5_output``)::

        /outputs/run_quality/{radiosity_actual_target_ratio,
                              actual_internal_temperature, chi2_retrieval,
                              delta_temperature_layers}
        /outputs/levels/{pressure, temperature, radiosity_internal, altitude,
                         kernel_temperature, is_convective,
                         gradiant_temperature, radiosity_error}
        /outputs/layers/{pressure, temperature, gravity, molar_mass,
                         mean_molar_mass}
        /outputs/layers/volume_mixing_ratios/absorbers/<species>
        /outputs/layers/volume_mixing_ratios/gases/<gas>
        /outputs/spectra/wavenumber
        /outputs/spectra/emission/spectral_radiosity
        /outputs/spectra/transmission/spectral_radius        (if computed)
        /model_parameters/target/{mass, internal_temperature, radius_1e5Pa}
        /model_parameters/species/elemental_abundances/<element>
        /model_parameters/species/solar_elemental_abundances/<element>

    Every dataset carries a ``units`` attribute identical to the Fortran's, and
    is written only when its source is present and non-empty in *payload*, so a
    partial state still yields a valid file.

    Units note: the Fortran writes ``pressures*1e2``, ``gravities*1e-2``,
    ``z*1e3``, ``molar_mass*1e-3`` because its internal arrays are
    mbar/cgs/km/(g mol-1).  The Python model state is already SI
    (Pa / m s-2 / m / kg mol-1), so the caller passes those arrays unscaled.
    The one quantity still in CGS internally is the radiosity, so the caller
    passes ``radiosity * 1e-3`` to obtain W m-2.
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        raise ImportError("h5py is required for HDF5 output.")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def dset(grp, name, value, units):
        """Create one dataset with a units attribute, skipping empty/None."""
        if value is None:
            return
        arr = np.asarray(value)
        if arr.size == 0:
            return
        d = grp.create_dataset(name, data=arr)
        d.attrs["units"] = units

    def dmap(grp, mapping, units):
        """Write each {name: array} pair of *mapping* as a dataset."""
        if not mapping:
            return
        for nm, val in mapping.items():
            dset(grp, str(nm), val, units)

    g = payload.get
    with h5py.File(p, "w") as f:
        out = f.create_group("outputs")

        # ---- run quality -------------------------------------------------
        rq = out.create_group("run_quality")
        dset(rq, "radiosity_actual_target_ratio", g("radiosity_target_ratio"), "None")
        dset(rq, "actual_internal_temperature", g("actual_internal_temperature"), "K")
        dset(rq, "chi2_retrieval", g("chi2_retrieval"), "None")
        dset(rq, "delta_temperature_layers", g("delta_temperature_layers"), "K")

        # ---- levels ------------------------------------------------------
        lv = out.create_group("levels")
        dset(lv, "pressure", g("levels_pressure_Pa"), "Pa")
        dset(lv, "temperature", g("levels_temperature_K"), "K")
        dset(lv, "radiosity_internal", g("levels_radiosity_W_m2"), "W.m2")
        dset(lv, "altitude", g("levels_altitude_m"), "m")
        dset(lv, "kernel_temperature", g("kernel_temperature"), "K")
        dset(lv, "is_convective", g("is_convective"), "None")
        dset(lv, "gradiant_temperature", g("gradiant_temperature"), "None")
        dset(lv, "radiosity_error", g("levels_radiosity_error_W_m2"), "W.m-2")
        # convective flux (diagnostics) — matches the Fortran levels dataset name
        dset(lv, "radiosity_convective", g("levels_radiosity_convective_W_m2"), "W.m2")

        # ---- layers ------------------------------------------------------
        ly = out.create_group("layers")
        dset(ly, "pressure", g("layers_pressure_Pa"), "Pa")
        dset(ly, "temperature", g("layers_temperature_K"), "K")
        dset(ly, "gravity", g("layers_gravity_m_s2"), "m.s-2")
        dset(ly, "molar_mass", g("layers_molar_mass_kg_mol"), "kg.mol-1")
        dset(ly, "mean_molar_mass", g("mean_molar_mass_kg_mol"), "kg.mol-1")
        # Kzz + scale height so the harness can reconstruct the quench timescales
        # (tmix = scale_height_cm**2 / Kzz) and verify the CO/CH4 quench level.
        # `/outputs/layers/eddy_diffusion_coefficient` matches the key ExoremOut.kzz
        # already reads from the Fortran reference output.
        dset(ly, "eddy_diffusion_coefficient",
             g("layers_eddy_diffusion_coefficient_cm2_s"), "cm2.s-1")
        dset(ly, "scale_height", g("layers_scale_height_m"), "m")
        # molar heat capacity (diagnostics) — matches the Fortran layers dataset
        dset(ly, "isobaric_molar_heat_capacity",
             g("layers_isobaric_molar_heat_capacity_J_K_mol"), "J.K-1.mol-1")

        vmr = ly.create_group("volume_mixing_ratios")
        dmap(vmr.create_group("absorbers"), g("absorbers_vmr") or {}, "None")
        dmap(vmr.create_group("gases"), g("gases_vmr") or {}, "None")

        # ---- spectra -----------------------------------------------------
        sp = out.create_group("spectra")
        dset(sp, "wavenumber", g("wavenumber_cm1"), "cm-1")
        em = sp.create_group("emission")
        dset(em, "spectral_radiosity",
             g("emission_spectral_radiosity_W_m2_cm1"), "W.m-2/cm-1")
        if g("transmission_spectral_radius_m") is not None:
            tr = sp.create_group("transmission")
            dset(tr, "spectral_radius", g("transmission_spectral_radius_m"), "m")

        # ---- model parameters -------------------------------------------
        mp = f.create_group("model_parameters")
        tg = mp.create_group("target")
        dset(tg, "mass", g("target_mass_kg"), "kg")
        dset(tg, "internal_temperature", g("target_internal_temperature_K"), "K")
        dset(tg, "radius_1e5Pa", g("target_radius_m"), "m")

        spg = mp.create_group("species")
        dmap(spg.create_group("elemental_abundances"),
             g("elemental_abundances") or {}, "None")
        dmap(spg.create_group("solar_elemental_abundances"),
             g("solar_elemental_abundances") or {}, "None")

    print(f"HDF5 output written to '{p}'")
