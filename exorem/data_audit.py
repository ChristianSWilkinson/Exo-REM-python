"""
Validate that the data files referenced by an Exorem ``.nml`` exist on disk.

Run before :func:`exorem.run_exorem` so missing files surface as one clean
report rather than as silent zero-fallbacks deep inside the loaders.

Path resolution
---------------
Paths inside the namelist are resolved **relative to the .nml file itself**,
not the current working directory.  This matches the convention of running
``python -m exorem.runner inputs/example.nml`` from the project root.

Usage
-----
    python -m exorem.data_audit path/to/example.nml
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from .nml_parser import parse_input_file


@dataclass
class AuditReport:
    found:   list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes:   list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    def format(self) -> str:
        lines: list[str] = []
        if self.found:
            lines.append(f"FOUND ({len(self.found)}):")
            lines.extend(f"  OK   {p}" for p in self.found)
        if self.notes:
            lines.append("\nNOTES:")
            lines.extend(f"  ..   {n}" for n in self.notes)
        if self.missing:
            lines.append(f"\nMISSING ({len(self.missing)}):")
            lines.extend(f"  !!   {p}" for p in self.missing)
        return "\n".join(lines)


def _resolve(base: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base / p).resolve()


def _glob_exists(directory: Path, patterns: tuple[str, ...]) -> Path | None:
    for pat in patterns:
        m = sorted(directory.glob(pat))
        if m:
            return m[0]
    return None


def audit_input(nml_path: str | Path) -> AuditReport:
    """Audit the data layout referenced by an Exorem namelist."""
    rep = AuditReport()
    nml_path = Path(nml_path).resolve()
    cfg = parse_input_file(nml_path)
    base = nml_path.parent

    paths  = cfg.get("paths", {})
    sp     = cfg.get("species_parameters", {})
    cp     = cfg.get("clouds_parameters", {})
    lp     = cfg.get("light_source_parameters", {})
    rp     = cfg.get("retrieval_parameters", {})

    def _check_dir(key: str) -> Path | None:
        raw = paths.get(key)
        if raw is None:
            return None
        path = _resolve(base, str(raw))
        if path.exists():
            rep.found.append(f"DIR  {key:25s} {path}")
            return path
        rep.missing.append(f"DIR  {key:25s} {path}")
        return path  # may still be useful for output dirs

    # directories
    _check_dir("path_data")
    cia_dir    = _check_dir("path_cia")
    cloud_dir  = _check_dir("path_clouds")
    k_dir      = _check_dir("path_k_coefficients")
    tprof_dir  = _check_dir("path_temperature_profile")
    thermo_dir = _check_dir("path_thermochemical_tables")
    vmr_dir    = _check_dir("path_vmr_profiles")
    star_dir   = _check_dir("path_light_source_spectra")

    out_raw = paths.get("path_outputs")
    if out_raw:
        out = _resolve(base, str(out_raw))
        if out.exists():
            rep.found.append(f"DIR  path_outputs              {out}")
        else:
            rep.notes.append(f"path_outputs '{out}' will be created if needed")

    # K-tables
    species_list = sp.get("species_names", []) or []
    if isinstance(species_list, str):
        species_list = [species_list]
    if k_dir and k_dir.exists():
        for s in species_list:
            f = _glob_exists(k_dir, (f"{s}.h5", f"{s}.ktable.exorem.h5",
                                       f"{s}.ktable.*.h5", f"{s}.*.h5"))
            if f:
                rep.found.append(f"K-table  {s:6s} -> {f.name}")
            else:
                rep.missing.append(f"K-table  {s:6s} in {k_dir}")

    # CIA
    cia_list = sp.get("cia_names", []) or []
    if isinstance(cia_list, str):
        cia_list = [cia_list]
    if cia_dir and cia_dir.exists():
        for c in cia_list:
            f = _glob_exists(cia_dir, (f"{c}.cia", f"{c}.cia.txt", f"{c}.txt", f"{c}.*"))
            if f:
                rep.found.append(f"CIA      {c:9s} -> {f.name}")
            else:
                rep.missing.append(f"CIA      {c:9s} in {cia_dir}")

    # Cloud optical constants
    cloud_list = cp.get("cloud_names", []) or []
    if isinstance(cloud_list, str):
        cloud_list = [cloud_list]
    if cloud_dir and cloud_dir.exists():
        for c in cloud_list:
            f = _glob_exists(cloud_dir, (f"{c}.ocst", f"{c}.ocst.txt", f"{c}.*"))
            if f:
                rep.found.append(f"cloud    {c:7s} -> {f.name}")
            else:
                rep.missing.append(f"cloud    {c:7s} in {cloud_dir}")

    # Thermochemical tables (per-species .tct.dat under gases/ and condensates/)
    if thermo_dir and thermo_dir.exists():
        gases_dir = thermo_dir / "gases"
        cond_dir  = thermo_dir / "condensates"
        if gases_dir.exists():
            n = len(list(gases_dir.glob("*.tct.dat")))
            rep.found.append(f"thermo    gases dir contains {n} .tct.dat files")
        else:
            rep.missing.append(f"thermo    {gases_dir}")
        if cond_dir.exists():
            n = len(list(cond_dir.glob("*.tct.dat")))
            rep.found.append(f"thermo    condensates dir contains {n} .tct.dat files")
        else:
            rep.missing.append(f"thermo    {cond_dir}")

    # Stellar spectrum (only if requested)
    sp_file = lp.get("light_source_spectrum_file", "None")
    use_star = lp.get("use_light_source_spectrum", False)
    if use_star and isinstance(sp_file, str) and sp_file.lower() not in ("none", ""):
        if star_dir:
            f = star_dir / sp_file
            if f.exists():
                rep.found.append(f"stellar  {f}")
            else:
                rep.missing.append(f"stellar  {f}")
    elif isinstance(sp_file, str) and sp_file.lower() not in ("none", ""):
        rep.notes.append(f"stellar spectrum '{sp_file}' not required "
                         "(use_light_source_spectrum=False)")

    # A-priori temperature profile
    tp_file = rp.get("temperature_profile_file", "None")
    if isinstance(tp_file, str) and tp_file.lower() not in ("none", "") and tprof_dir:
        f = tprof_dir / tp_file
        if f.exists():
            rep.found.append(f"T-P prof {f}")
        else:
            rep.missing.append(f"T-P prof {f}")

    # VMR profile (only if load_vmr_profiles=True)
    if sp.get("load_vmr_profiles", False):
        vmr_file = sp.get("vmr_profiles_file", "")
        if isinstance(vmr_file, str) and vmr_file and vmr_dir:
            f = vmr_dir / vmr_file
            if f.exists():
                rep.found.append(f"VMR prof {f}")
            else:
                rep.missing.append(f"VMR prof {f}")

    return rep


def _main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m exorem.data_audit <input.nml>", file=sys.stderr)
        return 2
    print(audit_input(sys.argv[1]).format())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
