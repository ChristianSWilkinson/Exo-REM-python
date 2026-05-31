"""
Single-shot entry point for Exorem.

All the bug-fix / wiring monkey-patches that used to live here have been
folded directly into the relevant source files.  This runner now does just
three things:

    1.  Optional pre-run audit of the data directory tree referenced by the
        namelist (use ``--no-audit`` to skip, ``--audit-only`` to dry-run).
    2.  Hand the .nml off to :func:`exorem.exorem_main.run_exorem`.
    3.  Surface tracebacks cleanly when something goes wrong.

Usage
-----
    python -m exorem.runner inputs/example.nml
    python -m exorem.runner --no-audit inputs/example.nml
    python -m exorem.runner --audit-only inputs/example.nml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import exorem_main
from .data_audit import audit_input

RUNNER_VERSION = "2025-05-25-r37"


def run(nml_file: str | Path,
        *, audit: bool = True, audit_only: bool = False,
        strict: bool = False) -> dict | None:
    """
    Run Exorem on *nml_file*.

    Parameters
    ----------
    nml_file    : path to the input namelist
    audit       : run the data-layout audit before kicking off
    audit_only  : run only the audit, then return ``None``
    strict      : abort if the audit reports any missing files
    """
    print(f"exorem.runner version {RUNNER_VERSION}")
    nml_path = Path(nml_file).resolve()
    if not nml_path.exists():
        raise FileNotFoundError(nml_path)

    if audit or audit_only:
        print("=" * 70)
        print(" Data layout audit")
        print("=" * 70)
        rep = audit_input(nml_path)
        print(rep.format())
        if audit_only:
            return None
        if not rep.ok and strict:
            raise SystemExit(
                "Audit found missing files. Re-run without --strict to proceed anyway.")
        print()

    print("=" * 70)
    print(f" Running Exorem on {nml_path}")
    print("=" * 70)
    return exorem_main.run_exorem(str(nml_path))


def _cli() -> int:
    ap = argparse.ArgumentParser(prog="exorem.runner")
    ap.add_argument("nml",                 help="path to the input namelist")
    ap.add_argument("--no-audit",  action="store_true",
                    help="skip the pre-run data-layout audit")
    ap.add_argument("--audit-only", action="store_true",
                    help="run the audit and exit without running Exorem")
    ap.add_argument("--strict",    action="store_true",
                    help="abort if any data files are missing")
    args = ap.parse_args()

    try:
        run(args.nml,
            audit=not args.no_audit,
            audit_only=args.audit_only,
            strict=args.strict)
    except SystemExit:
        raise
    except Exception as err:
        print(f"\nError: {err}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
