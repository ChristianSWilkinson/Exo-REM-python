"""
Robust Fortran-namelist parser to replace the broken one in
``exorem.interface.parse_input_file``.

The shipped parser splits the *body* of each ``&group ... /`` block on
``,\\s*(?=\\w)`` — which incorrectly splits inside comma-separated arrays
such as ``cloud_particle_radius = 50e-6, 5e-6``.  As a result, every key
after the first in each ``&group`` block ends up as part of the *value* of
the preceding key.

This module re-parses the file by:
  1. Stripping ``!`` comments.
  2. Splitting on newlines (each Fortran namelist assignment is one line).
  3. Detecting array literals by counting top-level commas after the ``=``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def parse_input_file(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse a Fortran-style namelist file into a nested dict."""
    text = Path(path).read_text()

    # 1. Strip Fortran-style comments (! to end-of-line) BEFORE any block matching.
    #    Comments can legally contain '/' which would otherwise terminate the block.
    text = re.sub(r"!.*", "", text)

    groups: dict[str, dict[str, Any]] = {}
    # 2. Walk line-by-line, building blocks delimited by &group and the next '/'
    #    that is NOT inside a quoted string.  Comma-separated arrays often span
    #    multiple lines after the '=' — accumulate continuation lines until we
    #    hit either the next ``key =`` or the block-terminator ``/``.
    current_group: str | None = None
    current_body: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"&(\w+)\b", line)
        if m:
            current_group = m.group(1).lower()
            current_body  = []
            continue
        if current_group is None:
            continue
        # End-of-block detector: '/' that is NOT inside quotes
        if _has_top_level_slash(line):
            groups[current_group] = _parse_body(current_body)
            current_group = None
            current_body  = []
            continue
        current_body.append(line)

    # If a block is unterminated, still parse it
    if current_group is not None:
        groups[current_group] = _parse_body(current_body)

    return groups


def _has_top_level_slash(line: str) -> bool:
    """True if *line* contains a '/' outside any quoted substring."""
    in_str = None
    for ch in line:
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch == "/":
            return True
    return False


def _parse_body(lines: list[str]) -> dict[str, Any]:
    """Parse the lines of a namelist block into ``{key: value, …}``.

    Continuation lines (no '=' yet, but the previous line ended on a trailing
    comma) are appended to the value of the most-recent key.
    """
    pairs: dict[str, Any] = {}
    last_key: str | None = None
    last_raw: list[str] = []

    def _commit() -> None:
        nonlocal last_key, last_raw
        if last_key is not None:
            pairs[last_key] = _parse_value(" ".join(last_raw))
        last_key = None
        last_raw = []

    for line in lines:
        if "=" in line:
            _commit()
            key, _, raw_value = line.partition("=")
            last_key = key.strip().lower()
            last_raw = [raw_value.strip()]
        else:
            # Continuation of the previous value
            if last_key is not None:
                last_raw.append(line.strip())

    _commit()
    return pairs


def _parse_value(raw: str) -> Any:
    """Convert a Fortran scalar or array literal to a Python value."""
    raw = raw.strip().rstrip(",")
    if raw == "":
        return ""

    # Lists: top-level commas (we approximate "top-level" by ignoring commas
    # that are inside a pair of single or double quotes).
    if _contains_top_level_comma(raw):
        tokens = _split_top_level_commas(raw)
        return [_parse_scalar(t) for t in tokens if t.strip()]

    # Space-separated array: 'val1 val2 val3' (rare but legal in Fortran).
    # Only treat as array if there are no quotes and at least one space
    # separating two tokens, AND the tokens look like scalars.
    if "'" not in raw and '"' not in raw:
        parts = raw.split()
        if len(parts) > 1 and all(_looks_scalar(p) for p in parts):
            return [_parse_scalar(p) for p in parts]

    return _parse_scalar(raw)


def _looks_scalar(s: str) -> bool:
    """True if *s* looks like a number, bool, or single bare token."""
    try:
        float(s.replace("d", "e").replace("D", "e"))
        return True
    except ValueError:
        pass
    return s.lower() in (".true.", ".false.", "t", "f", "true", "false")


def _contains_top_level_comma(s: str) -> bool:
    in_str = None
    for ch in s:
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            continue
        if ch == ",":
            return True
    return False


def _split_top_level_commas(s: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    in_str = None
    for ch in s:
        if in_str:
            buf.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
            buf.append(ch)
            continue
        if ch == ",":
            tokens.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        tokens.append("".join(buf).strip())
    return tokens


def _parse_scalar(s: str) -> Any:
    s = s.strip().strip("'\"").strip()
    if not s:
        return ""
    if s.lower() in (".true.", "t", "true"):
        return True
    if s.lower() in (".false.", "f", "false"):
        return False
    # Fortran double-precision exponent: 1.5d-3 → 1.5e-3
    s_e = re.sub(r"[dD]([+-]?\d)", r"e\1", s)
    try:
        return int(s_e)
    except ValueError:
        pass
    try:
        return float(s_e)
    except ValueError:
        pass
    return s


def patch_parser() -> None:
    """Replace ``exorem.interface.parse_input_file`` with this module's version."""
    from . import interface as _iface
    _iface.parse_input_file = parse_input_file


__all__ = ["parse_input_file", "patch_parser"]
