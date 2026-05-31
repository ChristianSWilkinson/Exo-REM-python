"""
Backward-compatibility shim.

All runtime patches that used to live here have been folded directly into
the source files they were patching:

  - parser bug                → interface.parse_input_file (now delegates to nml_parser)
  - target gravity from M, R  → interface.read_exorem_input_parameters
  - retrieval level swap      → interface.read_exorem_input_parameters
  - path resolution           → interface.read_exorem_input_parameters
  - placeholder K/CIA/thermo  → exorem_main._load_* (delegate to loaders.py)
  - elements_in_gases plumbing → exorem_main._calculate_thermochemical_equilibrium (signature)
  - species_vmr_layers wiring → exorem_main._do_radiative_transfer (computed from gases_vmr)
  - rayleigh shape            → exorem_main._init_rayleigh_scattering (returns (N_GASES, n_wn))
  - print summary guard       → exorem_main._print_iteration_summary
  - initial H2/He VMR seeding → exorem_main._init_exorem
  - cloud / species wiring    → exorem_main._init_exorem
  - off-by-one fixes          → radiative_transfer.calculate_radiative_transfer +
                                 radiative_transfer.calculate_two_stream_fluxes
  - vectorised inner loop     → radiative_transfer.calculate_radiative_transfer
  - unit-mismatch fixes       → exorem_main._calculate_altitude,
                                 exorem_main._calculate_eddy_diffusion_coefficient,
                                 chemistry._calculate_time_constants
  - Numba JIT on hot kernels  → radiative_transfer (calculate_two_stream_fluxes,
                                 _combine_k_distributions, _planck_array, _up_down_fluxes)

This module is intentionally tiny and exists only so that older callers
(e.g. `from .bootstrap import wire_state`) keep working without changes.
"""

from __future__ import annotations


def wire_state(state: dict) -> None:
    """No-op: wiring is done inside ``_init_exorem`` now."""
    return


__all__ = ["wire_state"]
