"""
Optical properties — refractive indices of common atmospheric gases and
Rayleigh scattering cross-sections.

Mirrors the Fortran ``optics`` module (optics.f90).
All wavenumbers are in cm-1 and wavelengths in micrometres unless stated.

Sources for refractive-index formulas (each documented inline):
    Ar  : Borzsonyi et al. 2008
    CH4 : Loria et al. 1909
    CO  : Sneep et al. 2004
    CO2 : Bideau-Mehu et al. 1973
    H2  : Peck et al. 1977
    H2O : IAPWS (T = 373.15 K, 1 atm)
    He  : Ermolov et al. 2015
    Kr  : Borzsonyi et al. 2008
    N2  : Borzsonyi et al. 2008
    Ne  : Borzsonyi et al. 2008
    NH3 : Cuthbertson et al. 1914
    Xe  : Borzsonyi et al. 2008

For species not in the table a constant 1.0003 is returned.

Rayleigh cross-section follows Bodhaine et al. 1999, assuming a depolarisation
factor of zero (single-gas, isotropic-molecule approximation).
"""

from __future__ import annotations

import math
from typing import Callable

from .physics import CST_N0, PI

DEFAULT_REFRACTIVE_INDEX: float = 1.0003


# ===========================================================================
# Per-species refractive index formulas
# ===========================================================================

def _wavelength_um(wavenumber: float, lo: float, hi: float | None = None) -> float:
    """Convert wavenumber (cm-1) to wavelength (um) and clip to range."""
    wl = 1e4 / wavenumber
    if hi is not None:
        wl = min(wl, hi)
    return max(wl, lo)


def refractive_index_ar(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.3)
    wl2 = wl * wl
    return math.sqrt(
        1.0
        + 20332.29e-8 * wl2 / (wl2 - 206.12e-6)
        + 34458.31e-8 * wl2 / (wl2 - 8.066e-3)
    )


def refractive_index_ch4(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.3, 0.6585)
    return 1.00042607 + 6.1396687e-6 * wl ** -2


def refractive_index_co(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.1672)
    return 1.0 + (22851.0 + 0.456e14 / (71427.0 ** 2 - (1e4 / wl) ** 2)) * 1e-8


def refractive_index_co2(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.1807, 1.6945)
    wl_2 = wl ** -2
    return (1.0
            + 6.99100e-2 / (166.175 - wl_2)
            + 1.44720e-3 / (79.609 - wl_2)
            + 6.42941e-5 / (56.3064 - wl_2)
            + 5.21306e-5 / (46.0196 - wl_2)
            + 1.46847e-6 / (0.0584738 - wl_2))


def refractive_index_h2(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.168)
    wl_2 = wl ** -2
    return (1.0
            + 0.0148956 / (180.7 - wl_2)
            + 0.0049037 / (92.0  - wl_2))


def refractive_index_h2o(wavenumber: float) -> float:
    """H2O at 373.15 K, 1 atm — IAPWS standard formulation."""
    rho = 8.272535e-1 / 1e3          # density ratio
    t   = 373.15 / 273.15            # temperature ratio
    wl0  = 0.589
    wl_uv = 0.2292020
    wl_ir = 5.432937

    wl_um = min(1e4 / wavenumber, 1.1)
    wl_um = max(wl_um, 0.2)
    wl    = wl_um / wl0

    a = (
        0.244257733
        + 9.74634476e-3 * rho
        - 3.73234996e-3 * t
        + 2.68678472e-4 * t * wl ** 2
        + 1.58920570e-3 / wl ** 2
        + 2.45934259e-3 / (wl ** 2 - wl_uv ** 2)
        + 0.900704920 / (wl ** 2 - wl_ir ** 2)
        - 1.66626219e-2 * rho ** 2
    ) * rho
    return math.sqrt((2.0 * a + 1.0) / (1.0 - a))


def refractive_index_he(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.09)
    wl2 = wl * wl
    return math.sqrt(
        1.0
        + 2.16463842e-5 * wl2 / (wl2 + 6.80769781e-4)
        + 2.10561127e-7 * wl2 / (wl2 - 5.13251289e-3)
        + 4.75092720e-5 * wl2 / (wl2 - 3.18621354e-3)
    )


def refractive_index_kr(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.3)
    wl2 = wl * wl
    return math.sqrt(
        1.0
        + 26102.88e-8 * wl2 / (wl2 - 2.01e-6)
        + 56946.82e-8 * wl2 / (wl2 - 10.043e-3)
    )


def refractive_index_n2(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.3)
    wl2 = wl * wl
    return math.sqrt(
        1.0
        + 39209.95e-8 * wl2 / (wl2 - 1146.24e-6)
        + 18806.48e-8 * wl2 / (wl2 - 13.476e-3)
    )


def refractive_index_ne(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.3)
    wl2 = wl * wl
    return math.sqrt(
        1.0
        + 9154.48e-8 * wl2 / (wl2 - 656.97e-6)
        + 4018.63e-8 * wl2 / (wl2 - 5.728e-3)
    )


def refractive_index_nh3(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.3)
    return 1.0 + 0.032953 / (90.392 - wl ** -2)


def refractive_index_xe(wavenumber: float) -> float:
    wl = _wavelength_um(wavenumber, 0.3)
    wl2 = wl * wl
    return math.sqrt(
        1.0
        + 103701.61e-8 * wl2 / (wl2 - 12750e-6)
        + 31228.61e-8  * wl2 / (wl2 - 0.561e-3)
    )


_REFRACTIVE_INDEX_TABLE: dict[str, Callable[[float], float]] = {
    "Ar":  refractive_index_ar,
    "CH4": refractive_index_ch4,
    "CO":  refractive_index_co,
    "CO2": refractive_index_co2,
    "H2":  refractive_index_h2,
    "H2O": refractive_index_h2o,
    "He":  refractive_index_he,
    "Kr":  refractive_index_kr,
    "N2":  refractive_index_n2,
    "Ne":  refractive_index_ne,
    "NH3": refractive_index_nh3,
    "Xe":  refractive_index_xe,
}


# ===========================================================================
# Public API
# ===========================================================================

def get_refractive_index(species: str, wavenumber: float) -> float:
    """
    Refractive index of *species* at *wavenumber* (cm-1).

    Falls back to :data:`DEFAULT_REFRACTIVE_INDEX` for species that have no
    formula in the table.
    """
    fn = _REFRACTIVE_INDEX_TABLE.get(species.strip())
    if fn is None:
        return DEFAULT_REFRACTIVE_INDEX
    return fn(wavenumber)


def rayleigh_scattering_coefficient(refractive_index: float, wavenumber: float) -> float:
    """
    Rayleigh-scattering cross-section in cm² (Bodhaine 1999, no depolarisation
    correction).

    Parameters
    ----------
    refractive_index : real refractive index of the gas at *wavenumber*
    wavenumber       : cm-1

    Returns
    -------
    sigma : cm²
    """
    wavelength_m = 1e-2 / wavenumber                       # cm-1 -> m
    n2 = refractive_index ** 2
    sigma_m2 = (24.0 * PI ** 3 * (n2 - 1.0) ** 2
                / (wavelength_m ** 4 * CST_N0 ** 2 * (n2 + 2.0) ** 2))
    return sigma_m2 * 1e4                                  # m2 -> cm2


__all__ = [
    "DEFAULT_REFRACTIVE_INDEX",
    "get_refractive_index",
    "rayleigh_scattering_coefficient",
    "refractive_index_ar", "refractive_index_ch4", "refractive_index_co",
    "refractive_index_co2", "refractive_index_h2", "refractive_index_h2o",
    "refractive_index_he", "refractive_index_kr", "refractive_index_n2",
    "refractive_index_ne", "refractive_index_nh3", "refractive_index_xe",
]
