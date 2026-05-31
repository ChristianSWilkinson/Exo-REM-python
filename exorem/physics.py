"""
Physical constants and basic physics helpers.

Mirrors the Fortran ``physics`` module (physics.f90).
All constants are CODATA 2018 (exact where applicable).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------
PI:     float = math.pi
CST_PI: float = math.pi          # Fortran-compatible alias

# ---------------------------------------------------------------------------
# Element catalog
# ---------------------------------------------------------------------------
N_ELEMENTS: int = 118

ELEMENTS_SYMBOL: list[str] = [
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

# (kg.mol-1)  IUPAC 2005 standard atomic weights (Wieser 2005);
# the most stable isotope is used for unstable / synthetic elements.
ELEMENTS_MOLAR_MASS: np.ndarray = np.array([
    1.00794e-3, 4.002602e-3,
    6.941e-3, 9.012182e-3, 10.811e-3, 12.0107e-3, 14.0067e-3, 15.9994e-3, 18.9984032e-3, 20.1797e-3,
    22.98976928e-3, 24.3050e-3, 26.9815386e-3, 28.0855e-3, 30.973762e-3, 32.065e-3, 35.453e-3, 39.948e-3,
    39.0983e-3, 40.078e-3, 44.955912e-3, 47.867e-3, 50.9415e-3, 51.9961e-3, 54.938045e-3, 55.845e-3,
    58.933195e-3, 58.6934e-3, 63.546e-3,
    65.409e-3, 69.723e-3, 72.64e-3, 74.92160e-3, 78.96e-3, 79.904e-3, 83.798e-3,
    85.4678e-3, 87.62e-3, 88.90585e-3, 91.224e-3, 92.90638e-3, 95.94e-3, 98e-3, 101.07e-3, 102.90550e-3,
    106.42e-3, 107.8682e-3,
    112.411e-3, 114.818e-3, 118.710e-3, 121.760e-3, 127.60e-3, 126.90447e-3, 131.293e-3,
    132.9054519e-3, 137.327e-3, 138.90547e-3,
    140.116e-3, 140.90765e-3, 144.242e-3, 145e-3, 150.36e-3, 151.964e-3, 157.25e-3, 158.92535e-3, 162.500e-3,
    164.93032e-3,
    167.259e-3, 168.93421e-3, 173.04e-3, 174.967e-3,
    178.49e-3, 180.94788e-3, 183.84e-3, 186.207e-3, 190.23e-3, 192.217e-3, 195.084e-3, 196.966569e-3,
    200.59e-3, 204.3833e-3, 207.2e-3, 208.98040e-3, 210e-3, 210e-3, 222e-3,
    223e-3, 226e-3, 227e-3,
    232.03806e-3, 231.03588e-3, 238.02891e-3,
    237e-3, 244e-3, 243e-3, 247e-3, 247e-3, 252e-3, 252e-3, 257e-3, 258e-3, 259e-3, 266e-3,
    267e-3, 268e-3, 269e-3, 278e-3, 269e-3, 282e-3, 281e-3, 286e-3, 285e-3, 286e-3,
    290e-3, 290e-3, 293e-3, 294e-3, 295e-3,
])

# ---------------------------------------------------------------------------
# Physical constants  (CODATA 2018; SI units unless stated)
# ---------------------------------------------------------------------------
CST_C:   float = 2.99792458e8        # (m s-1) speed of light in vacuum (exact)
CST_H:   float = 6.62607015e-34      # (J s)    Planck constant (exact)
CST_K:   float = 1.380649e-23        # (J K-1)  Boltzmann constant (exact)
CST_N_A: float = 6.02214076e23       # (mol-1)  Avogadro constant (exact)
CST_P0:  float = 1.01325e5           # (Pa)     1 atm
CST_T0:  float = 273.15              # (K)      Loschmidt reference temperature
CST_T_REF: float = 296.0             # (K)      GEISA 2015 reference temperature
CST_G:   float = 6.67430e-11         # (m3 kg-1 s-2) Newtonian gravitation constant

CST_HBAR: float = CST_H / (2.0 * PI)                       # (J s)
CST_N0:   float = CST_P0 / (CST_K * CST_T0)                # (m-3) Loschmidt constant
CST_R:    float = CST_N_A * CST_K                          # (J mol-1 K-1) molar gas constant
CST_SIGMA: float = (PI**2 / 60.0) * CST_K**4 / (CST_HBAR**3 * CST_C**2)  # Stefan-Boltzmann


# ===========================================================================
# Planck function
# ===========================================================================
_Number = Union[float, np.ndarray]


def planck_function(wavenumber: _Number, temperature: _Number) -> _Number:
    """
    Planck blackbody spectral radiance.

    Parameters
    ----------
    wavenumber  : cm-1
    temperature : K

    Returns
    -------
    spectral radiance, W m-2 sr-1 / cm-1

    Notes
    -----
    The factor 1e2 converts the speed of light from m/s to cm/s so that the
    units match the wavenumber input.  The trailing factor 1e4 converts the
    final answer from W cm-2 sr-1 / cm-1 to W m-2 sr-1 / cm-1.
    """
    wn = np.asarray(wavenumber, dtype=float) if np.ndim(wavenumber) else float(wavenumber)
    t  = np.asarray(temperature, dtype=float) if np.ndim(temperature) else float(temperature)

    c_cms = CST_C * 1e2                                   # m/s -> cm/s
    arg   = CST_H * c_cms * wn / (CST_K * t)
    # Clip to avoid overflow in exp() for very large wn/T
    if isinstance(arg, np.ndarray):
        arg = np.clip(arg, -700.0, 700.0)
        denom = np.exp(arg) - 1.0
    else:
        denom = math.exp(min(arg, 700.0)) - 1.0

    return 2.0 * CST_H * c_cms**2 * wn**3 / denom * 1e4


def spherical_black_body_spectral_radiance(
    wavenumber: _Number,
    temperature: _Number,
    radius: float,
    distance: float,
    redistribution_factor: float,
) -> _Number:
    """
    Spectral radiance from a spherical blackbody source seen at a given distance.

    Parameters
    ----------
    wavenumber             : cm-1
    temperature            : K  -- effective temperature of the source
    radius                 : m  -- source radius
    distance               : m  -- distance to the observer
    redistribution_factor  : dimensionless heat-redistribution factor

    Returns
    -------
    spectral radiance, W m-2 sr-1 / cm-1
    """
    return (planck_function(wavenumber, temperature)
            * (radius / distance) ** 2
            * redistribution_factor)


__all__ = [
    "PI", "CST_PI",
    "N_ELEMENTS", "ELEMENTS_SYMBOL", "ELEMENTS_MOLAR_MASS",
    "CST_C", "CST_H", "CST_K", "CST_N_A", "CST_P0",
    "CST_T0", "CST_T_REF", "CST_G", "CST_HBAR", "CST_N0",
    "CST_R", "CST_SIGMA",
    "planck_function", "spherical_black_body_spectral_radiance",
]
