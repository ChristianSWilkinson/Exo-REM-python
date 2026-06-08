"""
Thermochemical equilibrium chemistry for exoplanet atmospheres.

Implements the full Exorem chemistry pipeline:
  - 43 gas species, 20 condensate species
  - Gibbs free-energy minimisation (JANAF database)
  - Quench levels (Zahnle & Marley 2014)
  - Saturation pressure curves (H₂O, NH₃, NH₄SH, …)

Mirrors the Fortran ``chemistry`` module (chemistry.f90).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .math_utils import interp_ex_0d, interp
from .physics import CST_R, ELEMENTS_SYMBOL, ELEMENTS_MOLAR_MASS, N_ELEMENTS


def _safe_exp(x: float) -> float:
    """
    ``math.exp`` that clamps the argument to ±700 to avoid OverflowError.

    Many chemistry equilibrium expressions are of the form
    ``exp(-ΔG/(R·T))``.  When T is small (T → 50–100 K, the retrieval's
    safety floor) and ΔG is strongly negative, the bare ``math.exp`` raises
    OverflowError.  Beyond ±700 the value is either ~1e305 or ~1e-305; in
    either limit the downstream chemistry treats K_eq as "very large" or
    "essentially zero", so capping is physically appropriate.
    """
    if x > 700.0:
        return math.exp(700.0)
    if x < -700.0:
        return 0.0
    return math.exp(x)


# ---------------------------------------------------------------------------
# Species catalogues
# ---------------------------------------------------------------------------

N_CONDENSATES: int = 20
N_GASES:       int = 43

CONDENSATE_NAMES: list[str] = [
    "Al2O3", "Ca",    "CaTiO3", "Cr",    "Cr2O3",
    "Fe",    "H2O",   "H3PO4",  "KCl",   "Mg2SiO4",
    "MgSiO3","MnS",   "Na2S",   "NH3",   "NH4Cl",
    "NH4SH", "SiO2",  "TiN",    "VO",    "ZnS",
]

GASES_NAMES: list[str] = [
    "Al",  "Ar",  "Ca",  "CH3", "CH4",
    "CO",  "CO2", "Cr",  "Fe",  "FeH",
    "H",   "H2",  "H2O", "H2S", "HCl",
    "HCN", "He",  "K",   "KCl", "Kr",
    "Mg",  "Mn",  "N2",  "Na",  "NaCl",
    "Ne",  "NH3", "Ni",  "P",   "P2",
    "PH2", "PH3", "PO",  "SiH4","SiO",
    "Ti",  "TiO", "TiO2","V",   "VO",
    "VO2", "Xe",  "Zn",
]


def gas_id(gas_name: str) -> int:
    """Return 0-based index of *gas_name* in GASES_NAMES."""
    name = gas_name.strip()
    for i, n in enumerate(GASES_NAMES):
        if n == name:
            return i
    raise ValueError(f"chemistry: gas '{gas_name}' not implemented")


def condensate_id(condensate_name: str) -> int:
    """Return 0-based index of *condensate_name* in CONDENSATE_NAMES."""
    name = condensate_name.strip()
    for i, n in enumerate(CONDENSATE_NAMES):
        if n == name:
            return i
    raise ValueError(f"chemistry: condensate '{condensate_name}' not implemented")


# ---------------------------------------------------------------------------
# Molar mass utilities
# ---------------------------------------------------------------------------

_ELEMENT_MASS: dict[str, float] = {
    sym.strip(): float(m)
    for sym, m in zip(ELEMENTS_SYMBOL, ELEMENTS_MOLAR_MASS)
}


def count_all_elements(formula: str) -> np.ndarray:
    """
    Parse a chemical formula and return an atom-count array of length N_ELEMENTS.

    Example: 'NH4SH' → {'N': 1, 'H': 5, 'S': 1}
    """
    counts = np.zeros(N_ELEMENTS)
    i = 0
    while i < len(formula):
        if formula[i].isupper():
            # Element symbol (1 or 2 chars)
            j = i + 1
            while j < len(formula) and formula[j].islower():
                j += 1
            symbol = formula[i:j]
            # Count
            k = j
            while k < len(formula) and formula[k].isdigit():
                k += 1
            count = int(formula[j:k]) if j < k else 1
            # Map to element index
            try:
                idx = next(e for e, s in enumerate(ELEMENTS_SYMBOL)
                           if s.strip() == symbol)
                counts[idx] += count
            except StopIteration:
                pass  # unknown symbol – skip
            i = k
        else:
            i += 1
    return counts


def calculate_species_molar_mass(species_name: str) -> float:
    """
    Calculate the molar mass of a species from its chemical formula.

    Returns
    -------
    molar_mass : (kg mol⁻¹)
    """
    counts = count_all_elements(species_name.strip())
    return float(np.dot(counts, ELEMENTS_MOLAR_MASS))


def calculate_gases_molar_mass() -> tuple[np.ndarray, np.ndarray]:
    """
    Compute molar masses and elemental composition matrices for all gases.

    Returns
    -------
    gases_molar_mass : (N_GASES,)  kg mol⁻¹
    elements_in_gases: (N_GASES, N_ELEMENTS) atom counts
    """
    elements_in_gases = np.zeros((N_GASES, N_ELEMENTS))
    gases_molar_mass  = np.zeros(N_GASES)

    for i, name in enumerate(GASES_NAMES):
        elements_in_gases[i] = count_all_elements(name)
        gases_molar_mass[i]  = float(np.dot(elements_in_gases[i], ELEMENTS_MOLAR_MASS))

    return gases_molar_mass, elements_in_gases


# ---------------------------------------------------------------------------
# Equilibrium constant
# ---------------------------------------------------------------------------

def equilibrium_constant_gases(
    stoichiometric_coefficients: list[int],
    delta_g_species: list[float],
    pressure: float,
    temperature: float,
) -> float:
    """
    Equilibrium constant for a gas-phase reaction.

    Given aA + bB → cC + dD, stoichiometric coefficients are positive for
    products and negative for reactants.

    Parameters
    ----------
    stoichiometric_coefficients : (n_species,)  signed integers
    delta_g_species             : (n_species,)  kJ mol⁻¹ standard Gibbs energy of formation
    pressure    : (bar)
    temperature : (K)

    Returns
    -------
    K_eq : dimensionless equilibrium constant in VMR units
    """
    delta_g_reaction = sum(c * g * 1e3   # kJ → J
                           for c, g in zip(stoichiometric_coefficients, delta_g_species))
    p_power = -sum(stoichiometric_coefficients)
    return _safe_exp(-delta_g_reaction / (CST_R * temperature)) * pressure ** p_power


# ---------------------------------------------------------------------------
# Saturation pressures
# ---------------------------------------------------------------------------

def h2o_saturation_pressure(temperature: float) -> float:
    """
    H₂O saturation vapour pressure (bar).

    Sources:
      - Gas–Ice I:       Feistel & Wagner 2007
      - Gas–Liquid:      Wagner & Pruss 1993 (IAPWS 2011)
      - Gas–Ice VII:     Lin et al. 2004
    """
    T_tp = 273.16       # K – triple point
    p_tp = 6.11657e-3   # bar
    T_cp = 647.096      # K – critical point
    p_cp = 220.64       # bar
    T_ice7_tp = 355.0   # K – liquid–ice6–ice7 triple point
    p_ice7_tp = 2.17e4  # bar
    p_c = 0.85e4        # bar (Lin 2004)
    alpha = 3.47        # (Lin 2004)

    # Feistel–Wagner coefficients for gas↔ice-I
    e = np.array([20.996967, 3.724375, -13.920548, 29.698877,
                  -40.197239, 29.788048, -9.130510])

    # Wagner–Pruss coefficients for gas↔liquid
    a_wp = np.array([-7.85951783, 1.84408259, -11.7866497,
                     22.6807411, -15.9618719, 1.80122502])
    tau_pow = np.array([1.0, 1.5, 3.0, 3.5, 4.0, 7.5])

    if temperature <= T_tp:
        eta = sum(e[i] * (temperature / T_tp) ** i for i in range(len(e)))
        return p_tp * math.exp(
            1.5 * math.log(temperature / T_tp)
            + (1.0 - T_tp / temperature) * eta
        )
    elif temperature <= T_cp:
        tau = 1.0 - temperature / T_cp
        eta = sum(a_wp[i] * tau ** tau_pow[i] for i in range(len(a_wp)))
        return p_cp * math.exp(T_cp / temperature * eta)
    else:
        return p_c * ((temperature / T_ice7_tp) ** alpha - 1.0) + p_ice7_tp


def nh3_saturation_pressure(temperature: float) -> float:
    """
    NH₃ saturation vapour pressure (bar).

    Sources: Fray & Schmitt 2009; Lide 2006.
    """
    T_tp = 195.41   # K
    p_tp = 6.09e-2  # bar
    T_cp = 405.5    # K
    p_cp = 113.5    # bar

    a = np.array([1.596e1, -3.537e3, -3.310e4, 1.742e6, -2.995e7])

    temps_ref = np.array([T_tp, 200, 205, 210, 215, 220, 225, 230, 235, 240,
                          245, 250, 255, 260, 265, 270, 275, 280, 285, 290,
                          295, 300, T_cp])
    pres_ref  = np.array([p_tp,
                          8.7e-2, 12.6e-2, 17.9e-2, 24.9e-2, 34.1e-2,
                          45.9e-2, 60.8e-2, 79.6e-2, 1.03, 1.31, 1.65,
                          2.07, 2.56, 3.13, 3.81, 4.60, 5.52, 6.55, 7.74,
                          9.09, 10.62, p_cp])

    if temperature <= T_tp:
        return math.exp(sum(a[i] / temperature ** i for i in range(len(a))))
    elif temperature <= T_cp:
        return math.exp(
            float(interp(
                np.array([math.log(temperature)]),
                np.log(temps_ref),
                np.log(pres_ref)
            )[0])
        )
    else:
        return 1e10


def nh4sh_saturation_pressure(temperature: float) -> float:
    """
    NH₄SH saturation pressure (bar).

    Source: Stull 1947.
    """
    return 10.0 ** (14.581 - 4636.3 / temperature) * 1.01325  # atm → bar


# ---------------------------------------------------------------------------
# Main thermochemistry routine
# ---------------------------------------------------------------------------

def calculate_chemistry(
    at_equilibrium: bool,
    pressures_layers:    np.ndarray,      # (n_layers,) Pa
    temperatures_layers: np.ndarray,      # (n_layers,)
    gravities_layers:    np.ndarray,      # (n_layers,)
    eddy_diffusion_coefficient: np.ndarray,  # (n_layers,) cm² s⁻¹
    pressures:           np.ndarray,      # (n_levels,) Pa
    scale_height:        np.ndarray,      # (n_layers,) km
    elemental_h_ratio:   np.ndarray,      # (N_ELEMENTS,)
    temperatures_thermochemistry: np.ndarray,  # thermochemistry T grid
    gases_delta_g:       np.ndarray,      # (N_GASES, n_thermo)  kJ mol⁻¹
    condensates_delta_g: np.ndarray,      # (N_CONDENSATES, n_thermo)
    gases_c_p:           np.ndarray,      # (N_GASES, n_thermo)
    gases_molar_mass:    np.ndarray,      # (N_GASES,)
    elements_in_gases:   np.ndarray,      # (N_GASES, N_ELEMENTS)
    solar_h_ratio:       np.ndarray = None,  # (N_ELEMENTS,) solar X/H, for metallicity factors
) -> dict:
    """
    Compute thermochemical equilibrium VMR profiles.

    This is a layer-by-layer, bottom-to-top integration following the
    Fortran ``calculate_chemistry`` routine.  The returned dictionary
    contains all outputs that the Fortran subroutine populates via
    ``intent(out)`` arguments.

    Parameters
    ----------
    at_equilibrium : if True, compute full equilibrium (no quenching)

    Returns
    -------
    dict with keys:
      - gases_vmr         : (N_GASES, n_layers)
      - gas_element_abd   : (N_ELEMENTS, n_layers)
      - p_c_condensates   : (N_CONDENSATES,)   condensation pressure (mbar)
      - vmr_sat_condensates: (N_CONDENSATES, n_layers)
      - vmr_c_condensates : (N_CONDENSATES,)
      - layer_condensates : (N_CONDENSATES,)   int condensation layer index
    """
    print("Calculating thermochemistry at equilibrium..." if at_equilibrium
          else "Calculating non-equilibrium thermochemistry...")

    nlay = len(pressures_layers)

    gases_vmr         = np.zeros((N_GASES, nlay))
    gas_element_abd   = np.zeros((N_ELEMENTS, nlay))
    p_c_condensates   = np.zeros(N_CONDENSATES)
    vmr_sat_condensates = np.zeros((N_CONDENSATES, nlay))
    vmr_c_condensates = np.zeros(N_CONDENSATES)
    layer_condensates = np.zeros(N_CONDENSATES, dtype=int)

    is_condensed = np.zeros(N_CONDENSATES, dtype=bool)
    is_quenched  = np.zeros(N_GASES, dtype=bool)

    # ----------- metallicity factors for quench timescales -----------------
    # metallicity_X = (X/H)_atmosphere / (X/H)_solar  for C(Z=6), N(7), O(8).
    def _met(zi: int) -> float:
        if (solar_h_ratio is not None and len(solar_h_ratio) > zi
                and solar_h_ratio[zi] > 0.0):
            return float(elemental_h_ratio[zi] / solar_h_ratio[zi])
        return 1.0
    metallicity_c = _met(5)   # C  (index = Z-1)
    metallicity_n = _met(6)   # N
    metallicity_o = _met(7)   # O

    # ----------- quench state (carry-forward freezing) ---------------------
    idx_n2, idx_nh3, idx_hcn = gas_id("N2"), gas_id("NH3"), gas_id("HCN")
    n2_quench = hcn_quench = False
    n2_frozen = nh3_frozen = hcn_frozen = 0.0
    idx_co, idx_co2, idx_ch4, idx_ch3, idx_h2o = (
        gas_id("CO"), gas_id("CO2"), gas_id("CH4"), gas_id("CH3"), gas_id("H2O"))
    co_ch4_quench = co_co2_quench = False
    qcoco2 = qch4q = 0.0

    # Previous-layer chemical/mixing timescales, needed to interpolate the
    # quench point (Fortran tracks tchemC1/tchemCO21/tmix1).  Seed from the
    # init_non_equilibrium virtual deeper layer (1.198·T₀, 2·P₀); tmix1 is
    # seeded as 1.4352·tmix on the first layer (tmix needs the loop's Kzz).
    _pold0 = 2.0 * pressures_layers[0] * 1e-5
    _told0 = temperatures_layers[0] * 1.198
    _mco0  = 0.5 * (metallicity_c + metallicity_o)
    _tq1_0 = 1.5e-6 * math.exp(4.2e4 / _told0) * _mco0 ** (-0.70) / _pold0
    _tq2_0 = 4.0e1 * _mco0 ** (-0.70) * math.exp(2.5e4 / _told0) / (_pold0 * _pold0)
    tchemc1   = 1.0 / (1.0 / _tq1_0 + 1.0 / _tq2_0)
    tchemco21 = 1.0e-10 * math.exp(3.8e4 / _told0) / math.sqrt(_pold0)
    tmix1     = None

    # ----------- initialise VMR from elemental abundances ------------------
    _init_gas_element_abd(gases_vmr, gas_element_abd, elemental_h_ratio,
                          elements_in_gases, nlay)

    # -----------------------------------------------------------------------
    # Layer loop (pressure decreasing = altitude increasing)
    # -----------------------------------------------------------------------
    for ip in range(nlay):
        t = temperatures_layers[ip]
        p = pressures_layers[ip] * 1e-3     # Pa → mbar
        pbar = pressures[ip] * 1e-3         # Pa → mbar

        # Interpolate Gibbs free energies to current temperature
        gases_delta_g_i = np.array([
            interp_ex_0d(t, temperatures_thermochemistry, gases_delta_g[k, :])
            for k in range(N_GASES)
        ])
        condensates_delta_g_i = np.array([
            interp_ex_0d(t, temperatures_thermochemistry, condensates_delta_g[k, :])
            for k in range(N_CONDENSATES)
        ])

        if ip > 0:
            gas_element_abd[:, ip] = gas_element_abd[:, ip - 1]
            gases_vmr[:, ip]       = gases_vmr[:, ip - 1]

        # -- Time constants for quenching (bar, metres, cm² s⁻¹) --
        p_bar_ip = pressures_layers[ip] * 1e-5      # Pa → bar
        tmix, tchemc, tchemco2, tchemn, tchemhcn = _calculate_time_constants(
            t, p_bar_ip, scale_height[ip], eddy_diffusion_coefficient[ip],
            metallicity_c, metallicity_n, metallicity_o, at_equilibrium)
        if tmix1 is None:
            tmix1 = 1.4352 * tmix      # init_non_equilibrium seed

        # -- H₂ / H equilibrium --
        gases_vmr[:, ip] = _calculate_h2_h_equilibrium(
            t, p, gases_vmr[:, ip], gases_delta_g_i, elemental_h_ratio)

        # -- H₂O / CH₄ / CO / CO₂ / SiO / SiH₄  (equilibrium below quench) --
        if (not co_ch4_quench) and (tchemc <= tmix):
            gases_vmr[:, ip], is_condensed, vmr_sat_condensates[:, ip], layer_condensates = \
                _equil_co_si_o(
                    t, p, ip, gases_vmr[:, ip], gases_delta_g_i, is_condensed,
                    vmr_sat_condensates[:, ip], layer_condensates,
                    gas_element_abd[:, ip], elements_in_gases)
        else:
            # --- CARBON QUENCH: freeze C totals, re-solve O via osiqco --------
            # Mirrors the Fortran: below tchemC≤tmix → osi (equilibrium, above);
            # at each crossing the quench routines INTERPOLATE the true (T,P) and
            # re-solve there to capture qcoco2=CO+CO2 / qch4q=CH4·(1+ech3q), then
            # osiqco (CO frozen) / osiqcoco2 (CO+CO2 frozen) hold them aloft while
            # re-solving H2O from oxygen conservation.
            if not co_ch4_quench:
                co_ch4_quench = True
                # Interpolated CO/CH4-quench capture (Fortran co_ch4_quenching):
                # interpolate the true (T,P) of the tchemC=tmix crossing, re-solve
                # the C/O/Si equilibrium there, and capture the totals from it.
                # The grid layer is ~40-50 K colder than the crossing, where
                # equilibrium has already shifted toward CH4 → grid capture would
                # freeze CO+CO2 (and thus CO2) too low.
                _told, _pold = _get_pold_told(ip, temperatures_layers, pressures_layers)
                _tqC, _pqC, _ = _quench_interp_pt(
                    _told, _pold, t, p_bar_ip, tchemc1, tmix1, tchemc, tmix)
                _dgq = np.array([
                    interp_ex_0d(_tqC, temperatures_thermochemistry, gases_delta_g[k, :])
                    for k in range(N_GASES)])
                _vmrq = gases_vmr[:, ip].copy()
                _qh2q = _vmrq[gas_id("H2")] if _vmrq[gas_id("H2")] > 0 else 1e-300
                _kcoq  = _keq(["H2O", "CH4", "CO", "H2"], [-1, -1, 1, 3], _dgq, _pqC, _tqC, _qh2q)
                _ech3q = math.sqrt(max(
                    _keq(["CH4", "CH3", "H2"], [-2, 2, 1], _dgq, _pqC, _tqC, _qh2q), 0.0))
                _kco2q = _keq(["CO", "H2O", "CO2", "H2"], [-1, -1, 1, 1], _dgq, _pqC, _tqC, _qh2q)
                _ksiq  = _keq(["SiH4", "H2O", "SiO", "H2"], [-1, -1, 1, 3], _dgq, _pqC, _tqC, _qh2q)
                _vmrq, _, _, _, _ = _osi(
                    _vmrq, gas_element_abd[:, ip], elements_in_gases, _dgq,
                    _kcoq, _ech3q, _kco2q, _ksiq, _pqC, _tqC, False)
                qcoco2 = _vmrq[idx_co] + _vmrq[idx_co2]
                qch4q  = _vmrq[idx_ch4] * (1.0 + _ech3q)
            gases_vmr[idx_ch4, ip] = qch4q
            gases_vmr[idx_ch3, ip] = 1e-300
            # H₂O condensation (same closure as the equilibrium routine)
            p_bar_eq = p * 1e-2          # bar (was p·1e-3 = bar/10)
            qsat_h2o = h2o_saturation_pressure(t) / p_bar_eq
            vmr_sat_condensates[condensate_id("H2O"), ip] = qsat_h2o
            h2o_sat = gases_vmr[idx_h2o, ip] >= qsat_h2o
            if h2o_sat:
                if not is_condensed[condensate_id("H2O")]:
                    is_condensed[condensate_id("H2O")] = True
                    layer_condensates[condensate_id("H2O")] = ip
                gases_vmr[idx_h2o, ip] = qsat_h2o
            if tchemco2 <= tmix:
                gases_vmr[:, ip] = _osiqco(
                    t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
                    gas_element_abd[:, ip], qcoco2, h2o_sat)
            else:
                if not co_co2_quench:
                    co_co2_quench = True
                    # Interpolated CO/CO2-quench capture (Fortran co_co2_quenching):
                    # re-solve osiqco at the interpolated tchemCO2=tmix crossing so
                    # the CO/CO2 that osiqcoco2 then freezes are taken at the true
                    # (hotter) quench point rather than the colder grid layer.
                    _told, _pold = _get_pold_told(ip, temperatures_layers, pressures_layers)
                    _tqC2, _pqC2, _ = _quench_interp_pt(
                        _told, _pold, t, p_bar_ip, tchemco21, tmix1, tchemco2, tmix)
                    _dgq2 = np.array([
                        interp_ex_0d(_tqC2, temperatures_thermochemistry, gases_delta_g[k, :])
                        for k in range(N_GASES)])
                    gases_vmr[:, ip] = _osiqco(
                        _tqC2, _pqC2 * 1e2, ip, gases_vmr[:, ip], _dgq2,
                        gas_element_abd[:, ip], qcoco2, h2o_sat)
                gases_vmr[:, ip] = _osiqcoco2(
                    t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
                    gas_element_abd[:, ip], h2o_sat)

        # -- NH₃ / N₂ / HCN  (equilibrium below the quench level) --
        # Carried-forward values == the last fully-equilibrium layer's values.
        n2_cf, nh3_cf, hcn_cf = (gases_vmr[idx_n2, ip],
                                 gases_vmr[idx_nh3, ip],
                                 gases_vmr[idx_hcn, ip])
        gases_vmr[:, ip] = _calculate_nh3_n2_hcn(
            t, p, ip, gases_vmr[:, ip], gases_delta_g_i, at_equilibrium,
            is_quenched, tchemn, tmix)

        # -- N₂/NH₃ quench: freeze at the quench-level value above it --------
        # Mirrors the Fortran (skip the update above the quench → carry-forward).
        if tchemn > tmix:
            if not n2_quench:
                n2_quench = True
                n2_frozen, nh3_frozen = n2_cf, nh3_cf
            gases_vmr[idx_n2, ip]  = n2_frozen
            gases_vmr[idx_nh3, ip] = nh3_frozen
        # -- HCN quench (its own, higher quench level) -----------------------
        if tchemhcn > tmix:
            if not hcn_quench:
                hcn_quench = True
                hcn_frozen = hcn_cf
            gases_vmr[idx_hcn, ip] = hcn_frozen

        # -- Cl / Na / K --
        gases_vmr[:, ip], is_condensed, vmr_sat_condensates[:, ip], \
            layer_condensates = _calculate_cl_na_k(
                t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
                condensates_delta_g_i, is_condensed,
                vmr_sat_condensates[:, ip], layer_condensates,
                pressures_layers, temperatures_layers, gas_element_abd)

        # -- Al₂O₃ --
        gases_vmr[:, ip], is_condensed, vmr_sat_condensates[:, ip], \
            layer_condensates = _calculate_al_o(
                t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
                condensates_delta_g_i, is_condensed,
                vmr_sat_condensates[:, ip], layer_condensates,
                pressures_layers, temperatures_layers)

        # -- Cr --
        gases_vmr[:, ip], is_condensed, vmr_sat_condensates[:, ip], \
            layer_condensates = _calculate_cr(
                t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
                condensates_delta_g_i, is_condensed,
                vmr_sat_condensates[:, ip], layer_condensates,
                pressures_layers, temperatures_layers)

        # -- Fe / Ni / Co --
        gases_vmr[:, ip] = _calculate_fe_ni_co(
            t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
            condensates_delta_g_i, is_condensed,
            vmr_sat_condensates[:, ip], layer_condensates,
            pressures_layers, temperatures_layers)

        # -- Ca / Ti / V --
        gases_vmr[:, ip] = _calculate_ca_o_ti_v(
            t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
            condensates_delta_g_i, is_condensed, gas_element_abd,
            elemental_h_ratio)

        # -- Mg / Si / O silicate clouds (forsterite/enstatite/SiO2 O-sink) --
        gases_vmr[:, ip] = _calculate_mg_si_o(
            t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
            condensates_delta_g_i, is_condensed,
            vmr_sat_condensates[:, ip], layer_condensates,
            pressures_layers, temperatures_layers,
            gas_element_abd=gas_element_abd, elements_in_gases=elements_in_gases,
            co_ch4_quench=co_ch4_quench, co_co2_quench=co_co2_quench,
            qcoco2=qcoco2)

        # -- MnS --
        gases_vmr[:, ip] = _calculate_mn_s(
            t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
            condensates_delta_g_i, is_condensed,
            vmr_sat_condensates[:, ip], layer_condensates,
            pressures_layers, temperatures_layers)

        # -- ZnS --
        gases_vmr[:, ip] = _calculate_zn_s(
            t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
            condensates_delta_g_i, is_condensed,
            vmr_sat_condensates[:, ip], layer_condensates,
            pressures_layers, temperatures_layers)

        # -- PH₃ / P / P₂ / PH₂ / PO --
        gases_vmr[:, ip] = _calculate_p(
            t, p, ip, gases_vmr[:, ip], gases_delta_g_i,
            condensates_delta_g_i, is_condensed, gas_element_abd, at_equilibrium)

        # Update elemental abundances from VMR
        _update_gas_element_abd(gases_vmr[:, ip], gas_element_abd[:, ip],
                                elements_in_gases)

        # Carry this layer's timescales forward for the next layer's quench
        # interpolation (Fortran: tchemC1=tchemC, tchemCO21=tchemCO2, tmix1=tmix).
        tchemc1, tchemco21, tmix1 = tchemc, tchemco2, tmix

    # ---------------------------------------------------------------------
    # Normalisation step
    # ---------------------------------------------------------------------
    # The per-layer chemistry above computes VMRs relative to the *hydrogen*
    # number density (e.g. VMR(H₂) = elemental_h_ratio[H]/2 = 0.5,
    # VMR(He) = elemental_h_ratio[He] = 0.084, …).  Their sum is therefore
    # close to (1 + He/H × 2)/2 ≈ 0.58, not 1.0.
    #
    # Downstream code (mean molar mass, scale height, optical depth, …)
    # assumes ``sum(gases_vmr[:, ip]) == 1`` per the standard VMR definition.
    # Without this step the mean molar mass is under-counted by the same
    # factor (≈ 0.58), which dramatically reduces the apparent atmospheric
    # opacity and destabilises the radiative-convective iteration.
    for ip in range(nlay):
        s = float(np.sum(gases_vmr[:, ip]))
        if s > 0.0:
            gases_vmr[:, ip] /= s

    return {
        "gases_vmr":          gases_vmr,
        "gas_element_abd":    gas_element_abd,
        "p_c_condensates":    p_c_condensates,
        "vmr_sat_condensates": vmr_sat_condensates,
        "vmr_c_condensates":  vmr_c_condensates,
        "layer_condensates":  layer_condensates,
    }


# ===========================================================================
# Internal helpers (abbreviated for readability; full logic from Fortran)
# ===========================================================================

def _init_gas_element_abd(
    gases_vmr: np.ndarray,
    gas_element_abd: np.ndarray,
    elemental_h_ratio: np.ndarray,
    elements_in_gases: np.ndarray,
    nlay: int,
) -> None:
    """Initialise gas VMRs from elemental abundances.

    r36: Port of the Fortran ``init_gas_element_abd`` (chemistry.f90 lines
    328-475).  The previous Python stub only set H2, He, and H2O, leaving
    EVERY OTHER GAS SPECIES at zero — including CH4, CO, NH3, N2, Na, K,
    Fe, FeH, Ti, TiO, V, VO, H2S, PH3, etc.  Downstream equilibrium
    calculations (e.g. ``_calculate_h2o_ch4_co_co2_sih4_sio``) compute
    secondary species from already-initialised primary ones — e.g. CH4
    via ``qch4 = K_eq * qH2**3 * qCO``.  With qCO = 0, every secondary
    stayed at zero forever, and the atmosphere modelled by the radiative
    transfer was essentially pure H2 + He + H2O for the whole 50-iteration
    run.  Hence the persistent 1019 K stratospheric bump: with no CH4,
    CO, alkali, or oxide opacity, the model atmosphere had no way to
    radiate efficiently in the 1-5 μm window that real brown-dwarf
    photospheres depend on.

    The Fortran algorithm:

      For each gas species:
        n_unique = number of distinct elements
        * n_unique == 1 with H only (H, H2)  → flag has_H, init later
        * n_unique == 1 without H            → vmr = X[Z]/n_Z         (e.g. Na, K, Fe, Ti, V, N2, He)
        * n_unique == 2 with H (hydride)     → vmr = X[non-H]/n_non-H (e.g. H2O, CH4, HCl, FeH, H2S, PH3, SiH4)
        * n_unique == 2 without H            → vmr = min over elements of X[Z]/n_Z   (e.g. CO, SiO, NaCl)
        * n_unique >= 3                      → vmr = 0  (filled in later by equilibrium)

      Then explicit overrides: zero out species that should be
      populated by downstream equilibrium (CH3, CO2, FeH, H, HCN, KCl,
      NH3, NaCl, P, P2, PH2, PO, SiH4, TiO, TiO2, VO, VO2).

      C/O partitioning:
        * If O > C + Si: oxygen-rich, all C goes to CO, H2O = O - C - Si
        * Else if O > Si: H2O = O - Si, all C goes to CH4 (CO=0)
        * Else: C-rich → unsupported

      Finally compute H2 from the constraint  Σ(VMR) = 1, accumulating
      H/2-equivalents over H-bearing species.  Multiply everything by
      h_abd = 1/(0.5 + sum_non_H + sum_H_fraction) so that the absolute
      element abundances respect ``elemental_h_ratio``.

    Note: I fixed an apparent Fortran bug where ``sum_vmr_tmp_h_frac``
    is assigned (not accumulated) in the loop — see line 451 of
    chemistry.f90.  The Python version accumulates correctly.
    """
    n_gases = gases_vmr.shape[0]

    # Element indices in ELEMENTS_SYMBOL (0-based)
    iH  = 0    # H
    iC  = 5    # C
    iO  = 7    # O
    iSi = 13   # Si

    vmr_tmp = np.zeros(n_gases)
    has_h   = np.zeros(n_gases, dtype=bool)

    # ---------------- first-pass assignment by structure ------------------
    for i in range(n_gases):
        counts = elements_in_gases[i]
        n_unique = int(np.sum(counts > 0))
        n_H = int(counts[iH]) if iH < len(counts) else 0

        if n_unique == 1:
            if n_H > 0:
                # Pure H species (H, H2) — handled at the end via H2 closure.
                has_h[i] = True
                continue
            # Non-H single element (He, Ne, Ar, Kr, Xe, Na, K, Mg, Fe, ...)
            j_max = int(np.argmax(counts))
            if elemental_h_ratio[j_max] > 0 and counts[j_max] > 0:
                vmr_tmp[i] = elemental_h_ratio[j_max] / counts[j_max]

        elif n_unique == 2:
            if n_H > 0:
                # Hydride: VMR set by the non-H partner
                # find non-H element
                j_partner = -1
                for j in range(len(counts)):
                    if j != iH and counts[j] > 0:
                        j_partner = j
                        break
                if j_partner >= 0 and counts[j_partner] > 0:
                    has_h[i] = True
                    vmr_tmp[i] = elemental_h_ratio[j_partner] / counts[j_partner]
            else:
                # Two non-H elements: limited by rarer element.  Note we do
                # NOT skip elements with zero elemental abundance — if a
                # species depends on an element absent from the input
                # (e.g. SiO when no Si is supplied), the min is zero and
                # the species correctly gets vmr = 0.
                v_min = np.inf
                for j in range(len(counts)):
                    if counts[j] > 0:
                        candidate = elemental_h_ratio[j] / counts[j]
                        if candidate < v_min:
                            v_min = candidate
                if np.isfinite(v_min):
                    vmr_tmp[i] = v_min
        # else: n_unique >= 3 → leave at zero (filled by downstream equilibrium)

    # ---------------- explicit zeros (Fortran lines 407-423) ----------------
    # These species are populated by downstream equilibrium reactions, not
    # by the first-pass element apportionment.
    for name in ("CH3", "CO2", "FeH", "H", "HCN", "KCl",
                 "NH3", "NaCl", "P", "P2", "PH2", "PO",
                 "SiH4", "TiO", "TiO2", "VO", "VO2"):
        try:
            vmr_tmp[gas_id(name)] = 0.0
        except ValueError:
            pass

    # ---------------- C / O partitioning (Fortran 425-438) ------------------
    x_C  = float(elemental_h_ratio[iC])
    x_O  = float(elemental_h_ratio[iO])
    x_Si = float(elemental_h_ratio[iSi]) if iSi < len(elemental_h_ratio) else 0.0

    try:
        idx_CH4 = gas_id("CH4")
        idx_CO  = gas_id("CO")
        idx_H2O = gas_id("H2O")
    except ValueError:
        idx_CH4 = idx_CO = idx_H2O = -1

    if idx_H2O >= 0:
        if x_O > x_C + x_Si:
            # oxygen-rich: all C → CO, all Si → SiO, remainder → H2O
            if idx_CH4 >= 0:
                vmr_tmp[idx_CH4] = 0.0
            vmr_tmp[idx_H2O] = x_O - x_C - x_Si
        elif x_O > x_Si:
            # intermediate: all C → CH4 (CO = 0), Si → SiO, remainder → H2O
            if idx_CO >= 0:
                vmr_tmp[idx_CO] = 0.0
            vmr_tmp[idx_H2O] = x_O - x_Si
        else:
            # C-rich: not supported by this initialiser
            print("  Warning: C+Si > O — chemistry initialiser falls back to "
                  "x_H2O=0; results may be unphysical.")
            vmr_tmp[idx_H2O] = 0.0

    # ---------------- close VMRs via H2 (Fortran 440-468) -------------------
    # Each entry in vmr_tmp is currently in units of elemental_h_ratio (i.e.,
    # relative to gas_element_abd_H, the absolute H abundance per gas
    # molecule).  We need to find h_abd such that the final ΣVMR = 1.
    #
    # Constraint:  Σ_i VMR_i = 1, with VMR_H2 = 0.5*h_abd - Σ_(H-bearing,
    # other than H2) (n_H/2)*vmr_i, where the vmr_i values still need to be
    # multiplied by h_abd.  Algebra (see Fortran 440-456):
    #   h_abd * (0.5 + Σ_non-H vmr_tmp + Σ_H-bearing (1 - n_H/2) vmr_tmp) = 1
    sum_non_h         = 0.0
    sum_h_frac        = 0.0
    sum_h_bearing_h2eq = 0.0   # in "H2-equivalents" for the H2 closure

    try:
        idx_H2 = gas_id("H2")
    except ValueError:
        idx_H2 = -1

    for i in range(n_gases):
        if i == idx_H2:
            continue                     # H2 itself excluded
        if has_h[i]:
            n_H_i = int(elements_in_gases[i, iH])
            sum_h_frac        += (1.0 - n_H_i / 2.0) * vmr_tmp[i]
            sum_h_bearing_h2eq += (n_H_i / 2.0) * vmr_tmp[i]
        else:
            sum_non_h += vmr_tmp[i]

    denom = 0.5 + sum_non_h + sum_h_frac
    if denom <= 0:
        print(f"  Warning: chemistry init denom={denom:.3e}; setting h_abd=1.")
        h_abd = 1.0
    else:
        h_abd = 1.0 / denom

    if idx_H2 >= 0:
        vmr_tmp[idx_H2] = 0.5 - sum_h_bearing_h2eq
        if vmr_tmp[idx_H2] < 0:
            print(f"  Warning: chemistry init computed VMR(H2)={vmr_tmp[idx_H2]:.3e} "
                  f"< 0 — H-bearing species exceed H2 budget.  Clamping to 0.")
            vmr_tmp[idx_H2] = 0.0

    # Scale by h_abd
    vmr_tmp *= h_abd

    # ---------------- copy to all layers and update element abundances -----
    for ip in range(nlay):
        gases_vmr[:, ip] = vmr_tmp
        gas_element_abd[:, ip] = vmr_tmp @ elements_in_gases   # Σ_i VMR_i * n_X_i


def _update_gas_element_abd(
    vmr: np.ndarray,
    element_abd: np.ndarray,
    elements_in_gases: np.ndarray,
) -> None:
    """Recompute elemental abundances from the current VMR array."""
    element_abd[:] = 0.0
    for i in range(N_GASES):
        element_abd += vmr[i] * elements_in_gases[i]


def _calculate_time_constants(
    t: float,
    p_bar: float,
    scale_height_m: float,
    kzz: float,
    metallicity_c: float,
    metallicity_n: float,
    metallicity_o: float,
    at_equilibrium: bool,
) -> tuple[float, float, float, float, float]:
    """
    Quench mixing/chemical timescales — faithful port of the Fortran
    ``calculate_time_constants`` (chemistry.f90 478-496) plus the N2/HCN
    timescales from ``calculate_nh3_n2_hcn_equilibrium``.

    Zahnle & Marley (2014) expressions; ALL pressures in **bar**, scale
    height in **metres** (→ cm), Kzz in cm² s⁻¹.

    Returns
    -------
    (tmix, tchemc, tchemco2, tchemn, tchemhcn)   all in seconds
    """
    h_cm = scale_height_m * 1e2                       # m → cm (matches Fortran km×1e5)
    if at_equilibrium:
        tmix = 1.0e300                                # effectively infinite ⇒ no quench
    else:
        tmix = h_cm ** 2 / max(kzz, 1e-300)

    zco = 0.5 * (metallicity_c + metallicity_o)
    if not (zco > 0.0):
        zco = 1.0
    zn = metallicity_n if metallicity_n > 0.0 else 1.0
    zc = metallicity_c if metallicity_c > 0.0 else 1.0

    # CO–CH4: combined timescale tchemC = 1/(1/tq1 + 1/tq2)  (Zahnle & Marley Eqs. 11-13)
    tq1 = 1.5e-6 * _safe_exp(4.2e4 / t) * zco ** (-0.70) / p_bar
    tq2 = 4.0e1 * zco ** (-0.70) * _safe_exp(2.5e4 / t) / (p_bar ** 2)
    tchemc = 1.0 / (1.0 / tq1 + 1.0 / tq2)

    # CO–CO2 (Eq. 44), N2–NH3 (Eq. 32), HCN (Eq. with metallicity factor)
    tchemco2 = 1.0e-10 * _safe_exp(3.8e4 / t) / math.sqrt(p_bar)
    tchemn   = 1.0e-7 * _safe_exp(5.2e4 / t) / p_bar
    tchemhcn = 1.5e-4 * _safe_exp(3.6e4 / t) / p_bar / (zn ** 0.35 * zc ** 0.35)

    return tmix, tchemc, tchemco2, tchemn, tchemhcn


def _calculate_h2_h_equilibrium(
    t: float,
    p: float,
    vmr: np.ndarray,
    gases_delta_g_i: np.ndarray,
    elemental_h_ratio: np.ndarray,
) -> np.ndarray:
    """H₂ ⇌ 2H equilibrium with proper element conservation.

    The previous implementation used  qH = sqrt(qH2/K),  qH2_new = qH2 - qH/2.
    That is correct ONLY if there were no H atoms entering the layer.  But
    the layer-loop COPIES the previous (deeper) layer's full VMR vector
    before this routine runs, so qH(initial) is generally non-zero — and
    the old formula silently OVERWROTE qH (losing those atoms) while still
    subtracting qH/2 from qH2.  Iterating layer-by-layer leaked H atoms
    until both H2 and H went to zero, leaving an unphysical
    "He-only" atmosphere (mu → 4.4) above the bottom layer.

    The fix below solves the system properly: starting from total
    ``n_H = 2·qH2_in + qH_in`` (conserved by 2H ⇌ H₂), and the equilibrium
    constraint ``qH2 = K·qH²``, we get a quadratic
    ``2K·qH² + qH − n_H = 0`` whose positive root is

        qH_new  = 2 n_H / (1 + √(1 + 8 K n_H))

    written in the numerically-stable form that converges to ``n_H`` for
    K→0 and to ``√(n_H/2K)`` for K→∞.
    """
    idx_h2 = gas_id("H2")
    idx_h  = gas_id("H")

    dg_h  = gases_delta_g_i[idx_h]

    # 2H → H₂  :  ΔG = -2·g_H  (H₂ is reference, g_H₂ = 0)
    k_eq = equilibrium_constant_gases([-2, 1], [dg_h, 0.0],
                                      p * 1e-3, t)        # mbar → bar

    # Conserved total H atoms (per total atmosphere mole)
    total_h = 2.0 * vmr[idx_h2] + vmr[idx_h]

    K = max(k_eq, 1e-300)
    # Stable form (no catastrophic cancellation at small K):
    #   qH = 2·n_H / (1 + √(1 + 8 K n_H))
    qh = (2.0 * total_h) / (1.0 + math.sqrt(1.0 + 8.0 * K * total_h))
    qh2 = 0.5 * (total_h - qh)

    vmr[idx_h2] = max(qh2, 0.0)
    vmr[idx_h]  = max(qh,  0.0)
    return vmr


def _calculate_h2o_ch4_co_co2_sih4_sio(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, at_equilibrium, is_quenched,
    tchemc, tmix, pressures_layers, temperatures_layers
):
    """H₂O, CH₄, CO, CO₂, CH₃, SiO, SiH₄ equilibrium with C/O/Si conservation.

    r38: Port of the Fortran ``osi`` element-conservation form
    (chemistry.f90 lines 2490-2785).  Replaces the prior shortcut

        qch4 = k_co_ch4 * qH2^3 * qco

    which was deferred in HANDOFF_r37 as "doesn't blow up because total C
    is much larger than total Ti" -- but in the r37 test run CH4 reached
    VMR = 0.9995 at L79, 0.989 at L78, 0.873 at L77 because:

      * vmr[CO] was never reduced when CH4 formed (the formula above sets
        qch4 from qco but doesn't take qch4 out of qco),
      * the new qch4 was not capped at the elemental carbon budget,
      * at T = 350 K and p = 0.1 Pa the equilibrium constant for
        CO+3H2 -> CH4+H2O is huge, so qch4 = K * 1 * X_C >> 1,
      * the post-chemistry normalisation ``gases_vmr /= sum(gases_vmr)``
        (line ~470) then crushed H2 from 0.85 down to 4e-4 and pushed
        CH4 to 0.9995.

    The CH4-rich cap blocked OLR escape and caused the model to settle at
    T_eff ≈ 400 K instead of the target 500 K, with a permanent ~900 K
    hot bubble at ~10 Pa where heat was trapped under the opaque cap.

    Conservation form (Fortran ``osi`` cold-T branch, lines 2719-2761).
    All ratios assume qH2 ≈ 1 (H2-dominated atmosphere), consistent with
    every other simplified branch in this file.  K_eq values follow the
    Fortran convention:

      K_co   = K(H2O + CH4 -> CO + 3H2)    large at high T, small at low T
      K_ch3  = sqrt K(2CH4 -> 2CH3 + H2)   small everywhere
      K_co2  = K(CO + H2O -> CO2 + H2)
      K_sih4 = K(SiH4 + H2O -> SiO + 3H2)

    Equilibrium ratios:
      qCO/qCH4   = K_co  · qH2O
      qCH3/qCH4  = K_ch3
      qCO2/qCO   = K_co2 · qH2O
      qSiO/qSiH4 = K_sih4 · qH2O

    Element conservation closes the system:
      C-balance: qCH4 = X_C / (1 + K_ch3 + K_co · qH2O · (1 + K_co2 · qH2O))
      Si-balance: qSiH4 = X_Si / (1 + K_sih4 · qH2O)
      O-balance: qH2O = X_O - qCO - 2·qCO2 - qSiO

    qH2O appears on both sides → Picard iteration (≤8 sweeps, fully
    converged in practice by 3-4) with under-relaxation for stability.
    H2O is fixed to its saturation value when condensation occurs and
    the iteration is skipped.

    Asymptotic correctness:
      T → ∞  (K_co → ∞):  qCH4 → 0, qCO → X_C, qH2O → X_O - X_C
      T → 0  (K_co → 0):  qCH4 → X_C, qCO → 0, qH2O → X_O
    Both limits match physical chemistry.  The previous code had
    no T → 0 limit at all (qCH4 → ∞).
    """
    idx_h2o  = gas_id("H2O")
    idx_ch4  = gas_id("CH4")
    idx_co   = gas_id("CO")
    idx_co2  = gas_id("CO2")
    idx_ch3  = gas_id("CH3")
    # SiH4 / SiO are present in some species lists but not all; guard.
    try:
        idx_sih4 = gas_id("SiH4")
        idx_sio  = gas_id("SiO")
        have_si  = True
    except ValueError:
        idx_sih4 = idx_sio = -1
        have_si  = False

    dg_h2o  = gases_delta_g_i[idx_h2o]
    dg_ch4  = gases_delta_g_i[idx_ch4]
    dg_co   = gases_delta_g_i[idx_co]
    dg_co2  = gases_delta_g_i[idx_co2]
    dg_ch3  = gases_delta_g_i[idx_ch3]
    if have_si:
        dg_sih4 = gases_delta_g_i[idx_sih4]
        dg_sio  = gases_delta_g_i[idx_sio]

    p_bar = p * 1e-3   # mbar → bar

    # ---- Equilibrium constants in Fortran sign convention ------------
    # H2O + CH4 → CO + 3H2
    k_co = equilibrium_constant_gases([-1, -1, 1, 3],
                                      [dg_h2o, dg_ch4, dg_co, 0.0],
                                      p_bar, t)
    # 2CH4 → 2CH3 + H2  → sqrt to get qCH3/qCH4 ratio
    k_ch3_full = equilibrium_constant_gases([-2, 2, 1],
                                            [dg_ch4, dg_ch3, 0.0],
                                            p_bar, t)
    k_ch3 = math.sqrt(max(k_ch3_full, 0.0))
    # CO + H2O → CO2 + H2
    k_co2 = equilibrium_constant_gases([-1, -1, 1, 1],
                                       [dg_co, dg_h2o, dg_co2, 0.0],
                                       p_bar, t)
    # SiH4 + H2O → SiO + 3H2  (only if Si species present)
    if have_si:
        k_sih4 = equilibrium_constant_gases([-1, -1, 1, 3],
                                            [dg_sih4, dg_h2o, dg_sio, 0.0],
                                            p_bar, t)
    else:
        k_sih4 = 0.0

    # ---- Total element budgets (from CURRENT vmr) --------------------
    # These are the *conserved* totals; the chemistry redistributes
    # mass among species but cannot change the sums.
    x_C = vmr[idx_ch4] + vmr[idx_co] + vmr[idx_co2] + vmr[idx_ch3]
    if have_si:
        x_Si = vmr[idx_sio] + vmr[idx_sih4]
        x_O  = vmr[idx_h2o] + vmr[idx_co] + 2.0 * vmr[idx_co2] + vmr[idx_sio]
    else:
        x_Si = 0.0
        x_O  = vmr[idx_h2o] + vmr[idx_co] + 2.0 * vmr[idx_co2]

    # Numerical safety: floor very small / negative K (from extrapolated dG)
    if not math.isfinite(k_co)   or k_co   < 0: k_co   = 0.0
    if not math.isfinite(k_co2)  or k_co2  < 0: k_co2  = 0.0
    if not math.isfinite(k_ch3)  or k_ch3  < 0: k_ch3  = 0.0
    if not math.isfinite(k_sih4) or k_sih4 < 0: k_sih4 = 0.0

    # ---- H₂O condensation (compute saturation, possibly clamp qH2O) --
    qh2o = max(vmr[idx_h2o], 0.0)
    qsat_h2o = h2o_saturation_pressure(t) / p_bar
    vmr_sat[condensate_id("H2O")] = qsat_h2o
    h2o_saturated = (qh2o >= qsat_h2o)
    if h2o_saturated:
        if not is_condensed[condensate_id("H2O")]:
            is_condensed[condensate_id("H2O")] = True
            layer_cond[condensate_id("H2O")] = ip
        qh2o = qsat_h2o
        # When H2O condenses, x_O is no longer conserved here (oxygen
        # leaves the gas phase as ice).  Reduce x_O to match.
        # The simple closure: assume condensed H2O carries away
        # (vmr[H2O] - qsat) worth of O.
        x_O = qh2o + vmr[idx_co] + 2.0 * vmr[idx_co2] \
              + (vmr[idx_sio] if have_si else 0.0)

    # ---- Picard iteration to couple qH2O ↔ qCH4 ----------------------
    # When saturated, qH2O is fixed and one pass suffices.
    n_iter = 1 if h2o_saturated else 8
    if x_C <= 0.0:
        # No carbon: solve only Si and trivial CH4=CO=CO2=CH3=0
        qch4 = qco = qco2 = qch3 = 0.0
        # Si budget still updates qH2O via SiO
        for _ in range(n_iter):
            denom_si = 1.0 + k_sih4 * qh2o
            qsih4 = x_Si / denom_si if (x_Si > 0 and have_si) else 0.0
            qsio  = k_sih4 * qh2o * qsih4 if have_si else 0.0
            if h2o_saturated:
                break
            qh2o_new = max(x_O - qsio, x_O * 1e-30)
            qh2o = 0.5 * (qh2o + qh2o_new)
    else:
        prev = -1.0
        for _ in range(n_iter):
            denom_c = 1.0 + k_ch3 + k_co * qh2o * (1.0 + k_co2 * qh2o)
            if not math.isfinite(denom_c) or denom_c <= 0.0:
                denom_c = 1.0
            qch4 = x_C / denom_c
            qco  = k_co * qh2o * qch4
            qco2 = k_co2 * qh2o * qco
            qch3 = k_ch3 * qch4
            if have_si:
                denom_si = 1.0 + k_sih4 * qh2o
                qsih4 = x_Si / denom_si if x_Si > 0 else 0.0
                qsio  = k_sih4 * qh2o * qsih4
            else:
                qsih4 = qsio = 0.0

            if h2o_saturated:
                break

            qh2o_new = x_O - qco - 2.0 * qco2 - qsio
            if qh2o_new <= 0.0:
                # Overshot: halve and retry next sweep
                qh2o_new = 0.5 * qh2o
            elif qh2o_new > x_O:
                qh2o_new = x_O

            # Under-relaxation for stability; converges in 3-4 sweeps
            # for typical brown-dwarf conditions
            qh2o = 0.5 * (qh2o + qh2o_new)

            if abs(qh2o - prev) < 1e-12 * max(qh2o, 1e-30):
                break
            prev = qh2o

    # ---- Write back, clipping negatives from numerical roundoff ------
    vmr[idx_h2o] = max(qh2o, 0.0)
    vmr[idx_ch4] = max(qch4, 0.0)
    vmr[idx_co]  = max(qco,  0.0)
    vmr[idx_co2] = max(qco2, 0.0)
    vmr[idx_ch3] = max(qch3, 0.0)
    if have_si:
        vmr[idx_sih4] = max(qsih4, 0.0)
        vmr[idx_sio]  = max(qsio,  0.0)

    return vmr, is_condensed, vmr_sat, layer_cond


# ---------------------------------------------------------------------------
#  Interpolated-quench helpers (Fortran get_pold_told + the log-interpolation
#  in co_ch4_quenching / co_co2_quenching).  At the crossing where a chemical
#  timescale overtakes the mixing timescale, the Fortran does NOT capture the
#  quenched abundances at the grid layer; it interpolates the true (T,P) of the
#  crossing and re-solves the equilibrium there.  The grid layer sits ~40-50 K
#  COLDER than that crossing, where equilibrium has already shifted toward CH4,
#  so capturing at the grid layer freezes CO+CO2 (hence CO2) too low.
# ---------------------------------------------------------------------------
def _get_pold_told(ip, temperatures_layers, pressures_layers):
    """Previous-layer T (K) and P (bar); at ip==0 the Fortran seed: a virtual
    deeper layer at 1.198·T and 2·P (Fortran get_pold_told, chemistry.f90 1581).
    pressures_layers is in Pa here, so bar = ·1e-5."""
    if ip == 0:
        told = temperatures_layers[0] * 1.198
        pold = 2.0 * pressures_layers[0] * 1e-5
    else:
        told = temperatures_layers[ip - 1]
        pold = pressures_layers[ip - 1] * 1e-5
    return told, pold


def _quench_interp_pt(told, pold, t, p_bar, tchem1, tmix1, tchem, tmix):
    """Log-interpolate the quench (T[K], P[bar]) between the previous layer
    (subscript 1) and the current layer (Fortran co_ch4/co_co2_quenching):
        x = log(tchem1/tmix1) / log(tchem1·tmix / (tchem·tmix1))
        Tq = told·(t/told)^x ,  Pq = pold·(p/pold)^x .
    x is clamped to [0,1] for safety (it is analytically in-range at a genuine
    crossing where tchem1<tmix1 and tchem>tmix)."""
    denom = math.log(tchem1 * tmix / (tchem * tmix1))
    if denom == 0.0 or not math.isfinite(denom):
        return t, p_bar, 0.0
    x = math.log(tchem1 / tmix1) / denom
    if not math.isfinite(x):
        return t, p_bar, 0.0
    x = min(1.0, max(0.0, x))
    tq = told * (t / told) ** x
    pq = pold * (p_bar / pold) ** x
    return tq, pq, x


def _osiqco(t, p, ip, vmr, gases_delta_g_i, gas_element_abd_ip, qcoco2,
            h2o_saturated):
    """CO quenched: CO+CO₂ frozen at total ``qcoco2``; re-solve H₂O (O-balance),
    split CO/CO₂ by the *local* equilibrium, and derive SiO/SiH₄.

    Faithful port of the Fortran ``osiqco`` (chemistry.f90 2787-2871), written
    in this module's convention (qH₂≈1, ``p_bar = p·1e-3``).  H₂O is re-solved
    from oxygen conservation ``qH2O = X_O − qcoco2 − qSiO`` only when water is
    NOT condensing; when saturated, H₂O stays at its saturation value (set by
    the caller) and we simply distribute the frozen carbon over it.
    """
    iH2O = gas_id("H2O"); iCO = gas_id("CO"); iCO2 = gas_id("CO2")
    try:
        iSiO = gas_id("SiO"); iSiH4 = gas_id("SiH4"); have_si = True
    except ValueError:
        have_si = False

    O  = float(gas_element_abd_ip[7])                 # O  (Z=8 → idx 7)
    Si = float(gas_element_abd_ip[13]) if have_si else 0.0   # Si (Z=14 → idx 13)
    # bar = p·1e-2 (= pressures_layers·1e-5), matching _equil_co_si_o and the
    # Fortran (whose osiqco receives k's built at the layer's bar pressure).
    # The previous p·1e-3 was bar/10, the same unit bug fixed in NH3/HCN: it
    # made k_sih4 (∝ P⁻²) ~100× too large → SiO O-sink too deep → H2O (hence
    # CO2 = CO·H2O·k_co2/H2) biased low in this quench region.
    p_bar = p * 1e-2

    dg_h2o = gases_delta_g_i[iH2O]; dg_co = gases_delta_g_i[iCO]
    dg_co2 = gases_delta_g_i[iCO2]
    # CO + H2O → CO2 + H2 ;  SiH4 + H2O → SiO + 3H2   (qH2≈1)
    k_co2 = equilibrium_constant_gases([-1, -1, 1, 1],
                                       [dg_co, dg_h2o, dg_co2, 0.0], p_bar, t)
    if have_si:
        dg_sio = gases_delta_g_i[iSiO]; dg_sih4 = gases_delta_g_i[iSiH4]
        k_sih4 = equilibrium_constant_gases([-1, -1, 1, 3],
                                            [dg_sih4, dg_h2o, dg_sio, 0.0], p_bar, t)
    else:
        k_sih4 = 0.0
    if not math.isfinite(k_co2)  or k_co2  < 0: k_co2  = 0.0
    if not math.isfinite(k_sih4) or k_sih4 < 0: k_sih4 = 0.0

    if not h2o_saturated:
        # initial guess (Fortran: branch on k_sih4·(Si−O+qcoco2) vs 1)
        denom0 = k_sih4 * (Si - O + qcoco2)
        if denom0 <= 1.0:
            vmr[iH2O] = O - qcoco2
        else:
            vmr[iH2O] = (O - qcoco2) / denom0
        for _ in range(200):
            prev = vmr[iH2O]
            h2o = vmr[iH2O] if vmr[iH2O] > 0.0 else 0.0
            si_term = (k_sih4 * Si / (1.0 + k_sih4 * h2o)) if have_si else 0.0
            vmr[iH2O] = (O - qcoco2) / (1.0 + si_term
                                        + k_co2 * qcoco2 / (1.0 + k_co2 * h2o))
            if abs(vmr[iH2O] - prev) <= 1e-9 * max(abs(vmr[iH2O]), 1e-300):
                break

    h2o = max(vmr[iH2O], 0.0)
    vmr[iCO]  = qcoco2 / (1.0 + k_co2 * h2o)
    vmr[iCO2] = vmr[iCO] * h2o * k_co2
    if vmr[iH2O] < 0.0:
        # H2O drove negative: sacrifice strict O conservation (Fortran fallback)
        vmr[iH2O] = max(O - (vmr[iCO] + vmr[iCO2]), 1e-300)
        h2o = vmr[iH2O]
    if have_si:
        vmr[iSiO]  = h2o * Si * k_sih4 / (1.0 + k_sih4 * h2o)
        vmr[iSiH4] = (vmr[iSiO] / (k_sih4 * h2o)) if (k_sih4 > 0.0 and h2o > 0.0) else 0.0
    return vmr


def _osiqcoco2(t, p, ip, vmr, gases_delta_g_i, gas_element_abd_ip,
               h2o_saturated):
    """CO *and* CO₂ both quenched (frozen at their carried-forward values);
    re-solve H₂O from oxygen conservation and derive SiO/SiH₄.

    Faithful port of the Fortran ``osiqcoco2`` (chemistry.f90 2873-2942).
    """
    iH2O = gas_id("H2O"); iCO = gas_id("CO"); iCO2 = gas_id("CO2")
    try:
        iSiO = gas_id("SiO"); iSiH4 = gas_id("SiH4"); have_si = True
    except ValueError:
        have_si = False

    O  = float(gas_element_abd_ip[7])
    Si = float(gas_element_abd_ip[13]) if have_si else 0.0
    p_bar = p * 1e-2                  # bar (was p·1e-3 = bar/10; see _osiqco note)
    co  = max(vmr[iCO], 0.0)
    co2 = max(vmr[iCO2], 0.0)

    dg_h2o = gases_delta_g_i[iH2O]
    if have_si:
        dg_sio = gases_delta_g_i[iSiO]; dg_sih4 = gases_delta_g_i[iSiH4]
        k_sih4 = equilibrium_constant_gases([-1, -1, 1, 3],
                                            [dg_sih4, dg_h2o, dg_sio, 0.0], p_bar, t)
    else:
        k_sih4 = 0.0
    if not math.isfinite(k_sih4) or k_sih4 < 0: k_sih4 = 0.0

    if not h2o_saturated:
        o_free = O - co - 2.0 * co2
        denom0 = k_sih4 * (Si - O + co + 2.0 * co2)
        if denom0 <= 1.0:
            vmr[iH2O] = o_free
        else:
            vmr[iH2O] = o_free / denom0
        for _ in range(200):
            prev = vmr[iH2O]
            h2o = vmr[iH2O] if vmr[iH2O] > 0.0 else 0.0
            si_term = (k_sih4 * Si / (1.0 + k_sih4 * h2o)) if have_si else 0.0
            vmr[iH2O] = o_free / (1.0 + si_term)
            if abs(vmr[iH2O] - prev) <= 1e-9 * max(abs(vmr[iH2O]), 1e-300):
                break

    h2o = max(vmr[iH2O], 0.0)
    if have_si:
        vmr[iSiO]  = h2o * Si * k_sih4 / (1.0 + k_sih4 * h2o)
        vmr[iSiH4] = (vmr[iSiO] / (k_sih4 * h2o)) if (k_sih4 > 0.0 and h2o > 0.0) else 0.0
    return vmr


# ===========================================================================
#  FULL osi C/O/Si EQUILIBRIUM SOLVER  (faithful port of chemistry.f90 osi,
#  k_eq, calculate_h2_h_equilibrium).  Conventions matched to the Fortran:
#    * pressure in BAR  (p_bar = p * 1e-2 = pressures_layers * 1e-5)
#    * k_eq = equilibrium_constant_gases(...) / qH2^{H2 stoichiometric coeff}
#    * solver tolerances  PREC_HIGH=1e-15, PREC_LOW=1e-6, I_MAX=100
#  This replaces the simplified C/O/Si equilibrium for layers below the quench.
# ===========================================================================
_PREC_HIGH = 1.0e-15      # = 10**(-precision(0d0))  (double precision)
_PREC_LOW  = 1.0e-6       # = 10**(-precision(0.))    (single precision)
_I_MAX     = 100

# --- Mg/Si/O silicate O-sink (forsterite/enstatite/SiO2) ------------------
# Master switch for the faithful silicate condensation O-sink (Fortran
# calculate_mg_si_o_equilibrium).  Set False to fall back to the simplified
# enstatite-only stub `_calculate_mg_si_o_simple` (the pre-sink "3b" state).
_USE_MG_SI_O_SINK = True
_TINY    = 2.2250738585072014e-308   # Fortran tiny(0d0)  (smallest normal f64)
_HUGE_SP = 3.4028234663852886e+38    # Fortran huge(0.)   (single-precision max)


def _keq(species, stoich, gases_delta_g_i, p_bar, t, qh2):
    """Fortran ``k_eq`` wrapper: equilibrium_constant_gases / qH2^{H2 stoich}."""
    idx = [gas_id(s) for s in species]
    dg  = [gases_delta_g_i[i] for i in idx]
    k = equilibrium_constant_gases(stoich, dg, p_bar, t)
    ih2 = gas_id("H2")
    for i, s in zip(idx, stoich):
        if i == ih2 and qh2 > 0.0:
            k = k / (qh2 ** s)
    return k


def _h2_h_eq(vmr, gas_element_abd_ip, elements_in_gases, gases_delta_g_i, p_bar, t):
    """H2 <-> 2H equilibrium with full element conservation.

    Faithful port of the Fortran ``calculate_h2_h_equilibrium``:
        k_eq_h = k_eq(['H2','H'],[-1,2]) / qH2
        sum_h  = X_H - (H in species other than H2/H)
        H      = -k/4 + sqrt(k*sum_h/2 + k^2/16)
        H2     = H^2 / k
    """
    ih2 = gas_id("H2"); ih = gas_id("H")
    qh2 = vmr[ih2] if vmr[ih2] > 0.0 else 1e-300
    k_h = _keq(["H2", "H"], [-1, 2], gases_delta_g_i, p_bar, t, qh2)
    if not math.isfinite(k_h) or k_h <= 0.0:
        return vmr
    # H locked in all gas species, then everything except H2/H
    sum_h = float(np.dot(elements_in_gases[:, 0], vmr))
    sum_h = sum_h - 2.0 * vmr[ih2] - vmr[ih]
    sum_h = gas_element_abd_ip[0] - sum_h
    if sum_h < 0.0:
        sum_h = gas_element_abd_ip[0]
    disc = k_h * sum_h / 2.0 + k_h ** 2 / 16.0
    vmr[ih]  = -k_h / 4.0 + math.sqrt(disc) if disc > 0 else 0.0
    vmr[ih2] = vmr[ih] ** 2 / k_h
    return vmr


def _osi(vmr, gas_element_abd_ip, elements_in_gases, gases_delta_g_i,
         k_co, k_ch3, k_co2, k_sih4, p_bar, t, h2o_saturated):
    """Coupled C/O/Si gas equilibrium — faithful port of the Fortran ``osi``
    (chemistry.f90 2490-2785).  Four branches (CO- vs CH4-dominated × Si-oxide
    balance) plus the H2O-saturated case, each with the high-metallicity
    divergence damping.  ``k_co/k_ch3/k_co2/k_sih4`` are updated in place each
    iteration via the Fortran ``k_eq`` (qH2 division, bar pressure).
    """
    iH2O = gas_id("H2O"); iCO = gas_id("CO"); iCO2 = gas_id("CO2")
    iCH4 = gas_id("CH4"); iCH3 = gas_id("CH3"); iSiO = gas_id("SiO")
    iSiH4 = gas_id("SiH4"); iH2 = gas_id("H2")
    O  = gas_element_abd_ip[7]    # O  (Z=8)
    C  = gas_element_abd_ip[5]    # C  (Z=6)
    Si = gas_element_abd_ip[13]   # Si (Z=14)
    lp = math.log(_PREC_LOW)

    def recompute_k():
        qh2 = vmr[iH2] if vmr[iH2] > 0 else 1e-300
        kc  = _keq(["H2O", "CH4", "CO", "H2"], [-1, -1, 1, 3], gases_delta_g_i, p_bar, t, qh2)
        kc3 = math.sqrt(max(_keq(["CH4", "CH3", "H2"], [-2, 2, 1], gases_delta_g_i, p_bar, t, qh2), 0.0))
        kc2 = _keq(["CO", "H2O", "CO2", "H2"], [-1, -1, 1, 1], gases_delta_g_i, p_bar, t, qh2)
        ks  = _keq(["SiH4", "H2O", "SiO", "H2"], [-1, -1, 1, 3], gases_delta_g_i, p_bar, t, qh2)
        return kc, kc3, kc2, ks

    def h2h():
        _h2_h_eq(vmr, gas_element_abd_ip, elements_in_gases, gases_delta_g_i, p_bar, t)

    if not h2o_saturated:
        if k_co * (O - C - Si) >= 1.0:
            # ---------- branch A : CO-dominated ----------
            if k_sih4 * (O - Si) - k_co * C >= 1.0:
                # --- A1 ---
                vmr[iH2O] = O - C - Si
                i = 0; d_sum = 1e300
                while d_sum > _PREC_HIGH and i < _I_MAX:
                    i += 1
                    vmr_tmp = vmr[iH2O]; h2_vmr_tmp = vmr[iH2O]
                    h2o = vmr[iH2O]
                    vmr[iCO]   = h2o * C * k_co / (1.0 + k_ch3 + k_co * h2o * (1.0 + k_co2 * h2o))
                    vmr[iCO2]  = vmr[iCO] * h2o * k_co2
                    vmr[iSiO]  = h2o * Si * k_sih4 / (1.0 + k_sih4 * h2o)
                    vmr[iSiH4] = vmr[iSiO] / (k_sih4 * h2o) if (k_sih4 * h2o) > 0 else 0.0
                    vmr[iH2O]  = O - float(np.dot(elements_in_gases[:, 7], vmr)) + vmr[iH2O]
                    if ((d_sum < abs(vmr[iH2O] - vmr_tmp) and i > 5) or i > 30
                            or vmr[iH2O] <= 0.0):
                        if vmr[iH2O] <= 0.0:
                            vmr[iH2O] = vmr_tmp * (1.0 - math.exp(lp / _I_MAX * i))
                        elif vmr[iH2O] >= 1.0:
                            vmr[iH2O] = vmr_tmp * (1.0 + math.exp(lp / _I_MAX * i))
                        else:
                            if vmr[iH2O] > vmr_tmp:
                                vmr[iH2O] = vmr_tmp * (1.0 + math.exp(lp / _I_MAX * (i - 5)))
                            else:
                                vmr[iH2O] = vmr_tmp * (1.0 - math.exp(lp / _I_MAX * (i - 5)))
                    h2h()
                    if vmr[iH2] <= 0.0 or vmr[iH2] >= 1.0:
                        if vmr[iH2] <= 0.0:
                            vmr[iH2] = h2_vmr_tmp * (1.0 - math.exp(lp / _I_MAX * i))
                        else:
                            vmr[iH2] = h2_vmr_tmp * (1.0 + math.exp(lp / _I_MAX * i))
                    k_co, k_ch3, k_co2, k_sih4 = recompute_k()
                    d_sum = abs(vmr[iH2O] - vmr_tmp)
                vmr[iCH4] = vmr[iCO] / (k_co * vmr[iH2O]) if (k_co * vmr[iH2O]) > 0 else 0.0
                h2h()
            else:
                # --- A2 ---
                vmr[iH2O] = (O - C) / (1.0 + k_sih4 * Si)
                i = 0; d_sum = 1e300
                while d_sum > _PREC_HIGH and i < _I_MAX:
                    i += 1
                    vmr_tmp = vmr[iH2O]; h2o = vmr[iH2O]
                    vmr[iSiH4] = Si / (1.0 + k_sih4 * h2o)
                    vmr[iCO]   = h2o * C * k_co / (1.0 + k_ch3 + k_co * h2o * (1.0 + k_co2 * h2o))
                    vmr[iCO2]  = vmr[iCO] * h2o * k_co2
                    vmr[iH2O]  = (O - vmr[iCO] * (1.0 + 2.0 * k_co2 * h2o)) / (1.0 + k_sih4 * vmr[iSiH4])
                    if ((d_sum < abs(vmr[iH2O] - vmr_tmp) and i > 5) or i > 30):
                        if vmr[iH2O] > vmr_tmp:
                            vmr[iH2O] = vmr_tmp * (1.0 + math.exp(lp / _I_MAX * (i - 5)))
                        else:
                            vmr[iH2O] = vmr_tmp * (1.0 - math.exp(lp / _I_MAX * (i - 5)))
                    h2h()
                    k_co, k_ch3, k_co2, k_sih4 = recompute_k()
                    d_sum = abs(vmr[iH2O] - vmr_tmp)
                vmr[iSiO] = vmr[iH2O] * k_sih4 * vmr[iSiH4]
                vmr[iCH4] = vmr[iCO] / (k_co * vmr[iH2O]) if (k_co * vmr[iH2O]) > 0 else 0.0
                h2h()
        else:
            # ---------- branch B : CH4-dominated ----------
            if k_sih4 * (O - Si) - k_co * C >= 1.0:
                # --- B1 ---  (converges on H2)
                vmr[iH2O] = (O - Si) / (1.0 + k_co * C)
                i = 0; d_sum = 1e300
                while d_sum > _PREC_HIGH and i < _I_MAX:
                    i += 1
                    vmr_tmp = vmr[iH2]; h2o = vmr[iH2O]
                    vmr[iCH4] = C / (1.0 + k_ch3 + k_co * h2o * (1.0 + k_co2 * h2o))
                    vmr[iCH3] = vmr[iCH4] * k_ch3
                    vmr[iSiO] = h2o * Si * k_sih4 / (1.0 + k_sih4 * h2o)
                    vmr[iSiH4] = vmr[iSiO] / (k_sih4 * h2o) if (k_sih4 * h2o) > 0 else 0.0
                    vmr[iH2O] = (O - vmr[iSiO]) / (1.0 + k_co * vmr[iCH4] * (1.0 + 2.0 * k_co2 * h2o))
                    h2h()
                    if ((d_sum < abs(vmr[iH2] - vmr_tmp) and i > 5) or i > 30
                            or vmr[iH2] <= 0.0 or vmr[iH2] >= 1.0):
                        if vmr[iH2] <= 0.0:
                            vmr[iH2] = vmr_tmp * (1.0 - math.exp(lp / _I_MAX * i))
                        elif vmr[iH2] >= 1.0:
                            vmr[iH2] = vmr_tmp * (1.0 + math.exp(lp / _I_MAX * i))
                        else:
                            if vmr[iH2] > vmr_tmp:
                                vmr[iH2] = vmr_tmp * (1.0 + math.exp(lp / _I_MAX * (i - 5)))
                            else:
                                vmr[iH2] = vmr_tmp * (1.0 - math.exp(lp / _I_MAX * (i - 5)))
                    k_co, k_ch3, k_co2, k_sih4 = recompute_k()
                    d_sum = abs(vmr[iH2] - vmr_tmp)
                vmr[iCO]  = vmr[iH2O] * k_co * vmr[iCH4]
                vmr[iCO2] = vmr[iCO] * vmr[iH2O] * k_co2
                h2h()
            else:
                # --- B2 ---
                vmr[iH2O] = O / (1.0 + k_co * C + k_sih4 * Si)
                i = 0; d_sum = 1e300
                while d_sum > _PREC_HIGH and i < _I_MAX:
                    i += 1
                    vmr_tmp = vmr[iH2O]; h2o = vmr[iH2O]
                    vmr[iSiH4] = Si / (1.0 + k_sih4 * h2o)
                    vmr[iCH4]  = C / (1.0 + k_ch3 + k_co * h2o * (1.0 + k_co2 * h2o))
                    vmr[iCH3]  = k_ch3 * vmr[iCH4]
                    vmr[iH2O]  = O / (1.0 + k_co * vmr[iCH4] * (1.0 + 2.0 * k_co2 * h2o)
                                      + k_sih4 * vmr[iSiH4])
                    if ((d_sum < abs(vmr[iH2O] - vmr_tmp) and i > 5) or i > 30):
                        if vmr[iH2O] > vmr_tmp:
                            vmr[iH2O] = vmr_tmp * (1.0 + math.exp(lp / _I_MAX * (i - 5)))
                        else:
                            vmr[iH2O] = vmr_tmp * (1.0 - math.exp(lp / _I_MAX * (i - 5)))
                    h2h()
                    k_co, k_ch3, k_co2, k_sih4 = recompute_k()
                    d_sum = abs(vmr[iH2O] - vmr_tmp)
                vmr[iCO]  = vmr[iH2O] * k_co * vmr[iCH4]
                vmr[iCO2] = vmr[iCO] * vmr[iH2O] * k_co2
                vmr[iSiO] = vmr[iH2O] * k_sih4 * vmr[iSiH4]
                if vmr[iH2O] < 0.0:
                    vmr[iH2O] = O / (1.0 + k_co * vmr[iCH4] * (1.0 + 2.0 * k_co2 * vmr[iH2O])
                                     + k_sih4 * vmr[iSiH4])
                h2h()
    else:
        # ---------- H2O saturated : distribute C/Si over the fixed H2O ----
        h2o = vmr[iH2O]
        vmr[iCO]  = h2o * C * k_co / (1.0 + k_ch3 + k_co * h2o * (1.0 + k_co2 * h2o))
        vmr[iCO2] = vmr[iCO] * h2o * k_co2
        vmr[iSiO] = h2o * Si * k_sih4 / (1.0 + k_sih4 * h2o)

    # numerical safety: no negative / NaN carbon-silicon VMRs
    for _ix in (iH2O, iCO, iCO2, iCH4, iCH3, iSiO, iSiH4):
        if (not math.isfinite(vmr[_ix])) or vmr[_ix] < 0.0:
            vmr[_ix] = 1e-300
    return vmr, k_co, k_ch3, k_co2, k_sih4


def _equil_co_si_o(t, p, ip, vmr, gases_delta_g_i, is_condensed, vmr_sat,
                   layer_cond, gas_element_abd_ip, elements_in_gases):
    """Equilibrium path of the Fortran ``calculate_h2o_ch4_co_co2_sih4_sio_equilibirum``:
    build the four C/O/Si equilibrium constants (qH2 division, bar pressure),
    seed CH4 at the bottom layer, apply H2O condensation, then call ``osi``.
    Used for layers below the CO/CH4 quench level (tchemC <= tmix)."""
    iH2O = gas_id("H2O"); iCH4 = gas_id("CH4"); iCH3 = gas_id("CH3")
    iH2 = gas_id("H2")
    p_bar = p * 1e-2                       # Pa·1e-3 → ·1e-2 = pressures·1e-5 = bar
    qh2 = vmr[iH2] if vmr[iH2] > 0 else 1e-300

    k_co  = _keq(["H2O", "CH4", "CO", "H2"], [-1, -1, 1, 3], gases_delta_g_i, p_bar, t, qh2)
    k_ch3 = math.sqrt(max(_keq(["CH4", "CH3", "H2"], [-2, 2, 1], gases_delta_g_i, p_bar, t, qh2), 0.0))
    k_co2 = _keq(["CO", "H2O", "CO2", "H2"], [-1, -1, 1, 1], gases_delta_g_i, p_bar, t, qh2)
    k_sih4 = _keq(["SiH4", "H2O", "SiO", "H2"], [-1, -1, 1, 3], gases_delta_g_i, p_bar, t, qh2)

    if ip == 0:
        # initial CH4 guess (Fortran seeds qch4q/(1+k_ch3); qch4q≈X_C at the base)
        vmr[iCH4] = gas_element_abd_ip[5] / (1.0 + k_ch3) if (1.0 + k_ch3) > 0 else gas_element_abd_ip[5]
        vmr[iCH3] = 1e-300

    # H2O condensation (cap at saturation; flag for osi's saturated branch)
    qsath2o = h2o_saturation_pressure(t) / p_bar
    vmr_sat[condensate_id("H2O")] = qsath2o
    h2o_sat = vmr[iH2O] > qsath2o
    if h2o_sat:
        if not is_condensed[condensate_id("H2O")]:
            is_condensed[condensate_id("H2O")] = True
            layer_cond[condensate_id("H2O")] = ip
        vmr[iH2O] = qsath2o

    vmr, _kc, _kc3, _kc2, _ks = _osi(
        vmr, gas_element_abd_ip, elements_in_gases, gases_delta_g_i,
        k_co, k_ch3, k_co2, k_sih4, p_bar, t, h2o_sat)
    return vmr, is_condensed, vmr_sat, layer_cond


def _calculate_nh3_n2_hcn(
    t, p, ip, vmr, gases_delta_g_i, at_equilibrium, is_quenched, tchemn, tmix
):
    """NH₃ ⇌ N₂ and HCN equilibria with nitrogen conservation.

    r38: Replaces the prior shortcut

        qnh3 = sqrt(K * qN2 * qH2^3)

    which never reduced qN2 nor capped at the elemental N budget.  Same
    bug pattern as the CH4 issue: in cold layers K_eq grows large,
    qNH3 = sqrt(K * X_N) can exceed X_N, and the subsequent normalisation
    crushes other species.  At brown-dwarf TOA conditions the effect is
    smaller than CH4 (because X_N < X_C and the sqrt softens it) but
    still drives µ ~ 0.5 g/mol error in upper layers.

    Conservation form (Fortran ``calculate_nh3_n2_hcn_equilibrium``,
    chemistry.f90 lines 644-656):

      Equilibrium: N2 + 3H2 → 2NH3   K_nh3 = qNH3² / (qN2 · qH2³) ≈ qNH3² / qN2
      Element:     X_N = 2·qN2 + qNH3

    Substituting qN2 = (X_N - qNH3)/2 gives the quadratic
      2·qNH3² + K_nh3·qNH3 − K_nh3·X_N = 0
    with positive root
      qNH3 = [-K_nh3 + sqrt(K_nh3² + 8·K_nh3·X_N)] / 4
      qN2  = (X_N - qNH3) / 2

    Asymptotics:
      K_nh3 → 0  (hot):  qNH3 → sqrt(K_nh3 · X_N / 2),  qN2 → X_N/2
      K_nh3 → ∞  (cold): qNH3 → X_N,                    qN2 → 0

    HCN from N2 (Fortran line 701):
      qHCN = qCH4 · sqrt(K_hcn_bare · qN2)
    where K_hcn_bare is the equilibrium constant for 2CH4 + N2 → 2HCN + 3H2
    (without the qCH4² prefactor that the Fortran rolls in).  The HCN
    typically removes a small fraction of N from N2; we cap qHCN ≤ 0.5·X_N
    to avoid double-counting near solver edge cases.
    """
    idx_nh3 = gas_id("NH3")
    idx_n2  = gas_id("N2")
    idx_hcn = gas_id("HCN")
    idx_ch4 = gas_id("CH4")
    idx_h2  = gas_id("H2")

    dg_nh3 = gases_delta_g_i[idx_nh3]
    dg_n2  = gases_delta_g_i[idx_n2]
    dg_hcn = gases_delta_g_i[idx_hcn]
    dg_ch4 = gases_delta_g_i[idx_ch4]

    # Pressure in BAR.  `p = pressures_layers·1e-3`, so bar = p·1e-2
    # (= pressures_layers·1e-5).  The prior `p*1e-3` was bar/10 — the legacy
    # N/metal-routine unit bug.  k_eq_nh3 carries P² and k_eq_hcn carries P⁻²,
    # so the 10× pressure error pushed NH3 ~1 dex LOW and HCN ~1 dex HIGH.
    p_bar = p * 1e-2
    qh2 = max(vmr[idx_h2], 1e-30)        # H2 mole fraction (~0.85), tracked

    # K(N2 + 3H2 → 2NH3).  The Fortran k_eq divides equilibrium_constant_gases by
    # h2_factor = qH2^(stoich_H2) = qH2^(-3), i.e. multiplies by qH2³.  Omitting
    # it (the qH2≈1 shortcut) left NH3 ~0.2 dex high; with the bar fix the two
    # together resolve the −0.85 dex offset.
    k_nh3 = equilibrium_constant_gases([-1, -3, 2],
                                       [dg_n2, 0.0, dg_nh3],
                                       p_bar, t) * qh2 ** 3
    if not math.isfinite(k_nh3) or k_nh3 < 0:
        k_nh3 = 0.0

    # ---- Conserved nitrogen budget --------------------------------------
    # Sum atoms of N currently in N2 (×2) and NH3 (×1) plus any pre-existing
    # HCN.  HCN will be re-derived below; subtract it from x_N before
    # solving N2/NH3 to avoid double-counting.
    x_N = 2.0 * vmr[idx_n2] + vmr[idx_nh3] + vmr[idx_hcn]

    if x_N > 0.0 and k_nh3 > 0.0:
        # Solve  2·qNH3² + K·qNH3 − K·X_N = 0   (positive root only)
        # Use stable form for very small K (avoid 0 - 0 cancellation):
        disc = k_nh3 * k_nh3 + 8.0 * k_nh3 * x_N
        if disc < 0.0:
            disc = 0.0
        qnh3 = (math.sqrt(disc) - k_nh3) * 0.25
        # Numerical safety: clamp to physical range
        qnh3 = max(0.0, min(qnh3, x_N))
        qn2  = max(0.0, 0.5 * (x_N - qnh3))
    elif x_N > 0.0:
        # K_nh3 = 0 → no NH3 forms
        qnh3 = 0.0
        qn2  = 0.5 * x_N
    else:
        qnh3 = 0.0
        qn2  = 0.0

    vmr[idx_nh3] = qnh3
    vmr[idx_n2]  = qn2

    # ---- HCN from CH4 + N2 ---------------------------------------------
    # K_hcn for 2CH4 + N2 → 2HCN + 3H2.  H2 stoich = +3 ⇒ h2_factor = qH2³ ⇒
    # divide by qH2³ (Fortran k_eq).  p_power = -2, so the corrected bar drops
    # HCN back ~1 dex toward the Fortran (it was ~1 dex high from bar/10).
    k_hcn_bare = (equilibrium_constant_gases([-2, -1, 2, 3],
                                             [dg_ch4, dg_n2, dg_hcn, 0.0],
                                             p_bar, t) / qh2 ** 3)
    if math.isfinite(k_hcn_bare) and k_hcn_bare > 0 and qn2 > 0:
        qch4 = vmr[idx_ch4]
        # qHCN² = K · qCH4² · qN2   (qH2³ already folded into k_hcn_bare)
        qhcn_sq = k_hcn_bare * qch4 * qch4 * qn2
        qhcn = math.sqrt(max(qhcn_sq, 0.0))
        # Cap at 0.5·X_N to keep N conservation reasonable (Fortran
        # warns and limits if HCN dominates; we just clip silently
        # because the layer loop will re-balance).
        max_hcn = 0.5 * x_N
        if qhcn > max_hcn:
            qhcn = max_hcn
        # Remove the N taken by HCN from N2 (each HCN has 1 N).
        # Without this step the post-chemistry normalisation re-introduces
        # a small inconsistency.
        n_taken = min(qhcn, qn2)
        vmr[idx_n2] = max(qn2 - n_taken, 0.0)
        vmr[idx_hcn] = qhcn
    else:
        vmr[idx_hcn] = 0.0

    return vmr


def _calculate_cl_na_k(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, pressures_layers, temperatures_layers,
    gas_element_abd
):
    """Cl/Na/K gas equilibrium + Na₂S and KCl condensation.

    Faithful (focused) port of the Fortran ``calculate_cl_na_k_equilibrium``
    (chemistry.f90 718-975).  The previous Python stub only capped gaseous
    KCl and never depleted atomic Na or K, so both stayed flat
    (Na≈3.2e-5, K≈2e-6) through the photosphere — a ~67/83 dex error and the
    spurious Na/K resonance opacity at 0.589/0.77 µm that inflates the
    transit spectrum.  Here:

      * Below the clouds: gas-phase Na–K–Cl–HCl equilibrium
        (Na/(1+k_nacl·HCl), K/(1+k_kcl·HCl), HCl from Cl conservation).
      * Na₂S(s)+H2 -> 2Na+H2S:  above the cloud Na is capped at
        √(k_na2s/H2S) (drops steeply as T falls), and H2S is reduced by
        the sulphur locked into Na₂S.
      * KCl(s): once gaseous KCl exceeds its saturation k_kcl_solid, KCl is
        capped and atomic K = KCl/(k_kcl·HCl) is dragged down with it.

    Element budgets from ``gas_element_abd`` (rained-out, carried upward);
    ``p_bar=p·1e-2`` = true bar (= pressures_layers·1e-5), matching the Fortran's
    pressure in k_eq (FIXED: was p·1e-3 = bar/10, a 10x under-pressure that left
    K/Na under-condensed aloft).  NH4Cl condensation (T≲200 K) and the
    gas_element_abd rain-out write-back remain to be added — secondary here.
    """
    iNa  = gas_id("Na");  iK   = gas_id("K");   iNaCl = gas_id("NaCl")
    iKCl = gas_id("KCl"); iHCl = gas_id("HCl"); iH2S  = gas_id("H2S")
    iH2  = gas_id("H2")
    cNa2S = condensate_id("Na2S"); cKCl = condensate_id("KCl")

    p_bar = p * 1e-2   # true bar = pressures_layers*1e-5 ; legacy p*1e-3 was bar/10
    qh2   = vmr[iH2] if vmr[iH2] > 0 else 1.0

    # element totals (Z-1): Na=10, K=18, Cl=16, S=15
    na_tot = max(gas_element_abd[10, ip], 0.0)
    k_tot  = max(gas_element_abd[18, ip], 0.0)
    cl_tot = max(gas_element_abd[16, ip], 0.0)
    s_tot  = max(gas_element_abd[15, ip], 0.0)
    h2s    = max(vmr[iH2S], 1e-300)

    g_na = gases_delta_g_i[iNa];  g_k  = gases_delta_g_i[iK]
    g_h2s = gases_delta_g_i[iH2S]; g_hcl = gases_delta_g_i[iHCl]
    g_nacl = gases_delta_g_i[iNaCl]; g_kcl = gases_delta_g_i[iKCl]
    g_na2s_c = condensates_delta_g_i[cNa2S]; g_kcl_c = condensates_delta_g_i[cKCl]

    # --- equilibrium constants (Fortran k_eq = ECG / qH2^{H2 stoich}) -----
    # Na2S(s) + H2 -> 2Na + H2S
    k_na2s = _safe_exp((g_na2s_c - 2.0 * g_na - g_h2s) * 1e3 / (CST_R * t)) * qh2 / p_bar ** 2
    # 2Na + 2HCl -> 2NaCl + H2   (per-atom constant = sqrt)
    ecg_nacl = equilibrium_constant_gases([-2, -2, 2, 1], [g_na, g_hcl, g_nacl, 0.0], p_bar, t)
    k_nacl = math.sqrt(max(ecg_nacl / qh2, 0.0)) if math.isfinite(ecg_nacl) else 0.0
    # 2K + 2HCl -> 2KCl + H2
    ecg_kcl = equilibrium_constant_gases([-2, -2, 2, 1], [g_k, g_hcl, g_kcl, 0.0], p_bar, t)
    k_kcl = math.sqrt(max(ecg_kcl / qh2, 0.0)) if math.isfinite(ecg_kcl) else 0.0
    # KCl(s) -> KCl(g)
    k_kcl_solid = _safe_exp((g_kcl_c - g_kcl) * 1e3 / (CST_R * t)) / p_bar

    qnasat = math.sqrt(max(k_na2s / h2s, 0.0))
    vmr_sat[cNa2S] = qnasat
    vmr_sat[cKCl]  = k_kcl_solid

    na_prev  = vmr[iNa]  if ip > 0 else na_tot     # carried-forward Na
    kcl_prev = vmr[iKCl] if ip > 0 else 0.0

    if na_prev <= qnasat and not is_condensed[cNa2S]:
        # ---- gas-phase Na-K-Cl equilibrium (no condensation) -------------
        hcl = cl_tot
        na = na_tot; k_at = k_tot
        for _ in range(10):
            na   = na_tot / (1.0 + k_nacl * hcl)
            k_at = k_tot  / (1.0 + k_kcl  * hcl)
            hcl  = cl_tot / (1.0 + k_nacl * na + k_kcl * k_at)
            hcl  = max(hcl, 1e-300)
        vmr[iNa]   = na
        vmr[iK]    = k_at
        vmr[iNaCl] = k_nacl * na * hcl
        vmr[iKCl]  = k_kcl  * k_at * hcl
        vmr[iHCl]  = hcl
    else:
        # ---- Na2S condensation: atomic Na capped at saturation -----------
        is_condensed[cNa2S] = True
        na = min(qnasat, na_prev)
        vmr[iNa]  = max(na, 1e-300)
        vmr[iH2S] = max(s_tot - 0.5 * (na_prev - na), 1e-300)
        # K-Cl gas equilibrium with Na fixed
        hcl = cl_tot
        k_at = k_tot
        for _ in range(10):
            k_at = k_tot / (1.0 + k_kcl * hcl)
            nacl = k_nacl * na * hcl
            kcl  = k_kcl  * k_at * hcl
            hcl  = max(cl_tot - nacl - kcl, 1e-300)
        kcl_gas = k_kcl * k_at * hcl
        # ---- KCl condensation: cap gaseous KCl, drag atomic K down -------
        if kcl_gas > k_kcl_solid or is_condensed[cKCl]:
            is_condensed[cKCl] = True
            kcl_gas = min(k_kcl_solid, kcl_prev) if kcl_prev > 0.0 else k_kcl_solid
            if k_kcl * hcl > 0.0:
                k_at = kcl_gas / (k_kcl * hcl)
        vmr[iK]    = max(k_at, 1e-300)
        vmr[iKCl]  = max(kcl_gas, 1e-300)
        vmr[iNaCl] = max(k_nacl * na * hcl, 1e-300)
        vmr[iHCl]  = max(hcl, 1e-300)

    # ---- physical bounds: a gas species cannot exceed its element total --
    # (critical: when HCl is exhausted into condensates, K = KCl/(k_kcl·HCl)
    #  can otherwise overflow to a huge *finite* value, spiking the opacity
    #  and diverging the radiative-convective solver.)
    na_cap = max(na_tot, 1e-300)
    k_cap  = max(k_tot,  1e-300)
    cl_cap = max(cl_tot, 1e-300)
    vmr[iNa]   = min(max(vmr[iNa],   1e-300), na_cap)
    vmr[iNaCl] = min(max(vmr[iNaCl], 1e-300), na_cap)
    vmr[iK]    = min(max(vmr[iK],    1e-300), k_cap)
    vmr[iKCl]  = min(max(vmr[iKCl],  1e-300), k_cap)
    vmr[iHCl]  = min(max(vmr[iHCl],  1e-300), cl_cap)
    for _ix in (iNa, iK, iNaCl, iKCl, iHCl, iH2S):
        if (not math.isfinite(vmr[_ix])) or vmr[_ix] < 0.0:
            vmr[_ix] = 1e-300
    return vmr, is_condensed, vmr_sat, layer_cond


def _calculate_al_o(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, pressures_layers, temperatures_layers
):
    """Al₂O₃ condensation."""
    cond_al2o3 = condensate_id("Al2O3")
    idx_al = gas_id("Al")

    dg_al2o3 = condensates_delta_g_i[cond_al2o3]
    dg_al    = gases_delta_g_i[idx_al]
    dg_h2o   = gases_delta_g_i[gas_id("H2O")]

    p_bar = p * 1e-2   # true bar = pressures_layers*1e-5 ; legacy p*1e-3 was bar/10
    qh2   = vmr[gas_id("H2")]
    qh2o  = vmr[gas_id("H2O")]

    # 2Al + 3H₂O → Al₂O₃(s) + 3H₂
    k_al2o3 = equilibrium_constant_gases([-2, -3, 3],
                                         [dg_al, dg_h2o, 0.0],
                                         p_bar, t)
    if qh2 > 0 and qh2o > 0:
        qsat_al2o3 = math.sqrt(max(qh2**3 * qh2o**(-3) / max(k_al2o3, 1e-300), 0.0))
        vmr_sat[cond_al2o3] = qsat_al2o3

        if vmr[idx_al] > qsat_al2o3 and not is_condensed[cond_al2o3]:
            is_condensed[cond_al2o3] = True
            layer_cond[cond_al2o3] = ip
            # r39 (Gemini): the previous version only updated vmr[Al] and
            # silently left H2O / H2 unchanged.  The reaction stoichiometry
            #   2 Al + 3 H₂O → Al₂O₃(s) + 3 H₂
            # requires that for every 2·Δ(Al) consumed, 3·Δ(Al)/2 H₂O are
            # consumed and 3·Δ(Al)/2 H₂ are produced (per Al-atom basis:
            # 1 Al loses → 1.5 H₂O consumed, 1.5 H₂ produced).  In a Solar
            # atmosphere with X_Al/H ≈ 3e-6 the effect is tiny (<0.1% of
            # the H₂O budget), but at higher metallicities or with the
            # over-initialised heavy-element abundances observed in this
            # build (Al ≈ 4e-5, ~15× solar) the discrepancy grows.  We
            # patch it here so the bug stops being latent.
            delta_al = vmr[idx_al] - max(qsat_al2o3, 0.0)   # >0
            if delta_al > 0:
                vmr[idx_al] = max(qsat_al2o3, 0.0)
                delta_h2o = 1.5 * delta_al
                # Don't drive H2O negative; cap at what's available.
                delta_h2o = min(delta_h2o, vmr[gas_id("H2O")])
                vmr[gas_id("H2O")] -= delta_h2o
                vmr[gas_id("H2")]  += delta_h2o   # 1 H₂ per H₂O reacted

    return vmr, is_condensed, vmr_sat, layer_cond


def _calculate_cr(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, pressures_layers, temperatures_layers
):
    """Cr and Cr₂O₃ condensation."""
    idx_cr = gas_id("Cr")
    cond_cr = condensate_id("Cr")
    dg_cr_cond = condensates_delta_g_i[cond_cr]
    dg_cr_gas  = gases_delta_g_i[idx_cr]

    p_bar = p * 1e-2   # true bar = pressures_layers*1e-5 ; legacy p*1e-3 was bar/10
    k_cr = _safe_exp(-(dg_cr_cond - dg_cr_gas) * 1e3 / (CST_R * t)) if t > 0 else 0.0
    qsat_cr = k_cr / p_bar if p_bar > 0 else 0.0
    vmr_sat[cond_cr] = qsat_cr

    if vmr[idx_cr] > qsat_cr and not is_condensed[cond_cr]:
        is_condensed[cond_cr] = True
        layer_cond[cond_cr] = ip
        vmr[idx_cr] = max(qsat_cr, 0.0)

    return vmr, is_condensed, vmr_sat, layer_cond


def _calculate_fe_ni_co(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, pressures_layers, temperatures_layers
):
    """Fe / FeH equilibrium + Fe condensation.

    Faithful port of the Fortran ``calculate_fe_ni_co_equilibrium``
    (chemistry.f90 1202-1263).  The previous version did only the Fe
    condensation cap and never touched FeH, so FeH stayed at its zero
    initialisation at every level.  This adds the three pieces needed to
    reproduce the Fortran FeH profile:

      1. FeH/Fe ratio (Visscher et al. 2010, 2Fe + H2 -> 2FeH):
             rfeh = 10**(-1.85 - 1905/T) * sqrt(p_bar * qH2)
         with FeH = Fe * rfeh, and atomic Fe drawn into FeH via
         Fe[ip] = Fe[ip-1] / (1 + rfeh).  On entry vmr[idx_fe] already holds
         the layer-below value (carried forward at calculate_chemistry:399);
         at the deepest layer it is the elemental Fe abundance, matching the
         Fortran's Fe[1] = Fe[1] / (1 + rfeh).

      2. The Fe(g)->Fe(s) saturation written with the Fortran's sign,
             qsat = exp(+(g_cond - g_gas)*1e3/RT) / p_bar
         (= 1/(K_eq * p) under the standard Gibbs convention used by
         equilibrium_constant_gases; the old code used K_eq/p, the reciprocal).
         Above the Fe cloud this is what drives atomic Fe -- and hence FeH --
         down steeply: the Fortran FeH falls ~22 dex deep->top, of which only
         ~6 dex is rfeh; the rest is Fe tracking this saturation.

      3. The cap is applied at EVERY layer where Fe >= qsat (matching the
         Fortran), not only at the first condensation level -- the old code
         capped once and then froze Fe at the cloud-base value.

    Pressure: ``p_bar = p * 1e-2`` is the *true* bar (p = pressures_layers*1e-3
    with pressures_layers in Pa, so bar = p*1e-2 = pressures_layers*1e-5).  This
    matches the Fortran's ``p`` and the already-corrected gas-phase routines
    (_equil_co_si_o, _calculate_nh3_n2_hcn).  NB the other condensation routines
    (_calculate_cr, _calculate_cl_na_k, _calculate_p, _calculate_ca_o_ti_v,
    _calculate_mn_s, _calculate_zn_s) have NOW been corrected to the same
    ``p*1e-2`` = true bar (previously p*1e-3 = bar/10).

    Ni/Co alloy formation and the interpolated cloud-base (pcfe/tc) bookkeeping
    in the Fortran are not reproduced: neither feeds back into gas-phase Fe or
    FeH, so they are out of scope here (and were already absent before).
    """
    idx_fe  = gas_id("Fe")
    idx_feh = gas_id("FeH")
    idx_h2  = gas_id("H2")
    cond_fe = condensate_id("Fe")
    dg_fe_cond = condensates_delta_g_i[cond_fe]
    dg_fe_gas  = gases_delta_g_i[idx_fe]

    p_bar = p * 1e-2                      # true bar (= pressures_layers*1e-5)
    qh2   = vmr[idx_h2] if vmr[idx_h2] > 0 else 0.0

    # FeH/Fe ratio -- Visscher et al. (2010), 2Fe + H2 -> 2FeH
    if t > 0 and p_bar > 0:
        rfeh = (10.0 ** (-1.85 - 1.905e3 / t)) * math.sqrt(p_bar * qh2)
    else:
        rfeh = 0.0

    # Atomic Fe drawn down into FeH (Fortran: Fe[ip] = Fe[ip-1] / (1 + rfeh)).
    vmr[idx_fe] = vmr[idx_fe] / (1.0 + rfeh)

    # Fe(g) -> Fe(s) saturation, Fortran sign (= 1/(K_eq * p)).
    if t > 0 and p_bar > 0:
        qsat_fe = _safe_exp((dg_fe_cond - dg_fe_gas) * 1e3 / (CST_R * t)) / p_bar
    else:
        qsat_fe = 0.0
    vmr_sat[cond_fe] = qsat_fe

    if vmr[idx_fe] >= qsat_fe and not is_condensed[cond_fe]:
        is_condensed[cond_fe] = True
        layer_cond[cond_fe] = ip

    if vmr[idx_fe] >= qsat_fe:
        vmr[idx_fe] = max(qsat_fe, 0.0)

    # FeH = Fe * rfeh  (both Fortran branches).
    vmr[idx_feh] = max(vmr[idx_fe] * rfeh, 0.0)

    return vmr


def _calculate_ca_o_ti_v(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i, is_condensed,
    gas_element_abd, elemental_h_ratio
):
    """Ca/O/Ti/V gas equilibrium + condensation.

    Faithful port of the Fortran ``calculate_ca_o_ti_v_equilibirum``
    (chemistry.f90 977-1164).  The gas-phase Ti/TiO/TiO2 and V/VO/VO2
    partition was already present; this version ADDS the condensation that
    actually removes the refractories from the gas (previously skipped):

      * TiN(s)     : TiO + ... -> TiN ;   caps TiO at H2O/(k_tin·√N2)
      * CaTiO3(s)  : Ca + TiO + 2H2O -> CaTiO3 + 2H2 ;  the dominant Ti sink,
                     drives TiO down to 1/(k_catio3·H2O²·Ca) and depletes Ca;
                     also dissolves VO into the perovskite.
      * Ca(s)      : simple saturation cap on atomic Ca.
      * VO(s)      : simple saturation cap on VO (the dominant V sink at low T).

    Without these, TiO/VO stay at their gas-equilibrium values (~1e-8…1e-36
    aloft) instead of condensing out (~1e-300), the ~270 dex error seen in the
    comparison and the spurious optical/NIR opacity that inflates the transit
    spectrum.  Element budgets use ``gas_element_abd`` (rained-out, carried up)
    exactly as the Fortran does; pressure uses ``p_bar=p·1e-2`` = true bar
    (= pressures_layers·1e-5), matching the Fortran (FIXED: was p·1e-3 = bar/10,
    which left TiO/VO under-condensed at depth).
    """
    idx_ti   = gas_id("Ti");   idx_tio  = gas_id("TiO");  idx_tio2 = gas_id("TiO2")
    idx_v    = gas_id("V");    idx_vo   = gas_id("VO");   idx_vo2  = gas_id("VO2")
    idx_h2   = gas_id("H2");   idx_h2o  = gas_id("H2O")
    idx_ca   = gas_id("Ca");   idx_n2   = gas_id("N2")

    dg_ti   = gases_delta_g_i[idx_ti];   dg_tio  = gases_delta_g_i[idx_tio]
    dg_tio2 = gases_delta_g_i[idx_tio2]; dg_v    = gases_delta_g_i[idx_v]
    dg_vo   = gases_delta_g_i[idx_vo];   dg_vo2  = gases_delta_g_i[idx_vo2]
    dg_h2o  = gases_delta_g_i[idx_h2o];  dg_ca   = gases_delta_g_i[idx_ca]

    p_bar = p * 1e-2   # true bar = pressures_layers*1e-5 ; legacy p*1e-3 was bar/10
    qh2   = vmr[idx_h2] if vmr[idx_h2] > 0 else 1.0
    qh2o  = max(vmr[idx_h2o], 0.0)
    TINY  = 1e-300

    # element totals (Z-1 indices): Ti=21, V=22, Ca=19
    abd_ti = gas_element_abd[21, ip]; abd_v = gas_element_abd[22, ip]
    abd_ca = gas_element_abd[19, ip]

    # ---- gas-phase equilibrium constants (Fortran k_eq, qH2≈1) -----------
    k_tio  = equilibrium_constant_gases([-1, -1, 1, 1], [dg_ti, dg_h2o, dg_tio, 0.0], p_bar, t)
    k_tio2 = equilibrium_constant_gases([-1, -2, 1, 2], [dg_ti, dg_h2o, dg_tio2, 0.0], p_bar, t)
    k_vo   = equilibrium_constant_gases([-1, -1, 1, 1], [dg_v, dg_h2o, dg_vo, 0.0], p_bar, t)
    k_vo2  = equilibrium_constant_gases([-1, -2, 1, 2], [dg_v, dg_h2o, dg_vo2, 0.0], p_bar, t)
    k_tio  = k_tio  if (math.isfinite(k_tio)  and k_tio  >= 0) else 0.0
    k_tio2 = k_tio2 if (math.isfinite(k_tio2) and k_tio2 >= 0) else 0.0
    k_vo   = k_vo   if (math.isfinite(k_vo)   and k_vo   >= 0) else 0.0
    k_vo2  = k_vo2  if (math.isfinite(k_vo2)  and k_vo2  >= 0) else 0.0

    # ---- Ti / TiO / TiO2 partition (Fortran form, X_Ti = gas_element_abd) -
    if abd_ti > 0.0 and qh2o > 0.0:
        vmr[idx_ti]   = abd_ti / (1.0 + k_tio * qh2o + k_tio2 * qh2o ** 2)
        denom_tio = 1.0 + (1.0 / (k_tio * qh2o) if (k_tio * qh2o) > 0 else 1e300) \
                    + (k_tio2 * qh2o / k_tio if k_tio > 0 else 0.0)
        vmr[idx_tio]  = abd_ti / denom_tio
        denom_tio2 = 1.0 + (1.0 / (k_tio2 * qh2o ** 2) if (k_tio2 * qh2o ** 2) > 0 else 1e300) \
                     + (k_tio / (k_tio2 * qh2o) if (k_tio2 * qh2o) > 0 else 1e300)
        vmr[idx_tio2] = abd_ti / denom_tio2
    if vmr[idx_tio] < TINY:
        vmr[idx_tio] = TINY
    r1 = vmr[idx_ti] / vmr[idx_tio]
    r2 = vmr[idx_tio2] / vmr[idx_tio]

    if abd_v > 0.0 and qh2o > 0.0:
        vmr[idx_v]   = abd_v / (1.0 + k_vo * qh2o + k_vo2 * qh2o ** 2)
        denom_vo = 1.0 + (1.0 / (k_vo * qh2o) if (k_vo * qh2o) > 0 else 1e300) \
                   + (k_vo2 * qh2o / k_vo if k_vo > 0 else 0.0)
        vmr[idx_vo]  = abd_v / denom_vo
        denom_vo2 = 1.0 + (1.0 / (k_vo2 * qh2o ** 2) if (k_vo2 * qh2o ** 2) > 0 else 1e300) \
                    + (k_vo / (k_vo2 * qh2o) if (k_vo2 * qh2o) > 0 else 1e300)
        vmr[idx_vo2] = abd_v / denom_vo2
    if vmr[idx_vo] < TINY:
        vmr[idx_vo] = TINY
    r1v = vmr[idx_v] / vmr[idx_vo]
    r2v = vmr[idx_vo2] / vmr[idx_vo]

    # ---- condensation equilibrium constants ------------------------------
    cid_tin    = condensate_id("TiN")
    cid_catio3 = condensate_id("CaTiO3")
    cid_ca     = condensate_id("Ca")
    cid_vo     = condensate_id("VO")
    dg_tin_c    = condensates_delta_g_i[cid_tin]
    dg_catio3_c = condensates_delta_g_i[cid_catio3]
    dg_ca_c     = condensates_delta_g_i[cid_ca]
    dg_vo_c     = condensates_delta_g_i[cid_vo]
    qn2 = max(vmr[idx_n2], 0.0)

    # 2TiO + 2H2 + N2 -> 2TiN + 2H2O
    k_tin = _safe_exp((dg_tio - dg_tin_c - dg_h2o) * 1e3 / (CST_R * t)) * qh2 * p_bar ** 1.5
    # VO(s) -> VO(g)
    k_vo_solid = _safe_exp((dg_vo_c - dg_vo) * 1e3 / (CST_R * t)) / p_bar
    # Ca + TiO + 2H2O -> CaTiO3(s) + 2H2
    k_catio3 = _safe_exp((dg_ca + dg_tio + 2.0 * dg_h2o - dg_catio3_c) * 1e3 / (CST_R * t)) \
               / (qh2 ** 2 / p_bar ** 2)
    # Ca(s) -> Ca(g)
    k_ca_solid = _safe_exp(-dg_ca * 1e3 / (CST_R * t)) / p_bar

    # ---- TiN condensation (only if CaTiO3 not yet condensing) ------------
    if qn2 > 0.0 and k_tin > 0.0:
        tio_tin_sat = qh2o / (k_tin * math.sqrt(qn2))
        if vmr[idx_tio] > tio_tin_sat and not is_condensed[cid_catio3]:
            is_condensed[cid_tin] = True
            vmr[idx_tio]  = tio_tin_sat
            vmr[idx_ti]   = vmr[idx_tio] * r1
            vmr[idx_tio2] = vmr[idx_tio] * r2
            gas_element_abd[21, ip] = vmr[idx_tio] * (1.0 + r1 + r2)

    # ---- CaTiO3 condensation (dominant Ti sink) + VO dissolution ---------
    if k_catio3 > 0.0 and qh2o > 0.0 and gas_element_abd[19, ip] * vmr[idx_tio] \
            > 1.0 / (k_catio3 * qh2o ** 2):
        is_condensed[cid_catio3] = True
        ca = gas_element_abd[19, ip]
        for _ in range(5):
            ti_tmp = vmr[idx_tio] * (1.0 + r1 + r2)
            vmr[idx_tio] = 1.0 / (k_catio3 * qh2o ** 2 * ca) if ca > 0 else TINY
            ca = ca - (ti_tmp - vmr[idx_tio] * (1.0 + r1 + r2))
        vmr[idx_ti]   = vmr[idx_tio] * r1
        vmr[idx_tio2] = vmr[idx_tio] * r2
        gas_element_abd[19, ip] = ca
        gas_element_abd[21, ip] = vmr[idx_tio] * (1.0 + r1 + r2)
        vmr[idx_ca] = max(ca, 0.0)
        # VO dissolves into CaTiO3
        if vmr[idx_vo] < k_vo_solid:
            abd_v_prev  = gas_element_abd[22, max(ip - 1, 0)]
            abd_ti_prev = gas_element_abd[21, max(ip - 1, 0)]
            abd_ti_now  = gas_element_abd[21, ip]
            for _ in range(5):
                denom = (abd_ti_prev - abd_ti_now + abd_v_prev
                         + (1.0 + r1v + r2v) * (k_vo_solid - vmr[idx_vo]))
                vmr[idx_vo] = (abd_v_prev * k_vo_solid / denom) if denom != 0 else vmr[idx_vo]
            vmr[idx_v]   = vmr[idx_vo] * r1v
            vmr[idx_vo2] = vmr[idx_vo] * r2v
            gas_element_abd[22, ip] = vmr[idx_vo] * (1.0 + r1v + r2v)

    # ---- Ca(s) condensation (simple saturation cap) ----------------------
    if vmr[idx_ca] > k_ca_solid:
        if not is_condensed[cid_ca]:
            is_condensed[cid_ca] = True
        vmr[idx_ca] = k_ca_solid

    # ---- VO(s) condensation (dominant V sink at low T) -------------------
    if vmr[idx_vo] > k_vo_solid:
        if not is_condensed[cid_vo]:
            is_condensed[cid_vo] = True
        vmr[idx_vo]  = k_vo_solid
        vmr[idx_v]   = vmr[idx_vo] * r1v
        vmr[idx_vo2] = vmr[idx_vo] * r2v
        gas_element_abd[22, ip] = vmr[idx_vo] * (1.0 + r1v + r2v)

    # ---- numerical safety: no negative / NaN refractory VMRs -------------
    for _ix in (idx_ti, idx_tio, idx_tio2, idx_v, idx_vo, idx_vo2, idx_ca):
        if (not math.isfinite(vmr[_ix])) or vmr[_ix] < 0.0:
            vmr[_ix] = TINY

    # ---- final VO cap relative to TiO (solar V/Ti) -----------------------
    if elemental_h_ratio[21] > 0:
        vmr[idx_vo] = min(vmr[idx_vo],
                          vmr[idx_tio] * elemental_h_ratio[22] / elemental_h_ratio[21])
    return vmr


def _calculate_mg_si_o_simple(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, pressures_layers, temperatures_layers
):
    """Simplified silicate condensation (FALLBACK, pre-sink "3b" behaviour).

    Enstatite-only, single-shot at the first supersaturated layer; uses the
    pressure now ``p_bar = p*1e-2`` = true bar (was p*1e-3 = bar/10); still never
    touches ``gas_element_abd``
    nor re-solves C/O/Si, so the O it removes cannot propagate upward.
    Retained verbatim and selected when ``_USE_MG_SI_O_SINK is False``.
    """
    # Simplified: handle MgSiO3 only
    cond_mgsio3 = condensate_id("MgSiO3")
    idx_sio     = gas_id("SiO")

    dg_mgsio3 = condensates_delta_g_i[cond_mgsio3]
    dg_sio    = gases_delta_g_i[idx_sio]
    dg_h2o    = gases_delta_g_i[gas_id("H2O")]

    p_bar = p * 1e-2   # true bar = pressures_layers*1e-5 ; legacy p*1e-3 was bar/10
    qh2   = vmr[gas_id("H2")]
    qmg   = vmr[gas_id("Mg")]
    qsio  = vmr[idx_sio]
    qh2o  = vmr[gas_id("H2O")]

    # Mg + SiO + 2H₂O → MgSiO₃(s) + 2H₂
    k_mgsio3 = equilibrium_constant_gases([-1, -1, -2, 2],
                                          [0.0, dg_sio, dg_h2o, 0.0],
                                          p_bar, t)
    if qmg > 0 and qh2o > 0 and qsio > 0:
        qsat_sio3 = qh2**2 / max(k_mgsio3 * qmg * qh2o**2, 1e-300)
        vmr_sat[cond_mgsio3] = qsat_sio3

        if qsio > qsat_sio3 and not is_condensed[cond_mgsio3]:
            is_condensed[cond_mgsio3] = True
            layer_cond[cond_mgsio3] = ip
            # r39 (Gemini): previously only vmr[SiO] was reduced.  The
            # stoichiometry  Mg + SiO + 2 H₂O → MgSiO₃(s) + 2 H₂  requires
            # that for every Δ(SiO) condensed, 1·Δ(SiO) Mg is consumed,
            # 2·Δ(SiO) H₂O are consumed, and 2·Δ(SiO) H₂ are produced.
            # The previous code left H₂O at its pre-condensation value,
            # which in the worst case (full Si condensation at ~5e-4) was
            # 2 × 5e-4 = 1e-3 of "phantom" oxygen, ~18% of the elemental
            # O budget, double-counted as both gas-phase H₂O and locked-up
            # silicate.  Here we close the budget by debiting H₂O and
            # crediting H₂, capped against availability.
            delta_sio = qsio - max(qsat_sio3, 0.0)
            vmr[idx_sio] = max(qsat_sio3, 0.0)
            if delta_sio > 0:
                # Mg → consumed 1:1 with SiO (cap against availability).
                delta_mg = min(delta_sio, vmr[gas_id("Mg")])
                vmr[gas_id("Mg")] -= delta_mg
                # H2O → consumed 2:1 (cap, then back-calculate true SiO
                # consumption if H2O is the limiting reagent).
                delta_h2o_wanted = 2.0 * delta_sio
                delta_h2o = min(delta_h2o_wanted, vmr[gas_id("H2O")])
                vmr[gas_id("H2O")] -= delta_h2o
                vmr[gas_id("H2")]  += delta_h2o   # 1 H₂ per H₂O consumed

    return vmr


def _partition_o_mg_si(ii, alf1, alf2, alpha1, alf, alpha, qmg, asi, ao, ic):
    """Secant accelerator on the condensation step ``alpha`` — faithful port of
    the Fortran ``partition_o_mg_si`` (chemistry.f90 3592-3626).

    For ii<=2 it only seeds the (alf, alpha) history and leaves ``alpha``
    unchanged; for ii>=3 it overrides ``alpha`` with a log-secant estimate
    ``x = -alpha1·ln(alf2)/ln(alf2/alf1)`` capped by per-condensate availability
    limits (ic selects the cap set: 1=forsterite, 2=enstatite, 3=silica).
    Returns the updated (alf1, alf2, alpha1, alpha).
    """
    if ii <= 2:
        alf1 = alf
        alpha1 = alpha
    else:
        alf2 = alf
        if alf2 * (1.0 - _PREC_HIGH) < alf1 < alf2 * (1.0 + _PREC_HIGH):
            x = _HUGE_SP
        else:
            x = -alpha1 * math.log(alf2) / math.log(alf2 / alf1)
        if ic == 1:
            alpha = min(x, 0.4 * qmg, 0.8 * asi, 0.2 * ao)
        elif ic == 2:
            alpha = min(x, 0.8 * qmg, 0.8 * asi, 0.2667 * ao)
        else:
            alpha = min(x, 0.8 * asi, 0.4 * ao)
        alf1 = alf2
        alpha1 = alpha
    return alf1, alf2, alpha1, alpha


def _calculate_mg_si_o(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, pressures_layers, temperatures_layers,
    gas_element_abd=None, elements_in_gases=None,
    co_ch4_quench=False, co_co2_quench=False, qcoco2=0.0,
):
    """Mg/Si/O silicate condensation O-sink — faithful port of the Fortran
    ``calculate_mg_si_o_equilibrium`` (chemistry.f90 1166-1200) and its three
    condensation routines (mg2sio4_/mgsio3_/sio2_condensation):

        Mg2SiO4(s) + 3 H2 -> 2 Mg + SiO + 3 H2O   (forsterite, removes 4 O / unit)
        MgSiO3(s)  + 2 H2 ->   Mg + SiO + 2 H2O   (enstatite,  removes 3 O / unit)
        SiO2(s)    +   H2 ->        SiO +   H2O    (silica,     removes 2 O / unit)

    Each supersaturated layer relaxes the gas SiO/H2O/Mg toward the solubility
    product with a 15-step damped iteration, debiting the carried-forward
    elemental oxygen (``gas_element_abd[7]``) and silicon (``gas_element_abd[13]``)
    and re-solving the coupled C/O/Si gas partition (``osi`` / ``osiqco`` /
    ``osiqcoco2``) after every step.  Removing O from the gas here is what lets
    the model shed the ~16% of gas-phase oxygen between the silicate cloud bases
    and aloft that the simplified stub could not.

    Conventions match the rest of the port: pressure in bar (``p*1e-2``), the
    Fortran ``k_eq`` qH2 division (via ``_keq``), and un-normalised VMRs (which
    the initialiser pins to Σ≈1, i.e. the Fortran's normalised scale).

    Faithful to the Fortran *main* partition loops at each layer's own (T,p).
    DEFERRED (documented; onset-layer-only, 2nd-order — revisit if the measured
    per-layer O-budget needs it): the crossing-level 5-iteration ΔG(@tc)
    re-equilibration loops and the frac/pc/tc onset bookkeeping (which feed only
    the untracked p_c/vmr_c condensation-pressure outputs).  Also: the alf>1
    over-shoot max-branch uses the SiO-vmr form uniformly (the Fortran's
    onset-iteration variant momentarily uses the Si abundance instead — same
    2nd-order rare branch).  Set ``_USE_MG_SI_O_SINK = False`` to fall back to
    the simplified enstatite-only stub.
    """
    if (not _USE_MG_SI_O_SINK) or gas_element_abd is None or elements_in_gases is None:
        return _calculate_mg_si_o_simple(
            t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
            is_condensed, vmr_sat, layer_cond, pressures_layers,
            temperatures_layers)

    iH2 = gas_id("H2"); iH2O = gas_id("H2O"); iMg = gas_id("Mg")
    iSiO = gas_id("SiO"); iCO = gas_id("CO"); iCO2 = gas_id("CO2")
    cMg2SiO4 = condensate_id("Mg2SiO4")
    cMgSiO3 = condensate_id("MgSiO3")
    cSiO2 = condensate_id("SiO2")

    gae_ip = gas_element_abd[:, ip]           # view: decrements write through
    p_bar = p * 1e-2                          # Pa·1e-3 → ·1e-2 = pressures·1e-5 = bar
    h2o_sat = bool(is_condensed[condensate_id("H2O")])
    qh2 = vmr[iH2] if vmr[iH2] > 0.0 else 1e-300

    # --- three solubility products (exact Fortran formula; qH2 & p in bar) ---
    a_fo = (condensates_delta_g_i[cMg2SiO4]
            - 2.0 * gases_delta_g_i[iMg] - gases_delta_g_i[iSiO]
            - 3.0 * gases_delta_g_i[iH2O]) * 1e3 / (CST_R * t)
    k_eq_mg2sio4 = max(_TINY, _safe_exp(a_fo) * qh2 ** 3 / p_bar ** 3)

    a_en = (condensates_delta_g_i[cMgSiO3]
            - gases_delta_g_i[iMg] - gases_delta_g_i[iSiO]
            - 2.0 * gases_delta_g_i[iH2O]) * 1e3 / (CST_R * t)
    k_eq_mgsio3 = max(_TINY, _safe_exp(a_en) * qh2 ** 2 / p_bar ** 2)

    a_si = (condensates_delta_g_i[cSiO2]
            - gases_delta_g_i[iSiO] - gases_delta_g_i[iH2O]) * 1e3 / (CST_R * t)
    k_eq_sio2 = max(_TINY, _safe_exp(a_si) * qh2 / p_bar)

    # ---- shared re-equilibration of the C/O/Si gas partition (Fortran
    #      update_c_o_si): re-solve H2O/CO/CO2/SiO from the (now reduced) O & Si.
    def update_c_o_si():
        if not co_ch4_quench:
            qh2_now = vmr[iH2] if vmr[iH2] > 0.0 else 1e-300
            kc  = _keq(["H2O", "CH4", "CO", "H2"], [-1, -1, 1, 3], gases_delta_g_i, p_bar, t, qh2_now)
            kc3 = math.sqrt(max(_keq(["CH4", "CH3", "H2"], [-2, 2, 1], gases_delta_g_i, p_bar, t, qh2_now), 0.0))
            kc2 = _keq(["CO", "H2O", "CO2", "H2"], [-1, -1, 1, 1], gases_delta_g_i, p_bar, t, qh2_now)
            ks  = _keq(["SiH4", "H2O", "SiO", "H2"], [-1, -1, 1, 3], gases_delta_g_i, p_bar, t, qh2_now)
            _osi(vmr, gae_ip, elements_in_gases, gases_delta_g_i,
                 kc, kc3, kc2, ks, p_bar, t, h2o_sat)
        elif not co_co2_quench:
            _osiqco(t, p, ip, vmr, gases_delta_g_i, gae_ip, qcoco2, h2o_sat)
        else:
            _osiqcoco2(t, p, ip, vmr, gases_delta_g_i, gae_ip, h2o_sat)

    def ao_quench_adjusted():
        """``gas_element_abd[O] − SiO`` (− CO − 2·CO2 when CO/CO2 is quenched)."""
        a = gae_ip[7] - vmr[iSiO]
        if co_co2_quench:
            a = a - vmr[iCO] - 2.0 * vmr[iCO2]
        return a

    def clamp_elements():
        """Defensive: keep O, Si abundances and Mg finite & non-negative.

        Bounds the elemental sinks from below so a float-noise over-shoot can't
        feed osi a negative O/Si (handoff safety lesson: a finite-but-rogue
        value silently diverges the RC solver).  The Fortran relies on the
        analytic + secant caps alone; in practice these never bind."""
        if (not math.isfinite(gae_ip[7])) or gae_ip[7] < _TINY:
            gae_ip[7] = _TINY
        if (not math.isfinite(gae_ip[13])) or gae_ip[13] < _TINY:
            gae_ip[13] = _TINY
        if (not math.isfinite(vmr[iMg])) or vmr[iMg] < 0.0:
            vmr[iMg] = 1e-300

    # ---- forsterite : Mg2SiO4(s) + 3H2 -> 2Mg + SiO + 3H2O ----------------
    def mg2sio4_condensation():
        sio = vmr[iSiO]; h2o = vmr[iH2O]
        denom = sio * h2o ** 3
        if denom > 0.0:
            # qsat = sqrt(k_eq/(SiO·H2O^3)); guard the divide so a far-subsaturated
            # (enormous k_eq) layer yields a finite sentinel instead of inf + a
            # numpy overflow warning (vmr_sat feeds cloud opacity only).
            vmr_sat[cMg2SiO4] = (math.sqrt(k_eq_mg2sio4 / denom)
                                 if k_eq_mg2sio4 < 1.7e308 * denom else _HUGE_SP)
        if not (vmr[iMg] ** 2 * vmr[iSiO] * vmr[iH2O] ** 3 > k_eq_mg2sio4):
            return
        first_cross = not is_condensed[cMg2SiO4]
        if first_cross:
            is_condensed[cMg2SiO4] = True
            layer_cond[cMg2SiO4] = ip
            # DEFERRED: frac/pc/tc onset bookkeeping (untracked outputs only).
        elif is_condensed[cMgSiO3] or is_condensed[cSiO2]:
            return                        # Fortran gates the already-condensed loop
        extra_div3 = not first_cross      # Fortran loop-B max-branch /3 nuance
        alf1 = alf2 = alpha1 = alpha = 0.0
        for ii in range(1, 16):
            prod = vmr[iMg] ** 2 * vmr[iSiO] * vmr[iH2O] ** 3
            if not (prod > 0.0 and math.isfinite(prod)):
                break
            alf = k_eq_mg2sio4 / prod
            if abs(alf - 1.0) < 1e-8:
                break
            ao = ao_quench_adjusted()
            t1 = 0.5 * vmr[iMg] * (1.0 - math.sqrt(alf))
            t2 = vmr[iSiO] * (1.0 - alf)
            cube = 1.0 - alf ** (1.0 / 3.0)
            if alf < 1.0:
                alpha = min(t1, t2, ao / 3.0 * cube)
            elif extra_div3:
                alpha = max(t1, t2, ao / 3.0 * cube / 3.0)
            else:
                alpha = max(t1, t2, ao / 3.0 * cube)
            alf1, alf2, alpha1, alpha = _partition_o_mg_si(
                ii, alf1, alf2, alpha1, alf, alpha,
                vmr[iMg], gae_ip[13], gae_ip[7], 1)
            if vmr[iMg] - 2.0 * alpha > _TINY:
                vmr[iMg] = vmr[iMg] - 2.0 * alpha
            else:
                vmr[iMg] = vmr[iMg] * math.sqrt(alf)
            gae_ip[7] = gae_ip[7] - 4.0 * alpha
            gae_ip[13] = gae_ip[13] - alpha
            clamp_elements()
            update_c_o_si()
        clamp_elements()

    # ---- enstatite : MgSiO3(s) + 2H2 -> Mg + SiO + 2H2O -------------------
    def mgsio3_condensation():
        sio = vmr[iSiO]; h2o = vmr[iH2O]
        denom = sio * h2o ** 2
        if denom > 0.0:
            q = (k_eq_mgsio3 / denom) if k_eq_mgsio3 < 1.7e308 * denom else _HUGE_SP
            vmr_sat[cMgSiO3] = min(vmr[iMg], q)
        if not (vmr[iMg] * vmr[iSiO] * vmr[iH2O] ** 2 > k_eq_mgsio3):
            return
        if not is_condensed[cMgSiO3]:
            is_condensed[cMgSiO3] = True
            layer_cond[cMgSiO3] = ip
            # DEFERRED: 5-iter crossing-level forsterite re-equilibration +
            #           frac/pc/tc onset bookkeeping.
        # main enstatite loop (Fortran first-cross & already-condensed paths are
        # the same loop modulo the deferred crossing-level block).
        alf1 = alf2 = alpha1 = alpha = 0.0
        for ii in range(1, 16):
            prod = vmr[iMg] * vmr[iSiO] * vmr[iH2O] ** 2
            if not (prod > 0.0 and math.isfinite(prod)):
                break
            alf = k_eq_mgsio3 / prod
            if abs(alf - 1.0) < 1e-8:
                break
            ao = ao_quench_adjusted()
            root = 1.0 - math.sqrt(alf)
            t1 = vmr[iMg] * (1.0 - alf)
            t2 = vmr[iSiO] * (1.0 - alf)
            if alf < 1.0:
                alpha = min(t1, t2, ao / 2.0 * root)
            else:
                alpha = max(t1, t2, ao / 2.0 * root)
            alf1, alf2, alpha1, alpha = _partition_o_mg_si(
                ii, alf1, alf2, alpha1, alf, alpha,
                vmr[iMg], gae_ip[13], gae_ip[7], 2)
            if vmr[iMg] - alpha > _TINY:
                vmr[iMg] = vmr[iMg] - alpha
            else:
                vmr[iMg] = vmr[iMg] * alf
            gae_ip[7] = gae_ip[7] - 3.0 * alpha
            gae_ip[13] = gae_ip[13] - alpha
            clamp_elements()
            update_c_o_si()
        clamp_elements()

    # ---- silica : SiO2(s) + H2 -> SiO + H2O -------------------------------
    def sio2_loop(use_quench_ao):
        alf1 = alf2 = alpha1 = alpha = 0.0
        for ii in range(1, 16):
            prod = vmr[iSiO] * vmr[iH2O]
            if not (prod > 0.0 and math.isfinite(prod)):
                break
            alf = k_eq_sio2 / prod
            if abs(alf - 1.0) < 1e-8:
                break
            ao = ao_quench_adjusted() if use_quench_ao else (gae_ip[7] - vmr[iSiO])
            t1 = vmr[iSiO] * (1.0 - alf)
            t2 = ao * (1.0 - alf)
            alpha = min(t1, t2) if alf < 1.0 else max(t1, t2)
            alf1, alf2, alpha1, alpha = _partition_o_mg_si(
                ii, alf1, alf2, alpha1, alf, alpha,
                vmr[iMg], gae_ip[13], gae_ip[7], 3)
            if gae_ip[13] - alpha > _TINY:
                gae_ip[13] = gae_ip[13] - alpha
            else:
                gae_ip[13] = gae_ip[13] * alf
            gae_ip[7] = gae_ip[7] - 2.0 * alpha
            clamp_elements()
            update_c_o_si()
        clamp_elements()

    def sio2_condensation():
        if vmr[iSiO] * vmr[iH2O] > k_eq_sio2:
            if not is_condensed[cSiO2]:
                is_condensed[cSiO2] = True
                layer_cond[cSiO2] = ip
                # DEFERRED: 5-iter crossing-level enstatite re-equilibration +
                #           frac/pc/tc onset bookkeeping.
                sio2_loop(use_quench_ao=True)     # onset-layer equilibration
            # maintained on every supersaturated layer (unless H2O is saturated)
            if (vmr[iSiO] * vmr[iH2O] >= k_eq_sio2) and (not h2o_sat):
                sio2_loop(use_quench_ao=False)
        vmr_sat[cSiO2] = vmr[iSiO]

    mg2sio4_condensation()
    mgsio3_condensation()
    sio2_condensation()
    return vmr


def _calculate_mn_s(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, pressures_layers, temperatures_layers
):
    """MnS condensation."""
    idx_mn = gas_id("Mn")
    idx_h2s = gas_id("H2S")
    cond_mns = condensate_id("MnS")

    dg_mns = condensates_delta_g_i[cond_mns]
    dg_mn  = gases_delta_g_i[idx_mn]
    dg_h2s = gases_delta_g_i[idx_h2s]

    p_bar = p * 1e-2   # true bar = pressures_layers*1e-5 ; legacy p*1e-3 was bar/10
    qh2   = vmr[gas_id("H2")]
    qmn   = vmr[idx_mn]
    qh2s  = vmr[idx_h2s]

    # Mn + H₂S → MnS(s) + H₂
    k_mns = equilibrium_constant_gases([-1, -1, 1],
                                       [dg_mn, dg_h2s, 0.0],
                                       p_bar, t)
    if qmn > 0 and qh2s > 0:
        qsat_mns = qh2 / max(k_mns * qh2s, 1e-300)
        vmr_sat[cond_mns] = qsat_mns

        if qmn > qsat_mns and not is_condensed[cond_mns]:
            is_condensed[cond_mns] = True
            layer_cond[cond_mns] = ip
            vmr[idx_mn] = max(qsat_mns, 0.0)

    return vmr


def _calculate_zn_s(
    t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
    is_condensed, vmr_sat, layer_cond, pressures_layers, temperatures_layers
):
    """ZnS condensation."""
    idx_zn  = gas_id("Zn")
    idx_h2s = gas_id("H2S")
    cond_zns = condensate_id("ZnS")

    dg_zns = condensates_delta_g_i[cond_zns]
    dg_zn  = gases_delta_g_i[idx_zn]
    dg_h2s = gases_delta_g_i[idx_h2s]

    p_bar = p * 1e-2   # true bar = pressures_layers*1e-5 ; legacy p*1e-3 was bar/10
    qh2   = vmr[gas_id("H2")]

    k_zns = equilibrium_constant_gases([-1, -1, 1],
                                       [dg_zn, dg_h2s, 0.0],
                                       p_bar, t)
    qzn  = vmr[idx_zn]
    qh2s = vmr[idx_h2s]
    if qzn > 0 and qh2s > 0:
        qsat_zns = qh2 / max(k_zns * qh2s, 1e-300)
        vmr_sat[cond_zns] = qsat_zns

        if qzn > qsat_zns and not is_condensed[cond_zns]:
            is_condensed[cond_zns] = True
            layer_cond[cond_zns] = ip
            vmr[idx_zn] = max(qsat_zns, 0.0)

    return vmr


def _calculate_p(t, p, ip, vmr, gases_delta_g_i, condensates_delta_g_i,
                 is_condensed, gas_element_abd, at_equilibrium):
    """P / PH3 / P2 / PH2 / PO equilibrium + H3PO4 condensation.

    Faithful port of the Fortran ``calculate_p_equilibrium`` (chemistry.f90
    1265-1317).  The previous version was a one-reaction simplification
    (P + 1.5 H2 -> PH3 from a held atomic-P value) that left PH3 flat at ~the
    elemental P abundance (~4.6e-6) at every level.  This partitions the
    *fixed* elemental P budget across PH3/P/P2/PH2/PO and adds the H3PO4
    condensation sink, reproducing the Fortran's fall in PH3 at depth (P and
    P2 take over at high T) and aloft (H3PO4 rains out at low T).

    Ratios relative to PH3 (Fortran k_eq = ECG / qH2^{H2 stoich}; _keq is the
    matching wrapper):
        P   = k_eq_ph3 * PH3          2PH3 -> 2P  + 3H2
        P2  = k_eq_p2  * PH3^2        2PH3 -> P2  + 3H2
        PH2 = k_eq_ph2 * PH3          2PH3 -> 2PH2 + H2
        PO  = k_eq_po  * PH3          2PH3 + 2H2O -> 2PO + 5H2
    Mass balance PH3*term + 2*P2 = P_elem gives PH3 (linear when P2 is
    negligible, else the quadratic root), then the rest are back-substituted.

    Pressure: ``p_bar = p * 1e-2`` is true bar (= pressures_layers*1e-5),
    matching the Fortran's ``p`` and the gas-phase routines; the previous code
    used ``p*1e-3`` = bar/10.

    P budget index: gas_element_abd[14] is phosphorus (Z=15 -> Z-1=14), the
    same (Z-1) convention used by the alkali routine (S sits at 15).
    """
    idx_ph3 = gas_id("PH3"); idx_p = gas_id("P");   idx_p2 = gas_id("P2")
    idx_ph2 = gas_id("PH2"); idx_po = gas_id("PO"); idx_h2o = gas_id("H2O")
    idx_h2  = gas_id("H2")
    cid_h3po4 = condensate_id("H3PO4")
    iP = 14                                # phosphorus elemental abundance (Z-1)

    p_bar  = p * 1e-2                       # true bar
    qh2    = vmr[idx_h2]  if vmr[idx_h2]  > 0 else 1e-300
    qh2o   = vmr[idx_h2o] if vmr[idx_h2o] > 0 else 0.0
    p_elem = max(gas_element_abd[iP, ip], 0.0)

    # Equilibrium ratios relative to PH3.
    k_eq_ph3 = math.sqrt(max(_keq(["PH3", "P", "H2"], [-2, 2, 3],
                                  gases_delta_g_i, p_bar, t, qh2), 0.0))
    k_eq_p2  = _keq(["PH3", "P2", "H2"], [-2, 1, 3], gases_delta_g_i, p_bar, t, qh2)
    k_eq_ph2 = math.sqrt(max(_keq(["PH3", "PH2", "H2"], [-2, 2, 1],
                                  gases_delta_g_i, p_bar, t, qh2), 0.0))
    k_eq_po  = math.sqrt(max(_keq(["PH3", "H2O", "PO", "H2"], [-2, -2, 2, 5],
                                  gases_delta_g_i, p_bar, t, qh2), 0.0)) * qh2o

    # H3PO4 -> PH3 + 4H2O + 4H2  (saturation: PH3_sat = k_eq_h3po4 / H2O^4).
    if t > 0 and p_bar > 0:
        k_eq_h3po4 = _safe_exp((condensates_delta_g_i[cid_h3po4]
                                - gases_delta_g_i[idx_ph3]
                                - 4.0 * gases_delta_g_i[idx_h2o]) * 1e3
                               / (CST_R * t)) * qh2 ** 4 / p_bar
    else:
        k_eq_h3po4 = 0.0

    term = 1.0 + k_eq_ph2 + k_eq_ph3 + k_eq_po

    # Partition the fixed P budget: 2*k_eq_p2*PH3^2 + term*PH3 - P_elem = 0.
    if k_eq_p2 <= 0.0 or term ** 2 > 1e6 * p_elem * k_eq_p2:
        vmr[idx_ph3] = p_elem / term if term > 0 else 0.0
    else:
        vmr[idx_ph3] = (-term + math.sqrt(term ** 2 + 8.0 * p_elem * k_eq_p2)) \
                       / (4.0 * k_eq_p2)

    # H3PO4 condensation (equilibrium only) -- the low-T phosphorus sink.
    if at_equilibrium and qh2o > 0.0:
        ph3_sat = k_eq_h3po4 / qh2o ** 4
        if ph3_sat < vmr[idx_ph3]:
            if not is_condensed[cid_h3po4]:
                # NB the Fortran marks NH4Cl here (a copy-paste typo); the PH3
                # cap and budget update below are unaffected by which flag is set.
                is_condensed[cid_h3po4] = True
            vmr[idx_ph3] = ph3_sat
            gas_element_abd[iP, ip] = vmr[idx_ph3] * (term + 2.0 * k_eq_p2 * vmr[idx_ph3])

    # Back out the other P-bearing species from PH3.
    vmr[idx_p]   = k_eq_ph3 * vmr[idx_ph3]
    vmr[idx_p2]  = k_eq_p2  * vmr[idx_ph3] ** 2
    vmr[idx_ph2] = k_eq_ph2 * vmr[idx_ph3]
    vmr[idx_po]  = k_eq_po  * vmr[idx_ph3]

    return vmr
