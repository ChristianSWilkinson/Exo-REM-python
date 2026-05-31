"""
Cloud microphysics (Ackerman & Marley 2001 with Hansen 1974 size distribution).

Mirrors the Fortran ``cloud_mixing`` module (cloud_mixing.f90):
  - :func:`calculate_cloud_mixing`  : standard A&M solver
  - :func:`calculate_cloud_mixing2` : variant including a condensation
                                      timescale and coalescence

The Fortran code uses 1-based indexing; layer indices have been shifted by
−1 in Python. The "layer_clouds" argument from the caller follows the same
convention (1-based in the call site translated from Fortran). To preserve
behaviour we treat ``layer_clouds`` as a *0-based* index of the bottom-most
cloud level. The caller in ``exorem_main.py`` passes this in already.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .physics import CST_N_A, CST_R, PI

# ---------------------------------------------------------------------------
# Module-wide constants (mirroring Fortran ``parameter`` blocks)
# ---------------------------------------------------------------------------
_AVOCADO_MOLRAD2: float = 12032.12883
_ANUEFF:          float = 0.3
_TINY:            float = 2.0 * float(np.finfo(np.float64).tiny)
_HUGE_LOG:        float = math.log(np.finfo(np.float64).max)

_AMOLRAD:           float = 1.4135e-10
_RADIUS_SED_MIN:    float = 1.37e-7
_RADIUS_SED_MAX:    float = 1.37e-4
_F_COAL:            float = 2.0
_ALPHA_COAL:        float = 1.0
_TINIEST32:         float = float(np.finfo(np.float32).tiny)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_log(arg: float, fallback: float = 1e30) -> float:
    """log() that returns +/- fallback rather than raising on bad input."""
    if arg <= 0.0:
        return -fallback
    if arg == float("inf"):
        return fallback
    return math.log(arg)


def _safe_exp(x: float) -> float:
    """
    exp(x) that mirrors Fortran's silent overflow/underflow behaviour:
    underflows to 0.0 and overflows to a large finite value (~1e308),
    rather than raising :class:`OverflowError` like :func:`math.exp`.
    """
    if x <= -745.0:           # exp(-745) ≈ 5e-324 (smallest subnormal)
        return 0.0
    if x >= 709.0:            # exp(709) ≈ 8.2e307 (largest representable)
        return 1.0e308
    return math.exp(x)


# ===========================================================================
# Standard Ackerman & Marley solver
# ===========================================================================

def calculate_cloud_mixing(
    nlay:           int,
    p:              np.ndarray,    # (n_levels,) mbar
    t:              np.ndarray,    # (n_levels,)
    pl:             np.ndarray,    # (n_layers,) layer pressures (mbar)
    tl:             np.ndarray,    # (n_layers,) layer temperatures
    gl:             np.ndarray,    # (n_layers,) layer gravities (m/s^2)
    ml:             np.ndarray,    # (n_layers,) layer mean molar masses (kg/mol)
    layer_clouds:   int,           # 0-based index of the bottom cloud level
    pbot:           float,
    qsat_cloud:     np.ndarray,    # (n_layers,) saturation mass mixing ratio
    q0:             float,         # surface (deep) value of the vapour MMR
    kzz:            np.ndarray,    # (n_layers,) eddy diffusion coefficient (cm^2/s)
    cloud_mode:     str,
    eddy_mode:      str,
    fsed:           float,
    radius:         np.ndarray,    # (n_layers,) particle radius (m), in/out
    rho_c:          float,         # condensate density (g/cm^3 in F90 convention)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the cloud mass mixing ratio profile, the particle radius profile,
    the sedimentation velocity ``vsed`` and the eddy mixing velocity ``vmixing``.

    Returns
    -------
    q_cloud   : (n_layers,) cloud mass mixing ratio
    radius    : (n_layers,) updated particle radius (m)
    vsed      : (n_layers,) sedimentation velocity (cm/s in F90 units)
    vmixing   : (n_layers,) eddy mixing velocity
    """
    print("Calculating cloud mixing...")

    radius = np.array(radius, dtype=float, copy=True)
    n_layers = nlay
    sg = math.sqrt(math.log(1.0 + _ANUEFF))   # Hansen 1974

    variation_constante = True

    q_cloud = np.zeros(n_layers)
    q_vap   = np.full(n_layers, q0)
    dq_cond = np.zeros(n_layers)

    pl2 = pl[:n_layers] * 100.0                                  # mbar → Pa
    rho = p[:n_layers] * 100.0 / (CST_R * t[:n_layers] / ml[:n_layers] * 1000.0)
    Hscale = 1e5 * (CST_R * tl[:n_layers] / ml[:n_layers] / gl[:n_layers])

    vsed       = np.zeros(n_layers)
    vmixing    = np.zeros(n_layers)
    radius_sed = np.zeros(n_layers)

    # --- Sedimentation velocity in each layer ---
    for i in range(n_layers):
        visc = 2.0123e-7 * tl[i] ** (2.0 / 3.0)        # dynamic viscosity (H2)
        a = math.sqrt(2.0) / 2.0 * CST_R / (4.0 * PI * _AVOCADO_MOLRAD2)
        b = (2.0 / 9.0) * (rho_c - rho[i]) * gl[i] * 1e-2 / visc

        if eddy_mode == "infinity" and cloud_mode == "fixedRadius":
            vsed[i] = 0.0
        elif radius[i] > 0.0:
            radius_sed[i] = radius[i] * _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2)
            vsed[i] = (b * radius_sed[i] ** 2
                       * (1.0 + 4.0 / 3.0 * (a * tl[i] / pl2[i]) / radius_sed[i]))

        vmixing[i] = 1e-4 * kzz[i] / Hscale[i]

        if cloud_mode == "fixedSedimentation":
            vsed[i] = vmixing[i] * fsed
            c = 4.0 / 3.0 * (a * tl[i] / pl2[i])
            radius_sed[i] = 0.5 * (-c + math.sqrt(c ** 2 + 4.0 * fsed * vmixing[i] / b))
            radius[i] = radius_sed[i] * _safe_exp(-(1.4 + 1.0) / 2.0 * sg ** 2)
            radius[i] = max(radius[i], 1e-8)

        elif cloud_mode == "fixedRadiusCondensation":
            c = 4.0 / 3.0 * (a * tl[i] / pl2[i])

            if i > layer_clouds + 1:
                radius_sed[i] = radius_sed[layer_clouds + 1]
                vsed[i] = (b * radius_sed[i] ** 2
                           * (1.0 + 4.0 / 3.0 * (a * tl[i] / pl2[i]) / radius_sed[i]))
            else:
                vsed[i] = vmixing[i] * fsed
                radius_sed[i] = 0.5 * (-c + math.sqrt(c ** 2 + 4.0 * fsed * vmixing[i] / b))

            radius[i] = radius_sed[i] * _safe_exp(-(1.4 + 1.0) / 2.0 * sg ** 2)
            radius[i] = max(radius[i], 1e-8)

    # --- Vertical loop for q_cloud ---
    if 0 < layer_clouds < n_layers:
        # Case A — cloud forms at a level inside the grid
        for i in range(layer_clouds, n_layers - 1):
            dz = (pl[i - 1] - pl[i]) * 100.0 * 100.0 / rho[i] / gl[i]
            dq_cond[i] = min(qsat_cloud[i - 1], q_vap[i - 1]) - qsat_cloud[i]

            if cloud_mode in ("fixedRadius", "fixedRadiusCondensation"):
                num = (dq_cond[i]
                       + q_cloud[i - 1] * (1.0 - 0.5 * vsed[i - 1] / (1e-4 * kzz[i - 1]) * dz))
                den = 1.0 + 0.5 * vsed[i] / (1e-4 * kzz[i]) * dz
                q_cloud[i] = num / den

                if variation_constante:
                    var_a = -0.5 / 1e-4 * (vsed[i - 1] / kzz[i - 1] + vsed[i] / kzz[i]) * dz

                    if (qsat_cloud[i] > qsat_cloud[i - 1] * (1 - 1e-12)
                            and qsat_cloud[i] < qsat_cloud[i - 1] * (1 + 1e-12)):
                        H2 = dz * math.copysign(1e10, qsat_cloud[i] - qsat_cloud[i - 1])
                    else:
                        H2 = dz / math.log(qsat_cloud[i] / qsat_cloud[i - 1])

                    H3 = 1.0 / (1.0 / H2 + 0.5 / 1e-4
                                * (vsed[i - 1] / kzz[i - 1] + vsed[i] / kzz[i]))

                    if qsat_cloud[i] > 1.0 and q_cloud[i - 1] < _TINY * (1.0 + 1e-12):
                        q_cloud[i] = 0.0
                    elif dz / H3 > _HUGE_LOG:
                        var_b = _safe_log(abs(qsat_cloud[i - 1] / H2 * H3)) + dz / H3
                        var_b = max(0.0, min(var_b, q_vap[i - 1] - qsat_cloud[i]))
                        q_cloud[i] = math.copysign(_safe_exp(var_a + var_b),
                                                   -qsat_cloud[i - 1] / H2 * H3)
                    else:
                        var_b = -qsat_cloud[i - 1] / H2 * H3 * (_safe_exp(dz / H3) - 1.0)
                        var_b = min(var_b, q_vap[i - 1] - qsat_cloud[i])
                        q_cloud[i] = _safe_exp(var_a) * (q_cloud[i - 1] + var_b)

            elif cloud_mode == "fixedSedimentation":
                num = (dq_cond[i]
                       + q_cloud[i - 1] * (1.0 - 0.5 * fsed / Hscale[i - 1] * dz))
                den = 1.0 + 0.5 * fsed / Hscale[i] * dz
                q_cloud[i] = num / den

                if variation_constante:
                    var_a = _safe_exp(-0.5 * (fsed / Hscale[i - 1] + fsed / Hscale[i]) * dz)

                    if (qsat_cloud[i] > qsat_cloud[i - 1] * (1 - 1e-12)
                            and qsat_cloud[i] < qsat_cloud[i - 1] * (1 + 1e-12)):
                        H2 = dz * math.copysign(1e10, qsat_cloud[i] - qsat_cloud[i - 1])
                    else:
                        H2 = dz / math.log(qsat_cloud[i] / qsat_cloud[i - 1])

                    H3 = 1.0 / (1.0 / H2 + 0.5 * (fsed / Hscale[i - 1] + fsed / Hscale[i]))
                    var_b = -qsat_cloud[i - 1] / H2 * H3 * (_safe_exp(dz / H3) - 1.0)
                    var_b = min(var_b, q_vap[i - 1] - qsat_cloud[i])
                    q_cloud[i] = var_a * (q_cloud[i - 1] + var_b)
            else:
                q_cloud[i] = 0.0

            q_cloud[i] = max(_TINY, q_cloud[i])
            q_cloud[i] = min(q_cloud[i], q0)
            q_cloud[i] = min(q_cloud[i], qsat_cloud[0])
            q_vap[i]   = min(qsat_cloud[i], q_cloud[i - 1] + q_vap[i - 1])

    elif layer_clouds <= 0:
        # Case B — cloud forms below the model grid; first layer comes from
        # a power-law continuation, then propagate as in case A.
        if cloud_mode in ("fixedSedimentation", "fixedRadiusCondensation"):
            q_cloud[0] = (q0 - qsat_cloud[0]) * (p[0] / pbot) ** fsed
        else:
            q_cloud[0] = 0.0

        q_cloud[0] = max(0.0, min(q0 - qsat_cloud[0], q_cloud[0]))
        q_vap[0]   = qsat_cloud[0]

        for i in range(1, n_layers - 1):
            dz = (pl[i - 1] - pl[i]) * 100.0 * 100.0 / rho[i] / gl[i]
            dq_cond[i] = max(0.0,
                             min(qsat_cloud[i - 1], q_vap[i - 1]) - qsat_cloud[i])

            if cloud_mode in ("fixedRadius", "fixedRadiusCondensation"):
                num = (dq_cond[i]
                       + q_cloud[i - 1] * (1.0 - 0.5 * vsed[i - 1] / (1e-4 * kzz[i - 1]) * dz))
                den = 1.0 + 0.5 * vsed[i] / (1e-4 * kzz[i]) * dz
                q_cloud[i] = num / den

                if variation_constante:
                    var_a = _safe_exp(-0.5 / 1e-4
                                     * (vsed[i - 1] / kzz[i - 1] + vsed[i] / kzz[i]) * dz)
                    if (qsat_cloud[i] > qsat_cloud[i - 1] * (1 - 1e-12)
                            and qsat_cloud[i] < qsat_cloud[i - 1] * (1 + 1e-12)):
                        H2 = dz * math.copysign(1e10, qsat_cloud[i] - qsat_cloud[i - 1])
                    else:
                        H2 = dz / math.log(qsat_cloud[i] / qsat_cloud[i - 1])
                    H3 = 1.0 / (1.0 / H2 + 0.5 / 1e-4
                                * (vsed[i - 1] / kzz[i - 1] + vsed[i] / kzz[i]))
                    if dz / H3 > _HUGE_LOG:
                        var_b = -np.finfo(np.float64).max
                    else:
                        var_b = -qsat_cloud[i - 1] / H2 * H3 * (_safe_exp(dz / H3) - 1.0)
                        var_b = min(var_b, q_vap[i - 1] - qsat_cloud[i])
                    q_cloud[i] = var_a * (q_cloud[i - 1] + var_b)

            elif cloud_mode == "fixedSedimentation":
                num = (dq_cond[i]
                       + q_cloud[i - 1] * (1.0 - 0.5 * fsed / Hscale[i - 1] * dz))
                den = 1.0 + 0.5 * fsed / Hscale[i] * dz
                q_cloud[i] = num / den

                if variation_constante:
                    var_a = _safe_exp(-0.5 * (fsed / Hscale[i - 1] + fsed / Hscale[i]) * dz)
                    if (qsat_cloud[i] > qsat_cloud[i - 1] * (1 - 1e-12)
                            and qsat_cloud[i] < qsat_cloud[i - 1] * (1 + 1e-12)):
                        H2 = dz * math.copysign(1e10, qsat_cloud[i] - qsat_cloud[i - 1])
                    else:
                        H2 = dz / math.log(qsat_cloud[i] / qsat_cloud[i - 1])
                    H3 = 1.0 / (1.0 / H2 + 0.5 * (fsed / Hscale[i - 1] + fsed / Hscale[i]))
                    var_b = -qsat_cloud[i - 1] / H2 * H3 * (_safe_exp(dz / H3) - 1.0)
                    var_b = min(var_b, q_vap[i - 1] - qsat_cloud[i])
                    q_cloud[i] = var_a * (q_cloud[i - 1] + var_b)
            else:
                q_cloud[i] = 0.0

            q_cloud[i] = max(_TINY, q_cloud[i])
            q_cloud[i] = min(q_cloud[i], q0)
            q_vap[i]   = min(qsat_cloud[i], q_cloud[i - 1] + q_vap[i - 1])

        q_cloud[0] = q_cloud[1]

    # --- Vapor-conservation guard ---
    for i in range(1, n_layers - 1):
        q_cloud[i] = min(q_cloud[i], qsat_cloud[i - 1] + q_cloud[i - 1])

    # --- Cloud MMR at the condensation level ---
    # r31: was `< n_layers`, but q_cloud has shape (n_layers,) and the
    # block accesses q_cloud[layer_clouds + 1], so the safe bound is
    # `< n_layers - 1`.  Mirrors Fortran cloud_mixing.f90 line 220:
    # `if (layer_clouds >= 2 .and. layer_clouds < nlay)` — in Fortran's
    # 1-based indexing, `layer_clouds < nlay` means layer_clouds + 1 <= nlay
    # which translates to `layer_clouds + 1 <= n_layers - 1` here.
    if 1 <= layer_clouds < n_layers - 1:
        ratio = (pbot - p[layer_clouds + 1]) / (p[layer_clouds] - p[layer_clouds + 1])
        ratio = max(min(1.0, ratio), 0.0)

        if q_cloud[layer_clouds + 1] < _TINY * (1.0 + 1e-12):
            pass   # keep existing q_cloud[layer_clouds]
        elif ratio > np.finfo(np.float64).tiny / max(q_cloud[layer_clouds + 1], _TINY):
            q_cloud[layer_clouds] = 0.5 * q_cloud[layer_clouds + 1] * ratio
        else:
            q_cloud[layer_clouds] = 0.5 * q_cloud[layer_clouds + 1]
    elif layer_clouds == 0:
        q_cloud[layer_clouds] = 0.5 * q_cloud[layer_clouds + 1]
    # else: layer_clouds >= n_layers - 1, i.e. condensation only at or above
    # the topmost layer — nothing useful we can do here, leave q_cloud alone
    # (matches Fortran's implicit no-op when layer_clouds == nlay).

    return q_cloud, radius, vsed, vmixing


# ===========================================================================
# Variant including a condensation timescale + coalescence
# ===========================================================================

def calculate_cloud_mixing2(
    nlay:           int,
    p:              np.ndarray,
    t:              np.ndarray,
    pl:             np.ndarray,
    tl:             np.ndarray,
    gl:             np.ndarray,
    ml:             np.ndarray,
    layer_clouds:   int,
    pbot:           float,
    qsat_cloud:     np.ndarray,
    q0:             float,
    kzz:            np.ndarray,
    cloud_mode:     str,           # unused but kept for signature symmetry
    eddy_mode:      str,           # unused but kept for signature symmetry
    stick_ef0:      float,
    radius:         np.ndarray,
    rho_c:          float,
    smax:           float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Alternative cloud-mixing solver including a condensation timescale and a
    coalescence timescale (Charnay et al. 2018).

    Returns
    -------
    q_cloud, radius, vsed, vmixing  (each of shape (n_layers,))
    """
    print("Calculating cloud mixing (alt)...")

    radius = np.array(radius, dtype=float, copy=True)
    n_layers = nlay
    sg = math.sqrt(math.log(1.0 + _ANUEFF))
    variation_constante = True
    coalescence         = True

    stick_ef = min(1.0, stick_ef0)
    stick_ef = max(1e-6, stick_ef)

    q_cloud = np.zeros(n_layers)
    q_vap   = np.full(n_layers, q0)

    pl2 = pl[:n_layers] * 100.0
    rho = p[:n_layers] * 100.0 / (CST_R * t[:n_layers] / ml[:n_layers] * 1000.0)
    Hscale = 1e5 * (CST_R * tl[:n_layers] / ml[:n_layers] / gl[:n_layers])

    radius_sed = radius[:n_layers] * _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2)
    vsed       = np.zeros(n_layers)
    tau_mixing = 1e4 * Hscale ** 2 / kzz[:n_layers]
    rho_s      = rho * qsat_cloud[:n_layers]
    vmixing    = 1e-4 * kzz[:n_layers] / Hscale

    # First pass — initial sedimentation velocity
    for i in range(n_layers):
        visc = 2.0123e-7 * tl[i] ** 0.66
        a = math.sqrt(2.0) / 2.0 * CST_R / (4.0 * PI * _AVOCADO_MOLRAD2)
        b = (2.0 / 9.0) * (rho_c - rho[i]) * gl[i] * 1e-2 / visc
        vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0) * (a * tl[i] / pl2[i]) / radius_sed[i])

    tau_sed         = np.zeros(n_layers)
    tau_sed_mixing  = np.zeros(n_layers)
    tau_cond        = np.zeros(n_layers)
    tau_coal        = np.zeros(n_layers)
    vsed_cond       = np.zeros(n_layers)
    tau_sed_cond    = np.zeros(n_layers)
    tau_coal_cond   = np.zeros(n_layers)

    # Outer loop refining radius/vsed/q_cloud self-consistently
    for ii in range(4):
        a = math.sqrt(2.0) / 2.0 * CST_R / (4.0 * PI * _AVOCADO_MOLRAD2)

        for i in range(n_layers):
            visc = 2.0123e-7 * tl[i] ** (2.0 / 3.0)
            b  = (2.0 / 9.0) * (rho_c - rho[i]) * gl[i] * 1e-2 / visc
            aa = b
            bb = a * b * (4.0 / 3.0) * tl[i] / pl2[i]

            radius_sed[i] = math.sqrt(
                tau_mixing[i] / rho[i]
                * (2.0 * _F_COAL * visc * rho_s[i] * smax / rho_c)
            )
            if i == 0 and layer_clouds < 0:
                radius_sed[i] = math.sqrt(
                    tau_mixing[i] * (2.0 * _F_COAL * visc
                                     * math.sqrt(qsat_cloud[i] * q0) * smax / rho_c)
                )
            radius_sed[i] = max(radius_sed[i], _RADIUS_SED_MIN)
            radius_sed[i] = min(radius_sed[i], _RADIUS_SED_MAX)
            radius[i] = radius_sed[i] / _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2)

            vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0)
                                                 * (a * tl[i] / pl2[i]) / radius_sed[i])
            aKn = (CST_R / CST_N_A * tl[i]
                   / (math.sqrt(2.0) * PI * _AMOLRAD ** 2 * pl2[i])
                   / radius_sed[i])

            if aKn >= _ALPHA_COAL:
                radius_sed[i] = (tau_mixing[i]
                                 * (3.0 * _ALPHA_COAL * _F_COAL * rho_s[i] * smax / 2.0 / rho_c)
                                 * math.sqrt(2.0 * CST_R * tl[i] / PI / ml[i] * 1e3))
                if i == 0 and layer_clouds < 0:
                    radius_sed[i] = (tau_mixing[i]
                                     * (3.0 * _ALPHA_COAL * _F_COAL
                                        * math.sqrt(qsat_cloud[i] * q0) * rho[i]
                                        * smax / 2.0 / rho_c)
                                     * math.sqrt(2.0 * CST_R * tl[i] / PI / ml[i] * 1e3))
                radius_sed[i] = max(radius_sed[i], _RADIUS_SED_MIN)
                radius_sed[i] = min(radius_sed[i], _RADIUS_SED_MAX)

                vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0)
                                                     * (a * tl[i] / pl2[i]) / radius_sed[i])
                tau_sed_mixing[i] = Hscale[i] / max(1e-7, abs(vsed[i] - 1e-4 * kzz[i] / Hscale[i]))
                tau_cond[i] = tau_mixing[i]
                tau_sed[i]  = Hscale[i] / vsed[i]

                # Newton: shrink radius until t_sed >= t_cond
                if tau_sed[i] < tau_cond[i]:
                    r0 = radius_sed[i]
                    for _ in range(100):
                        fnewton  = (tau_cond[i] / r0 * aa * radius_sed[i] ** 3
                                    + tau_cond[i] / r0 * bb * radius_sed[i] ** 2
                                    - Hscale[i])
                        dfnewton = (3.0 * tau_cond[i] / r0 * aa * radius_sed[i] ** 2
                                    + 2.0 * tau_cond[i] / r0 * bb * radius_sed[i])
                        if dfnewton == 0.0:
                            break
                        radius_sed[i] -= fnewton / dfnewton
                        if abs(fnewton / dfnewton / radius_sed[i]) <= 1e-3:
                            break
                    tau_cond[i] *= radius_sed[i] / r0
                    vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0)
                                                         * (a * tl[i] / pl2[i]) / radius_sed[i])
                    tau_sed[i]        = Hscale[i] / vsed[i]
                    tau_sed_mixing[i] = Hscale[i] / max(1e-7, abs(vsed[i] - 1e-4 * kzz[i] / Hscale[i]))

                # Coalescence Newton step
                tau_coal[i] = (4.0 * radius_sed[i] * rho_c
                               / (3.0 * stick_ef * _ALPHA_COAL * vsed[i] * rho[i]
                                  * max(_TINIEST32, q_cloud[i])))
                if coalescence and tau_coal[i] < tau_cond[i]:
                    r0 = radius_sed[i]
                    for _ in range(100):
                        fnewton  = (tau_cond[i] / r0 * radius_sed[i]
                                    * (aa * radius_sed[i] ** 2 + bb * radius_sed[i])
                                    - tau_coal[i] * vsed[i] * radius_sed[i] / r0)
                        dfnewton = (3.0 * tau_cond[i] / r0 * aa * radius_sed[i] ** 2
                                    + 2.0 * tau_cond[i] / r0 * bb * radius_sed[i]
                                    - tau_coal[i] * vsed[i] / r0)
                        if dfnewton == 0.0:
                            break
                        radius_sed[i] -= fnewton / dfnewton
                        if abs(fnewton / dfnewton / radius_sed[i]) <= 1e-3:
                            break
                    tau_cond[i] *= radius_sed[i] / r0
                    vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0)
                                                         * (a * tl[i] / pl2[i]) / radius_sed[i])
                    tau_sed[i]        = Hscale[i] / vsed[i]
                    tau_sed_mixing[i] = Hscale[i] / max(1e-7, abs(vsed[i] - 1e-4 * kzz[i] / Hscale[i]))
                    tau_coal[i] = (4.0 * radius_sed[i] * rho_c
                                   / (3.0 * _ALPHA_COAL * vsed[i] * rho[i]
                                      * max(_TINIEST32, q_cloud[i])))
            else:
                tau_cond[i] = tau_mixing[i]
                tau_sed[i]  = Hscale[i] / vsed[i]

                if tau_sed[i] < tau_cond[i]:
                    r0 = radius_sed[i]
                    for _ in range(100):
                        fnewton  = (tau_cond[i] / r0 ** 2 * aa * radius_sed[i] ** 4
                                    + tau_cond[i] / r0 ** 2 * bb * radius_sed[i] ** 3
                                    - Hscale[i])
                        dfnewton = (4.0 * tau_cond[i] / r0 ** 2 * aa * radius_sed[i] ** 3
                                    + 3.0 * tau_cond[i] / r0 ** 2 * bb * radius_sed[i] ** 2)
                        if dfnewton == 0.0:
                            break
                        radius_sed[i] -= fnewton / dfnewton
                        if abs(fnewton / dfnewton / radius_sed[i]) <= 1e-3:
                            break
                    tau_cond[i] *= (radius_sed[i] / r0) ** 2
                    vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0)
                                                         * (a * tl[i] / pl2[i]) / radius_sed[i])
                    tau_sed[i]        = Hscale[i] / vsed[i]
                    tau_sed_mixing[i] = Hscale[i] / max(1e-7, abs(vsed[i] - 1e-4 * kzz[i] / Hscale[i]))

            tau_coal[i] = (4.0 * radius_sed[i] * rho_c
                           / (3.0 * stick_ef * _ALPHA_COAL * vsed[i] * rho[i]
                              * max(_TINIEST32, q_cloud[i])))
            if coalescence and tau_coal[i] < tau_cond[i]:
                r0 = radius_sed[i]
                for _ in range(100):
                    fnewton  = (tau_cond[i] / r0 ** 2 * radius_sed[i] ** 2
                                * (aa * radius_sed[i] ** 2 + bb * radius_sed[i])
                                - tau_coal[i] * vsed[i] * radius_sed[i] / r0)
                    dfnewton = (4.0 * tau_cond[i] / r0 ** 2 * aa * radius_sed[i] ** 3
                                + 3.0 * tau_cond[i] / r0 ** 2 * bb * radius_sed[i] ** 2
                                - tau_coal[i] * vsed[i] / r0)
                    if dfnewton == 0.0:
                        break
                    radius_sed[i] -= fnewton / dfnewton
                    if abs(fnewton / dfnewton / radius_sed[i]) <= 1e-3:
                        break
                tau_cond[i] *= (radius_sed[i] / r0) ** 2
                vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0)
                                                     * (a * tl[i] / pl2[i]) / radius_sed[i])
                tau_sed[i]        = Hscale[i] / vsed[i]
                tau_sed_mixing[i] = Hscale[i] / max(1e-7, abs(vsed[i] - 1e-4 * kzz[i] / Hscale[i]))
                tau_coal[i] = (4.0 * radius_sed[i] * rho_c
                               / (3.0 * _ALPHA_COAL * vsed[i] * rho[i]
                                  * max(_TINIEST32, q_cloud[i])))

            tau_sed_mixing[i] = Hscale[i] / max(1e-7, abs(vsed[i] - 1e-4 * kzz[i] / Hscale[i]))
            radius[i] = radius_sed[i] / _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2)

            vsed_cond[i]     = vsed[i]
            tau_sed_cond[i]  = tau_sed[i]
            tau_coal_cond[i] = tau_coal[i]

        # --- Propagation of q_cloud + radius_sed correction ---
        if 0 < layer_clouds < n_layers:
            for i in range(layer_clouds + 1, n_layers - 1):
                visc = 2.0123e-7 * tl[i] ** (2.0 / 3.0)
                b = (2.0 / 9.0) * (rho_c - rho[i]) * gl[i] * 1e-2 / visc

                dz = (pl[i - 1] - pl[i]) * 100.0 * 100.0 / rho[i] / gl[i]
                var_a  = _safe_exp(-0.5 / 1e-4 * (vsed[i - 1] / kzz[i - 1]
                                                  + vsed[i - 1] / kzz[i]) * dz)
                var_a2 = _safe_exp(-0.5 / 1e-4 * (vsed_cond[i - 1] / kzz[i - 1]
                                                  + vsed_cond[i] / kzz[i]) * dz)

                if coalescence:
                    var_a = _safe_exp(-0.5 / 1e-4 * (
                        vsed[i - 1] / kzz[i - 1] * (1 + tau_sed[i - 1] / tau_coal[i - 1])
                        + vsed[i - 1] / kzz[i]   * (1 + tau_sed[i - 1] / tau_coal[i - 1])
                    ) * dz)
                    var_a2 = _safe_exp(-0.5 / 1e-4 * (
                        vsed_cond[i - 1] / kzz[i - 1] * (1 + tau_sed_cond[i - 1] / tau_coal_cond[i - 1])
                        + vsed_cond[i]   / kzz[i]     * (1 + tau_sed_cond[i]     / tau_coal_cond[i])
                    ) * dz)

                aa = b
                bb = a * b * (4.0 / 3.0) * tl[i] / pl2[i]

                if (qsat_cloud[i] > qsat_cloud[i - 1] * (1 - 1e-12)
                        and qsat_cloud[i] < qsat_cloud[i - 1] * (1 + 1e-12)):
                    H2 = dz * math.copysign(1e10, qsat_cloud[i] - qsat_cloud[i - 1])
                else:
                    # Guard math.log: skip layer if ratio is non-positive
                    # (can happen during early iterations before T(p) stabilises).
                    qratio = (qsat_cloud[i] / qsat_cloud[i - 1]
                              if qsat_cloud[i - 1] > 0 else 0.0)
                    if qratio <= 0 or not math.isfinite(qratio):
                        continue
                    H2 = dz / math.log(qratio)

                if coalescence:
                    H3 = 1.0 / (1.0 / H2 + 0.5 / 1e-4 * (
                        vsed_cond[i - 1] / kzz[i - 1] * (1 + tau_sed_cond[i - 1] / tau_coal_cond[i - 1])
                        + vsed_cond[i]   / kzz[i]     * (1 + tau_sed_cond[i]     / tau_coal_cond[i])))
                else:
                    H3 = 1.0 / (1.0 / H2 + 0.5 / 1e-4 * (
                        vsed_cond[i - 1] / kzz[i - 1] + vsed_cond[i] / kzz[i]))

                var_b = -qsat_cloud[i - 1] / H2 * H3 * (_safe_exp(dz / H3) - 1.0)
                var_b = min(var_b, q_vap[i - 1] - qsat_cloud[i])
                q_cloud[i] = var_a * q_cloud[i - 1] + var_a2 * var_b

                cc = (b * (var_a2 * var_b * radius_sed[i] ** 2
                           + var_a  * q_cloud[i - 1] * radius_sed[i - 1] ** 2)
                      + bb * (var_a2 * var_b * radius_sed[i]
                              + var_a * q_cloud[i - 1] * radius_sed[i - 1]))
                denom = var_a2 * var_b + var_a * q_cloud[i - 1]
                cc = -cc / denom if denom != 0.0 else 0.0

                disc = bb ** 2 - 4.0 * aa * cc
                if disc > 0.0:
                    radius_sed[i] = (math.sqrt(disc) - bb) / 2.0 / aa
                else:
                    radius_sed[i] = _RADIUS_SED_MIN

                radius_sed[i] = max(radius_sed[i], _RADIUS_SED_MIN)
                radius_sed[i] = min(radius_sed[i], _RADIUS_SED_MAX)
                radius[i] = radius_sed[i] / _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2)
                radius[i] = max(radius[i],
                                _RADIUS_SED_MIN / _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2))
                radius[i] = min(radius[i],
                                _RADIUS_SED_MAX / _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2))

                vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0)
                                                     * (a * tl[i] / pl2[i]) / radius_sed[i])
                tau_sed_mixing[i] = Hscale[i] / max(1e-7, abs(vsed[i] - 1e-4 * kzz[i] / Hscale[i]))
                tau_coal[i] = (4.0 * radius_sed[i] * rho_c
                               / (3.0 * _ALPHA_COAL * vsed[i] * rho[i]
                                  * max(_TINIEST32, q_cloud[i])))

                q_cloud[i] = max(1e-30, q_cloud[i])
                q_cloud[i] = min(q_cloud[i], q0)
                q_cloud[i] = min(q_cloud[i], qsat_cloud[0])
                q_vap[i]   = min(qsat_cloud[i], q_cloud[i - 1] + q_vap[i - 1])

        elif layer_clouds <= 0:
            # When the cloud never reaches saturation anywhere in the column,
            # pbot is zero (or the air parcel never condensed).  In that case
            # there is nothing to compute — q_cloud / q_vap / radius arrays
            # stay at their initial values and we skip the rest of this branch
            # to avoid divide-by-zero and overflow in the cloud-physics math.
            if pbot <= 0.0:
                # Nothing condenses: leave q_cloud, q_vap, radius_sed at zero.
                pass
            else:
                q_cloud[0] = (q0 - qsat_cloud[0]) * (p[0] / pbot) ** 4.0
                q_cloud[0] = max(0.0, min(q0 - qsat_cloud[0], q_cloud[0]))

                for i in range(1, n_layers - 1):
                    visc = 2.0123e-7 * tl[i] ** (2.0 / 3.0)
                    b = (2.0 / 9.0) * (rho_c - rho[i]) * gl[i] * 1e-2 / visc
                    aa = b
                    bb = a * b * (4.0 / 3.0) * tl[i] / pl2[i]

                    dz = (pl[i - 1] - pl[i]) * 100.0 * 100.0 / rho[i] / gl[i]
                    var_a  = _safe_exp(-0.5 / 1e-4 * (vsed[i - 1] / kzz[i - 1]
                                                      + vsed[i - 1] / kzz[i]) * dz)
                    var_a2 = _safe_exp(-0.5 / 1e-4 * (vsed_cond[i - 1] / kzz[i - 1]
                                                      + vsed_cond[i] / kzz[i]) * dz)
                    if coalescence:
                        var_a = _safe_exp(-0.5 / 1e-4 * (
                            vsed[i - 1] / kzz[i - 1] * (1 + tau_sed[i - 1] / tau_coal[i - 1])
                            + vsed[i - 1] / kzz[i]   * (1 + tau_sed[i - 1] / tau_coal[i - 1])) * dz)
                        var_a2 = _safe_exp(-0.5 / 1e-4 * (
                            vsed_cond[i - 1] / kzz[i - 1] * (1 + tau_sed_cond[i - 1] / tau_coal_cond[i - 1])
                            + vsed_cond[i]   / kzz[i]     * (1 + tau_sed_cond[i]     / tau_coal_cond[i])) * dz)

                    if (qsat_cloud[i] > qsat_cloud[i - 1] * (1 - 1e-12)
                            and qsat_cloud[i] < qsat_cloud[i - 1] * (1 + 1e-12)):
                        H2 = dz * math.copysign(1e10, qsat_cloud[i] - qsat_cloud[i - 1])
                    else:
                        H2 = dz / math.log(qsat_cloud[i] / qsat_cloud[i - 1])

                    if coalescence:
                        H3 = 1.0 / (1.0 / H2 + 0.5 / 1e-4 * (
                            vsed_cond[i - 1] / kzz[i - 1] * (1 + tau_sed_cond[i - 1] / tau_coal_cond[i - 1])
                            + vsed_cond[i]   / kzz[i]     * (1 + tau_sed_cond[i]     / tau_coal_cond[i])))
                    else:
                        H3 = 1.0 / (1.0 / H2 + 0.5 / 1e-4 * (
                            vsed_cond[i - 1] / kzz[i - 1] + vsed_cond[i] / kzz[i]))

                    var_b = -qsat_cloud[i - 1] / H2 * H3 * (_safe_exp(dz / H3) - 1.0)
                    var_b = min(var_b, q_vap[i - 1] - qsat_cloud[i])
                    q_cloud[i] = var_a * q_cloud[i - 1] + var_a2 * var_b

                    cc = (aa * (var_a2 * var_b * radius_sed[i] ** 2
                                + var_a * q_cloud[i - 1] * radius_sed[i - 1] ** 2)
                          + bb * (var_a2 * var_b * radius_sed[i]
                                  + var_a * q_cloud[i - 1] * radius_sed[i - 1]))
                    denom = var_a2 * var_b + var_a * q_cloud[i - 1]
                    cc = -cc / denom if denom != 0.0 else 0.0
                    disc = bb ** 2 - 4.0 * aa * cc
                    if disc < 0.0:
                        radius_sed[i] = _RADIUS_SED_MIN
                    else:
                        radius_sed[i] = (math.sqrt(disc) - bb) / 2.0 / aa

                    radius_sed[i] = max(radius_sed[i], _RADIUS_SED_MIN)
                    radius_sed[i] = min(radius_sed[i], _RADIUS_SED_MAX)
                    radius[i] = radius_sed[i] / _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2)
                    radius[i] = max(radius[i],
                                    _RADIUS_SED_MIN / _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2))
                    radius[i] = min(radius[i],
                                    _RADIUS_SED_MAX / _safe_exp((1.4 + 1.0) / 2.0 * sg ** 2))

                    vsed[i] = b * radius_sed[i] ** 2 * (1.0 + (4.0 / 3.0)
                                                         * (a * tl[i] / pl2[i]) / radius_sed[i])
                    tau_coal[i] = (4.0 * radius_sed[i] * rho_c
                                   / (3.0 * stick_ef * _ALPHA_COAL * vsed[i] * rho[i]
                                      * max(_TINIEST32, q_cloud[i])))
                    q_cloud[i] = max(_TINIEST32, q_cloud[i])
                    q_cloud[i] = min(q_cloud[i], q0)
                    q_vap[i]   = min(qsat_cloud[i], q_cloud[i - 1] + q_vap[i - 1])

                q_cloud[0] = q_cloud[1]

        # vapor conservation guard
        for i in range(1, n_layers - 1):
            q_cloud[i] = min(q_cloud[i], q_vap[i - 1] + q_cloud[i - 1])

        # r31: was `< n_layers`, but q_cloud has shape (n_layers,) and the
        # block accesses q_cloud[layer_clouds + 1], so the safe bound is
        # `< n_layers - 1` (matches Fortran cloud_mixing.f90 line 429-430:
        # `if (layer_clouds > 1 .and. layer_clouds < nlay)`).
        if 1 <= layer_clouds < n_layers - 1:
            ratio = (pbot - p[layer_clouds + 1]) / (p[layer_clouds] - p[layer_clouds + 1])
            ratio = max(min(1.0, ratio), 0.0)
            q_cloud[layer_clouds] = 0.5 * q_cloud[layer_clouds + 1] * ratio
        elif layer_clouds == 0:
            q_cloud[layer_clouds] = 0.5 * q_cloud[layer_clouds + 1]
        # else: layer_clouds >= n_layers - 1 — see note in calculate_cloud_mixing.

    return q_cloud, radius, vsed, vmixing


__all__ = ["calculate_cloud_mixing", "calculate_cloud_mixing2"]
