"""
Transmission (transit) spectrum calculation.

Mirrors the Fortran ``transit_spectrum`` module (transit_spectrum.f90).
The geometry follows the standard secant-of-the-airmass approach: each
line-of-sight crossing a tangent layer ``l`` is decomposed into the
contributions of all layers ``j ≥ l``.
"""

from __future__ import annotations

import numpy as np


def calculate_transit_spectrum(
    tau:           np.ndarray,    # (n_levels, n_wavenumbers, n_g)
    tau_rayleigh:  np.ndarray,    # (n_levels, n_wavenumbers)
    weights_k:     np.ndarray,    # (n_g,)
    z:             np.ndarray,    # (n_levels,) altitudes in km
    target_radius: float,         # (m or km) consistent with z, see notes
    *,
    calculate_contribution: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the wavelength-dependent effective transit radius.

    Parameters
    ----------
    tau
        Layer-cumulative optical depth at each level, wavenumber and g-point.
    tau_rayleigh
        Rayleigh optical depth at each level and wavenumber.
    weights_k
        Quadrature weights of the k-distribution.
    z
        Altitudes of the level grid. Must be in the same length unit as
        ``target_radius`` for the squared sums to be dimensionally consistent
        (the original Fortran uses km throughout and rescales by 1e3 at the
        end to return metres).
    target_radius
        Planetary radius at z[0].
    calculate_contribution
        If True, also return the per-layer contribution function
        ``d_spectral_radius``.

    Returns
    -------
    spectral_radius     : (n_wavenumbers,) effective radius in metres
    d_spectral_radius   : (n_levels, n_wavenumbers) per-level contribution.
                          Zero everywhere when ``calculate_contribution`` is False.
    """
    print("Calculating transit spectrum...")

    n_levels      = z.size
    n_layers      = n_levels - 1
    n_wavenumbers = tau.shape[1]
    n_g           = int(weights_k.size)

    # --- Geometry: secant factor and "annulus" area for each layer ---
    sec_mat = np.zeros((n_layers, n_layers))   # sec_mat[j, l]
    area_l  = np.zeros(n_levels)

    for l in range(n_layers):
        z02 = (target_radius + z[l]) ** 2
        if l == 0:
            area_l[l] = (target_radius + z[l]) * (z[1] - z[0])
        else:
            area_l[l] = (target_radius + z[l]) * (z[l + 1] - z[l - 1])

        for j in range(l, n_layers):
            z12 = (target_radius + z[j])     ** 2
            z22 = (target_radius + z[j + 1]) ** 2
            denom = (z[j + 1] - z[j])
            sec_mat[j, l] = (np.sqrt(z22 - z02) - np.sqrt(z12 - z02)) / denom if denom != 0 else 0.0

    spectral_radius   = np.zeros(n_wavenumbers)
    d_spectral_radius = np.zeros((n_levels, n_wavenumbers))
    transmittance_l   = np.zeros(n_layers)
    tiny_floor        = float(np.finfo(np.float64).tiny)

    # --- Main spectral loop ---
    for i in range(n_wavenumbers):
        # initial geometric area (planet disc)
        spectral_radius[i] = (target_radius + z[0]) ** 2

        # delta-tau along the line of sight, per layer & per g-point
        delta_tau_g = (tau[:n_layers, i, :] - tau[1:n_levels, i, :]
                       + (tau_rayleigh[:n_layers, i] - tau_rayleigh[1:n_levels, i])[:, None])
        # shape: (n_layers, n_g)

        # transmittance of the line of sight that has its tangent in layer l
        for l in range(n_layers):
            # cumulative tau over all layers traversed by line-of-sight l
            tau_g = (delta_tau_g[l:, :] * sec_mat[l:, l, None]).sum(axis=0)   # (n_g,)
            trans_g = np.exp(-2.0 * tau_g)
            tr = float(np.dot(trans_g, weights_k))
            if tr < tiny_floor:
                tr = 0.0
            elif tr > 1.0:
                tr = 1.0
            transmittance_l[l] = tr
            spectral_radius[i] += area_l[l] * (1.0 - tr)

        # Contribution-function calculation (expensive — only if requested)
        if calculate_contribution:
            for j in range(n_layers):
                for l in range(j + 1):
                    tau_g_total = (delta_tau_g[l:, :] * sec_mat[l:, l, None]).sum(axis=0)
                    alpha_j = delta_tau_g[j, :]
                    rest    = tau_g_total - alpha_j * sec_mat[j, l]
                    trans_rest = np.exp(-2.0 * rest)
                    contrib    = (1.0 - np.exp(-2.0 * alpha_j * sec_mat[j, l])) * trans_rest
                    d_spectral_radius[j, i] += area_l[l] * float(np.dot(contrib, weights_k))

            d_spectral_radius[n_levels - 1, i] = transmittance_l[n_layers - 1]
            d_spectral_radius[:, i] = np.sqrt(np.maximum(d_spectral_radius[:, i], 0.0)) * 1e3

        spectral_radius[i] = np.sqrt(spectral_radius[i]) * 1e3   # km → m

    return spectral_radius, d_spectral_radius


__all__ = ["calculate_transit_spectrum"]
