"""
Data containers replacing the Fortran module-level ``save`` state.

Each of the Fortran modules ``atmosphere``, ``cloud``, ``exorem_retrieval``,
``light_source``, ``species``, ``spectrometrics``, ``target`` and
``thermodynamics`` becomes a single ``@dataclass`` here.  Attribute names
mirror the original Fortran variable names — typically with the leading
module prefix stripped (e.g. ``light_source_radius`` → ``light.radius``).

Mirrors the Fortran ``objects`` file (objects.f90).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# File-naming constants
# ---------------------------------------------------------------------------
FILE_NAME_SIZE: int = 255
ELEMENT_SYMBOL_SIZE: int = 4
SPECIES_NAME_SIZE: int = 32


# ===========================================================================
# Atmosphere
# ===========================================================================

@dataclass
class Atmosphere:
    """
    Atmospheric structure and bulk properties.

    Notes
    -----
    Arrays are sized either on the *level* grid (``n_levels``) or on the
    *layer* grid (``n_layers = n_levels − 1``).
    """
    # Eddy diffusion mode:
    #   'constant'           : uniform K_zz across the grid
    #   'Ackerman'           : Ackerman & Marley 2001
    #   'AckermanConvective' : adds convective contribution
    #   'infinity'           : as 'AckermanConvective' but K_zz → ∞ for clouds
    eddy_mode: str = "Ackerman"

    # Vertical grid
    n_layers: int = 0
    n_levels: int = 0

    # Bulk composition / pressure range
    h2_vmr:       float = 0.0
    he_vmr:       float = 0.0
    metallicity:  float = 1.0
    pressure_min: float = 1e-6   # mbar
    pressure_max: float = 1e3    # mbar
    z_vmr:        float = 0.0

    # 1-D profiles
    eddy_diffusion_coefficient: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gravities_layers:           np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale_height:               np.ndarray = field(default_factory=lambda: np.zeros(0))
    pressures:                  np.ndarray = field(default_factory=lambda: np.zeros(0))   # n_levels
    temperatures:               np.ndarray = field(default_factory=lambda: np.zeros(0))   # n_levels
    pressures_layers:           np.ndarray = field(default_factory=lambda: np.zeros(0))   # n_layers
    temperatures_layers:        np.ndarray = field(default_factory=lambda: np.zeros(0))   # n_layers
    molar_masses_layers:        np.ndarray = field(default_factory=lambda: np.zeros(0))
    z:                          np.ndarray = field(default_factory=lambda: np.zeros(0))   # altitude (km)

    # Optical depths
    tau_rayleigh: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tau:          np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))

    def check_eddy_mode(self) -> None:
        """Validate ``eddy_mode`` against the allowed list."""
        allowed = ("constant", "Ackerman", "AckermanConvective", "infinity")
        if self.eddy_mode not in allowed:
            raise ValueError(
                f"Atmosphere: eddy mode '{self.eddy_mode}' not implemented "
                f"(allowed: {', '.join(allowed)})")


# ===========================================================================
# Retrieval parameters
# ===========================================================================

@dataclass
class ExoremRetrieval:
    """Convergence and smoothing parameters for the RC equilibrium solver."""
    chemistry_iteration_interval: int = 1
    cloud_iteration_interval:     int = 1
    n_burn_iterations:            int = 0
    n_iterations:                 int = 20
    n_non_adiabatic_iterations:   int = 5
    retrieval_level_top:          int = 1
    retrieval_level_bottom:       int = 0

    retrieval_flux_error_top:     float = 1e-2
    retrieval_flux_error_bottom:  float = 1e-2
    retrieval_tolerance:          float = 1e-3
    smoothing_top:                float = 1.0
    smoothing_bottom:             float = 1.0
    weight_apriori:               float = 1.0


# ===========================================================================
# Light source (star / illuminating body)
# ===========================================================================

@dataclass
class LightSource:
    """
    Parameters of the illuminating body.

    Attribute names use the short form (``radius``, ``range``, …);
    the Fortran prefix ``light_source_`` is dropped.
    """
    radius:                 float = 0.0   # (m)
    range:                  float = 0.0   # (m) distance to the target
    effective_temperature:  float = 0.0   # (K)
    irradiation:            float = 0.0   # (W m-2)
    incidence_angle:        float = 0.0   # (deg)
    irradiance: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (n_wavenumbers,)


# ===========================================================================
# Species (line lists, broadening, etc.)
# ===========================================================================

@dataclass
class Species:
    """Per-species spectroscopic, thermodynamic and abundance data."""
    # Counts
    n_species:               int = 0
    n_broadenings_max:       int = 0
    n_cia:                   int = 0
    n_electronic_states_max: int = 0
    n_vibrational_modes_max: int = 0

    # Names
    elements_names: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    line_shape:     np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    species_names:  np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    cia_names:      np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    # File paths
    cia_files:      np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    lines_files:    np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    # Flags
    species_at_equilibrium: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Per-species integer arrays
    n_broadenings_species:       np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    n_electronic_states_species: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    n_vibrational_modes_species: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # 2-D integer arrays
    electronic_states_degeneracies: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    vibrational_modes_degeneracies: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))

    # Elemental abundances
    elemental_abundances: np.ndarray = field(default_factory=lambda: np.zeros(0))
    elemental_h_ratio:    np.ndarray = field(default_factory=lambda: np.zeros(0))
    solar_h_ratio:        np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Per-species real arrays
    cutoffs:                       np.ndarray = field(default_factory=lambda: np.zeros(0))
    intensities_min:               np.ndarray = field(default_factory=lambda: np.zeros(0))
    molar_masses:                  np.ndarray = field(default_factory=lambda: np.zeros(0))
    rotational_partition_exponents:np.ndarray = field(default_factory=lambda: np.zeros(0))
    species_metallicity:           np.ndarray = field(default_factory=lambda: np.zeros(0))
    species_vmr:                   np.ndarray = field(default_factory=lambda: np.zeros(0))
    temperature_intensities_min:   np.ndarray = field(default_factory=lambda: np.zeros(0))

    # 2-D real arrays
    electronic_states_wavenumbers:   np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rayleigh_scattering_coefficients:np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    species_broadenings:             np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    species_broadening_temperature_coefficients: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)))
    species_vmr_layers:              np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sublimation_profiles_coefficients: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    vibrational_modes_wavenumbers:   np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # --- Allocators (parity with Fortran subroutines) ---
    def allocate_primary(self) -> None:
        """Zero/empty allocations dependent only on ``n_species``."""
        n = self.n_species
        self.cutoffs                        = np.zeros(n)
        self.intensities_min                = np.zeros(n)
        self.lines_files                    = np.array([""] * n, dtype=object)
        self.line_shape                     = np.array([""] * n, dtype=object)
        self.molar_masses                   = np.zeros(n)
        self.n_broadenings_species          = np.zeros(n, dtype=int)
        self.n_electronic_states_species    = np.zeros(n, dtype=int)
        self.n_vibrational_modes_species    = np.zeros(n, dtype=int)
        self.rotational_partition_exponents = np.zeros(n)
        self.species_at_equilibrium         = np.ones(n, dtype=bool)
        self.species_names                  = np.array([""] * n, dtype=object)
        self.species_vmr                    = np.zeros(n)
        self.temperature_intensities_min    = np.zeros(n)

    def allocate_secondary(self) -> None:
        """Allocate per-species 2-D arrays once the maxima are known."""
        n = self.n_species
        self.n_broadenings_max       = int(self.n_broadenings_species.max(initial=0))
        self.n_electronic_states_max = int(self.n_electronic_states_species.max(initial=0))
        self.n_vibrational_modes_max = int(self.n_vibrational_modes_species.max(initial=0))

        self.electronic_states_degeneracies = np.zeros(
            (n, self.n_electronic_states_max), dtype=int)
        self.electronic_states_wavenumbers  = np.zeros(
            (n, self.n_electronic_states_max))
        self.species_broadenings = np.zeros((n, self.n_broadenings_max))
        self.species_broadening_temperature_coefficients = np.zeros(
            (n, self.n_broadenings_max))
        self.vibrational_modes_degeneracies = np.zeros(
            (n, self.n_vibrational_modes_max), dtype=int)
        self.vibrational_modes_wavenumbers  = np.zeros(
            (n, self.n_vibrational_modes_max))

    # --- PH3 correction factor (Sousa-Silva et al. 2014) -----------------
    @staticmethod
    def get_ph3_absorption_cross_section_correction_factor(temperature: float) -> float:
        """
        Correction factor accounting for the incompleteness of the
        Sousa-Silva et al. 2014 PH3 line list at high temperatures.
        """
        import math
        temperatures_ref = np.array([1014.0, 1146.0, 1500.0, 1797.0, 2000.0, 2500.0])
        completeness    = np.array([1.0 - 1e-15, 0.99, 0.91, 0.80, 0.70, 0.50])

        if temperature <= temperatures_ref[0]:
            return 1.0

        def _ab(x0: float, x1: float, y0: float, y1: float) -> tuple[float, float]:
            r = math.log(1.0 - y0) / math.log(1.0 - y1)
            b = (r * x0 - x1) / (r - 1.0)
            a = -math.log(1.0 - y1) * (x1 - b)
            return a, b

        for i in range(temperatures_ref.size - 1):
            if temperature > temperatures_ref[i]:
                a, b = _ab(temperatures_ref[i], temperatures_ref[i + 1],
                           completeness[i], completeness[i + 1])
                cf = 1.0 - math.exp(-a / (temperature - b))
                print(f"Warning: applying a correction factor of {1.0 / cf:.3f}"
                      " to PH3 lines intensity, according to Sousa-Silva et al. 2014\n"
                      "Ensure that the PH3 line list you are using comes from this source")
                return cf
        return 1.0


# ===========================================================================
# Spectrometrics — wavenumber grid
# ===========================================================================

@dataclass
class Spectrometrics:
    """Spectral grid and downstream scalar outputs."""
    n_wavenumbers: int = 0

    wavenumber_max:               float = 0.0
    wavenumber_min:               float = 0.0
    wavenumber_step:              float = 0.0
    doppler_deviation_tolerance:  float = 1e-2
    min_total_interval_size:      float = 0.0

    spectral_radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wavenumbers:     np.ndarray = field(default_factory=lambda: np.zeros(0))


# ===========================================================================
# Cloud
# ===========================================================================

@dataclass
class Cloud:
    """Cloud microphysics and optical properties."""
    # cloud_mode:
    #   'fixedRadius'              : particle radius is a fixed input
    #   'fixedSedimentation'       : Ackerman & Marley sedimentation parameter fixed
    #   'fixedRadiusCondensation'  : radius set by f_sed at the condensation level
    #   'fixedRadiusTime'          : radius set by condensation timescale (variant 2)
    cloud_mode:     str = "fixedRadius"
    cloud_fraction: float = 1.0

    cloud_opacity_files: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    cloud_names:         np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    n_clouds: int = 0

    n_cloud_particle_radii: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    sedimentation_parameter:  np.ndarray = field(default_factory=lambda: np.zeros(0))
    supersaturation_parameter:np.ndarray = field(default_factory=lambda: np.zeros(0))
    sticking_efficiency:      np.ndarray = field(default_factory=lambda: np.zeros(0))
    cloud_particle_density:   np.ndarray = field(default_factory=lambda: np.zeros(0))
    cloud_molar_mass:         np.ndarray = field(default_factory=lambda: np.zeros(0))
    reference_wavenumber:     np.ndarray = field(default_factory=lambda: np.zeros(0))

    cloud_particle_radius: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    q_ext_ref:             np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # Raw optical tables, indexed by [cloud, radius, wavenumber]
    cloud_particle_radius_data: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    qext:    np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    gfactor: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    omeg:    np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    qscat:   np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    qabs:    np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))

    # Interpolated quantities indexed by [cloud, wavenumber, layer]
    asymetry_factor:          np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    single_scattering_albedo: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    q_scat:                   np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    q_ext:                    np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))

    def check_cloud_mode(self) -> None:
        allowed = ("fixedRadius", "fixedSedimentation",
                   "fixedRadiusCondensation", "fixedRadiusTime")
        if self.cloud_mode not in allowed:
            raise ValueError(
                f"Cloud: cloud mode '{self.cloud_mode}' not implemented "
                f"(allowed: {', '.join(allowed)})")

    def update_cloud_optical_parameters(self, wavenumbers: np.ndarray) -> None:
        """
        Re-interpolate ``q_ext``, ``q_scat``, ``single_scattering_albedo`` and
        ``asymetry_factor`` onto the current particle-radius profile.

        Mirrors Fortran ``update_cloud_optical_parameters``.
        """
        from .math_utils import interp_ex_0d
        n_clouds = self.n_clouds
        if n_clouds == 0:
            return
        n_layers      = self.cloud_particle_radius.shape[1]
        n_wavenumbers = self.q_ext.shape[1]

        for i in range(n_clouds):
            n_r = int(self.n_cloud_particle_radii[i])
            r_data = self.cloud_particle_radius_data[i, :n_r, :]
            for j in range(n_layers):
                r = self.cloud_particle_radius[i, j]
                for k in range(n_wavenumbers):
                    q_e  = max(interp_ex_0d(r, r_data[:, k], self.qext[i, :n_r, k]), 0.0)
                    q_s  = max(interp_ex_0d(r, r_data[:, k], self.qscat[i, :n_r, k]), 0.0)
                    ssa  = min(max(interp_ex_0d(r, r_data[:, k], self.omeg[i, :n_r, k]), 0.0), 1.0)
                    g    = min(max(interp_ex_0d(r, r_data[:, k], self.gfactor[i, :n_r, k]), -1.0), 1.0)
                    self.q_ext[i, k, j]                    = q_e
                    self.q_scat[i, k, j]                   = q_s
                    self.single_scattering_albedo[i, k, j] = ssa
                    self.asymetry_factor[i, k, j]          = g
                self.q_ext_ref[i, j] = interp_ex_0d(
                    self.reference_wavenumber[i], wavenumbers, self.q_ext[i, :, j])


# ===========================================================================
# Target — planetary body
# ===========================================================================

@dataclass
class Target:
    """Target (planet) parameters."""
    cos_average_angle:          float = 2.0 / 3.0
    emission_angle:             float = 0.0     # (deg)
    latitude:                   float = 0.0     # (deg)
    target_internal_temperature:float = 0.0     # (K)
    target_flattening:          float = 0.0
    target_gravity:             float = 0.0     # (m s-2)
    target_equatorial_radius:   float = 0.0     # (m)
    target_equatorial_gravity:  float = 0.0     # (m s-2)
    target_mass:                float = 0.0     # (kg)
    target_polar_radius:        float = 0.0     # (m)
    target_radius:              float = 0.0     # (m)


# ===========================================================================
# Thermodynamics
# ===========================================================================

@dataclass
class Thermodynamics:
    """Pressure / temperature grids used by absorption cross-section tables."""
    n_levels:         int = 0
    size_thermospace: int = 1   # 1 when in profile mode
    pressure_space:    np.ndarray = field(default_factory=lambda: np.zeros(0))
    temperature_space: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def allocate(self) -> None:
        self.pressure_space    = np.zeros(self.n_levels)
        self.temperature_space = np.zeros(self.size_thermospace)


__all__ = [
    "FILE_NAME_SIZE", "ELEMENT_SYMBOL_SIZE", "SPECIES_NAME_SIZE",
    "Atmosphere", "ExoremRetrieval", "LightSource",
    "Species", "Spectrometrics", "Cloud",
    "Target", "Thermodynamics",
]
