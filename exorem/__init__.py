"""
ExoREM — 1-D Radiative-Convective Equilibrium Model (Python port)
==================================================================

A Python translation of the Fortran Exorem code for computing
thermal emission and transmission spectra of planetary atmospheres.

Key modules
-----------
physics          Physical constants and Planck function
math_utils       Mathematical utility functions (interpolation, FFT, …)
optics           Refractive indices and Rayleigh scattering
objects          Data-classes replacing Fortran module-level state
chemistry        Thermochemical equilibrium (43 gases, 20 condensates)
cloud_mixing     Ackerman & Marley cloud microphysics
radiative_transfer  Two-stream radiative transfer with k-distributions
transit_spectrum  Transmission (transit) spectrum calculation
interface        I/O — input-file parsing, data loading, output writing
exorem_main      Main loop: radiative-convective equilibrium solver

Quick start
-----------
>>> from exorem import run_exorem
>>> results = run_exorem("path/to/input.nml")
>>> wavenumbers = results["wavenumbers"]
>>> spectral_radiosity = results["spectral_radiosity"]

References
----------
- Baudino et al. 2015  https://doi.org/10.1051/0004-6361/201526332
- Baudino et al. 2017  https://doi.org/10.3847/1538-4357/aa95be
- Charnay et al. 2018  https://doi.org/10.3847/1538-4357/aaac7d
- Blain et al. 2020    https://doi.org/10.1051/0004-6361/202039072
"""

from .exorem_main import run_exorem
from .physics import (
    planck_function,
    spherical_black_body_spectral_radiance,
    CST_C, CST_H, CST_K, CST_N_A, CST_P0,
    CST_R, CST_SIGMA, CST_G, CST_N0,
    ELEMENTS_SYMBOL, ELEMENTS_MOLAR_MASS,
)
from .math_utils import (
    interp, interp_ex, interp_ex_0d,
    chi2, chi2_reduced,
    gaussian, gaussian_noise,
    voigt, erfinv, matinv,
    arange, arange_include,
    convolve, fft, ifft,
)
from .optics import (
    get_refractive_index,
    rayleigh_scattering_coefficient,
    DEFAULT_REFRACTIVE_INDEX,
)
from .objects import (
    Atmosphere, Cloud, ExoremRetrieval,
    LightSource, Species, Spectrometrics, Target, Thermodynamics,
)
from .chemistry import (
    gas_id, condensate_id,
    GASES_NAMES, CONDENSATE_NAMES, N_GASES, N_CONDENSATES,
    equilibrium_constant_gases,
    h2o_saturation_pressure, nh3_saturation_pressure, nh4sh_saturation_pressure,
    calculate_chemistry, calculate_gases_molar_mass, calculate_species_molar_mass,
)
from .cloud_mixing import calculate_cloud_mixing, calculate_cloud_mixing2
from .radiative_transfer import calculate_radiative_transfer, calculate_two_stream_fluxes
from .transit_spectrum import calculate_transit_spectrum
from .interface import (
    parse_input_file,
    read_exorem_input_parameters,
    build_stellar_irradiance,
    read_cia_file, read_data_file,
    write_spectrum, write_temperature_profile,
    write_vmr_profile, write_hdf5_output,
)

__version__ = "1.0.0"
__all__ = [
    # Entry point
    "run_exorem",
    # Physics
    "planck_function", "spherical_black_body_spectral_radiance",
    "CST_C", "CST_H", "CST_K", "CST_N_A", "CST_P0",
    "CST_R", "CST_SIGMA", "CST_G", "CST_N0",
    "ELEMENTS_SYMBOL", "ELEMENTS_MOLAR_MASS",
    # Math
    "interp", "interp_ex", "interp_ex_0d",
    "chi2", "chi2_reduced",
    "gaussian", "gaussian_noise",
    "voigt", "erfinv", "matinv",
    "arange", "arange_include",
    "convolve", "fft", "ifft",
    # Optics
    "get_refractive_index", "rayleigh_scattering_coefficient",
    "DEFAULT_REFRACTIVE_INDEX",
    # Data classes
    "Atmosphere", "Cloud", "ExoremRetrieval",
    "LightSource", "Species", "Spectrometrics", "Target", "Thermodynamics",
    # Chemistry
    "gas_id", "condensate_id",
    "GASES_NAMES", "CONDENSATE_NAMES", "N_GASES", "N_CONDENSATES",
    "equilibrium_constant_gases",
    "h2o_saturation_pressure", "nh3_saturation_pressure", "nh4sh_saturation_pressure",
    "calculate_chemistry", "calculate_gases_molar_mass", "calculate_species_molar_mass",
    # Cloud
    "calculate_cloud_mixing", "calculate_cloud_mixing2",
    # Radiative transfer
    "calculate_radiative_transfer", "calculate_two_stream_fluxes",
    # Transit spectrum
    "calculate_transit_spectrum",
    # I/O
    "parse_input_file", "read_exorem_input_parameters",
    "build_stellar_irradiance",
    "read_cia_file", "read_data_file",
    "write_spectrum", "write_temperature_profile",
    "write_vmr_profile", "write_hdf5_output",
]
