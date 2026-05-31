# ExoREM (Python port)

A 1-D **radiative–convective equilibrium** model for the atmospheres of brown
dwarfs and self-luminous giant exoplanets. Given a planet/object's internal
temperature, gravity, bulk metallicity (and optionally an external irradiation
source), the code iterates a temperature profile to radiative–convective
equilibrium and returns the converged profile, the emergent spectrum, the
chemical composition, and (optionally) a transmission spectrum.

This is a Python translation of the original Fortran ExoREM. It has been
validated against the Fortran reference on the non-irradiated `T_int = 500 K`,
10× solar, 81-level test case: the emergent spectrum agrees channel-by-channel
to a few percent and the bolometric effective temperature to ~0.5 %
(`T_eff = 500.2 K` vs the 500 K target), and the full self-consistent loop
converges in ~71 iterations with a well-conditioned inversion.

---

## 1. Installation

The code is a plain Python package with a small scientific stack:

```
numpy
scipy
h5py        # k-tables and HDF5 output
numba       # JIT for the two-stream layer-opacity kernels
```

No build step is required. Put the `exorem/` package on your `PYTHONPATH`
(or run from its parent directory) and make sure the data directory referenced
by your namelist (k-tables, CIA tables, thermochemical tables, profiles) is in
place.

---

## 2. Running the model

The single entry point is the runner module:

```bash
python -m exorem.runner inputs/example_no_irr.nml
```

Useful flags:

- `--no-audit` — skip the pre-run check that every data file referenced by the
  namelist exists.
- `--audit-only` — run only that check and exit (a dry run).
- `--strict` — abort if the audit finds missing files.

You can also call it from Python:

```python
from exorem import run_exorem
results = run_exorem("inputs/example_no_irr.nml")
```

`run_exorem` returns a dictionary containing the wavenumber grid, the spectral
radiosity, the level pressures/temperatures, the per-level fluxes, the layer
volume mixing ratios, and related arrays.

During the run, each iteration prints the radiative-transfer status, the trace
of `K Sₐ Kᵀ`, the Tikhonov λ and the regularized condition number, the mean
molar mass, the flux ratio `J_int / (σ T_int⁴)`, the implied `T_int`/`T_eff`,
and χ². Convergence is reached when the internal flux matches the target to the
configured tolerance (default `1e-3`).

---

## 3. Inputs

### 3.1 The namelist

The namelist is grouped into Fortran-style sections. The parameters most users
will set:

**`target_parameters`**

| Key | Meaning |
| --- | --- |
| `target_internal_temperature` | Interior temperature `T_int` (K) that the equilibrium must reproduce |
| `target_mass`, `target_equatorial_radius` | Used to derive surface gravity via `G·M/R²` |
| `target_equatorial_gravity` | Explicit gravity (an alternative to `M`,`R`) |
| `target_polar_radius`, `target_flattening` | Oblateness (optional) |

When both an explicit gravity and `G·M/R²` are available, the code reports both
candidates and resolves to one (the test case resolves to `25.946 m/s²` from
`G·M/R²`).

**`atmosphere` / model grid**

| Key | Meaning |
| --- | --- |
| `n_levels` | Number of pressure levels (e.g. 81) |
| `pressure_min`, `pressure_max` | Top and bottom pressures of the grid (**Pa**) |
| `use_pressure_grid` | Use a supplied pressure grid instead of building one |
| `metallicity`, `use_metallicity` | Bulk metallicity (linear, ×solar) |
| `elements_metallicity`, `elements_names`, `use_elements_metallicity` | Per-element metallicity overrides |
| `he_vmr` | Helium mixing ratio |

**`spectrum`**

| Key | Meaning |
| --- | --- |
| `wavenumber_min`, `wavenumber_max`, `wavenumber_step` | Output spectral grid (**cm⁻¹**) |
| `reference_wavenumber` | Reference for the spectral radius / transit |

Note: if `wavenumber_max` exceeds the k-table coverage (the supplied tables stop
near 8130 cm⁻¹), the extension region is handled with Rayleigh scattering and
the Planck source only (no line opacity). This is intended behaviour.

**`retrieval_parameters`** (controls the equilibrium solver)

| Key | Meaning |
| --- | --- |
| `n_iterations` | Maximum number of iterations |
| `n_non_adiabatic_iterations` | Pure-radiative burn-in before the adiabat is projected |
| `n_burn_iterations` | Additional burn-in control |
| `chemistry_iteration_interval` | How often thermochemistry is recomputed |
| `cloud_iteration_interval` | How often cloud mixing is recomputed |
| `retrieval_level_top`, `retrieval_level_bottom` | Level range whose temperatures are retrieved |
| `retrieval_flux_error_top`, `retrieval_flux_error_bottom` | Measurement-error envelope (sets `Sₑ`) |
| `retrieval_tolerance` | Convergence threshold on the internal-flux ratio (default `1e-3`) |

**Physics switches**: `use_chemistry`, `use_rayleigh`, `use_irradiation`,
`add_light_source`, `species_at_equilibrium`, `species_names`, `cia_names`,
`cloud_names`, `n_species`, `n_cia`, `n_clouds`, eddy-diffusion controls
(`eddy_mode`, `eddy_diffusion_coefficient`, `load_kzz_profile`), cloud
microphysics (`cloud_particle_radius`, `cloud_particle_density`,
`sedimentation_parameter`, `sticking_efficiency`, `supersaturation_parameter`),
and the Guillot analytic-profile parameters used to seed the first guess
(`guillot_kappa_ir`, `guillot_gamma_v`, `guillot_grad_ad`, `guillot_p_photo_bar`).

**Profile loading / output toggles**: `temperature_profile_file`,
`vmr_profiles_file`, `load_vmr_profiles`, `load_cloud_profiles`,
`output_fluxes`, `output_full`, `output_transmission_spectra`,
`output_thermal_spectral_contribution`,
`output_species_spectral_contributions`, `output_cia_spectral_contribution`.

**Paths**: `path_data`, `path_k_coefficients`, `path_cia`,
`path_thermochemical_tables`, `path_clouds`, `path_temperature_profile`,
`path_vmr_profiles`, `path_light_source_spectra`, `path_outputs`.

### 3.2 The data directory

Referenced relative to `path_data`:

- **k-tables** — one correlated-k HDF5 per absorber
  (`<species>.ktable.exorem.h5`), each storing the g-ordered absorption
  coefficient on a (pressure, temperature, wavenumber, g) grid plus the g-point
  quadrature weights.
- **CIA tables** — collision-induced absorption text files
  (`H2-H2.cia.txt`, `H2-He.cia.txt`, `H2O-H2O.cia.txt`, …).
- **thermochemical tables** — equilibrium-chemistry tables used by the
  composition step.
- **profiles** — optional a-priori temperature profile and/or VMR profiles, and
  an optional `Kzz` profile.

---

## 4. Outputs

Written under `path_outputs`:

| File | Contents |
| --- | --- |
| `spectra<suffix>.dat` | Emergent (and/or transit) spectrum on the output wavenumber grid |
| `temperature_profile<suffix>.dat` | Converged `T(P)` |
| `vmr<suffix>.dat` | Converged volume mixing ratios per species and level |
| `output<suffix>.h5` | Full HDF5 dump (profiles, spectra, fluxes, optical depths, run-quality metadata) |
| `iteration_timings.csv`, `rt_internal_timings.csv` | Per-phase wall-clock profiling |

---

## 5. Units convention

The original Fortran worked internally in **mbar / CGS** in several places; this
Python port standardises the *model state* on **SI** (Pa, metres, kelvin,
kg/mol) while keeping the *radiative quantities* in **CGS** to match the
reference's flux normalisation. The boundaries between unit systems are the most
error-prone part of the code, so they are tabulated explicitly below.

### 5.1 Verified during validation (load-bearing for the physics)

| Quantity | Variable(s) | Unit |
| --- | --- | --- |
| Pressure — model state | `atm.pressures`, `atm.pressures_layers` | **Pa** |
| Pressure — k-table axis | `p_k_species` (as stored in the HDF5) | **bar**; converted Pa→bar (`×1e-5`) immediately before the log-pressure interpolation |
| Temperature | `atm.temperatures`, `atm.temperatures_layers`, `t_k_species` | **K** |
| Wavenumber | `wavenumbers`, `wavenumbers_k` | **cm⁻¹** |
| Altitude | `atm.z` | **m** |
| Scale height | `atm.scale_height` | **m** (`R·T / (μ·g)` with R in J·mol⁻¹·K⁻¹, μ in kg·mol⁻¹, g in m·s⁻²) |
| Surface / layer gravity | `target_gravity`, `atm.gravities_layers` | **m·s⁻²** |
| Radius | `target_radius` | **m** |
| Molar mass | `gases_molar_mass`, `atm.molar_masses_layers` | **kg·mol⁻¹** (the “Mean molar mass: … g mol⁻¹” console line is a display-only ×10³ conversion) |
| Volume mixing ratio | `gases_vmr`, `species_vmr_layers` | dimensionless mole fraction |
| Planck source, spectral radiosity, fluxes | `_planck_array`, `spectral_radiosity`, `flux` | **erg·s⁻¹·cm⁻²·sr⁻¹ / cm⁻¹** (CGS); `π∫B dν = σ_cgs·T⁴` was checked to 0.04 % |
| Internal radiosity target | `radiosity_internal_target` | **erg·s⁻¹·cm⁻²** = `σ_SI·T_int⁴ × 1e3` (the `1e3` is the W·m⁻² → erg·s⁻¹·cm⁻² conversion, putting the target in the same CGS units as the modelled radiosity) |
| Optical depth | `tau`, `tau_rayleigh`, `dtau`, `dtauc` | dimensionless |
| Layer column number density | `cmam · h0` | **cm⁻²** (validated against the hydrostatic column `n·Δz` to 0.2 %) |
| Correlated-k coefficient | `kcoeff_species` | absorption **cross-section per molecule** (cm²·molecule⁻¹): `τ = k · VMR · (cmam·h0)` is dimensionless |
| CIA coefficient | `h2_h2_cia`, `h2_he_cia`, `h2o_h2o_cia`, `h2o_n2_cia` | **cm⁻¹·amagat⁻²**; the prefactor `fac_cont` supplies `(n/n_L)²·path` in cm |

### 5.2 Physical constants (`physics.py`, CODATA 2018, SI)

| Constant | Symbol | Value / unit |
| --- | --- | --- |
| Speed of light | `CST_C` | 2.99792458×10⁸ m·s⁻¹ |
| Planck | `CST_H` | 6.62607015×10⁻³⁴ J·s |
| Boltzmann | `CST_K` | 1.380649×10⁻²³ J·K⁻¹ |
| Avogadro | `CST_N_A` | 6.02214076×10²³ mol⁻¹ |
| 1 atm | `CST_P0` | 1.01325×10⁵ Pa |
| Loschmidt reference T | `CST_T0` | 273.15 K |
| Newton gravitation | `CST_G` | 6.67430×10⁻¹¹ m³·kg⁻¹·s⁻² |
| Loschmidt density | `CST_N0` | `CST_P0/(CST_K·CST_T0)` (m⁻³) |
| Molar gas constant | `CST_R` | `CST_N_A·CST_K` (J·mol⁻¹·K⁻¹) |
| Stefan–Boltzmann | `CST_SIGMA` | derived (W·m⁻²·K⁻⁴, SI) |

Note the asymmetry that catches everyone: `CST_SIGMA` and `CST_N0` are **SI**,
but the radiative quantities they feed into are expressed in **CGS**, which is
why `radiosity_internal_target` carries an explicit `×1e3`.

### 5.3 Per namelist documentation, not independently re-derived here

The cloud microphysics (`cloud_particle_radius`, `cloud_particle_density`,
`sedimentation_parameter`), the eddy diffusion coefficient
(`eddy_diffusion_coefficient`), and the irradiation source quantities follow the
conventions documented in the namelist and the original Fortran. They were not
exercised by the non-irradiated, cloud-free validation case, so confirm their
units against the loaders before relying on them quantitatively.

---

## 6. Method

ExoREM solves for the temperature profile `T(P)` at which the net radiative +
convective flux is constant with depth and equal to the internal flux
`σ T_int⁴`. It does this by linearised iteration: at each step it builds the
radiation field and its sensitivity to temperature, then takes a regularised
Newton step toward flux balance, with a convective adjustment ensuring stability
in the deep, optically thick interior.

### 6.1 Radiative transfer

For each atmospheric layer and each spectral interval the code assembles the
optical depth from three sources:

- **Molecular lines** via the **correlated-k** method. Each absorber's k-table
  is interpolated bilinearly in (log-pressure, log-temperature) — with the
  pressure converted Pa→bar to match the table axis — at every g-point. The
  layer optical depth for a species is `k · VMR · (cmam·h0)`, where `cmam·h0` is
  the hydrostatic column number density of the layer (cm⁻²). Per-species
  contributions are summed, and the g-point quadrature integrates the
  transmission over each spectral bin.
- **Collision-induced absorption** (H₂–H₂, H₂–He, H₂O–H₂O, H₂O–N₂), a smooth
  continuum scaling as density² × path through the `fac_cont` prefactor.
- **Rayleigh scattering** (optional), giving `tau_rayleigh`.

The intensity field is propagated with a **two-stream** solver (the Numba-JIT
`_twostream_loop_parallel` kernel), which returns the upward and downward
fluxes, the spectral radiosity at every level, and — crucially for the
retrieval — the **Jacobian** `matrix_t = ∂(radiosity)/∂T`. The Planck source
function (`_planck_array`) is evaluated in CGS and integrates to `σ_cgs T⁴`.

### 6.2 Convection

The deep interior is convective and near-adiabatic. The solver runs a
pure-radiative burn-in for `n_non_adiabatic_iterations`, then projects the deep
super-adiabatic zone onto `∇_ad` (anchored at the radiative–convective
boundary) once via `_init_adiabat`. Thereafter convection is handled in two
self-consistent ways every iteration:

- `_add_convective_term` adds the convective-flux closure to the energy balance
  **and** couples it into `matrix_t`, so the retrieval "sees" convective
  stabilisation in its Jacobian and the deep zone can settle slightly
  super-adiabatic while carrying the internal flux.
- `_convective_adjustment` performs an enthalpy-conserving adjustment toward
  `∇_ad + ΔΦ`, removing super-adiabatic cold dips at the boundary while
  preserving the convective Jacobian coupling.

### 6.3 Chemistry, mixing, clouds, structure

On a configurable cadence the code recomputes:

- **Thermochemical equilibrium** composition (`_calculate_thermochemical_equilibrium`),
  which sets the VMRs that feed back into the opacities and mean molecular
  weight.
- **Eddy mixing** (`Kzz`) updates.
- **Cloud mixing** and cloud optical depth (for the species in `cloud_names`).
- **Altitude / scale height / gravity** (`_calculate_altitude`): re-integrates
  the hydrostatic altitude grid, recomputes the mean molar mass from the current
  VMRs, and updates the layer gravities with the `(R/(R+z))²` falloff.

---

## 7. The linear algebra: optimal-estimation retrieval

The temperature update is a **Tikhonov-regularised optimal-estimation (Rodgers)
inversion**, implemented in `_temperature_profile_retrieval`. It treats the
per-level internal radiosity as the "measurement" and the retrieved-level
temperatures as the state, and takes a constrained Gauss–Newton step that pulls
the modelled internal flux toward the target while staying close to the a-priori
profile.

### 7.1 Matrices

- **K** — the Jacobian, `K = matrix_t[:, retrieval window]`, with
  `K_ij = ∂(radiosity at level i) / ∂(temperature at retrieved level j)`. It is
  produced directly by the two-stream solver.
- **Sₐ** — the a-priori / state covariance, `matrix_s`. It encodes how strongly
  neighbouring levels are correlated and how far the solution may stray from the
  prior.
- **Sₑ** — the measurement-error covariance, `Se = diag(rad_noise²)`. The
  per-level flux error `rad_noise` is set from `retrieval_flux_error_top/bottom`.

### 7.2 The step actually computed

```
SK     = Sₐ · K                         # (n_levels × n_retrieved)
KSK    = Kᵀ · SK                         # (n_retrieved × n_retrieved)
M      = KSK + Sₑ                        # observation-space curvature
M_reg  = M + λ · (trace(M)/n) · I        # Tikhonov ridge
M_inv  = matinv(M_reg)
R      = SK · M_inv                      # gain matrix
ΔT     = R · Δradiosity                  # proposed temperature update
```

where `Δradiosity` is the residual between the modelled internal radiosity and
the target, and `matinv` is a LAPACK matrix inverse
(`numpy.linalg.inv`).

### 7.3 Why the Tikhonov ridge is there

Many vertical modes of the temperature profile have almost no spectral
signature: deep optically thick layers contribute little to the emergent flux,
and very tenuous upper layers contribute nothing. The bare matrix `KSK + Sₑ` is
therefore near-singular (condition number ≳ 10⁶ for this problem), and inverting
it amplifies noise in those modes into nonsensical proposed temperature changes
(values of order thousands of times the current temperature were observed).

The ridge

```
M_reg = M + λ · (trace(M)/n) · I,   λ = 1e-3
```

adds a small isotropic term proportional to the mean diagonal eigenvalue.
Well-constrained modes (large eigenvalues) are essentially untouched;
near-singular modes (small eigenvalues) are damped, capping the condition number
at roughly `1/λ ≈ 10³` and keeping the Newton direction reliable. In the
validated run the regularised condition number sits steadily around 2–6×10⁹ for
the full coupled (radiative + convective) matrix, and the inversion never
diverges.

### 7.4 Safeguards on the step

The proposed `ΔT` passes through several guards before it is applied:

1. **Non-finite guard** — if any element is NaN/Inf the whole step is rejected
   (`ΔT = 0`).
2. **Upper-atmosphere information weighting** (optional) — using the per-level
   cumulative optical depth `tau_rep`, the step is faded out of the
   optically-thin, data-empty upper atmosphere (`w = τ/(τ+τ_info)`), and those
   levels are instead relaxed gently toward the a-priori radiative-equilibrium
   profile. This prevents the inversion from inventing structure where the
   spectrum carries no information.
3. **Per-layer rate limit** — if any layer's proposed change is extreme
   (> 5× its current temperature), the step is clipped per-layer to ±30 %.
4. **Physical floor/ceiling and line-search** — temperatures are kept within
   `[100 K, 10000 K]` (below ~100 K the equilibrium chemistry becomes
   numerically singular; the ceiling is well above any `T_int < 2000 K`
   interior), with a final line-search ensuring no level underflows the floor.

### 7.5 Convergence

After the radiative transfer each iteration evaluates

```
solution_deviation = | 1 − radiosity_internal[bottom] / radiosity_internal_target |
```

equivalently the printed `J_int / (σ T_int⁴)` ratio. When this falls below
`retrieval_tolerance` (default `1e-3`) the loop reports convergence, optionally
computes the transmission spectrum, writes all outputs, and stops. χ² and its
reduced form are printed alongside as a goodness-of-fit diagnostic.

### 7.6 Iteration skeleton

Per iteration, in order: optional adiabat projection (once, at the burn-in
boundary) → **radiative transfer** (clear, then cloudy if clouds are active) →
convergence check (and final write if converged) → **convective term** (flux
closure + Jacobian coupling) → **optimal-estimation retrieval** (the step above)
→ **convective adjustment** → altitude/scale-height/gravity update → `Kzz`
update → thermochemistry (on its interval) → cloud mixing (on its interval) →
diagnostics.

---

## 8. Validation provenance

This port diverged from the Fortran reference (runaway upper-atmosphere
heating) until four unit mismatches in `radiative_transfer.py` were corrected —
all artefacts of porting an mbar/CGS Fortran code into a Pa/SI Python one:

1. **Scale-height conversion** in `h0`: `scale_height` is SI metres, so the
   m→cm factor is `1e2`, not the Fortran's `1e5` (km→cm). Removed a spurious
   1000×.
2. **Line column density** `cmam`: ΔP is in Pa, so the column divides by
   `CST_P0` (Pa), not `CST_P0·1e-2` (mbar). Removed a verified 100× over-count in
   the line optical depth (the column density now matches the hydrostatic value
   to 0.2 %).
3. **CIA prefactor** `fac_cont`: the same Pa-vs-mbar error squared — `CST_P0²`,
   not `(CST_P0·1e-2)²`. Removed 10⁴×.
4. **k-table pressure axis**: the tables store pressure in **bar**, so the model
   pressure is converted Pa→bar before the log-pressure interpolation; otherwise
   every layer above ~10⁻³ bar clipped to the 100-bar table entry and the lines
   were evaluated at maximum pressure broadening throughout the column.

With these in place the forward model reproduces the Fortran emergent spectrum
to a few percent per channel and `T_eff` to ~0.5 %, and the full loop converges
without the heat-bubble runaway or matrix-singularity failures.

### Validation tooling (optional, not part of the core model)

Three standalone scripts were used to localise the above and can be reused to
re-validate after changes:

- `inspect_ktable_grid.py` — prints each k-table's (P, T, wavenumber) grid next
  to the model's pressure range, flagging pressure-axis unit mismatches.
- `dump_planck_cia.py` — checks the Planck source against an analytic CGS
  blackbody (and `π∫B dν` against `σT⁴`), and isolates the emergent flux into
  line-only / CIA-only / full contributions.
- `verify_with_fortran_vmrs.py` — injects the Fortran reference VMRs, profile,
  and mean molar mass into a single Python radiative-transfer call and compares
  the emergent spectrum channel-by-channel against the Fortran output.
