# Accuracy study

The numerical scheme is unchanged. This study separates four questions: approximation of a smooth known solution, discretisation sensitivity of the cavity, settling of the final state, and agreement with published numerical references.

## Experiments

| Question | Controlled experiment | Evidence |
|---|---|---|
| Smooth spatial accuracy | Exact-wall and Thom MMS; four grids at fixed small dt, integrated to T = 0.5 so that wall-generated vorticity diffuses into the interior | `verification.json` |
| Robustness of MMS | An asymmetric two-mode field to T = 0.5; halve dt on the finest spatial grid | `verification.json` |
| Smooth temporal accuracy | Same-grid fine-time reference | `verification.json` |
| Actual cavity time accuracy | From rest to T = 5 at N = 129, using dt, dt/2, dt/4 for each Re | `accuracy/temporal_Re*.json` |
| Cavity grid sensitivity | Three independently converged grids per Re | `accuracy.json`, `grid` |
| Remaining settling | Continue each final state for ten time units using dt and dt/2 | `accuracy/steady_*.json` |
| Effect of earlier stopping times | Compare fresh T = 150/300 snapshots with the converged fine-grid state | `accuracy.json`, `steady[].milestone_changes` |
| Reference agreement | Ghia at all three Re; reflected Botella–Peyret at Re = 1000 | `validation.json`, `centerline_comparisons.csv` |

All nine production states start from rest. The grid sequences are N = 65, 129, 257 at Re = 400 and 1000, and N = 129, 193, 257 at Re = 3200. The latter avoids a severely under-resolved 65² mesh in the fit. Every run uses the same absolute update-rate stopping threshold, 1e-6. Reaching the time cap without meeting it is a failed steady study, not a successful completion.

## Conditional Richardson estimates

For coarse-to-fine spacings `hc > hm > hf` and scalar outputs `qc, qm, qf`, assume `q(h) = q0 + C h^p` and solve

`(qc - qm)/(qm - qf) = (hc^p - hm^p)/(hm^p - hf^p)`.

The implementation uses the actual ratios, including the unequal Re = 3200 spacings. The extrapolated limit is `qf + (qf - qm)/((hm/hf)^p - 1)`. The absolute fine-grid GCI estimate is `1.25 * abs(qf - qm)/((hm/hf)^p - 1)`. The percentage divides by `abs(qf)` and multiplies by 100. This uses the basic three-grid procedure of [Celik et al. (2008)](https://doi.org/10.1115/1.2960953), also described in the [Turbulence Modeling Resource summary](https://tmbwg.github.io/turbmodels/Papers/uncertainty_summary.pdf). The optional order-dependent modifications in that summary are not applied.

These are conditional engineering estimates. Three values determine three model parameters, so a good fit does not independently verify the assumed asymptotic model. Non-monotone sequences, roundoff-sized differences, and sequences with no positive order between 0.01 and 10 receive an explicit status and no extrapolation. All outcomes are retained. An independent 513² integration at Re = 400 tests the original three-grid prediction and compares the orders/limits of the overlapping 65/129/257 and 129/257/513 triples. Its measurements are in `data/fourth_grid.json`. This check applies to the Re = 400 primary-streamfunction estimate; additional levels would be needed to test the other cases in the same way.

The scalar outputs include the primary streamfunction, vortex coordinates, kinetic energy, center vorticity and bottom-right eddy strength. The primary-vortex GCI is not an error estimate for the full flow. Profiles are also compared at 65 common uniform stations, which coincide with nodes on every selected grid. Their L2 difference uses trapezoidal integration over the unit interval. This norm differs from the RMS over the 17 irregular literature stations.

## Time refinement and steady states

At a fixed point of this semi-implicit method the time step cancels: the solution satisfies `C_h(psi, omega) = (1/Re) Delta_h omega` with its consistent Thom wall values, regardless of dt. Consequently, a steady solution has no separate backward-Euler time bias once fully converged. The time step still affects stability, transient accuracy, and convergence toward a discrete solution; multiple solutions, if present, can have different basins of attraction.

Temporal order is therefore measured at the same transient time T = 5, starting each run from rest. Ratios of successive centerline differences measure self-convergence on a fixed grid. They do not measure total continuum error. The reported primary-vortex temporal extrapolation in the JSON is similarly conditional.

For settling, the final state is independently continued with dt and dt/2 for ten more time units. The study reports both changes relative to that state and the difference between the two continuation endpoints. It also evaluates the steady PDE defect after refreshing the wall ring. If `rate = (omega_next - omega)/dt`, then `(I + dt L/Re) rate = -F_h`, where `L` is the interior negative Laplacian and `F_h = C_h - Delta_h omega/Re`. The update rate is thus a filtered defect, not the same residual.

Halving the step reduces the implicit filtering, so the initial update rate can marginally exceed the original threshold despite identical initial fields. The JSON records these cases with `stayed_below_threshold: false`; the report prints the maximum rate across both continuations. This does not change the criterion used to stop the original production run.

Finite-window drift is a measurement, not an infinite-time bound. Neither residual is converted into a solution error without a justified stability estimate. Grid GCI, temporal differences, observed settling and benchmark discrepancies are not independent random errors and are not combined in quadrature.

## Reproduction and provenance

```bash
python scripts/run_production.py --force
python scripts/check_fourth_grid.py
python scripts/run_accuracy.py
python scripts/build_results.py
python scripts/build_notebook.py --execute
python scripts/build_report.py
```

For independent or partial runs:

```bash
python scripts/run_accuracy.py --temporal-only
python scripts/run_accuracy.py --steady-only --case Re3200_N257 --fft-workers 4
python scripts/run_accuracy.py --assemble-only
```

Production records contain numerical-source and runner hashes, case settings, the initial condition, timestamps, software versions, FFT worker count, and field/cache SHA-256 hashes. Accuracy caches include their input-cache hashes and analysis-source fingerprints. Builders reject mismatched evidence. Numerical outputs may differ at roundoff across library versions or thread configurations. Historical fields remain only in the local ignored archive and do not supply the current headline results.
