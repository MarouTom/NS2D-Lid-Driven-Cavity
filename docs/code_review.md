# Numerical and software review

The solver is deliberately small: `Grid` defines array orientation; `Problem` supplies walls and forcing; operators and linear solvers are separate; `Solver` implements one time step; post-processing and reference data are independent of integration. This is a suitable design for studying a uniform-grid 2-D cavity. The code is not intended as a general geometry or turbulence solver.

## Confirmed numerical choices

- With u = ψ_y and v = −ψ_x, convection is ψ_y ω_x − ψ_x ω_y. The implementation uses this sign consistently.
- Boundary terms enter the Helmholtz right-hand side with the correct positive coefficient. All four Thom wall formulas have the appropriate velocity signs.
- The interior matrices are symmetric positive definite. DST and sparse LU provide cross-checks for conjugate gradient with an SPD incomplete-Cholesky preconditioner.
- Corner vorticity values enter no interior stencil. Their sum convention serves the discrete circulation quadrature; it does not resolve the physical corner singularity.
- Smooth manufactured solutions exercise both convection and moving walls. An isolated Laplacian eigenmode would fail to exercise the convection term.

## Changes made in this review

| Issue | Resolution |
|---|---|
| Restart could accept a different dt and domain while retaining old solver matrices | Versioned checkpoints check dt, dimensions, lengths, Reynolds number, wall mode, corner convention and built-in problem parameters before replacing state |
| Invalid mesh, Re, dt or run controls failed late or silently did nothing | Validate finite positive physical parameters, integer sizes and monitoring intervals, and mutually exclusive stop controls |
| Automatic step selection assumed a unit lid | Uses the configured cavity lid-speed magnitude, including a stationary lid |
| Initializing only one field left an inconsistent pair | Reconstruct the missing field with the discrete Poisson relation; check shape, finiteness and zero streamfunction boundary |
| Vorticity L² error included inert corners while its maximum norm excluded them | Apply the same corner exclusion to both reported vorticity norms |
| CG used `rtol` despite an older declared SciPy minimum | Require SciPy ≥ 1.14 |
| Ghia data contained a solver-fitted replacement and a transcription error | Restore printed values, flag anomalies, report full and secondary metrics, and add Botella & Peyret |
| Documentation asserted that refinement ruled out bugs | State measured results and limitations without treating them as a proof |
| README, notebook, tables and report could disagree | Generate figures, CSV/JSON evidence and LaTeX table inputs through one results script |
| Placeholder repository URLs and a manually fixed test badge | Remove them until an actual remote exists; CI runs the tests on supported Python versions |

## Validation and limits

Run `python -m pytest -q`. The suite covers stencils, SPD matrices, three backends, four wall formulas, spatial/temporal refinement, discrete identities, end-to-end cavity comparison, instability detection, and restart compatibility. Expected overflow warnings are scoped to deliberately unstable tests; production warnings are not globally suppressed.

In version 1.3 all nine main production fields were integrated afresh from rest, with an additional fourth-grid check at Re = 400. Case settings, source/runner fingerprints, timestamps and cache hashes establish their provenance. The numerical scheme is unchanged. Builders reject stale or unconverged production states, recompute the benchmark metrics and audit a continuation step.

The accuracy study adds three-grid sensitivity, actual-cavity transient time refinement, an unfiltered steady PDE defect and ten-time-unit continuations at dt and dt/2. The Richardson analysis handles unequal ratios and refuses to manufacture an order for non-monotone or roundoff-level sequences. Tests verify known power-law limits and the relationship between the PDE defect and the semi-implicit update rate. An asymmetric MMS and a finest-grid step-halving check challenge the original verification setup.

The step-size rule combines an interior Fourier bound with a finite-grid Stokes boundary experiment. It remains guidance, not a nonlinear stability theorem. Conditional GCI estimates are not rigorous bounds; finite-window settling is not an infinite-time error bound. Pressure, adaptive grids, higher-order time integration and energy/enstrophy-conserving advection are outside the current implementation. See [accuracy.md](accuracy.md) for the complete interpretation.
