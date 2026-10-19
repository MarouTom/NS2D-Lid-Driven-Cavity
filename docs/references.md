# Reference data and conventions

The benchmark arrays are in [`ns2d/benchmark.py`](../ns2d/benchmark.py).
The source pages were checked visually on 20 September 2026; OCR alone was not used to settle signs or digits.

## Ghia, Ghia & Shin (1982)

**High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method**, *Journal of Computational Physics* 48, 387-411.
[DOI](https://doi.org/10.1016/0021-9991(82)90058-4) · [University-hosted scan](https://www.ljll.fr/~frey/papers/Navier-Stokes/Ghia%20V.%2C%20High-Resolutions%20for%20incompressible%20flow%20using%20the%20Navier-Stokes%20equations%20and%20a%20multi-grid%20method.pdf).

- Table I, p. 398: horizontal velocity on the vertical centerline.
- Table II, p. 399: vertical velocity on the horizontal centerline.
- Table V, pp. 408-409: primary-vortex properties. Its final row explicitly lists the grid used: 129² at Re = 100, 1000, 3200; 257² at Re = 400, 5000. No grid is inferred from rounded vortex coordinates.

The headline RMS uses **all 17 published stations**, including the zero-velocity endpoints. Secondary RMS values omit only the flagged entries listed below. Those secondary metrics diagnose sensitivity to the reference; they are not replacement headline scores. Velocities are interpolated linearly to the printed four-decimal coordinates, rather than silently treating them as exact fractions of 128.

| Table | Re | Coordinate | Printed value | Treatment |
|---|---:|---:|---:|---|
| I, u | 3200 | y = 0.4531 | −0.86636 | Retained and flagged; no decimal-point correction |
| I, u | 10000 | y = 0.5000 | +0.03111 | Retained and flagged; not used in the production study |
| II, v | 400 | x = 0.9063 | −0.23827 | Retained and flagged |
| II, v | 1000 | x = 0.9063 | −0.51550 | Corrected a local transcription of −0.51500 |
| V, vortex x | 3200 | primary center | 0.5165 | Retained as printed, even though it is off the listed grid |

Flags express an apparent inconsistency with surrounding data and/or the computed profiles. They are **not publisher errata**. In particular, changing −0.86636 to −0.086636 because it fits the present solution would contaminate the comparison. Earlier local drafts made that change; the current project preserves the published number.

## Botella & Peyret (1998)

**Benchmark spectral results on the lid-driven cavity flow**, *Computers & Fluids* 27, 421-433.
[DOI](https://doi.org/10.1016/S0045-7930(98)00002-4) · [Paper](https://cats2d.com/documentation/botellapeyret98.pdf).

This is an independent comparison method: Chebyshev collocation with corner singularity subtraction. We use Table 6 (p. 425, N = 160 primary vortex), Table 9 (p. 426, u), and Table 10 (p. 427, v). These are numerical reference solutions, not experimental measurements.

Their lid moves **left** with u = −1. For our right-moving lid the reflection is

$$x_R=1-x_L,\quad y_R=y_L,\quad u_R(x_R,y)=-u_L(1-x_R,y),\quad v_R(x_R,y)=v_L(1-x_R,y).$$

The streamfunction changes sign: ψ_R(x_R,y) = −ψ_L(1−x_R,y). Thus their primary vortex becomes ψ = −0.1189366 at (0.5308, 0.5652). The code stores the published left-lid velocities and applies this transformation explicitly. Notice that **v does not change sign**.

## Historical and mathematical background

The report introduction draws on these sources:

- S. R. Bistafa, **200 Years of the Navier-Stokes Equation**, [arXiv:2401.13669](https://arxiv.org/abs/2401.13669), for Navier's 1822 contribution, Stokes's 1845 formulation, and the development of viscous-flow modelling.
- C. L. Fefferman, **Existence and smoothness of the Navier-Stokes equation**, [official Clay problem statement](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf), for the precise three-dimensional regularity question and the distinction from the two-dimensional theory. The [Clay problem page](https://www.claymath.org/millennium/navier-stokes-equation/) lists its current status.
- O. R. Burggraf (1966), **Analytical and numerical studies of the structure of steady separated flows**, *Journal of Fluid Mechanics* 24(1), 113-151, [DOI](https://doi.org/10.1017/S0022112066000545), for early numerical cavity calculations up to Re = 400 and their corner eddies. This is historical context, not an additional dataset used in the present error tables.

The later benchmark studies are Ghia et al. (1982) and Botella & Peyret (1998), documented above. The discussion distinguishes the difficulty of computing a particular flow accurately from the general three-dimensional existence and smoothness problem.

## What the comparisons establish

The MMS studies test formal accuracy for a smooth forced solution. Literature comparisons test agreement on the singular cavity benchmark. Neither a small linear-system residual nor a small discrepancy from one reference establishes continuum accuracy by itself. The cavity study includes three grids per Reynolds number, transient time refinement and further-continuation measurements from states satisfying the same steady criterion. Its Richardson/GCI estimates are conditional on a leading power-law error; three data points alone cannot establish an asymptotic regime. An independent fourth grid checks that assumption for primary-vortex strength at Re = 400. See [accuracy.md](accuracy.md).

## Grid-uncertainty procedure

Celik, Ghia, Roache, Freitas, Coleman & Raad (2008), **Procedure for estimation and reporting of uncertainty due to discretization in CFD applications**, *Journal of Fluids Engineering* 130(7), 078001. [DOI](https://doi.org/10.1115/1.2960953). The [Turbulence Modeling Resource summary](https://tmbwg.github.io/turbmodels/Papers/uncertainty_summary.pdf) gives the basic unequal-ratio procedure and factor 1.25. This project uses that basic GCI expression, rejects non-positive/non-monotone fits, and does not apply the summary's optional order-dependent limiters.
