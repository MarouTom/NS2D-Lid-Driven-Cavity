"""Linear stability of the semi-implicit step, boundary condition included.

The von Neumann analysis of Section 6.1 of the report linearises the *interior*
scheme on an infinite grid.  It gives ``dt <= 2 nu/|u|^2`` and, because ``h``
cancels, no mesh-dependent restriction at all: the implicit viscous term is
supposed to buy unconditional stability in ``h``.

It does not, and the reason is the boundary.  Thom's formula supplies the wall
vorticity from the streamfunction of the *previous* time level, so the Dirichlet
data of the implicit solve is explicit.  An analysis that omits the boundary
cannot see this, and the resulting time-step rule is wrong wherever the
mesh-dependent bound binds -- the low-Reynolds, fine-grid regime.

This module measures the effect directly, on the code that is actually run.
Switching the convection term off and holding the walls at rest leaves a linear
map

    omega_int^{n+1} = M omega_int^n ,

namely one call to :meth:`ns2d.solver.Solver.step` of the Stokes problem.  ``M``
is assembled column by column by stepping unit vectors, and its spectral radius
is a function of the single dimensionless group

    d = nu dt / h^2                     (the diffusion number).

:func:`threshold` locates ``rho(M) = 1``.  The answer is independent of ``Re``
(as it must be -- ``M`` depends on ``dt``, ``nu`` and ``h`` only through ``d``)
and decreases monotonically towards ``3/2`` under refinement::

    N        17      25      33      41      49      65
    d_crit   1.5809  1.5465  1.5320  1.5241  1.5192  1.5135

With ``wall="frozen"`` -- the same scheme with the lagged formula replaced by
prescribed wall data -- no finite threshold exists, which is what identifies the
boundary condition as the cause rather than the interior stencils.

The cost is ``O(N_int^3)`` for the eigenvalues, so this is a small-grid
diagnostic; ``N <= 33`` runs in a second or two.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid
from .problems import Cavity
from .solver import Solver

__all__ = [
    "StokesStep",
    "FrozenWallStep",
    "amplification_matrix",
    "spectral_radius",
    "threshold",
]


class StokesStep(Solver):
    """The production time step with the convection term removed."""

    def convection(self, psi: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return np.zeros(self.grid.interior_shape)


class FrozenWallStep(StokesStep):
    """Same, but with the wall vorticity *prescribed* instead of reconstructed.

    The control experiment: everything is identical except that Thom's formula
    no longer feeds ``psi^n`` back into the boundary data.  Any threshold that
    survives here is a property of the interior scheme; any threshold that
    disappears was caused by the lagged boundary condition.
    """

    def boundary_omega(self, psi: np.ndarray, t: float) -> np.ndarray:
        return self.grid.zeros()


def amplification_matrix(
    n: int, d: float, Re: float = 100.0, wall: str = "thom"
) -> np.ndarray:
    """Dense ``M`` with ``omega_int^{n+1} = M omega_int^n`` at diffusion number ``d``.

    ``wall`` is ``"thom"`` (the production boundary condition) or ``"frozen"``
    (wall vorticity prescribed, no feedback).  ``psi^n`` is reconstructed from
    ``omega^n`` by the same Poisson solve the run uses, so the state is exactly
    the one a real step would see.
    """
    grid = Grid.square(n)
    nu = 1.0 / Re
    dt = d * grid.h_min**2 / nu
    cls = {"thom": StokesStep, "frozen": FrozenWallStep}[wall]
    solver = cls(grid, Cavity(Re, u_lid=0.0), dt=dt)

    m = grid.n_interior
    M = np.empty((m, m))
    e = np.zeros(grid.interior_shape)
    for k in range(m):
        e.flat[k] = 1.0
        solver.omega = grid.zeros()
        solver.omega[1:-1, 1:-1] = e
        solver.psi = grid.zeros()
        solver.psi[1:-1, 1:-1] = solver.solver_psi.solve(e)
        solver.t = 0.0
        solver.step()
        M[:, k] = solver.omega[1:-1, 1:-1].ravel()
        e.flat[k] = 0.0
    return M


def spectral_radius(n: int, d: float, Re: float = 100.0, wall: str = "thom") -> float:
    """Spectral-radius diagnostic for the finite-grid linear step.

    Values below one imply asymptotic decay and values above one imply growth.
    At exactly one, power boundedness also depends on the Jordan structure;
    the radius alone does not establish stability at the crossing itself.
    """
    return float(
        np.max(np.abs(np.linalg.eigvals(amplification_matrix(n, d, Re, wall))))
    )


def threshold(
    n: int,
    Re: float = 100.0,
    wall: str = "thom",
    bracket: tuple[float, float] = (0.5, 4.0),
    iterations: int = 20,
) -> float:
    """Bisect for the ``d = nu dt/h^2`` at which ``rho(M)`` crosses 1.

    Returns ``inf`` if the scheme is still stable at the top of ``bracket``,
    which is what happens with ``wall="frozen"``.
    """
    lo, hi = bracket
    if spectral_radius(n, lo, Re, wall) > 1.0:
        raise ValueError(f"already unstable at d={lo}; widen the bracket downwards")
    if spectral_radius(n, hi, Re, wall) <= 1.0:
        return float("inf")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if spectral_radius(n, mid, Re, wall) <= 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
