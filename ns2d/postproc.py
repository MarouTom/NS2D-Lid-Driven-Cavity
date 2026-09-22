"""Post-processing and physical diagnostics.

Everything here is a *check on the answer*: velocities reconstructed from the
streamfunction, integral quantities that have known exact values, and the
locations of the vortices that the benchmark tabulates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import WallVelocities
from .grid import Grid
from .operators import laplacian

__all__ = [
    "trapezoid_weights",
    "l2_norm",
    "velocity_from_psi",
    "kinetic_energy",
    "enstrophy",
    "total_circulation",
    "circulation_identity",
    "expected_circulation",
    "max_divergence",
    "poisson_residual",
    "centerline_u",
    "centerline_v",
    "Vortex",
    "primary_vortex",
    "corner_vortices",
]


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------
def trapezoid_weights(grid: Grid) -> np.ndarray:
    """Composite-trapezoid weights: 1 inside, 1/2 on edges, 1/4 at corners."""
    wx = np.ones(grid.nx)
    wx[0] = wx[-1] = 0.5
    wy = np.ones(grid.ny)
    wy[0] = wy[-1] = 0.5
    return np.outer(wx, wy)


def integrate(f: np.ndarray, grid: Grid) -> float:
    return float(grid.hx * grid.hy * np.sum(trapezoid_weights(grid) * f))


def l2_norm(f: np.ndarray, grid: Grid) -> float:
    """Discrete ``L2`` norm consistent with the continuous one."""
    return float(np.sqrt(integrate(f**2, grid)))


# ---------------------------------------------------------------------------
# velocity
# ---------------------------------------------------------------------------
def velocity_from_psi(
    psi: np.ndarray, grid: Grid, walls: WallVelocities | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``u = dpsi/dy``, ``v = -dpsi/dx``.

    Centered differences at interior points.  On the walls the velocity is
    *known data*, so it is imposed rather than differenced; this keeps the
    no-slip condition exact in the plots and in the integral diagnostics.
    """
    u = np.zeros(grid.shape)
    v = np.zeros(grid.shape)
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * grid.hy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * grid.hx)

    if walls is not None:
        u_top, u_bot, v_left, v_right = walls.arrays(grid)
        u[:, -1], v[:, -1] = u_top, 0.0
        u[:, 0], v[:, 0] = u_bot, 0.0
        # The left/right walls are applied last, so at the two upper corners --
        # the singular points of the continuous problem, where the lid asks for
        # u = 1 and the side wall for u = 0 -- the stationary wall wins.  Some
        # definite choice has to be made; it affects plots only.
        u[0, :], v[0, :] = 0.0, v_left
        u[-1, :], v[-1, :] = 0.0, v_right
    return u, v


def kinetic_energy(u: np.ndarray, v: np.ndarray, grid: Grid) -> float:
    return 0.5 * integrate(u**2 + v**2, grid)


def enstrophy(omega: np.ndarray, grid: Grid) -> float:
    return 0.5 * integrate(omega**2, grid)


def total_circulation(omega: np.ndarray, grid: Grid) -> float:
    """``\\int\\int omega dA`` by the trapezoid rule.

    By Green's theorem this equals the circulation of the velocity around the
    boundary, which is known exactly from the wall velocities
    (:func:`expected_circulation`).

    The discrete version of the identity is *exact*, not merely ``O(h)``: with
    the wall values of ``omega`` given by Thom's formula and the interior
    values satisfying ``Delta_h psi = -omega``, the trapezoid sum reproduces
    the boundary integral to round-off.  Each wall's ``-2 u_wall/h`` term
    telescopes into the trapezoid quadrature of that wall's velocity, and the
    ``-2 psi/h^2`` terms cancel against the summed Laplacian of the interior.
    The identity is therefore a check on the Poisson solve and on all four Thom
    formulas -- signs, wall velocities and all -- in a single number.

    It does require the corner convention ``corner="sum"``
    (:func:`ns2d.boundary.thom_vorticity`), because a corner must carry the
    contribution of both walls that meet there.  With ``"average"`` the result
    is off by ``h/2`` per unit lid velocity and with ``"zero"`` by ``h`` --
    an ``O(h)`` discrepancy that is pure bookkeeping and says nothing about
    the accuracy of the scheme.

    Applied to a *stored* solver state the identity holds only up to the
    one-step lag of the boundary ring: :meth:`ns2d.solver.Solver.step` writes
    the wall values obtained from ``psi^n`` alongside the interior values of
    ``omega^{n+1}``, so the pair is consistent only in the limit of steady
    state.  :func:`circulation_identity` removes the lag and is the exact
    version.
    """
    return integrate(omega, grid)


def circulation_identity(psi: np.ndarray, grid: Grid, walls: WallVelocities) -> float:
    """Trapezoid integral of the vorticity field *consistent with* ``psi``.

    Builds ``omega`` from a single streamfunction -- Thom's formula on the
    walls, ``-Delta_h psi`` inside -- and integrates it.  The result equals
    :func:`expected_circulation` to round-off for **any** ``psi`` vanishing on
    the boundary, so this is an algebraic identity of the discretisation rather
    than an accuracy statement.  It fails if any of the four Thom formulas, the
    corner convention or the sign of the Poisson equation is wrong.
    """
    from .boundary import thom_vorticity

    omega = thom_vorticity(psi, grid, walls, corner="sum")
    omega[1:-1, 1:-1] = -laplacian(psi, grid)
    return integrate(omega, grid)


def expected_circulation(grid: Grid, walls: WallVelocities) -> float:
    """``\\oint u . dl`` counter-clockwise around the boundary, computed from
    the prescribed wall velocities.  For the cavity with ``u_lid = 1`` this is
    ``-1``."""
    u_top, u_bot, v_left, v_right = walls.arrays(grid)
    wx = np.ones(grid.nx)
    wx[0] = wx[-1] = 0.5
    wy = np.ones(grid.ny)
    wy[0] = wy[-1] = 0.5
    bottom = grid.hx * np.sum(wx * u_bot)
    top = grid.hx * np.sum(wx * u_top)
    right = grid.hy * np.sum(wy * v_right)
    left = grid.hy * np.sum(wy * v_left)
    return float(bottom - top + right - left)


def max_divergence(u: np.ndarray, v: np.ndarray, grid: Grid) -> float:
    """Centered discrete divergence at interior points.

    Because ``u`` and ``v`` are obtained from the *same* streamfunction with the
    *same* centered stencil, the mixed differences cancel identically and this
    must be zero to round-off.  It is a check on the code, not on the physics.
    """
    div = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * grid.hx) + (
        v[1:-1, 2:] - v[1:-1, :-2]
    ) / (2.0 * grid.hy)
    return float(np.max(np.abs(div)))


def poisson_residual(psi: np.ndarray, omega: np.ndarray, grid: Grid) -> float:
    """``max |Delta_h psi + omega|`` at interior points: how well the linear
    solve satisfied the discrete Poisson equation."""
    return float(np.max(np.abs(laplacian(psi, grid) + omega[1:-1, 1:-1])))


# ---------------------------------------------------------------------------
# centerlines
# ---------------------------------------------------------------------------
def _interp_axis(
    values: np.ndarray, coords: np.ndarray, x0: float, axis: int
) -> np.ndarray:
    """Linear interpolation of a 2-D array along one axis at coordinate ``x0``."""
    i = int(np.clip(np.searchsorted(coords, x0) - 1, 0, len(coords) - 2))
    w = (x0 - coords[i]) / (coords[i + 1] - coords[i])
    lo = np.take(values, i, axis=axis)
    hi = np.take(values, i + 1, axis=axis)
    return (1.0 - w) * lo + w * hi


def centerline_u(u: np.ndarray, grid: Grid, x0: float = 0.5) -> np.ndarray:
    """``u`` along the vertical line ``x = x0`` (Ghia Table I)."""
    return _interp_axis(u, grid.x, x0, axis=0)


def centerline_v(v: np.ndarray, grid: Grid, y0: float = 0.5) -> np.ndarray:
    """``v`` along the horizontal line ``y = y0`` (Ghia Table II)."""
    return _interp_axis(v, grid.y, y0, axis=1)


# ---------------------------------------------------------------------------
# vortices
# ---------------------------------------------------------------------------
@dataclass
class Vortex:
    psi: float
    x: float
    y: float
    i: int
    j: int

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.psi, self.x, self.y)


def _refine_extremum(psi: np.ndarray, grid: Grid, i: int, j: int) -> Vortex:
    """Sub-grid location of an extremum by a least-squares biquadratic fit on
    the 3x3 neighbourhood."""
    if not (1 <= i < grid.nx - 1 and 1 <= j < grid.ny - 1):
        return Vortex(float(psi[i, j]), float(grid.x[i]), float(grid.y[j]), i, j)
    di = np.array([-1, 0, 1])
    dx = di[:, None] * grid.hx * np.ones(3)[None, :]
    dy = np.ones(3)[:, None] * di[None, :] * grid.hy
    p = psi[i - 1 : i + 2, j - 1 : j + 2]
    A = np.column_stack(
        [
            np.ones(9),
            dx.ravel(),
            dy.ravel(),
            0.5 * dx.ravel() ** 2,
            dx.ravel() * dy.ravel(),
            0.5 * dy.ravel() ** 2,
        ]
    )
    c, *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
    c0, cx, cy, cxx, cxy, cyy = c
    H = np.array([[cxx, cxy], [cxy, cyy]])
    try:
        shift = np.linalg.solve(H, [-cx, -cy])
    except np.linalg.LinAlgError:
        shift = np.zeros(2)
    if np.max(np.abs(shift)) > max(grid.hx, grid.hy):
        shift = np.zeros(2)
    sx, sy = shift
    val = c0 + cx * sx + cy * sy + 0.5 * cxx * sx**2 + cxy * sx * sy + 0.5 * cyy * sy**2
    return Vortex(float(val), float(grid.x[i] + sx), float(grid.y[j] + sy), i, j)


def primary_vortex(psi: np.ndarray, grid: Grid) -> Vortex:
    """Centre and strength of the primary (clockwise, ``psi < 0``) vortex."""
    i, j = np.unravel_index(np.argmin(psi), psi.shape)
    return _refine_extremum(psi, grid, int(i), int(j))


def corner_vortices(
    psi: np.ndarray, grid: Grid, frac: float = 0.5
) -> dict[str, Vortex]:
    """Strongest counter-rotating (``psi > 0``) cell in each corner quadrant.

    Returns an entry only where a counter-rotating cell actually exists.
    """
    nx, ny = grid.shape
    ix = int(frac * nx)
    iy = int(frac * ny)
    windows = {
        "BL": (slice(0, ix), slice(0, iy)),
        "BR": (slice(nx - ix, nx), slice(0, iy)),
        "TL": (slice(0, ix), slice(ny - iy, ny)),
        "TR": (slice(nx - ix, nx), slice(ny - iy, ny)),
    }
    out: dict[str, Vortex] = {}
    for name, (sx, sy) in windows.items():
        block = psi[sx, sy]
        k = int(np.argmax(block))
        bi, bj = np.unravel_index(k, block.shape)
        i = sx.start + int(bi)
        j = sy.start + int(bj)
        if psi[i, j] <= 0.0:
            continue
        out[name] = _refine_extremum(psi, grid, i, j)
    return out
