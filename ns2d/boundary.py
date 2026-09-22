"""Vorticity boundary conditions (Thom's formula).

The streamfunction problem carries both Dirichlet data (``psi = 0``, from
impermeability) and Neumann data (``dpsi/dn = g``, from no-slip), while the
vorticity carries none.  The Neumann data is therefore converted into Dirichlet
data for ``omega`` on the walls.

Derivation for the top wall ``y = 1`` (report, "Boundary Vorticity").  Taylor expansion
of ``psi`` one node below the wall,

    psi(x, 1-h) = psi(x,1) - h psi_y(x,1) + h^2/2 psi_yy(x,1) + O(h^3),

with ``psi(x,1) = 0``, ``psi_y(x,1) = u_top`` and -- because ``psi`` is constant
along the wall so its tangential derivatives vanish there --

    psi_yy(x,1) = -omega(x,1) - psi_xx(x,1) = -omega(x,1),

gives

    omega(x,1) = -2 psi(x,1-h)/h^2 - 2 u_top/h + O(h).

The other three walls follow identically; only the sign of the ``O(1/h)``
velocity term changes, according to the outward normal.

The local truncation error is ``O(h)``, i.e. the boundary treatment is formally
first order even though the interior stencils are second order.  This is the
classical Thom condition and does not, in general, destroy the second-order
global accuracy -- but that is a claim to be *measured*, which is what
``ns2d.verification`` does.

Note on corners: the four corner values of ``omega`` never enter any interior
stencil (neither the five-point Laplacian nor the centered convection term
reaches a corner from an interior node), so they are inert -- no choice of
corner value can change the computed solution.  They still have to be filled
in, because plots and norms read them, and one choice is distinguished:

``corner="sum"`` (the default) assigns each corner the *sum* of the two
adjacent wall formulas.  With that rule -- and only with that rule -- the
discrete Green identity

    hx hy sum_ij sigma_ij omega_ij  =  contour integral of u.dl        (exact)

holds to round-off, where ``sigma`` are the trapezoid weights.  The reason is
bookkeeping: in the two-dimensional trapezoid rule a corner carries weight
``1/4``, but the wall it belongs to needs it at weight ``1/2 * 1/2 = 1/4``
*for each* of the two walls that meet there, so the single stored value must
carry both contributions.  Averaging them supplies half of what the identity
needs and leaves an ``O(h)`` deficit; zeroing them leaves the whole ``O(h)``.
See :func:`ns2d.postproc.total_circulation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .grid import Grid

__all__ = ["WallVelocities", "thom_vorticity", "apply_thom"]

CornerRule = Literal["zero", "average", "sum"]


@dataclass
class WallVelocities:
    """Tangential wall velocities.

    ``top``/``bottom`` hold ``u`` along ``y = ly`` / ``y = 0`` (length ``nx``),
    ``left``/``right`` hold ``v`` along ``x = 0`` / ``x = lx`` (length ``ny``).
    Scalars are broadcast.
    """

    top: np.ndarray | float = 0.0
    bottom: np.ndarray | float = 0.0
    left: np.ndarray | float = 0.0
    right: np.ndarray | float = 0.0

    def arrays(
        self, grid: Grid
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.broadcast_to(np.asarray(self.top, dtype=float), (grid.nx,)),
            np.broadcast_to(np.asarray(self.bottom, dtype=float), (grid.nx,)),
            np.broadcast_to(np.asarray(self.left, dtype=float), (grid.ny,)),
            np.broadcast_to(np.asarray(self.right, dtype=float), (grid.ny,)),
        )


def thom_vorticity(
    psi: np.ndarray,
    grid: Grid,
    walls: WallVelocities,
    corner: CornerRule = "sum",
) -> np.ndarray:
    """Return the wall values of ``omega`` implied by ``psi`` and the wall velocities.

    Only the boundary entries of the returned array are meaningful; the interior
    is filled with zeros.

    ``corner`` selects the (inert) value stored at the four corners: ``"sum"``
    adds the two adjacent wall formulas, which is what makes the discrete
    circulation identity exact; ``"average"`` halves that; ``"zero"`` stores
    nothing.  See the module docstring.
    """
    u_top, u_bot, v_left, v_right = walls.arrays(grid)
    w = np.zeros(grid.shape)

    hx, hy = grid.hx, grid.hy
    # psi is zero on the wall, so psi(wall) does not appear below.
    w[:, -1] = -2.0 * psi[:, -2] / hy**2 - 2.0 * u_top / hy  # y = ly
    w[:, 0] = -2.0 * psi[:, 1] / hy**2 + 2.0 * u_bot / hy  # y = 0
    w[0, :] = -2.0 * psi[1, :] / hx**2 - 2.0 * v_left / hx  # x = 0
    w[-1, :] = -2.0 * psi[-2, :] / hx**2 + 2.0 * v_right / hx  # x = lx

    if corner == "zero":
        w[0, 0] = w[-1, 0] = w[0, -1] = w[-1, -1] = 0.0
        return w

    try:
        weight = {"sum": 1.0, "average": 0.5}[corner]
    except KeyError:
        raise ValueError(f"unknown corner rule {corner!r}") from None

    # each corner currently holds the x-wall formula (assigned last); combine it
    # with the y-wall formula for the same node
    y_wall = {
        (0, 0): -2.0 * psi[0, 1] / hy**2 + 2.0 * u_bot[0] / hy,
        (-1, 0): -2.0 * psi[-1, 1] / hy**2 + 2.0 * u_bot[-1] / hy,
        (0, -1): -2.0 * psi[0, -2] / hy**2 - 2.0 * u_top[0] / hy,
        (-1, -1): -2.0 * psi[-1, -2] / hy**2 - 2.0 * u_top[-1] / hy,
    }
    for (i, j), value in y_wall.items():
        w[i, j] = weight * (w[i, j] + value)
    return w


def apply_thom(
    omega: np.ndarray,
    psi: np.ndarray,
    grid: Grid,
    walls: WallVelocities,
    corner: CornerRule = "sum",
) -> None:
    """In-place update of the boundary ring of ``omega`` from ``psi``."""
    w = thom_vorticity(psi, grid, walls, corner=corner)
    omega[:, 0] = w[:, 0]
    omega[:, -1] = w[:, -1]
    omega[0, :] = w[0, :]
    omega[-1, :] = w[-1, :]
