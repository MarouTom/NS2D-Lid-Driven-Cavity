"""Problem definitions: the lid-driven cavity and manufactured solutions.

A *problem* supplies everything the solver needs beyond the discretisation:
the Reynolds number, the wall velocities, an optional source term, and -- for
verification -- the exact fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .boundary import WallVelocities
from .grid import Grid

__all__ = [
    "Problem",
    "Cavity",
    "ManufacturedSolution",
    "PSI_SMOOTH",
    "PSI_WALL_DRIVEN",
    "PSI_SINGLE_MODE",
]


class Problem:
    """Base class.  ``Re`` and ``walls`` are the minimum a solver needs."""

    Re: float

    def checkpoint_signature(self) -> dict:
        """Configuration needed for restart compatibility.

        Custom problems should override this and include every parameter that
        affects their source or walls. Built-in problems include their inputs.
        """
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "Re": float(self.Re),
        }

    def walls(self, grid: Grid, t: float) -> WallVelocities:  # pragma: no cover
        raise NotImplementedError

    def source(self, grid: Grid, t: float) -> np.ndarray | None:
        """Forcing ``f`` in ``omega_t + C(psi, omega) - Re^-1 Delta omega = f``."""
        return None

    def exact_psi(self, grid: Grid, t: float) -> np.ndarray | None:
        return None

    def exact_omega(self, grid: Grid, t: float) -> np.ndarray | None:
        return None


@dataclass
class Cavity(Problem):
    """The lid-driven cavity: top wall slides at ``u = u_lid``, no-slip elsewhere."""

    Re: float
    u_lid: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.Re) or self.Re <= 0:
            raise ValueError("Re must be finite and positive")
        if not np.isfinite(self.u_lid):
            raise ValueError("u_lid must be finite")

    def checkpoint_signature(self) -> dict:
        return {**super().checkpoint_signature(), "u_lid": float(self.u_lid)}

    def walls(self, grid: Grid, t: float) -> WallVelocities:
        return WallVelocities(top=self.u_lid, bottom=0.0, left=0.0, right=0.0)


# ---------------------------------------------------------------------------
# manufactured solutions
# ---------------------------------------------------------------------------
# The derivatives and the source term are derived symbolically with SymPy at
# construction time and turned into NumPy functions.  Nothing is differentiated
# by hand, which removes the most common source of error in an MMS study.

PSI_SMOOTH = "sin(pi*x)**2 * sin(pi*y)**2 * exp(-t)"
"""``psi`` and ``dpsi/dn`` both vanish on the whole boundary, so all four wall
velocities are zero.  Exercises the interior scheme and the ``psi``-dependent
part of Thom's formula, but *not* its velocity term."""

PSI_WALL_DRIVEN = "sin(pi*x)*sin(pi*y)*exp(-t) + sin(2*pi*x)*sin(2*pi*y)*exp(-2*t)/2"
"""``psi = 0`` on the boundary but ``dpsi/dn != 0``, so all four walls move.

This is the field that exercises the ``-2 u_wall/h`` term of Thom's formula,
i.e. the very mechanism by which the lid drives the cavity.

The second mode is not decorative.  For a single mode ``psi = sin(pi x)
sin(pi y) f(t)`` one gets ``omega = 2 pi^2 psi``, and the convection term

    psi_y omega_x - psi_x omega_y = 2 pi^2 (psi_y psi_x - psi_x psi_y) = 0

vanishes *identically*: the nonlinearity would never be tested.  Any
manufactured ``psi`` that is a single eigenfunction of the Laplacian has this
defect.  Superposing two modes with different eigenvalues removes it."""

PSI_SINGLE_MODE = "sin(pi*x)*sin(pi*y)*exp(-t)"
"""Single Laplacian eigenmode.  Kept only to demonstrate the trap described
above: :attr:`ManufacturedSolution.has_convection` is ``False`` for it."""


class ManufacturedSolution(Problem):
    """Manufactured solution built from a prescribed streamfunction.

    Given ``psi_ex(x, y, t)`` the class sets

        omega_ex = -Delta psi_ex

    so that the Poisson equation ``Delta psi = -omega`` is satisfied *exactly*
    at the continuous level and needs no source term of its own.  (Prescribing
    ``psi_ex`` and ``omega_ex`` independently would require a second source
    term in the Poisson equation -- a classical oversight.)

    The residual of the vorticity transport equation,

        f = omega_t + psi_y omega_x - psi_x omega_y - Re^-1 Delta omega ,

    is then the forcing that must be added to the discrete equation.

    Parameters
    ----------
    psi_expr
        SymPy-parsable expression in ``x``, ``y``, ``t``.
    Re
        Reynolds number.
    """

    def __init__(self, psi_expr: str = PSI_WALL_DRIVEN, Re: float = 100.0) -> None:
        import sympy as smp

        self.psi_expr_str = psi_expr
        self.Re = float(Re)
        if not np.isfinite(self.Re) or self.Re <= 0:
            raise ValueError("Re must be finite and positive")

        x, y, t = smp.symbols("x y t", real=True)
        psi = smp.sympify(psi_expr, locals={"x": x, "y": y, "t": t})

        omega = -(smp.diff(psi, x, 2) + smp.diff(psi, y, 2))
        u = smp.diff(psi, y)
        v = -smp.diff(psi, x)
        conv = smp.diff(psi, y) * smp.diff(omega, x) - smp.diff(psi, x) * smp.diff(
            omega, y
        )
        lap_omega = smp.diff(omega, x, 2) + smp.diff(omega, y, 2)
        f = smp.simplify(smp.diff(omega, t) + conv - lap_omega / self.Re)

        self._sym = {
            "psi": psi,
            "omega": omega,
            "u": u,
            "v": v,
            "f": f,
            "conv": smp.simplify(conv),
        }
        # A manufactured field that is a single Laplacian eigenmode makes the
        # convective term vanish identically, silently leaving the nonlinearity
        # untested.  Record whether that has happened.
        self.has_convection = bool(self._sym["conv"] != 0)
        args = (x, y, t)
        mods = ["numpy"]
        self._psi = smp.lambdify(args, psi, modules=mods)
        self._omega = smp.lambdify(args, omega, modules=mods)
        self._u = smp.lambdify(args, u, modules=mods)
        self._v = smp.lambdify(args, v, modules=mods)
        self._f = smp.lambdify(args, f, modules=mods)

    # -- helpers ---------------------------------------------------------
    def checkpoint_signature(self) -> dict:
        return {**super().checkpoint_signature(), "psi_expr": self.psi_expr_str}

    @staticmethod
    def _eval(fn: Callable, X: np.ndarray, Y: np.ndarray, t: float) -> np.ndarray:
        out = fn(X, Y, t)
        return np.broadcast_to(np.asarray(out, dtype=float), X.shape).copy()

    # -- Problem interface ------------------------------------------------
    def walls(self, grid: Grid, t: float) -> WallVelocities:
        X, Y = grid.X, grid.Y
        u = self._eval(self._u, X, Y, t)
        v = self._eval(self._v, X, Y, t)
        return WallVelocities(
            top=u[:, -1].copy(),
            bottom=u[:, 0].copy(),
            left=v[0, :].copy(),
            right=v[-1, :].copy(),
        )

    def source(self, grid: Grid, t: float) -> np.ndarray:
        return self._eval(self._f, grid.X, grid.Y, t)

    def exact_psi(self, grid: Grid, t: float) -> np.ndarray:
        return self._eval(self._psi, grid.X, grid.Y, t)

    def exact_omega(self, grid: Grid, t: float) -> np.ndarray:
        return self._eval(self._omega, grid.X, grid.Y, t)

    def exact_velocity(self, grid: Grid, t: float) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._eval(self._u, grid.X, grid.Y, t),
            self._eval(self._v, grid.X, grid.Y, t),
        )

    def max_speed(self, grid: Grid, t: float = 0.0) -> float:
        u, v = self.exact_velocity(grid, t)
        return float(np.max(np.hypot(u, v)))

    def pretty(self) -> str:  # pragma: no cover - cosmetic
        import sympy as smp

        lines = [
            f"psi    = {smp.sstr(self._sym['psi'])}",
            f"omega  = {smp.sstr(smp.simplify(self._sym['omega']))}",
            f"C      = {smp.sstr(self._sym['conv'])}",
            f"f      = {smp.sstr(self._sym['f'])}",
            f"tests convection: {self.has_convection}",
        ]
        return "\n".join(lines)
