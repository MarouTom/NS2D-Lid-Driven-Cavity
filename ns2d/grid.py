"""Cartesian grid for the vorticity-streamfunction solver.

Fields are stored as 2-D arrays ``phi[i, j]`` approximating ``phi(x_i, y_j)``
(i.e. ``indexing='ij'``, x is the first axis).  This matches the index
convention of the report and keeps the finite-difference stencils readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Grid"]


@dataclass(frozen=True)
class Grid:
    """Uniform Cartesian mesh on ``[0, lx] x [0, ly]``.

    Parameters
    ----------
    nx, ny
        Number of grid *points* per direction, boundary points included.
        A ``129 x 129`` grid therefore has ``h = 1/128``.
    """

    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    hx: float = field(init=False)
    hy: float = field(init=False)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if any(
            isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            for n in (self.nx, self.ny)
        ):
            raise ValueError("grid point counts must be integers")
        if self.nx < 3 or self.ny < 3:
            raise ValueError("need at least 3 points per direction")
        if not all(np.isfinite(v) and v > 0 for v in (self.lx, self.ly)):
            raise ValueError("domain lengths must be finite and positive")
        object.__setattr__(self, "hx", self.lx / (self.nx - 1))
        object.__setattr__(self, "hy", self.ly / (self.ny - 1))
        x = np.linspace(0.0, self.lx, self.nx)
        y = np.linspace(0.0, self.ly, self.ny)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        X, Y = np.meshgrid(x, y, indexing="ij")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)

    # -- convenience ------------------------------------------------------
    @property
    def h(self) -> float:
        """Representative mesh size ``max(hx, hy)``."""
        return max(self.hx, self.hy)

    @property
    def h_min(self) -> float:
        """Smallest mesh size ``min(hx, hy)``.

        This is the one that binds in the stability bounds of
        :func:`~ns2d.operators.suggest_dt`, since both the convective CFL and
        the diffusion number are worst where the mesh is finest.
        """
        return min(self.hx, self.hy)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def interior_shape(self) -> tuple[int, int]:
        return (self.nx - 2, self.ny - 2)

    @property
    def n_interior(self) -> int:
        return (self.nx - 2) * (self.ny - 2)

    def zeros(self) -> np.ndarray:
        return np.zeros(self.shape)

    @staticmethod
    def square(n: int, length: float = 1.0) -> "Grid":
        return Grid(nx=n, ny=n, lx=length, ly=length)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Grid({self.nx}x{self.ny}, hx={self.hx:.4g}, hy={self.hy:.4g})"
