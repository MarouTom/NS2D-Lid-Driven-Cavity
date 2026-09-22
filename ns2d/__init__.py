"""``ns2d`` -- 2-D incompressible Navier-Stokes in vorticity-streamfunction form.

Semi-implicit finite-difference solver for the lid-driven cavity:

    omega_t + psi_y omega_x - psi_x omega_y - Re^-1 Delta omega = 0
    Delta psi = -omega,      u = psi_y,   v = -psi_x

Quick start
-----------
>>> from ns2d import Grid, Cavity, Solver, suggest_dt
>>> grid = Grid.square(129)
>>> problem = Cavity(Re=400)
>>> solver = Solver(grid, problem, dt=suggest_dt(grid, 400))
>>> result = solver.run(T=30.0, steady_tol=1e-6, save_every=200)
"""

from .grid import Grid
from .boundary import WallVelocities, apply_thom, thom_vorticity
from .operators import (
    laplacian,
    laplacian_matrix,
    neg_laplacian_matrix,
    helmholtz_matrix,
    incomplete_cholesky,
    make_solver,
    suggest_dt,
    D_MAX,
)
from .problems import (
    Cavity,
    ManufacturedSolution,
    PSI_SMOOTH,
    PSI_WALL_DRIVEN,
    PSI_SINGLE_MODE,
)
from .solver import Solver, History, RunResult
from . import benchmark, postproc, stability, verification, viz

__version__ = "1.3.0"

__all__ = [
    "Grid",
    "WallVelocities",
    "apply_thom",
    "thom_vorticity",
    "laplacian",
    "laplacian_matrix",
    "neg_laplacian_matrix",
    "helmholtz_matrix",
    "incomplete_cholesky",
    "make_solver",
    "suggest_dt",
    "D_MAX",
    "Cavity",
    "ManufacturedSolution",
    "PSI_SMOOTH",
    "PSI_WALL_DRIVEN",
    "PSI_SINGLE_MODE",
    "Solver",
    "History",
    "RunResult",
    "benchmark",
    "postproc",
    "stability",
    "verification",
    "viz",
    "__version__",
]
