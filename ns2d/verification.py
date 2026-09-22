"""Verification: method of manufactured solutions and observed convergence order.

Manufactured solutions test discretisation accuracy. Numerical literature
comparisons are benchmark validation, not experimental validation of a physical
model or a proof of correctness for every flow.

Two boundary-condition modes are studied, because they test different code:

``"exact"``
    The wall values of ``omega`` are taken from the manufactured solution.
    This exercises the interior stencils, the time integration and the linear
    solvers, and nothing else.
``"thom"``
    The wall values are reconstructed by Thom's formula from the *numerical*
    streamfunction, exactly as in the cavity run, with the wall velocity taken
    from ``dpsi_ex/dn``.  This is the only mode that exercises
    :mod:`ns2d.boundary` -- the part of the code most likely to be wrong.

Reporting only the first mode would leave the boundary treatment unverified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Grid
from .postproc import l2_norm
from .problems import PSI_WALL_DRIVEN, ManufacturedSolution
from .solver import Solver

__all__ = [
    "ErrorSet",
    "ConvergenceStudy",
    "run_mms",
    "mms_errors",
    "spatial_convergence",
    "temporal_convergence",
    "joint_convergence",
    "observed_orders",
    "format_table",
]


@dataclass
class ErrorSet:
    n: int
    h: float
    dt: float
    steps: int
    psi_l2: float
    psi_inf: float
    omega_l2: float
    omega_inf: float

    def as_row(self) -> dict:
        return {
            "N": self.n,
            "h": self.h,
            "dt": self.dt,
            "steps": self.steps,
            "psi_L2": self.psi_l2,
            "psi_inf": self.psi_inf,
            "omega_L2": self.omega_l2,
            "omega_inf": self.omega_inf,
        }


@dataclass
class ConvergenceStudy:
    label: str
    refine: str  # 'space', 'time' or 'joint'
    errors: list[ErrorSet] = field(default_factory=list)

    @property
    def orders(self) -> dict[str, np.ndarray]:
        return {
            key: observed_orders(
                np.array([getattr(e, key) for e in self.errors]),
                np.array([e.h if self.refine != "time" else e.dt for e in self.errors]),
            )
            for key in ("psi_l2", "psi_inf", "omega_l2", "omega_inf")
        }

    def table(self, keys=("psi_l2", "omega_l2", "omega_inf")) -> str:
        return format_table(self, keys)


def observed_orders(errors: np.ndarray, spacings: np.ndarray) -> np.ndarray:
    """``p_k = ln(E_{k-1}/E_k) / ln(s_{k-1}/s_k)``; first entry is NaN."""
    errors = np.asarray(errors, dtype=float)
    spacings = np.asarray(spacings, dtype=float)
    p = np.full(errors.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        p[1:] = np.log(errors[:-1] / errors[1:]) / np.log(spacings[:-1] / spacings[1:])
    return p


def format_table(
    study: ConvergenceStudy, keys=("psi_l2", "omega_l2", "omega_inf")
) -> str:
    orders = study.orders
    head = f"{'N':>5} {'h':>10} {'dt':>10}"
    for k in keys:
        head += f" {k:>12} {'p':>6}"
    lines = [study.label, "-" * len(head), head, "-" * len(head)]
    for i, e in enumerate(study.errors):
        row = f"{e.n:>5} {e.h:>10.3e} {e.dt:>10.3e}"
        for k in keys:
            p = orders[k][i]
            row += (
                f" {getattr(e, k):>12.4e} {'    --' if np.isnan(p) else f'{p:>6.2f}'}"
            )
        lines.append(row)
    lines.append("-" * len(head))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def run_mms(
    n: int,
    dt: float,
    T: float,
    Re: float = 100.0,
    psi_expr: str = PSI_WALL_DRIVEN,
    omega_bc: str = "thom",
    backend: str = "dst",
    problem: ManufacturedSolution | None = None,
) -> tuple[Solver, ManufacturedSolution, int]:
    """Integrate the manufactured problem from the exact state at ``t = 0`` to ``T``."""
    grid = Grid.square(n)
    prob = problem if problem is not None else ManufacturedSolution(psi_expr, Re=Re)
    solver = Solver(grid, prob, dt=dt, backend=backend, omega_bc=omega_bc)
    solver.initialise_from_exact(t=0.0)

    n_steps = int(round(T / dt))
    if abs(n_steps * dt - T) > 1e-9 * max(1.0, T):
        raise ValueError(f"T={T} is not an integer multiple of dt={dt}")
    solver.run(n_steps=n_steps, progress=False, record=False)
    return solver, prob, n_steps


def _error_set(
    solver: Solver,
    psi_ref: np.ndarray,
    omega_ref: np.ndarray,
    n: int,
    dt: float,
    steps: int,
) -> ErrorSet:
    grid = solver.grid
    dpsi = solver.psi - psi_ref
    domega = solver.omega - omega_ref
    # Exclude inert corner values consistently from both vorticity norms.
    # Retain the original quadrature weights at all remaining nodes.
    mask = np.ones(grid.shape, dtype=bool)
    mask[0, 0] = mask[-1, 0] = mask[0, -1] = mask[-1, -1] = False
    domega = np.where(mask, domega, 0.0)
    return ErrorSet(
        n=n,
        h=grid.h,
        dt=dt,
        steps=steps,
        psi_l2=l2_norm(dpsi, grid),
        psi_inf=float(np.max(np.abs(dpsi))),
        omega_l2=l2_norm(domega, grid),
        omega_inf=float(np.max(np.abs(domega[mask]))),
    )


def mms_errors(
    n: int,
    dt: float,
    T: float,
    Re: float = 100.0,
    psi_expr: str = PSI_WALL_DRIVEN,
    omega_bc: str = "thom",
    backend: str = "dst",
    problem: ManufacturedSolution | None = None,
) -> ErrorSet:
    """Error against the *exact* manufactured solution at time ``T``."""
    solver, prob, steps = run_mms(n, dt, T, Re, psi_expr, omega_bc, backend, problem)
    return _error_set(
        solver,
        prob.exact_psi(solver.grid, solver.t),
        prob.exact_omega(solver.grid, solver.t),
        n,
        dt,
        steps,
    )


def spatial_convergence(
    sizes=(17, 33, 65, 129),
    dt: float = 1.25e-5,
    T: float = 0.01,
    Re: float = 100.0,
    psi_expr: str = PSI_WALL_DRIVEN,
    omega_bc: str = "thom",
    backend: str = "dst",
    verbose: bool = True,
) -> ConvergenceStudy:
    """Refine ``h`` at *fixed*, small ``dt``.

    The time step must be small enough that the ``O(dt)`` temporal error stays
    below the spatial error on the *finest* grid; otherwise the error stagnates
    and the observed order falls spuriously towards zero.
    """
    label = (
        f"Spatial convergence  (omega BC: {omega_bc}, Re={Re:g}, "
        f"dt={dt:g} fixed, T={T:g})"
    )
    study = ConvergenceStudy(label=label, refine="space")
    prob = ManufacturedSolution(psi_expr, Re=Re)
    for n in sizes:
        e = mms_errors(
            n, dt=dt, T=T, Re=Re, omega_bc=omega_bc, backend=backend, problem=prob
        )
        study.errors.append(e)
        if verbose:
            print(
                f"  N={n:4d}  h={e.h:.4e}  |psi|_L2={e.psi_l2:.4e}  "
                f"|omega|_L2={e.omega_l2:.4e}"
            )
    return study


def temporal_convergence(
    n: int = 65,
    dts=(3.2e-4, 1.6e-4, 8e-5, 4e-5),
    T: float = 0.032,
    Re: float = 100.0,
    psi_expr: str = PSI_WALL_DRIVEN,
    omega_bc: str = "thom",
    backend: str = "dst",
    reference: str = "self",
    ref_factor: int = 16,
    verbose: bool = True,
) -> ConvergenceStudy:
    """Refine ``dt`` on a *fixed* grid.

    Measuring the temporal order against the *exact* solution does not work
    here: the spatial error is a fixed floor of the same size as the temporal
    error at the accessible time steps (``dt`` is capped by stability), so the
    error stagnates and the observed order collapses towards zero.  That is the
    mirror image of the trap described in the report for spatial refinement.

    The fix is to measure against a *reference solution computed on the same
    grid* with ``dt_ref = min(dts)/ref_factor``.  The spatial error is then
    largely shared by both solutions, so its leading contribution cancels.
    The finite reference step still slightly biases the measured order.
    Set ``reference='exact'`` to compare with the continuous field instead.
    """
    label = (
        f"Temporal convergence  (omega BC: {omega_bc}, Re={Re:g}, "
        f"N={n} fixed, T={T:g}, reference: {reference})"
    )
    study = ConvergenceStudy(label=label, refine="time")
    prob = ManufacturedSolution(psi_expr, Re=Re)

    if reference == "self":
        dt_ref = min(dts) / ref_factor
        steps_ref = int(round(T / dt_ref))
        dt_ref = T / steps_ref
        ref_solver, _, _ = run_mms(
            n, dt_ref, T, Re, omega_bc=omega_bc, backend=backend, problem=prob
        )
        psi_ref, omega_ref = ref_solver.psi, ref_solver.omega
        if verbose:
            print(f"  reference solution: dt_ref = {dt_ref:.3e} ({steps_ref} steps)")
    else:
        grid = Grid.square(n)
        psi_ref = prob.exact_psi(grid, T)
        omega_ref = prob.exact_omega(grid, T)

    for dt in dts:
        solver, _, steps = run_mms(
            n, dt, T, Re, omega_bc=omega_bc, backend=backend, problem=prob
        )
        e = _error_set(solver, psi_ref, omega_ref, n, dt, steps)
        study.errors.append(e)
        if verbose:
            print(
                f"  dt={dt:.4e}  |psi|_L2={e.psi_l2:.4e}  |omega|_L2={e.omega_l2:.4e}"
            )
    return study


def joint_convergence(
    sizes=(17, 33, 65),
    dt0: float = 1.0e-3,
    n0: int = 17,
    T: float = 0.016,
    Re: float = 100.0,
    psi_expr: str = PSI_WALL_DRIVEN,
    omega_bc: str = "thom",
    backend: str = "dst",
    verbose: bool = True,
) -> ConvergenceStudy:
    """Refine ``dt`` proportionally to ``h**2`` so both error contributions fall
    at the same rate; the observed order measured against ``h`` should be 2."""
    label = f"Joint refinement dt ~ h^2  (omega BC: {omega_bc}, Re={Re:g}, T={T:g})"
    study = ConvergenceStudy(label=label, refine="space")
    prob = ManufacturedSolution(psi_expr, Re=Re)
    h0 = 1.0 / (n0 - 1)
    for n in sizes:
        h = 1.0 / (n - 1)
        dt = dt0 * (h / h0) ** 2
        steps = max(1, int(round(T / dt)))
        dt = T / steps
        e = mms_errors(
            n, dt=dt, T=T, Re=Re, omega_bc=omega_bc, backend=backend, problem=prob
        )
        study.errors.append(e)
        if verbose:
            print(
                f"  N={n:4d}  dt={dt:.3e}  |psi|_L2={e.psi_l2:.4e}  "
                f"|omega|_L2={e.omega_l2:.4e}"
            )
    return study
