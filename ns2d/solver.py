"""Semi-implicit vorticity-streamfunction solver.

One time step (report, Algorithm 1, "The Algorithm"):

1. wall values of ``omega^n`` from ``psi^n`` (Thom, or prescribed in MMS mode);
2. explicit centered convection ``C(psi^n, omega^n)`` at interior points;
3. solve ``(I + (dt/Re)(-Delta_h)) omega^{n+1} = omega^n - dt C + dt f``;
4. solve ``(-Delta_h) psi^{n+1} = omega^{n+1}`` with ``psi = 0`` on the wall.

Known boundary values are moved to the right-hand side by applying the discrete
Laplacian to a field that carries the boundary ring and zeros elsewhere; this is
algebraically identical to eliminating the Dirichlet columns of the matrix.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .boundary import CornerRule, thom_vorticity
from .grid import Grid
from .operators import Backend, ddx, ddy, laplacian, make_solver, suggest_dt
from .problems import Cavity, Problem

__all__ = ["Solver", "History", "RunResult", "suggest_dt"]

OmegaBC = Literal["thom", "exact"]


# ---------------------------------------------------------------------------
@dataclass
class History:
    """Snapshots recorded during a run (used for the animations)."""

    times: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    psi: list[np.ndarray] = field(default_factory=list)
    omega: list[np.ndarray] = field(default_factory=list)

    def append(self, t: float, n: int, psi: np.ndarray, omega: np.ndarray) -> None:
        self.times.append(float(t))
        self.steps.append(int(n))
        self.psi.append(psi.copy())
        self.omega.append(omega.copy())

    def __len__(self) -> int:
        return len(self.times)

    def stack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.times), np.asarray(self.psi), np.asarray(self.omega)

    def save(self, path: str) -> None:
        t, p, w = self.stack()
        np.savez_compressed(path, times=t, psi=p, omega=w, steps=np.asarray(self.steps))

    @staticmethod
    def load(path: str) -> "History":
        d = np.load(path)
        h = History()
        h.times = list(map(float, d["times"]))
        h.steps = list(map(int, d["steps"]))
        h.psi = list(d["psi"])
        h.omega = list(d["omega"])
        return h


@dataclass
class RunResult:
    """Diagnostics returned by :meth:`Solver.run`."""

    steps: int
    t_final: float
    residuals: np.ndarray  # ||omega^{n+1}-omega^n||_inf / dt, interior
    residual_times: np.ndarray
    kinetic_energy: np.ndarray
    converged: bool
    wall_time: float
    history: History | None = None

    @property
    def final_residual(self) -> float:
        return float(self.residuals[-1]) if len(self.residuals) else np.nan


# ---------------------------------------------------------------------------
class Solver:
    """Semi-implicit finite-difference solver for the 2-D vorticity equation.

    Parameters
    ----------
    grid, problem
        Mesh and problem definition (:class:`~ns2d.problems.Cavity` or
        :class:`~ns2d.problems.ManufacturedSolution`).
    dt
        Time step.  ``None`` uses :func:`~ns2d.operators.suggest_dt`.
    backend
        Linear-solver backend: ``"dst"`` (default, exact and fastest),
        ``"lu"``, or ``"cg"`` (preconditioned conjugate gradient).
    omega_bc
        ``"thom"`` reconstructs the wall vorticity from the numerical ``psi``
        (this is what the cavity uses).  ``"exact"`` prescribes it from the
        manufactured solution and is only meaningful for verification.
    corner
        Value stored at the four corner nodes of ``omega``.  These enter no
        stencil and cannot change the solution; the default ``"sum"`` is the
        one that makes the discrete circulation identity exact (see
        :mod:`ns2d.boundary`).
    """

    def __init__(
        self,
        grid: Grid,
        problem: Problem | None = None,
        dt: float | None = None,
        backend: Backend = "dst",
        omega_bc: OmegaBC = "thom",
        corner: CornerRule = "sum",
        safety: float = 0.5,
        solver_kwargs: dict | None = None,
    ) -> None:
        self.grid = grid
        self.problem = problem if problem is not None else Cavity(Re=400.0)
        self.Re = float(self.problem.Re)
        if not np.isfinite(self.Re) or self.Re <= 0:
            raise ValueError("Re must be finite and positive")
        speed = abs(self.problem.u_lid) if isinstance(self.problem, Cavity) else 1.0
        self.dt = (
            float(dt)
            if dt is not None
            else suggest_dt(grid, self.Re, u_max=speed, safety=safety)
        )
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if omega_bc not in ("thom", "exact"):
            raise ValueError("omega_bc must be 'thom' or 'exact'")
        if corner not in ("sum", "average", "zero"):
            raise ValueError("corner must be 'sum', 'average', or 'zero'")
        self.backend = backend
        self.omega_bc = omega_bc
        self.corner = corner

        kw = dict(solver_kwargs or {})
        beta = self.dt / self.Re
        self.solver_omega = make_solver(
            grid, alpha=1.0, beta=beta, backend=backend, **kw
        )
        self.solver_psi = make_solver(grid, alpha=0.0, beta=1.0, backend=backend, **kw)
        self._beta = beta

        self.psi = grid.zeros()
        self.omega = grid.zeros()
        self.t = 0.0
        self.n = 0

    # -- initial conditions ----------------------------------------------
    def set_initial_condition(
        self,
        psi: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        t: float = 0.0,
    ) -> None:
        """Reset state and step count; derive the missing field if one is given.

        Supplying both fields is useful for continuous manufactured solutions:
        they need not satisfy the discrete Poisson equation initially.
        """
        if not np.isfinite(t) or t < 0:
            raise ValueError("initial time must be finite and nonnegative")
        for name, value in (("psi", psi), ("omega", omega)):
            if value is not None and (
                np.shape(value) != self.grid.shape or not np.isfinite(value).all()
            ):
                raise ValueError(f"{name} must be finite with shape {self.grid.shape}")
        if psi is not None:
            p = np.array(psi, dtype=float)
            if not np.allclose(
                np.concatenate((p[0], p[-1], p[:, 0], p[:, -1])),
                0.0,
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("psi must vanish on the boundary")
        else:
            p = self.grid.zeros()
        if omega is not None:
            w = np.array(omega, dtype=float)
            if psi is None:
                p[1:-1, 1:-1] = self.solver_psi.solve(w[1:-1, 1:-1])
        else:
            w = self.boundary_omega(p, t)
            w[1:-1, 1:-1] = -laplacian(p, self.grid)
        self.psi, self.omega = p, w
        self.t = float(t)
        self.n = 0

    def initialise_from_exact(self, t: float = 0.0) -> None:
        """Start an MMS run from the exact fields at time ``t``."""
        psi = self.problem.exact_psi(self.grid, t)
        omega = self.problem.exact_omega(self.grid, t)
        if psi is None or omega is None:
            raise ValueError("problem has no exact solution")
        self.set_initial_condition(psi=psi, omega=omega, t=t)

    # -- pieces of one step ----------------------------------------------
    def boundary_omega(self, psi: np.ndarray, t: float) -> np.ndarray:
        """Full-grid array holding the wall vorticity on its ring, zero inside."""
        if self.omega_bc == "exact":
            exact = self.problem.exact_omega(self.grid, t)
            if exact is None:
                raise ValueError(
                    "omega_bc='exact' requires a problem with an exact solution"
                )
            w = np.zeros(self.grid.shape)
            w[0, :] = exact[0, :]
            w[-1, :] = exact[-1, :]
            w[:, 0] = exact[:, 0]
            w[:, -1] = exact[:, -1]
            return w
        return thom_vorticity(
            psi, self.grid, self.problem.walls(self.grid, t), corner=self.corner
        )

    def convection(self, psi: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """``psi_y omega_x - psi_x omega_y`` with centered differences (interior)."""
        g = self.grid
        return ddy(psi, g) * ddx(omega, g) - ddx(psi, g) * ddy(omega, g)

    def step(self) -> None:
        """Advance the solution by one time step."""
        g, dt = self.grid, self.dt

        # 1. wall vorticity from the most recent streamfunction
        w_ring = self.boundary_omega(self.psi, self.t)
        self.omega[0, :] = w_ring[0, :]
        self.omega[-1, :] = w_ring[-1, :]
        self.omega[:, 0] = w_ring[:, 0]
        self.omega[:, -1] = w_ring[:, -1]

        # 2. explicit convection
        conv = self.convection(self.psi, self.omega)

        # 3. right-hand side of the vorticity equation
        rhs = self.omega[1:-1, 1:-1] - dt * conv
        src = self.problem.source(g, self.t + dt)
        if src is not None:
            rhs = rhs + dt * src[1:-1, 1:-1]

        # Dirichlet data for omega^{n+1}: the same ring -- a second update here
        # could only re-use psi^n and would reproduce the first one -- or,
        # in MMS/exact mode, the exact values at the new time level.
        ring_new = (
            w_ring
            if self.omega_bc == "thom"
            else self.boundary_omega(self.psi, self.t + dt)
        )
        ring_only = np.zeros(g.shape)
        ring_only[0, :] = ring_new[0, :]
        ring_only[-1, :] = ring_new[-1, :]
        ring_only[:, 0] = ring_new[:, 0]
        ring_only[:, -1] = ring_new[:, -1]
        rhs = rhs + self._beta * laplacian(ring_only, g)

        omega_int = self.solver_omega.solve(rhs)

        # 4. Poisson problem for the streamfunction, psi = 0 on the wall
        psi_int = self.solver_psi.solve(omega_int)

        new_omega = ring_only
        new_omega[1:-1, 1:-1] = omega_int
        self.omega = new_omega
        self.psi = np.zeros(g.shape)
        self.psi[1:-1, 1:-1] = psi_int

        self.t += dt
        self.n += 1

    # -- driver -----------------------------------------------------------
    def run(
        self,
        T: float | None = None,
        n_steps: int | None = None,
        steady_tol: float | None = None,
        save_every: int | None = None,
        monitor_every: int = 20,
        callback: Callable[["Solver"], None] | None = None,
        progress: bool = True,
        record: bool = True,
    ) -> RunResult:
        """Time-march the solution.

        Parameters
        ----------
        T, n_steps
            Supply exactly one: advance to the first fixed-step time at or
            beyond absolute time ``T``, or take ``n_steps`` additional steps.
            The final step is not shortened; overshoot is less than ``dt``.
        steady_tol
            Stop early when ``||omega^{n+1}-omega^n||_inf / dt < steady_tol``
            (measured on interior points, so the mesh-dependent corner values do
            not pollute the criterion).
        save_every
            Record a snapshot every this many steps (for the animations).
        """
        if (T is None) == (n_steps is None):
            raise ValueError("give exactly one of T or n_steps")
        for name, value in (
            ("monitor_every", monitor_every),
            ("save_every", save_every),
            ("n_steps", n_steps),
        ):
            if value is None and name != "monitor_every":
                continue
            minimum = 0 if name == "n_steps" else 1
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if T is not None and (not np.isfinite(T) or T < self.t):
            raise ValueError("T must be finite and no earlier than the current time")
        if steady_tol is not None and (not np.isfinite(steady_tol) or steady_tol <= 0):
            raise ValueError("steady_tol must be finite and positive")
        if n_steps is None:
            remaining = (T - self.t) / self.dt
            nearest = round(remaining)
            total = (
                nearest
                if np.isclose(remaining, nearest, rtol=0, atol=1e-9)
                else int(np.ceil(remaining))
            )
        else:
            total = n_steps
        history = History() if (record and save_every) else None
        if history is not None:
            history.append(self.t, self.n, self.psi, self.omega)

        residuals: list[float] = []
        res_times: list[float] = []
        energies: list[float] = []
        converged = False
        t0 = time.perf_counter()
        last_print = t0

        for k in range(total):
            # step() rebinds self.omega but also mutates the boundary ring of
            # the old array in place, so keep an explicit copy of the interior
            # rather than relying on that aliasing
            measuring = (k + 1) % monitor_every == 0 or k == total - 1
            omega_prev = self.omega[1:-1, 1:-1].copy() if measuring else None
            self.step()

            if not np.isfinite(self.omega).all():
                raise FloatingPointError(
                    f"solution blew up at step {self.n} (t={self.t:.4g}); "
                    f"dt={self.dt:.3g} is probably too large -- see suggest_dt()"
                )

            if measuring:
                r = float(np.max(np.abs(self.omega[1:-1, 1:-1] - omega_prev)) / self.dt)
                residuals.append(r)
                res_times.append(self.t)
                energies.append(self.kinetic_energy())
                if steady_tol is not None and r < steady_tol:
                    converged = True
            if history is not None and (k + 1) % save_every == 0:
                history.append(self.t, self.n, self.psi, self.omega)
            if callback is not None:
                callback(self)

            if progress and (time.perf_counter() - last_print) > 2.0:
                last_print = time.perf_counter()
                r = residuals[-1] if residuals else np.nan
                print(
                    f"\r  step {k + 1}/{total}  t={self.t:8.3f}  R={r:.3e}"
                    f"  ({(k + 1) / (time.perf_counter() - t0):.0f} steps/s)",
                    end="",
                    flush=True,
                )
            if converged:
                break

        wall = time.perf_counter() - t0
        if progress:
            print(
                f"\r  {self.n} steps, t={self.t:.3f}, "
                f"R={residuals[-1] if residuals else float('nan'):.3e}, "
                f"{wall:.1f} s{' (steady)' if converged else ''}" + " " * 20
            )
        if history is not None and (history.steps[-1] != self.n):
            history.append(self.t, self.n, self.psi, self.omega)

        return RunResult(
            steps=self.n,
            t_final=self.t,
            residuals=np.asarray(residuals),
            residual_times=np.asarray(res_times),
            kinetic_energy=np.asarray(energies),
            converged=converged,
            wall_time=wall,
            history=history,
        )

    # -- diagnostics -------------------------------------------------------
    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """``u = dpsi/dy``, ``v = -dpsi/dx`` (see :mod:`ns2d.postproc`)."""
        from .postproc import velocity_from_psi

        return velocity_from_psi(
            self.psi, self.grid, self.problem.walls(self.grid, self.t)
        )

    def kinetic_energy(self) -> float:
        from .postproc import kinetic_energy

        u, v = self.velocity()
        return kinetic_energy(u, v, self.grid)

    # -- checkpointing ----------------------------------------------------
    def save(self, path: str) -> None:
        """Write state and configuration; strict reload checks compatibility."""
        np.savez_compressed(
            path,
            psi=self.psi,
            omega=self.omega,
            t=self.t,
            n=self.n,
            dt=self.dt,
            Re=self.Re,
            nx=self.grid.nx,
            ny=self.grid.ny,
            lx=self.grid.lx,
            ly=self.grid.ly,
            checkpoint_version=2,
            configuration=json.dumps(self._checkpoint_configuration(), sort_keys=True),
        )

    def _checkpoint_configuration(self) -> dict:
        return {
            "problem": self.problem.checkpoint_signature(),
            "omega_bc": self.omega_bc,
            "corner": self.corner,
        }

    def load(self, path: str, strict: bool = True) -> None:
        """Load state without changing this solver's numerical configuration.

        Strict mode rejects incompatible time steps, domains, wall models and
        problem parameters. ``strict=False`` explicitly imports a state for a
        different configuration (also permits old checkpoints); shape and
        finite-value checks still apply. Backend changes are permitted, with
        agreement to solver tolerance rather than bitwise identity.
        """
        with np.load(path, allow_pickle=False) as d:
            if (int(d["nx"]), int(d["ny"])) != self.grid.shape:
                raise ValueError("checkpoint grid does not match this solver")
            if strict:
                if "checkpoint_version" not in d or int(d["checkpoint_version"]) != 2:
                    raise ValueError(
                        "legacy or unsupported checkpoint; use strict=False to import state"
                    )
                for key, expected in (
                    ("dt", self.dt),
                    ("Re", self.Re),
                    ("lx", self.grid.lx),
                    ("ly", self.grid.ly),
                ):
                    if not np.isclose(float(d[key]), expected, rtol=1e-14, atol=0):
                        raise ValueError(f"checkpoint {key} does not match this solver")
                if (
                    json.loads(str(d["configuration"]))
                    != self._checkpoint_configuration()
                ):
                    raise ValueError(
                        "checkpoint problem or boundary configuration does not match"
                    )
            p, w = d["psi"].copy(), d["omega"].copy()
            t, n = float(d["t"]), float(d["n"])
            if (
                p.shape != self.grid.shape
                or w.shape != self.grid.shape
                or not np.isfinite(p).all()
                or not np.isfinite(w).all()
                or not np.isfinite(t)
                or t < 0
                or not np.isfinite(n)
                or n < 0
                or n != int(n)
            ):
                raise ValueError("checkpoint contains invalid state")
        self.psi, self.omega, self.t, self.n = p, w, t, int(n)
        for solver in (self.solver_psi, self.solver_omega):
            if hasattr(solver, "_x0"):
                solver._x0 = None
