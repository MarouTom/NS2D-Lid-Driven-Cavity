#!/usr/bin/env python
"""Run the production cavity cases and cache the results in ``data/``.

The notebook loads these caches instead of recomputing, so it opens quickly.
Delete a ``.npz`` file (or pass ``--force``) to recompute that case.

    python scripts/run_production.py            # all default cases
    python scripts/run_production.py --quick    # coarse grids only
    python scripts/run_production.py --case Re3200_N257
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
import platform
import scipy
from pathlib import Path
from datetime import datetime, timezone
from scipy.fft import set_workers

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ns2d import Cavity, Grid, Solver, suggest_dt, __version__  # noqa: E402
from ns2d.benchmark import GHIA_VORTEX, compare_to_ghia  # noqa: E402
from ns2d.postproc import (  # noqa: E402
    circulation_identity,
    corner_vortices,
    expected_circulation,
    max_divergence,
    poisson_residual,
    primary_vortex,
    total_circulation,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Three grid levels per Reynolds number. Re=3200 uses closer refinements to
# avoid using a severely under-resolved 65² mesh to infer an asymptotic order.
CASES: dict[str, dict] = {
    "Re400_N65": dict(Re=400, N=65, T=150.0, frames=False),
    "Re400_N129": dict(Re=400, N=129, T=100.0, save_every=250, frames=True),
    "Re400_N257": dict(Re=400, N=257, T=100.0, save_every=None, frames=False),
    "Re1000_N65": dict(Re=1000, N=65, T=400.0, frames=False),
    "Re1000_N129": dict(Re=1000, N=129, T=400.0, frames=False),
    "Re1000_N257": dict(Re=1000, N=257, T=400.0, frames=False),
    "Re3200_N129": dict(Re=3200, N=129, T=1000.0, save_every=10000, frames=True),
    "Re3200_N193": dict(Re=3200, N=193, T=1000.0, frames=False),
    "Re3200_N257": dict(Re=3200, N=257, T=1000.0, frames=False),
}
QUICK = ["Re400_N129", "Re3200_N129"]


def kernel_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1] / "ns2d"
    digest = hashlib.sha256()
    for filename in ("grid.py", "operators.py", "boundary.py", "problems.py", "solver.py"):
        digest.update(filename.encode())
        digest.update((root / filename).read_bytes())
    return digest.hexdigest()


def run_case(
    name: str, spec: dict, steady_tol: float = 1e-6, force: bool = False,
    fft_workers: int = 1,
) -> dict:
    path = os.path.join(DATA, f"cavity_{name}.npz")
    meta_path = os.path.join(DATA, f"cavity_{name}.json")
    signature = {
        "spec": spec,
        "steady_tol": steady_tol,
        "kernel_sha256": kernel_fingerprint(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_condition": "rest",
        "fft_workers": fft_workers,
    }
    if os.path.exists(path) and os.path.exists(meta_path) and not force:
        with open(meta_path) as fh:
            existing = json.load(fh)
        if existing.get("run_signature") == signature:
            print(f"[skip] {name}: matching configuration and solver source")
            return existing
        raise SystemExit(
            f"{name}: legacy or stale cache. Use --force to recompute, "
            "or scripts/build_results.py to audit existing fields."
        )

    Re, N, T = spec["Re"], spec["N"], spec["T"]
    grid = Grid.square(N)
    problem = Cavity(Re)
    dt = spec.get("dt") or suggest_dt(grid, Re)
    n_steps = int(np.ceil(T / dt))
    print(f"[run ] {name}: Re={Re} N={N} dt={dt:.4g} -> {n_steps} steps to T={T}")

    solver = Solver(grid, problem, dt=dt)
    started = datetime.now(timezone.utc).isoformat()
    milestone_fields = {}
    milestones = []
    pending_times = iter((100.0, 150.0, 300.0))
    next_time = next(pending_times)

    def capture(s):
        nonlocal next_time
        if next_time is not None and s.t + 1e-8 >= next_time:
            label = f"t{int(next_time)}"
            milestone_fields[f"{label}_psi"] = s.psi.copy()
            milestone_fields[f"{label}_omega"] = s.omega.copy()
            milestones.append({"label": label, "requested_time": next_time, "actual_time": s.t})
            next_time = next(pending_times, None)

    t0 = time.perf_counter()
    with set_workers(fft_workers):
        result = solver.run(
            T=T, steady_tol=steady_tol, save_every=spec.get("save_every"),
            monitor_every=50, progress=True, record=spec.get("frames", False),
            callback=capture,
        )
    wall = time.perf_counter() - t0

    u, v = solver.velocity()
    pv = primary_vortex(solver.psi, grid)
    ref = GHIA_VORTEX.get(Re)
    cmp_ = compare_to_ghia(u, v, grid, Re) if Re in GHIA_VORTEX else None
    cv = corner_vortices(solver.psi, grid)

    meta = {
        "name": name,
        "Re": Re,
        "N": N,
        "h": grid.h,
        "dt": dt,
        "T": T,
        "steps": result.steps,
        "t_final": result.t_final,
        "final_residual": result.final_residual,
        "converged": bool(result.converged),
        "wall_time_s": wall,
        "run_signature": signature,
        "provenance": "fresh integration from rest with recorded numerical-source and runner hashes",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "steady_tol": steady_tol,
        "milestones": milestones,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "ns2d": __version__,
        },
        "max_divergence": max_divergence(u, v, grid),
        "poisson_residual": poisson_residual(solver.psi, solver.omega, grid),
        # total_circulation reads the stored state, whose boundary ring lags the
        # interior by one step; circulation_identity removes the lag and is the
        # exact discrete Green identity (report, "Exact discrete identities")
        "circulation": total_circulation(solver.omega, grid),
        "circulation_identity": circulation_identity(
            solver.psi, grid, problem.walls(grid, solver.t)
        ),
        "circulation_exact": expected_circulation(grid, problem.walls(grid, 0.0)),
        "psi_min": pv.psi,
        "vortex_x": pv.x,
        "vortex_y": pv.y,
        "psi_min_ref": ref.psi if ref else None,
        "vortex_x_ref": ref.x if ref else None,
        "vortex_y_ref": ref.y if ref else None,
        # all tabulated stations; v_rms_unflagged repeats v without the
        # stations of benchmark.FLAGGED_V_POINTS (see that module)
        "u_rms": cmp_.u_rms if cmp_ else None,
        "v_rms": cmp_.v_rms if cmp_ else None,
        "u_rms_unflagged": cmp_.u_rms_unflagged if cmp_ else None,
        "flagged_u_y": cmp_.flagged_u_y if cmp_ else [],
        "flagged_v_x": cmp_.flagged_v_x if cmp_ else [],
        "v_rms_unflagged": cmp_.v_rms_unflagged if cmp_ else None,
        "corner_vortices": {k: [w.psi, w.x, w.y] for k, w in cv.items()},
    }

    payload = dict(
        psi=solver.psi,
        omega=solver.omega,
        residuals=result.residuals,
        residual_times=result.residual_times,
        kinetic_energy=result.kinetic_energy,
        **milestone_fields,
    )
    if result.history is not None:
        # float32 keeps the animation caches small; they are only for plotting
        t_h, psi_h, omg_h = result.history.stack()
        payload.update(
            hist_t=t_h,
            hist_psi=psi_h.astype(np.float32),
            hist_omega=omg_h.astype(np.float32),
        )
    np.savez_compressed(path + ".pending.npz", **payload)
    os.replace(path + ".pending.npz", path)
    meta["cache_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    meta["field_sha256"] = hashlib.sha256(solver.psi.tobytes() + solver.omega.tobytes()).hexdigest()
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    err = abs(pv.psi - ref.psi) / abs(ref.psi) * 100 if ref else float("nan")
    print(
        f"       psi_min={pv.psi:.6f} (Ghia {ref.psi if ref else 'n/a'}), "
        f"rel. err {err:.2f}%, R={result.final_residual:.2e}, {wall:.0f} s"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse grids only")
    ap.add_argument("--case", action="append", help="run only these cases")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--tol", type=float, default=1e-6, help="steady-state tolerance")
    ap.add_argument("--fft-workers", type=int, default=1, help="SciPy FFT worker threads")
    args = ap.parse_args()
    if args.fft_workers < 1:
        ap.error("--fft-workers must be positive")

    os.makedirs(DATA, exist_ok=True)
    names = args.case or (QUICK if args.quick else list(CASES))
    for name in names:
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}; choose from {list(CASES)}")
        meta = run_case(name, CASES[name], steady_tol=args.tol, force=args.force,
                        fft_workers=args.fft_workers)
        if not meta["converged"]:
            raise SystemExit(f"{name}: time cap reached before the steady criterion; results are not ready for the steady study")


if __name__ == "__main__":
    main()
