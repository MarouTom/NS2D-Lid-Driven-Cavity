#!/usr/bin/env python3
"""Run a small cavity example: python scripts/run_demo.py --re 400 --n 65."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))
from ns2d import Grid, Cavity, Solver
from ns2d.benchmark import compare_to_ghia, available_reynolds
from ns2d.postproc import primary_vortex
from ns2d.viz import plot_solution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re", type=int, default=400)
    ap.add_argument("--n", type=int, default=65)
    ap.add_argument("--time", type=float, default=80.0)
    ap.add_argument("--backend", choices=["dst", "lu", "cg"], default="dst")
    ap.add_argument("--output", type=Path, default=ROOT / "output/demo")
    args = ap.parse_args()
    solver = Solver(Grid.square(args.n), Cavity(args.re), backend=args.backend)
    result = solver.run(T=args.time, steady_tol=1e-6, record=False)
    print(
        f"Steady criterion met: {result.converged}; psi minimum: {primary_vortex(solver.psi,solver.grid).psi:.6f}"
    )
    if args.re in available_reynolds():
        print(compare_to_ghia(*solver.velocity(), solver.grid, args.re).summary())
    args.output.mkdir(parents=True, exist_ok=True)
    solver.save(args.output / "checkpoint.npz")
    fig, _ = plot_solution(
        solver.psi,
        solver.omega,
        solver.grid,
        suptitle=f"Re = {args.re} | {args.n}² | t = {solver.t:.2f} | residual = {result.final_residual:.2e}",
    )
    fig.savefig(args.output / "cavity.png", dpi=180, bbox_inches="tight")
    print(f"Wrote {args.output}/cavity.png and checkpoint.npz")


if __name__ == "__main__":
    main()
