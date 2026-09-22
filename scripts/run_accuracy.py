#!/usr/bin/env python3
"""Measure cavity grid sensitivity, transient time order, and steady drift.

Production states must first be regenerated with scripts/run_production.py.
--temporal-only is independent of those runs. Per-case studies are cached with
source/configuration hashes; --assemble-only combines completed measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.fft import set_workers
from ns2d import Cavity, Grid, Solver, suggest_dt
from ns2d.accuracy import observables, profile_difference, richardson_fit, state_difference, steady_defect
from run_production import CASES, kernel_fingerprint

DATA = ROOT / "data"
OUT = DATA / "accuracy"
GRIDS = {400: (65, 129, 257), 1000: (65, 129, 257), 3200: (129, 193, 257)}


def analysis_fingerprint():
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "ns2d/accuracy.py", ROOT / "ns2d/postproc.py"):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def signature(**kwargs):
    return dict(kernel_sha256=kernel_fingerprint(), analysis_sha256=analysis_fingerprint(), **kwargs)


def cached(path, sig, refresh):
    if path.exists() and not refresh:
        row = json.loads(path.read_text())
        if row.get("signature") == sig:
            print(f"[cached] {path.name}", flush=True)
            return row
    return None


def save(path, row):
    row["measured_utc"] = datetime.now(timezone.utc).isoformat()
    row["environment"] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    pending.replace(path)
    return row


def load_case(name):
    meta = json.loads((DATA / f"cavity_{name}.json").read_text())
    path = DATA / f"cavity_{name}.npz"
    sig = meta.get("run_signature", {})
    if (sig.get("kernel_sha256") != kernel_fingerprint()
            or sig.get("initial_condition") != "rest" or not meta["converged"]):
        raise ValueError(f"{name}: requires a fresh, converged integration with the current numerical source")
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("cache_sha256"):
        raise ValueError(f"{name}: cache does not match its recorded hash")
    with np.load(path, allow_pickle=False) as d:
        fields = {k: d[k].copy() for k in d.files if k == "psi" or k == "omega" or k.startswith("t")}
    return meta, fields


def temporal_study(re, refresh=False):
    n, target = 129, 5.0
    grid = Grid.square(n)
    base_dt = suggest_dt(grid, re)
    # All baseline values divide T exactly for this grid and these Re.
    dts = [base_dt / k for k in (1, 2, 4)]
    sig = signature(Re=re, N=n, T=target, dts=dts, initial_condition="rest")
    path = OUT / f"temporal_Re{re}.json"
    hit = cached(path, sig, refresh)
    if hit is not None:
        return hit
    walls = Cavity(re).walls(grid, target)
    observations = []
    for dt in dts:
        steps = round(target / dt)
        if not np.isclose(steps * dt, target, atol=1e-12, rtol=0):
            raise ValueError("time-study target must be an integer number of steps")
        s = Solver(grid, Cavity(re), dt=dt)
        s.run(n_steps=steps, progress=False, record=False)
        observations.append(observables(s.psi, s.omega, grid, walls))
        print(f"[time] Re={re}, N={n}, dt={dt:g}, T={s.t:g}", flush=True)
    differences = [profile_difference(a, b) for a, b in zip(observations, observations[1:])]
    orders = {key: float(np.log2(differences[0][key] / differences[1][key]))
              for key in ("u_l2", "v_l2") if differences[1][key] > 0}
    scalar = richardson_fit(dts, [q["psi_min"] for q in observations], safety=1.0)
    return save(path, dict(signature=sig, Re=re, N=n, target_time=target, dts=dts,
                          observations=observations, successive_profile_differences=differences,
                          profile_orders=orders, psi_min_time_fit=scalar,
                          interpretation="fixed-grid transient self-convergence; excludes spatial error and is not a steady-state error estimate"))


def steady_study(name, refresh=False, duration=10.0):
    meta, fields = load_case(name)
    sig = signature(case=name, cache_sha256=meta["cache_sha256"], duration=duration,
                    dt_factors=[1.0, 0.5])
    path = OUT / f"steady_{name}.json"
    hit = cached(path, sig, refresh)
    if hit is not None:
        return hit
    g, re = Grid.square(meta["N"]), meta["Re"]
    walls = Cavity(re).walls(g, meta["t_final"])
    p, w = fields["psi"], fields["omega"]
    trials, end_states = [], []
    initial_defect = None
    for factor in (1.0, 0.5):
        dt = meta["dt"] * factor
        s = Solver(g, Cavity(re), dt=dt)
        s.set_initial_condition(p, w, t=meta["t_final"])
        if initial_defect is None:
            initial_defect = steady_defect(s)
        result = s.run(n_steps=int(np.ceil(duration / dt)), progress=False, record=False,
                       monitor_every=50)
        drift = state_difference(p, w, s.psi, s.omega, g, walls)
        trials.append(dict(dt_factor=factor, dt=dt, duration=s.t-meta["t_final"],
                           max_update_rate=float(max(result.residuals)),
                           final_update_rate=result.final_residual,
                           final_steady_defect=steady_defect(s),
                           stayed_below_threshold=bool(max(result.residuals) < meta["steady_tol"]),
                           drift=drift))
        end_states.append((s.psi.copy(), s.omega.copy()))
        print(f"[settling] {name}, dt factor {factor}, psi drift {drift['psi_min_change']:.3e}", flush=True)
    milestone_changes = []
    for row in meta.get("milestones", []):
        label = row["label"]
        milestone_changes.append(dict(**row, change_to_converged=state_difference(
            fields[f"{label}_psi"], fields[f"{label}_omega"], p, w, g, walls)))
    return save(path, dict(signature=sig, name=name, Re=re, N=g.nx,
                          initial_steady_defect=initial_defect, trials=trials,
                          half_dt_vs_base_dt=state_difference(*end_states[0], *end_states[1], g, walls),
                          milestone_changes=milestone_changes,
                          interpretation="measured finite-window sensitivity, not a rigorous remaining-transient bound"))


def assemble():
    grid_studies = []
    for re, sizes in GRIDS.items():
        observations, names, spacings, hashes = [], [], [], []
        for n in sizes:
            name = f"Re{re}_N{n}"
            meta, fields = load_case(name)
            g = Grid.square(n)
            observations.append(observables(fields["psi"], fields["omega"], g, Cavity(re).walls(g, 0)))
            names.append(name)
            spacings.append(g.h)
            hashes.append(meta["cache_sha256"])
        scalar_keys = ("psi_min", "vortex_x", "vortex_y", "kinetic_energy", "omega_center", "eddy_BR_psi")
        fits = {key: richardson_fit(spacings, [q[key] for q in observations])
                for key in scalar_keys if all(key in q for q in observations)}
        grid_studies.append(dict(Re=re, names=names, N=list(sizes), h=spacings,
                                 cache_sha256=hashes, observations=observations, scalar_fits=fits,
                                 successive_profile_differences=[profile_difference(a,b) for a,b in zip(observations, observations[1:])]))
    temporal = [json.loads((OUT / f"temporal_Re{re}.json").read_text()) for re in GRIDS]
    steady = [json.loads((OUT / f"steady_{name}.json").read_text()) for name in CASES]
    for row in temporal + steady:
        if row["signature"]["analysis_sha256"] != analysis_fingerprint() or row["signature"]["kernel_sha256"] != kernel_fingerprint():
            raise ValueError("outdated accuracy measurements; rerun the study")
        if "name" in row:
            meta = json.loads((DATA / f"cavity_{row['name']}.json").read_text())
            if row["signature"]["cache_sha256"] != meta["cache_sha256"]:
                raise ValueError(f"outdated steady drift measurement: {row['name']}")
    return save(DATA / "accuracy.json", dict(signature=signature(), grid=grid_studies,
                                             temporal=temporal, steady=steady,
                                             caveat="Three-grid Richardson/GCI estimates assume a leading power-law error. They do not establish asymptotic convergence or statistical coverage."))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--temporal-only", action="store_true")
    modes.add_argument("--steady-only", action="store_true")
    modes.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--case", action="append", choices=list(CASES))
    ap.add_argument("--fft-workers", type=int, default=1)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.fft_workers < 1:
        ap.error("--fft-workers must be positive")
    with set_workers(args.fft_workers):
        if args.assemble_only:
            assemble()
            return
        if not args.steady_only:
            for re in GRIDS:
                temporal_study(re, args.refresh)
        if not args.temporal_only:
            for name in args.case or CASES:
                steady_study(name, args.refresh)
        if not args.steady_only and not args.temporal_only:
            assemble()


if __name__ == "__main__":
    main()
