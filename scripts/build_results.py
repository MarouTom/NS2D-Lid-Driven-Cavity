#!/usr/bin/env python3
"""Audit cached fields and regenerate publication figures, tables and evidence.

Run production simulations first if the ignored data/*.npz caches are absent.
No simulation is silently started by this script. --skip-verification reuses the
committed MMS measurements when only regenerating the plots.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
from importlib.metadata import PackageNotFoundError, version
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter
import numpy as np
import scipy
from ns2d import Grid, Cavity, Solver, __version__
from ns2d import benchmark as bm, postproc as pp, verification as vf
from ns2d.boundary import thom_vorticity
from ns2d.problems import ManufacturedSolution, PSI_WALL_DRIVEN
from ns2d.stability import threshold
from ns2d.solver import History
from ns2d.viz import animate_fields, save_gif
from run_production import CASES as PRODUCTION_CASES, kernel_fingerprint
from run_accuracy import analysis_fingerprint

DATA, FIGS = ROOT / "data", ROOT / "figures"
TEAL, ORANGE, BLUE = "#007f83", "#cf703b", "#486ca3"
COLORS = {400: TEAL, 1000: BLUE, 3200: ORANGE}
CASES = list(PRODUCTION_CASES)


def style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.fallback": "stix",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#263d4b",
            "text.color": "#172f3e",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b8c6cc",
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 180,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def savefig(fig, name):
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def audit_case(name):
    path = DATA / f"cavity_{name}.npz"
    if not path.exists():
        raise SystemExit(
            f"Missing {path.name}. Run python scripts/run_production.py first."
        )
    meta = json.loads((DATA / f"cavity_{name}.json").read_text())
    if meta.get("run_signature", {}).get("kernel_sha256") != kernel_fingerprint():
        raise ValueError(f"{name}: missing or stale numerical-source provenance; regenerate production")
    if meta["run_signature"].get("initial_condition") != "rest" or not meta["converged"]:
        raise ValueError(f"{name}: steady results require a converged integration from rest")
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("cache_sha256"):
        raise ValueError(f"{name}: the field cache does not match its recorded hash")
    g = Grid.square(meta["N"])
    with np.load(path, allow_pickle=False) as d:
        fields = {k: d[k].copy() for k in d.files}
    p, w = fields["psi"], fields["omega"]
    if (
        p.shape != g.shape
        or w.shape != g.shape
        or not np.isfinite(p).all()
        or not np.isfinite(w).all()
    ):
        raise ValueError(f"{name}: invalid cached fields")
    if pp.poisson_residual(p, w, g) > 1e-8:
        raise ValueError(f"{name}: cached fields violate the discrete Poisson equation")
    solver = Solver(g, Cavity(meta["Re"]), dt=meta["dt"])
    solver.set_initial_condition(p, w, t=meta["t_final"])
    u, v = solver.velocity()
    cmp = bm.compare_to_ghia(u, v, g, meta["Re"])
    vortex = pp.primary_vortex(p, g)
    ref = bm.GHIA_VORTEX[meta["Re"]]
    solver.step()
    continuation = float(
        np.max(np.abs(solver.omega[1:-1, 1:-1] - w[1:-1, 1:-1])) / solver.dt
    )
    row = {
        k: meta[k]
        for k in (
            "name",
            "Re",
            "N",
            "h",
            "dt",
            "t_final",
            "final_residual",
            "converged",
        )
    }
    row.update(
        psi_min=vortex.psi,
        vortex_x=vortex.x,
        vortex_y=vortex.y,
        ghia_psi=ref.psi,
        psi_error_percent=100 * abs(vortex.psi - ref.psi) / abs(ref.psi),
        u_rms=cmp.u_rms,
        u_rms_unflagged=cmp.u_rms_unflagged,
        v_rms=cmp.v_rms,
        v_rms_unflagged=cmp.v_rms_unflagged,
        flagged_u_y=cmp.flagged_u_y,
        flagged_v_x=cmp.flagged_v_x,
        poisson_residual=pp.poisson_residual(p, w, g),
        max_divergence=pp.max_divergence(u, v, g),
        circulation_stored=pp.total_circulation(w, g),
        circulation_identity=pp.circulation_identity(
            p, g, Cavity(meta["Re"]).walls(g, 0)
        ),
        continuation_residual=continuation,
        field_sha256=hashlib.sha256(p.tobytes() + w.tobytes()).hexdigest(),
        provenance=meta["provenance"],
        run_signature=meta["run_signature"],
        cache_sha256=meta["cache_sha256"],
        started_utc=meta["started_utc"], finished_utc=meta["finished_utc"],
    )
    if meta["Re"] == 1000:
        bc = bm.compare_to_botella(u, v, g)
        row.update(
            botella_psi=bm.BP_VORTEX.psi,
            botella_psi_error_percent=100
            * abs(vortex.psi - bm.BP_VORTEX.psi)
            / abs(bm.BP_VORTEX.psi),
            botella_u_rms=bc.u_rms,
            botella_v_rms=bc.v_rms,
        )
    return row, dict(g=g, p=p, w=w, u=u, v=v, cmp=cmp, fields=fields)


# Spatial MMS horizon. At Re = 100 the diffusion length sqrt(T/Re) is about 0.07,
# several cells on every grid, so wall-generated vorticity errors reach the
# interior. Over a much shorter horizon the psi error is almost entirely the
# Poisson discretisation error of the exact initial vorticity.
MMS_T, MMS_DT = 0.5, 5e-6


def verification():
    studies = {
        "spatial_exact": vf.spatial_convergence(
            sizes=(17, 33, 65, 129), dt=MMS_DT, T=MMS_T, omega_bc="exact", verbose=False
        ),
        "spatial_thom": vf.spatial_convergence(
            sizes=(17, 33, 65, 129), dt=MMS_DT, T=MMS_T, omega_bc="thom", verbose=False
        ),
        "temporal": vf.temporal_convergence(
            n=65, dts=(3.2e-4, 1.6e-4, 8e-5, 4e-5), T=0.032, verbose=False
        ),
        "joint": vf.joint_convergence(
            sizes=(17, 33, 65), dt0=4e-4, n0=17, T=MMS_T, verbose=False
        ),
        "asymmetric": vf.spatial_convergence(
            sizes=(17, 33, 65), dt=MMS_DT, T=MMS_T,
            psi_expr="sin(pi*x)*sin(2*pi*y)*exp(-t) + 0.3*sin(3*pi*x)*sin(pi*y)*exp(-2*t)",
            verbose=False,
        ),
    }
    data = {
        name: {
            "label": st.label,
            "refine": st.refine,
            "errors": [asdict(e) for e in st.errors],
            "orders": {
                k: [None if not np.isfinite(v) else float(v) for v in vs]
                for k, vs in st.orders.items()
            },
        }
        for name, st in studies.items()
    }
    prob = ManufacturedSolution(PSI_WALL_DRIVEN, Re=100)
    wall = []
    for n in (17, 33, 65, 129):
        g = Grid.square(n)
        dw = thom_vorticity(
            prob.exact_psi(g, 0), g, prob.walls(g, 0)
        ) - prob.exact_omega(g, 0)
        err = max(
            np.max(abs(dw[1:-1, 0])),
            np.max(abs(dw[1:-1, -1])),
            np.max(abs(dw[0, 1:-1])),
            np.max(abs(dw[-1, 1:-1])),
        )
        wall.append({"N": n, "h": g.h, "max_error": float(err)})
    data["wall"] = wall
    # Temporal contribution on the finest spatial grid: compare MMS_DT with 2*MMS_DT.
    data["spatial_time_sensitivity"] = [
        asdict(vf.mms_errors(n=129, dt=dt, T=MMS_T)) for dt in (2 * MMS_DT, MMS_DT)
    ]
    data["stokes_threshold"] = [
        {"N": n, "d_critical": threshold(n)} for n in (17, 25, 33)
    ]
    data["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "ns2d": __version__,
    }
    data["kernel_sha256"] = kernel_fingerprint()
    (DATA / "verification.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    return data


def verification_figure(ver):
    """Spatial, temporal and local-wall convergence panels (figures/verification)."""
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4), layout="constrained")
    for name, lab, color in (
        ("spatial_exact", "Exact wall vorticity", BLUE),
        ("spatial_thom", "Thom wall vorticity", TEAL),
    ):
        es = ver[name]["errors"]
        h = np.array([e["h"] for e in es])
        err = [e["psi_l2"] for e in es]
        axs[0].loglog(h, err, "o-", color=color, label=lab)
    axs[0].loglog(h, err[-1] * (h / h[-1]) ** 2, ":", color="#8b969a", label="Slope 2")
    es = ver["temporal"]["errors"]
    dt = np.array([e["dt"] for e in es])
    err = np.array([e["psi_l2"] for e in es])
    axs[1].loglog(dt, err, "o-", color=ORANGE, label=r"$\psi$, same-grid reference")
    axs[1].loglog(dt, err[-1] * dt / dt[-1], ":", color="#8b969a", label="Slope 1")
    wall = ver["wall"]
    h = np.array([e["h"] for e in wall])
    err = np.array([e["max_error"] for e in wall])
    axs[2].loglog(h, err, "o-", color=BLUE, label="Local Thom error")
    axs[2].loglog(h, err[-1] * h / h[-1], ":", color="#8b969a", label="Slope 1")
    for ax, title, xlabel, ylabel in zip(
        axs,
        ("Space: streamfunction", "Time: self-convergence", "Local wall error"),
        ("Grid spacing h", "Time step Δt", "Grid spacing h"),
        (r"$L^2$ error", r"$L^2$ difference", "Maximum error"),
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=9)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=9)
    for ax in (axs[0], axs[2]):
        ax.set_xticks(
            [1 / 128, 1 / 64, 1 / 32, 1 / 16], ["1/128", "1/64", "1/32", "1/16"]
        )
    axs[1].set_xticks([4e-5, 8e-5, 1.6e-4, 3.2e-4], ["0.4", "0.8", "1.6", "3.2"])
    axs[1].set_xlabel(r"Time step $\Delta t$ ($\times 10^{-4}$)")
    savefig(fig, "verification")


def figures(rows, cases, ver):
    # Three Reynolds numbers with one common speed scale.
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.2))
    fig.subplots_adjust(left=0.055, right=0.885, bottom=0.13, top=0.80, wspace=0.22)
    for ax, re in zip(axs, (400, 1000, 3200)):
        c = cases[f"Re{re}_N257"]
        g = c["g"]
        speed = np.hypot(c["u"], c["v"])
        ax.imshow(
            speed.T,
            extent=(0, 1, 0, 1),
            origin="lower",
            cmap="viridis",
            norm=Normalize(0, 1),
        )
        ax.streamplot(
            g.x,
            g.y,
            c["u"].T,
            c["v"].T,
            color="#f2f8f5",
            linewidth=0.52,
            density=1.55,
            arrowsize=0.65,
        )
        ax.annotate(
            "",
            xy=(0.94, 1.035),
            xytext=(0.05, 1.035),
            arrowprops={"arrowstyle": "->", "color": COLORS[re], "lw": 2.3},
            annotation_clip=False,
        )
        ax.set_title(f"Re = {re}", pad=20, color=COLORS[re])
        ax.set(xlabel="x", ylabel="y", aspect="equal", xlim=(0, 1), ylim=(0, 1))
        ax.set_xticks((0, 0.5, 1))
        ax.set_yticks((0, 0.5, 1))
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
        cax=fig.add_axes([0.916, 0.20, 0.014, 0.56]),
    )
    cb.set_label("Speed / lid speed")
    fig.suptitle(
        "THE FLOW CHANGES WITH REYNOLDS NUMBER", fontsize=16, fontweight="bold"
    )
    savefig(fig, "flow_atlas")

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.9), layout="constrained")
    for ax, re in zip(axs, (400, 1000, 3200)):
        c = cases[f"Re{re}_N257"]
        g = c["g"]
        im = ax.contourf(
            g.x,
            g.y,
            c["w"].T,
            levels=np.linspace(-20, 20, 41),
            cmap="RdBu_r",
            extend="both",
        )
        ax.contour(
            g.x,
            g.y,
            c["p"].T,
            levels=[-0.115, -0.10, -0.07, -0.04, -0.01, 0, 0.0001, 0.0005, 0.001],
            colors="#183846",
            linewidths=0.55,
        )
        ax.set(title=f"Re = {re}", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(
        im, ax=axs, shrink=0.8, label=r"Vorticity $\omega$; saturated beyond $\pm20$"
    )
    savefig(fig, "vorticity")

    fig, axs = plt.subplots(3, 2, figsize=(8.8, 10.5), layout="constrained")
    for axes, re in zip(axs, (400, 1000, 3200)):
        sizes = ((65, ":"), (129, "--"), (257, "-")) if re != 3200 else ((129, ":"), (193, "--"), (257, "-"))
        for n, ls in sizes:
            name = f"Re{re}_N{n}"
            if name not in cases:
                continue
            c = cases[name]
            g = c["g"]
            axes[0].plot(
                pp.centerline_u(c["u"], g),
                g.y,
                ls,
                color=COLORS[re],
                lw=1.5,
                label=f"Present, {n}²",
            )
            axes[1].plot(
                g.x,
                pp.centerline_v(c["v"], g),
                ls,
                color=COLORS[re],
                lw=1.5,
                label=f"Present, {n}²",
            )
        c = cases[f"Re{re}_N257"]
        comp = c["cmp"]
        axes[0].plot(
            comp.u_ref,
            comp.y,
            "o",
            mfc="white",
            mec="#263d4b",
            ms=4,
            label="Ghia (1982)",
        )
        axes[1].plot(
            comp.x,
            comp.v_ref,
            "o",
            mfc="white",
            mec="#263d4b",
            ms=4,
            label="Ghia (1982)",
        )
        for ys in comp.flagged_u_y:
            k = np.argmin(abs(comp.y - ys))
            axes[0].plot(
                comp.u_ref[k],
                ys,
                "x",
                color="#c84246",
                ms=9,
                mew=2,
                label="Flagged, retained",
            )
        for xs in comp.flagged_v_x:
            k = np.argmin(abs(comp.x - xs))
            axes[1].plot(
                xs,
                comp.v_ref[k],
                "x",
                color="#c84246",
                ms=9,
                mew=2,
                label="Flagged, retained",
            )
        if re == 1000:
            bc = bm.compare_to_botella(c["u"], c["v"], c["g"])
            axes[0].plot(
                bc.u_ref, bc.y, "+", color=ORANGE, ms=7, label="Botella & Peyret (1998)"
            )
            axes[1].plot(
                bc.x, bc.v_ref, "+", color=ORANGE, ms=7, label="Botella & Peyret (1998)"
            )
        axes[0].set(
            title=f"Re = {re}  |  vertical centerline", xlabel="u(0.5, y)", ylabel="y"
        )
        axes[1].set(
            title=f"Re = {re}  |  horizontal centerline", xlabel="x", ylabel="v(x, 0.5)"
        )
        for ax in axes:
            ax.grid(alpha=0.18)
            ax.legend(fontsize=9, loc="best")
    savefig(fig, "centerlines")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for row in rows:
        if row["N"] not in (129, 257):
            continue
        d = cases[row["name"]]["fields"]
        color = COLORS[row["Re"]]
        ls = "--" if row["N"] == 129 else "-"
        axs[0].semilogy(
            d["residual_times"],
            d["residuals"],
            ls,
            color=color,
            label=f"Re {row['Re']}, {row['N']}²",
        )
        axs[1].plot(d["residual_times"], d["kinetic_energy"], ls, color=color)
    axs[0].axhline(1e-6, color="#607783", ls=":", label="Stopping threshold")
    axs[0].set(
        xlabel="Dimensionless time",
        ylabel=r"$\|\omega^{n+1}-\omega^n\|_\infty/\Delta t$",
        title="Steady convergence",
    )
    axs[1].set(
        xlabel="Dimensionless time", ylabel="Kinetic energy", title="Kinetic energy"
    )
    axs[0].legend(fontsize=9, ncol=2)
    for ax in axs:
        ax.grid(alpha=0.15)
    savefig(fig, "residuals")

    verification_figure(ver)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.7), layout="constrained")
    for ax, (re, corner, limits) in zip(
        axs,
        (
            (400, "BR", (0.65, 1, 0, 0.35)),
            (1000, "BL", (0, 0.3, 0, 0.3)),
            (3200, "TL", (0, 0.3, 0.65, 1)),
        ),
    ):
        c = cases[f"Re{re}_N257"]
        g = c["g"]
        p = c["p"]
        ed = pp.corner_vortices(p, g)[corner]
        levels = ed.psi * np.array([0.05, 0.15, 0.35, 0.6, 0.85, 0.97])
        ax.contour(g.x, g.y, p.T, levels=levels, colors=COLORS[re], linewidths=1)
        ax.contour(g.x, g.y, p.T, levels=[0], colors="#839299", linewidths=0.6)
        ax.plot(ed.x, ed.y, "+", color="#172f3e", ms=9)
        ax.set(
            xlim=limits[:2],
            ylim=limits[2:],
            aspect="equal",
            xlabel="x",
            ylabel="y",
            title=f"Re {re} · {corner}\nψ max = {ed.psi:.2e}",
        )
    savefig(fig, "corner_eddies")

    c = cases["Re400_N129"]
    fields = c["fields"]
    history = History()
    for i in np.linspace(0, len(fields["hist_t"]) - 1, 60, dtype=int):
        history.append(fields["hist_t"][i], int(i), fields["hist_psi"][i], fields["hist_omega"][i])
    anim = animate_fields(history, c["g"], title="Re = 400 | 129² | fresh integration", pct=99.0)
    save_gif(anim, str(FIGS / "cavity_Re400.gif"), fps=8)


def export_tables(rows, cases):
    (DATA / "validation.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    keys = [
        "name",
        "Re",
        "N",
        "dt",
        "t_final",
        "converged",
        "final_residual",
        "psi_min",
        "ghia_psi",
        "psi_error_percent",
        "u_rms",
        "u_rms_unflagged",
        "v_rms",
        "v_rms_unflagged",
    ]
    with (DATA / "validation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (DATA / "centerline_comparisons.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "reference",
                "component",
                "coordinate",
                "computed",
                "reference_value",
                "flagged",
            ]
        )
        for row in rows:
            c = cases[row["name"]]
            comps = [c["cmp"]]
            if row["Re"] == 1000:
                comps.append(bm.compare_to_botella(c["u"], c["v"], c["g"]))
            for comp in comps:
                for component, coords, num, ref, flags in [
                    ("u", comp.y, comp.u_num, comp.u_ref, comp.flagged_u_y),
                    ("v", comp.x, comp.v_num, comp.v_ref, comp.flagged_v_x),
                ]:
                    for x, a, b in zip(coords, num, ref):
                        w.writerow(
                            [
                                row["name"],
                                comp.reference,
                                component,
                                x,
                                a,
                                b,
                                bool(x in flags),
                            ]
                        )
    lines = [
        "| Re | Grid | Final time | Residual | Steady? | ψ minimum | Ghia difference |",
        "|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['Re']} | {r['N']}² | {r['t_final']:.2f} | {r['final_residual']:.2e} | {'yes' if r['converged'] else 'no'} | {r['psi_min']:.6f} | {r['psi_error_percent']:.2f}% |"
        )
    (DATA / "results_table.md").write_text("\n".join(lines) + "\n")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        start, end = "<!-- results:start -->", "<!-- results:end -->"
        if start in text and end in text:
            before, rest = text.split(start, 1)
            _, after = rest.split(end, 1)
            readme.write_text(
                before + start + "\n" + "\n".join(lines) + "\n" + end + after
            )
    lines = []
    for r in rows:
        lines.append(
            f"{r['Re']} & {r['N']}$^2$ & {r['t_final']:.2f} & {r['final_residual']:.2e} & {'yes' if r['converged'] else 'no'} & {r['psi_min']:.6f} & {r['psi_error_percent']:.2f}\\\\"
        )
    (ROOT / "report/results_rows.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for r in rows:
        lines.append(
            f"{r['Re']} & {r['N']}$^2$ & {r['u_rms']:.3e} & {r['u_rms_unflagged']:.3e} & {r['v_rms']:.3e} & {r['v_rms_unflagged']:.3e}\\\\"
        )
    (ROOT / "report/profile_rows.tex").write_text("\n".join(lines) + "\n")


def accuracy_outputs(accuracy, rows, cases, fourth):
    """Generate every accuracy figure and report number from measured JSON."""
    def table(name, lines):
        (ROOT / "report" / name).write_text("\n".join(lines) + "\n")
    def fmt(value, precision=3):
        return "N/A" if value is None else f"{value:.{precision}f}"
    def sci(value):
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    table("grid_rows.tex", [
        f"{r['Re']} & {', '.join(map(str,r['N']))} & {fmt(r['scalar_fits']['psi_min']['order'])} & "
        f"{fmt(r['scalar_fits']['psi_min']['extrapolated'],7)} & {fmt(r['scalar_fits']['psi_min']['gci_percent'])}\\\\"
        for r in accuracy["grid"]
    ])
    table("fourth_grid_rows.tex", [
        f"65, 129, 257 & {fourth['original_fit']['order']:.3f} & {fourth['original_fit']['extrapolated']:.7f} & {fourth['original_fit']['gci_percent']:.3f}\\\\",
        f"129, 257, 513 & {fourth['finer_fit']['order']:.3f} & {fourth['finer_fit']['extrapolated']:.7f} & {fourth['finer_fit']['gci_percent']:.3f}\\\\",
    ])
    table("fourth_grid_summary.tex", [
        f"The original three-grid model predicts $\\psi_{{\\min}}={fourth['predicted_fourth_value']:.7f}$ on $513^2$; "
        f"the independent integration gives ${fourth['psi_min'][-1]:.7f}$, a difference of ${sci(fourth['fourth_prediction_error'])}$. "
        f"The two extrapolated limits differ by ${sci(fourth['extrapolated_limit_change'])}$. "
        "This is an independent check of the primary-streamfunction model at $\\Rey=400$ only. "
        "It does not establish the same reliability for other Reynolds numbers or observables.",
    ])
    table("time_rows.tex", [
        f"{r['Re']} & {r['dts'][0]:.3e} & {r['successive_profile_differences'][0]['u_l2']:.3e} & "
        f"{r['successive_profile_differences'][1]['u_l2']:.3e} & {r['profile_orders']['u_l2']:.3f} & {r['profile_orders']['v_l2']:.3f}\\\\"
        for r in accuracy["temporal"]
    ])
    table("settling_rows.tex", [
        f"{r['Re']} & {r['N']}$^2$ & {r['initial_steady_defect']:.2e} & "
        f"{max(t['max_update_rate'] for t in r['trials']):.3e} & "
        f"{max(abs(t['drift']['psi_min_change']) for t in r['trials']):.2e} & "
        f"{max(t['drift']['u_l2'] for t in r['trials']):.2e} & {max(t['drift']['v_l2'] for t in r['trials']):.2e}\\\\"
        for r in accuracy["steady"]
    ])
    milestone_lines = []
    for r in accuracy["steady"]:
        if r["N"] != 257:
            continue
        target = 150 if r["Re"] == 1000 else 300
        for m in r["milestone_changes"]:
            if m["requested_time"] == target:
                d = m["change_to_converged"]
                milestone_lines.append(f"{r['Re']} & {target} & {d['psi_min_change_percent']:.4f} & {d['u_l2']:.3e} & {d['v_l2']:.3e}\\\\")
    table("transient_rows.tex", milestone_lines)
    profile_lines = []
    for r in accuracy["grid"]:
        for pair, diff in zip(zip(r["N"],r["N"][1:]),r["successive_profile_differences"]):
            profile_lines.append(f"{r['Re']} & {pair[0]}$^2$ to {pair[1]}$^2$ & {diff['u_l2']:.3e} & {diff['v_l2']:.3e}\\\\")
    table("grid_profile_rows.tex", profile_lines)
    # Retain unsuccessful fits instead of silently selecting only neat orders.
    fit_lines = []
    labels = {"vortex_x": "$x_c$", "vortex_y": "$y_c$", "kinetic_energy": "$E_k$", "omega_center": "$\\omega(1/2,1/2)$", "eddy_BR_psi": "$\\psi_{BR}$"}
    for r in accuracy["grid"]:
        for key, label in labels.items():
            f = r["scalar_fits"].get(key)
            if f is None:
                continue
            order = f"{f['order']:.3f}" if f["order"] is not None else "N/A"
            gci = f"{f['gci_percent']:.3f}" if f["gci_percent"] is not None else "N/A"
            status = "conditional" if f["status"] == "conditional_fit" else "no admissible fit"
            fit_lines.append(f"{r['Re']} & {label} & {order} & {gci} & {status}\\\\")
    table("observable_rows.tex", fit_lines)
    eddy_lines=[]
    for re,corner,label in ((400,"BR","Bottom right"),(1000,"BL","Bottom left"),(3200,"TL","Top left")):
        c=cases[f"Re{re}_N257"]
        ed=pp.corner_vortices(c["p"],c["g"])[corner]
        eddy_lines.append(f"{re} & {label} & {ed.psi:.4e} & {ed.x:.5f} & {ed.y:.5f}\\\\")
    table("eddy_rows.tex",eddy_lines)
    r=next(r for r in rows if r["Re"]==1000 and r["N"]==257)
    ref=bm.GHIA_VORTEX[1000]
    table("botella_rows.tex",[
        f"$\\psi_{{\\min}}$ & {r['psi_min']:.7f} & {ref.psi:.7f} & {bm.BP_VORTEX.psi:.7f}\\\\",
        f"Vortex $x$ & {r['vortex_x']:.5f} & {ref.x:.4f} & {bm.BP_VORTEX.x:.4f}\\\\",
        f"Vortex $y$ & {r['vortex_y']:.5f} & {ref.y:.4f} & {bm.BP_VORTEX.y:.4f}\\\\",
        f"Strength discrepancy & N/A & {r['psi_error_percent']:.3f}\\% & {r['botella_psi_error_percent']:.3f}\\%\\\\",
        f"$u$ RMS discrepancy & N/A & {r['u_rms']:.3e} & {r['botella_u_rms']:.3e}\\\\",
        f"$v$ RMS discrepancy & N/A & {r['v_rms']:.3e} & {r['botella_v_rms']:.3e}\\\\",
    ])
    budget_lines=[]
    grid_to_drift=[]
    md=["| Re | Three-grid fitted order | Conditional GCI for ψ minimum | Ten-time-unit ψ drift |", "|---:|---:|---:|---:|"]
    for gridrow in accuracy["grid"]:
        re=gridrow["Re"]
        settling=next(r for r in accuracy["steady"] if r["Re"]==re and r["N"]==257)
        fit=gridrow["scalar_fits"]["psi_min"]
        drift=max(t["drift"]["psi_min_change_percent"] for t in settling["trials"])
        if fit["gci_percent"] is not None and drift>0:
            grid_to_drift.append(fit["gci_percent"]/drift)
        budget_lines.append(f"{re} & {fmt(fit['gci_percent'])} & {drift:.3e} & {abs(settling['half_dt_vs_base_dt']['psi_min_change']):.2e}\\\\")
        md.append(f"| {re} | {fmt(fit['order'])} | {fmt(fit['gci_percent'])}% | {drift:.3e}% |")
    table("accuracy_summary.tex",[
        r"\begin{center}\small",
        r"\begin{tabular}{@{}rrrr@{}}\toprule",
        r"$\Rey$ & Grid GCI (\%) & Settling drift (\%) & Step-halving $|\delta\psi_{\min}|$\\\midrule",
        *budget_lines,
        r"\bottomrule\end{tabular}\end{center}",
        "The three columns are distinct sensitivity measures on the $257^2$ grids, "
        "not independent random errors to combine in quadrature. The first assumes a "
        "power-law spatial error; the second measures ten-time-unit drift; the last "
        "compares the endpoints of the two continuations. The measured settling "
        "effects can thus be assessed against spatial sensitivity without treating "
        "the residual threshold as an uncertainty estimate.",
        (f"Across these three fine-grid cases, the conditional grid estimate is at least "
         f"${sci(min(grid_to_drift))}$ times the measured primary-vortex settling drift. "
         "Within these experiments, spatial resolution is the dominant measured limitation "
         "for primary-vortex strength. This conclusion does not transfer automatically to every flow quantity.")
        if len(grid_to_drift)==3 else "No common scale separation is asserted without three admissible grid fits.",
    ])
    md.extend(["", "GCI is a conditional power-law estimate, not a rigorous bound. Settling drift is the larger change during two independent ten-time-unit continuations with dt and dt/2; it is not an infinite-time error bound."])
    (DATA / "accuracy_summary.md").write_text("\n".join(md)+"\n")
    readme=ROOT/"README.md"
    text=readme.read_text()
    start,end="<!-- accuracy:start -->","<!-- accuracy:end -->"
    if start in text and end in text:
        before,rest=text.split(start,1)
        _,after=rest.split(end,1)
        readme.write_text(before+start+"\n"+"\n".join(md)+"\n"+end+after)
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.7),layout="constrained")
    for ax,r in zip(axs,accuracy["grid"]):
        fit=r["scalar_fits"]["psi_min"]
        h=np.asarray(r["h"])
        q=np.asarray(fit["values"])
        ax.plot(h,q,"o",color=COLORS[r["Re"]],label="Computed grids")
        if r["Re"]==400:
            ax.plot(fourth["h"][-1],fourth["psi_min"][-1],"s",mfc="white",mec=TEAL,label="Independent 513²")
        if fit["order"] is not None:
            xx=np.linspace(0,max(h),100)
            coefficient=(q[-1]-fit["extrapolated"])/h[-1]**fit["order"]
            ax.plot(xx,fit["extrapolated"]+coefficient*xx**fit["order"],"--",color=COLORS[r["Re"]],label="Conditional power fit")
            ax.plot(0,fit["extrapolated"],"+",color="#172f3e",ms=9)
        ax.axhline(bm.GHIA_VORTEX[r["Re"]].psi,color="#839299",lw=1,label="Ghia")
        if r["Re"]==1000:
            ax.axhline(bm.BP_VORTEX.psi,color=ORANGE,lw=1,label="Botella-Peyret")
        ax.set(xlabel="Grid spacing h",ylabel=r"Primary $\psi_{\min}$",title=f"Re = {r['Re']}")
        ax.ticklabel_format(axis="x",style="sci",scilimits=(0,0))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    savefig(fig,"grid_convergence")
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.7),layout="constrained")
    for ax,r in zip(axs,accuracy["temporal"]):
        dt=np.asarray(r["dts"][:2])
        for key,label,color in (("u_l2","u profile",TEAL),("v_l2","v profile",ORANGE)):
            e=[d[key] for d in r["successive_profile_differences"]]
            ax.loglog(dt,e,"o-",color=color,label=f"{label}, p={r['profile_orders'][key]:.3f}")
        ax.loglog(dt,e[-1]*dt/dt[-1],":",color="#839299",label="Slope 1")
        ax.set_xticks(dt[::-1], [f"{step:.2e}" for step in dt[::-1]])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(xlabel=r"Larger step $\Delta t$ in each pair",ylabel=r"Profile $L^2$ difference",title=f"Re = {r['Re']} | T = 5")
        ax.grid(alpha=.18,which="both")
        ax.legend(fontsize=8)
    savefig(fig,"cavity_time_convergence")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-verification", action="store_true")
    args = ap.parse_args()
    accuracy = json.loads((DATA / "accuracy.json").read_text())
    if accuracy["signature"] != {"kernel_sha256": kernel_fingerprint(), "analysis_sha256": analysis_fingerprint()}:
        raise ValueError("accuracy evidence is stale; run scripts/run_accuracy.py")
    fourth = json.loads((DATA / "fourth_grid.json").read_text())
    if fourth["analysis_sha256"] != hashlib.sha256((ROOT / "scripts/check_fourth_grid.py").read_bytes()).hexdigest():
        raise ValueError("fourth-grid evidence is stale; run scripts/check_fourth_grid.py")
    for meta in fourth["run_records"]:
        path=DATA / f"cavity_{meta['name']}.npz"
        if (meta["run_signature"]["kernel_sha256"] != kernel_fingerprint()
                or not meta["converged"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != meta["cache_sha256"]):
            raise ValueError("fourth-grid study uses stale production fields")
    FIGS.mkdir(exist_ok=True)
    style()
    rows = []
    cases = {}
    for name in CASES:
        row, c = audit_case(name)
        rows.append(row)
        cases[name] = c
        print(
            f"audited {name}: Poisson {row['poisson_residual']:.2e}, next-step residual {row['continuation_residual']:.2e}",
            flush=True,
        )
    for r in accuracy["steady"]:
        current=next(row for row in rows if row["name"]==r["name"])
        if r["signature"]["cache_sha256"] != current["cache_sha256"]:
            raise ValueError(f"{r['name']}: steady accuracy study uses an outdated field")
    ver = (
        json.loads((DATA / "verification.json").read_text())
        if args.skip_verification
        else verification()
    )
    export_tables(rows, cases)
    figures(rows, cases, ver)
    accuracy_outputs(accuracy, rows, cases, fourth)
    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "sympy", "pillow", "pytest", "nbformat", "nbclient"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    (DATA / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "ns2d": __version__, "kernel_sha256": kernel_fingerprint(),
        "note": "Measured build environment; production and accuracy records retain their own execution metadata."
    }, indent=2)+"\n")
    print("Updated figures, CSV/JSON evidence and report table inputs.")


if __name__ == "__main__":
    main()
