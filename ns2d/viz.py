"""Plotting and animation helpers (matplotlib only, no external encoders).

Animations are produced with ``FuncAnimation`` and embedded in the notebook
through ``to_jshtml()``, which needs no ffmpeg; ``save_gif`` uses Pillow.

A note on colour scales for the vorticity: the two upper corners are singular
points of the continuous problem, so ``omega`` there is unbounded and grows
without limit under mesh refinement.  Auto-scaling to ``min``/``max`` would
therefore render every plot as a uniform background with two hot pixels.  All
vorticity plots here clip the colour scale to a symmetric percentile range.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

from .grid import Grid

__all__ = [
    "symmetric_limits",
    "plot_field",
    "plot_streamlines",
    "plot_solution",
    "animate_fields",
    "save_gif",
    "plot_centerlines",
    "plot_convergence",
    "plot_residual_history",
]


def symmetric_limits(field: np.ndarray, pct: float = 99.0) -> float:
    """Symmetric colour limit that ignores the singular corner values."""
    return float(np.percentile(np.abs(field), pct)) or 1.0


def plot_field(
    field: np.ndarray,
    grid: Grid,
    ax=None,
    levels: int | np.ndarray = 30,
    cmap: str = "RdBu_r",
    symmetric: bool = False,
    pct: float = 99.0,
    title: str | None = None,
    colorbar: bool = True,
    contour_lines: bool = False,
):
    """Filled contour plot of a field on the cavity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    if symmetric:
        lim = symmetric_limits(field, pct)
        lv = np.linspace(-lim, lim, levels if isinstance(levels, int) else len(levels))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cs = ax.contourf(
            grid.X.T,
            grid.Y.T,
            np.clip(field, -lim, lim).T,
            levels=lv,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
    else:
        cs = ax.contourf(grid.X.T, grid.Y.T, field.T, levels=levels, cmap=cmap)
    if contour_lines:
        ax.contour(
            grid.X.T,
            grid.Y.T,
            field.T,
            levels=cs.levels,
            colors="k",
            linewidths=0.3,
            alpha=0.5,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if title:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    return ax, cs


def plot_streamlines(
    psi,
    grid: Grid,
    ax=None,
    density: float = 1.4,
    title: str | None = None,
    walls_top: float = 1.0,
):
    """Streamlines of the flow, drawn from the velocity field."""
    from .boundary import WallVelocities
    from .postproc import velocity_from_psi

    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.8))
    u, v = velocity_from_psi(psi, grid, WallVelocities(top=walls_top))
    speed = np.hypot(u, v)
    ax.streamplot(
        grid.x,
        grid.y,
        u.T,
        v.T,
        density=density,
        color=speed.T,
        cmap="viridis",
        linewidth=0.8,
        arrowsize=0.7,
    )
    ax.set_xlim(0, grid.lx)
    ax.set_ylim(0, grid.ly)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if title:
        ax.set_title(title)
    return ax


def plot_solution(
    psi,
    omega,
    grid: Grid,
    suptitle: str | None = None,
    psi_levels: int = 30,
    figsize=(13.5, 4.2),
):
    """Three-panel summary: streamfunction, streamlines, vorticity."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_field(
        psi,
        grid,
        ax=axes[0],
        levels=psi_levels,
        cmap="viridis",
        title=r"streamfunction $\psi$",
        contour_lines=True,
    )
    plot_streamlines(psi, grid, ax=axes[1], title="streamlines")
    plot_field(
        omega,
        grid,
        ax=axes[2],
        symmetric=True,
        levels=31,
        title=r"vorticity $\omega$ (colour scale clipped)",
    )
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
def animate_fields(
    history,
    grid: Grid,
    kind: str = "both",
    interval: int = 120,
    levels: int = 26,
    pct: float = 99.0,
    title: str = "",
):
    """Animate the recorded snapshots.

    ``kind`` is ``"psi"``, ``"omega"`` or ``"both"``.  Colour scales are fixed
    over the whole animation so frames are comparable.
    """
    times = np.asarray(history.times)
    psi_all = np.asarray(history.psi)
    omg_all = np.asarray(history.omega)

    panels = {"psi": 1, "omega": 1, "both": 2}[kind]
    fig, axes = plt.subplots(1, panels, figsize=(5.6 * panels, 4.8), squeeze=False)
    axes = axes[0]

    psi_lim = float(np.max(np.abs(psi_all))) or 1.0
    psi_levels = np.linspace(-psi_lim, psi_lim, levels)
    omg_lim = float(np.percentile(np.abs(omg_all), pct)) or 1.0
    omg_levels = np.linspace(-omg_lim, omg_lim, levels)

    specs = []
    if kind in ("psi", "both"):
        specs.append(("psi", psi_all, psi_levels, "viridis", r"$\psi$"))
    if kind in ("omega", "both"):
        specs.append(("omega", omg_all, omg_levels, "RdBu_r", r"$\omega$ (clipped)"))

    for ax, (_, data, lv, cmap, name) in zip(axes, specs):
        cs = ax.contourf(
            grid.X.T,
            grid.Y.T,
            np.clip(data[0], lv[0], lv[-1]).T,
            levels=lv,
            cmap=cmap,
            extend="both",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(name)
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    sup = fig.suptitle("")

    def update(frame):
        for ax, (_, data, lv, cmap, name) in zip(axes, specs):
            for coll in list(ax.collections):
                coll.remove()
            ax.contourf(
                grid.X.T,
                grid.Y.T,
                np.clip(data[frame], lv[0], lv[-1]).T,
                levels=lv,
                cmap=cmap,
                extend="both",
            )
            ax.set_title(name)
        sup.set_text(f"{title}   $t = {times[frame]:.3f}$")
        return axes

    fig.tight_layout()
    anim = animation.FuncAnimation(
        fig, update, frames=len(times), interval=interval, blit=False
    )
    plt.close(fig)
    return anim


def save_gif(anim, path: str, fps: int = 8) -> str:
    """Write an animation to an animated GIF using Pillow (no ffmpeg needed)."""
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    return path


# ---------------------------------------------------------------------------
def plot_centerlines(comparison, axes=None, label: str = "present"):
    """Computed centerline profiles against the Ghia et al. stations."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1, ax2 = axes
    ax1.plot(comparison.u_num, comparison.y, "-", lw=1.8, label=label)
    ax1.plot(
        comparison.u_ref,
        comparison.y,
        "ko",
        ms=5,
        mfc="none",
        label=comparison.reference,
    )
    for ys in comparison.flagged_u_y:
        k = int(np.argmin(np.abs(comparison.y - ys)))
        ax1.plot(
            comparison.u_ref[k],
            comparison.y[k],
            "rx",
            ms=9,
            label="flagged station (kept in RMS)",
        )
    ax1.set_xlabel("$u$")
    ax1.set_ylabel("$y$")
    ax1.set_title(rf"$u(0.5,\,y)$, $Re={comparison.Re}$")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(comparison.x, comparison.v_num, "-", lw=1.8, label=label)
    ax2.plot(
        comparison.x,
        comparison.v_ref,
        "ko",
        ms=5,
        mfc="none",
        label=comparison.reference,
    )
    for xs in comparison.flagged_v_x:
        k = int(np.argmin(np.abs(comparison.x - xs)))
        ax2.plot(
            comparison.x[k],
            comparison.v_ref[k],
            "rx",
            ms=9,
            label="flagged station (kept in RMS)",
        )
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$v$")
    ax2.set_title(rf"$v(x,\,0.5)$, $Re={comparison.Re}$")
    ax2.grid(alpha=0.3)
    ax2.legend()
    return axes


def plot_convergence(
    studies, keys=("psi_l2", "omega_l2"), ax=None, reference_slopes=(1, 2)
):
    """Log-log error plot with reference slopes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.6))
    if not isinstance(studies, (list, tuple)):
        studies = [studies]
    hmin, hmax, emax = np.inf, -np.inf, -np.inf
    for st in studies:
        s = np.array([e.h if st.refine != "time" else e.dt for e in st.errors])
        for key in keys:
            e = np.array([getattr(err, key) for err in st.errors])
            ax.loglog(s, e, "o-", label=f"{st.label.split('(')[0].strip()} : {key}")
            emax = max(emax, e.max())
        hmin, hmax = min(hmin, s.min()), max(hmax, s.max())
    ref = np.array([hmin, hmax])
    for p in reference_slopes:
        ax.loglog(ref, emax * (ref / ref[1]) ** p, "k--", lw=0.9, alpha=0.6)
        ax.annotate(
            f"slope {p}", (ref[0], emax * (ref[0] / ref[1]) ** p), fontsize=8, alpha=0.7
        )
    ax.set_xlabel("$h$  (or $\\Delta t$)")
    ax.set_ylabel("error")
    ax.grid(which="both", alpha=0.25)
    ax.legend(fontsize=8)
    return ax


def plot_residual_history(results: dict, ax=None, tol: float | None = None):
    """Steady-state residual ``||omega^{n+1}-omega^n||_inf / dt`` versus time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, res in results.items():
        ax.semilogy(res.residual_times, res.residuals, lw=1.5, label=label)
    if tol:
        ax.axhline(tol, color="k", ls="--", lw=0.9, label=f"tolerance {tol:g}")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\|\omega^{n+1}-\omega^{n}\|_\infty/\Delta t$")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    return ax
