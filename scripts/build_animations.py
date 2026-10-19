#!/usr/bin/env python3
"""Build README GIFs from audited production fields; never run a new CFD case.

The particle movie integrates passive tracers in frozen, converged velocities.
The startup movie interpolates recorded fields for display only. Neither changes
the production fields or supplies additional numerical accuracy evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

from build_results import ROOT, FIGS, audit_case, plt, pp
from ns2d.boundary import WallVelocities
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

BG, PANEL, TEXT, MUTED = "#09151f", "#102330", "#eaf2f5", "#9db4c2"
SPEED = LinearSegmentedColormap.from_list(
    "cavity_speed", ["#122c3d", "#197f87", "#54cebb", "#f4dfa2"]
)
TRACER = LinearSegmentedColormap.from_list(
    "cavity_tracer", ["#419fa7", "#6ce3c6", "#ffe2a2"]
)
VORTICITY = LinearSegmentedColormap.from_list(
    "cavity_vorticity", ["#63d8ec", "#102330", "#ffad71"]
)
QA = ROOT / "tmp/animation_review"


def theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": TEXT, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#526572", "figure.facecolor": BG,
        "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "mathtext.fontset": "dejavusans",
    })


def cavity_axes(ax, title):
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="x", ylabel="y")
    ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, .5, 1], ["0", "0.5", "1"])
    ax.tick_params(length=3, labelsize=9)
    ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=27)
    ax.annotate("", xy=(.97, 1.035), xytext=(.03, 1.035),
                xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "#71e2c3", "lw": 2})
    ax.text(.5, 1.055, "moving lid  /  U = 1", transform=ax.transAxes,
            ha="center", fontsize=8, color=MUTED)


def capture(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def save_movie(frames, filename, duration=100):
    """Use one palette throughout the loop to avoid per-frame colour flicker."""
    width = 960
    height = round(frames[0].height * width / frames[0].width)
    frames = [im.resize((width, height), Image.Resampling.LANCZOS) for im in frames]
    strip = Image.new("RGB", (320 * 8, 150))
    for j, k in enumerate(np.linspace(0, len(frames) - 1, 8, dtype=int)):
        small = frames[k].resize((320, 150))
        strip.paste(small, (320 * j, 0))
    palette = strip.quantize(colors=128)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    path = FIGS / filename
    indexed[0].save(path, save_all=True, append_images=indexed[1:],
                    duration=duration, loop=0, optimize=True, disposal=1)
    # Reopen the delivery file to check frame count and review actual GIF colours.
    with Image.open(path) as gif:
        if gif.n_frames != len(frames):
            raise RuntimeError(f"{filename}: unexpected frame count {gif.n_frames}")
        for k in (0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1):
            gif.seek(k)
            gif.convert("RGB").save(QA / f"{path.stem}-{k:03}.png")
    print(f"Wrote {filename}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB", flush=True)
    return {"file": str(path.relative_to(ROOT)), "frames": len(frames),
            "duration_ms_per_frame": duration, "size_bytes": path.stat().st_size,
            "width": width, "height": height, "palette_colours": 128,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def trajectories(case, seed, frames=81, frame_dt=.2, substeps=160):
    g = case["g"]
    streamfunction = RectBivariateSpline(g.x, g.y, case["p"], kx=3, ky=3, s=0)

    def velocity(points):
        # Derive both components from one interpolant to preserve zero divergence
        # and make its level sets the steady tracer trajectories.
        x, y = points.T
        return np.column_stack((streamfunction.ev(x, y, dy=1),
                                -streamfunction.ev(x, y, dx=1)))
    x = seed.copy()
    dt = frame_dt / substeps
    positions, speeds = [], []
    for frame in range(frames):
        positions.append(x.copy())
        speeds.append(np.linalg.norm(velocity(x), axis=1))
        if frame == frames - 1:
            break
        for _ in range(substeps):
            k1 = velocity(x)
            k2 = velocity(x + dt * k1 / 2)
            k3 = velocity(x + dt * k2 / 2)
            k4 = velocity(x + dt * k3)
            x += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.isfinite(x).all() or np.any(x <= 0) or np.any(x >= 1):
                raise RuntimeError("A displayed particle left the cavity; refine tracer integration")
    return np.asarray(positions), np.asarray(speeds)


def particle_movie():
    rng = np.random.default_rng(2026)
    xx, yy = np.meshgrid(np.linspace(.035, .965, 16), np.linspace(.035, .965, 16))
    seed = np.column_stack((xx.ravel(), yy.ravel())) + rng.uniform(-.016, .016, (256, 2))
    fig = plt.figure(figsize=(13, 6.4), dpi=100)
    fig.text(.045, .957, "NAVIER-STOKES  /  LID-DRIVEN CAVITY", fontsize=10, color="#70d6c0", weight="bold")
    fig.text(.045, .894, "Fluid motion inside the cavity", fontsize=23, weight="bold")
    fig.text(.955, .90, "257 × 257 grids", fontsize=10, color=MUTED, ha="right")
    movies, heads, trails, records = [], [], [], []
    norm = Normalize(0, 1)
    for j, re in enumerate((400, 1000, 3200)):
        row, case = audit_case(f"Re{re}_N257")
        path, speed = trajectories(case, seed)
        # Measure integration sensitivity on exactly the particles shown.
        refined, _ = trajectories(case, seed, substeps=320)
        difference = np.linalg.norm(path - refined, axis=-1)
        error = float(np.max(difference))
        if error > case["g"].h / 10:
            raise RuntimeError(f"Re {re}: tracer-step sensitivity too large: {error}")
        movies.append((path, speed))
        records.append({"case": row["name"], "cache_sha256": row["cache_sha256"],
                        "max_position_change_when_tracer_step_halved": error})
        ax = fig.add_axes([.048 + j * .317, .24, .268, .54])
        cavity_axes(ax, f"Re = {re}")
        ax.imshow(np.hypot(case["u"], case["v"]).T, origin="lower", extent=(0, 1, 0, 1),
                  cmap=SPEED, norm=norm, alpha=.40, interpolation="bilinear")
        ax.contour(case["g"].x, case["g"].y, case["p"].T,
                   levels=np.linspace(float(case["p"].min()) * .95, -.001, 11),
                   colors="#bfd8df", linewidths=.35, alpha=.17)
        trail = LineCollection([], linewidths=.9)
        ax.add_collection(trail)
        heads.append(ax.scatter([], [], s=6, edgecolors="none", zorder=4))
        trails.append(trail)
    cax = fig.add_axes([.07, .105, .245, .014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=TRACER), cax=cax,
                      orientation="horizontal", ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=8, length=2)
    cb.set_label("Particle speed / lid speed", fontsize=9, labelpad=3)
    clock = fig.text(.954, .115, "", ha="right", fontsize=12)
    fig.text(.405, .080, "Passive tracers in converged, steady velocity fields", fontsize=10, color=MUTED)
    fig.text(.405, .042, "Same physical clock in all panels • particle positions reset when the GIF loops", fontsize=8, color=MUTED)
    frames = []
    for i in range(len(movies[0][0])):
        for (path, speed), head, trail in zip(movies, heads, trails):
            first = max(0, i - 6)
            points = path[first:i + 1]
            segments = np.stack((points[:-1], points[1:]), axis=-2).reshape(-1, 2, 2)
            colors = TRACER(norm(speed[first + 1:i + 1])).reshape(-1, 4)
            if i > first:
                colors[:, 3] = np.repeat(np.linspace(.08, .8, i - first), len(seed))
            trail.set_segments(segments)
            trail.set_color(colors)
            head.set_offsets(path[i])
            head.set_color(TRACER(norm(speed[i])))
        clock.set_text(f"Tracer time  τ = {i * .2:4.1f} L/U")
        frames.append(capture(fig))
    result = save_movie(frames, "cavity_particles.gif")
    plt.close(fig)
    return {**result, "meaning": "RK4 passive tracers in frozen steady fields; velocity is the rotated gradient of a bicubic streamfunction interpolant",
            "seed": 2026, "particles_per_panel": len(seed), "frame_dt": .2,
            "tracer_dt": .00125, "tracer_time_interval": [0, 16], "inputs": records}


def startup_movie():
    row, case = audit_case("Re400_N129")
    fields, g = case["fields"], case["g"]
    times = fields["hist_t"]
    fig = plt.figure(figsize=(11.2, 7.1), dpi=100)
    fig.text(.065, .945, "STARTING FROM REST  /  Re = 400", fontsize=10, color="#70d6c0", weight="bold")
    fig.text(.065, .883, "The moving lid sets the fluid in motion", fontsize=21, weight="bold")
    axes = [fig.add_axes([left, .22, .36, .565]) for left in (.075, .565)]
    images, contours = [], [None, None]
    for ax, title, cmap, lo, hi in zip(axes, ("Speed and instantaneous streamlines", "Vorticity"),
                                      (SPEED, VORTICITY), (0, -20), (1, 20)):
        cavity_axes(ax, title)
        ax.title.set_fontsize(12)
        images.append(ax.imshow(np.zeros(g.shape).T, origin="lower", extent=(0, 1, 0, 1),
                                cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear"))
    for left, im, label, ticks in zip((.12, .61), images,
            ("Speed / lid speed", "Vorticity ω  /  colour clipped at ±20"), ([0, .5, 1], [-20, 0, 20])):
        cb = fig.colorbar(im, cax=fig.add_axes([left, .13, .27, .016]), orientation="horizontal", ticks=ticks)
        cb.ax.tick_params(length=2, labelsize=9)
        cb.set_label(label, fontsize=9)
    clock = fig.text(.935, .94, "", ha="right", fontsize=12)
    fig.text(.065, .035, "129 × 129 grid • first 20 lid-travel times • linear display interpolation between recorded snapshots",
             fontsize=9, color=MUTED)
    levels = np.array([-.11, -.10, -.08, -.06, -.04, -.02, -.01, -.003])
    frames = []
    display_times = np.linspace(0, 20, 129)
    for t in display_times:
        right = min(int(np.searchsorted(times, t, side="right")), len(times) - 1)
        left = right - 1
        weight = (t - times[left]) / (times[right] - times[left])
        psi = (1 - weight) * fields["hist_psi"][left] + weight * fields["hist_psi"][right]
        omega = (1 - weight) * fields["hist_omega"][left] + weight * fields["hist_omega"][right]
        u, v = pp.velocity_from_psi(psi, g, WallVelocities(top=1.0))
        images[0].set_data(np.hypot(u, v).T)
        images[1].set_data(omega.T)
        for contour in contours:
            if contour is not None:
                contour.remove()
        available = levels[(levels > psi.min()) & (levels < psi.max())]
        contours = [ax.contour(g.x, g.y, psi.T, levels=available,
                     colors="#e6f3f3", linewidths=.65, alpha=.5)
                    if len(available) else None for ax in axes]
        clock.set_text(f"Simulation time  t = {t:5.2f} L/U")
        frames.append(capture(fig))
    result = save_movie(frames, "cavity_startup.gif", duration=80)
    plt.close(fig)
    return {**result, "meaning": "Transient fields, linearly interpolated in time for display only",
            "case": row["name"], "cache_sha256": row["cache_sha256"],
            "simulation_time_interval": [0, 20], "snapshot_spacing": .625,
            "vorticity_colour_limits": [-20, 20]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=("particles", "startup"))
    args = ap.parse_args()
    theme()
    QA.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/animations.json"
    evidence = json.loads(path.read_text()) if path.exists() else {}
    for name, builder in (("particles", particle_movie), ("startup", startup_movie)):
        if args.only is None or args.only == name:
            evidence[name] = builder()
            evidence[name]["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            path.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
