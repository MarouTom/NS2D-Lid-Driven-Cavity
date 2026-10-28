#!/usr/bin/env python3
"""Draw the README hero from the audited Re = 1000, 257² production field.

Everything inside the cavity comes from the committed steady streamfunction:

* the pixel mosaic is the speed |u| / U averaged onto a coarse grid, coloured
  with the same sequential map as the README animations;
* the thin curves are contours of ψ at the Ghia, Ghia & Shin (1982) levels;
* the bright pixel trails are passive tracers integrated backwards through the
  frozen velocity field, so their length is proportional to the local speed.

The pixel streak along the top encodes the sliding lid and is the only purely
decorative element. Paths are simplified to 0.25 SVG units, below one pixel.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_hex
from scipy.interpolate import RectBivariateSpline

ROOT = Path(__file__).resolve().parents[1]
CASE = "cavity_Re1000_N257"
OUT = ROOT / "assets/readme/hero.svg"

PRIMARY = [-0.1175, -0.115, -0.11, -0.10, -0.09, -0.07, -0.05, -0.03, -0.01,
           -1e-4, -1e-5, -1e-7]
SECONDARY = [1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3]

# cavity square in SVG units, drawn as CELLS x CELLS pixels
X0, Y0, CELLS, PITCH = 748, 84, 28, 12
SIDE = CELLS * PITCH

# palette shared with scripts/manim_scenes.py
BG, TEXT, MUTED, RULE, ACCENT = "#08131c", "#eef5f7", "#9fb6c3", "#4a5f6c", "#6fe0c4"
SPEED = LinearSegmentedColormap.from_list("cavity_speed", [
    "#0b1d2a", "#123a4f", "#13606e", "#1b8a86", "#3fb59a", "#86d9a6", "#d6f0b4", "#fff6d5"])
TRACER = LinearSegmentedColormap.from_list("cavity_tracer", [
    "#2f8f9d", "#4fc3b0", "#8fe3b5", "#e3f5b8", "#fff1c9"])
SERIF = "Georgia, 'Times New Roman', serif"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def simplify(pts, tol):
    """Ramer-Douglas-Peucker on an (n, 2) polyline."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    norm = np.hypot(*ab)
    if norm == 0:
        d = np.hypot(*(pts - a).T)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / norm
    k = int(np.argmax(d))
    if d[k] <= tol:
        return np.array([a, b])
    left = simplify(pts[: k + 1], tol)
    return np.vstack([left[:-1], simplify(pts[k:], tol)])


def sx(x):
    return X0 + x * SIDE


def sy(y):
    return Y0 + (1.0 - y) * SIDE


def to_svg(seg):
    xy = simplify(np.column_stack([sx(seg[:, 0]), sy(seg[:, 1])]), 0.25)
    head = f"M{xy[0, 0]:.1f} {xy[0, 1]:.1f}"
    return head + "L" + " ".join(f"{x:.1f} {y:.1f}" for x, y in xy[1:])


def contour_paths(psi, levels):
    s = np.linspace(0.0, 1.0, psi.shape[0])
    fig, ax = plt.subplots()
    cs = ax.contour(s, s, psi.T, levels=sorted(levels))
    paths = [to_svg(seg) for segs in cs.allsegs for seg in segs if len(seg) > 2]
    plt.close(fig)
    return paths


def velocity(psi):
    """Centred-difference u = ψ_y, v = -ψ_x as splines on the unit square."""
    s = np.linspace(0.0, 1.0, psi.shape[0])
    dpdx, dpdy = np.gradient(psi, s, s)
    return RectBivariateSpline(s, s, dpdy), RectBivariateSpline(s, s, -dpdx)


def speed_mosaic(u, v):
    """Speed averaged over each pixel from a 4 x 4 sub-sample."""
    sub = (np.arange(CELLS * 4) + 0.5) / (CELLS * 4)
    sp = np.hypot(u(sub, sub), v(sub, sub))
    sp = sp.reshape(CELLS, 4, CELLS, 4).mean(axis=(1, 3))
    norm = PowerNorm(0.8, 0.0, 1.0)
    rects = []
    for i in range(CELLS):
        for j in range(CELLS):
            colour = to_hex(SPEED(norm(sp[i, j])))
            rects.append(f'<rect x="{X0 + i * PITCH}" y="{Y0 + (CELLS - 1 - j) * PITCH}" '
                         f'width="{PITCH - 1}" height="{PITCH - 1}" fill="{colour}"/>')
    return rects


def tracer_trails(u, v, centre):
    """Backward RK2 paths from seeds on rings around the primary vortex."""
    rng = np.random.default_rng(7)
    seeds = []
    for r in (0.10, 0.18, 0.26, 0.33, 0.40, 0.45):
        for a in np.linspace(0, 2 * np.pi, 4, endpoint=False) + rng.uniform(0, 2 * np.pi):
            p = centre + r * np.array([np.cos(a), np.sin(a)])
            if np.all((p > 0.04) & (p < 0.96)):
                seeds.append(p)
    marks = []
    snap = 3.0
    steps, dt = 10, 0.07
    for p in seeds:
        pts = [p.copy()]
        for _ in range(steps - 1):
            q = pts[-1]
            k1 = np.array([u(*q)[0, 0], v(*q)[0, 0]])
            mid = np.clip(q - 0.5 * dt * k1, 0.0, 1.0)
            k2 = np.array([u(*mid)[0, 0], v(*mid)[0, 0]])
            pts.append(np.clip(q - dt * k2, 0.0, 1.0))
        for k, q in enumerate(reversed(pts)):
            f = k / (steps - 1)                       # 0 = tail, 1 = head
            size = 3.0 + 4.0 * f
            x = round(sx(q[0]) / snap) * snap - size / 2
            y = round(sy(q[1]) / snap) * snap - size / 2
            colour = to_hex(TRACER(f))
            marks.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{size:.0f}" height="{size:.0f}" '
                         f'fill="{colour}" opacity="{0.25 + 0.75 * f:.2f}"/>')
    return marks


def lid_streak():
    """Row of pixels sliding in from the left and becoming the lid."""
    cells = []
    y = Y0 - 14
    for k, x in enumerate(range(64, X0, PITCH)):
        f = k / ((X0 - 64) / PITCH)
        if f < 0.35 and k % 3:
            continue
        cells.append(f'<rect x="{x}" y="{y}" width="{PITCH - 1}" height="5" '
                     f'fill="{ACCENT}" opacity="{0.06 + 0.5 * f ** 2:.2f}"/>')
    cells += [f'<rect x="{X0 + i * PITCH}" y="{y}" width="{PITCH - 1}" height="5" fill="#fff6d5"/>'
              for i in range(CELLS)]
    return cells


def main():
    psi = np.load(ROOT / f"data/{CASE}.npz")["psi"]
    rec = json.loads((ROOT / f"data/{CASE}.json").read_text())
    u, v = velocity(psi)
    centre = np.array([rec["vortex_x"], rec["vortex_y"]])
    gap = abs(rec["psi_min"] / rec["psi_min_ref"] - 1.0) * 100

    def block(items, indent="    "):
        return "\n".join(indent + s for s in items)

    def group(paths, colour, width, opacity):
        body = "\n".join(f'      <path d="{d}"/>' for d in paths)
        return (f'    <g fill="none" stroke="{colour}" stroke-width="{width}" opacity="{opacity}" '
                f'stroke-linejoin="round" stroke-linecap="round">\n{body}\n    </g>')

    primary = contour_paths(psi, PRIMARY)
    eddies = contour_paths(psi, SECONDARY)
    mosaic = speed_mosaic(u, v)
    trails = tracer_trails(u, v, centre)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="480" viewBox="0 0 1200 480" role="img" aria-labelledby="title desc">
  <title id="title">Lid-Driven Cavity: a verified 2-D Navier-Stokes solver</title>
  <desc id="desc">Project title beside the computed flow in the lid-driven cavity at Reynolds number 1000 on a 257 by 257 grid, drawn as a pixel mosaic of fluid speed with streamlines and tracer trails: one large clockwise vortex and two small counter-rotating corner eddies. Generated by scripts/build_readme_art.py from data/{CASE}.npz.</desc>
  <rect width="1200" height="480" rx="24" fill="{BG}"/>

  <g id="lid">
{block(lid_streak())}
    <text x="{X0 + SIDE}" y="{Y0 - 30}" text-anchor="end" font-family="{MONO}" font-size="18" fill="{TEXT}">lid  u = 1  →</text>
  </g>

  <g id="title-block">
    <text x="64" y="128" font-family="{MONO}" font-size="18" letter-spacing="2.5" fill="{ACCENT}">CFD BENCHMARK · FINITE DIFFERENCES · PYTHON</text>
    <text x="60" y="210" font-family="{SERIF}" font-size="72" fill="{TEXT}">Lid-Driven Cavity</text>
    <text x="64" y="260" font-family="{SANS}" font-size="25" fill="{MUTED}">2-D incompressible Navier-Stokes, solved from rest</text>
    <text x="64" y="294" font-family="{SANS}" font-size="25" fill="{MUTED}">and validated against published benchmarks.</text>
  </g>

  <g id="evidence" font-family="{SANS}">
    <line x1="64" y1="340" x2="660" y2="340" stroke="{RULE}" stroke-width="1.5"/>
    <text x="64" y="394" font-family="{SERIF}" font-size="40" fill="{TEXT}">2.04</text>
    <text x="64" y="424" font-size="18" fill="{MUTED}">spatial order</text>
    <text x="64" y="446" font-size="18" fill="{MUTED}">manufactured solution</text>
    <text x="276" y="394" font-family="{SERIF}" font-size="40" fill="{TEXT}">{gap:.2f} %</text>
    <text x="276" y="424" font-size="18" fill="{MUTED}">from Ghia et al.</text>
    <text x="276" y="446" font-size="18" fill="{MUTED}">ψ minimum, Re = 1000</text>
    <text x="488" y="394" font-family="{SERIF}" font-size="40" fill="{TEXT}">10<tspan dy="-17" font-size="24">−14</tspan></text>
    <text x="488" y="424" font-size="18" fill="{MUTED}">discrete divergence</text>
    <text x="488" y="446" font-size="18" fill="{MUTED}">every production field</text>
  </g>

  <g id="cavity">
    <g id="speed-mosaic">
{block(mosaic, "      ")}
    </g>
{group(primary, "#e6f3f3", 1.0, 0.45)}
{group(eddies, "#f09a5e", 1.0, 0.8)}
    <g id="tracers">
{block(trails, "      ")}
    </g>
    <rect x="{X0 - 0.5}" y="{Y0 - 0.5}" width="{SIDE}" height="{SIDE}" fill="none" stroke="{RULE}" stroke-width="1"/>
    <text x="{X0}" y="{Y0 + SIDE + 32}" font-family="{MONO}" font-size="17" fill="{MUTED}">computed speed + ψ · Re = 1000 · 257²</text>
  </g>
</svg>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KiB, "
          f"{len(mosaic)} pixels, {len(primary)} + {len(eddies)} contours, "
          f"{len(trails)} tracer marks)")


if __name__ == "__main__":
    main()
