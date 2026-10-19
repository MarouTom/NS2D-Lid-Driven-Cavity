#!/usr/bin/env python3
"""Build README videos from audited production fields; never run a new CFD case.

The particle movie integrates passive tracers in frozen, converged velocities.
The startup movie interpolates recorded fields for display only. Neither changes
the production fields or supplies additional numerical accuracy evidence.

Drawing is done by Manim (``scripts/manim_scenes.py``, Cairo renderer). Frames
are rendered at twice the delivered resolution and downsampled with a Lanczos
filter, then encoded as H.264 MP4 (yuv420p, web fast-start). A full-colour MP4
is a fraction of the size of a GIF of the same frames, because thousands of thin
moving trails defeat GIF's palette and LZW compression. Each video is decoded
again; the frame count, size and PSNR against the rendered frames are recorded.
A still poster frame is written next to each video for the README.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline

from build_results import ROOT, FIGS, audit_case, pp
from ns2d.boundary import WallVelocities

SUPERSAMPLE = 2       # Manim renders at this multiple of the delivered size
FPS = 20
CRF = 22              # x264 constant-rate factor: lower is higher quality
QA = ROOT / "tmp/animation_review"
PREVIEW = False       # --preview: short, 1x renders kept in tmp/ only


def render(scene_cls, data, width, height, workdir):
    """Render a Manim scene and store each downsampled frame as a PNG in ``workdir``."""
    from manim import tempconfig
    from manim_scenes import BG

    scale = 1 if PREVIEW else SUPERSAMPLE
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    options = {
        "pixel_width": width * scale, "pixel_height": height * scale,
        "frame_height": 8.0, "frame_width": 8.0 * width / height, "frame_rate": FPS,
        "background_color": BG, "write_to_movie": False, "save_last_frame": False,
        "disable_caching": True, "progress_bar": "none", "verbosity": "WARNING",
        "media_dir": str(workdir / "media"),
    }
    count = 0
    with tempconfig(options):
        scene_cls.data = data
        scene = scene_cls()

        def keep(frame, num_frames=1):
            nonlocal count
            im = Image.fromarray(np.asarray(frame)[..., :3])
            im = im.resize((width, height), Image.Resampling.LANCZOS)
            for _ in range(num_frames):
                im.save(workdir / f"{count:05d}.png", compress_level=1)
                count += 1

        scene.renderer.file_writer.write_frame = keep
        scene.render()
    return sorted(workdir.glob("*.png"))


def save_video(paths, stem, poster_index):
    """Encode frames as H.264 MP4, write a poster PNG, and check the decoded video."""
    import av

    first = Image.open(paths[0])
    width, height = first.size
    path = FIGS / f"{stem}.mp4"
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as box:
        stream = box.add_stream("libx264", rate=Fraction(FPS))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": str(CRF), "preset": "slow", "tune": "animation"}
        for p in paths:
            frame = av.VideoFrame.from_ndarray(np.asarray(Image.open(p).convert("RGB")), format="rgb24")
            for packet in stream.encode(frame):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)

    # Decode the delivery file: every frame must be present, at the right size.
    psnr = []
    with av.open(str(path)) as box:
        for k, frame in enumerate(box.decode(video=0)):
            shown = frame.to_ndarray(format="rgb24").astype(float)
            if shown.shape != (height, width, 3) or k >= len(paths):
                raise RuntimeError(f"{path.name}: unexpected decoded frame {k} {shown.shape}")
            reference = np.asarray(Image.open(paths[k]).convert("RGB"), dtype=float)
            mse = np.mean((shown - reference) ** 2)
            psnr.append(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse))
            if k in (0, len(paths) // 2, len(paths) - 1):
                Image.fromarray(shown.astype(np.uint8)).save(QA / f"{stem}-decoded-{k:03}.png")
    if len(psnr) != len(paths):
        raise RuntimeError(f"{path.name}: decoded {len(psnr)} of {len(paths)} frames")

    # Poster: one rendered frame in full colour (a palette would band the colour bars).
    poster = FIGS / f"{stem}.png"
    Image.open(paths[poster_index]).convert("RGB").save(poster, optimize=True)

    size = path.stat().st_size
    print(f"Wrote {path.name}: {len(paths)} frames, {width}×{height}, {size / 1e6:.2f} MB, "
          f"PSNR mean {np.mean(psnr):.1f} dB / min {np.min(psnr):.1f} dB; "
          f"poster {poster.stat().st_size / 1e3:.0f} kB", flush=True)
    return {"file": str(path.relative_to(ROOT)), "poster": str(poster.relative_to(ROOT)),
            "poster_frame": int(poster_index), "frames": len(paths), "fps": FPS,
            "duration_s": round(len(paths) / FPS, 2), "size_bytes": size,
            "width": width, "height": height, "renderer": f"manim {version('manim')} (Cairo)",
            "supersampling": SUPERSAMPLE,
            "codec": f"H.264 (libx264, crf {CRF}, preset slow, tune animation, yuv420p)",
            "psnr_db_vs_rendered_frames": {"mean": round(float(np.mean(psnr)), 2),
                                           "min": round(float(np.min(psnr)), 2)},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "poster_sha256": hashlib.sha256(poster.read_bytes()).hexdigest()}


def trajectories(case, seed, frames, frame_dt, substeps):
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
    from manim_scenes import SPEED, TracerScene, rgba
    from matplotlib.colors import Normalize

    frame_dt, tracer_dt, n_frames, trail_frames, fade_frames = .1, .00125, 161, 12, 6
    if PREVIEW:
        n_frames = 31
    substeps = round(frame_dt / tracer_dt)
    rng = np.random.default_rng(2026)
    xx, yy = np.meshgrid(np.linspace(.035, .965, 20), np.linspace(.035, .965, 20))
    seed = np.column_stack((xx.ravel(), yy.ravel())) + rng.uniform(-.012, .012, (400, 2))
    panels, records = [], []
    for re in (400, 1000, 3200):
        row, case = audit_case(f"Re{re}_N257")
        path, speed = trajectories(case, seed, n_frames, frame_dt, substeps)
        # Measure integration sensitivity on exactly the particles shown.
        refined, _ = trajectories(case, seed, n_frames, frame_dt, 2 * substeps)
        error = float(np.max(np.linalg.norm(path - refined, axis=-1)))
        if error > case["g"].h / 10:
            raise RuntimeError(f"Re {re}: tracer-step sensitivity too large: {error}")
        records.append({"case": row["name"], "cache_sha256": row["cache_sha256"],
                        "max_position_change_when_tracer_step_halved": error})
        g = case["g"]
        panels.append({
            "Re": re, "path": path, "speed": speed, "x": g.x, "y": g.y, "psi": case["p"],
            "levels": np.linspace(float(case["p"].min()) * .95, -.001, 11),
            "background": rgba(np.hypot(case["u"], case["v"]), SPEED, Normalize(0, 1), alpha=.45),
        })
    data = {"panels": panels, "fps": FPS, "frames": n_frames, "frame_dt": frame_dt,
            "trail_frames": trail_frames, "trail_length": trail_frames * frame_dt,
            "fade_frames": fade_frames, "particles": len(seed)}
    paths = render(TracerScene, data, 1440, 720, QA / "frames_particles")
    if PREVIEW:
        return None
    # Poster: tracers fully faded in, trails long, about 60 % through the loop.
    result = save_video(paths, "cavity_particles", poster_index=int(.6 * len(paths)))
    return {**result, "meaning": "RK4 passive tracers in frozen steady fields; velocity is the rotated gradient of a bicubic streamfunction interpolant",
            "seed": 2026, "particles_per_panel": len(seed), "frame_dt": frame_dt,
            "tracer_dt": tracer_dt, "tracer_time_interval": [0, (n_frames - 1) * frame_dt],
            "trail_length": trail_frames * frame_dt, "loop_fade_frames": fade_frames,
            "inputs": records}


def startup_movie():
    from manim_scenes import StartupScene

    row, case = audit_case("Re400_N129")
    fields, g = case["fields"], case["g"]
    times = fields["hist_t"]
    t_end, n_frames = 20.0, 161
    if PREVIEW:
        t_end, n_frames = 5.0, 21
    display = np.linspace(0, t_end, n_frames)
    speeds, omegas, psis = [], [], []
    for t in display:
        right = min(int(np.searchsorted(times, t, side="right")), len(times) - 1)
        left = right - 1
        weight = (t - times[left]) / (times[right] - times[left])
        psi = (1 - weight) * fields["hist_psi"][left] + weight * fields["hist_psi"][right]
        omega = (1 - weight) * fields["hist_omega"][left] + weight * fields["hist_omega"][right]
        u, v = pp.velocity_from_psi(psi, g, WallVelocities(top=1.0))
        speeds.append(np.hypot(u, v))
        omegas.append(omega)
        psis.append(psi)
    t_e, energy = fields["residual_times"], fields["kinetic_energy"]
    keep = t_e <= t_end
    data = {
        "fps": FPS, "frames": n_frames, "times": display, "t_end": t_end,
        "x": g.x, "y": g.y, "speed": speeds, "omega": omegas, "psi": psis,
        "levels": np.array([-.11, -.10, -.08, -.06, -.04, -.02, -.01, -.003]),
        "energy_t": np.concatenate(([0.0], t_e[keep])),
        "energy": np.concatenate(([0.0], energy[keep])),
        "energy_final": float(energy[-1]), "t_steady": float(t_e[-1]),
    }
    paths = render(StartupScene, data, 1280, 1024, QA / "frames_startup")
    if PREVIEW:
        return None
    # Poster: the last frame, with the whole energy curve drawn up to t = 20.
    result = save_video(paths, "cavity_startup", poster_index=len(paths) - 1)
    return {**result, "meaning": "Transient fields, linearly interpolated in time for display only; energy curve uses recorded solver diagnostics",
            "case": row["name"], "cache_sha256": row["cache_sha256"],
            "simulation_time_interval": [0, t_end], "snapshot_spacing": float(times[1] - times[0]),
            "display_time_step": t_end / (n_frames - 1), "vorticity_colour_limits": [-20, 20]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=("particles", "startup"))
    ap.add_argument("--preview", action="store_true",
                    help="render short 1x sequences to tmp/animation_review only")
    args = ap.parse_args()
    global PREVIEW
    PREVIEW = args.preview
    QA.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/animations.json"
    evidence = json.loads(path.read_text()) if path.exists() else {}
    builders = Path(__file__).read_bytes() + (Path(__file__).parent / "manim_scenes.py").read_bytes()
    for name, builder in (("particles", particle_movie), ("startup", startup_movie)):
        if args.only is None or args.only == name:
            record = builder()
            if record is None:
                continue
            evidence[name] = record
            evidence[name]["builder_sha256"] = hashlib.sha256(builders).hexdigest()
            path.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
