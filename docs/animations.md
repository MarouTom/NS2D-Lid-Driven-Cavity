# Reading and reproducing the animations

The README shows two views of the computed flow. Both are drawn with [Manim](https://www.manim.community/) from audited production caches. No velocity field is invented for an animation, and Manim never integrates the flow equations: it only draws arrays that `scripts/build_animations.py` has prepared and checked.

| File | What it shows | Source data | Time axis |
|---|---|---|---|
| `figures/cavity_particles.mp4` (poster `.png`) | Passive tracers in the converged steady fields at Re = 400, 1000, 3200 | `Re{400,1000,3200}_N257` caches | Tracer time τ ∈ [0, 16] |
| `figures/cavity_startup.mp4` (poster `.png`) | Speed, streamlines, vorticity and kinetic energy as the Re = 400 flow starts from rest | `Re400_N129` cache (recorded history) | Simulation time t ∈ [0, 20] |

## Fluid motion in a steady flow

`figures/cavity_particles.mp4` compares Re = 400, 1000 and 3200 on the converged 257² grids. Each panel contains 400 passive tracers seeded on the same jittered 20 × 20 lattice (seed 2026). Their paths solve

`dX/dτ = u_h(X), dY/dτ = v_h(X)`

with classical fourth-order Runge-Kutta integration. The velocity is obtained as `(∂ψ/∂y, −∂ψ/∂x)` from a bicubic interpolant of the saved streamfunction. Using a single differentiable streamfunction preserves zero divergence in this interpolated velocity and makes its level sets the steady tracer trajectories. The field is held fixed. These paths show how fluid moves after the circulation has settled; they do not show further evolution of the velocity field or the onset of turbulence.

The tracer step is 0.00125 in units of L/U. Frames are separated by 0.1, covering tracer times 0 to 16 at 20 frames per second. All three panels share this clock and a speed scale from 0 to the lid speed. Fading trails retain the previous 1.2 time units; trail and head colours are binned into 12 speed classes so that thousands of segments can be drawn as a few hundred Manim objects. The background shows the nodal speed (bicubic display interpolation, 45 % opacity) and faint streamfunction contours. The tracer clock starts from zero independently of each production simulation's final time. Tracers fade in over the first six frames and out over the last six, then reset if the video loops; the short title sequence before them is static decoration.

The builder repeats the particle integration with half the tracer step, records the largest position change and requires it to remain below one tenth of a grid spacing. It also rejects any particle that leaves the cavity. These checks assess the drawing of the paths; the accuracy of the underlying fluid solution is assessed separately in [the accuracy study](accuracy.md).

## Starting the fluid from rest

`figures/cavity_startup.mp4` uses the first 20 time units of the recorded Re = 400, 129² integration. The left panel shows speed and instantaneous streamlines; the right shows signed vorticity. The upper wall starts moving at t = 0. Negative vorticity corresponds to clockwise local rotation under the project's sign convention.

The production snapshots are spaced by 0.625 time units in this interval. Linear interpolation between saved streamfunction and vorticity fields supplies display frames every 0.125 units. This interpolation only smooths playback; it is not a more finely resolved numerical simulation. Velocity is reconstructed from the displayed streamfunction with the same wall values as the solver. The streamfunction contours show the direction of the instantaneous velocity; during the transient they are not particle trajectories.

The lower strip plots the kinetic energy `E = ½∫|u|² dA` recorded by the solver every 0.125 time units. It is **not** interpolated. The dashed line marks the energy of the final steady state, reached at t ≈ 61.4, beyond the end of the animation. By t = 20 the energy is within about 1.3 % of that value.

The speed colour scale is fixed at [0, 1]. Vorticity uses a fixed [−20, 20] colour range, marked by end triangles, so the singular lid corners do not hide the interior structure. Only the colour mapping saturates; the cached field is unchanged.

## Colour and accessibility choices

- **Speed** uses a sequential map whose CIE L\* lightness rises monotonically from about 10 to 97, so faster flow always reads as brighter, including in greyscale.
- **Vorticity** uses a diverging map with a dark centre: zero rotation merges into the panel and both signs gain lightness symmetrically (L\* ≈ 95 and 92 at the ends). The sign is carried by hue (blue for clockwise, orange for anticlockwise), and the contours provide a second cue.
- Colour is never the only encoding: streamlines, labelled colour bars and the energy curve carry the same information in a different form.
- Every README image has alt text, and all plotted quantities are available as numbers in `data/`.

## Video quality

Manim renders each frame with Cairo at **twice** the delivered resolution (2880 × 1440 and 2560 × 2048). Each frame is then downsampled with a Lanczos filter, which anti-aliases thin trails and small markers. The frames are encoded as H.264 MP4 (libx264, CRF 22, `tune=animation`, yuv420p, fast-start for streaming) at 20 frames per second.

A GIF was tried first. Thousands of thin, partly transparent trails change a large share of the pixels in every frame, which defeats GIF's 256-colour palette and LZW compression. A faithful 1440-pixel GIF of the tracer scene was about 47 MB; the full-colour MP4 is a small fraction of that.

The builder decodes each delivered MP4 and checks its frame count and size. It records the PSNR of every decoded frame against the rendered frame (mean and minimum) in `data/animations.json`, together with the file size and hashes. A still **poster** frame (`figures/*.png`, full-colour PNG) is written next to each video as a fallback for viewers that cannot play video.

### Embedding in the GitHub README

GitHub plays inline video only for files uploaded through its web editor, which produces a `https://github.com/user-attachments/assets/…` URL. A `<video>` tag that points at an MP4 in the repository does not play. The README embeds both videos this way: each MP4 was dragged into the GitHub web editor, which uploaded it and inserted its `user-attachments` URL on a line of its own. After rebuilding a video, upload it again the same way and replace the old URL. The poster images remain available for viewers that cannot play video, such as PyPI or local Markdown previews.

## Rebuild

Manim needs Cairo and Pango. Where `pip install manim` cannot find prebuilt `pycairo` wheels (for example, Python 3.14 on Intel macOS), the conda-forge package includes the native libraries:

```bash
conda create -n ns2d-manim -c conda-forge python=3.12 manim numpy scipy matplotlib pillow contourpy sympy
conda activate ns2d-manim
python -m pip install -e . --no-deps
```

Alternatively, with system Cairo/Pango available: `python -m pip install -e ".[animation]"`.

After generating the production caches:

```bash
python scripts/build_animations.py
# Or rebuild one view:
python scripts/build_animations.py --only particles
python scripts/build_animations.py --only startup
# Quick layout check: short 1x render into tmp/animation_review/ only
python scripts/build_animations.py --only startup --preview
```

The script checks the source and cache hashes, final-state convergence, and Poisson consistency through the existing results auditor. It writes both MP4s and their poster frames to `figures/`, review frames to the ignored `tmp/animation_review/`, and provenance to `data/animations.json`. The record includes input-cache hashes, a combined hash of the builder and scene sources, the Manim version, playback timing and tracer-step sensitivity. Scene code lives in `scripts/manim_scenes.py`.
