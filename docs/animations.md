# Reading and reproducing the animations

The README shows two different views of the computed flow. Both are generated from audited production caches. No velocity field is invented for an animation.

## Fluid motion in a steady flow

`figures/cavity_particles.gif` compares Re = 400, 1000 and 3200 on the converged 257² grids. Each panel contains 256 passive tracers seeded at the same positions. Their paths solve

`dX/dτ = u_h(X), dY/dτ = v_h(X)`

with classical fourth-order Runge–Kutta integration. The velocity is obtained as `(∂ψ/∂y, −∂ψ/∂x)` from a bicubic interpolant of the saved streamfunction. Using a single differentiable streamfunction preserves zero divergence in this interpolated velocity and makes its level sets the steady tracer trajectories. The field is held fixed. These paths show how fluid moves after the circulation has settled; they do not show further evolution of the velocity field or the onset of turbulence.

The tracer step is 0.00125 in units of L/U. Frames are separated by 0.2, covering tracer times 0 to 16. All three panels share this clock and a speed scale from 0 to the lid speed. Fading trails retain the previous 1.2 time units. The background gives the nodal speed distribution and faint streamfunction contours. The tracer clock starts from zero independently of each production simulation's final time. Each GIF loop resets the initial particle positions.

The builder repeats the particle integration with half the tracer step, records the largest position change and requires it to remain below one tenth of a grid spacing. It also rejects any particle that leaves the cavity. These checks assess the drawing of the paths; the accuracy of the underlying fluid solution is assessed separately in [the accuracy study](accuracy.md).

## Starting the fluid from rest

`figures/cavity_startup.gif` uses the first 20 time units of the recorded Re = 400, 129² integration. The left panel shows speed and instantaneous streamlines; the right shows signed vorticity. The upper wall starts moving at t = 0. Negative vorticity corresponds to clockwise local rotation under the project's sign convention.

The production snapshots are spaced by 0.625 time units in this interval. Linear interpolation between saved streamfunction and vorticity fields supplies display frames every 0.15625 units. This interpolation only smooths playback; it is not a more finely resolved numerical simulation. Velocity is reconstructed from the displayed streamfunction with the same wall values as the solver. The streamfunction contours show the direction of the instantaneous velocity; during the transient they are not particle trajectories.

The speed colour scale is fixed at [0, 1]. Vorticity uses a fixed [-20, 20] colour range so the singular lid corners do not hide the interior structure. Only the colour mapping saturates; the cached field is unchanged. The GIF shows startup, ending before the steady stopping time of approximately 61.4.

## Rebuild

After generating the production caches:

```bash
python scripts/build_animations.py
# Or rebuild one view:
python scripts/build_animations.py --only particles
python scripts/build_animations.py --only startup
```

The script checks the source and cache hashes, final-state convergence, and Poisson consistency through the existing results auditor. It writes both GIFs to `figures/`, review frames to ignored `tmp/animation_review/`, and provenance to `data/animations.json`. The record includes input-cache hashes, the animation-builder hash, playback timing and tracer-step sensitivity. GIFs use a fixed 128-colour palette across frames to reduce colour flicker, at 960 pixels wide to limit download size. Pillow is the only encoder required.
