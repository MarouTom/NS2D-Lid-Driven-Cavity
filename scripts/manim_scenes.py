"""Manim scenes for the README animations.

The scenes only *draw* arrays prepared and audited by ``build_animations.py``:
converged or recorded fields, RK4 tracer positions and solver diagnostics. They
never integrate the flow equations, and nothing here adds numerical evidence.

Rendering notes
---------------
* Text uses Pango (``Text``), so no LaTeX installation is needed.
* Hundreds of tracer trails are drawn as a few hundred VMobjects, grouped by
  trail age (opacity) and speed (colour), instead of one mobject per segment.
* Field images are colour-mapped here and resampled bicubically for display.
"""
from __future__ import annotations

import contourpy
import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, AnimationGroup, Axes, Circle, Create, DashedLine, Dot,
    FadeIn, GrowArrow, ImageMobject, LaggedStart, Rectangle, Scene, Square, Text,
    ValueTracker, VGroup, VMobject, Arrow, linear,
)
from manim.constants import RESAMPLING_ALGORITHMS
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgb

BG, PANEL, TEXT, MUTED, ACCENT, EDGE = (
    "#08131c", "#0e1f2b", "#eef5f7", "#9fb6c3", "#6fe0c4", "#4a5f6c")
FONT = "Avenir Next"  # Pango substitutes per glyph if this font is missing
# Sequential speed map: CIE L* rises monotonically from about 10 to 97.
SPEED = LinearSegmentedColormap.from_list("cavity_speed", [
    "#0b1d2a", "#123a4f", "#13606e", "#1b8a86", "#3fb59a", "#86d9a6", "#d6f0b4", "#fff6d5"])
TRACER = LinearSegmentedColormap.from_list("cavity_tracer", [
    "#2f8f9d", "#4fc3b0", "#8fe3b5", "#e3f5b8", "#fff1c9"])
# Diverging vorticity map with a dark centre: omega -> 0 fades into the panel and
# the two signs reach nearly symmetric lightness (L* about 95 and 92).
VORTICITY = LinearSegmentedColormap.from_list("cavity_vorticity", [
    "#d4f6ff", "#62c9e6", "#1f7fa6", "#163e57", "#0f1c26",
    "#5a2f22", "#b45a33", "#f09a5e", "#ffe2c4"])
UNIT_CIRCLE = Circle(radius=1, num_components=9).points  # 8 cubic arcs


def label(text, size=18, color=TEXT, weight="NORMAL", font=FONT):
    return Text(text, font=font, font_size=size, color=color, weight=weight)


def rgba(field, cmap, norm, alpha=1.0):
    """Colour-map a nodal field indexed [x, y] into an image with rows of falling y."""
    colours = cmap(norm(np.asarray(field).T[::-1]))[..., :3]
    if alpha < 1:
        colours = alpha * colours + (1 - alpha) * np.asarray(to_rgb(PANEL))
    out = np.empty(colours.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.round(255 * colours)
    out[..., 3] = 255
    return out


class Panel:
    """A unit-square cavity drawn with side ``size`` centred at ``center``."""

    def __init__(self, center, size):
        self.c, self.s = np.array([center[0], center[1], 0.0]), size

    def to_scene(self, xy):
        xy = np.asarray(xy, dtype=float)
        out = np.zeros(xy.shape[:-1] + (3,))
        out[..., 0] = self.c[0] + self.s * (xy[..., 0] - .5)
        out[..., 1] = self.c[1] + self.s * (xy[..., 1] - .5)
        return out

    def image(self, pixels):
        im = ImageMobject(pixels)
        im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bicubic"])
        im.stretch_to_fit_width(self.s).stretch_to_fit_height(self.s).move_to(self.c)
        return im

    def border(self):
        return Square(side_length=self.s, stroke_color=EDGE, stroke_width=1.6).move_to(self.c)

    def lid(self):
        top = self.c[1] + self.s / 2 + .16
        arrow = Arrow(self.to_scene((.03, 0))[0] * RIGHT + top * UP,
                      self.to_scene((.97, 0))[0] * RIGHT + top * UP, buff=0,
                      color=ACCENT, stroke_width=3, tip_length=.13,
                      max_tip_length_to_length_ratio=1, max_stroke_width_to_length_ratio=100)
        note = label("moving lid   U = 1", 11, MUTED).next_to(arrow, UP, buff=.05)
        return arrow, note

    def ticks(self, size=11):
        lo, hi = self.to_scene((0, 0)), self.to_scene((1, 1))
        group = VGroup(
            label("0", size, MUTED).next_to(lo, DOWN + LEFT, buff=.06),
            label("1", size, MUTED).next_to([hi[0], lo[1], 0], DOWN, buff=.08),
            label("1", size, MUTED).next_to([lo[0], hi[1], 0], LEFT, buff=.08),
            label("x", size, MUTED).next_to([self.c[0], lo[1], 0], DOWN, buff=.08),
            label("y", size, MUTED).next_to([lo[0], self.c[1], 0], LEFT, buff=.1),
        )
        return group

    def contours(self, x, y, psi, levels, width=.7, opacity=.25, color="#c6dde3"):
        generator = contourpy.contour_generator(x, y, np.asarray(psi).T)
        group = VGroup()
        for level in levels:
            for line in generator.lines(level):
                if len(line) > 2:
                    group.add(VMobject(stroke_color=color, stroke_width=width,
                                       stroke_opacity=opacity).set_points_as_corners(self.to_scene(line)))
        return group


def colourbar(cmap, center, width, height, ticks, tick_labels, caption, extend=False):
    """Gradient strip, frame, tick labels and caption centred at ``center``."""
    strip = (255 * np.repeat(cmap(np.linspace(0, 1, 512))[None], 6, axis=0)).astype(np.uint8)
    bar = ImageMobject(strip)
    bar.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    bar.stretch_to_fit_width(width).stretch_to_fit_height(height).move_to([*center, 0])
    extras = VGroup(Rectangle(width=width, height=height, stroke_color=EDGE,
                              stroke_width=1.2).move_to(bar))
    for value, text in zip(ticks, tick_labels):
        x = bar.get_left()[0] + value * width
        extras.add(label(text, 11, MUTED).move_to([x, bar.get_bottom()[1] - .15, 0]))
    extras.add(label(caption, 12, MUTED).next_to(bar, DOWN, buff=.3))
    if extend:  # triangles mark that colours saturate beyond the end values
        extras.add(label("◀", 9, to_hex(cmap(0.0))).next_to(bar, LEFT, buff=.04),
                   label("▶", 9, to_hex(cmap(1.0))).next_to(bar, RIGHT, buff=.04))
    return bar, extras


def live(*mobjects):
    """Mark mobjects changed by another updater as moving.

    Manim draws mobjects without updaters into a cached background during an
    animation, so in-place changes to them would otherwise not be shown.
    """
    for mob in mobjects:
        mob.add_updater(lambda m: None)


class TextCache:
    """Reuse rendered Text for repeated strings such as clock readings."""

    def __init__(self, **style):
        self.style, self.cache = style, {}

    def __call__(self, text):
        if text not in self.cache:
            self.cache[text] = label(text, **self.style)
        return self.cache[text].copy()


class Clock:
    """Right-aligned text readout replaced in place each frame."""

    def __init__(self, template, anchor, size=20):
        self.render = TextCache(size=size, color=TEXT, font="Menlo")  # fixed-width digits
        self.mob = self.render(template).move_to(anchor, aligned_edge=RIGHT)
        self.anchor = self.mob.get_right()
        live(self.mob)

    def show(self, text):
        self.mob.become(self.render(text).move_to(self.anchor, aligned_edge=RIGHT))


class TracerLayer(VGroup):
    """Fading trails plus glowing heads for every tracer in one panel."""

    def __init__(self, panel, path, speed, trail_frames, bins=12):
        super().__init__()
        self.panel, self.path, self.speed = panel, path, speed
        self.trail_frames, self.bins = trail_frames, bins
        self.colours = [to_hex(TRACER(c)) for c in (np.arange(bins) + .5) / bins]
        self.trails = [[VMobject(stroke_color=self.colours[b], stroke_width=1.15, fill_opacity=0)
                        for b in range(bins)] for _ in range(trail_frames)]
        self.glows = [VMobject(stroke_width=0, fill_color=self.colours[b]) for b in range(bins)]
        self.heads = [VMobject(stroke_width=0, fill_color=self.colours[b]) for b in range(bins)]
        for age in self.trails:
            self.add(*age)
        self.add(*self.glows, *self.heads)

    def _bin(self, speed):
        return np.clip((speed * self.bins).astype(int), 0, self.bins - 1)

    def show(self, frame, envelope):
        panel = self.panel
        for age in range(1, self.trail_frames + 1):
            start = frame - age
            row = self.trails[age - 1]
            if start < 0:
                for vm in row:
                    vm.clear_points()
                continue
            opacity = envelope * .85 * ((self.trail_frames + 1 - age) / self.trail_frames) ** 1.6
            a = panel.to_scene(self.path[start])
            b = panel.to_scene(self.path[start + 1])
            groups = self._bin(self.speed[start + 1])
            for colour, vm in enumerate(row):
                pick = groups == colour
                if not pick.any():
                    vm.clear_points()
                    continue
                pa, pb = a[pick], b[pick]
                pts = np.stack((pa, pa + (pb - pa) / 3, pa + 2 * (pb - pa) / 3, pb), axis=1)
                vm.set_points(pts.reshape(-1, 3))
                vm.set_stroke(opacity=opacity)
        centres = panel.to_scene(self.path[frame])
        groups = self._bin(self.speed[frame])
        for colour in range(self.bins):
            pick = groups == colour
            for vm, radius, opacity in ((self.glows[colour], .055, .17 * envelope),
                                        (self.heads[colour], .021, envelope)):
                if not pick.any():
                    vm.clear_points()
                    continue
                pts = centres[pick][:, None, :] + radius * UNIT_CIRCLE[None]
                vm.set_points(pts.reshape(-1, 3))
                vm.set_fill(opacity=opacity)


def header(kicker, title, subtitle=None, title_size=34):
    group = VGroup(label(kicker, 15, ACCENT, "BOLD"), label(title, title_size, TEXT, "BOLD"))
    if subtitle:
        group.add(label(subtitle, 13, MUTED))
    group.arrange(DOWN, aligned_edge=LEFT, buff=.12).to_corner(UP + LEFT, buff=.34)
    return group


class TracerScene(Scene):
    """Passive tracers in the three converged 257² fields (data set by the builder).

    Canvas: 16 × 8 scene units.
    """

    data: dict = {}

    def construct(self):
        d = self.data
        top = header("NAVIER–STOKES  ·  LID-DRIVEN CAVITY", "Fluid motion inside the cavity",
                     "Steady flow at three Reynolds numbers  ·  converged 257 × 257 fields")
        clock = Clock("τ = 16.0 L/U", [7.6, top[0].get_center()[1], 0], size=19)

        panels = [Panel((x, -.27), 4.0) for x in (-5.15, 0, 5.15)]
        images, borders, arrows, decor, layers = [], [], [], [], []
        for panel, rec in zip(panels, d["panels"]):
            images.append(panel.image(rec["background"]))
            borders.append(VGroup(panel.contours(rec["x"], rec["y"], rec["psi"], rec["levels"]),
                                  panel.border()))
            arrow, note = panel.lid()
            name = label(f"Re = {rec['Re']}", 22, TEXT, "BOLD").next_to(note, UP, buff=.08)
            name.align_to(panel.border(), LEFT)
            arrows.append(arrow)
            decor.append(VGroup(note, name, panel.ticks()))
            layers.append(TracerLayer(panel, rec["path"], rec["speed"], d["trail_frames"]))

        bar, bar_extras = colourbar(TRACER, (-5.15, -3.12), 3.6, .1, [0, .25, .5, .75, 1],
                                    ["0", "0.25", "0.5", "0.75", "1"], "Particle speed / lid speed")
        caption = VGroup(
            label(f"{d['particles']} passive tracers per panel, advected by RK4 through the converged velocity field",
                  13, MUTED),
            label("Same seeds and clock in every panel  ·  trails show the last "
                  f"{d['trail_length']:.1f} L/U  ·  tracers fade in and out over the loop", 11, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=.12)
        caption.move_to([-2.75, -3.2, 0], aligned_edge=LEFT)

        self.play(FadeIn(top, shift=.15 * UP), run_time=.6)
        self.play(LaggedStart(*[AnimationGroup(FadeIn(im), Create(b)) for im, b in zip(images, borders)],
                              lag_ratio=.25), run_time=1.0)
        self.play(*[GrowArrow(a) for a in arrows], *[FadeIn(g, shift=.08 * UP) for g in decor],
                  FadeIn(bar), FadeIn(bar_extras), FadeIn(caption), run_time=.5)

        frames, fade = d["frames"], d["fade_frames"]
        tracker = ValueTracker(0)
        tracers = VGroup(*layers)

        def redraw(group):
            i = int(np.clip(round(tracker.get_value()), 0, frames - 1))
            envelope = min(1.0, (i + 1) / fade, (frames - i) / fade)
            for layer in layers:
                layer.show(i, envelope)
            clock.show(f"τ = {i * d['frame_dt']:4.1f} L/U")

        redraw(tracers)
        tracers.add_updater(redraw)
        self.add(tracers, clock.mob)
        self.play(tracker.animate.set_value(frames - 1), run_time=(frames - 1) / d["fps"],
                  rate_func=linear)
        self.wait(.25)


class StartupScene(Scene):
    """Recorded Re = 400 transient: speed, vorticity and kinetic energy.

    Canvas: 10 × 8 scene units.
    """

    data: dict = {}

    def construct(self):
        d = self.data
        top = header("STARTING FROM REST  ·  Re = 400", "The moving lid sets the fluid in motion",
                     "Production run on a 129 × 129 grid  ·  fields interpolated linearly between "
                     "saved snapshots for display", title_size=28)
        clock = Clock("t = 20.00 L/U", [4.66, top[0].get_center()[1], 0], size=18)

        panels = [Panel((x, .1), 3.45) for x in (-2.3, 2.3)]
        speed_norm, vort_norm = Normalize(0, 1), Normalize(-20, 20)
        speed_img = panels[0].image(rgba(d["speed"][0], SPEED, speed_norm))
        vort_img = panels[1].image(rgba(d["omega"][0], VORTICITY, vort_norm))
        arrows, decor, borders = [], [], []
        for panel, name in zip(panels, ("Speed and instantaneous streamlines", "Vorticity ω")):
            arrow, note = panel.lid()
            heading = label(name, 17, TEXT, "BOLD").next_to(note, UP, buff=.08)
            heading.align_to(panel.border(), LEFT)
            arrows.append(arrow)
            decor.append(VGroup(note, heading, panel.ticks()))
            borders.append(panel.border())
        bars = [
            colourbar(SPEED, (panels[0].c[0], -2.2), 2.8, .085, [0, .25, .5, .75, 1],
                      ["0", "0.25", "0.5", "0.75", "1"], "Speed / lid speed"),
            colourbar(VORTICITY, (panels[1].c[0], -2.2), 2.8, .085, [0, .25, .5, .75, 1],
                      ["−20", "−10", "0", "10", "20"], "Vorticity ω   (colour saturates beyond ±20)",
                      extend=True),
        ]

        # Kinetic energy from the solver's recorded diagnostics (not interpolated).
        t_e, energy, e_final, t_end = d["energy_t"], d["energy"], d["energy_final"], d["t_end"]
        axes = Axes(x_range=[0, t_end, 5], y_range=[0, .05, .02], x_length=8.0, y_length=.95,
                    tips=False, axis_config={"color": EDGE, "stroke_width": 1.4, "tick_size": .035})
        axes.move_to([.2, -3.25, 0])
        curve = np.array([axes.c2p(t, e) for t, e in zip(t_e, energy)])
        ghost = VMobject(stroke_color="#33495a", stroke_width=2.2).set_points_as_corners(curve)
        steady = DashedLine(axes.c2p(0, e_final), axes.c2p(t_end, e_final),
                            color=MUTED, stroke_width=1.2, dash_length=.06)
        steady_note = label(f"steady value {e_final:.4f}, reached at t ≈ {d['t_steady']:.1f}", 10, MUTED)
        steady_note.next_to(axes.c2p(t_end, e_final), UP, buff=.05).align_to(axes.c2p(t_end, 0), RIGHT)
        ticks = VGroup(
            *[label(f"{t:g}", 10, MUTED).next_to(axes.c2p(t, 0), DOWN, buff=.07)
              for t in np.arange(0, t_end + 1e-9, 5)],
            *[label(f"{e:g}", 10, MUTED).next_to(axes.c2p(0, e), LEFT, buff=.07) for e in (0, .02, .04)],
            label("t  (L/U)", 10, MUTED).next_to(axes.c2p(t_end, 0), RIGHT, buff=.14),
            label("E = ½∫|u|² dA", 10, MUTED).rotate(np.pi / 2).next_to(axes, LEFT, buff=.42),
        )
        played = VMobject(stroke_color=ACCENT, stroke_width=3.2)
        cursor = Dot(axes.c2p(0, 0), radius=.05, color=ACCENT)

        self.play(FadeIn(top, shift=.15 * UP), run_time=.6)
        self.play(*[Create(b) for b in borders], FadeIn(speed_img), FadeIn(vort_img), Create(axes),
                  *[FadeIn(m) for pair in bars for m in pair], run_time=.8)
        self.play(*[GrowArrow(a) for a in arrows], *[FadeIn(g, shift=.08 * UP) for g in decor],
                  Create(ghost), FadeIn(steady), FadeIn(steady_note), FadeIn(ticks),
                  FadeIn(cursor), FadeIn(clock.mob), run_time=.7)

        frames, levels = d["frames"], d["levels"]
        tracker = ValueTracker(0)
        streamlines = VGroup()
        live(speed_img, vort_img, played, streamlines, cursor)

        def redraw(group):
            i = int(np.clip(round(tracker.get_value()), 0, frames - 1))
            t = d["times"][i]
            speed_img.pixel_array = rgba(d["speed"][i], SPEED, speed_norm)
            vort_img.pixel_array = rgba(d["omega"][i], VORTICITY, vort_norm)
            psi = d["psi"][i]
            available = levels[(levels > psi.min()) & (levels < psi.max())]
            streamlines.become(VGroup(*[p.contours(d["x"], d["y"], psi, available, width=1.0,
                                                   opacity=.55, color="#e6f3f3") for p in panels]))
            shown = t_e <= t + 1e-9
            if shown.sum() >= 2:
                played.set_points_as_corners(np.vstack((curve[shown], axes.c2p(t, np.interp(t, t_e, energy)))))
            cursor.move_to(axes.c2p(t, np.interp(t, t_e, energy)))
            clock.show(f"t = {t:5.2f} L/U")

        driver = VGroup()
        driver.add_updater(redraw)
        self.add(played, streamlines, cursor, clock.mob, driver)
        self.play(tracker.animate.set_value(frames - 1), run_time=(frames - 1) / d["fps"],
                  rate_func=linear)
        self.wait(.5)
