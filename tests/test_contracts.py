"""Regression tests for public API, restart compatibility, and source data."""

import numpy as np
import pytest
from ns2d import Grid, Cavity, Solver, suggest_dt
from ns2d.benchmark import (
    GHIA_U,
    GHIA_V,
    BP_VORTEX,
    BP_U_LEFT,
    BP_V_LEFT,
    BP_X_LEFT,
    compare_to_ghia,
    compare_to_botella,
)
from ns2d.operators import laplacian
from ns2d.verification import _error_set


@pytest.mark.parametrize(
    "args", [(2, 9), (5.5, 9), (True, 9), (9, 9, 0, 1), (9, 9, 1, np.nan)]
)
def test_invalid_grids(args):
    with pytest.raises(ValueError):
        Grid(*args)


@pytest.mark.parametrize("dt", [0, -1, np.inf, np.nan])
def test_invalid_timesteps(dt):
    with pytest.raises(ValueError):
        Solver(Grid.square(9), Cavity(100), dt=dt)


@pytest.mark.parametrize("re", [0, -1, np.inf, np.nan])
def test_invalid_reynolds(re):
    with pytest.raises(ValueError):
        Cavity(re)
    with pytest.raises(ValueError):
        suggest_dt(Grid.square(9), re)


@pytest.mark.parametrize(
    "kw",
    [
        {"monitor_every": 0},
        {"save_every": 0},
        {"n_steps": -1},
        {"n_steps": 2.5},
        {"steady_tol": -1},
        {"T": -1},
    ],
)
def test_invalid_run_controls(kw):
    s = Solver(Grid.square(9), Cavity(100))
    if "n_steps" not in kw and "T" not in kw:
        kw["n_steps"] = 1
    with pytest.raises(ValueError):
        s.run(**kw, progress=False)


def test_time_target_and_zero_step_run():
    s = Solver(Grid.square(9), Cavity(100), dt=0.003)
    assert s.run(n_steps=0, progress=False).steps == 0
    s.run(T=0.01, progress=False)
    assert s.t == pytest.approx(0.012)
    with pytest.raises(ValueError):
        s.run(T=0.1, n_steps=2)


def test_automatic_timestep_uses_lid_speed():
    g = Grid.square(33)
    assert Solver(g, Cavity(400, 4)).dt == suggest_dt(g, 400, u_max=4)
    assert np.isfinite(Solver(g, Cavity(400, 0)).dt)


def test_initial_fields_are_derived_and_validated():
    g = Grid.square(13)
    s = Solver(g, Cavity(100))
    p = np.sin(np.pi * g.X) * np.sin(np.pi * g.Y)
    s.set_initial_condition(psi=p)
    assert np.max(np.abs(laplacian(s.psi, g) + s.omega[1:-1, 1:-1])) < 1e-12
    w = s.omega.copy()
    s.set_initial_condition(omega=w)
    assert np.max(np.abs(s.psi - p)) < 1e-12
    for bad in (np.zeros((2, 2)), np.full(g.shape, np.nan), np.ones(g.shape)):
        with pytest.raises(ValueError):
            s.set_initial_condition(psi=bad)


@pytest.mark.parametrize("changed", ["dt", "domain", "lid", "bc", "corner"])
def test_checkpoint_rejects_incompatible_configuration(tmp_path, changed):
    g = Grid.square(13)
    a = Solver(g, Cavity(100), dt=0.001)
    a.run(n_steps=3, progress=False)
    f = tmp_path / "state.npz"
    a.save(f)
    b = Solver(
        Grid.square(13, 2) if changed == "domain" else g,
        Cavity(100, 2 if changed == "lid" else 1),
        dt=0.002 if changed == "dt" else 0.001,
        omega_bc="exact" if changed == "bc" else "thom",
        corner="zero" if changed == "corner" else "sum",
    )
    before = b.psi.copy()
    with pytest.raises(ValueError):
        b.load(f)
    assert np.array_equal(b.psi, before)
    b.load(f, strict=False)
    assert np.array_equal(b.psi, a.psi)


def test_legacy_checkpoints_require_explicit_state_import(tmp_path):
    g = Grid.square(9)
    f = tmp_path / "old.npz"
    np.savez(f, psi=g.zeros(), omega=g.zeros(), nx=9, ny=9, Re=100, dt=0.001, t=0, n=0)
    s = Solver(g, Cavity(100), dt=0.001)
    with pytest.raises(ValueError, match="legacy"):
        s.load(f)
    s.load(f, strict=False)


def test_vorticity_norms_ignore_inert_corners():
    g = Grid.square(9)
    s = Solver(g, Cavity(100), dt=0.001)
    s.omega[0, 0] = 1e8
    err = _error_set(s, g.zeros(), g.zeros(), 9, 0.001, 0)
    assert err.omega_l2 == err.omega_inf == 0


def test_published_data_and_all_station_metrics():
    # Visually checked against Ghia Tables I and II, pp.398-399.
    assert GHIA_U[3200][9] == -0.86636
    assert GHIA_U[10000][8] == 0.03111
    assert GHIA_V[1000][5] == -0.51550
    g = Grid.square(17)
    c = compare_to_ghia(g.zeros(), g.zeros(), g, 3200)
    assert c.flagged_u_y == [0.4531]
    assert c.u_rms == pytest.approx(np.sqrt(np.mean(GHIA_U[3200] ** 2)))
    assert c.u_rms > c.u_rms_unflagged


def test_botella_reflection_preserves_vertical_velocity():
    g = Grid.square(17)
    c = compare_to_botella(g.zeros(), g.zeros(), g)
    assert c.u_ref[-1] == 1  # right-moving lid
    assert BP_VORTEX.x == pytest.approx(0.5308)
    assert BP_VORTEX.psi < 0
    assert c.u_ref[0] == 0
    assert np.array_equal(c.u_ref, -BP_U_LEFT[::-1])
    assert np.allclose(c.x, 1 - BP_X_LEFT[::-1])
    assert np.array_equal(c.v_ref, BP_V_LEFT[::-1])
