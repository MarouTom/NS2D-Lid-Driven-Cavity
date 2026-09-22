"""Test suite for ``ns2d``.

    python -m pytest tests/ -v

The tests are organised from the smallest checkable unit upwards: stencils,
then matrices, then solvers, then boundary conditions, then the full scheme's
convergence order, then the physics.  A failure high up the list localises the
bug much faster than a failing cavity run does.
"""

from __future__ import annotations

import numpy as np
import pytest

from ns2d import (
    Cavity,
    Grid,
    ManufacturedSolution,
    PSI_SINGLE_MODE,
    PSI_SMOOTH,
    PSI_WALL_DRIVEN,
    Solver,
    WallVelocities,
    suggest_dt,
)
from ns2d.operators import (
    ddx,
    ddy,
    helmholtz_matrix,
    laplacian,
    laplacian_matrix,
    make_solver,
)
from ns2d.boundary import thom_vorticity
from ns2d.postproc import (
    expected_circulation,
    l2_norm,
    max_divergence,
    poisson_residual,
    primary_vortex,
    total_circulation,
    velocity_from_psi,
)
from ns2d.verification import (
    joint_convergence,
    spatial_convergence,
    temporal_convergence,
)


# ---------------------------------------------------------------------------
# stencils
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [17, 33, 65])
def test_stencils_are_second_order(n):
    """Interior stencils must show 4x error reduction per halving of h."""
    errs = {}
    for m in (n, 2 * n - 1):
        g = Grid.square(m)
        f = np.sin(2 * np.pi * g.X) * np.cos(3 * np.pi * g.Y)
        lap_ex = -(4 + 9) * np.pi**2 * f
        dx_ex = 2 * np.pi * np.cos(2 * np.pi * g.X) * np.cos(3 * np.pi * g.Y)
        dy_ex = -3 * np.pi * np.sin(2 * np.pi * g.X) * np.sin(3 * np.pi * g.Y)
        errs[m] = (
            np.max(np.abs(laplacian(f, g) - lap_ex[1:-1, 1:-1])),
            np.max(np.abs(ddx(f, g) - dx_ex[1:-1, 1:-1])),
            np.max(np.abs(ddy(f, g) - dy_ex[1:-1, 1:-1])),
        )
    coarse, fine = errs[n], errs[2 * n - 1]
    for c, f_ in zip(coarse, fine):
        assert 3.5 < c / f_ < 4.5


def test_matrix_matches_stencil():
    g = Grid.square(21)
    rng = np.random.default_rng(0)
    phi = rng.standard_normal(g.shape)
    phi[0, :] = phi[-1, :] = phi[:, 0] = phi[:, -1] = 0.0
    A = laplacian_matrix(g)
    got = (A @ phi[1:-1, 1:-1].ravel()).reshape(g.interior_shape)
    assert np.allclose(got, laplacian(phi, g), atol=1e-10)


def test_operators_are_symmetric_positive_definite():
    g = Grid.square(13)
    for alpha, beta in [(0.0, 1.0), (1.0, 0.01)]:
        A = helmholtz_matrix(g, alpha, beta).toarray()
        assert np.allclose(A, A.T)
        assert np.min(np.linalg.eigvalsh(A)) > 0.0


# ---------------------------------------------------------------------------
# solver backends
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,beta", [(0.0, 1.0), (1.0, 2e-3)])
def test_backends_agree(alpha, beta):
    """DST, LU and PCG must return the same solution."""
    g = Grid.square(33)
    rng = np.random.default_rng(1)
    f = rng.standard_normal(g.interior_shape)
    sols = {
        b: make_solver(g, alpha, beta, backend=b).solve(f) for b in ("dst", "lu", "cg")
    }
    scale = np.max(np.abs(sols["lu"]))
    assert np.max(np.abs(sols["dst"] - sols["lu"])) < 1e-12 * scale
    assert np.max(np.abs(sols["cg"] - sols["lu"])) < 1e-9 * scale


def test_incomplete_cholesky_is_lower_triangular_with_the_right_pattern():
    from ns2d.operators import incomplete_cholesky

    g = Grid.square(17)
    A = helmholtz_matrix(g, 0.0, 1.0)
    L = incomplete_cholesky(A)
    assert L.nnz == sp_tril_nnz(A)
    assert np.all(L.diagonal() > 0)
    dense = L.toarray()
    assert np.allclose(np.triu(dense, 1), 0.0)


def sp_tril_nnz(A):
    import scipy.sparse as sp

    return sp.tril(A.tocsr()).nnz


def test_ic0_preconditioner_beats_no_preconditioner():
    """IC(0) must be SPD and must actually help; ILU is neither."""
    g = Grid.square(33)
    rng = np.random.default_rng(7)
    f = rng.standard_normal(g.interior_shape)
    ref = make_solver(g, 0.0, 1.0, backend="lu").solve(f)

    s_ic = make_solver(g, 0.0, 1.0, backend="cg", preconditioner="ic0")
    s_no = make_solver(g, 0.0, 1.0, backend="cg", preconditioner="none")
    x_ic, x_no = s_ic.solve(f), s_no.solve(f)

    assert np.max(np.abs(x_ic - ref)) < 1e-11
    assert np.max(np.abs(x_no - ref)) < 1e-11
    assert s_ic.n_iterations < 0.75 * s_no.n_iterations


def test_dst_solver_is_exact_on_an_eigenmode():
    """-Delta_h sin(pi x) sin(pi y) has a known discrete eigenvalue."""
    g = Grid.square(33)
    mode = np.sin(np.pi * g.X) * np.sin(np.pi * g.Y)
    lam = (2 - 2 * np.cos(np.pi * g.hx)) / g.hx**2 + (
        2 - 2 * np.cos(np.pi * g.hy)
    ) / g.hy**2
    solver = make_solver(g, 0.0, 1.0, backend="dst")
    got = solver.solve(lam * mode[1:-1, 1:-1])
    assert np.max(np.abs(got - mode[1:-1, 1:-1])) < 1e-12


# ---------------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------------
def test_thom_reproduces_exact_wall_vorticity():
    """Thom's formula applied to an exact psi must give O(h) wall vorticity."""
    prob = ManufacturedSolution(PSI_WALL_DRIVEN, Re=100.0)
    errs = []
    for n in (33, 65, 129):
        g = Grid.square(n)
        psi = prob.exact_psi(g, 0.0)
        w = thom_vorticity(psi, g, prob.walls(g, 0.0))
        ex = prob.exact_omega(g, 0.0)
        errs.append(
            max(
                np.max(np.abs(w[1:-1, -1] - ex[1:-1, -1])),
                np.max(np.abs(w[1:-1, 0] - ex[1:-1, 0])),
                np.max(np.abs(w[0, 1:-1] - ex[0, 1:-1])),
                np.max(np.abs(w[-1, 1:-1] - ex[-1, 1:-1])),
            )
        )
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    # Thom is first-order accurate locally: error should roughly halve
    assert all(1.7 < r < 2.4 for r in ratios), ratios


def test_thom_velocity_term_has_the_right_sign():
    """A lid moving in +x must generate negative vorticity underneath it.

    This is the term a sign error would silently break, and the one a
    manufactured solution with zero wall velocities cannot exercise.
    """
    g = Grid.square(33)
    psi = np.zeros(g.shape)
    w = thom_vorticity(psi, g, WallVelocities(top=1.0))
    assert np.all(w[1:-1, -1] < 0)
    assert np.allclose(w[1:-1, -1], -2.0 / g.hy)
    w2 = thom_vorticity(psi, g, WallVelocities(bottom=1.0))
    assert np.all(w2[1:-1, 0] > 0)


def test_corner_vorticity_is_inert():
    """Corner values of omega must not influence the solution at all."""
    g = Grid.square(21)
    s1 = Solver(g, Cavity(100.0), dt=1e-3, corner="zero")
    s2 = Solver(g, Cavity(100.0), dt=1e-3, corner="average")
    for _ in range(20):
        s1.step()
        s2.step()
    assert np.max(np.abs(s1.psi - s2.psi)) < 1e-14
    assert np.max(np.abs(s1.omega[1:-1, 1:-1] - s2.omega[1:-1, 1:-1])) < 1e-14


# ---------------------------------------------------------------------------
# manufactured solutions
# ---------------------------------------------------------------------------
def test_single_eigenmode_has_no_convection():
    """A single-eigenmode manufactured field silently kills the nonlinearity."""
    assert ManufacturedSolution(PSI_SINGLE_MODE, Re=100.0).has_convection is False
    assert ManufacturedSolution(PSI_WALL_DRIVEN, Re=100.0).has_convection is True
    assert ManufacturedSolution(PSI_SMOOTH, Re=100.0).has_convection is True


def test_psi_smooth_has_no_wall_velocity():
    """A field with zero normal derivative cannot test Thom's velocity term."""
    g = Grid.square(33)
    w = ManufacturedSolution(PSI_SMOOTH, Re=100.0).walls(g, 0.0)
    assert np.allclose(w.top, 0.0) and np.allclose(w.bottom, 0.0)
    w2 = ManufacturedSolution(PSI_WALL_DRIVEN, Re=100.0).walls(g, 0.0)
    assert np.max(np.abs(w2.top)) > 1.0


def test_manufactured_omega_satisfies_poisson():
    """omega_ex = -Delta psi_ex, so no Poisson source term is needed."""
    prob = ManufacturedSolution(PSI_WALL_DRIVEN, Re=100.0)
    g = Grid.square(129)
    psi, omega = prob.exact_psi(g, 0.3), prob.exact_omega(g, 0.3)
    res = np.max(np.abs(laplacian(psi, g) + omega[1:-1, 1:-1]))
    assert res < 5e-3  # discrete truncation error only


# ---------------------------------------------------------------------------
# convergence orders  (the core verification)
# ---------------------------------------------------------------------------
def test_spatial_order_two_with_exact_boundary_vorticity():
    st = spatial_convergence(
        sizes=(17, 33, 65), dt=1e-5, T=0.005, omega_bc="exact", verbose=False
    )
    for key in ("psi_l2", "omega_l2"):
        p = st.orders[key][1:]
        assert np.all(p > 1.85), (key, p)


def test_spatial_order_two_with_thom_boundary_vorticity():
    """psi must still be second order when omega's wall values come from Thom."""
    st = spatial_convergence(
        sizes=(17, 33, 65, 129), dt=1e-5, T=0.005, omega_bc="thom", verbose=False
    )
    p = st.orders["psi_l2"][1:]
    assert np.all(p > 1.9), p
    assert st.orders["omega_l2"][-1] > 1.8


def test_temporal_order_one():
    st = temporal_convergence(
        n=65,
        dts=(3.2e-4, 1.6e-4, 8e-5),
        T=0.032,
        omega_bc="thom",
        reference="self",
        verbose=False,
    )
    p = st.orders["psi_l2"][1:]
    assert np.all((p > 0.85) & (p < 1.2)), p


def test_joint_refinement_order_two():
    st = joint_convergence(
        sizes=(17, 33, 65), dt0=4e-4, n0=17, T=0.04, omega_bc="thom", verbose=False
    )
    p = st.orders["psi_l2"][1:]
    assert np.all(p > 1.9), p


# ---------------------------------------------------------------------------
# physics / identities
# ---------------------------------------------------------------------------
def test_velocity_field_is_discretely_divergence_free():
    """Exact to round-off by construction, not merely small."""
    g = Grid.square(65)
    s = Solver(g, Cavity(400.0), dt=1e-3)
    for _ in range(30):
        s.step()
    u, v = s.velocity()
    assert max_divergence(u, v, g) < 1e-12


def test_poisson_equation_is_satisfied_after_each_step():
    g = Grid.square(65)
    s = Solver(g, Cavity(400.0), dt=1e-3)
    for _ in range(10):
        s.step()
    assert poisson_residual(s.psi, s.omega, g) < 1e-10


def test_circulation_identity_is_exact():
    """int int omega dA = closed-loop integral of u, to round-off.

    This is an *algebraic* identity of the discretisation, not an accuracy
    statement: it holds for any psi vanishing on the boundary and for arbitrary
    wall velocities.  It fails if any of the four Thom formulas, the corner
    convention or the sign of the Poisson equation is wrong.
    """
    from ns2d.postproc import circulation_identity

    rng = np.random.default_rng(11)
    for g in (Grid.square(33), Grid(41, 27), Grid(24, 60)):
        psi = rng.standard_normal(g.shape)
        psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0.0
        walls = WallVelocities(
            top=rng.standard_normal(g.nx),
            bottom=rng.standard_normal(g.nx),
            left=rng.standard_normal(g.ny),
            right=rng.standard_normal(g.ny),
        )
        got = circulation_identity(psi, g, walls)
        exact = expected_circulation(g, walls)
        assert abs(got - exact) < 1e-11 * max(1.0, abs(exact)), (g, got, exact)


def test_circulation_of_a_converged_run_matches_greens_theorem():
    """On the stored state the identity holds up to the one-step boundary lag,
    which vanishes as the flow reaches steady state."""
    g = Grid.square(65)
    s = Solver(g, Cavity(400.0), dt=suggest_dt(g, 400.0))
    r = s.run(T=100.0, steady_tol=1e-6, progress=False, record=False)
    assert r.converged
    exact = expected_circulation(g, Cavity(400.0).walls(g, 0.0))
    assert np.isclose(exact, -1.0)
    assert abs(total_circulation(s.omega, g) - exact) < 1e-6


def test_corner_convention_shifts_the_circulation_by_a_known_amount():
    """The O(h) 'circulation error' of the first write-up was this, and nothing
    else: with 'average' the corners supply half of what the identity needs."""
    from ns2d.postproc import circulation_identity, integrate
    from ns2d.boundary import thom_vorticity
    from ns2d.operators import laplacian as lap

    for n in (33, 65, 129):
        g = Grid.square(n)
        walls = Cavity(400.0).walls(g, 0.0)
        psi = np.zeros(g.shape)
        psi[1:-1, 1:-1] = np.random.default_rng(n).standard_normal(g.interior_shape)
        exact = circulation_identity(psi, g, walls)
        for rule, deficit in (("average", 0.5 * g.hx), ("zero", g.hx)):
            w = thom_vorticity(psi, g, walls, corner=rule)
            w[1:-1, 1:-1] = -lap(psi, g)
            assert abs((integrate(w, g) - exact) - deficit) < 1e-12


def test_cavity_matches_ghia_at_re400():
    """End-to-end validation against the benchmark."""
    from ns2d.benchmark import GHIA_VORTEX, compare_to_ghia

    g = Grid.square(129)
    s = Solver(g, Cavity(400.0), dt=suggest_dt(g, 400.0))
    s.run(T=100.0, steady_tol=1e-6, progress=False, record=False)
    pv = primary_vortex(s.psi, g)
    ref = GHIA_VORTEX[400]
    assert abs(pv.psi - ref.psi) / abs(ref.psi) < 0.02
    assert abs(pv.x - ref.x) < 0.01 and abs(pv.y - ref.y) < 0.01
    u, v = s.velocity()
    c = compare_to_ghia(u, v, g, 400)
    assert c.u_rms < 5e-3
    assert c.v_rms_unflagged < 5e-3


def test_flagged_reference_stations_are_not_silently_dropped():
    """Every tabulated station must enter the headline metric.

    The Re=400, x=0.9063 station of Table II disagrees with the converged
    solution by 30x the next largest error in its column.  Whatever the
    explanation, excluding it by default would be indefensible, so
    ``compare_to_ghia`` reports the full RMS and the without-it RMS side by
    side and lets the reader compare them.
    """
    from ns2d.benchmark import FLAGGED_V_POINTS, compare_to_ghia, v_profile

    g = Grid.square(129)
    s = Solver(g, Cavity(400.0), dt=suggest_dt(g, 400.0))
    s.run(T=100.0, steady_tol=1e-6, progress=False, record=False)
    u, v = s.velocity()
    c = compare_to_ghia(u, v, g, 400)

    assert c.flagged_v_x == [0.9063]
    # the headline metric keeps the station, so it must be the larger of the two
    assert c.v_rms > 10 * c.v_rms_unflagged
    # and the full metric is exactly the RMS over all 17 stations
    x_ref, v_ref = v_profile(400)
    assert c.v_rms == pytest.approx(np.sqrt(np.mean((c.v_num - v_ref) ** 2)))
    assert len(c.v_num) == len(x_ref) == 17
    # Re with no flagged station: the two metrics coincide
    c3200 = compare_to_ghia(u, v, g, 3200)
    assert c3200.flagged_v_x == []
    assert c3200.v_rms == pytest.approx(c3200.v_rms_unflagged)
    assert 400 in FLAGGED_V_POINTS


def test_the_flagged_station_disagreement_is_isolated_and_converged():
    """Our value there is as grid-converged as at its neighbours."""
    from ns2d.benchmark import compare_to_ghia

    got = {}
    for n in (65, 129):
        g = Grid.square(n)
        s = Solver(g, Cavity(400.0), dt=suggest_dt(g, 400.0))
        s.run(T=100.0, steady_tol=1e-6, progress=False, record=False)
        u, v = s.velocity()
        got[n] = compare_to_ghia(u, v, g, 400)

    c = got[129]
    k = int(np.argmin(np.abs(c.x - 0.9063)))
    err = np.abs(c.v_num - c.v_ref)
    # the disagreement at this one station dwarfs every other
    assert err[k] > 10 * np.max(np.delete(err, k))
    # yet our own value is converging there at the same rate as elsewhere
    drift = np.abs(got[129].v_num - got[65].v_num)
    assert drift[k] < 5 * np.median(drift[drift > 0])


def test_lid_drives_the_flow_clockwise():
    """A +x lid must produce a clockwise primary vortex, i.e. psi < 0.

    This fails immediately if the convection term or the Thom velocity term
    carries the wrong sign -- the failure mode Remark 2 of the report warns of.
    """
    g = Grid.square(65)
    s = Solver(g, Cavity(400.0), dt=suggest_dt(g, 400.0))
    s.run(T=20.0, progress=False, record=False)
    assert primary_vortex(s.psi, g).psi < -0.05
    assert np.max(s.psi) < 1e-2  # no large positive cell
    u, _ = s.velocity()
    assert u[g.nx // 2, -2] > 0  # fluid drags along with lid


# ---------------------------------------------------------------------------
# infrastructure
# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path):
    g = Grid.square(33)
    s = Solver(g, Cavity(400.0), dt=1e-3)
    s.run(n_steps=25, progress=False, record=False)
    path = str(tmp_path / "chk.npz")
    s.save(path)

    s2 = Solver(g, Cavity(400.0), dt=1e-3)
    s2.load(path)
    assert np.array_equal(s.psi, s2.psi) and np.array_equal(s.omega, s2.omega)
    assert s.t == s2.t and s.n == s2.n

    s.run(n_steps=15, progress=False, record=False)
    s2.run(n_steps=15, progress=False, record=False)
    assert np.max(np.abs(s.psi - s2.psi)) < 1e-15


def test_restart_equals_uninterrupted_run(tmp_path):
    """Stopping and restarting must reproduce the continuous run bit for bit."""
    g = Grid.square(33)
    ref = Solver(g, Cavity(400.0), dt=1e-3)
    ref.run(n_steps=40, progress=False, record=False)

    a = Solver(g, Cavity(400.0), dt=1e-3)
    a.run(n_steps=17, progress=False, record=False)
    path = str(tmp_path / "mid.npz")
    a.save(path)
    b = Solver(g, Cavity(400.0), dt=1e-3)
    b.load(path)
    b.run(n_steps=23, progress=False, record=False)
    assert np.max(np.abs(ref.psi - b.psi)) < 1e-15


def test_suggest_dt_respects_all_three_bounds():
    g = Grid.square(129)
    # high Re: the convective (von Neumann) bound 2 nu / U^2 binds
    assert suggest_dt(g, 3200.0) == pytest.approx(0.5 * 2.0 / 3200.0)
    # low Re on a fine grid: the lagged-boundary bound d_max h^2 / nu binds.
    # The old rule returned 0.5*h here, which is unstable -- see the test below.
    assert suggest_dt(g, 10.0) == pytest.approx(0.5 * 1.5 * 10.0 * g.h**2)
    # intermediate (2/3 < Re*h < 2): the CFL safeguard binds
    g11 = Grid.square(11)
    assert suggest_dt(g11, 10.0) == pytest.approx(0.5 * g11.h)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_the_lagged_thom_condition_imposes_a_diffusion_number_limit():
    """nu dt/h^2 must stay below ~1.5 even though the viscous term is implicit.

    The interior von Neumann analysis cannot see this: it is the explicitly
    lagged wall vorticity.  dt = 0.5*h at Re=10 on 129^2 -- what the rule
    returned before this bound was added -- is 4.3x over the limit.
    """
    g = Grid.square(129)
    Re, nu = 10.0, 0.1
    dt_old = 0.5 * g.h  # the previous suggest_dt result
    assert nu * dt_old / g.h**2 > 1.5
    s = Solver(g, Cavity(Re), dt=dt_old)
    with pytest.raises(FloatingPointError):
        s.run(T=2.0, progress=False, record=False)

    dt_new = suggest_dt(g, Re)
    assert nu * dt_new / g.h**2 < 1.5
    Solver(g, Cavity(Re), dt=dt_new).run(T=2.0, progress=False, record=False)


@pytest.mark.parametrize("n,expected", [(17, 1.58), (25, 1.55)])
def test_step_operator_spectral_radius_crosses_one_at_three_halves(n, expected):
    """Linear (Stokes) stability of the implemented step, boundary included."""
    from ns2d import stability

    d = stability.threshold(n)
    assert d == pytest.approx(expected, abs=0.02)
    assert stability.spectral_radius(n, 1.0) < 1.0
    assert stability.spectral_radius(n, 2.0) > 1.0


def test_the_limit_is_caused_by_the_boundary_condition_not_the_interior():
    """Freeze the wall vorticity and the restriction disappears entirely."""
    from ns2d import stability

    assert stability.threshold(17, wall="frozen") == float("inf")
    assert stability.spectral_radius(17, 50.0, wall="frozen") < 1.0


def test_step_operator_spectrum_depends_only_on_the_diffusion_number():
    from ns2d import stability

    a = stability.spectral_radius(17, 1.2, Re=10.0)
    b = stability.spectral_radius(17, 1.2, Re=1000.0)
    assert a == pytest.approx(b, rel=1e-10)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_blowup_is_reported_not_silent():
    g = Grid.square(33)
    s = Solver(g, Cavity(5000.0), dt=0.5)  # far beyond 2/Re
    with pytest.raises(FloatingPointError):
        s.run(n_steps=400, progress=False, record=False)


def test_l2_norm_matches_analytic_integral():
    for n in (65, 129):
        g = Grid.square(n)
        f = np.sin(np.pi * g.X) * np.sin(np.pi * g.Y)
        assert l2_norm(f, g) == pytest.approx(0.5, rel=2e-3)


def test_velocity_from_psi_imposes_no_slip():
    g = Grid.square(33)
    rng = np.random.default_rng(3)
    psi = rng.standard_normal(g.shape)
    u, v = velocity_from_psi(psi, g, WallVelocities(top=1.0))
    assert np.allclose(u[1:-1, -1], 1.0)  # lid, corners excluded
    assert np.allclose(u[:, 0], 0.0) and np.allclose(v[0, :], 0.0)


def test_lid_corners_resolve_to_the_stationary_wall():
    """The two upper corners are the singular points of the continuous problem:
    the lid prescribes u=1 there and the side wall prescribes u=0.  The code
    must make a definite choice; it gives the stationary wall priority."""
    g = Grid.square(33)
    u, _ = velocity_from_psi(np.zeros(g.shape), g, WallVelocities(top=1.0))
    assert u[0, -1] == 0.0 and u[-1, -1] == 0.0
