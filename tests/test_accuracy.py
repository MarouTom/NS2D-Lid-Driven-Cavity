"""Check the mathematical assumptions behind the reported accuracy estimates."""
import numpy as np
import pytest

from ns2d import Grid, Cavity, Solver
from ns2d.accuracy import richardson_fit, steady_defect
from ns2d.operators import laplacian


@pytest.mark.parametrize("h", [[1/64, 1/128, 1/256], [1/128, 1/192, 1/256]])
@pytest.mark.parametrize("p", [1.0, 1.8, 2.0, 3.0])
def test_richardson_recovers_known_limit_on_equal_and_unequal_ratios(h, p):
    h = np.asarray(h)
    limit = -0.12
    q = limit + 1.7 * h**p
    fit = richardson_fit(h, q)
    assert fit["status"] == "conditional_fit"
    assert fit["order"] == pytest.approx(p, abs=1e-7)
    assert fit["extrapolated"] == pytest.approx(limit, abs=1e-12)
    assert fit["gci_absolute"] == pytest.approx(1.25 * abs(q[-1] - limit))


@pytest.mark.parametrize("values,status", [
    ([1.0, 1.1, 1.05], "nonmonotone"),
    ([1.0, 1.0, 1.0], "differences_at_roundoff"),
    ([1.0, 2.0, 4.0], "no_positive_order_in_0.01_to_10"),
])
def test_richardson_does_not_invent_an_order(values, status):
    fit = richardson_fit([0.25, 0.125, 0.0625], values)
    assert fit["status"] == status
    assert fit["order"] is None
    assert fit["gci_percent"] is None


def test_steady_defect_matches_the_unsmoothed_time_step_equation():
    g = Grid.square(17)
    s = Solver(g, Cavity(400), dt=0.002)
    s.run(n_steps=25, progress=False, record=False)
    w = s.boundary_omega(s.psi, s.t)
    w[1:-1, 1:-1] = s.omega[1:-1, 1:-1]
    defect = s.convection(s.psi, w) - laplacian(w, g) / s.Re
    assert steady_defect(s) == pytest.approx(np.max(abs(defect)))
    old = s.omega.copy()
    s.step()
    rate = g.zeros()
    rate[1:-1, 1:-1] = (s.omega[1:-1, 1:-1] - old[1:-1, 1:-1]) / s.dt
    # Dirichlet diffusion is implicit, so the update rate is a filtered PDE
    # defect. Check the actual equation rather than equating the two norms.
    unfiltered = rate[1:-1, 1:-1] - s.dt / s.Re * laplacian(rate, g)
    assert np.max(abs(unfiltered + defect)) < 1e-10
