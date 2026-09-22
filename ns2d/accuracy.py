"""Observable-based accuracy diagnostics, independent of literature values.

Richardson estimates are conditional on q(h) = q0 + C h**p. A three-grid fit
is not proof of an asymptotic regime, and non-monotone sequences are rejected.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from . import postproc as pp
from .operators import laplacian


def richardson_fit(h, q, safety: float = 1.25) -> dict:
    """Fit three coarse-to-fine scalar values, allowing unequal grid ratios.

    Return an explicit status instead of forcing an order for oscillatory,
    flat, or non-convergent data. The GCI is an engineering estimate, not a
    statistical confidence interval or a rigorous bound.
    """
    h, q = np.asarray(h, dtype=float), np.asarray(q, dtype=float)
    if h.shape != (3,) or q.shape != (3,) or not np.isfinite([h, q]).all():
        raise ValueError("need three finite spacings and three finite observables")
    if np.any(h <= 0) or not np.all(np.diff(h) < 0) or safety < 1:
        raise ValueError("spacings must decrease and safety must be at least one")
    dc, df = q[0] - q[1], q[1] - q[2]
    out = dict(h=h.tolist(), values=q.tolist(), coarse_change=float(dc),
               fine_change=float(df), status="unresolved", order=None,
               extrapolated=None, estimated_fine_error=None, gci_absolute=None,
               gci_percent=None, safety_factor=safety)
    tiny = 100 * np.finfo(float).eps * max(1.0, float(np.max(abs(q))))
    if abs(dc) <= tiny or abs(df) <= tiny:
        out["status"] = "differences_at_roundoff"
        return out
    if dc * df <= 0:
        out["status"] = "nonmonotone"
        return out
    # Scale by the finest h before taking powers; avoid tiny h**p values.
    a, b = h[0] / h[2], h[1] / h[2]
    ratio = dc / df
    def equation(p):
        return np.expm1(p * np.log(a)) / np.expm1(p * np.log(b)) - 1 - ratio
    if equation(0.01) * equation(10.0) >= 0:
        out["status"] = "no_positive_order_in_0.01_to_10"
        return out
    p = brentq(equation, 0.01, 10.0)
    correction = (q[2] - q[1]) / np.expm1(p * np.log(b))
    estimate = abs(correction)
    out.update(status="conditional_fit", order=float(p),
               extrapolated=float(q[2] + correction), estimated_fine_error=float(estimate),
               gci_absolute=float(safety * estimate),
               gci_percent=float(100 * safety * estimate / abs(q[2])) if q[2] else None)
    return out


def observables(psi, omega, grid, walls) -> dict:
    u, v = pp.velocity_from_psi(psi, grid, walls)
    vortex = pp.primary_vortex(psi, grid)
    stations = np.linspace(0, 1, 65)
    result = dict(psi_min=vortex.psi, vortex_x=vortex.x, vortex_y=vortex.y,
                  kinetic_energy=pp.kinetic_energy(u, v, grid),
                  omega_center=float(omega[grid.nx // 2, grid.ny // 2]),
                  stations=stations.tolist(),
                  u_profile=np.interp(stations, grid.y, pp.centerline_u(u, grid)).tolist(),
                  v_profile=np.interp(stations, grid.x, pp.centerline_v(v, grid)).tolist())
    for name, cell in pp.corner_vortices(psi, grid).items():
        result[f"eddy_{name}_psi"] = cell.psi
    return result


def profile_difference(a: dict, b: dict) -> dict:
    x = np.asarray(a["stations"])
    if not np.array_equal(x, b["stations"]):
        raise ValueError("profiles must use the same stations")
    out = {}
    for component in ("u", "v"):
        d = np.asarray(b[f"{component}_profile"]) - a[f"{component}_profile"]
        out[f"{component}_l2"] = float(np.sqrt(trapezoid(d*d, x)))
        out[f"{component}_linf"] = float(np.max(abs(d)))
    return out


def state_difference(p0, w0, p1, w1, grid, walls) -> dict:
    a, b = (observables(p, w, grid, walls) for p, w in ((p0, w0), (p1, w1)))
    out = profile_difference(a, b)
    out.update(psi_linf=float(np.max(abs(p1 - p0))),
               omega_interior_linf=float(np.max(abs(w1[1:-1, 1:-1] - w0[1:-1, 1:-1]))),
               psi_min_change=float(b["psi_min"] - a["psi_min"]),
               psi_min_change_percent=float(100 * abs(b["psi_min"] - a["psi_min"]) / abs(a["psi_min"])) if a["psi_min"] else None)
    return out


def steady_defect(solver) -> float:
    """Maximum residual of the discrete steady PDE with refreshed wall data.

    Unlike the update-rate monitor this does not include the inverse implicit
    diffusion operator. Neither residual is an a posteriori solution bound.
    """
    w = solver.boundary_omega(solver.psi, solver.t)
    w[1:-1, 1:-1] = solver.omega[1:-1, 1:-1]
    return float(np.max(abs(solver.convection(solver.psi, w) - laplacian(w, solver.grid) / solver.Re)))
