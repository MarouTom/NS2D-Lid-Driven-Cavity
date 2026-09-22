"""Published cavity benchmarks with explicit provenance and anomaly flags.

Ghia, Ghia & Shin (1982), Tables I, II and V, DOI 10.1016/0021-9991(82)90058-4.
Botella & Peyret (1998), Tables 6, 9 and 10, DOI 10.1016/S0045-7930(98)00002-4.
See docs/references.md for the source audit and coordinate transformation.
Published values are preserved, including suspect entries; no fitted corrections
are applied. Headline metrics include every station. Secondary metrics omit
explicitly flagged entries and must be reported as such.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# -- Table I: u along x = 0.5 (y descending, as printed) ---------------------
GHIA_Y = np.array(
    [
        1.0000,
        0.9766,
        0.9688,
        0.9609,
        0.9531,
        0.8516,
        0.7344,
        0.6172,
        0.5000,
        0.4531,
        0.2813,
        0.1719,
        0.1016,
        0.0703,
        0.0625,
        0.0547,
        0.0000,
    ]
)

GHIA_U: dict[int, np.ndarray] = {
    100: np.array(
        [
            1.00000,
            0.84123,
            0.78871,
            0.73722,
            0.68717,
            0.23151,
            0.00332,
            -0.13641,
            -0.20581,
            -0.21090,
            -0.15662,
            -0.10150,
            -0.06434,
            -0.04775,
            -0.04192,
            -0.03717,
            0.00000,
        ]
    ),
    400: np.array(
        [
            1.00000,
            0.75837,
            0.68439,
            0.61756,
            0.55892,
            0.29093,
            0.16256,
            0.02135,
            -0.11477,
            -0.17119,
            -0.32726,
            -0.24299,
            -0.14612,
            -0.10338,
            -0.09266,
            -0.08186,
            0.00000,
        ]
    ),
    1000: np.array(
        [
            1.00000,
            0.65928,
            0.57492,
            0.51117,
            0.46604,
            0.33304,
            0.18719,
            0.05702,
            -0.06080,
            -0.10648,
            -0.27805,
            -0.38289,
            -0.29730,
            -0.22220,
            -0.20196,
            -0.18109,
            0.00000,
        ]
    ),
    3200: np.array(
        [
            1.00000,
            0.53236,
            0.48296,
            0.46547,
            0.46101,
            0.34682,
            0.19791,
            0.07156,
            -0.04272,
            -0.86636,
            -0.24427,
            -0.34323,
            -0.41933,
            -0.37827,
            -0.35344,
            -0.32407,
            0.00000,
        ]
    ),
    5000: np.array(
        [
            1.00000,
            0.48223,
            0.46120,
            0.45992,
            0.46036,
            0.33556,
            0.20087,
            0.08183,
            -0.03039,
            -0.07404,
            -0.22855,
            -0.33050,
            -0.40435,
            -0.43643,
            -0.42901,
            -0.41165,
            0.00000,
        ]
    ),
    7500: np.array(
        [
            1.00000,
            0.47244,
            0.47048,
            0.47323,
            0.47167,
            0.34228,
            0.20591,
            0.08342,
            -0.03800,
            -0.07503,
            -0.23176,
            -0.32393,
            -0.38324,
            -0.43025,
            -0.43590,
            -0.43154,
            0.00000,
        ]
    ),
    10000: np.array(
        [
            1.00000,
            0.47221,
            0.47783,
            0.48070,
            0.47804,
            0.34635,
            0.20673,
            0.08344,
            0.03111,
            -0.07540,
            -0.23186,
            -0.32709,
            -0.38000,
            -0.41657,
            -0.42537,
            -0.42735,
            0.00000,
        ]
    ),
}

# -- Table II: v along y = 0.5 (x descending, as printed) -------------------
GHIA_X = np.array(
    [
        1.0000,
        0.9688,
        0.9609,
        0.9531,
        0.9453,
        0.9063,
        0.8594,
        0.8047,
        0.5000,
        0.2344,
        0.2266,
        0.1563,
        0.0938,
        0.0781,
        0.0703,
        0.0625,
        0.0000,
    ]
)

GHIA_V: dict[int, np.ndarray] = {
    100: np.array(
        [
            0.00000,
            -0.05906,
            -0.07391,
            -0.08864,
            -0.10313,
            -0.16914,
            -0.22445,
            -0.24533,
            0.05454,
            0.17527,
            0.17507,
            0.16077,
            0.12317,
            0.10890,
            0.10091,
            0.09233,
            0.00000,
        ]
    ),
    400: np.array(
        [
            0.00000,
            -0.12146,
            -0.15663,
            -0.19254,
            -0.22847,
            -0.23827,
            -0.44993,
            -0.38598,
            0.05186,
            0.30174,
            0.30203,
            0.28124,
            0.22965,
            0.20920,
            0.19713,
            0.18360,
            0.00000,
        ]
    ),
    1000: np.array(
        [
            0.00000,
            -0.21388,
            -0.27669,
            -0.33714,
            -0.39188,
            -0.51550,
            -0.42665,
            -0.31966,
            0.02526,
            0.32235,
            0.33075,
            0.37095,
            0.32627,
            0.30353,
            0.29012,
            0.27485,
            0.00000,
        ]
    ),
    3200: np.array(
        [
            0.00000,
            -0.39017,
            -0.47425,
            -0.52357,
            -0.54053,
            -0.44307,
            -0.37401,
            -0.31184,
            0.00999,
            0.28188,
            0.29030,
            0.37119,
            0.42768,
            0.41906,
            0.40917,
            0.39560,
            0.00000,
        ]
    ),
    5000: np.array(
        [
            0.00000,
            -0.49774,
            -0.55069,
            -0.55408,
            -0.52876,
            -0.41442,
            -0.36214,
            -0.30018,
            0.00945,
            0.27280,
            0.28066,
            0.35368,
            0.42951,
            0.43648,
            0.43329,
            0.42447,
            0.00000,
        ]
    ),
    7500: np.array(
        [
            0.00000,
            -0.53858,
            -0.55216,
            -0.52347,
            -0.48590,
            -0.41050,
            -0.36213,
            -0.30448,
            0.00824,
            0.27348,
            0.28117,
            0.35060,
            0.41824,
            0.43564,
            0.44030,
            0.43979,
            0.00000,
        ]
    ),
    10000: np.array(
        [
            0.00000,
            -0.54302,
            -0.52987,
            -0.49099,
            -0.45863,
            -0.41496,
            -0.36737,
            -0.30719,
            0.00831,
            0.27224,
            0.28003,
            0.35070,
            0.41487,
            0.43124,
            0.43733,
            0.43983,
            0.00000,
        ]
    ),
}


FLAGGED_U_POINTS = {3200: [0.4531], 10000: [0.5]}
FLAGGED_V_POINTS = {400: [0.9063]}
SUSPECT_V_POINTS = FLAGGED_V_POINTS  # compatibility alias
GHIA_VORTEX_GRID = {100: 129, 400: 257, 1000: 129, 3200: 129, 5000: 257}


@dataclass(frozen=True)
class VortexData:
    psi: float
    x: float
    y: float


GHIA_VORTEX = {
    100: VortexData(-0.103423, 0.6172, 0.7344),
    400: VortexData(-0.113909, 0.5547, 0.6055),
    1000: VortexData(-0.117929, 0.5313, 0.5625),
    3200: VortexData(-0.120377, 0.5165, 0.5469),
    5000: VortexData(-0.118966, 0.5117, 0.5352),
}
# The printed Re=3200 x=0.5165 is retained despite not being a 129-grid node.

# Botella & Peyret use a LEFT-moving lid. Preserve the published data here;
# transform x -> 1-x, u -> -u, v -> v, psi -> -psi for our right-moving lid.
BP_Y = GHIA_Y.copy()
BP_U_LEFT = np.array(
    [
        -1.0,
        -0.6644227,
        -0.5808359,
        -0.5169277,
        -0.4723329,
        -0.3372212,
        -0.1886747,
        -0.0570178,
        0.0620561,
        0.1081999,
        0.2803696,
        0.3885691,
        0.3004561,
        0.2228955,
        0.2023300,
        0.1812881,
        0.0,
    ]
)
BP_X_LEFT = np.array(
    [
        0.0,
        0.0312,
        0.0391,
        0.0469,
        0.0547,
        0.0937,
        0.1406,
        0.1953,
        0.5,
        0.7656,
        0.7734,
        0.8437,
        0.9062,
        0.9219,
        0.9297,
        0.9375,
        1.0,
    ]
)
BP_V_LEFT = np.array(
    [
        0.0,
        -0.2279225,
        -0.2936869,
        -0.3553213,
        -0.4103754,
        -0.5264392,
        -0.4264545,
        -0.3202137,
        0.0257995,
        0.3253592,
        0.3339924,
        0.3769189,
        0.3330442,
        0.3099097,
        0.2962703,
        0.2807056,
        0.0,
    ]
)
BP_VORTEX = VortexData(-0.1189366, 1.0 - 0.4692, 0.5652)


def available_reynolds() -> list[int]:
    return sorted(GHIA_U)


def _ascending(coord, values):
    order = np.argsort(coord)
    return coord[order].copy(), values[order].copy()


def u_profile(Re: int):
    """Published (y,u) at x=0.5, sorted by y; includes flagged stations."""
    if Re not in GHIA_U:
        raise KeyError(f"no Ghia data for Re={Re}; available: {available_reynolds()}")
    return _ascending(GHIA_Y, GHIA_U[Re])


def v_profile(Re: int):
    """Published (x,v) at y=0.5, sorted by x; includes flagged stations."""
    if Re not in GHIA_V:
        raise KeyError(f"no Ghia data for Re={Re}; available: {available_reynolds()}")
    return _ascending(GHIA_X, GHIA_V[Re])


@dataclass
class GhiaComparison:
    """RMS errors include endpoints and all flagged stations by default."""

    Re: int
    y: np.ndarray
    u_num: np.ndarray
    u_ref: np.ndarray
    x: np.ndarray
    v_num: np.ndarray
    v_ref: np.ndarray
    u_rms: float
    v_rms: float
    u_max_abs: float
    v_max_abs: float
    u_rms_unflagged: float
    v_rms_unflagged: float
    u_max_abs_unflagged: float
    v_max_abs_unflagged: float
    flagged_u_y: list[float]
    flagged_v_x: list[float]
    reference: str = "Ghia et al. (1982)"

    @property
    def excluded_v_x(self):
        """Deprecated alias; stations are only excluded from secondary metrics."""
        return self.flagged_v_x

    def summary(self):
        lines = [
            f"{self.reference}, Re={self.Re} (all {len(self.y)} u / {len(self.x)} v stations)",
            f"  u RMS = {self.u_rms:.3e}, max = {self.u_max_abs:.3e}",
            f"  v RMS = {self.v_rms:.3e}, max = {self.v_max_abs:.3e}",
        ]
        if self.flagged_u_y:
            lines.append(
                f"  without flagged u stations {self.flagged_u_y}: RMS = {self.u_rms_unflagged:.3e}"
            )
        if self.flagged_v_x:
            lines.append(
                f"  without flagged v stations {self.flagged_v_x}: RMS = {self.v_rms_unflagged:.3e}"
            )
        return "\n".join(lines)


def _compare(
    u,
    v,
    grid,
    Re,
    y_ref,
    u_ref,
    x_ref,
    v_ref,
    flagged_u=(),
    flagged_v=(),
    reference="Ghia et al. (1982)",
):
    from .postproc import centerline_u, centerline_v

    if not (np.isclose(grid.lx, 1.0) and np.isclose(grid.ly, 1.0)):
        raise ValueError("cavity benchmark requires a unit-square domain")
    if np.shape(u) != grid.shape or np.shape(v) != grid.shape:
        raise ValueError("velocity fields must match the grid")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("velocity fields must be finite")
    un = np.interp(y_ref, grid.y, centerline_u(u, grid))
    vn = np.interp(x_ref, grid.x, centerline_v(v, grid))
    du, dv = un - u_ref, vn - v_ref
    ku = ~np.isin(y_ref, flagged_u)
    kv = ~np.isin(x_ref, flagged_v)
    rms = lambda a: float(np.sqrt(np.mean(a * a)))
    maxabs = lambda a: float(np.max(np.abs(a)))
    return GhiaComparison(
        Re,
        y_ref,
        un,
        u_ref,
        x_ref,
        vn,
        v_ref,
        rms(du),
        rms(dv),
        maxabs(du),
        maxabs(dv),
        rms(du[ku]),
        rms(dv[kv]),
        maxabs(du[ku]),
        maxabs(dv[kv]),
        list(flagged_u),
        list(flagged_v),
        reference,
    )


def compare_to_ghia(u, v, grid, Re: int):
    """Compare right-moving unit-lid velocities with all published stations."""
    y, ur = u_profile(Re)
    x, vr = v_profile(Re)
    return _compare(
        u,
        v,
        grid,
        Re,
        y,
        ur,
        x,
        vr,
        FLAGGED_U_POINTS.get(Re, ()),
        FLAGGED_V_POINTS.get(Re, ()),
    )


def compare_to_botella(u, v, grid):
    """Compare an Re=1000 field after reflecting the left-lid reference."""
    y, ur = _ascending(BP_Y, -BP_U_LEFT)
    x, vr = _ascending(1.0 - BP_X_LEFT, BP_V_LEFT)
    return _compare(u, v, grid, 1000, y, ur, x, vr, reference="Botella & Peyret (1998)")
