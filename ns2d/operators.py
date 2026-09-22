"""Discrete operators and Dirichlet solvers.

Two linear systems are solved at every time step (report, "Time Discretisation
and Linear Systems"):

    A_omega omega^{n+1} = b_omega ,    A_omega = I + (dt/Re) (-Delta_h)
    A_psi   psi^{n+1}   = b_psi   ,    A_psi   = -Delta_h

Both are written with ``-Delta_h`` so that the matrices are symmetric
*positive* definite, which is what the conjugate gradient method requires.

Three interchangeable backends solve ``(alpha I + beta (-Delta_h)) u = f`` on
the interior unknowns with homogeneous Dirichlet conditions:

``"dst"``
    Fast diagonalisation by the type-I discrete sine transform.  Exact (up to
    round-off), ``O(N log N)``, no iteration.  Default.
``"lu"``
    Sparse LU factorisation, computed once and reused.
``"cg"``
    Preconditioned conjugate gradient with an incomplete-Cholesky preconditioner --
    the iterative path.

The three agree to solver tolerance; ``ns2d.verification`` checks that.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import dstn, idstn

from .grid import Grid

__all__ = [
    "laplacian_matrix",
    "neg_laplacian_matrix",
    "helmholtz_matrix",
    "laplacian",
    "ddx",
    "ddy",
    "DirichletSolver",
    "make_solver",
    "suggest_dt",
    "D_MAX",
]

Backend = Literal["dst", "lu", "cg"]


# ---------------------------------------------------------------------------
# stencils acting on full fields (boundary values are used, not modified)
# ---------------------------------------------------------------------------
def laplacian(phi: np.ndarray, grid: Grid) -> np.ndarray:
    """Five-point Laplacian at interior points; returns an interior-shaped array."""
    d2x = (phi[2:, 1:-1] - 2.0 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / grid.hx**2
    d2y = (phi[1:-1, 2:] - 2.0 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / grid.hy**2
    return d2x + d2y


def ddx(phi: np.ndarray, grid: Grid) -> np.ndarray:
    """Centered d/dx at interior points."""
    return (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2.0 * grid.hx)


def ddy(phi: np.ndarray, grid: Grid) -> np.ndarray:
    """Centered d/dy at interior points."""
    return (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2.0 * grid.hy)


# ---------------------------------------------------------------------------
# sparse matrices
# ---------------------------------------------------------------------------
def _d2_1d(n: int, h: float) -> sp.csr_matrix:
    """Second-difference matrix on the ``n-2`` interior points of ``n`` nodes."""
    m = n - 2
    main = -2.0 * np.ones(m)
    off = np.ones(m - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr") / h**2


def laplacian_matrix(grid: Grid) -> sp.csr_matrix:
    """``Delta_h`` on interior unknowns, homogeneous Dirichlet, row-major in i."""
    dxx = _d2_1d(grid.nx, grid.hx)
    dyy = _d2_1d(grid.ny, grid.hy)
    ix = sp.identity(grid.nx - 2, format="csr")
    iy = sp.identity(grid.ny - 2, format="csr")
    # unknowns ordered as u[i, j] flattened C-style -> kron(x-part, y-part)
    return (sp.kron(dxx, iy) + sp.kron(ix, dyy)).tocsr()


def neg_laplacian_matrix(grid: Grid) -> sp.csr_matrix:
    """``-Delta_h``: symmetric positive definite."""
    return (-laplacian_matrix(grid)).tocsr()


def helmholtz_matrix(grid: Grid, alpha: float, beta: float) -> sp.csr_matrix:
    """``alpha I + beta (-Delta_h)``, symmetric positive definite for alpha, beta > 0."""
    n = grid.n_interior
    return (
        alpha * sp.identity(n, format="csr") + beta * neg_laplacian_matrix(grid)
    ).tocsr()


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------
class DirichletSolver:
    """Solve ``(alpha I + beta (-Delta_h)) u = f`` on the interior points.

    ``solve`` takes and returns arrays of shape ``grid.interior_shape``.
    """

    def __init__(self, grid: Grid, alpha: float, beta: float) -> None:
        self.grid = grid
        self.alpha = float(alpha)
        self.beta = float(beta)
        if (
            not np.isfinite([self.alpha, self.beta]).all()
            or self.beta <= 0.0
            or self.alpha < 0.0
        ):
            raise ValueError("need beta > 0 and alpha >= 0 for positive definiteness")
        self.n_solves = 0
        self.n_iterations = 0

    def solve(self, rhs: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class DSTSolver(DirichletSolver):
    """Fast diagonalisation with the type-I DST (exact, non-iterative).

    The eigenvectors of the 1-D Dirichlet second-difference matrix are
    ``sin(p pi i / (N-1))``, which is precisely the DST-I basis, so the 2-D
    operator is diagonal in transform space with eigenvalues

        lam[p, q] = (2 - 2 cos(p pi/(Nx-1))) / hx**2
                  + (2 - 2 cos(q pi/(Ny-1))) / hy**2 .
    """

    def __init__(self, grid: Grid, alpha: float, beta: float) -> None:
        super().__init__(grid, alpha, beta)
        p = np.arange(1, grid.nx - 1)
        q = np.arange(1, grid.ny - 1)
        lam_x = (2.0 - 2.0 * np.cos(p * np.pi / (grid.nx - 1))) / grid.hx**2
        lam_y = (2.0 - 2.0 * np.cos(q * np.pi / (grid.ny - 1))) / grid.hy**2
        self.eig = self.alpha + self.beta * (lam_x[:, None] + lam_y[None, :])

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        self.n_solves += 1
        fhat = dstn(rhs, type=1, norm="ortho")
        return idstn(fhat / self.eig, type=1, norm="ortho")


class LUSolver(DirichletSolver):
    """Sparse LU factorisation computed once, reused at every time step."""

    def __init__(self, grid: Grid, alpha: float, beta: float) -> None:
        super().__init__(grid, alpha, beta)
        self.A = helmholtz_matrix(grid, alpha, beta)
        self.lu = spla.splu(self.A.tocsc())

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        self.n_solves += 1
        return self.lu.solve(rhs.ravel()).reshape(self.grid.interior_shape)


def incomplete_cholesky(A: sp.spmatrix) -> sp.csr_matrix:
    """Incomplete Cholesky factorisation with zero fill-in, IC(0).

    Returns the lower-triangular ``L`` with the sparsity pattern of
    ``tril(A)`` such that ``L L^T ~ A``.

    SciPy provides no incomplete Cholesky --- only ``spilu``, which is an
    incomplete *LU*.  That matters: the ILU factors of a symmetric matrix are
    not symmetric, so ``M^-1 = U^-1 L^-1`` is not a symmetric positive definite
    operator, and preconditioned conjugate gradient (which assumes it is)
    stagnates.  Measured on the 65x65 Poisson system, ILU-preconditioned CG
    stalls at a relative residual of 0.22 and never converges, while the same
    solve takes 248 unpreconditioned iterations.  Hence this routine.

    IC(0) is guaranteed to exist for an M-matrix, which both ``-Delta_h`` and
    ``A_omega = I + (dt/Re)(-Delta_h)`` are: symmetric positive definite, with
    non-positive off-diagonal entries and weak diagonal dominance.
    """
    L = sp.tril(A.tocsr(), format="csr")
    L.sort_indices()
    indptr, indices, data = L.indptr, L.indices, L.data
    n = L.shape[0]

    rows: list[dict[int, float]] = [dict() for _ in range(n)]
    diag = np.zeros(n)
    for i in range(n):
        ri = rows[i]
        for p in range(indptr[i], indptr[i + 1]):
            j = int(indices[p])
            a = float(data[p])
            if j < i:
                rj = rows[j]
                # subtract the inner product over the shared pattern, k < j
                s = a
                if len(ri) <= len(rj):
                    for k, v in ri.items():
                        if k < j:
                            w = rj.get(k)
                            if w is not None:
                                s -= v * w
                else:
                    for k, w in rj.items():
                        if k < j:
                            v = ri.get(k)
                            if v is not None:
                                s -= v * w
                ri[j] = s / diag[j]
            else:  # j == i, and CSR rows are sorted so ri is complete by now
                s = a - sum(v * v for v in ri.values())
                if s <= 0.0:
                    raise np.linalg.LinAlgError(
                        f"IC(0) breakdown at row {i}: matrix is not an M-matrix"
                    )
                diag[i] = np.sqrt(s)

    ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        ptr[i + 1] = ptr[i] + len(rows[i]) + 1
    idx = np.empty(ptr[-1], dtype=np.int64)
    val = np.empty(ptr[-1], dtype=float)
    for i in range(n):
        cols = sorted(rows[i])
        k = ptr[i]
        for j in cols:
            idx[k] = j
            val[k] = rows[i][j]
            k += 1
        idx[k] = i
        val[k] = diag[i]
    return sp.csr_matrix((val, idx, ptr), shape=(n, n))


class CGSolver(DirichletSolver):
    """Preconditioned conjugate gradient, the iterative path.

    Preconditioned with the IC(0) factorisation above, so that the
    preconditioner really is symmetric positive definite and CG's assumptions
    hold.  The previous solution is used as the initial guess, which makes the
    per-step iteration count drop sharply once the flow settles.
    """

    def __init__(
        self,
        grid: Grid,
        alpha: float,
        beta: float,
        rtol: float = 1e-12,
        maxiter: int = 5000,
        preconditioner: str = "ic0",
        warm_start: bool = True,
    ) -> None:
        super().__init__(grid, alpha, beta)
        self.A = helmholtz_matrix(grid, alpha, beta)
        self.rtol = rtol
        self.maxiter = maxiter
        self.preconditioner = preconditioner

        if preconditioner == "ic0":
            L = incomplete_cholesky(self.A).tocsc()
            lu = spla.splu(
                L,
                permc_spec="NATURAL",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
            lu_t = spla.splu(
                L.T.tocsc(),
                permc_spec="NATURAL",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
            self.L = L
            self.M = spla.LinearOperator(
                self.A.shape, lambda r: lu_t.solve(lu.solve(r))
            )
        elif preconditioner == "jacobi":
            d = self.A.diagonal()
            self.M = spla.LinearOperator(self.A.shape, lambda r: r / d)
        elif preconditioner in (None, "none"):
            self.M = None
        else:
            raise ValueError(f"unknown preconditioner {preconditioner!r}")

        self.warm_start = warm_start
        self._x0: np.ndarray | None = None

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        self.n_solves += 1
        counter = {"n": 0}

        def _cb(_xk):
            counter["n"] += 1

        x, info = spla.cg(
            self.A,
            rhs.ravel(),
            x0=self._x0 if self.warm_start else None,
            rtol=self.rtol,
            maxiter=self.maxiter,
            M=self.M,
            callback=_cb,
        )
        if info != 0:
            raise RuntimeError(f"conjugate gradient failed to converge (info={info})")
        self.n_iterations += counter["n"]
        if self.warm_start:
            self._x0 = x
        return x.reshape(self.grid.interior_shape)


def make_solver(
    grid: Grid, alpha: float, beta: float, backend: Backend = "dst", **kwargs
) -> DirichletSolver:
    """Factory for the three interchangeable Dirichlet solvers."""
    backends = {"dst": DSTSolver, "lu": LUSolver, "cg": CGSolver}
    try:
        cls = backends[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}, expected one of {sorted(backends)}"
        )
    return cls(grid, alpha, beta, **kwargs)


# ---------------------------------------------------------------------------
# time step
# ---------------------------------------------------------------------------
D_MAX = 1.5
"""Conservative square-grid Stokes boundary limit suggested by a finite-grid
spectral study; not a proof of nonlinear stability. See :func:`suggest_dt`."""


def suggest_dt(
    grid: Grid,
    Re: float,
    u_max: float = 1.0,
    safety: float = 0.5,
    d_max: float = D_MAX,
) -> float:
    """Suggested time step for explicit convection and implicit diffusion.

    ``u_max`` is an estimate of the maximum speed, not a measured CFL bound.
    The square-grid Stokes boundary experiment does not prove stability for
    arbitrary nonlinear flows or stretched rectangular grids. Check residuals
    and repeat with a smaller step when using a new configuration.

    Three bounds are combined (report, "Stability and the Time Step"):

    1. *Convective* (von Neumann analysis of the linearised interior scheme)::

           dt <= 2 nu / |u|^2 ,      nu = 1/Re .

       Centered convection with forward Euler is unconditionally unstable on
       its own; it is the implicit viscous term that stabilises it, and enough
       of it must act per step.  Note that ``h`` cancels: this bound alone
       implies no CFL-type restriction.

    2. *Boundary* (lagged Thom condition)::

           nu dt / h^2 <= d_max ~ 1.5 .

       This is the bound the interior von Neumann analysis cannot see, and it
       is the reason the implicit viscous term does *not* buy unconditional
       stability in ``h``.  Thom's formula supplies the wall vorticity from the
       streamfunction of the *previous* step, so the boundary data is explicit
       even though the interior operator is implicit.  Measuring the spectral
       radius of the one-step operator of the convection-free (Stokes) problem
       -- assembled from this very code by :mod:`ns2d.stability` -- gives
       ``rho = 1`` at ``nu dt/h^2 = 1.581, 1.546, 1.532, 1.524, 1.519, 1.514``
       on ``17^2 ... 65^2``, decreasing monotonically towards ``3/2`` and
       independent of ``Re`` to machine precision. Replacing Thom's formula by
       exact wall data removes the restriction entirely, which identifies the
       cause.

       Without this term the rule returns unstable steps whenever
       ``Re h < 1/3`` -- the low-Reynolds, fine-grid corner of parameter space.

    3. *Convective CFL* ``dt <= h/|u|``, which is not required for linear
       stability but is retained as an accuracy and nonlinear-robustness
       safeguard.

    ``safety`` multiplies the minimum of the three.
    """
    if not np.isfinite(Re) or Re <= 0:
        raise ValueError("Re must be finite and positive")
    if not np.isfinite(u_max) or u_max < 0.0:
        raise ValueError("u_max must be finite and nonnegative")
    if not np.isfinite(safety) or not 0 < safety <= 1:
        raise ValueError("safety must lie in (0, 1]")
    if not np.isfinite(d_max) or d_max <= 0:
        raise ValueError("d_max must be finite and positive")
    nu = 1.0 / Re
    h = grid.h_min
    dt_visc = 2.0 * nu / u_max**2 if u_max else np.inf
    dt_bc = d_max * h**2 / nu
    dt_cfl = h / u_max if u_max else np.inf
    return safety * min(dt_visc, dt_bc, dt_cfl)
