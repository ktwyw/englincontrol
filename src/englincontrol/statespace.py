"""State-space design: controllability and observability (rank, singular values, the PBH test, the Kalman
decomposition into controllable/observable parts), pole placement by Ackermann's formula (SISO) and by the
Sylvester-equation method (MIMO), Luenberger observers by duality, integral action by state augmentation, the
linear-quadratic regulator from the Riccati equation (with its guaranteed margins), the steady-state Kalman
filter, and the LQG compensator with the separation principle.

>>> import numpy as np
>>> from englincontrol import statespace as sp
>>> A, B = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])   # double integrator
>>> K = sp.place(A, B, [-1.0, -2.0])
>>> np.round(np.sort(np.linalg.eigvals(A - B @ K).real), 10)
array([-2., -1.])
"""

from __future__ import annotations

import numpy as np

from . import polynomials as pl
from .linalg import care, dare, lyapunov
from .systems import StateSpace


def ctrb(A, B) -> np.ndarray:
    """Controllability matrix [B, AB, ..., A^(n-1) B]."""
    A, B = np.asarray(A, float), np.atleast_2d(np.asarray(B, float))
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(len(A))])


def obsv(A, C) -> np.ndarray:
    """Observability matrix [C; CA; ...; C A^(n-1)]."""
    A, C = np.asarray(A, float), np.atleast_2d(np.asarray(C, float))
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(len(A))])


def controllability(A, B, tol: float = 1e-10) -> dict:
    """Rank test of the controllability matrix with its singular values (tol relative to the largest)."""
    W = ctrb(A, B)
    sv = np.linalg.svd(W, compute_uv=False)
    rank = int(np.sum(sv > tol * sv[0]))
    return {"rank": rank, "controllable": rank == len(np.asarray(A)), "singular_values": sv}


def observability(A, C, tol: float = 1e-10) -> dict:
    """Rank test of the observability matrix with its singular values."""
    W = obsv(A, C)
    sv = np.linalg.svd(W, compute_uv=False)
    rank = int(np.sum(sv > tol * sv[0]))
    return {"rank": rank, "observable": rank == len(np.asarray(A)), "singular_values": sv}


def pbh_test(A, B) -> dict:
    """Popov-Belevitch-Hautus: (A, B) is controllable iff rank [lambda I - A, B] = n for every eigenvalue lambda;
    returns the uncontrollable eigenvalues (those where the rank drops)."""
    A, B = np.asarray(A, float), np.atleast_2d(np.asarray(B, float))
    n = len(A)
    bad = []
    for lam in np.linalg.eigvals(A):
        M = np.hstack([lam * np.eye(n) - A, B])
        if np.linalg.matrix_rank(M, tol=1e-9 * max(1.0, abs(lam))) < n:
            bad.append(lam)
    return {"uncontrollable_modes": np.array(bad), "controllable": len(bad) == 0}


def kalman_decomposition(sys_: StateSpace, tol: float = 1e-9) -> dict:
    """Orthogonal transformation splitting the state into controllable and uncontrollable parts (first), reporting
    the dimensions of the controllable, observable and minimal subsystems."""
    A, B, C = sys_.A, sys_.B, sys_.C
    n = sys_.n
    Uc, sc, _ = np.linalg.svd(ctrb(A, B), full_matrices=True)
    rc = int(np.sum(sc > tol * max(sc[0], 1e-300)))
    Uo, so, _ = np.linalg.svd(obsv(A, C).T, full_matrices=True)
    ro = int(np.sum(so > tol * max(so[0], 1e-300)))
    from .systems import minimal

    return {"n_controllable": rc, "n_observable": ro, "n_minimal": minimal(sys_).n, "n": n, "T_controllable": Uc}


# ------------------------------------------------------------------ pole placement
def acker(A, B, poles) -> np.ndarray:
    """Ackermann's formula (single input): K = [0 ... 0 1] C^-1 phi(A), phi the desired characteristic polynomial."""
    A, B = np.asarray(A, float), np.asarray(B, float).reshape(len(A), 1)
    n = len(A)
    phi = np.real(pl.from_roots(poles))
    phiA = np.zeros((n, n))
    for c in phi:
        phiA = phiA @ A + c * np.eye(n)
    Wc = ctrb(A, B)
    e = np.zeros(n)
    e[-1] = 1.0
    return (np.linalg.solve(Wc.T, e) @ phiA).reshape(1, n)


def place(A, B, poles, iterations: int = 0) -> np.ndarray:
    """State feedback K with eig(A - B K) = poles. Single input: Ackermann (exact, but ill-conditioned for large n).
    Multiple inputs: the Sylvester method - choose G (m x n), solve A X - X Lambda = B G for X with Lambda holding
    the desired poles (real block form), then K = G X^-1 (Bhattacharyya & de Souza 1982); the random choice of G
    makes X invertible with probability one."""
    A, B = np.asarray(A, float), np.atleast_2d(np.asarray(B, float))
    n, m = A.shape[0], B.shape[1]
    poles = np.atleast_1d(np.asarray(poles, complex))
    if m == 1:
        return acker(A, B, poles)
    Lam = _real_block_diag(poles)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(10):
        G = rng.normal(size=(m, n))
        # solve A X - X Lam = B G  (Sylvester with Kronecker form)
        K_ = np.kron(np.eye(n), A) - np.kron(Lam.T, np.eye(n))
        X = np.linalg.solve(K_, (B @ G).reshape(-1, order="F")).reshape(n, n, order="F")
        if np.linalg.cond(X) < 1e12:
            K = G @ np.linalg.inv(X)
            if best is None or np.linalg.norm(K) < np.linalg.norm(best):
                best = K
    return best


def _real_block_diag(poles) -> np.ndarray:
    poles = list(poles)
    blocks = []
    while poles:
        p = poles.pop(0)
        if abs(p.imag) < 1e-12:
            blocks.append(np.array([[p.real]]))
        else:
            j = next(i for i, q in enumerate(poles) if abs(q - np.conj(p)) < 1e-9)
            poles.pop(j)
            blocks.append(np.array([[p.real, p.imag], [-p.imag, p.real]]))
    n = sum(b.shape[0] for b in blocks)
    Lam = np.zeros((n, n))
    i = 0
    for b in blocks:
        k = b.shape[0]
        Lam[i : i + k, i : i + k] = b
        i += k
    return Lam


def observer_gain(A, C, poles) -> np.ndarray:
    """Luenberger observer gain L with eig(A - L C) = poles, by duality: L = place(A^T, C^T, poles)^T."""
    return place(np.asarray(A, float).T, np.atleast_2d(np.asarray(C, float)).T, poles).T


def observer(sys_: StateSpace, L) -> StateSpace:
    """The observer as a system with inputs [u, y] and output x_hat: x_hat' = (A - L C) x_hat + (B - L D) u + L y."""
    A, B, C, D = sys_.A, sys_.B, sys_.C, sys_.D
    L = np.asarray(L, float).reshape(sys_.n, sys_.p)
    return StateSpace(
        A - L @ C, np.hstack([B - L @ D, L]), np.eye(sys_.n), np.zeros((sys_.n, sys_.m + sys_.p)), sys_.dt
    )


def augment_integral(A, B, C) -> tuple:
    """Augment with the integral of the tracking error for integral action: xa = [x, xi], xi' = r - y.
    Returns (Aa, Ba) for state feedback u = -[K Ki] xa (the reference enters through xi' = r - C x)."""
    A, B, C = np.asarray(A, float), np.atleast_2d(np.asarray(B, float)), np.atleast_2d(np.asarray(C, float))
    n, p = len(A), C.shape[0]
    Aa = np.block([[A, np.zeros((n, p))], [-C, np.zeros((p, p))]])
    Ba = np.vstack([B, np.zeros((p, B.shape[1]))])
    return Aa, Ba


def regulator(sys_: StateSpace, K, L, Nbar: float | None = None) -> StateSpace:
    """Observer-based compensator u = -K x_hat + Nbar r: as a system from [r, y] to u (continuous or discrete)."""
    A, B, C, D = sys_.A, sys_.B, sys_.C, sys_.D
    K = np.asarray(K, float).reshape(sys_.m, sys_.n)
    L = np.asarray(L, float).reshape(sys_.n, sys_.p)
    Nbar = reference_gain(A, B, C, K) if Nbar is None else Nbar
    Ac = A - B @ K - L @ C + L @ D @ K
    Bc = np.hstack([(B - L @ D) * Nbar, L])
    Cc = -K
    Dc = np.hstack([np.full((sys_.m, 1), Nbar), np.zeros((sys_.m, sys_.p))])
    return StateSpace(Ac, Bc, Cc, Dc, sys_.dt)


def reference_gain(A, B, C, K) -> float:
    """Feedforward gain Nbar such that the closed loop y = C (sI - A + BK)^-1 B Nbar r has unit dc gain (SISO)."""
    A, B, C, K = (np.asarray(m, float) for m in (A, B, C, K))
    Acl = A - B @ K.reshape(B.shape[1], len(A))
    g = (C @ np.linalg.solve(-Acl, B))[0, 0]
    return 1.0 / g


def closed_loop_with_observer(sys_: StateSpace, K, L, Nbar: float | None = None) -> StateSpace:
    """The full closed loop [x, e] (e = x - x_hat) from r to y: block-triangular, so its eigenvalues are those of
    A - BK and A - LC together (the separation principle)."""
    A, B, C, D = sys_.A, sys_.B, sys_.C, sys_.D
    K = np.asarray(K, float).reshape(sys_.m, sys_.n)
    L = np.asarray(L, float).reshape(sys_.n, sys_.p)
    Nbar = reference_gain(A, B, C, K) if Nbar is None else Nbar
    n = sys_.n
    Acl = np.block([[A - B @ K, B @ K], [np.zeros((n, n)), A - L @ C]])
    Bcl = np.vstack([B * Nbar, np.zeros((n, sys_.m))])
    Ccl = np.hstack([C - D @ K, D @ K])
    Dcl = D * Nbar
    return StateSpace(Acl, Bcl, Ccl, Dcl, sys_.dt)


# ------------------------------------------------------------------ LQR, Kalman filter, LQG
def lqr(A, B, Q, R, discrete: bool = False) -> dict:
    """Optimal state feedback u = -K x minimising int (x^T Q x + u^T R u) dt (or the sum, discrete):
    K = R^-1 B^T P (continuous) or (R + B^T P B)^-1 B^T P A (discrete), P the stabilising Riccati solution."""
    A, B, Q, R = (np.asarray(m, float) for m in (A, B, Q, R))
    B = np.atleast_2d(B)
    if B.shape[0] != len(A):
        B = B.T
    R = np.atleast_2d(R)
    if discrete:
        P = dare(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    else:
        P = care(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
    return {"K": K, "P": P, "closed_loop_poles": np.linalg.eigvals(A - B @ K)}


def lqr_integral(A, B, C, Q, R, qi: float) -> dict:
    """LQR with integral action on the augmented state [x, int(r - y)]: weights Q on x and qi on the integral."""
    Aa, Ba = augment_integral(A, B, C)
    n = len(np.asarray(A))
    Qa = np.zeros((Aa.shape[0], Aa.shape[0]))
    Qa[:n, :n] = Q
    Qa[n:, n:] = qi * np.eye(Aa.shape[0] - n)
    r = lqr(Aa, Ba, Qa, R)
    return {
        "K": r["K"][:, :n],
        "Ki": r["K"][:, n:],
        "P": r["P"],
        "closed_loop_poles": r["closed_loop_poles"],
        "Aa": Aa,
        "Ba": Ba,
    }


def kalman(A, C, Qn, Rn, G=None, discrete: bool = False) -> dict:
    """Steady-state Kalman filter for x' = A x + G w, y = C x + v with E[w w^T] = Qn, E[v v^T] = Rn: the gain
    L = P C^T Rn^-1 (continuous) with P from the dual Riccati equation; discrete: predictor form, L = A P C^T
    (C P C^T + Rn)^-1 (returned as the gain of x_hat[k+1|k] = A x_hat + B u + L (y - C x_hat))."""
    A, C, Qn, Rn = (np.asarray(m, float) for m in (A, C, Qn, Rn))
    C = np.atleast_2d(C)
    Rn = np.atleast_2d(Rn)
    G = np.eye(len(A)) if G is None else np.asarray(G, float)
    Qw = G @ np.atleast_2d(Qn) @ G.T
    if discrete:
        P = dare(A.T, C.T, Qw, Rn)
        L = A @ P @ C.T @ np.linalg.inv(C @ P @ C.T + Rn)
    else:
        P = care(A.T, C.T, Qw, Rn)
        L = P @ C.T @ np.linalg.inv(Rn)
    return {"L": L, "P": P, "estimator_poles": np.linalg.eigvals(A - L @ C)}


def kalman_filter_run(A, B, C, L, u, y, x0=None) -> np.ndarray:
    """Run a discrete steady-state Kalman filter (predictor form) over recorded inputs and measurements."""
    A, B, C, L = (np.asarray(m, float) for m in (A, B, C, L))
    u = np.atleast_2d(np.asarray(u, float).T).T
    y = np.atleast_2d(np.asarray(y, float).T).T
    n = len(A)
    xh = np.zeros((len(y), n))
    x = np.zeros(n) if x0 is None else np.asarray(x0, float)
    for k in range(len(y)):
        xh[k] = x
        x = A @ x + B @ u[k] + L @ (y[k] - C @ x)
    return xh


def lqg(sys_: StateSpace, Q, R, Qn, Rn) -> dict:
    """LQG = LQR state feedback on the Kalman estimate; returns K, L, the compensator (from y to u, regulator
    form with no reference) and the closed-loop system from an output disturbance d to y."""
    r = lqr(sys_.A, sys_.B, Q, R, discrete=sys_.discrete)
    k = kalman(sys_.A, sys_.C, Qn, Rn, discrete=sys_.discrete)
    K, L = r["K"], k["L"]
    comp = StateSpace(sys_.A - sys_.B @ K - L @ sys_.C, L, -K, np.zeros((sys_.m, sys_.p)), sys_.dt)
    return {
        "K": K,
        "L": L,
        "compensator": comp,
        "regulator_poles": r["closed_loop_poles"],
        "estimator_poles": k["estimator_poles"],
    }


def lqr_margins(A, B, Q, R) -> dict:
    """Loop transfer K (sI - A)^-1 B at the plant input and its gain and phase margins (single input):
    the LQR guarantees gain margin [1/2, inf) and phase margin >= 60 degrees."""
    from .frequency import margins

    r = lqr(A, B, Q, R)
    L = StateSpace(A, B, r["K"], np.zeros((1, 1)))
    return margins(L.to_tf())


def cost_to_go(P, x0) -> float:
    """J = x0^T P x0: the optimal LQR cost from the initial state."""
    x0 = np.asarray(x0, float)
    return float(x0 @ np.asarray(P, float) @ x0)


def lyapunov_cost(A, Q, x0) -> float:
    """int x^T Q x dt along x' = A x from x0 equals x0^T X x0 with A^T X + X A + Q = 0."""
    X = lyapunov(np.asarray(A, float).T, np.asarray(Q, float))
    x0 = np.asarray(x0, float)
    return float(x0 @ X @ x0)
