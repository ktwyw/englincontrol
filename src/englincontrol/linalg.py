"""Matrix computations of control theory, in NumPy: the matrix exponential by scaling and squaring with a
degree-13 Pade approximant (Higham's algorithm), the matrix exponential together with its integral (for
discretisation), Lyapunov and Sylvester equations (Bartels-Stewart via the real Schur form computed with the
QR algorithm, Kronecker form for small sizes), the continuous and discrete algebraic Riccati equations
(stable invariant subspace of the Hamiltonian / symplectic pencil, with Newton-Kleinman refinement), and
controllability/observability Gramians.

>>> import numpy as np
>>> from englincontrol import linalg as la
>>> A = np.array([[0.0, 1.0], [-2.0, -3.0]])
>>> np.round(la.expm(A), 6)
array([[ 0.600424,  0.232544],
       [-0.465088, -0.097209]])
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ matrix exponential
_PADE13 = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
]


def expm(A) -> np.ndarray:
    """exp(A) by scaling and squaring with the [13/13] Pade approximant (Higham 2005)."""
    A = np.asarray(A, float)
    n = len(A)
    norm = np.linalg.norm(A, 1)
    s = max(0, int(np.ceil(np.log2(norm / 5.371920351148152)))) if norm > 5.371920351148152 else 0
    A = A / 2**s
    b = _PADE13
    eye = np.eye(n)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2
    U = A @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * eye)
    V = A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * eye
    R = np.linalg.solve(V - U, V + U)
    for _ in range(s):
        R = R @ R
    return R


def expm_with_integral(A, B, h: float) -> tuple:
    """exp(A h) and the integral int_0^h exp(A tau) dtau B, from one exponential of the augmented matrix
    [[A, B], [0, 0]] (Van Loan): the zero-order-hold discretisation (Phi, Gamma)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * h
    M[:n, n:] = B * h
    E = expm(M)
    return E[:n, :n], E[:n, n:]


# ------------------------------------------------------------------ Schur form (for Lyapunov / Sylvester)
def schur(A, maxiter: int = 5000) -> tuple:
    """Real Schur form A = Q T Q^T (T quasi-triangular) by the shifted QR algorithm on the Hessenberg form,
    with deflation. Returns (T, Q)."""
    A = np.asarray(A, float).copy()
    n = len(A)
    Q = np.eye(n)
    # Hessenberg reduction by Householder
    for k in range(n - 2):
        x = A[k + 1 :, k].copy()
        if np.linalg.norm(x[1:]) < 1e-300:
            continue
        v = x.copy()
        v[0] += np.sign(x[0] if x[0] != 0 else 1.0) * np.linalg.norm(x)
        v /= np.linalg.norm(v)
        A[k + 1 :, :] -= 2 * np.outer(v, v @ A[k + 1 :, :])
        A[:, k + 1 :] -= 2 * np.outer(A[:, k + 1 :] @ v, v)
        Q[:, k + 1 :] -= 2 * np.outer(Q[:, k + 1 :] @ v, v)
    T = A
    hi = n - 1
    it = 0
    while hi > 0 and it < maxiter:
        it += 1
        # deflate zero subdiagonals
        lo = hi
        while lo > 0 and abs(T[lo, lo - 1]) > 1e-14 * (abs(T[lo, lo]) + abs(T[lo - 1, lo - 1])):
            lo -= 1
        if lo == hi:
            hi -= 1
            continue
        if lo == hi - 1:  # a 2 x 2 block: leave it if its eigenvalues are complex, else split it
            blk = T[hi - 1 : hi + 1, hi - 1 : hi + 1]
            tr, det = blk[0, 0] + blk[1, 1], blk[0, 0] * blk[1, 1] - blk[0, 1] * blk[1, 0]
            disc = tr * tr / 4 - det
            if disc < 0:
                hi -= 2
                continue
            # real eigenvalues: rotate to triangular form
            lam = tr / 2 + np.sqrt(disc) * np.sign(tr if tr != 0 else 1.0)
            x = (
                np.array([blk[0, 1], lam - blk[0, 0]])
                if abs(blk[0, 1]) > abs(blk[1, 0])
                else np.array([lam - blk[1, 1], blk[1, 0]])
            )
            if np.linalg.norm(x) < 1e-300:
                hi -= 2
                continue
            x /= np.linalg.norm(x)
            G = np.array([[x[0], x[1]], [-x[1], x[0]]])
            T[hi - 1 : hi + 1, :] = G @ T[hi - 1 : hi + 1, :]
            T[:, hi - 1 : hi + 1] = T[:, hi - 1 : hi + 1] @ G.T
            Q[:, hi - 1 : hi + 1] = Q[:, hi - 1 : hi + 1] @ G.T
            T[hi, hi - 1] = 0.0
            hi -= 2
            continue
        # shifted QR step on the active block lo..hi (Wilkinson shift; an explicit double step for complex pairs)
        blk = T[lo : hi + 1, lo : hi + 1]
        m = hi - lo + 1
        a, b_, c_, d = blk[-2, -2], blk[-2, -1], blk[-1, -2], blk[-1, -1]
        tr, det = a + d, a * d - b_ * c_
        disc = tr * tr / 4 - det
        if disc >= 0:
            mu = tr / 2 + np.sqrt(disc)
            if abs(mu - d) > abs(tr / 2 - np.sqrt(disc) - d):
                mu = tr / 2 - np.sqrt(disc)
            Qk, Rk = np.linalg.qr(blk - mu * np.eye(m))
            new = Rk @ Qk + mu * np.eye(m)
        else:  # complex pair: double step (blk - mu)(blk - conj mu) is real
            M = blk @ blk - tr * blk + det * np.eye(m)
            Qk, Rk = np.linalg.qr(M)
            new = Qk.T @ blk @ Qk
        T[lo : hi + 1, lo : hi + 1] = new
        T[lo : hi + 1, hi + 1 :] = Qk.T @ T[lo : hi + 1, hi + 1 :]
        T[:lo, lo : hi + 1] = T[:lo, lo : hi + 1] @ Qk
        Q[:, lo : hi + 1] = Q[:, lo : hi + 1] @ Qk
    return T, Q


def sylvester(A, B, C) -> np.ndarray:
    """Solve A X + X B = C by the Kronecker form (small matrices) - exact and simple; O((n m)^3)."""
    A, B, C = np.asarray(A, float), np.asarray(B, float), np.asarray(C, float)
    n, m = A.shape[0], B.shape[0]
    K = np.kron(np.eye(m), A) + np.kron(B.T, np.eye(n))
    return np.linalg.solve(K, C.reshape(-1, order="F")).reshape(n, m, order="F")


def lyapunov(A, Q) -> np.ndarray:
    """Solve the continuous Lyapunov equation A P + P A^T + Q = 0 (Bartels-Stewart on the real Schur form of A,
    which costs O(n^3); falls back to the Kronecker form for n <= 4)."""
    A, Q = np.asarray(A, float), np.asarray(Q, float)
    n = len(A)
    if n <= 4:
        return sylvester(A, A.T, -Q)
    T, U = schur(A)
    F = U.T @ Q @ U
    Y = _lyap_schur(T, F)
    P = U @ Y @ U.T
    return 0.5 * (P + P.T)


def _lyap_schur(T, F):
    """T Y + Y T^T + F = 0 with T quasi-upper-triangular, solved block by block from the bottom right."""
    n = len(T)
    Y = np.zeros((n, n))
    # identify blocks
    blocks = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1e-14 * (abs(T[i, i]) + abs(T[i + 1, i + 1])):
            blocks.append((i, i + 2))
            i += 2
        else:
            blocks.append((i, i + 1))
            i += 1
    for bi in range(len(blocks) - 1, -1, -1):
        i0, i1 = blocks[bi]
        for bj in range(len(blocks) - 1, bi - 1, -1):
            j0, j1 = blocks[bj]
            rhs = -F[i0:i1, j0:j1] - T[i0:i1, i1:] @ Y[i1:, j0:j1] - Y[i0:i1, j1:] @ T[j0:j1, j1:].T
            Y[i0:i1, j0:j1] = sylvester(T[i0:i1, i0:i1], T[j0:j1, j0:j1].T, rhs)
            Y[j0:j1, i0:i1] = Y[i0:i1, j0:j1].T
    return Y


def dlyapunov(A, Q) -> np.ndarray:
    """Solve the discrete Lyapunov equation A P A^T - P + Q = 0 (Kronecker form)."""
    A, Q = np.asarray(A, float), np.asarray(Q, float)
    n = len(A)
    K = np.eye(n * n) - np.kron(A, A)
    P = np.linalg.solve(K, Q.reshape(-1, order="F")).reshape(n, n, order="F")
    return 0.5 * (P + P.T)


# ------------------------------------------------------------------ Riccati equations
def care(A, B, Q, R, newton_steps: int = 5) -> np.ndarray:
    """Stabilising solution P of A^T P + P A - P B R^-1 B^T P + Q = 0 from the stable invariant subspace of the
    Hamiltonian matrix [[A, -B R^-1 B^T], [-Q, -A^T]] (eigenvector method), refined by Newton-Kleinman
    iterations (each a Lyapunov equation) to full accuracy."""
    A, B, Q, R = (np.asarray(m, float) for m in (A, B, Q, R))
    n = len(A)
    S = B @ np.linalg.solve(R, B.T)
    H = np.block([[A, -S], [-Q, -A.T]])
    lam, V = np.linalg.eig(H)
    scale = max(1.0, float(np.max(np.abs(lam))))
    stable = V[:, lam.real < -1e-10 * scale]  # eigenvalues on the axis are not stable: no stabilising solution
    if stable.shape[1] != n:
        raise ValueError(
            "the Hamiltonian has no n-dimensional stable subspace: (A, B) not stabilisable or (Q, A) not detectable"
        )
    X1, X2 = stable[:n, :], stable[n:, :]
    try:
        P = np.linalg.solve(X1.T, X2.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("the stable subspace is not a graph over the state space: no stabilising solution") from exc
    P = np.real(P)
    P = 0.5 * (P + P.T)
    for _ in range(newton_steps):  # Newton-Kleinman: (A - S P)^T dP + dP (A - S P) + residual = 0
        resid = A.T @ P + P @ A - P @ S @ P + Q
        if np.linalg.norm(resid) < 1e-13 * max(1.0, np.linalg.norm(P)):
            break
        Ak = A - S @ P
        dP = sylvester(Ak.T, Ak, -resid)
        P = 0.5 * ((P + dP) + (P + dP).T)
    return P


def dare(A, B, Q, R, newton_steps: int = 20) -> np.ndarray:
    """Stabilising solution P of A^T P A - P - A^T P B (R + B^T P B)^-1 B^T P A + Q = 0, by the structure-preserving
    doubling algorithm (Chu et al. 2004; needs no inverse of A), refined by Newton iterations."""
    A, B, Q, R = (np.asarray(m, float) for m in (A, B, Q, R))
    n = len(A)
    G = B @ np.linalg.solve(R, B.T)
    Ak, Gk, Hk = A.copy(), G.copy(), Q.copy()
    for _ in range(100):
        M = np.linalg.solve(np.eye(n) + Gk @ Hk, Ak)
        Hn = Hk + Ak.T @ Hk @ M
        Gn = Gk + Ak @ np.linalg.solve(np.eye(n) + Gk @ Hk, Gk) @ Ak.T
        An = Ak @ M
        if np.linalg.norm(Hn - Hk) <= 1e-14 * max(1.0, np.linalg.norm(Hn)):
            Hk = Hn
            break
        Ak, Gk, Hk = An, Gn, Hn
    P = 0.5 * (Hk + Hk.T)
    for _ in range(newton_steps):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Ac = A - B @ K
        resid = Ac.T @ P @ Ac - P + Q + K.T @ R @ K
        if np.linalg.norm(resid) < 1e-13 * max(1.0, np.linalg.norm(P)):
            break
        P = dlyapunov(Ac.T, Q + K.T @ R @ K)  # the Newton step: the closed-loop discrete Lyapunov equation
        P = 0.5 * (P + P.T)
    return P


def gramians(A, B, C) -> tuple:
    """Controllability Gramian Wc (A Wc + Wc A^T + B B^T = 0) and observability Gramian Wo (A^T Wo + Wo A + C^T C = 0)
    of a stable continuous-time system."""
    A, B, C = (np.asarray(m, float) for m in (A, B, C))
    return lyapunov(A, B @ B.T), lyapunov(A.T, C.T @ C)


def dgramians(A, B, C) -> tuple:
    """Controllability and observability Gramians of a stable discrete-time system (discrete Lyapunov equations)."""
    A, B, C = (np.asarray(m, float) for m in (A, B, C))
    return dlyapunov(A, B @ B.T), dlyapunov(A.T, C.T @ C)
