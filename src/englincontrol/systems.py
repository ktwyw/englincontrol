"""LTI system objects: StateSpace (A, B, C, D) and TransferFunction (numerator, denominator; SISO), with exact
conversions between them (controllable canonical realisation; the transfer function of a realisation by the
Leverrier-Faddeev / resolvent method), canonical forms (controllable, observable, modal), poles and zeros,
dc gain, interconnections (series, parallel, feedback) in both representations, similarity transformations,
and minimal realisation by removing uncontrollable and unobservable modes. Discrete-time systems carry a
sampling period dt.

>>> from englincontrol import systems as sy
>>> G = sy.tf([1.0], [1.0, 3.0, 2.0])             # 1 / ((s + 1)(s + 2))
>>> sorted(G.poles().real.tolist())
[-2.0, -1.0]
>>> G.to_ss().to_tf().den
array([1., 3., 2.])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import polynomials as pl


@dataclass
class StateSpace:
    """x' = A x + B u, y = C x + D u (continuous) or x[k+1] = A x[k] + B u[k] (discrete, dt > 0)."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    dt: float = 0.0

    def __post_init__(self):
        self.A = np.atleast_2d(np.asarray(self.A, float))
        self.B = np.atleast_2d(np.asarray(self.B, float))
        self.C = np.atleast_2d(np.asarray(self.C, float))
        self.D = np.atleast_2d(np.asarray(self.D, float))
        n = self.A.shape[0]
        if self.B.shape[0] != n:
            self.B = self.B.T
        if self.C.shape[1] != n:
            self.C = self.C.T
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            self.D = np.broadcast_to(self.D, (self.C.shape[0], self.B.shape[1])).copy()

    @property
    def n(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    @property
    def p(self) -> int:
        return self.C.shape[0]

    @property
    def discrete(self) -> bool:
        return self.dt > 0

    def poles(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    def zeros(self) -> np.ndarray:
        """Transmission zeros of a SISO system (roots of the transfer function's numerator after cancellation
        of common factors with the denominator)."""
        return self.to_tf().zeros()

    def dcgain(self) -> np.ndarray:
        """y/u at s = 0 (or z = 1): D - C A^-1 B, or C (I - A)^-1 B + D."""
        M = self.A if not self.discrete else self.A - np.eye(self.n)
        return self.D - self.C @ np.linalg.solve(M, self.B)

    def transform(self, T) -> StateSpace:
        """Similarity transformation z = T^-1 x: (T^-1 A T, T^-1 B, C T, D)."""
        T = np.asarray(T, float)
        Ti = np.linalg.inv(T)
        return StateSpace(Ti @ self.A @ T, Ti @ self.B, self.C @ T, self.D, self.dt)

    def to_tf(self) -> TransferFunction:
        """SISO transfer function by the Leverrier-Faddeev algorithm: den = det(sI - A) and
        num = C adj(sI - A) B + D den, computed exactly in the coefficients (no eigenvalues needed)."""
        if self.m != 1 or self.p != 1:
            raise ValueError("to_tf is for SISO systems; use transfer_matrix for MIMO")
        A, B, C, D = self.A, self.B, self.C, self.D
        n = self.n
        den = np.zeros(n + 1)
        den[0] = 1.0
        Rk = np.zeros((n, n))  # adj(sI - A) = sum_k R_k s^{n-1-k}
        num = np.zeros(n + 1)
        N = np.eye(n)  # R_0
        for k in range(n):
            Rk = N
            num[k + 1] = (C @ Rk @ B)[0, 0]
            ck = -np.trace(A @ Rk) / (k + 1)
            den[k + 1] = ck
            N = A @ Rk + ck * np.eye(n)
        num = pl.add(num, D[0, 0] * den)
        return TransferFunction(num, den, self.dt).cancel()

    def to_ss(self) -> StateSpace:
        return self

    def __repr__(self):
        kind = f"discrete (dt = {self.dt:g})" if self.discrete else "continuous"
        return f"StateSpace({kind}, n = {self.n}, inputs = {self.m}, outputs = {self.p})"


@dataclass
class TransferFunction:
    """num(s)/den(s) with coefficients in descending powers; proper (deg num <= deg den)."""

    num: np.ndarray
    den: np.ndarray
    dt: float = 0.0

    def __post_init__(self):
        self.num, self.den = pl.trim(self.num), pl.trim(self.den)
        if len(self.num) > len(self.den):
            raise ValueError("improper transfer function (numerator degree exceeds denominator degree)")
        lead = self.den[0]
        self.num, self.den = self.num / lead, self.den / lead

    @property
    def discrete(self) -> bool:
        return self.dt > 0

    @property
    def order(self) -> int:
        return len(self.den) - 1

    def poles(self) -> np.ndarray:
        return pl.roots(self.den)

    def zeros(self) -> np.ndarray:
        return pl.roots(self.num)

    def dcgain(self) -> float:
        s0 = 1.0 if self.discrete else 0.0
        return float(pl.evaluate(self.num, s0) / pl.evaluate(self.den, s0))

    def __call__(self, s):
        """G(s) for scalar or array s (complex allowed)."""
        return pl.evaluate(self.num, s) / pl.evaluate(self.den, s)

    def cancel(self, tol: float = 1e-8) -> TransferFunction:
        """Remove pole-zero pairs that coincide within tol (relative to the larger magnitude)."""
        z, p = list(self.zeros()), list(self.poles())
        kept_z = []
        for zk in z:
            hit = [i for i, pk in enumerate(p) if abs(zk - pk) <= tol * max(1.0, abs(zk), abs(pk))]
            if hit:
                p.pop(hit[0])
            else:
                kept_z.append(zk)
        gain = self.num[0] if len(self.num) == len(self.den) else self.num[0]
        num = pl.from_roots(kept_z, gain) if len(kept_z) else np.array([gain])
        den = pl.from_roots(p) if len(p) else np.array([1.0])
        if np.iscomplexobj(num) or np.iscomplexobj(den):
            num, den = np.real(num), np.real(den)
        return TransferFunction(num, den, self.dt)

    def to_ss(self, form: str = "controllable") -> StateSpace:
        """Controllable or observable canonical realisation."""
        n = self.order
        den, num = self.den, np.r_[np.zeros(n + 1 - len(self.num)), self.num]
        d = num[0]
        b = num[1:] - d * den[1:]  # strictly proper part's numerator coefficients (descending)
        A = np.zeros((n, n))
        A[0, :] = -den[1:]
        A[1:, :-1] = np.eye(n - 1)
        B = np.zeros((n, 1))
        B[0, 0] = 1.0
        C = b.reshape(1, n)
        D = np.array([[d]])
        if form == "observable":
            return StateSpace(A.T, C.T, B.T, D, self.dt)
        return StateSpace(A, B, C, D, self.dt)

    def to_tf(self) -> TransferFunction:
        return self

    def __repr__(self):
        var = "z" if self.discrete else "s"
        return f"TransferFunction({_poly_str(self.num, var)} / {_poly_str(self.den, var)}" + (
            f", dt = {self.dt:g})" if self.discrete else ")"
        )


def _poly_str(c, var):
    terms = []
    n = len(c) - 1
    for k, ck in enumerate(c):
        if abs(ck) < 1e-12:
            continue
        pw = n - k
        coef = f"{ck:.4g}" if pw == 0 or abs(ck - 1) > 1e-12 else ""
        terms.append(coef + (f"{var}^{pw}" if pw > 1 else var if pw == 1 else ""))
    return " + ".join(terms).replace("+ -", "- ") or "0"


# ------------------------------------------------------------------ constructors and conversions
def ss(A, B, C, D=0.0, dt: float = 0.0) -> StateSpace:
    """State-space constructor (A, B, C, D, dt)."""
    return StateSpace(A, B, C, D, dt)


def tf(num, den, dt: float = 0.0) -> TransferFunction:
    """Transfer-function constructor from numerator and denominator coefficients (descending powers)."""
    return TransferFunction(num, den, dt)


def zpk(zeros, poles, gain: float, dt: float = 0.0) -> TransferFunction:
    """Transfer function from zeros, poles and gain: gain * prod(s - z) / prod(s - p)."""
    num = pl.from_roots(zeros, gain) if len(np.atleast_1d(zeros)) else np.array([gain])
    return TransferFunction(np.real(num), np.real(pl.from_roots(poles)), dt)


def modal_form(sys_: StateSpace) -> tuple:
    """Diagonalise A (distinct eigenvalues): returns (modal system, T). Complex pairs are kept as real 2 x 2 blocks."""
    lam, V = np.linalg.eig(sys_.A)
    T = np.zeros((sys_.n, sys_.n))
    cols = []
    i = 0
    used = set()
    while i < sys_.n:
        if i in used:
            i += 1
            continue
        if abs(lam[i].imag) < 1e-10:
            cols.append(V[:, i].real)
            used.add(i)
        else:
            j = next(k for k in range(sys_.n) if k not in used and k != i and abs(lam[k] - np.conj(lam[i])) < 1e-8)
            cols += [V[:, i].real, V[:, i].imag]
            used.update({i, j})
        i += 1
    T = np.column_stack(cols)
    return sys_.transform(T), T


def series(g1, g2):
    """g2 * g1: the output of g1 feeds g2 (SISO transfer functions, or state-space systems of any size)."""
    if isinstance(g1, TransferFunction) and isinstance(g2, TransferFunction):
        return TransferFunction(pl.mul(g1.num, g2.num), pl.mul(g1.den, g2.den), g1.dt).cancel()
    s1, s2 = g1.to_ss(), g2.to_ss()
    A = np.block([[s1.A, np.zeros((s1.n, s2.n))], [s2.B @ s1.C, s2.A]])
    B = np.vstack([s1.B, s2.B @ s1.D])
    C = np.hstack([s2.D @ s1.C, s2.C])
    D = s2.D @ s1.D
    return StateSpace(A, B, C, D, s1.dt)


def parallel(g1, g2):
    """Sum g1 + g2 of two systems (transfer functions or state-space systems)."""
    if isinstance(g1, TransferFunction) and isinstance(g2, TransferFunction):
        return TransferFunction(
            pl.add(pl.mul(g1.num, g2.den), pl.mul(g2.num, g1.den)), pl.mul(g1.den, g2.den), g1.dt
        ).cancel()
    s1, s2 = g1.to_ss(), g2.to_ss()
    A = np.block([[s1.A, np.zeros((s1.n, s2.n))], [np.zeros((s2.n, s1.n)), s2.A]])
    return StateSpace(A, np.vstack([s1.B, s2.B]), np.hstack([s1.C, s2.C]), s1.D + s2.D, s1.dt)


def feedback(g, h=None, sign: int = -1):
    """Closed loop g / (1 - sign * g h): negative feedback by default (sign = -1); h = None means unity."""
    if isinstance(g, TransferFunction):
        h = h if h is not None else TransferFunction([1.0], [1.0], g.dt)
        num = pl.mul(g.num, h.den)
        den = pl.sub(pl.mul(g.den, h.den), sign * pl.mul(g.num, h.num))
        return TransferFunction(num, den, g.dt).cancel()
    s1 = g.to_ss()
    s2 = (
        h.to_ss()
        if h is not None
        else StateSpace(np.zeros((0, 0)), np.zeros((0, s1.p)), np.zeros((s1.m, 0)), np.eye(s1.m), s1.dt)
    )
    # u = r + sign * y2, y2 = C2 x2 + D2 y1, y1 = C1 x1 + D1 u  (assumes well-posed: I - sign D1 D2 invertible)
    E = np.linalg.inv(np.eye(s1.m) - sign * s1.D @ s2.D)
    A = np.block(
        [
            [s1.A + sign * s1.B @ E @ s2.D @ s1.C, sign * s1.B @ E @ s2.C],
            [s2.B @ (s1.C + sign * s1.D @ E @ s2.D @ s1.C), s2.A + sign * s2.B @ s1.D @ E @ s2.C],
        ]
    )
    B = np.vstack([s1.B @ E, s2.B @ s1.D @ E])
    C = np.hstack([s1.C + sign * s1.D @ E @ s2.D @ s1.C, sign * s1.D @ E @ s2.C])
    D = s1.D @ E
    if s2.n == 0:
        A, B, C = A[: s1.n, : s1.n], B[: s1.n], C[:, : s1.n]
    return StateSpace(A, B, C, D, s1.dt)


def controllability_matrix(sys_: StateSpace) -> np.ndarray:
    """Controllability matrix [B, AB, ..., A^(n-1) B] of a state-space system."""
    A, B = sys_.A, sys_.B
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(sys_.n)])


def observability_matrix(sys_: StateSpace) -> np.ndarray:
    """Observability matrix [C; CA; ...; C A^(n-1)] of a state-space system."""
    A, C = sys_.A, sys_.C
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(sys_.n)])


def minimal(sys_: StateSpace, tol: float = 1e-8) -> StateSpace:
    """Minimal realisation: remove the uncontrollable and then the unobservable parts by orthogonal staircase
    decompositions (Kalman decomposition in two steps)."""
    s = _remove_uncontrollable(sys_, tol)
    sT = StateSpace(
        s.A.T, s.C.T, s.B.T, s.D.T, s.dt
    )  # the dual: unobservable modes of s are uncontrollable modes of the dual
    sT = _remove_uncontrollable(sT, tol)
    return StateSpace(sT.A.T, sT.C.T, sT.B.T, sT.D.T, s.dt)


def _remove_uncontrollable(sys_: StateSpace, tol: float) -> StateSpace:
    Wc = controllability_matrix(sys_)
    U, sv, _ = np.linalg.svd(Wc, full_matrices=True)
    r = int(np.sum(sv > tol * max(sv[0], 1e-300))) if len(sv) else 0
    if r == sys_.n:
        return sys_
    T = U  # columns 0..r-1 span the controllable subspace
    A = T.T @ sys_.A @ T
    B = T.T @ sys_.B
    C = sys_.C @ T
    return StateSpace(A[:r, :r], B[:r], C[:, :r], sys_.D, sys_.dt)


def balanced_realisation(sys_: StateSpace) -> tuple:
    """Balanced form (equal diagonal Gramians) of a stable minimal system: returns (system, Hankel singular values)."""
    from .linalg import dgramians, gramians

    Wc, Wo = (dgramians if sys_.discrete else gramians)(sys_.A, sys_.B, sys_.C)
    Lc = np.linalg.cholesky(Wc + 1e-15 * np.eye(sys_.n))
    U, s, Vt = np.linalg.svd(Lc.T @ Wo @ Lc)
    T = Lc @ U @ np.diag(s**-0.25)
    return sys_.transform(T), np.sqrt(s)
