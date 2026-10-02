"""Stability tests: the Routh array (with the epsilon and auxiliary-polynomial special cases) and Hurwitz
determinants for continuous polynomials, the Jury table for discrete ones, Lyapunov's test on the state matrix,
the root locus as a function of a gain (with branches tracked, asymptotes, real-axis segments and breakaway
points from the rules), and the Nyquist stability criterion as a computation of encirclements, with the
indentation around poles on the imaginary axis.

>>> from englincontrol import stability as st
>>> st.routh([1, 2, 3, 4])["rhp_roots"]        # s^3 + 2s^2 + 3s + 4: Routh says stable
0
"""

from __future__ import annotations

import numpy as np

from . import polynomials as pl
from .systems import TransferFunction


def routh(c, eps: float = 1e-9) -> dict:
    """Routh array of a polynomial (descending coefficients). A zero first entry is replaced by eps; a row of
    zeros by the derivative of the auxiliary polynomial (whose roots are symmetric about the origin). Returns the
    table, the number of sign changes in the first column (= roots in the right half-plane), and the auxiliary
    polynomial if one occurred."""
    c = pl.trim(c)
    n = len(c) - 1
    cols = (n + 2) // 2
    T = np.zeros((n + 1, cols))
    T[0, : len(c[0::2])] = c[0::2]
    T[1, : len(c[1::2])] = c[1::2]
    aux = None
    for i in range(2, n + 1):
        if np.all(np.abs(T[i - 1]) < 1e-300):  # row of zeros: auxiliary polynomial from the row above
            order = n - (i - 2)
            coeffs = []
            for j in range(cols):
                if order - 2 * j >= 0:
                    coeffs += [T[i - 2, j], 0.0]
            aux = pl.trim(np.array(coeffs[: order + 1]))
            d = pl.derivative(aux)
            T[i - 1, : len(d[0::2])] = d[0::2]
        if abs(T[i - 1, 0]) < 1e-300:
            T[i - 1, 0] = eps
        for j in range(cols - 1):
            T[i, j] = (T[i - 1, 0] * T[i - 2, j + 1] - T[i - 2, 0] * T[i - 1, j + 1]) / T[i - 1, 0]
    first = T[:, 0]
    signs = np.sign(first[np.abs(first) > 0])
    changes = int(np.sum(signs[1:] != signs[:-1]))
    return {"table": T, "rhp_roots": changes, "stable": changes == 0 and aux is None, "auxiliary": aux}


def hurwitz(c) -> dict:
    """Hurwitz determinants: stable iff all leading principal minors of the Hurwitz matrix are positive."""
    c = pl.trim(c)
    n = len(c) - 1
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k = 2 * j - i + 1
            if 0 <= k <= n:
                H[i, j] = c[k]
    minors = [np.linalg.det(H[: k + 1, : k + 1]) for k in range(n)]
    return {"matrix": H, "minors": np.array(minors), "stable": bool(np.all(np.array(minors) > 0)) and c[0] > 0}


def jury(c) -> dict:
    """Jury's stability test for a discrete characteristic polynomial P(z): the necessary conditions P(1) > 0,
    (-1)^n P(-1) > 0, |a_n| < |a_0|, and the table conditions. Returns the table rows and stable (all roots
    inside the unit circle)."""
    a = pl.trim(c)
    n = len(a) - 1
    conds = [pl.evaluate(a, 1.0) > 0, (-1) ** n * pl.evaluate(a, -1.0) > 0, abs(a[-1]) < abs(a[0])]
    rows = [a.copy()]
    row = a.copy()
    while len(row) > 3:
        b = np.array([row[0] * row[k] - row[-1] * row[-1 - k] for k in range(len(row) - 1)])
        rows.append(b)
        conds.append(abs(b[-1]) < abs(b[0]))
        row = b
    return {"rows": rows, "conditions": conds, "stable": bool(np.all(conds))}


def lyapunov_test(A) -> dict:
    """Solve A^T P + P A = -I; the system is asymptotically stable iff P is positive definite."""
    from .linalg import lyapunov

    A = np.asarray(A, float)
    P = lyapunov(A.T, np.eye(len(A)))
    eig = np.linalg.eigvalsh(0.5 * (P + P.T))
    return {"P": P, "eigenvalues": eig, "stable": bool(np.all(eig > 0))}


def is_stable(sys_) -> bool:
    """True if every pole has negative real part (continuous) or lies inside the unit circle (discrete)."""
    lam = sys_.poles()
    return bool(np.all(np.abs(lam) < 1) if sys_.dt > 0 else np.all(lam.real < 0))


# ------------------------------------------------------------------ root locus
def root_locus(G: TransferFunction, gains=None) -> dict:
    """Closed-loop poles of 1 + K G(s) = 0 for each gain K (branches ordered by nearest-neighbour continuation),
    with the rules: asymptote centroid and angles, real-axis segments, breakaway/break-in points (roots of
    N'D - ND' on the real axis with positive K), and the departure angles from complex poles."""
    num, den = G.num, G.den
    gains = np.logspace(-3, 3, 600) if gains is None else np.asarray(gains, float)
    n, m = pl.degree(den), pl.degree(num)
    branches = np.zeros((len(gains), n), complex)
    prev = None
    for i, K in enumerate(gains):
        r = pl.roots(pl.add(den, K * num))
        if prev is None:
            r = r[np.argsort(r.real)]
        else:  # match to the previous roots (greedy assignment)
            order = []
            avail = list(range(n))
            for pk in prev:
                j = min(avail, key=lambda j: abs(r[j] - pk))
                order.append(j)
                avail.remove(j)
            r = r[order]
        branches[i] = r
        prev = r
    poles, zeros = pl.roots(den), pl.roots(num)
    q = n - m
    centroid = (np.sum(poles) - np.sum(zeros)).real / q if q > 0 else np.nan
    angles = np.degrees((2 * np.arange(q) + 1) * np.pi / q) if q > 0 else np.array([])
    dK = pl.sub(pl.mul(pl.derivative(num), den), pl.mul(num, pl.derivative(den)))  # d/ds (D/N) = 0  <=>  N'D - ND' = 0
    cand = pl.roots(dK)
    cand = cand[np.abs(cand.imag) < 1e-8].real
    Kc = -pl.evaluate(den, cand) / pl.evaluate(num, cand)
    breakaway = cand[(Kc.real > 0) & (np.abs(Kc.imag) < 1e-8)]
    # real-axis segments: points with an odd number of real poles+zeros to the right
    crit = np.sort(np.r_[poles[np.abs(poles.imag) < 1e-9].real, zeros[np.abs(zeros.imag) < 1e-9].real])
    segments = []
    for a, b in zip(crit[:-1], crit[1:]):
        if np.sum(crit > 0.5 * (a + b)) % 2 == 1:
            segments.append((a, b))
    if len(crit) and np.sum(crit > crit[0] - 1) % 2 == 1:
        segments.append((-np.inf, crit[0]))
    departure = {}
    for pk in poles[poles.imag > 1e-9]:
        ang = 180.0 + np.degrees(np.sum(np.angle(pk - zeros)) - np.sum(np.angle(pk - poles[np.abs(poles - pk) > 1e-9])))
        departure[complex(np.round(pk, 6))] = ((ang + 180.0) % 360.0) - 180.0
    return {
        "gains": gains,
        "branches": branches,
        "poles": poles,
        "zeros": zeros,
        "centroid": centroid,
        "asymptote_angles": angles,
        "breakaway": breakaway,
        "real_axis_segments": segments,
        "departure_angles": departure,
    }


def gain_for_damping(G: TransferFunction, zeta: float, gains=None) -> tuple:
    """Smallest gain at which a branch of the root locus crosses the damping-ratio line zeta (K, pole)."""
    rl = root_locus(G, gains)
    for K, r in zip(rl["gains"], rl["branches"]):
        cplx = r[np.abs(r.imag) > 1e-9]
        if len(cplx):
            z = -cplx.real / np.abs(cplx)
            if np.any(z <= zeta):
                i = int(np.argmin(z))
                return float(K), cplx[i]
    return np.nan, np.nan


# ------------------------------------------------------------------ Nyquist criterion
def _segment(lo: float, hi: float, n: int, floor: float = 1e-4) -> np.ndarray:
    """Points from lo to hi on the imaginary axis, logarithmically dense near the origin on either side."""
    parts = []
    if lo < -floor:
        parts.append(-np.logspace(np.log10(-lo), np.log10(floor if hi > -floor else -hi), n))
    if lo < floor and hi > -floor:
        parts.append(np.linspace(max(lo, -floor), min(hi, floor), n // 4))
    if hi > floor:
        parts.append(np.logspace(np.log10(floor if lo < floor else lo), np.log10(hi), n))
    seg = np.concatenate(parts)
    return seg[(seg >= lo) & (seg <= hi)]


def nyquist_criterion(L: TransferFunction, w_max: float = 1e4, eps: float = 1e-3, n: int = 20000) -> dict:
    """Count the encirclements N of -1 by L(s) along the Nyquist contour (the imaginary axis from -j w_max to
    +j w_max, indented by small semicircles into the right half-plane around poles on the axis), P = open-loop
    poles in the right half-plane; the closed-loop right-half-plane poles are Z = N + P. Returns N, P, Z and the
    actual count from the closed-loop characteristic polynomial for comparison. If closed-loop poles lie on the
    imaginary axis the contour passes through zeros of 1 + L and the count is meaningless: 'marginal' is set."""
    poles = L.poles()
    axis_poles = np.unique(
        np.round(poles[np.abs(poles.real) < 1e-9].imag, 9)
    )  # distinct axis poles, each indented once
    pts = []
    # build the contour as s values: up the imaginary axis with indentations
    breakpoints = [-w_max] + [v for v in axis_poles] + [w_max]
    for a, b in zip(breakpoints[:-1], breakpoints[1:]):
        lo, hi = a + (eps if a != -w_max else 0.0), b - (eps if b != w_max else 0.0)
        pts.append(1j * _segment(lo, hi, n // 4))
        if b != w_max:  # semicircle around the pole at j b into the RHP: theta from -90 to +90
            th = np.linspace(-np.pi / 2, np.pi / 2, n // 20)
            pts.append(1j * b + eps * np.exp(1j * th))
    s = np.concatenate(pts)
    Ls = pl.evaluate(L.num, s) / pl.evaluate(L.den, s)
    phase = np.unwrap(np.angle(1 + Ls))
    N = int(
        np.round((phase[-1] - phase[0]) / (2 * np.pi))
    )  # counterclockwise is positive; clockwise encirclements of -1 are -N
    N_cw = -N
    P = int(np.sum(poles.real > 1e-9))
    Z = N_cw + P
    cl = pl.roots(pl.add(L.den, L.num))
    marginal = bool(np.any(np.abs(cl.real) < 1e-7))  # closed-loop poles on the contour: the criterion does not apply
    return {
        "marginal": marginal,
        "N_clockwise": N_cw,
        "P": P,
        "Z": Z,
        "closed_loop_rhp_poles": int(np.sum(cl.real > 1e-9)),
        "contour": s,
        "L": Ls,
    }
