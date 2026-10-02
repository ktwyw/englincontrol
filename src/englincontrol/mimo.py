"""Multivariable basics: the transfer matrix of a state-space system, singular values (principal gains) across
frequency, the condition number, transmission zeros of square systems, the relative gain array and its reading
of interaction, static decoupling, and the Bristol pairing recommendation.

>>> import numpy as np
>>> from englincontrol import mimo
>>> np.round(mimo.rga(np.array([[2.0, 1.0], [1.0, 2.0]])), 4)
array([[ 1.3333, -0.3333],
       [-0.3333,  1.3333]])
"""

from __future__ import annotations

import numpy as np

from . import polynomials as pl
from .frequency import freqresp
from .systems import StateSpace, TransferFunction


def transfer_matrix(sys_: StateSpace) -> list:
    """p x m nested list of SISO transfer functions G_ij(s) (each by Leverrier-Faddeev on the (i, j) pair)."""
    out = []
    for i in range(sys_.p):
        row = []
        for j in range(sys_.m):
            row.append(
                StateSpace(sys_.A, sys_.B[:, [j]], sys_.C[[i], :], sys_.D[[i], [j]].reshape(1, 1), sys_.dt).to_tf()
            )
        out.append(row)
    return out


def sigma(sys_, w) -> np.ndarray:
    """Singular values of G(jw) for each frequency: shape (len(w), min(p, m)), largest first."""
    H = freqresp(sys_, np.atleast_1d(w))
    if H.ndim == 1:
        return np.abs(H).reshape(-1, 1)
    return np.linalg.svd(H, compute_uv=False)


def condition_number(sys_, w) -> np.ndarray:
    """sigma_max/sigma_min of G(jw) at each frequency: the direction-dependence of the gain."""
    sv = sigma(sys_, w)
    return sv[:, 0] / sv[:, -1]


def rga(G0) -> np.ndarray:
    """Relative gain array Lambda = G o (G^-1)^T (Bristol 1966) of a square gain matrix."""
    G0 = np.asarray(G0, float)
    return G0 * np.linalg.inv(G0).T


def rga_pairing(G0) -> dict:
    """Recommended input-output pairing: choose the pairs whose relative gains are closest to 1 (and positive);
    large or negative entries warn of strong interaction or sign reversal under control."""
    Lam = rga(G0)
    n = len(Lam)
    pairs = []
    used = set()
    for i in range(n):
        j = int(np.argmin([abs(Lam[i, jj] - 1) if jj not in used else np.inf for jj in range(n)]))
        pairs.append((i, j))
        used.add(j)
    return {
        "rga": Lam,
        "pairs": pairs,
        "max_abs": float(np.max(np.abs(Lam))),
        "negative_entries": bool(np.any(Lam < 0)),
    }


def static_decoupler(G0) -> np.ndarray:
    """D = G(0)^-1: the plant seen through it, G D, is the identity at dc."""
    return np.linalg.inv(np.asarray(G0, float))


def transmission_zeros(sys_: StateSpace) -> np.ndarray:
    """Zeros of a square system: roots of det G(s) computed from the transfer matrix (numerator of the
    determinant after cancellation of common factors) - adequate for the small systems of the course."""
    G = transfer_matrix(sys_)
    if sys_.p != sys_.m:
        raise ValueError("transmission zeros by the determinant need a square system")
    det = _det_tf(G)
    return det.zeros()


def _det_tf(G):
    n = len(G)
    if n == 1:
        return G[0][0]
    total = None
    for j in range(n):
        minor = [[G[i][k] for k in range(n) if k != j] for i in range(1, n)]
        term = _mul(G[0][j], _det_tf(minor))
        if j % 2:
            term = TransferFunction(-term.num, term.den, term.dt)
        total = term if total is None else _add(total, term)
    return total.cancel(tol=1e-6)  # roots of expanded products are less exact than 1e-8


def _mul(a: TransferFunction, b: TransferFunction) -> TransferFunction:
    return TransferFunction(pl.mul(a.num, b.num), pl.mul(a.den, b.den), a.dt)


def _add(a: TransferFunction, b: TransferFunction) -> TransferFunction:
    return TransferFunction(pl.add(pl.mul(a.num, b.den), pl.mul(b.num, a.den)), pl.mul(a.den, b.den), a.dt)
