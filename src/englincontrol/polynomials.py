"""Polynomial algebra for transfer functions: coefficients in descending powers (NumPy's convention), products,
sums, division with remainder, evaluation by Horner's scheme (real or complex argument), roots as the
eigenvalues of the companion matrix, and construction from roots.

>>> from englincontrol import polynomials as pl
>>> pl.mul([1, 1], [1, 2])        # (s + 1)(s + 2)
array([1., 3., 2.])
"""

from __future__ import annotations

import numpy as np


def trim(c) -> np.ndarray:
    """Remove leading zeros (keep at least one coefficient)."""
    c = np.atleast_1d(np.asarray(c, float))
    nz = np.flatnonzero(np.abs(c) > 0)
    return c[nz[0] :] if len(nz) else np.array([0.0])


def mul(a, b) -> np.ndarray:
    """Product of two polynomials (coefficients in descending powers)."""
    return trim(np.convolve(trim(a), trim(b)))


def add(a, b) -> np.ndarray:
    """Sum of two polynomials of possibly different degree."""
    a, b = trim(a), trim(b)
    n = max(len(a), len(b))
    return trim(np.r_[np.zeros(n - len(a)), a] + np.r_[np.zeros(n - len(b)), b])


def sub(a, b) -> np.ndarray:
    """Difference a - b of two polynomials."""
    return add(a, -np.asarray(b, float))


def div(a, b) -> tuple:
    """Polynomial long division a = q b + r: returns (q, r)."""
    a, b = trim(a), trim(b)
    if len(a) < len(b):
        return np.array([0.0]), a
    q = np.zeros(len(a) - len(b) + 1)
    r = a.copy()
    for i in range(len(q)):
        q[i] = r[i] / b[0]
        r[i : i + len(b)] -= q[i] * b
    return trim(q), trim(r[len(q) - 1 + 1 :]) if len(b) > 1 else np.array([0.0])


def evaluate(c, s):
    """p(s) by Horner's scheme; s may be an array (real or complex)."""
    c = trim(c)
    s = np.asarray(s)
    out = np.zeros(s.shape, dtype=np.result_type(s, float)) + c[0]
    for ck in c[1:]:
        out = out * s + ck
    return out


def roots(c) -> np.ndarray:
    """Roots as eigenvalues of the companion matrix (what numpy.roots does)."""
    c = trim(c)
    n = len(c) - 1
    if n < 1:
        return np.array([])
    C = np.zeros((n, n))
    C[0, :] = -c[1:] / c[0]
    C[1:, :-1] = np.eye(n - 1)
    return np.linalg.eigvals(C)


def from_roots(r, leading: float = 1.0) -> np.ndarray:
    """Monic polynomial with the given roots (complex-conjugate pairs give real coefficients)."""
    c = np.array([1.0 + 0j])
    for rk in np.atleast_1d(r):
        c = np.convolve(c, [1.0, -rk])
    c = leading * c
    return c.real if np.allclose(c.imag, 0, atol=1e-10 * max(1.0, np.max(np.abs(c)))) else c


def derivative(c) -> np.ndarray:
    """Derivative of a polynomial."""
    c = trim(c)
    n = len(c) - 1
    return trim(c[:-1] * np.arange(n, 0, -1)) if n >= 1 else np.array([0.0])


def degree(c) -> int:
    """Degree of a polynomial after removing leading zeros."""
    return len(trim(c)) - 1
