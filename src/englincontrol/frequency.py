"""Frequency response: G(j omega) for transfer functions and state-space systems (continuous or discrete), Bode
magnitude and unwrapped phase, Nyquist curves, Nichols coordinates, gain and phase margins with their crossover
frequencies (all crossings found, the critical ones reported), delay margin, bandwidth and resonant peak, the
Pade approximation of a time delay, and the H-infinity norm by the Boyd-Balakrishnan bisection on the
Hamiltonian (with a frequency-sweep cross-check).

>>> import numpy as np
>>> from englincontrol import systems as sy, frequency as fr
>>> G = sy.tf([1.0], [1.0, 1.0])
>>> round(float(abs(fr.freqresp(G, 1.0))), 6)       # |1/(1 + j)| = 1/sqrt 2
0.707107
"""

from __future__ import annotations

import numpy as np

from . import polynomials as pl
from .systems import StateSpace, TransferFunction


def freqresp(sys_, w, delay: float = 0.0) -> np.ndarray:
    """G(j w) (continuous) or G(e^{j w dt}) (discrete) for scalar or array w; SISO returns a complex array of
    w's shape, MIMO an array of shape (len(w), p, m). A pure delay e^{-s delay} may be appended."""
    scalar = np.ndim(w) == 0
    w = np.atleast_1d(np.asarray(w, float))
    if isinstance(sys_, TransferFunction):
        s = np.exp(1j * w * sys_.dt) if sys_.discrete else 1j * w
        out = pl.evaluate(sys_.num, s) / pl.evaluate(sys_.den, s)
    else:
        s = np.exp(1j * w * sys_.dt) if sys_.discrete else 1j * w
        n = sys_.n
        out = np.empty((len(w), sys_.p, sys_.m), complex)
        for k, sk in enumerate(s):
            out[k] = sys_.C @ np.linalg.solve(sk * np.eye(n) - sys_.A, sys_.B) + sys_.D
        if sys_.p == 1 and sys_.m == 1:
            out = out[:, 0, 0]
    if delay:
        out = out * np.exp(-1j * w * delay).reshape(w.shape + (1,) * (out.ndim - 1))
    return out[0] if scalar else out


def default_frequencies(sys_, n: int = 500) -> np.ndarray:
    """Logarithmic grid two decades below the slowest and above the fastest pole/zero (discrete: up to pi/dt)."""
    feats = np.abs(np.r_[_poles(sys_), _zeros(sys_)])
    feats = feats[feats > 1e-12]
    if sys_.dt > 0:
        return np.logspace(np.log10(max(feats.min() if len(feats) else 0.01, 1e-3)) - 2, np.log10(np.pi / sys_.dt), n)
    lo, hi = (feats.min(), feats.max()) if len(feats) else (0.1, 10.0)
    return np.logspace(np.log10(lo) - 2, np.log10(hi) + 2, n)


def _poles(sys_):
    return sys_.poles()


def _zeros(sys_):
    try:
        return sys_.zeros()
    except Exception:
        return np.array([])


def bode(sys_, w=None, delay: float = 0.0) -> tuple:
    """(w, magnitude in dB, phase in degrees, unwrapped)."""
    w = default_frequencies(sys_) if w is None else np.asarray(w, float)
    H = freqresp(sys_, w, delay)
    return w, 20 * np.log10(np.abs(H)), np.degrees(np.unwrap(np.angle(H)))


def nyquist(sys_, w=None, delay: float = 0.0) -> tuple:
    """(w, G(jw)) for positive w; the negative-frequency half is the conjugate."""
    w = default_frequencies(sys_) if w is None else np.asarray(w, float)
    return w, freqresp(sys_, w, delay)


def nichols(sys_, w=None, delay: float = 0.0) -> tuple:
    """(phase in degrees, magnitude in dB): the open-loop locus in Nichols coordinates."""
    w, mag, ph = bode(sys_, w, delay)
    return ph, mag


def _refine_crossing(f, a, b, iters: int = 60):
    fa = f(a)
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def margins(L, w=None, delay: float = 0.0) -> dict:
    """Gain margin (ratio and dB) at the phase crossover(s) where arg L = -180 degrees, phase margin (degrees)
    at the gain crossover(s) where |L| = 1, their frequencies, and the delay margin (phase margin / gain
    crossover frequency, seconds). With several crossings the smallest margins are reported; all are listed."""
    w = default_frequencies(L, 2000) if w is None else np.asarray(w, float)
    H = freqresp(L, w, delay)
    mag = np.abs(H)
    ph = np.degrees(np.unwrap(np.angle(H)))
    # gain crossovers: |L| = 1
    g = np.log10(mag)
    gc = [
        _refine_crossing(lambda x: np.log10(abs(freqresp(L, x, delay))), w[i], w[i + 1])
        for i in np.flatnonzero(g[:-1] * g[1:] < 0)
    ]
    pms = []
    for wg in gc:
        phase = np.degrees(np.angle(freqresp(L, wg, delay)))
        pm = ((phase + 360.0) % 360.0) - 180.0  # phase + 180, mapped to (-180, 180]
        pms.append((wg, pm))
    # phase crossovers: arg L = -180 (mod 360)
    pc = []
    shifted = (ph + 180.0) / 360.0  # crossings where this passes an integer
    k = np.floor(shifted)
    for i in np.flatnonzero(np.diff(k) != 0):
        target = 360.0 * max(k[i], k[i + 1]) - 180.0

        def f(x, target=target):
            # continuous phase near the bracket: unwrap relative to the grid value
            p_ = np.degrees(np.angle(freqresp(L, x, delay)))
            return ((p_ - target + 180.0) % 360.0) - 180.0

        pc.append(_refine_crossing(f, w[i], w[i + 1]))
    gms = [(wp, 1.0 / abs(freqresp(L, wp, delay))) for wp in pc]
    out = {
        "gain_crossovers": [wg for wg, _ in pms],
        "phase_margins": [pm for _, pm in pms],
        "phase_crossovers": [wp for wp, _ in gms],
        "gain_margins": [gm for _, gm in gms],
    }
    if pms:
        i = int(np.argmin([abs(pm) for _, pm in pms]))
        out["phase_margin"], out["w_gc"] = pms[i][1], pms[i][0]
        out["delay_margin"] = np.radians(pms[i][1]) / pms[i][0] if pms[i][1] > 0 else 0.0
    else:
        out["phase_margin"], out["w_gc"], out["delay_margin"] = np.inf, np.nan, np.inf
    if gms:
        i = int(np.argmin([abs(np.log(gm)) for _, gm in gms]))
        out["gain_margin"], out["w_pc"] = gms[i][1], gms[i][0]
        out["gain_margin_db"] = 20 * np.log10(gms[i][1])
    else:
        out["gain_margin"], out["w_pc"], out["gain_margin_db"] = np.inf, np.nan, np.inf
    return out


def bandwidth(sys_, w=None, level_db: float = -3.0) -> float:
    """First frequency at which |G| falls to the dc gain + level_db (closed-loop bandwidth for a low-pass T)."""
    w = default_frequencies(sys_, 4000) if w is None else np.asarray(w, float)
    mag = 20 * np.log10(np.abs(freqresp(sys_, w)))
    target = 20 * np.log10(abs(sys_.dcgain() if isinstance(sys_, TransferFunction) else sys_.dcgain()[0, 0])) + level_db
    idx = np.flatnonzero((mag[:-1] >= target) & (mag[1:] < target))
    if len(idx) == 0:
        return np.inf
    i = idx[0]
    return _refine_crossing(lambda x: 20 * np.log10(abs(freqresp(sys_, x))) - target, w[i], w[i + 1])


def resonant_peak(sys_, w=None) -> tuple:
    """(peak magnitude |G|max, frequency) over the grid, refined by a golden-section search."""
    w = default_frequencies(sys_, 4000) if w is None else np.asarray(w, float)
    mag = np.abs(freqresp(sys_, w))
    i = int(np.argmax(mag))
    a, b = w[max(i - 1, 0)], w[min(i + 1, len(w) - 1)]
    f = lambda x: -abs(freqresp(sys_, x))  # noqa: E731
    invphi = (np.sqrt(5) - 1) / 2
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    for _ in range(80):
        if f(c) < f(d):
            b, d = d, c
            c = b - invphi * (b - a)
        else:
            a, c = c, d
            d = a + invphi * (b - a)
    wm = 0.5 * (a + b)
    return float(abs(freqresp(sys_, wm))), float(wm)


def pade(delay: float, order: int = 2) -> TransferFunction:
    """Pade approximant of e^{-s T}: ratio of polynomials of the given order, exact through order 2n in sT."""
    T = delay
    n = order
    from math import factorial

    num = np.zeros(n + 1)
    den = np.zeros(n + 1)
    for k in range(n + 1):
        c = factorial(2 * n - k) * factorial(n) / (factorial(2 * n) * factorial(k) * factorial(n - k))
        num[n - k] = c * (-T) ** k
        den[n - k] = c * T**k
    return TransferFunction(num, den)


def hinf_norm(sys_: StateSpace, tol: float = 1e-6, w_max: float = 1e6) -> tuple:
    """H-infinity norm (peak gain over frequency) of a stable continuous-time system by Boyd-Balakrishnan-Kabamba
    bisection: gamma is an upper bound iff the Hamiltonian [[A + B R^-1 D^T C, B R^-1 B^T],
    [-C^T (I + D R^-1 D^T) C, -(A + B R^-1 D^T C)^T]], R = gamma^2 I - D^T D, has no eigenvalue on the imaginary
    axis. Returns (norm, frequency of the peak from a sweep)."""
    A, B, C, D = sys_.A, sys_.B, sys_.C, sys_.D
    if np.max(np.linalg.eigvals(A).real) >= 0:
        return np.inf, np.nan
    w = np.logspace(-3, np.log10(w_max), 2000)
    H = freqresp(sys_, w)
    gains = np.linalg.norm(H.reshape(len(w), -1, 1) if H.ndim == 1 else H, 2, axis=(1, 2)) if H.ndim == 3 else np.abs(H)
    lo = max(float(gains.max()), np.linalg.norm(D, 2))
    hi = 2 * lo + 1e-12

    def has_imag_eig(gamma):
        R = gamma**2 * np.eye(B.shape[1]) - D.T @ D
        Ri = np.linalg.inv(R)
        Ah = A + B @ Ri @ D.T @ C
        Ham = np.block([[Ah, B @ Ri @ B.T], [-C.T @ (np.eye(C.shape[0]) + D @ Ri @ D.T) @ C, -Ah.T]])
        lam = np.linalg.eigvals(Ham)
        return bool(np.any(np.abs(lam.real) < 1e-8 * max(1.0, np.max(np.abs(lam))))), lam

    while has_imag_eig(hi)[0] and hi < 1e12:  # make sure the upper end really is an upper bound
        hi *= 2
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if has_imag_eig(mid)[0]:
            lo = mid
        else:
            hi = mid
    gamma = 0.5 * (lo + hi)
    w_peak = float(w[int(np.argmax(gains))])
    return float(gamma), w_peak
