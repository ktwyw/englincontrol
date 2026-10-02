"""Time-domain responses of LTI systems: the exact solution on a time grid by the matrix exponential (zero-order
hold on the input between samples), step, impulse, initial-condition and arbitrary-input responses, discrete-time
recursions, and the characteristics of a step response (rise time, peak, overshoot, settling time, steady
state) together with the second-order formulas they are compared with.

>>> import numpy as np
>>> from englincontrol import systems as sy, response as rs
>>> t, y = rs.step(sy.tf([1.0], [1.0, 1.0]), t_end=5.0, n=501)
>>> bool(abs(y[-1] - (1 - np.exp(-5.0))) < 1e-12)
True
"""

from __future__ import annotations

import numpy as np

from .linalg import expm_with_integral
from .systems import StateSpace, TransferFunction


def _ss(sys_):
    return sys_.to_ss() if isinstance(sys_, TransferFunction) else sys_


def lsim(sys_, u, t=None, x0=None) -> tuple:
    """Response to an input sequence u (shape (N,) or (N, m)) at times t (continuous: zero-order hold between
    samples, exact for piecewise-constant u; discrete: the recursion with t = k dt). Returns (t, y, x)."""
    s = _ss(sys_)
    u = np.asarray(u, float)
    u = u.reshape(-1, 1) if u.ndim == 1 else u
    N = len(u)
    if s.discrete:
        t = np.arange(N) * s.dt if t is None else np.asarray(t, float)
        Phi, Gam = s.A, s.B
    else:
        t = np.asarray(t, float)
        h = t[1] - t[0]
        if not np.allclose(np.diff(t), h):
            raise ValueError("lsim needs a uniform time grid")
        Phi, Gam = expm_with_integral(s.A, s.B, h)
    x = np.zeros((N, s.n))
    x[0] = np.zeros(s.n) if x0 is None else np.asarray(x0, float)
    for k in range(N - 1):
        x[k + 1] = Phi @ x[k] + Gam @ u[k]
    y = x @ s.C.T + u @ s.D.T
    return t, y.squeeze(), x


def step(sys_, t_end: float | None = None, n: int = 1001, input_index: int = 0) -> tuple:
    """Unit step response (t, y); t_end defaults to about 7 times the slowest time constant."""
    s = _ss(sys_)
    if s.discrete:
        N = int(round(t_end / s.dt)) + 1 if t_end else 100
        u = np.zeros((N, s.m))
        u[:, input_index] = 1.0
        t, y, _ = lsim(s, u)
        return t, y
    t_end = t_end if t_end is not None else _default_horizon(s)
    t = np.linspace(0, t_end, n)
    u = np.zeros((n, s.m))
    u[:, input_index] = 1.0
    t, y, _ = lsim(s, u, t)
    return t, y


def impulse(sys_, t_end: float | None = None, n: int = 1001) -> tuple:
    """Impulse response: the initial-condition response from x0 = B (continuous) plus the D term at t = 0."""
    s = _ss(sys_)
    if s.discrete:
        N = int(round(t_end / s.dt)) + 1 if t_end else 100
        u = np.zeros((N, s.m))
        u[0, 0] = 1.0 / s.dt
        t, y, _ = lsim(s, u)
        return t, y
    t_end = t_end if t_end is not None else _default_horizon(s)
    t = np.linspace(0, t_end, n)
    t, y, _ = lsim(s, np.zeros((n, s.m)), t, x0=s.B[:, 0])
    return t, y


def initial(sys_, x0, t_end: float | None = None, n: int = 1001) -> tuple:
    """Response from an initial state with zero input: (t, y, x)."""
    s = _ss(sys_)
    t_end = t_end if t_end is not None else _default_horizon(s)
    t = np.linspace(0, t_end, n) if not s.discrete else np.arange(int(round(t_end / s.dt)) + 1) * s.dt
    t, y, x = lsim(s, np.zeros((len(t), s.m)), t, x0=x0)
    return t, y, x


def _default_horizon(s: StateSpace) -> float:
    lam = s.poles()
    rates = np.abs(lam.real[np.abs(lam.real) > 1e-9])
    if len(rates) == 0:
        return 10.0
    return float(min(7.0 / rates.min(), 1e4 / rates.max()))


def step_info(t, y, settle_fraction: float = 0.02, final: float | None = None) -> dict:
    """Rise time (10 % to 90 % of the final value), peak time, overshoot (% of the final value), settling time
    (last time outside the +-2 % band), final value. `final` defaults to y[-1]; pass the dc gain when the
    response has not fully settled."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    yf = y[-1] if final is None else float(final)
    if abs(yf) < 1e-300:
        return {"rise_time": np.nan, "peak_time": np.nan, "overshoot": np.nan, "settling_time": np.nan, "final": 0.0}
    yn = y / yf
    t10 = t[np.argmax(yn >= 0.1)]
    t90 = t[np.argmax(yn >= 0.9)]
    ipk = int(np.argmax(yn)) if yf > 0 else int(np.argmin(yn))
    outside = np.abs(yn - 1) > settle_fraction
    if not np.any(outside):
        ts = t[0]
    elif np.max(np.flatnonzero(outside)) + 1 < len(t):
        ts = t[np.max(np.flatnonzero(outside)) + 1]
    else:
        ts = np.nan  # never settled within the record
    return {
        "rise_time": float(t90 - t10),
        "peak_time": float(t[ipk]),
        "overshoot": float(100 * max(yn[ipk] - 1, 0.0)),
        "settling_time": float(ts),
        "final": float(yf),
    }


def second_order(zeta: float, wn: float) -> dict:
    """Closed-form step-response characteristics of wn^2/(s^2 + 2 zeta wn s + wn^2) for 0 < zeta < 1."""
    wd = wn * np.sqrt(1 - zeta**2)
    return {
        "peak_time": np.pi / wd,
        "overshoot": 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)),
        "settling_time_2pct": 4 / (zeta * wn),
        "rise_time_approx": (1.76 * zeta**3 - 0.417 * zeta**2 + 1.039 * zeta + 1) / wn,
        "damped_frequency": wd,
    }


def dominant_poles(sys_, count: int = 2) -> np.ndarray:
    """The `count` poles closest to the imaginary axis (slowest) - the dominant-pole approximation's basis."""
    lam = _ss(sys_).poles()
    return lam[np.argsort(np.abs(lam.real))][:count]
