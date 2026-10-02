"""Discrete-time control: discretisation of continuous systems (zero-order hold by the matrix exponential,
Tustin's bilinear transform with optional frequency prewarping, matched pole-zero mapping, forward and backward
Euler), the inverse (d2c by the matrix logarithm), sampling and aliasing, discrete PID forms, dead-beat
control, and sample-rate guidance.

>>> import numpy as np
>>> from englincontrol import systems as sy, discrete as dc
>>> Gd = dc.c2d(sy.tf([1.0], [1.0, 1.0]), 0.1)              # ZOH of 1/(s + 1)
>>> np.round(Gd.den, 6).tolist(), round(float(Gd.dcgain()), 10)
([1.0, -0.904837], 1.0)
"""

from __future__ import annotations

import numpy as np

from . import polynomials as pl
from .linalg import expm_with_integral
from .systems import StateSpace, TransferFunction


def c2d(sys_, dt: float, method: str = "zoh", w_prewarp: float | None = None):
    """Discretise with sampling period dt: 'zoh' (exact for piecewise-constant inputs), 'tustin' (bilinear,
    s = (2/dt)(z - 1)/(z + 1), optionally prewarped so that w_prewarp maps exactly), 'matched' (poles and zeros
    z = e^{s dt}, dc gain matched; SISO transfer functions), 'euler' (forward), 'backward'. Returns the same
    kind of object as given."""
    tf_in = isinstance(sys_, TransferFunction)
    if method == "matched":
        G = sys_ if tf_in else sys_.to_tf()
        zd = np.exp(G.zeros() * dt)
        pd = np.exp(G.poles() * dt)
        rel = len(G.den) - len(G.num)
        zd = np.r_[
            zd, -np.ones(max(rel - 1, 0))
        ]  # excess poles: zeros at z = -1 (all but one) keep the delay structure
        num, den = np.real(pl.from_roots(zd)), np.real(pl.from_roots(pd))
        gain = G.dcgain() * pl.evaluate(den, 1.0) / pl.evaluate(num, 1.0)
        return TransferFunction(gain * num, den, dt)
    s = sys_.to_ss()
    A, B, C, D = s.A, s.B, s.C, s.D
    n = s.n
    if method == "zoh":
        Ad, Bd = expm_with_integral(A, B, dt)
        Cd, Dd = C, D
    elif method in ("tustin", "prewarp"):
        k = 2.0 / dt if w_prewarp is None else w_prewarp / np.tan(w_prewarp * dt / 2)
        M = np.linalg.inv(k * np.eye(n) - A)
        Ad = M @ (k * np.eye(n) + A)
        Bd = M @ B * np.sqrt(2 * k)
        Cd = np.sqrt(2 * k) * C @ M
        Dd = D + C @ M @ B
    elif method == "euler":
        Ad, Bd, Cd, Dd = np.eye(n) + dt * A, dt * B, C, D
    elif method == "backward":
        M = np.linalg.inv(np.eye(n) - dt * A)
        Ad, Bd, Cd, Dd = M, M @ B * dt, C @ M, D + C @ M @ B * dt
    else:
        raise ValueError(method)
    out = StateSpace(Ad, Bd, Cd, Dd, dt)
    return out.to_tf() if tf_in else out


def logm(M) -> np.ndarray:
    """Principal matrix logarithm via the eigendecomposition (diagonalisable M with no eigenvalue on the
    negative real axis), returned real when M is real."""
    lam, V = np.linalg.eig(np.asarray(M, complex))
    L = V @ np.diag(np.log(lam)) @ np.linalg.inv(V)
    return L.real if np.allclose(L.imag, 0, atol=1e-9 * max(1.0, np.max(np.abs(L)))) else L


def d2c(sys_: StateSpace) -> StateSpace:
    """Inverse of the zero-order hold: A = log(Ad)/dt, B = A (Ad - I)^-1 Bd."""
    s = sys_.to_ss()
    A = logm(s.A) / s.dt
    B = A @ np.linalg.solve(s.A - np.eye(s.n), s.B)
    return StateSpace(A, B, s.C, s.D, 0.0)


# ------------------------------------------------------------------ sampling
def nyquist_frequency(dt: float) -> float:
    """pi/dt, the highest frequency a sampling period dt can represent."""
    return np.pi / dt


def alias(w: float, dt: float) -> float:
    """Apparent frequency (rad/s) of a sinusoid at w after sampling with period dt."""
    ws = 2 * np.pi / dt
    r = w % ws
    return min(r, ws - r)


def sample_rate_rules(w_bandwidth: float) -> dict:
    """Common rules for the sampling period from the closed-loop bandwidth: w_s between 10 and 30 times
    w_b (Astrom & Wittenmark: w_b dt in 0.2 - 0.6)."""
    return {"dt_max": 0.6 / w_bandwidth, "dt_min": 0.2 / w_bandwidth, "dt_recommended": 0.3 / w_bandwidth}


def zoh_frequency_response(w, dt: float):
    """Frequency response of the zero-order hold (1 - e^{-s dt})/(s dt): magnitude sinc, phase -w dt/2."""
    w = np.asarray(w, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(w == 0, 1.0 + 0j, (1 - np.exp(-1j * w * dt)) / (1j * w * dt))
    return H


# ------------------------------------------------------------------ discrete controllers
def pid_discrete(
    Kp: float, Ki: float = 0.0, Kd: float = 0.0, dt: float = 0.01, N: float = 10.0, method: str = "backward"
) -> TransferFunction:
    """Discrete PID: integral by the backward rectangle (Ki dt z/(z - 1)) or by Tustin (Ki dt/2 (z + 1)/(z - 1));
    derivative with the filter Tf = Kd/(Kp N) discretised by the backward rule, Kd (1 - a)/dt (z - 1)/(z - a) with
    a = Tf/(Tf + dt)."""
    z1 = np.array([1.0, 0.0])  # z
    one = np.array([1.0])
    num, den = Kp * one, one
    # integral
    if Ki:
        if method == "tustin":
            inum, iden = Ki * dt / 2 * np.array([1.0, 1.0]), np.array([1.0, -1.0])
        else:
            inum, iden = Ki * dt * z1, np.array([1.0, -1.0])
        num, den = pl.add(pl.mul(num, iden), pl.mul(inum, den)), pl.mul(den, iden)
    if Kd:
        Tf = Kd / (Kp * N) if Kp else dt
        a = Tf / (Tf + dt)  # backward-difference filtered derivative: Kd (1 - a) (z - 1)/(dt (z - a))
        dnum, dden = Kd * (1 - a) / dt * np.array([1.0, -1.0]), np.array([1.0, -a])
        num, den = pl.add(pl.mul(num, dden), pl.mul(dnum, den)), pl.mul(den, dden)
    return TransferFunction(num, den, dt)


def deadbeat(A, B) -> np.ndarray:
    """State feedback placing every closed-loop pole at z = 0: the state reaches zero in at most n steps."""
    from .statespace import place

    A = np.asarray(A, float)
    return place(A, B, np.zeros(len(A)))


def discrete_step_info(y, dt: float, settle_fraction: float = 0.02) -> dict:
    """Step characteristics of a sampled response y[k] with sampling period dt (see response.step_info)."""
    from .response import step_info

    return step_info(np.arange(len(y)) * dt, y, settle_fraction)
