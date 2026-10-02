"""Classical design: PID controllers in parallel and standard forms with a filtered derivative; tuning rules
(Ziegler-Nichols step and ultimate, Cohen-Coon, IMC / SIMC) for first-order-plus-dead-time models; a relay
feedback experiment that measures the ultimate gain and period by simulation; a discrete-time PID simulation
with actuator saturation and anti-windup (clamping or back-calculation); lead, lag and lead-lag compensators
designed from phase and gain requirements on the Bode plot; and model-inverse feedforward.

>>> from englincontrol import design as dg
>>> dg.ziegler_nichols_ultimate(2.0, 1.0)["PID"]       # (Kp, Ti, Td) = (0.6 Ku, Pu/2, Pu/8)
{'Kp': 1.2, 'Ti': 0.5, 'Td': 0.125}
"""

from __future__ import annotations

import numpy as np

from . import polynomials as pl
from .frequency import freqresp, margins
from .linalg import expm_with_integral
from .systems import TransferFunction, feedback, series


# ------------------------------------------------------------------ PID
def pid(Kp: float, Ki: float = 0.0, Kd: float = 0.0, N: float = 10.0) -> TransferFunction:
    """Parallel PID Kp + Ki/s + Kd s/(1 + s Tf) with the derivative filter time constant Tf = Kd/(Kp N) (N = 10
    by default; N = inf gives the ideal, improper derivative, only allowed when Kd = 0)."""
    if Kd == 0.0:
        return TransferFunction([Kp, Ki], [1.0, 0.0]) if Ki else TransferFunction([Kp], [1.0])
    Tf = Kd / (Kp * N) if Kp else 1.0 / N
    # (Kp s (1 + Tf s) + Ki (1 + Tf s) + Kd s^2) / (s (1 + Tf s))
    num = pl.add(pl.add(pl.mul([Kp, 0.0], [Tf, 1.0]), Ki * np.array([Tf, 1.0])), [Kd, 0.0, 0.0])
    den = pl.mul([1.0, 0.0], [Tf, 1.0])
    return TransferFunction(num, den)


def pid_standard(Kp: float, Ti: float = np.inf, Td: float = 0.0, N: float = 10.0) -> TransferFunction:
    """Standard (ISA) form Kp (1 + 1/(Ti s) + Td s/(1 + Td s/N)) converted to the parallel form."""
    Ki = Kp / Ti if np.isfinite(Ti) else 0.0
    return pid(Kp, Ki, Kp * Td, N)


def closed_loop(plant, controller, sensor=None) -> dict:
    """Standard loop: L = C P, T = L/(1 + L) (reference to output), S = 1/(1 + L) (disturbance at the output),
    CS (reference to control signal), PS (input disturbance to output). All as transfer functions."""
    L = series(controller, plant)
    one = TransferFunction([1.0], [1.0])
    T = feedback(L)
    S = feedback(one, L)
    return {"L": L, "T": T, "S": S, "CS": series(S, controller).cancel(), "PS": series(S, plant).cancel()}


# ------------------------------------------------------------------ tuning rules for K e^{-Ls}/(tau s + 1)
def fopdt_from_step(t, y, u_amplitude: float = 1.0) -> dict:
    """First-order-plus-dead-time (K, tau, L) from a step response by the two-point method (28.3 % and 63.2 %
    of the final value: tau = 1.5 (t63 - t28), L = t63 - tau)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    yf = y[-1]
    K = yf / u_amplitude
    t28 = np.interp(0.283 * yf, y, t)
    t63 = np.interp(0.632 * yf, y, t)
    tau = 1.5 * (t63 - t28)
    return {"K": K, "tau": tau, "L": max(t63 - tau, 0.0)}


def ziegler_nichols_step(K: float, tau: float, L: float) -> dict:
    """Ziegler-Nichols open-loop (reaction curve) rules; a = K L / tau."""
    a = K * L / tau
    return {
        "P": {"Kp": 1 / a, "Ti": np.inf, "Td": 0.0},
        "PI": {"Kp": 0.9 / a, "Ti": 3 * L, "Td": 0.0},
        "PID": {"Kp": 1.2 / a, "Ti": 2 * L, "Td": 0.5 * L},
    }


def ziegler_nichols_ultimate(Ku: float, Pu: float) -> dict:
    """Ziegler-Nichols closed-loop rules from the ultimate gain and period."""
    return {
        "P": {"Kp": 0.5 * Ku, "Ti": np.inf, "Td": 0.0},
        "PI": {"Kp": 0.45 * Ku, "Ti": Pu / 1.2, "Td": 0.0},
        "PID": {"Kp": 0.6 * Ku, "Ti": Pu / 2, "Td": Pu / 8},
    }


def cohen_coon(K: float, tau: float, L: float) -> dict:
    """Cohen-Coon rules for a FOPDT model (K, tau, L): P, PI and PID settings as (Kp, Ti, Td)."""
    r = L / tau
    a = K * L / tau
    return {
        "P": {"Kp": (1 + r / 3) / a, "Ti": np.inf, "Td": 0.0},
        "PI": {"Kp": (0.9 + r / 12) / a, "Ti": L * (30 + 3 * r) / (9 + 20 * r), "Td": 0.0},
        "PID": {"Kp": (4 / 3 + r / 4) / a, "Ti": L * (32 + 6 * r) / (13 + 8 * r), "Td": 4 * L / (11 + 2 * r)},
    }


def imc_pid(K: float, tau: float, L: float, lam: float | None = None, rule: str = "simc") -> dict:
    """IMC-based PI/PID for a FOPDT model. 'simc' (Skogestad 2003): Kp = tau/(K (lam + L)), Ti = min(tau, 4(lam + L)),
    with lam = L by default (the tight-but-robust choice); 'imc' (Rivera-Morari with a first-order Pade of the
    delay): Kp = (tau + L/2)/(K (lam + L/2)), Ti = tau + L/2, Td = tau L/(2 tau + L)."""
    if rule == "simc":
        lam = L if lam is None else lam
        return {"Kp": tau / (K * (lam + L)), "Ti": min(tau, 4 * (lam + L)), "Td": 0.0, "lambda": lam}
    lam = L if lam is None else lam
    return {"Kp": (tau + L / 2) / (K * (lam + L / 2)), "Ti": tau + L / 2, "Td": tau * L / (2 * tau + L), "lambda": lam}


def ultimate_gain(plant, delay: float = 0.0) -> dict:
    """Ultimate gain and period from the phase crossover of the plant (with a pure delay if given)."""
    m = margins(plant, delay=delay)
    return {"Ku": m["gain_margin"], "Pu": 2 * np.pi / m["w_pc"] if np.isfinite(m["w_pc"]) else np.inf, "w_u": m["w_pc"]}


# ------------------------------------------------------------------ simulations
def _discretise(plant, dt, delay=0.0):
    s = plant.to_ss()
    Phi, Gam = expm_with_integral(s.A, s.B, dt)
    nd = int(round(delay / dt))
    return s, Phi, Gam, nd


def relay_experiment(
    plant, delay: float = 0.0, amplitude: float = 1.0, hysteresis: float = 0.0, dt: float = 0.01, T: float = 60.0
) -> dict:
    """Astrom-Hagglund relay feedback: the loop oscillates at the ultimate period with an amplitude a; the
    ultimate gain is 4 d/(pi a) (describing function). Returns the estimates and the recorded signals."""
    s, Phi, Gam, nd = _discretise(plant, dt, delay)
    N = int(round(T / dt))
    x = np.zeros(s.n)
    u_hist = np.zeros(N + nd + 1)
    y = np.zeros(N)
    u = amplitude
    for k in range(N):
        y[k] = (s.C @ x + s.D @ [u_hist[k]])[0]
        if y[k] > hysteresis:
            u = -amplitude
        elif y[k] < -hysteresis:
            u = amplitude
        u_hist[k + nd] = u
        x = Phi @ x + Gam @ [u_hist[k]]
    t = np.arange(N) * dt
    tail = slice(N // 2, N)
    yt = y[tail]
    a = 0.5 * (yt.max() - yt.min())
    crossings = np.flatnonzero((yt[:-1] < 0) & (yt[1:] >= 0))
    Pu = float(np.mean(np.diff(crossings)) * dt) if len(crossings) > 2 else np.nan
    return {"Ku": 4 * amplitude / (np.pi * a), "Pu": Pu, "amplitude": a, "t": t, "y": y, "u": u_hist[:N]}


def simulate_pid(
    plant,
    Kp: float,
    Ki: float,
    Kd: float,
    setpoint,
    dt: float = 0.01,
    T: float = 10.0,
    delay: float = 0.0,
    u_limits=None,
    antiwindup: str | None = "back-calculation",
    N: float = 10.0,
    Tt: float | None = None,
    disturbance=None,
    derivative_on: str = "measurement",
) -> dict:
    """Discrete-time simulation of a PID loop: plant (continuous, zero-order hold), setpoint r(t) (number or
    function), input disturbance d(t), optional actuator limits (umin, umax) and anti-windup: 'clamping' stops
    integration when saturated, 'back-calculation' feeds the saturation error back with time constant Tt
    (default sqrt(Ti Td) or Ti/2). The derivative acts on the measurement (no setpoint kick) or, with
    derivative_on = 'error', on the error, in both cases through a first-order filter N."""
    s, Phi, Gam, nd = _discretise(plant, dt, delay)
    n = int(round(T / dt))
    r = (lambda t: float(setpoint)) if not callable(setpoint) else setpoint
    d = (lambda t: 0.0) if disturbance is None else disturbance
    Ti = Kp / Ki if Ki else np.inf
    Td = Kd / Kp if Kp else 0.0
    Tt = Tt if Tt is not None else (np.sqrt(Ti * Td) if (Ki and Kd) else (Ti / 2 if Ki else 1.0))
    Tf = Kd / (Kp * N) if (Kp and Kd) else 0.0
    x = np.zeros(s.n)
    I_ = 0.0
    y_prev = 0.0
    dterm = 0.0
    u_hist = np.zeros(n + nd + 1)
    t = np.arange(n) * dt
    y, u, uc = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        y[k] = (s.C @ x)[0] + (s.D @ [u_hist[k]])[0]
        e = r(t[k]) - y[k]
        sig = -y[k] if derivative_on == "measurement" else e  # the signal the derivative acts on
        if Tf > 0:  # filtered derivative
            dterm = (Tf * dterm + Kd * (sig - y_prev)) / (Tf + dt)
        else:
            dterm = Kd * (sig - y_prev) / dt if k else 0.0
        y_prev = sig
        v = Kp * e + I_ + dterm  # unsaturated control
        uk = v if u_limits is None else float(np.clip(v, *u_limits))
        if Ki:
            if antiwindup == "clamping" and u_limits is not None and uk != v and np.sign(e) == np.sign(v):
                pass  # hold the integrator
            elif antiwindup == "back-calculation":
                I_ += dt * (Ki * e + (uk - v) / Tt)
            else:
                I_ += dt * Ki * e
        uc[k], u[k] = v, uk
        u_hist[k + nd] = uk
        x = Phi @ x + Gam @ [u_hist[k] + d(t[k])]
    return {"t": t, "y": y, "u": u, "u_unsaturated": uc}


# ------------------------------------------------------------------ lead / lag compensators
def lead(phi_max_deg: float, w_max: float) -> TransferFunction:
    """Lead compensator (1 + T s)/(1 + alpha T s) giving phase lead phi_max at w_max:
    alpha = (1 - sin phi)/(1 + sin phi), T = 1/(w_max sqrt(alpha)). Its gain at w_max is 1/sqrt(alpha)."""
    phi = np.radians(phi_max_deg)
    alpha = (1 - np.sin(phi)) / (1 + np.sin(phi))
    T = 1 / (w_max * np.sqrt(alpha))
    return TransferFunction([T, 1.0], [alpha * T, 1.0])


def lag(beta: float, w_zero: float) -> TransferFunction:
    """Lag compensator beta (1 + T s)/(1 + beta T s), beta > 1: dc gain beta, high-frequency gain 1, so it raises the
    low-frequency loop gain by beta and leaves the crossover region untouched when its zero w_zero sits well
    below the crossover (a decade is the usual rule, where its phase lag has faded to a few degrees)."""
    T = 1 / w_zero
    return TransferFunction([beta * T, beta], [beta * T, 1.0])


def design_lead(plant, w_c: float, pm_target_deg: float, margin_extra_deg: float = 5.0) -> dict:
    """Lead design for a crossover w_c and phase margin: read the plant's phase at w_c, add the lead that is
    missing (plus a small extra), set the gain so that |K C P(j w_c)| = 1."""
    Pj = freqresp(plant, w_c)
    w_grid = np.logspace(
        np.log10(w_c) - 4, np.log10(w_c), 2000
    )  # unwrapped phase up to w_c (np.angle alone would wrap -185 to +175)
    phase_unw = np.degrees(np.unwrap(np.angle(freqresp(plant, w_grid))))
    phase_unw = phase_unw - 360.0 * np.ceil(
        phase_unw[0] / 360.0
    )  # start in (-360, 0]: a negative real low-frequency gain is -180, not +180
    phase_now = phase_unw[-1]
    needed = pm_target_deg - (180.0 + phase_now) + margin_extra_deg
    needed = max(needed, 0.0)
    n_leads = 1 if needed <= 65.0 else int(np.ceil(needed / 60.0))  # one lead gives at most ~70 degrees usefully
    C = lead(needed / n_leads, w_c)
    for _ in range(n_leads - 1):
        C = series(C, lead(needed / n_leads, w_c))
    K = 1.0 / abs(freqresp(C, w_c) * Pj)
    Ctot = TransferFunction(K * C.num, C.den)
    return {
        "controller": Ctot,
        "K": K,
        "lead_angle": needed,
        "n_leads": n_leads,
        "margins": margins(series(Ctot, plant)),
    }


def design_lag(plant, K_lowfreq_factor: float, w_c: float) -> TransferFunction:
    """Lag that raises the low-frequency loop gain by the given factor with its zero a decade below w_c."""
    return lag(K_lowfreq_factor, w_c / 10.0)


# ------------------------------------------------------------------ feedforward
def feedforward_inverse(plant, bandwidth_filter: float | None = None, order: int = 1) -> TransferFunction:
    """Approximate plant inverse P^-1 made proper with a low-pass filter 1/(1 + s/wf)^k (minimum-phase plants):
    the reference feedforward that would give y = r if the model were exact."""
    inv = TransferFunction(plant.den, plant.num) if len(plant.num) == len(plant.den) else None
    rel = len(plant.den) - len(plant.num)
    wf = bandwidth_filter if bandwidth_filter is not None else 10.0 * np.max(np.abs(plant.poles()))
    filt = TransferFunction([1.0], np.array([1.0]))
    for _ in range(max(rel, order if rel == 0 else 0)):
        filt = series(filt, TransferFunction([1.0], [1.0 / wf, 1.0]))
    num = pl.mul(plant.den, filt.num)
    den = pl.mul(plant.num, filt.den)
    return TransferFunction(num, den) if inv is None else series(inv, filt)
