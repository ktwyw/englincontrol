"""Example plants with physical parameters, as state-space models with named states and the properties the
course checks them against: a DC motor (speed and position), mass-spring-damper, two interacting tanks, a
thermal process with dead time, a cruise control, magnetic levitation, the cart-pendulum linearised upright,
an aircraft's short-period mode, a quarter-car suspension, and a 2 x 2 distillation column.

>>> from englincontrol import models
>>> m = models.dc_motor()
>>> m["system"].n, m["states"]
(2, ['current', 'speed'])
"""

from __future__ import annotations

import numpy as np

from .systems import StateSpace, TransferFunction


def dc_motor(
    R: float = 1.0,
    L: float = 0.5,
    Kt: float = 0.01,
    Ke: float = 0.01,
    J: float = 0.01,
    b: float = 0.1,
    position: bool = False,
) -> dict:
    """Armature-controlled DC motor: L di/dt = -R i - Ke w + V, J dw/dt = Kt i - b w. Output: speed w (or the
    angle theta with an added integrator state). Transfer function V -> w: Kt / ((L s + R)(J s + b) + Kt Ke)."""
    A = np.array([[-R / L, -Ke / L], [Kt / J, -b / J]])
    B = np.array([[1 / L], [0.0]])
    C = np.array([[0.0, 1.0]])
    states = ["current", "speed"]
    if position:
        A = np.block([[A, np.zeros((2, 1))], [np.array([[0.0, 1.0, 0.0]])]])
        B = np.vstack([B, [[0.0]]])
        C = np.array([[0.0, 0.0, 1.0]])
        states.append("angle")
    tf = TransferFunction([Kt], np.polymul([L, R], [J, b]) + np.array([0, 0, Kt * Ke]))
    return {
        "system": StateSpace(A, B, C, 0.0),
        "tf_speed": tf,
        "states": states,
        "input": "voltage (V)",
        "output": "angle (rad)" if position else "speed (rad/s)",
        "params": {"R": R, "L": L, "Kt": Kt, "Ke": Ke, "J": J, "b": b},
    }


def mass_spring_damper(m: float = 1.0, c: float = 0.4, k: float = 4.0) -> dict:
    """m x'' + c x' + k x = F: natural frequency sqrt(k/m), damping ratio c/(2 sqrt(k m))."""
    A = np.array([[0.0, 1.0], [-k / m, -c / m]])
    return {
        "system": StateSpace(A, [[0.0], [1 / m]], [[1.0, 0.0]], 0.0),
        "states": ["position", "velocity"],
        "input": "force (N)",
        "output": "position (m)",
        "wn": np.sqrt(k / m),
        "zeta": c / (2 * np.sqrt(k * m)),
        "params": {"m": m, "c": c, "k": k},
    }


def two_tanks(A1: float = 1.0, A2: float = 1.0, R1: float = 2.0, R2: float = 3.0) -> dict:
    """Two tanks in series (interacting): A1 h1' = q - (h1 - h2)/R1, A2 h2' = (h1 - h2)/R1 - h2/R2; input q, output h2.
    Overdamped with two real poles; dc gain R2."""
    A = np.array([[-1 / (A1 * R1), 1 / (A1 * R1)], [1 / (A2 * R1), -1 / (A2 * R1) - 1 / (A2 * R2)]])
    return {
        "system": StateSpace(A, [[1 / A1], [0.0]], [[0.0, 1.0]], 0.0),
        "states": ["level 1", "level 2"],
        "input": "inflow (m^3/s)",
        "output": "level 2 (m)",
        "dc_gain": R2,
        "params": {"A1": A1, "A2": A2, "R1": R1, "R2": R2},
    }


def thermal_process(K: float = 2.0, tau: float = 50.0, L: float = 5.0) -> dict:
    """A stirred heater seen from its thermocouple: K e^{-L s}/(tau s + 1) (the FOPDT model of process control);
    the delay is carried separately (use frequency.freqresp(..., delay=L) or design.simulate_pid(..., delay=L))."""
    return {
        "tf": TransferFunction([K], [tau, 1.0]),
        "delay": L,
        "input": "heater power (kW)",
        "output": "temperature (K)",
        "params": {"K": K, "tau": tau, "L": L},
    }


def cruise_control(m: float = 1000.0, b: float = 50.0) -> dict:
    """m v' = F - b v: a first-order plant with time constant m/b and gain 1/b."""
    return {
        "system": StateSpace([[-b / m]], [[1 / m]], [[1.0]], 0.0),
        "states": ["speed"],
        "input": "force (N)",
        "output": "speed (m/s)",
        "tau": m / b,
        "params": {"m": m, "b": b},
    }


def magnetic_levitation(m: float = 0.05, k: float = 1e-4, g: float = 9.81, x0: float = 0.01) -> dict:
    """Ball levitated by an electromagnet: m x'' = m g - k i^2/x^2, linearised about the equilibrium (x0, i0):
    x'' = (2 g/x0) dx - (2 sqrt(g k/m)/x0) di... states [x, x'], input di: unstable with poles +-sqrt(2 g/x0)."""
    i0 = np.sqrt(m * g / k) * x0
    a = 2 * g / x0
    bcoef = -2 * k * i0 / (m * x0**2)
    A = np.array([[0.0, 1.0], [a, 0.0]])
    return {
        "system": StateSpace(A, [[0.0], [bcoef]], [[1.0, 0.0]], 0.0),
        "states": ["gap", "gap rate"],
        "input": "current deviation (A)",
        "output": "gap (m)",
        "unstable_pole": np.sqrt(a),
        "i0": i0,
        "params": {"m": m, "k": k, "g": g, "x0": x0},
    }


def cart_pendulum(
    M: float = 0.5, m: float = 0.2, length: float = 0.3, b: float = 0.1, inertia: float | None = None, g: float = 9.81
) -> dict:
    """Cart of mass M with an inverted pendulum (mass m, distance `length` to its centre, inertia about the centre
    `inertia`, default m l^2/3 for a rod), linearised about the upright position from Lagrange's equations
    (M + m) x'' + b x' + m l theta'' = F, (I + m l^2) theta'' + m l x'' - m g l theta = 0, with theta measured from
    upright in the direction of x. States [x, x', theta, theta'], input: force on the cart, outputs [x, theta]."""
    ell = length
    inertia = m * ell * ell / 3 if inertia is None else inertia
    p = inertia * (M + m) + M * m * ell * ell
    A = np.array(
        [
            [0, 1, 0, 0],
            [0, -(inertia + m * ell * ell) * b / p, -((m * ell) ** 2) * g / p, 0],
            [0, 0, 0, 1],
            [0, m * ell * b / p, m * g * ell * (M + m) / p, 0],
        ]
    )
    B = np.array([[0.0], [(inertia + m * ell * ell) / p], [0.0], [-m * ell / p]])
    C = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    return {
        "system": StateSpace(A, B, C, np.zeros((2, 1))),
        "states": ["cart position", "cart velocity", "pendulum angle", "pendulum rate"],
        "input": "force (N)",
        "outputs": ["position (m)", "angle (rad)"],
        "params": {"M": M, "m": m, "l": ell, "b": b, "I": inertia, "g": g},
    }


def aircraft_short_period(
    Z_alpha: float = -1.2,
    M_alpha: float = -5.0,
    M_q: float = -1.5,
    Z_de: float = -0.1,
    M_de: float = -8.0,
    V: float = 100.0,
) -> dict:
    """Longitudinal short-period approximation: states [alpha, q] (angle of attack, pitch rate), input elevator
    deflection; alpha' = Z_alpha alpha + q + Z_de/V de, q' = M_alpha alpha + M_q q + M_de de. Output: pitch rate."""
    A = np.array([[Z_alpha, 1.0], [M_alpha, M_q]])
    B = np.array([[Z_de / V], [M_de]])
    wn = np.sqrt(-M_alpha + Z_alpha * M_q)
    zeta = -(Z_alpha + M_q) / (2 * wn)
    return {
        "system": StateSpace(A, B, [[0.0, 1.0]], 0.0),
        "states": ["angle of attack", "pitch rate"],
        "input": "elevator (rad)",
        "output": "pitch rate (rad/s)",
        "wn": wn,
        "zeta": zeta,
        "params": {"Z_alpha": Z_alpha, "M_alpha": M_alpha, "M_q": M_q, "Z_de": Z_de, "M_de": M_de, "V": V},
    }


def quarter_car(
    ms: float = 300.0, mu: float = 40.0, ks: float = 15000.0, kt: float = 150000.0, cs: float = 1000.0
) -> dict:
    """Quarter-car suspension: sprung mass ms on spring ks and damper cs over the unsprung mass mu on the tyre
    stiffness kt. States [zs, zs', zu, zu'], inputs [road height zr, active force F], outputs [body acceleration,
    suspension travel zs - zu]. The body-bounce and wheel-hop modes are its two resonances."""
    A = np.array(
        [
            [0, 1, 0, 0],
            [-ks / ms, -cs / ms, ks / ms, cs / ms],
            [0, 0, 0, 1],
            [ks / mu, cs / mu, -(ks + kt) / mu, -cs / mu],
        ]
    )
    B = np.array([[0, 0], [0, 1 / ms], [0, 0], [kt / mu, -1 / mu]])
    C = np.array([[-ks / ms, -cs / ms, ks / ms, cs / ms], [1, 0, -1, 0]])
    D = np.array([[0, 1 / ms], [0, 0]])
    return {
        "system": StateSpace(A, B, C, D),
        "states": ["body position", "body velocity", "wheel position", "wheel velocity"],
        "inputs": ["road (m)", "actuator force (N)"],
        "outputs": ["body acceleration (m/s^2)", "suspension travel (m)"],
        "params": {"ms": ms, "mu": mu, "ks": ks, "kt": kt, "cs": cs},
    }


def distillation_2x2() -> dict:
    """A delay-free version of the Wood-Berry column: top and bottom compositions from reflux and steam,
    G = [[12.8/(16.7s+1), -18.9/(21s+1)], [6.6/(10.9s+1), -19.4/(14.4s+1)]] (the delays of the original are
    dropped; the interaction, RGA 2.0 on the diagonal, is kept)."""
    gains = np.array([[12.8, -18.9], [6.6, -19.4]])
    taus = np.array([[16.7, 21.0], [10.9, 14.4]])
    A = np.diag(-1 / taus.ravel())
    B = np.zeros((4, 2))
    C = np.zeros((2, 4))
    k = 0
    for i in range(2):
        for j in range(2):
            B[k, j] = gains[i, j] / taus[i, j]
            C[i, k] = 1.0
            k += 1
    return {
        "system": StateSpace(A, B, C, np.zeros((2, 2))),
        "inputs": ["reflux", "steam"],
        "outputs": ["top composition", "bottom composition"],
        "gains": gains,
        "taus": taus,
    }


def linearise(f, x0, u0, eps: float = 1e-6) -> tuple:
    """Jacobians A = df/dx, B = df/du of x' = f(x, u) at (x0, u0) by central differences."""
    x0, u0 = np.asarray(x0, float), np.atleast_1d(np.asarray(u0, float))
    n, m = len(x0), len(u0)
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for j in range(n):
        d = np.zeros(n)
        d[j] = eps
        A[:, j] = (np.asarray(f(x0 + d, u0)) - np.asarray(f(x0 - d, u0))) / (2 * eps)
    for j in range(m):
        d = np.zeros(m)
        d[j] = eps
        B[:, j] = (np.asarray(f(x0, u0 + d)) - np.asarray(f(x0, u0 - d))) / (2 * eps)
    return A, B


def pendulum_nonlinear(
    M: float = 0.5, m: float = 0.2, length: float = 0.3, b: float = 0.1, inertia: float | None = None, g: float = 9.81
):
    """The nonlinear cart-pendulum x' = f(x, u) of the same equations (for linearise and for simulation beyond
    the linear range)."""
    ell = length
    inertia = m * ell * ell / 3 if inertia is None else inertia

    def f(x, u):
        _, xd, th, thd = x
        F = u[0]
        s_, c_ = np.sin(th), np.cos(th)
        den = (M + m) * (inertia + m * ell * ell) - (m * ell * c_) ** 2
        xdd = ((inertia + m * ell * ell) * (F - b * xd + m * ell * thd**2 * s_) - m * m * ell * ell * g * s_ * c_) / den
        thdd = (m * ell * (g * s_ * (M + m) - c_ * (F - b * xd + m * ell * thd**2 * s_))) / den
        return np.array([xd, xdd, thd, thdd])

    return f
