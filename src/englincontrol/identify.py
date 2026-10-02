"""System identification from data: FOPDT and second-order models from step responses, ARX models by least
squares (with the residual whiteness check), transfer-function fits to frequency-response data (Levy's
linearised least squares with iterative reweighting), the eigensystem realisation algorithm (Ho-Kalman) from
impulse/Markov parameters, model reduction by balanced truncation, and a PRBS test signal.

>>> import numpy as np
>>> from englincontrol import identify as idf
>>> u = idf.prbs(1000, seed=0)
>>> int(np.sum(np.abs(u) == 1.0))
1000
"""

from __future__ import annotations

import numpy as np

from . import polynomials as pl
from .systems import StateSpace, TransferFunction, balanced_realisation


def prbs(n: int, seed: int = 0, bits: int = 9, hold: int = 1) -> np.ndarray:
    """Pseudo-random binary sequence of +-1 from a maximal-length shift register (default 9 bits, period 511),
    each bit held for `hold` samples: the clock period sets the excitation bandwidth (about 1/(hold dt)), which
    should cover the plant's bandwidth but not exceed it by much, or the output drowns in noise."""
    taps = {9: (9, 5), 7: (7, 6), 10: (10, 7), 5: (5, 3)}[bits]
    state = [1] * bits
    for _ in range(seed * 37):  # a different starting point of the sequence per seed
        new = state[taps[0] - 1] ^ state[taps[1] - 1]
        state = [new] + state[:-1]
    bits_out = []
    while len(bits_out) * hold < n:
        new = state[taps[0] - 1] ^ state[taps[1] - 1]
        bits_out.append(1.0 if state[-1] else -1.0)
        state = [new] + state[:-1]
    return np.repeat(np.array(bits_out), hold)[:n]


def arx(u, y, na: int, nb: int, nk: int = 1, dt: float = 1.0, method: str = "ls") -> dict:
    """ARX model A(q) y = B(q) u + e: y[k] = -sum a_i y[k-i] + sum b_j u[k-nk-j]. method 'ls' is least squares,
    unbiased when the noise enters as the equation error e; with noise on the measured output it is biased (the
    lagged outputs in the regressor are correlated with the noise), the more so the closer the poles are to
    z = 1. method 'iv' uses delayed inputs as instruments (consistent for output noise). Returns the discrete
    transfer function B(z)/A(z), the parameter covariance, residuals and the fit."""
    u, y = np.asarray(u, float), np.asarray(y, float)
    N = len(y)
    k0 = max(na, nb + nk - 1)
    rows = []
    for k in range(k0, N):
        rows.append(np.r_[-y[k - na : k][::-1], u[k - nk - nb + 1 : k - nk + 1][::-1]])
    Phi = np.array(rows)
    Y = y[k0:]
    if method == "iv":  # instruments: na + nb delayed inputs, uncorrelated with the output noise
        k0i = max(k0, na + nb + nk)
        Z = np.array([u[k - nk - na - nb + 1 : k - nk + 1][::-1] for k in range(k0i, N)])
        Phi, Y = Phi[k0i - k0 :], Y[k0i - k0 :]
        theta = np.linalg.solve(Z.T @ Phi, Z.T @ Y)
    else:
        theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    res = Y - Phi @ theta
    s2 = res @ res / (len(Y) - len(theta))
    cov = s2 * np.linalg.inv(Phi.T @ Phi)
    a, b = theta[:na], theta[na:]
    num = np.r_[np.zeros(nk - 1), b] if nk >= 1 else b  # B(z)/A(z) with the delay as leading zeros... as z^-nk B / A
    den = np.r_[1.0, a]
    # as a proper transfer function in z: multiply numerator and denominator to equal length
    num_z = np.r_[np.zeros(max(len(den) - len(num) - nk, 0)), num]
    tfz = TransferFunction(
        np.r_[num_z, np.zeros(max(len(den) - len(num_z), 0))][: len(den)]
        if len(num_z) < len(den)
        else num_z[: len(den)],
        np.r_[den, np.zeros(max(len(num_z) - len(den), 0))],
        dt,
    )
    fit = 100 * (1 - np.linalg.norm(res) / np.linalg.norm(Y - Y.mean()))
    return {
        "a": a,
        "b": b,
        "tf": tfz,
        "covariance": cov,
        "residuals": res,
        "fit_percent": float(fit),
        "std_errors": np.sqrt(np.diag(cov)),
    }


def autocorrelation(e, lags: int = 20) -> np.ndarray:
    """Normalised autocorrelation of a residual sequence at lags 0 .. lags (white residuals give ~0 beyond lag 0)."""
    e = np.asarray(e, float) - np.mean(e)
    return np.array([np.sum(e[: len(e) - k] * e[k:]) / np.sum(e * e) for k in range(lags + 1)])


def fit_frequency_response(w, H, n_num: int, n_den: int, iterations: int = 10) -> TransferFunction:
    """Rational fit num(s)/den(s) to measured H(jw): Levy's linearised least squares (num - H den ~ 0), reweighted by
    1/|den| from the previous iteration (Sanathanan-Koerner) to undo Levy's high-frequency bias."""
    w = np.asarray(w, float)
    H = np.asarray(H, complex)
    s = 1j * w
    weight = np.ones_like(w)
    for _ in range(iterations):
        rows = []
        rhs = []
        for sk, Hk, wk in zip(s, H, weight):
            pn = sk ** np.arange(n_num, -1, -1)
            pd = sk ** np.arange(n_den - 1, -1, -1)  # den monic: leading coefficient 1
            row = np.r_[pn, -Hk * pd] * wk
            rows += [row.real, row.imag]
            val = Hk * sk**n_den * wk
            rhs += [val.real, val.imag]
        theta, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        num, den = theta[: n_num + 1], np.r_[1.0, theta[n_num + 1 :]]
        weight = 1 / np.abs(pl.evaluate(den, s))
    return TransferFunction(num, den)


def era(markov, order: int, dt: float = 1.0, n_inputs: int = 1, n_outputs: int = 1, rows: int | None = None) -> dict:
    """Eigensystem realisation algorithm (Ho-Kalman): from the impulse response (Markov parameters h_1, h_2, ...)
    build the block Hankel matrix, take its SVD, and read off a discrete state-space model of the given order.
    Returns the model, the Hankel singular values (the model order's evidence) and D = h_0."""
    h = np.asarray(markov, float)
    if h.ndim == 1:
        h = h.reshape(-1, 1, 1)
    N = len(h) - 1
    r = rows if rows is not None else N // 2
    c = N - r
    Hk = np.block([[h[i + j + 1] for j in range(c)] for i in range(r)])
    Hk1 = np.block([[h[i + j + 2] for j in range(c)] for i in range(r)])
    U, sv, Vt = np.linalg.svd(Hk, full_matrices=False)
    Ur, Sr, Vr = U[:, :order], np.diag(sv[:order]), Vt[:order].T
    Sih = np.diag(sv[:order] ** -0.5)
    A = Sih @ Ur.T @ Hk1 @ Vr @ Sih
    B = (np.sqrt(Sr) @ Vr.T)[:, :n_inputs]
    C = (Ur @ np.sqrt(Sr))[:n_outputs, :]
    return {"system": StateSpace(A, B, C, h[0], dt), "hankel_singular_values": sv}


def balanced_truncation(sys_: StateSpace, order: int) -> dict:
    """Keep the `order` states of largest Hankel singular value in the balanced realisation; the H-infinity error
    is at most twice the sum of the discarded singular values."""
    bal, hsv = balanced_realisation(sys_)
    k = order
    red = StateSpace(bal.A[:k, :k], bal.B[:k], bal.C[:, :k], bal.D, bal.dt)
    return {"system": red, "hankel_singular_values": hsv, "error_bound": 2 * float(np.sum(hsv[k:]))}


def fit_second_order_step(t, y, u_amplitude: float = 1.0) -> dict:
    """Underdamped second-order model K wn^2/(s^2 + 2 zeta wn s + wn^2) from a step response: zeta from the
    overshoot, wn from the peak time."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    yf = y[-1]
    K = yf / u_amplitude
    ipk = int(np.argmax(y))
    Mp = max(y[ipk] / yf - 1, 1e-9)
    zeta = -np.log(Mp) / np.sqrt(np.pi**2 + np.log(Mp) ** 2)
    wn = np.pi / (t[ipk] * np.sqrt(1 - zeta**2))
    return {"K": K, "zeta": zeta, "wn": wn, "tf": TransferFunction([K * wn**2], [1.0, 2 * zeta * wn, wn**2])}


def output_error(u, y, order: int, dt: float, init: TransferFunction | None = None, iterations: int = 50) -> dict:
    """Output-error model: the discrete transfer function B(z)/A(z) (strictly proper, one-sample delay) whose
    *simulated* output y_hat(u) is closest to y in least squares - unbiased for noise on the measured output
    (unlike ARX, whose regressor contains the noisy outputs). Levenberg-Marquardt on the simulation error with a
    finite-difference Jacobian, started from the ARX least-squares estimate unless `init` is given."""
    from .response import lsim

    u, y = np.asarray(u, float), np.asarray(y, float)
    if init is None:
        init = arx(u, y, order, order, 1, dt)["tf"]
    theta = np.r_[
        init.num[-order:] if len(init.num) >= order else np.r_[np.zeros(order - len(init.num)), init.num], init.den[1:]
    ]

    def simulate(th):
        num = np.r_[0.0, th[:order]]
        den = np.r_[1.0, th[order:]]
        G = TransferFunction(num, den, dt)
        if np.any(np.abs(G.poles()) >= 1.0):
            return None
        return lsim(G, u)[1]

    def residual(th):
        yh = simulate(th)
        return None if yh is None else yh - y

    r = residual(theta)
    lam = 1e-3
    for _ in range(iterations):
        J = np.zeros((len(y), len(theta)))
        for j in range(len(theta)):
            d = np.zeros(len(theta))
            d[j] = 1e-7 * max(1.0, abs(theta[j]))
            rp = residual(theta + d)
            rm = residual(theta - d)
            if rp is None or rm is None:
                rp = r if rp is None else rp
                rm = r if rm is None else rm
            J[:, j] = (rp - rm) / (2 * d[j]) if (rp is not rm) else 0.0
        g = J.T @ r
        H = J.T @ J
        improved = False
        for _ in range(20):
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
            r_new = residual(theta + step)
            if r_new is not None and r_new @ r_new < r @ r:
                theta, r = theta + step, r_new
                lam = max(lam / 3, 1e-12)
                improved = True
                break
            lam *= 4
        if not improved or np.linalg.norm(step) < 1e-10 * max(1.0, np.linalg.norm(theta)):
            break
    num = np.r_[0.0, theta[:order]]
    den = np.r_[1.0, theta[order:]]
    fit = 100 * (1 - np.linalg.norm(r) / np.linalg.norm(y - y.mean()))
    return {"tf": TransferFunction(num, den, dt), "residuals": r, "fit_percent": float(fit)}
