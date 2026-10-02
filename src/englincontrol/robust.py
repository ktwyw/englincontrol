"""Robustness and fundamental limitations: the gang of four with their peaks Ms and Mt, Bode's sensitivity
integral (the waterbed effect) evaluated numerically, the bandwidth limits imposed by right-half-plane zeros
and poles and by delay, the disk margin, the small-gain condition, and stability margins as the distance of
the Nyquist curve from -1.

>>> from englincontrol import systems as sy, robust as rb
>>> round(rb.sensitivity_peak(sy.tf([1.0], [1.0, 3.0, 2.0, 0.0]))["Ms"], 3)
1.568
"""

from __future__ import annotations

import numpy as np

from .frequency import default_frequencies, freqresp
from .systems import TransferFunction, feedback, series


def gang_of_four(P: TransferFunction, C: TransferFunction) -> dict:
    """L, S = 1/(1 + L), T = L/(1 + L), PS and CS for a plant P and controller C in a unity-feedback loop."""
    L = series(C, P)
    one = TransferFunction([1.0], [1.0])
    S = feedback(one, L)
    return {"L": L, "S": S, "T": feedback(L), "PS": series(S, P).cancel(), "CS": series(S, C).cancel()}


def sensitivity_peak(L: TransferFunction, w=None) -> dict:
    """Ms = max |S|, Mt = max |T| and the margins they imply: gain margin >= Ms/(Ms - 1), phase margin >=
    2 arcsin(1/(2 Ms)) (the disk of radius 1/Ms around -1 is avoided)."""
    w = default_frequencies(L, 4000) if w is None else np.asarray(w, float)
    Lj = freqresp(L, w)
    S = 1 / (1 + Lj)
    T = Lj / (1 + Lj)
    Ms, Mt = float(np.max(np.abs(S))), float(np.max(np.abs(T)))
    return {
        "Ms": Ms,
        "Mt": Mt,
        "w_Ms": float(w[np.argmax(np.abs(S))]),
        "gm_lower_bound": Ms / (Ms - 1) if Ms > 1 else np.inf,
        "pm_lower_bound_deg": float(np.degrees(2 * np.arcsin(1 / (2 * Ms)))),
    }


def bode_sensitivity_integral(L: TransferFunction, w_max: float = 1e4, n: int = 200000) -> dict:
    """int_0^inf ln|S(jw)| dw = pi sum Re(p_k) over the unstable open-loop poles p_k, for loops with relative
    degree >= 2 (Bode 1945, Freudenberg & Looze 1985): the waterbed effect."""
    w = np.logspace(-4, np.log10(w_max), n)
    S = 1 / (1 + freqresp(L, w))
    integral = float(np.sum(np.log(np.abs(S[1:])) * np.diff(w))) + float(np.log(abs(S[0])) * w[0])
    unstable = L.poles()[L.poles().real > 1e-9]
    return {"integral": integral, "theory": float(np.pi * np.sum(unstable.real)), "unstable_poles": unstable}


def rhp_limits(P: TransferFunction, delay: float = 0.0) -> dict:
    """Rules of thumb for achievable bandwidth: a RHP zero z limits w_c < z/2, a RHP pole p needs w_c > 2 p,
    a delay L limits w_c < 1/L; Ms at least |p + z|/|p - z| when both occur."""
    z = P.zeros()
    p = P.poles()
    zr, pr = z[z.real > 1e-9], p[p.real > 1e-9]
    out = {
        "rhp_zeros": zr,
        "rhp_poles": pr,
        "w_c_max_from_zero": float(np.min(np.abs(zr)) / 2) if len(zr) else np.inf,
        "w_c_min_from_pole": float(2 * np.max(np.abs(pr))) if len(pr) else 0.0,
        "w_c_max_from_delay": 1 / delay if delay else np.inf,
    }
    if len(zr) and len(pr):
        out["Ms_lower_bound"] = float(np.max([abs(pk + zk) / abs(pk - zk) for pk in pr for zk in zr]))
    return out


def disk_margin(L: TransferFunction, w=None) -> dict:
    """Symmetric disk margin: the largest alpha such that the loop stays stable for all gain/phase perturbations
    in the disk of radius alpha about 1 (balanced sensitivity): alpha = 1 / max|S - T|. The corresponding gain
    margins are (1 -+ alpha/2)/(1 +- alpha/2) and the phase margin 2 arctan(alpha/2)."""
    w = default_frequencies(L, 4000) if w is None else np.asarray(w, float)
    Lj = freqresp(L, w)
    S = 1 / (1 + Lj)
    T = Lj / (1 + Lj)
    alpha = 1.0 / float(np.max(np.abs(S - T)))
    return {
        "alpha": alpha,
        "gain_margin_range": ((1 - alpha / 2) / (1 + alpha / 2), (1 + alpha / 2) / (1 - alpha / 2))
        if alpha < 2
        else (0.0, np.inf),
        "phase_margin_deg": float(np.degrees(2 * np.arctan(alpha / 2))),
    }


def nyquist_distance(L: TransferFunction, w=None) -> dict:
    """Minimum distance of L(jw) from -1 (= 1/Ms) and the frequency where it occurs: the single most honest
    stability margin."""
    w = default_frequencies(L, 4000) if w is None else np.asarray(w, float)
    d = np.abs(1 + freqresp(L, w))
    i = int(np.argmin(d))
    return {"distance": float(d[i]), "w": float(w[i])}


def small_gain(M: TransferFunction, delta_bound: float) -> dict:
    """Small-gain theorem: the loop of M with any stable perturbation Delta with ||Delta||_inf < delta_bound is
    stable if ||M||_inf < 1/delta_bound. Returns the peak of |M| and the verdict."""
    from .frequency import hinf_norm

    norm, w_pk = hinf_norm(M.to_ss())
    return {"M_inf": norm, "w_peak": w_pk, "robustly_stable": norm * delta_bound < 1.0}


def multiplicative_robustness(L: TransferFunction, W=None, w=None) -> dict:
    """Robust stability against multiplicative plant uncertainty P (1 + W Delta), ||Delta||_inf <= 1: requires
    |W T| < 1 at every frequency; returns the peak of |W T| (W = 1 if not given)."""
    w = default_frequencies(L, 4000) if w is None else np.asarray(w, float)
    Lj = freqresp(L, w)
    T = Lj / (1 + Lj)
    Wj = freqresp(W, w) if W is not None else np.ones_like(w)
    peak = float(np.max(np.abs(Wj * T)))
    return {"peak_WT": peak, "robustly_stable": peak < 1.0, "w_peak": float(w[np.argmax(np.abs(Wj * T))])}
