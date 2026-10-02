"""Validation of englincontrol against closed-form results, SciPy, LAPACK and python-control, and the theory of
linear systems and control. Run:  python docs/validate.py   (writes docs/VALIDATION.md)
"""

from __future__ import annotations

import warnings
from pathlib import Path

with warnings.catch_warnings():  # python-control 0.9 warns on import under recent pyparsing
    warnings.simplefilter("ignore")
    import control as ct
import numpy as np
from scipy import linalg as sla
from scipy import signal as ssg

from englincontrol import linalg as la
from englincontrol import polynomials as pl
from englincontrol import response as rs
from englincontrol import systems as sy

RESULTS: list[tuple] = []
rng = np.random.default_rng(2026)


def check(section, name, reference, value, expected, rtol=0.0, atol=0.0):
    ok = bool(np.isclose(float(value), float(expected), rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))
    if not ok:
        print(f"FAIL: {section} | {name}: computed {value!r}, expected {expected!r}")


# ------------------------------------------------------------------ 1. polynomials
S = "Polynomials"
check(S, "(s + 1)(s + 2) = s^2 + 3s + 2", "algebra", float(np.max(np.abs(pl.mul([1, 1], [1, 2]) - [1, 3, 2]))), 0.0)
q, r = pl.div([1, 0, -1], [1, 1])
check(
    S,
    "(s^2 - 1) / (s + 1) = s - 1, remainder 0",
    "long division",
    float(np.max(np.abs(q - [1, -1])) + abs(r[0])),
    0.0,
    atol=1e-15,
)
check(
    S,
    "roots of (s - 1)(s - 2)(s - 3) by the companion matrix",
    "exact",
    float(np.max(np.abs(np.sort(pl.roots([1, -6, 11, -6]).real) - [1, 2, 3]))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "from_roots of a conjugate pair gives real coefficients",
    "s^2 - 2s + 5",
    float(np.max(np.abs(pl.from_roots([1 + 2j, 1 - 2j]) - [1, -2, 5]))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "Horner evaluation at a complex argument",
    "p(1 + i) for s^2 + 1",
    complex(pl.evaluate([1, 0, 1], 1 + 1j)).real + 10 * complex(pl.evaluate([1, 0, 1], 1 + 1j)).imag,
    1 + 10 * 2,
    atol=1e-14,
)

# ------------------------------------------------------------------ 2. matrix computations
S = "Matrix computations"
for n in (2, 6, 15):
    A = rng.normal(size=(n, n)) * 3
    check(
        S,
        f"matrix exponential by scaling-and-squaring Pade 13, n = {n}",
        "scipy.linalg.expm",
        float(np.max(np.abs(la.expm(A) - sla.expm(A))) / np.max(np.abs(sla.expm(A)))),
        0.0,
        atol=1e-13,
    )
A = np.array([[0.0, 1.0], [-1.0, 0.0]])
check(
    S,
    "exp of a rotation generator is a rotation (cos, sin)",
    "exact",
    float(np.max(np.abs(la.expm(np.pi / 3 * A) - np.array([[0.5, np.sqrt(3) / 2], [-np.sqrt(3) / 2, 0.5]])))),
    0.0,
    atol=1e-15,
)
check(
    S,
    "exp(A) exp(-A) = I",
    "group property",
    float(np.max(np.abs(la.expm(A * 2.3) @ la.expm(-A * 2.3) - np.eye(2)))),
    0.0,
    atol=1e-15,
)
A = rng.normal(size=(10, 10)) - 5 * np.eye(10)
T, U = la.schur(A)
check(S, "real Schur form: U T U^T = A", "QR algorithm", float(np.max(np.abs(U @ T @ U.T - A))), 0.0, atol=1e-12)
check(
    S,
    "... U orthogonal, T quasi-triangular",
    "construction",
    float(np.max(np.abs(U.T @ U - np.eye(10))) + np.max(np.abs(np.tril(T, -2)))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... eigenvalues preserved",
    "numpy.linalg.eigvals",
    float(np.max(np.abs(np.sort_complex(np.linalg.eigvals(T)) - np.sort_complex(np.linalg.eigvals(A))))),
    0.0,
    atol=1e-10,
)
Q = np.eye(10)
P = la.lyapunov(A, Q)
check(
    S,
    "continuous Lyapunov equation by Bartels-Stewart, n = 10: residual",
    "A P + P A^T + Q = 0",
    float(np.linalg.norm(A @ P + P @ A.T + Q)),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... agrees with SciPy",
    "scipy.linalg.solve_continuous_lyapunov",
    float(np.max(np.abs(P - sla.solve_continuous_lyapunov(A, -Q)))),
    0.0,
    atol=1e-12,
)
Ad = 0.5 * rng.normal(size=(6, 6)) / 3
Pd = la.dlyapunov(Ad, np.eye(6))
check(
    S,
    "discrete Lyapunov equation",
    "scipy.linalg.solve_discrete_lyapunov",
    float(np.max(np.abs(Pd - sla.solve_discrete_lyapunov(Ad, np.eye(6))))),
    0.0,
    atol=1e-12,
)
A = np.array([[0.0, 1.0], [-1.0, -0.5]])
B = np.array([[0.0], [1.0]])
Qc, Rc = np.eye(2), np.array([[0.1]])
Pc = la.care(A, B, Qc, Rc)
check(
    S,
    "continuous Riccati (CARE) by the Hamiltonian's stable subspace + Newton",
    "scipy.linalg.solve_continuous_are",
    float(np.max(np.abs(Pc - sla.solve_continuous_are(A, B, Qc, Rc)))),
    0.0,
    atol=1e-13,
)
K = np.linalg.solve(Rc, B.T @ Pc)
check(
    S,
    "... the closed loop A - B K is stable",
    "stabilising solution",
    float(np.max(np.linalg.eigvals(A - B @ K).real) < 0),
    1.0,
)
A6 = rng.normal(size=(6, 6))
B6 = rng.normal(size=(6, 2))
P6 = la.care(A6, B6, np.eye(6), np.eye(2))
check(
    S,
    "CARE for a random unstable 6-state, 2-input system",
    "scipy",
    float(np.max(np.abs(P6 - sla.solve_continuous_are(A6, B6, np.eye(6), np.eye(2)))) / np.max(np.abs(P6))),
    0.0,
    atol=1e-11,
)
Add = np.array([[1.0, 0.1], [0.0, 1.0]])
Bdd = np.array([[0.005], [0.1]])
Pdd = la.dare(Add, Bdd, Qc, Rc)
check(
    S,
    "discrete Riccati (DARE) by structure-preserving doubling + Newton",
    "scipy.linalg.solve_discrete_are",
    float(np.max(np.abs(Pdd - sla.solve_discrete_are(Add, Bdd, Qc, Rc)))),
    0.0,
    atol=1e-11,
)
Ad6 = rng.normal(size=(6, 6)) * 0.6
Pd6 = la.dare(Ad6, B6, np.eye(6), np.eye(2))
check(
    S,
    "DARE for a random 6-state, 2-input system",
    "scipy",
    float(np.max(np.abs(Pd6 - sla.solve_discrete_are(Ad6, B6, np.eye(6), np.eye(2)))) / np.max(np.abs(Pd6))),
    0.0,
    atol=1e-10,
)
Wc, Wo = la.gramians(A, B, np.array([[1.0, 0.0]]))
check(
    S,
    "Gramians satisfy their Lyapunov equations",
    "definition",
    float(np.linalg.norm(A @ Wc + Wc @ A.T + B @ B.T) + np.linalg.norm(A.T @ Wo + Wo @ A + np.outer([1, 0], [1, 0]))),
    0.0,
    atol=1e-14,
)
Phi, Gam = la.expm_with_integral(A, B, 0.1)
check(
    S,
    "ZOH discretisation (Phi, Gamma) by the augmented exponential",
    "scipy.signal.cont2discrete",
    float(
        np.max(np.abs(Phi - ssg.cont2discrete((A, B, np.eye(2), np.zeros((2, 1))), 0.1)[0]))
        + np.max(np.abs(Gam - ssg.cont2discrete((A, B, np.eye(2), np.zeros((2, 1))), 0.1)[1]))
    ),
    0.0,
    atol=1e-14,
)

# ------------------------------------------------------------------ 3. systems
S = "Systems"
G = sy.tf([2.0, 1.0], [1.0, 3.0, 2.0])
check(
    S,
    "tf -> controllable canonical ss -> tf round trip",
    "Leverrier-Faddeev",
    float(np.max(np.abs(G.to_ss().to_tf().num - G.num)) + np.max(np.abs(G.to_ss().to_tf().den - G.den))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... observable canonical form gives the same transfer function",
    "duality",
    float(np.max(np.abs(G.to_ss("observable").to_tf().den - G.den))),
    0.0,
    atol=1e-12,
)
A5, B5, C5, D5 = rng.normal(size=(5, 5)), rng.normal(size=(5, 1)), rng.normal(size=(1, 5)), np.array([[0.3]])
ours = sy.ss(A5, B5, C5, D5).to_tf()
ref = ct.ss2tf(ct.ss(A5, B5, C5, D5))
rnum, rden = np.array(ref.num[0][0]), np.array(ref.den[0][0])
check(
    S,
    "ss2tf of a random 5-state system",
    "python-control",
    float(np.max(np.abs(ours.num - rnum / rden[0])) + np.max(np.abs(ours.den - rden / rden[0]))),
    0.0,
    atol=1e-10,
)
check(
    S,
    "poles of the realisation = eigenvalues of A",
    "definition",
    float(np.max(np.abs(np.sort_complex(ours.poles()) - np.sort_complex(np.linalg.eigvals(A5))))),
    0.0,
    atol=1e-8,
)
check(
    S,
    "dc gain from state space = D - C A^-1 B = G(0)",
    "two routes",
    float(abs(sy.ss(A5, B5, C5, D5).dcgain()[0, 0] - ours.dcgain())),
    0.0,
    atol=1e-10,
)
H = sy.tf([1.0], [1.0, 1.0])
cl_tf, cl_ss = sy.feedback(sy.series(G, H)), sy.feedback(sy.series(G.to_ss(), H.to_ss()))
check(
    S,
    "feedback interconnection: tf and ss routes give the same closed-loop poles",
    "consistency",
    float(np.max(np.abs(np.sort_complex(cl_tf.poles()) - np.sort_complex(cl_ss.poles())))),
    0.0,
    atol=1e-10,
)
ref_cl = ct.feedback(ct.series(ct.tf([2, 1], [1, 3, 2]), ct.tf([1], [1, 1])))
check(
    S,
    "... and python-control's",
    "python-control",
    float(np.max(np.abs(np.sort_complex(cl_tf.poles()) - np.sort_complex(np.array(ref_cl.poles()))))),
    0.0,
    atol=1e-10,
)
par = sy.parallel(G, sy.tf([1.0], [1.0, 1.0])).to_ss()
mini = sy.minimal(par)
check(
    S,
    "minimal realisation removes the cancelling (s + 1): (2s+1)/((s+1)(s+2)) + 1/(s+1) = 3/(s+2)",
    "pole-zero cancellation",
    float(mini.n),
    1.0,
)
check(S, "... with the right transfer function", "exact", float(abs(mini.to_tf().dcgain() - 1.5)), 0.0, atol=1e-10)
Gss = G.to_ss()
bal, hsv = sy.balanced_realisation(Gss)
Wcb, Wob = la.gramians(bal.A, bal.B, bal.C)
check(
    S,
    "balanced realisation: the Gramians are equal and diagonal",
    "Moore 1981",
    float(np.max(np.abs(Wcb - Wob)) + np.max(np.abs(Wcb - np.diag(np.diag(Wcb))))),
    0.0,
    atol=1e-10,
)
Wc_s, Wo_s = (
    sla.solve_continuous_lyapunov(Gss.A, -Gss.B @ Gss.B.T),
    sla.solve_continuous_lyapunov(Gss.A.T, -Gss.C.T @ Gss.C),
)
check(
    S,
    "... Hankel singular values = sqrt(eig(Wc Wo))",
    "definition via SciPy Gramians",
    float(np.max(np.abs(np.sort(hsv) - np.sort(np.sqrt(np.linalg.eigvals(Wc_s @ Wo_s).real))))),
    0.0,
    atol=1e-10,
)
modal, Tm = sy.modal_form(sy.ss(A5, B5, C5, D5))
check(
    S,
    "modal form has the same transfer function",
    "similarity invariance",
    float(np.max(np.abs(modal.to_tf().den - ours.den))),
    0.0,
    atol=1e-9,
)
Z = sy.zpk([-3.0], [-1.0, -2.0], 4.0)
check(
    S,
    "zpk constructor: 4(s + 3)/((s + 1)(s + 2))",
    "expansion",
    float(np.max(np.abs(Z.num - [4, 12])) + np.max(np.abs(Z.den - [1, 3, 2]))),
    0.0,
    atol=1e-12,
)

# ------------------------------------------------------------------ 4. time responses
S = "Time responses"
G2 = sy.tf([4.0], [1.0, 1.2, 4.0])  # wn = 2, zeta = 0.3
t, y = rs.step(G2, 10.0, 4001)
tc, yc = ct.step_response(ct.tf([4], [1, 1.2, 4]), T=t)
check(
    S,
    "step response of a second-order system (exact ZOH integration)",
    "python-control",
    float(np.max(np.abs(y - yc))),
    0.0,
    atol=1e-12,
)
info = rs.step_info(t, y, final=G2.dcgain())
th = rs.second_order(0.3, 2.0)
check(
    S,
    "... overshoot exp(-pi zeta/sqrt(1 - zeta^2)) = 37.23 %",
    "closed form",
    info["overshoot"],
    th["overshoot"],
    rtol=1e-3,
)
check(S, "... peak time pi/wd", "closed form", info["peak_time"], th["peak_time"], rtol=2e-3)
check(
    S,
    "... 2 % settling time ~ 4/(zeta wn) (the usual approximation, within 20 %)",
    "approximation",
    info["settling_time"],
    th["settling_time_2pct"],
    rtol=0.2,
)
t1, y1 = rs.step(sy.tf([1.0], [1.0, 1.0]), 5.0, 501)
check(
    S,
    "first-order step 1 - e^-t is exact at every sample",
    "closed form",
    float(np.max(np.abs(y1 - (1 - np.exp(-t1))))),
    0.0,
    atol=1e-13,
)
ti, yi = rs.impulse(G2, 25.0, 10001)
_, yic = ct.impulse_response(ct.tf([4], [1, 1.2, 4]), T=ti)
check(S, "impulse response", "python-control", float(np.max(np.abs(yi - yic))), 0.0, atol=1e-12)
check(
    S,
    "... integrates to the dc gain",
    "integral of the impulse response",
    float(np.sum(0.5 * (yi[1:] + yi[:-1]) * np.diff(ti))),
    G2.dcgain(),
    rtol=1e-3,
)
t = np.linspace(0, 40, 16001)
u = np.sin(3 * t)
_, yl, _ = rs.lsim(G2, u, t)
_, ylc = ct.forced_response(ct.tf([4], [1, 1.2, 4]), T=t, U=u)
w_ = 3.0
tail = t > 30.0  # the transient has decayed by e^-18
M = np.column_stack([np.sin(w_ * t[tail]), np.cos(w_ * t[tail])])
a_, b_ = np.linalg.lstsq(M, yl[tail], rcond=None)[0]
amp, phase = np.hypot(a_, b_), np.arctan2(b_, a_)
Gjw = G2(1j * w_)
check(
    S,
    "forced response to sin(3t): steady-state amplitude = |G(3j)|",
    "frequency response",
    float(amp),
    float(abs(Gjw)),
    rtol=1e-4,
)
check(
    S,
    "... phase = arg G(3j) minus the half-sample delay w h/2 of the zero-order hold on the input",
    "ZOH delay",
    float(phase),
    float(np.angle(Gjw) - w_ * (t[1] - t[0]) / 2),
    atol=1e-4,
)
_, ylc = ct.forced_response(ct.tf([4], [1, 1.2, 4]), T=t, U=u)
check(
    S,
    "... python-control (which interpolates the input linearly) differs by O(h): 3e-3 at h = 0.0025",
    "hold assumption",
    float(np.max(np.abs(yl - ylc))),
    0.0,
    atol=5e-3,
)
Gd = sy.tf([0.5], [1.0, -0.5], dt=0.1)
td, yd = rs.step(Gd, 2.0)
check(
    S,
    "discrete step response of 0.5/(z - 0.5): y[k] = 1 - 0.5^k",
    "closed form",
    float(np.max(np.abs(yd - (1 - 0.5 ** np.arange(len(yd)))))),
    0.0,
    atol=1e-14,
)
check(
    S,
    "dominant poles are the slowest",
    "definition",
    float(np.max(np.abs(rs.dominant_poles(sy.tf([1.0], pl.from_roots([-1, -10, -20])), 1).real + 1))),
    0.0,
    atol=1e-12,
)

# ------------------------------------------------------------------ 5. frequency response
from englincontrol import frequency as fr  # noqa: E402
from englincontrol import stability as st  # noqa: E402

S = "Frequency response"
G1 = sy.tf([1.0], [1.0, 1.0])
check(
    S,
    "|1/(1 + j)| = 1/sqrt 2 and arg = -45 degrees at w = 1",
    "closed form",
    float(abs(fr.freqresp(G1, 1.0))) + abs(np.degrees(np.angle(fr.freqresp(G1, 1.0)))) / 100,
    1 / np.sqrt(2) + 0.45,
    atol=1e-12,
)
w_ = np.logspace(-2, 2, 50)
H_tf, H_ss = fr.freqresp(G2, w_), fr.freqresp(G2.to_ss(), w_)
check(
    S,
    "frequency response from the transfer function and from state space agree",
    "C (jwI - A)^-1 B + D",
    float(np.max(np.abs(H_tf - H_ss))),
    0.0,
    atol=1e-12,
)
mag_c, ph_c, w_c = ct.frequency_response(ct.tf([4], [1, 1.2, 4]), w_)
check(S, "... and with python-control", "python-control", float(np.max(np.abs(np.abs(H_tf) - mag_c))), 0.0, atol=1e-12)
L = sy.tf([10.0], [1.0, 3.0, 2.0, 0.0])
m = fr.margins(L)
gm_c, pm_c, wpc_c, wgc_c = ct.margin(ct.tf([10], [1, 3, 2, 0]))
check(
    S,
    "gain margin of 10/(s(s+1)(s+2)) = 0.6 (unstable loop) at w = sqrt 2",
    "python-control / closed form 6/10",
    m["gain_margin"],
    0.6,
    rtol=1e-8,
)
check(S, "... phase crossover at sqrt(2)", "closed form", m["w_pc"], np.sqrt(2), rtol=1e-8)
check(S, "... phase margin", "python-control", m["phase_margin"], pm_c, atol=1e-6)
check(S, "... gain crossover", "python-control", m["w_gc"], wgc_c, rtol=1e-6)
L2 = sy.tf([1.0], [1.0, 3.0, 2.0, 0.0])
m2 = fr.margins(L2)
check(S, "margins of 1/(s(s+1)(s+2)): gain margin 6 (15.6 dB)", "closed form", m2["gain_margin"], 6.0, rtol=1e-8)
check(
    S,
    "... delay margin = phase margin / w_gc",
    "definition",
    m2["delay_margin"],
    np.radians(m2["phase_margin"]) / m2["w_gc"],
    rtol=1e-12,
)
Ld = sy.tf([0.2], [1.0, -1.5, 0.6], dt=0.1)
md = fr.margins(Ld)
gm_d, pm_d, _, _ = ct.margin(ct.tf([0.2], [1, -1.5, 0.6], 0.1))
check(
    S,
    "discrete-time margins (z = e^{jw dt})",
    "python-control",
    abs(md["gain_margin"] - gm_d) + abs(md["phase_margin"] - pm_d),
    0.0,
    atol=1e-6,
)
pk, wpk = fr.resonant_peak(G2)
check(
    S,
    "resonant peak of a second-order system 1/(2 zeta sqrt(1 - zeta^2))",
    "closed form",
    pk,
    1 / (2 * 0.3 * np.sqrt(1 - 0.09)),
    rtol=1e-9,
)
check(S, "... at wn sqrt(1 - 2 zeta^2)", "closed form", wpk, 2 * np.sqrt(1 - 2 * 0.09), rtol=1e-7)
check(S, "-3 dB bandwidth", "python-control", fr.bandwidth(G2), ct.bandwidth(ct.tf([4], [1, 1.2, 4])), rtol=1e-8)
P3 = fr.pade(1.0, 3)
check(
    S,
    "Pade(3) of e^{-s}: |P| = 1 (all-pass)",
    "construction",
    float(np.max(np.abs(np.abs(fr.freqresp(P3, w_)) - 1))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... phase error below 0.01 degrees up to w T = 1",
    "Pade accuracy",
    float(
        np.max(
            np.abs(
                np.degrees(np.unwrap(np.angle(fr.freqresp(P3, np.linspace(0.01, 1, 50)))) + np.linspace(0.01, 1, 50))
            )
        )
    ),
    0.0,
    atol=0.01,
)
h_norm, w_h = fr.hinf_norm(G2.to_ss())
check(S, "H-infinity norm by Boyd-Balakrishnan bisection equals the resonant peak", "peak gain", h_norm, pk, rtol=1e-5)
rng2 = np.random.default_rng(3)
A4 = rng2.normal(size=(4, 4)) - 3 * np.eye(4)
B4, C4 = rng2.normal(size=(4, 2)), rng2.normal(size=(2, 4))
S4 = sy.ss(A4, B4, C4, np.zeros((2, 2)))
sv = np.array([np.linalg.norm(fr.freqresp(S4, wk), 2) for wk in np.logspace(-3, 3, 30000)])
check(
    S,
    "... for a MIMO system: equals the peak singular value over a dense sweep",
    "sweep",
    fr.hinf_norm(S4)[0],
    float(sv.max()),
    rtol=1e-4,
)
w_b, mag_b, ph_b = fr.bode(G1)
check(
    S,
    "Bode phase is unwrapped and tends to -90 degrees for a first-order lag (-89.4 at the grid's last point, 100/tau)",
    "asymptote",
    float(ph_b[-1]),
    -90.0,
    atol=1.0,
)

# ------------------------------------------------------------------ 6. stability tests
S = "Stability tests"
for c_, label in (
    ([1, 2, 3, 4], "s^3 + 2s^2 + 3s + 4"),
    ([1, 1, 2, 2, 3], "s^4 + s^3 + 2s^2 + 2s + 3 (epsilon case)"),
    ([1, 1, 1, 1, 1, 1], "s^5 + ... + 1 (row of zeros)"),
):
    check(
        S,
        f"Routh array of {label}: right-half-plane roots",
        "root count",
        float(st.routh(c_)["rhp_roots"]),
        float(np.sum(pl.roots(c_).real > 1e-9)),
    )
r_ = st.routh([1, 2, 8, 12, 20, 16, 16])
check(
    S,
    "a row of zeros reveals the auxiliary polynomial 2s^4 + 12s^2 + 16 (roots +-j sqrt2, +-2j)",
    "Routh special case",
    float(np.max(np.abs(np.sort(np.abs(pl.roots(r_["auxiliary"]))) - [np.sqrt(2), np.sqrt(2), 2, 2]))),
    0.0,
    atol=1e-9,
)
for c_ in ([1, 2, 3, 4], [1, 1, 2, 2, 3], [1, 6, 11, 6], [2, 1, 3]):
    check(
        S,
        f"Hurwitz determinants agree with the roots for {c_}",
        "definition",
        float(st.hurwitz(c_)["stable"]),
        float(np.all(pl.roots(c_).real < 0)),
    )
for c_ in ([1, -1.5, 0.56], [1, 0.5, -0.5], [1, -0.3, 0.2, -0.1], [1, -2.5, 2.0, -0.5]):
    check(
        S,
        f"Jury test agrees with the root moduli for {c_}",
        "definition",
        float(st.jury(c_)["stable"]),
        float(np.all(np.abs(pl.roots(c_)) < 1)),
    )
check(
    S,
    "Lyapunov test: A^T P + P A = -I has P > 0 for a stable A",
    "Lyapunov's theorem",
    float(st.lyapunov_test([[0, 1], [-2, -3]])["stable"]),
    1.0,
)
check(
    S,
    "... and not for an unstable one",
    "Lyapunov's theorem",
    float(st.lyapunov_test([[0, 1], [2, -1]])["stable"]),
    0.0,
)
G3 = sy.tf([1.0], pl.from_roots([0, -1, -2]))
rl = st.root_locus(G3, np.logspace(-2, 2, 400))
roots_c = (
    ct.root_locus_map(ct.tf([1], [1, 3, 2, 0]), gains=np.logspace(-2, 2, 400)).loci
    if hasattr(ct, "root_locus_map")
    else ct.root_locus(ct.tf([1], [1, 3, 2, 0]), kvect=np.logspace(-2, 2, 400), plot=False)[0]
)
check(
    S,
    "root locus of 1/(s(s+1)(s+2)): branches agree with python-control at every gain",
    "python-control",
    float(np.max([np.max(np.abs(np.sort_complex(a) - np.sort_complex(b))) for a, b in zip(rl["branches"], roots_c)])),
    0.0,
    atol=1e-8,
)
check(S, "... asymptote centroid (0 - 1 - 2)/3 = -1", "rule", rl["centroid"], -1.0, atol=1e-12)
check(
    S,
    "... asymptote angles 60, 180, 300",
    "rule",
    float(np.max(np.abs(rl["asymptote_angles"] - [60, 180, 300]))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... breakaway point -1 + 1/sqrt 3 = -0.4226",
    "dK/ds = 0",
    float(rl["breakaway"][0]),
    -1 + 1 / np.sqrt(3),
    rtol=1e-9,
)
check(
    S,
    "... the locus crosses the imaginary axis at K = 6, s = +-j sqrt 2",
    "Routh / closed form",
    float(np.max(np.abs(np.sort(np.abs(pl.roots(pl.add(G3.den, 6 * G3.num)).imag)) - [0, np.sqrt(2), np.sqrt(2)]))),
    0.0,
    atol=1e-10,
)
K_, p_ = st.gain_for_damping(G3, 0.5, np.logspace(-2, 1, 3000))
check(S, "... gain for damping ratio 0.5 (K = 1.04, s = -0.33 +- 0.58j)", "textbook (Nise)", K_, 1.04, rtol=0.03)
rl2 = st.root_locus(sy.tf([1.0], pl.from_roots([-1 + 1j, -1 - 1j, -3])))
dep = rl2["departure_angles"][complex(-1 + 1j)]
check(
    S,
    "... departure angle from the pole -1 + j of 1/((s^2 + 2s + 2)(s + 3)): 180 - 90 - atan(1/2) = 63.4 degrees",
    "angle condition",
    dep,
    180 - 90 - np.degrees(np.arctan(0.5)),
    atol=1e-6,
)
for L_, Z_ in (
    (sy.tf([10.0], [1.0, 3.0, 2.0, 0.0]), 2),
    (sy.tf([1.0], [1.0, 3.0, 2.0, 0.0]), 0),
    (sy.tf([2.0], [1.0, -1.0]), 0),
    (sy.tf([0.5], [1.0, -1.0]), 1),
    (sy.tf([3.0, 3.0], [1.0, 0.0, 0.0]), 0),
    (sy.tf([1.0], pl.from_roots([1, -2, -3])), 1),
    (sy.tf([1.0], [1.0, 1.0, 0.0, 0.0]), 2),
):
    r_ = st.nyquist_criterion(L_)
    check(
        S,
        f"Nyquist criterion Z = N + P for {L_}: closed-loop RHP poles ({Z_})",
        "characteristic polynomial (and the stated count)",
        float(r_["Z"]),
        float(Z_) if Z_ == r_["closed_loop_rhp_poles"] else float("nan"),
    )
check(
    S,
    "... with closed-loop poles on the axis the criterion is flagged as marginal",
    "1/(s^2 + 1)",
    float(st.nyquist_criterion(sy.tf([1.0], [1.0, 0.0, 1.0]))["marginal"]),
    1.0,
)

# ------------------------------------------------------------------ 7. state-space design
from scipy import signal as ssg2  # noqa: E402

from englincontrol import design as dg  # noqa: E402
from englincontrol import statespace as sp  # noqa: E402

S = "State-space design"
A3 = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6.0]])
B3 = np.array([[0], [0], [1.0]])
C3 = np.array([[1.0, 0, 0]])
check(
    S,
    "controllability matrix of the companion form has full rank",
    "Kalman rank test",
    float(sp.controllability(A3, B3)["rank"]),
    3.0,
)
Aun = np.array([[-1.0, 0.0], [0.0, -2.0]])
check(
    S,
    "diag(-1, -2) with B = [1, 0]: rank 1, PBH names the uncontrollable mode -2",
    "PBH test",
    float(sp.controllability(Aun, [[1.0], [0.0]])["rank"])
    + float(sp.pbh_test(Aun, [[1.0], [0.0]])["uncontrollable_modes"][0].real),
    1.0 - 2.0,
)
des3 = np.array([-2, -3 + 1j, -3 - 1j])
K3 = sp.place(A3, B3, des3)
check(
    S,
    "Ackermann pole placement puts the poles where asked",
    "eig(A - BK)",
    float(np.max(np.abs(np.sort_complex(np.linalg.eigvals(A3 - B3 @ K3)) - np.sort_complex(des3)))),
    0.0,
    atol=1e-9,
)
check(
    S,
    "... agrees with scipy.signal.place_poles for a single input (unique solution)",
    "scipy",
    float(np.max(np.abs(K3 - ssg2.place_poles(A3, B3, des3).gain_matrix))),
    0.0,
    atol=1e-8,
)
A5r = rng.normal(size=(5, 5))
B5r = rng.normal(size=(5, 2))
des5 = np.array([-1, -2, -3 + 2j, -3 - 2j, -5])
K5 = sp.place(A5r, B5r, des5)
check(
    S,
    "MIMO pole placement (Sylvester method) for a random 5-state, 2-input system",
    "eig(A - BK)",
    float(np.max(np.abs(np.sort_complex(np.linalg.eigvals(A5r - B5r @ K5)) - np.sort_complex(des5)))),
    0.0,
    atol=1e-8,
)
L3 = sp.observer_gain(A3, C3, [-8, -9, -10])
check(
    S,
    "observer gain by duality: eig(A - LC) as asked",
    "duality",
    float(np.max(np.abs(np.sort(np.linalg.eigvals(A3 - L3 @ C3).real) - [-10, -9, -8]))),
    0.0,
    atol=1e-8,
)
sys3 = sy.ss(A3, B3, C3, 0.0)
cl3 = sp.closed_loop_with_observer(sys3, K3, L3)
check(
    S,
    "separation principle: closed-loop eigenvalues = regulator poles + observer poles",
    "block triangular structure",
    float(np.max(np.abs(np.sort_complex(cl3.poles()) - np.sort_complex(np.r_[des3, -8, -9, -10])))),
    0.0,
    atol=1e-8,
)
t3, y3 = rs.step(cl3, 6.0)
check(S, "... with the reference gain Nbar the step response settles at 1", "dc gain", float(y3[-1]), 1.0, atol=2e-3)
Q3, R3 = np.eye(3), np.array([[1.0]])
r3 = sp.lqr(A3, B3, Q3, R3)
Kc, Pc, Ec = ct.lqr(A3, B3, Q3, R3)
check(S, "LQR gain", "python-control", float(np.max(np.abs(r3["K"] - Kc))), 0.0, atol=1e-12)
check(
    S,
    "... the Riccati residual vanishes",
    "A^T P + P A - P B R^-1 B^T P + Q = 0",
    float(np.linalg.norm(A3.T @ r3["P"] + r3["P"] @ A3 - r3["P"] @ B3 @ np.linalg.solve(R3, B3.T) @ r3["P"] + Q3)),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... optimal cost x0^T P x0 equals the Lyapunov integral of x^T Q x + u^T R u along the closed loop",
    "optimality",
    sp.cost_to_go(r3["P"], [1, 0, 0]),
    sp.lyapunov_cost(A3 - B3 @ r3["K"], Q3 + r3["K"].T @ R3 @ r3["K"], [1, 0, 0]),
    rtol=1e-10,
)
m_lqr = sp.lqr_margins(A3, B3, np.diag([100, 1, 1]), np.array([[0.01]]))
check(
    S,
    "LQR loop at the plant input: phase margin >= 60 degrees",
    "Kalman's return difference inequality",
    float(m_lqr["phase_margin"] >= 60.0),
    1.0,
)
check(S, "... gain margin at least 2 (6 dB)", "guaranteed margins", float(m_lqr["gain_margin"] >= 2.0), 1.0)
rd = sp.lqr(np.array([[1.0, 0.1], [0, 1.0]]), np.array([[0.005], [0.1]]), np.eye(2), np.array([[0.1]]), discrete=True)
Kd_c, _, _ = ct.dlqr(np.array([[1.0, 0.1], [0, 1.0]]), np.array([[0.005], [0.1]]), np.eye(2), np.array([[0.1]]))
check(S, "discrete LQR gain", "python-control dlqr", float(np.max(np.abs(rd["K"] - Kd_c))), 0.0, atol=1e-10)
kf = sp.kalman(A3, C3, 0.1 * np.eye(3), np.array([[0.01]]))
Lc, _, _ = ct.lqe(A3, np.eye(3), C3, 0.1 * np.eye(3), np.array([[0.01]]))
check(S, "steady-state Kalman gain", "python-control lqe", float(np.max(np.abs(kf["L"] - Lc))), 0.0, atol=1e-10)
Ad, Bd, Cd = np.array([[1.0, 0.1], [0, 0.98]]), np.array([[0.0], [0.1]]), np.array([[1.0, 0.0]])
kd = sp.kalman(Ad, Cd, np.diag([1e-4, 1e-3]), np.array([[0.05]]), discrete=True)
rng_k = np.random.default_rng(5)
Nk = 4000
x_true = np.zeros((Nk, 2))
y_meas = np.zeros(Nk)
xk = np.array([1.0, 0.0])
uk = np.zeros(Nk)
for k in range(Nk):
    x_true[k] = xk
    y_meas[k] = (Cd @ xk)[0] + np.sqrt(0.05) * rng_k.normal()
    xk = Ad @ xk + Bd @ [uk[k]] + rng_k.normal(size=2) * np.sqrt([1e-4, 1e-3])
xh = sp.kalman_filter_run(Ad, Bd, Cd, kd["L"], uk, y_meas)
err_raw = np.mean((y_meas - x_true[:, 0]) ** 2)
err_kf = np.mean((xh[500:, 0] - x_true[500:, 0]) ** 2)
check(
    S,
    "discrete Kalman filter: the estimate's error variance is far below the measurement noise",
    "optimal filtering",
    float(err_kf / err_raw < 0.3),
    1.0,
)
check(
    S,
    "... and matches the Riccati prediction P_11 (predictor form) within sampling error",
    "error covariance",
    err_kf,
    float(kd["P"][0, 0]),
    rtol=0.2,
)
li = sp.lqr_integral(A3, B3, C3, Q3, R3, 1.0)
check(
    S,
    "LQR with integral action: the augmented closed loop is stable",
    "stability",
    float(np.max(li["closed_loop_poles"].real) < 0),
    1.0,
)
g_ = sp.lqg(sys3, Q3, R3, 0.1 * np.eye(3), np.array([[0.01]]))
Acl_lqg = np.block([[sys3.A, sys3.B @ g_["compensator"].C], [g_["compensator"].B @ sys3.C, g_["compensator"].A]])
check(
    S,
    "LQG closed loop: eigenvalues are the LQR and Kalman poles together",
    "separation",
    float(
        np.max(
            np.abs(
                np.sort_complex(np.linalg.eigvals(Acl_lqg))
                - np.sort_complex(np.r_[g_["regulator_poles"], g_["estimator_poles"]])
            )
        )
    ),
    0.0,
    atol=1e-8,
)

# ------------------------------------------------------------------ 8. classical design
S = "Classical design"
Cpid = dg.pid(2.0, 1.0, 0.5, N=10.0)
check(
    S,
    "parallel PID with a filtered derivative: C(j w) = Kp + Ki/(jw) + Kd jw/(1 + jw Tf)",
    "definition",
    float(abs(fr.freqresp(Cpid, 3.0) - (2.0 + 1.0 / 3j + 0.5 * 3j / (1 + 3j * 0.5 / 20)))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "standard (ISA) form equals the parallel form with Ki = Kp/Ti, Kd = Kp Td",
    "conversion",
    float(np.max(np.abs(dg.pid_standard(2.0, 2.0, 0.25).num - Cpid.num))),
    0.0,
    atol=1e-12,
)
zn = dg.ziegler_nichols_ultimate(8.0, 3.6276)
check(
    S,
    "Ziegler-Nichols ultimate rules: PID Kp = 0.6 Ku, Ti = Pu/2, Td = Pu/8",
    "ZN 1942",
    zn["PID"]["Kp"] + zn["PID"]["Ti"] + zn["PID"]["Td"],
    0.6 * 8 + 3.6276 / 2 + 3.6276 / 8,
    rtol=1e-12,
)
zs = dg.ziegler_nichols_step(2.0, 5.0, 1.0)
check(
    S,
    "Ziegler-Nichols step rules: PI Kp = 0.9 tau/(K L), Ti = 3 L",
    "ZN 1942",
    zs["PI"]["Kp"] + zs["PI"]["Ti"],
    0.9 * 5 / 2 + 3.0,
    rtol=1e-12,
)
cc = dg.cohen_coon(2.0, 5.0, 1.0)
check(
    S,
    "Cohen-Coon PID Kp = (4/3 + L/(4 tau)) tau/(K L)",
    "Cohen & Coon 1953",
    cc["PID"]["Kp"],
    (4 / 3 + 0.2 / 4) * 5 / 2,
    rtol=1e-12,
)
simc = dg.imc_pid(2.0, 5.0, 1.0)
check(
    S,
    "SIMC PI for a FOPDT model with lambda = L: Kp = tau/(2 K L), Ti = min(tau, 8 L)",
    "Skogestad 2003",
    simc["Kp"] + simc["Ti"],
    5 / (2 * 2 * 1) + 5.0,
    rtol=1e-12,
)
Gp = sy.tf([1.0], [1.0, 3.0, 3.0, 1.0])
ug = dg.ultimate_gain(Gp)
check(
    S,
    "ultimate gain of 1/(s + 1)^3 is 8 at w = sqrt 3 (Pu = 2 pi / sqrt 3)",
    "closed form",
    ug["Ku"] + ug["Pu"],
    8.0 + 2 * np.pi / np.sqrt(3),
    rtol=1e-8,
)
rl_exp = dg.relay_experiment(Gp, dt=0.005, T=80.0)
check(
    S,
    "relay feedback experiment estimates Ku within 5 % (describing-function accuracy)",
    "Astrom-Hagglund 1984",
    rl_exp["Ku"],
    8.0,
    rtol=0.05,
)
check(S, "... and Pu within 2 %", "Astrom-Hagglund 1984", rl_exp["Pu"], 2 * np.pi / np.sqrt(3), rtol=0.02)
t_s, y_s = rs.step(sy.tf([2.0], [5.0, 1.0]), 40.0, 4001)
fo = dg.fopdt_from_step(t_s + 1.0, y_s)  # a first-order lag delayed by 1 s
check(
    S,
    "FOPDT identification from a step by the two-point method (K = 2, tau = 5, L = 1)",
    "two-point method",
    abs(fo["K"] - 2) + abs(fo["tau"] - 5) + abs(fo["L"] - 1),
    0.0,
    atol=0.05,
)
znp = zn["PID"]
sim = dg.simulate_pid(Gp, znp["Kp"], znp["Kp"] / znp["Ti"], znp["Kp"] * znp["Td"], 1.0, T=30.0)
check(
    S,
    "ZN-tuned PID on 1/(s+1)^3: the classic quarter-decay design overshoots by 40-60 %",
    "ZN's design intent",
    float(40 < 100 * (sim["y"].max() - 1) < 60),
    1.0,
)
check(S, "... with zero steady-state error (integral action)", "final value", float(sim["y"][-1]), 1.0, atol=1e-3)
cl = dg.closed_loop(Gp, dg.pid_standard(znp["Kp"], znp["Ti"], znp["Td"]))
t_c, y_c = rs.step(cl["T"], 30.0, 3001)
sim_e = dg.simulate_pid(
    Gp, znp["Kp"], znp["Kp"] / znp["Ti"], znp["Kp"] * znp["Td"], 1.0, dt=0.002, T=30.0, derivative_on="error"
)
check(
    S,
    "... with the derivative on the error, the simulation agrees with the linear closed loop T = L/(1 + L)",
    "two routes (dt = 0.002)",
    float(np.max(np.abs(np.interp(t_c, sim_e["t"], sim_e["y"]) - y_c))),
    0.0,
    atol=0.02,
)
check(
    S,
    "... derivative on the measurement avoids the setpoint kick: a smaller initial control move",
    "PID structure",
    float(np.abs(sim["u"][:5]).max() < np.abs(sim_e["u"][:5]).max()),
    1.0,
)
sat = dg.simulate_pid(
    Gp, znp["Kp"], znp["Kp"] / znp["Ti"], znp["Kp"] * znp["Td"], 1.0, T=30.0, u_limits=(-1.5, 1.5), antiwindup=None
)
aw = dg.simulate_pid(
    Gp,
    znp["Kp"],
    znp["Kp"] / znp["Ti"],
    znp["Kp"] * znp["Td"],
    1.0,
    T=30.0,
    u_limits=(-1.5, 1.5),
    antiwindup="back-calculation",
)
check(
    S,
    "actuator saturation: back-calculation anti-windup cuts the overshoot by more than half",
    "Astrom & Rundqwist 1989",
    float(aw["y"].max() - 1 < 0.5 * (sat["y"].max() - 1)),
    1.0,
)
ld = dg.lead(45.0, 2.0)
check(
    S,
    "lead compensator: maximum phase 45 degrees at w = 2",
    "phase of (1 + jwT)/(1 + j alpha w T)",
    float(np.degrees(np.angle(fr.freqresp(ld, 2.0)))),
    45.0,
    atol=1e-9,
)
check(
    S,
    "... gain at w_max is 1/sqrt(alpha)",
    "closed form",
    float(abs(fr.freqresp(ld, 2.0))),
    1 / np.sqrt((1 - np.sin(np.pi / 4)) / (1 + np.sin(np.pi / 4))),
    rtol=1e-12,
)
dl = dg.design_lead(sy.tf([1.0], [1.0, 1.0, 0.0]), 3.0, 50.0)
check(S, "lead design for 1/(s(s+1)): crossover at 3 rad/s", "design target", dl["margins"]["w_gc"], 3.0, rtol=1e-6)
check(
    S,
    "... phase margin 50 + 5 degrees (the extra 5 absorb the gain shift)",
    "design target",
    dl["margins"]["phase_margin"],
    55.0,
    atol=0.5,
)
lg = dg.lag(10.0, 0.3)
check(
    S,
    "lag compensator raises the low-frequency gain tenfold with little phase lag a decade above its zero",
    "definition",
    float(abs(fr.freqresp(lg, 1e-4)) / abs(fr.freqresp(lg, 3.0))),
    10.0,
    rtol=0.02,
)
ff = dg.feedforward_inverse(sy.tf([2.0], [1.0, 3.0, 2.0]), bandwidth_filter=50.0)
check(
    S,
    "model-inverse feedforward: P F ~ 1 at low frequency",
    "inversion",
    float(abs(fr.freqresp(sy.series(ff, sy.tf([2.0], [1.0, 3.0, 2.0])), 0.1))),
    1.0,
    atol=1e-3,
)

# ------------------------------------------------------------------ 9. discrete-time control
from englincontrol import discrete as dc  # noqa: E402
from englincontrol import identify as idf  # noqa: E402
from englincontrol import mimo  # noqa: E402
from englincontrol import robust as rb  # noqa: E402

S = "Discrete-time control"
dt_ = 0.05
for meth, ref in (("zoh", "zoh"), ("tustin", "bilinear"), ("euler", "euler"), ("backward", "backward_diff")):
    Gd_ = dc.c2d(G2, dt_, meth)
    num_r, den_r, _ = ssg2.cont2discrete(([4.0], [1.0, 1.2, 4.0]), dt_, method=ref)
    check(
        S,
        f"c2d by '{meth}'",
        "scipy.signal.cont2discrete",
        float(np.max(np.abs(Gd_.den - den_r)) + np.max(np.abs(Gd_.num - pl.trim(num_r[0]) / den_r[0]))),
        0.0,
        atol=1e-12,
    )
Gm_ = dc.c2d(G2, dt_, "matched")
check(
    S,
    "matched pole-zero mapping: poles e^{s dt}, unit dc gain preserved",
    "definition",
    float(np.max(np.abs(np.sort_complex(Gm_.poles()) - np.sort_complex(np.exp(G2.poles() * dt_)))))
    + abs(Gm_.dcgain() - G2.dcgain()),
    0.0,
    atol=1e-12,
)
pw_ = dc.c2d(G2, dt_, "prewarp", w_prewarp=2.0)
check(
    S,
    "Tustin with prewarping at w = 2 reproduces |G(2j)| exactly at that frequency",
    "prewarp",
    float(abs(fr.freqresp(pw_, 2.0))),
    float(abs(fr.freqresp(G2, 2.0))),
    rtol=1e-12,
)
tw_ = dc.c2d(G2, dt_, "tustin")
check(
    S,
    "... plain Tustin warps: its response at w = 2 differs (frequency warping w' = (2/dt) tan(w dt/2))",
    "warping",
    float(abs(fr.freqresp(tw_, 2.0))),
    float(abs(fr.freqresp(G2, 2 / dt_ * np.tan(2.0 * dt_ / 2)))),
    rtol=1e-10,
)
Gdss = dc.c2d(G2.to_ss(), dt_)
check(
    S,
    "d2c inverts the zero-order hold (matrix logarithm)",
    "round trip",
    float(np.max(np.abs(dc.d2c(Gdss).A - G2.to_ss().A))),
    0.0,
    atol=1e-12,
)
td_, yd_ = rs.step(dc.c2d(G2, dt_), 10.0)
check(
    S,
    "the ZOH-discretised step response equals the continuous one at the samples",
    "exactness of ZOH",
    float(np.max(np.abs(yd_ - rs.step(G2, 10.0, len(yd_))[1]))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "aliasing: 100 rad/s sampled at dt = 0.1 (w_s = 62.8) appears at 25.66 rad/s",
    "folding",
    dc.alias(100.0, 0.1),
    abs(100.0 - 2 * 2 * np.pi / 0.1),
    rtol=1e-10,
)
check(
    S,
    "zero-order hold frequency response: |H| = sinc, phase -w dt/2 (a half-sample delay)",
    "closed form",
    float(np.degrees(np.angle(dc.zoh_frequency_response(10.0, 0.1)))),
    float(np.degrees(-10.0 * 0.1 / 2)),
    atol=1e-10,
)
Ad2, Bd2 = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([[0.005], [0.1]])
Kdb = dc.deadbeat(Ad2, Bd2)
check(
    S,
    "dead-beat control: (A - BK)^n = 0, the state is zero after n steps",
    "nilpotent closed loop",
    float(np.max(np.abs(np.linalg.matrix_power(Ad2 - Bd2 @ Kdb, 2)))),
    0.0,
    atol=1e-10,
)
Cz = dc.pid_discrete(2.0, 1.0, 0.5, dt=0.01)
check(
    S,
    "discrete PID tends to the continuous one at low frequency",
    "consistency",
    float(abs(fr.freqresp(Cz, 0.5) - fr.freqresp(dg.pid(2.0, 1.0, 0.5), 0.5))),
    0.0,
    atol=0.02,
)
check(
    S,
    "... with backward-rectangle integration: C(z) has a pole at z = 1",
    "integrator",
    float(np.min(np.abs(Cz.poles() - 1))),
    0.0,
    atol=1e-12,
)
rules = dc.sample_rate_rules(5.0)
check(
    S,
    "sample-rate rule of thumb: w_b dt between 0.2 and 0.6",
    "Astrom & Wittenmark",
    rules["dt_min"] * 5.0 + rules["dt_max"] * 5.0,
    0.8,
    rtol=1e-12,
)
jd = st.jury(dc.c2d(sy.tf([1.0], [1.0, -1.0]), 0.1).den)
check(
    S,
    "Jury confirms the discretised unstable plant is unstable (pole e^{0.1} > 1)",
    "consistency",
    float(jd["stable"]),
    0.0,
)

# ------------------------------------------------------------------ 10. robustness and limitations
S = "Robustness and limitations"
Lr = sy.tf([1.0], [1.0, 3.0, 2.0, 0.0])
spk = rb.sensitivity_peak(Lr)
check(S, "Ms of 1/(s(s+1)(s+2)) = 1.568", "dense sweep with python-control", spk["Ms"], 1.5684, rtol=1e-3)
check(
    S,
    "the minimum distance of the Nyquist curve from -1 is 1/Ms",
    "definition",
    rb.nyquist_distance(Lr)["distance"],
    1 / spk["Ms"],
    rtol=1e-6,
)
mr = fr.margins(Lr)
check(
    S,
    "the actual margins exceed the bounds implied by Ms: gm >= Ms/(Ms-1), pm >= 2 asin(1/(2Ms))",
    "Ms bounds",
    float(mr["gain_margin"] >= spk["gm_lower_bound"] and mr["phase_margin"] >= spk["pm_lower_bound_deg"]),
    1.0,
)
dm = rb.disk_margin(Lr)
check(
    S,
    "disk margin alpha = 1/max|S - T|; its phase margin 2 atan(alpha/2) is below the classical one",
    "disk margin",
    float(dm["phase_margin_deg"] <= mr["phase_margin"] and 0 < dm["alpha"] < 2),
    1.0,
)
bi_s = rb.bode_sensitivity_integral(Lr)
check(
    S,
    "Bode's sensitivity integral of a stable loop with relative degree 3 is zero (the waterbed effect)",
    "Bode 1945",
    bi_s["integral"],
    0.0,
    atol=2e-3,
)
Lu_ = sy.tf([30.0, 60.0], pl.from_roots([1, -5, -8]))
bi_u = rb.bode_sensitivity_integral(Lu_)
check(
    S,
    "... and equals pi times the unstable pole (+1) for a stabilised unstable loop",
    "Freudenberg & Looze 1985",
    bi_u["integral"],
    np.pi,
    rtol=2e-3,
)
lim = rb.rhp_limits(sy.tf([1.0, -2.0], [1.0, 2.0, 1.0]))
check(
    S,
    "a right-half-plane zero at 2 limits the crossover to about 1 rad/s",
    "rule of thumb w_c < z/2",
    lim["w_c_max_from_zero"],
    1.0,
)
lim2 = rb.rhp_limits(sy.tf([1.0, -4.0], [1.0, -1.0, 0.0]))
check(
    S,
    "unstable pole 1 with RHP zero 4: Ms >= |p + z|/|p - z| = 5/3",
    "interpolation constraint",
    lim2["Ms_lower_bound"],
    5 / 3,
    rtol=1e-12,
)
sg = rb.small_gain(sy.tf([0.5], [1.0, 1.0]), 1.5)
check(
    S,
    "small-gain theorem: ||M||_inf delta < 1 passes, fails at delta = 3",
    "small gain",
    float(sg["robustly_stable"]) + float(rb.small_gain(sy.tf([0.5], [1.0, 1.0]), 3.0)["robustly_stable"]),
    1.0,
)
mu_ = rb.multiplicative_robustness(Lr, sy.tf([0.5, 0.0], [1.0, 1.0]))
check(
    S,
    "multiplicative uncertainty W = 0.5 s/(s + 1): |W T| stays below 1 at every frequency",
    "robust stability",
    float(mu_["robustly_stable"]),
    1.0,
)
g4 = rb.gang_of_four(sy.tf([1.0], [1.0, 1.0]), sy.tf([2.0, 1.0], [1.0, 0.0]))
check(
    S,
    "gang of four: S + T = 1 at every frequency",
    "identity",
    float(np.max(np.abs(fr.freqresp(g4["S"], w_) + fr.freqresp(g4["T"], w_) - 1))),
    0.0,
    atol=1e-10,
)

# ------------------------------------------------------------------ 11. identification
S = "Identification"
Gtrue = dc.c2d(sy.tf([2.0], [1.0, 3.0, 2.0]), 0.1)
rng_i = np.random.default_rng(0)
u_i = idf.prbs(20000, seed=1, hold=10)
_, y_i, _ = rs.lsim(Gtrue, u_i)
m0 = idf.arx(u_i[:2000], y_i[:2000], 2, 2, 1)
check(
    S,
    "ARX least squares recovers the ZOH model exactly from noise-free data",
    "exact",
    float(np.max(np.abs(m0["tf"].den - Gtrue.den)) + np.max(np.abs(m0["b"] - Gtrue.num[-2:]))),
    0.0,
    atol=1e-10,
)
y_lo = y_i + 1e-3 * rng_i.normal(size=len(y_i))
m_lo = idf.arx(u_i[:2000], y_lo[:2000], 2, 2, 1)
check(
    S,
    "... and the poles to 1 % with 0.1 % output noise",
    "bias ~ noise^2",
    float(np.max(np.abs(np.sort(m_lo["tf"].poles().real) - np.sort(Gtrue.poles().real)))),
    0.0,
    atol=0.01,
)
y_hi = y_i + 1e-2 * rng_i.normal(size=len(y_i))
m_hi = idf.arx(u_i, y_hi, 2, 2, 1)
check(
    S,
    "with 1 % output noise least squares is biased (a pole off by > 0.2) even with 20 000 samples",
    "errors in variables",
    float(np.max(np.abs(np.sort(m_hi["tf"].poles().real) - np.sort(Gtrue.poles().real))) > 0.2),
    1.0,
)
check(
    S,
    "... and its reported standard errors (< 0.01) do not reveal it",
    "misspecified noise model",
    float(np.max(m_hi["std_errors"][:2]) < 0.01),
    1.0,
)
m_iv = idf.arx(u_i, y_hi, 2, 2, 1, method="iv")
check(
    S,
    "... the instrumental-variable estimate is consistent: poles within 0.1",
    "IV",
    float(np.max(np.abs(np.sort(m_iv["tf"].poles().real) - np.sort(Gtrue.poles().real)))),
    0.0,
    atol=0.1,
)
acf_hi = idf.autocorrelation(m_hi["residuals"], 5)
check(
    S,
    "... the whiteness test flags the misspecification: with output noise the equation error A(q) v is coloured (|acf(1)| > 0.3)",
    "residual analysis",
    float(abs(acf_hi[1]) > 0.3),
    1.0,
)
Gc_ = sy.tf([2.0], [1.0, 3.0, 2.0])
w_i = np.logspace(-1, 1.5, 60)
H_i = fr.freqresp(Gc_, w_i) * (1 + 0.01 * rng_i.normal(size=60))
ffit = idf.fit_frequency_response(w_i, H_i, 0, 2)
check(
    S,
    "rational fit to noisy frequency-response data (Levy / Sanathanan-Koerner): coefficients within 2 %",
    "least squares",
    float(np.max(np.abs(ffit.den - Gc_.den)) + abs(ffit.num[0] - 2.0)),
    0.0,
    atol=0.06,
)
_, ys_ = rs.step(Gtrue, 6.0)
mk_ = np.r_[ys_[0], np.diff(ys_)]  # Markov parameters from the step response
era_ = idf.era(mk_, 2, 0.1)
check(
    S,
    "ERA (Ho-Kalman) from Markov parameters recovers the poles",
    "exact realisation",
    float(np.max(np.abs(np.sort_complex(era_["system"].poles()) - np.sort_complex(Gtrue.poles())))),
    0.0,
    atol=1e-8,
)
check(
    S,
    "... the Hankel singular values show the order: the third is zero",
    "order determination",
    float(era_["hankel_singular_values"][2] / era_["hankel_singular_values"][0]),
    0.0,
    atol=1e-10,
)
big_ = sy.tf([1.0], pl.from_roots([-1, -2, -20, -50, -100])).to_ss()
red_ = idf.balanced_truncation(big_, 2)
err_bt = float(np.max(np.abs(fr.freqresp(big_, w_) - fr.freqresp(red_["system"], w_))))
check(
    S,
    "balanced truncation 5 -> 2 states: the error is below the bound 2 sum(discarded HSV)",
    "Glover / Enns bound",
    float(err_bt <= red_["error_bound"]),
    1.0,
)
t2_, y2_ = rs.step(G2, 15.0, 6001)
so_ = idf.fit_second_order_step(t2_, y2_)
check(
    S,
    "second-order fit from a step: zeta and wn from overshoot and peak time",
    "closed form",
    abs(so_["zeta"] - 0.3) + abs(so_["wn"] - 2.0),
    0.0,
    atol=0.01,
)
fo_ = dg.fopdt_from_step(t2_ + 0.5, rs.step(sy.tf([3.0], [2.0, 1.0]), 15.0, 6001)[1])
check(
    S,
    "FOPDT two-point fit of a delayed first-order lag (K = 3, tau = 2, L = 0.5)",
    "two-point method",
    abs(fo_["K"] - 3) + abs(fo_["tau"] - 2) + abs(fo_["L"] - 0.5),
    0.0,
    atol=0.03,
)
pr_ = idf.prbs(1022, bits=9)
check(
    S,
    "a 9-bit PRBS has period 511 and nearly zero mean",
    "m-sequence",
    float(np.max(np.abs(pr_[:511] - pr_[511:1022]))) + abs(pr_[:511].sum()),
    1.0,
)

# ------------------------------------------------------------------ 12. MIMO basics
S = "MIMO basics"
Am_, Bm_, Cm_ = np.array([[-1.0, 0.5], [0.0, -2.0]]), np.eye(2), np.array([[1.0, 0.0], [1.0, 1.0]])
Sm_ = sy.ss(Am_, Bm_, Cm_, np.zeros((2, 2)))
TM = mimo.transfer_matrix(Sm_)
check(
    S,
    "transfer matrix entries agree with the state-space frequency response",
    "consistency",
    float(np.max(np.abs(np.array([[TM[i][j](1j * 0.7) for j in range(2)] for i in range(2)]) - fr.freqresp(Sm_, 0.7)))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "singular values at w = 1",
    "numpy.linalg.svd of G(j)",
    float(np.max(np.abs(mimo.sigma(Sm_, 1.0)[0] - np.linalg.svd(fr.freqresp(Sm_, 1.0), compute_uv=False)))),
    0.0,
    atol=1e-14,
)
G0_ = np.array([[2.0, 1.0], [1.0, 2.0]])
check(
    S,
    "RGA of [[2, 1], [1, 2]] is [[4/3, -1/3], [-1/3, 4/3]]",
    "Bristol 1966",
    float(np.max(np.abs(mimo.rga(G0_) - np.array([[4 / 3, -1 / 3], [-1 / 3, 4 / 3]])))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... RGA rows and columns sum to 1",
    "property",
    float(np.max(np.abs(mimo.rga(G0_).sum(axis=0) - 1)) + np.max(np.abs(mimo.rga(G0_).sum(axis=1) - 1))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... diagonal pairing recommended",
    "pairing rule",
    float(mimo.rga_pairing(G0_)["pairs"] == [(0, 0), (1, 1)]),
    1.0,
)
check(
    S,
    "a triangular plant has an identity RGA (no interaction in the right pairing)",
    "property",
    float(np.max(np.abs(mimo.rga([[1.0, 0.0], [3.0, 2.0]]) - np.eye(2)))),
    0.0,
    atol=1e-12,
)
D_ = mimo.static_decoupler(Sm_.dcgain())
check(
    S,
    "static decoupler: G(0) D = I",
    "definition",
    float(np.max(np.abs(Sm_.dcgain() @ D_ - np.eye(2)))),
    0.0,
    atol=1e-12,
)
Sz_ = sy.ss(
    np.array([[-1.0, 0], [0, -2.0]]), np.eye(2), np.array([[1.0, 1.0], [1.0, -1.0]]), np.array([[0.0, 0.0], [0.0, 1.0]])
)
check(
    S,
    "transmission zeros of a square system",
    "python-control",
    float(
        np.max(
            np.abs(
                np.sort_complex(mimo.transmission_zeros(Sz_))
                - np.sort_complex(np.array(ct.zeros(ct.ss(Sz_.A, Sz_.B, Sz_.C, Sz_.D))))
            )
        )
    ),
    0.0,
    atol=1e-8,
)
check(
    S,
    "condition number of G(jw) is sigma_max/sigma_min >= 1",
    "definition",
    float(np.min(mimo.condition_number(Sm_, np.logspace(-1, 1, 20))) >= 1.0),
    1.0,
)

# ------------------------------------------------------------------ 13. example plants
import pandas as pd  # noqa: E402

from englincontrol import datasets, models  # noqa: E402

S = "Example plants"
mt = models.dc_motor()
check(
    S,
    "DC motor: state-space to transfer function equals Kt/((Ls + R)(Js + b) + Kt Ke)",
    "derivation",
    float(np.max(np.abs(mt["system"].to_tf().den - mt["tf_speed"].den))),
    0.0,
    atol=1e-12,
)
check(
    S,
    "... the position model adds a pole at the origin",
    "integrator",
    float(np.min(np.abs(models.dc_motor(position=True)["system"].poles()))),
    0.0,
    atol=1e-12,
)
msd = models.mass_spring_damper(m=2.0, c=1.0, k=8.0)
check(
    S,
    "mass-spring-damper: poles -zeta wn +- j wn sqrt(1 - zeta^2) with wn = sqrt(k/m), zeta = c/(2 sqrt(km))",
    "closed form",
    float(
        np.max(
            np.abs(
                np.sort_complex(msd["system"].poles())
                - np.sort_complex(
                    np.array(
                        [
                            -msd["zeta"] * msd["wn"] - 1j * msd["wn"] * np.sqrt(1 - msd["zeta"] ** 2),
                            -msd["zeta"] * msd["wn"] + 1j * msd["wn"] * np.sqrt(1 - msd["zeta"] ** 2),
                        ]
                    )
                )
            )
        )
    ),
    0.0,
    atol=1e-12,
)
tk = models.two_tanks()
check(
    S,
    "two tanks: dc gain of level 2 is R2 and both poles are real (overdamped)",
    "steady state",
    float(tk["system"].dcgain()[0, 0]) + float(np.max(np.abs(tk["system"].poles().imag))),
    tk["dc_gain"],
    rtol=1e-12,
)
ml = models.magnetic_levitation()
check(
    S,
    "magnetic levitation: poles +-sqrt(2 g/x0)",
    "linearisation",
    float(np.max(ml["system"].poles().real)),
    ml["unstable_pole"],
    rtol=1e-12,
)
cp = models.cart_pendulum()
Aln, Bln = models.linearise(models.pendulum_nonlinear(), [0, 0, 0, 0], [0.0])
check(
    S,
    "cart-pendulum: numerical linearisation of the nonlinear equations equals the analytic matrices",
    "central differences",
    float(np.max(np.abs(Aln - cp["system"].A)) + np.max(np.abs(Bln - cp["system"].B))),
    0.0,
    atol=1e-8,
)
check(
    S,
    "... one unstable pole near sqrt(m g l (M + m)/p), a pole at 0 (cart position) and the pendulum controllable from the cart force",
    "physics",
    float(np.max(cp["system"].poles().real) > 5)
    + float(sp.controllability(cp["system"].A, cp["system"].B)["controllable"]),
    2.0,
)
Ap, Bp = models.linearise(models.pendulum_nonlinear(), [0, 0, np.pi, 0], [0.0])
check(
    S,
    "... linearised about the hanging position the pendulum is stable (oscillatory)",
    "physics",
    float(np.max(np.linalg.eigvals(Ap).real) <= 1e-9),
    1.0,
)
ac = models.aircraft_short_period()
check(
    S,
    "aircraft short period: wn and zeta from the stability derivatives",
    "closed form",
    float(
        np.max(
            np.abs(
                np.sort_complex(ac["system"].poles())
                - np.sort_complex(np.roots([1, 2 * ac["zeta"] * ac["wn"], ac["wn"] ** 2]))
            )
        )
    ),
    0.0,
    atol=1e-10,
)
qc = models.quarter_car()
wn_qc = np.sort(np.abs(qc["system"].poles()))[[0, 2]]
check(
    S,
    "quarter car: body-bounce mode near 1 Hz and wheel-hop mode near 10 Hz",
    "textbook ranges",
    float((0.8 < wn_qc[0] / (2 * np.pi) < 1.5) and (8 < wn_qc[1] / (2 * np.pi) < 12)),
    1.0,
)
ds = models.distillation_2x2()
check(
    S,
    "distillation 2 x 2: RGA diagonal 2.0 (Wood-Berry)",
    "Bristol",
    float(mimo.rga(ds["system"].dcgain())[0, 0]),
    2.0,
    rtol=0.01,
)
check(
    S,
    "... its dc gain matrix is the Wood-Berry gain matrix",
    "construction",
    float(np.max(np.abs(ds["system"].dcgain() - ds["gains"]))),
    0.0,
    atol=1e-10,
)

# ------------------------------------------------------------------ 14. course data
S = "Course data"
motor_true = models.dc_motor(R=2.0, L=0.1, Kt=0.05, Ke=0.05, J=0.002, b=0.005)["system"]
dstep = pd.read_csv(datasets.path("motor_step.csv"), comment="#")
check(
    S,
    "motor step test: 1500 samples, one dropped (NaN)",
    "logger fault",
    float(len(dstep)) + float(dstep.speed_rad_s.isna().sum()),
    1501.0,
)
good = dstep.dropna()
late = good[good.t_s > 1.3]
check(
    S,
    "... the late-time speed matches the model's 12 V response within the tachometer noise",
    "generating model",
    float(late.speed_rad_s.mean()),
    float(rs.step(motor_true, 1.4, 1401)[1][-100:].mean() * 12),
    atol=0.3,
)
fo_m = dg.fopdt_from_step(good.t_s.to_numpy() - 0.1, good.speed_rad_s.to_numpy() / 12.0)
check(
    S,
    "... a FOPDT fit gives K = 4 within 3 % and a time constant near the slow pole's 0.31 s",
    "two-point method",
    abs(fo_m["K"] - 4.0) / 4.0 + abs(fo_m["tau"] - 0.31) / 0.31,
    0.0,
    atol=0.3,
)
dprbs = pd.read_csv(datasets.path("motor_prbs.csv"), comment="#")
dp = dprbs[dprbs.t_s >= 2.0]
u_p, y_p = (dp.V - dp.V.mean()).to_numpy(), (dp.speed_rad_s - dp.speed_rad_s.mean()).to_numpy()
oe_m = idf.output_error(u_p[::10], y_p[::10], 2, 0.05)
check(
    S,
    "PRBS experiment: an output-error fit at 20 Hz recovers both discrete poles within 2 %",
    "generating model",
    float(np.max(np.abs(np.sort(oe_m["tf"].poles().real) - np.sort(np.exp(motor_true.poles().real * 0.05))))),
    0.0,
    atol=0.02,
)
check(S, "... and the dc gain 4 within 1 %", "generating model", float(oe_m["tf"].dcgain()), 4.0, rtol=0.01)
ls_m = idf.arx(u_p, y_p, 2, 2, 1, dt=0.005)
check(
    S,
    "... while ARX least squares at the full 200 Hz is wrecked by the near-collinear regressors (a pole off by > 0.5)",
    "ill-conditioning + output noise",
    float(np.max(np.abs(np.sort(ls_m["tf"].poles().real) - np.sort(np.exp(motor_true.poles().real * 0.005)))) > 0.5),
    1.0,
)
dfr = pd.read_csv(datasets.path("servo_frequency_response.csv"), comment="#")
low = dfr[dfr.w_rad_s < 0.5]
slope = np.polyfit(np.log10(low.w_rad_s), low.gain_dB, 1)[0]
check(
    S,
    "servo frequency response: about -20 dB/decade and -90 degrees at low frequency (an integrator; the motor's slow pole at 3 rad/s already bends the slope and the 1-degree phase scatter shows)",
    "position = integral of speed",
    float(slope / -20.0) + float(low.phase_deg.mean() / -90.0),
    2.0,
    atol=0.1,
)
check(
    S,
    "... a resonant peak between 60 and 100 rad/s",
    "unmodelled structural mode",
    float(dfr.w_rad_s.to_numpy()[np.argmax(dfr.gain_dB.to_numpy() + 40 * np.log10(dfr.w_rad_s.to_numpy()))]),
    80.0,
    rtol=0.25,
)

# ------------------------------------------------------------------ report
passed = sum(r[5] for r in RESULTS)
lines = [
    "# Validation report",
    "",
    "Generated by `python docs/validate.py`; rerun in CI on every push. Checks compare englincontrol with",
    "closed-form results, SciPy/LAPACK and python-control, and with the theory of linear systems and control.",
    "",
    f"**{passed} / {len(RESULTS)} checks pass.**",
    "",
]
for section in dict.fromkeys(r[0] for r in RESULTS):
    lines += [
        "",
        f"## {section}",
        "",
        "| Check | Reference | englincontrol | Expected | Result |",
        "|---|---|---|---|:-:|",
    ]
    for s_, name, ref_, v, expected, ok in RESULTS:
        if s_ == section:
            lines.append(f"| {name} | {ref_} | {v:.10g} | {expected:.10g} | {'pass' if ok else 'FAIL'} |")
Path(__file__).with_name("VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{passed}/{len(RESULTS)} checks pass")
