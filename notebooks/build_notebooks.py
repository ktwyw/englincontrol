"""Generate the course notebooks (single source of truth) and execute them.

python notebooks/build_notebooks.py            # write and execute all notebooks
python notebooks/build_notebooks.py 03 15      # only notebooks whose names start with 03 or 15
python notebooks/build_notebooks.py --no-run   # write without executing
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS: dict[str, list] = {}
EXTRAS: dict[str, dict] = {}

COLAB = """try:                      # on Google Colab (or anywhere englincontrol is missing): install it from GitHub
    import englincontrol
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ktwyw/englincontrol"], check=True)"""

STYLE = '''plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

import inspect
from IPython.display import Code

def show_source(obj):
    """Display the source code of a library function or class."""
    return Code(inspect.getsource(obj), language="python")

def heading(text):
    print(f"\\n{text}\\n" + "=" * len(text))'''


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def setup_cell(extra_imports=()):
    lines = ["import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"]
    for imp in extra_imports:
        if imp not in lines:
            lines.append(imp)
    return code(COLAB + "\n" + "\n".join(lines) + "\n" + STYLE)


def read_data(name):
    """Code snippet reading a bundled CSV file (comment lines start with #)."""
    return f'pd.read_csv(datasets.path("{name}"), comment="#")'


# =====================================================================================================



NOTEBOOKS["00_models_of_linear_systems"] = [
    md(r"""
# 00 · Models of linear systems

Everything in linear control starts with a model: a set of first-order differential equations x′ = Ax + Bu,
y = Cx + Du (state space), or a ratio of polynomials in the Laplace variable, G(s) = N(s)/D(s) (transfer
function). They describe the same object, and the first skill of the subject is moving between them without
losing anything: realising a transfer function as a state-space model, extracting the transfer function of a
state-space model, recognising when a model carries hidden (uncontrollable or unobservable) modes, and
assembling models of interconnected systems. This notebook does all of this on the plants the course will use.
"""),
    setup_cell(["from englincontrol import systems as sy, models, response as rs"]),
    md("## A plant from physics: the DC motor"),
    code(r"""m = models.dc_motor()
G = m["system"]
print(G); print("states:", m["states"], "| input:", m["input"], "| output:", m["output"])
print("A =\n", G.A, "\nB =", G.B.ravel(), " C =", G.C.ravel())
Gtf = G.to_tf()
print("\ntransfer function V -> speed:", Gtf)
print("poles", np.round(np.sort(G.poles().real), 4), "  (electrical and mechanical time constants:", np.round(-1 / np.sort(G.poles().real), 3), "s)")
print("dc gain", np.round(G.dcgain()[0, 0], 4), "rad/s per volt;  Kt/(R b + Kt Ke) =", m["params"]["Kt"] / (m["params"]["R"] * m["params"]["b"] + m["params"]["Kt"] * m["params"]["Ke"]))"""),
    md(r"""
Kirchhoff's law for the armature and Newton's law for the rotor give two first-order equations in the current
and the speed - a state-space model whose matrices contain the physical parameters directly. Its transfer
function from voltage to speed, Kt/((Ls + R)(Js + b) + KtKe), follows by eliminating the current; the library
computes it from (A, B, C, D) exactly, by the Leverrier-Faddeev recursion, without ever computing eigenvalues.
The poles are the electrical and mechanical time constants coupled by the back-emf, and the dc gain is the
steady speed per volt.

## State space and transfer function: two views of one object
"""),
    code(r"""H = sy.tf([2.0, 1.0], [1.0, 3.0, 2.0])                      # (2s + 1) / ((s + 1)(s + 2))
S_c = H.to_ss()                                                 # controllable canonical form
S_o = H.to_ss("observable")
print("controllable canonical A =\n", S_c.A, "\nobservable canonical A =\n", S_o.A)
print("both realise the same transfer function:", S_c.to_tf(), "and", S_o.to_tf())
modal, T = sy.modal_form(S_c)
print("modal form A =\n", np.round(modal.A, 10), " <- the poles on the diagonal")
print("a similarity transformation changes the state but not the input-output map:", np.allclose(S_c.transform(np.array([[2.0, 1.0], [0.5, 3.0]])).to_tf().den, H.den))"""),
    md(r"""
A transfer function has infinitely many realisations: the controllable canonical form (the denominator's
coefficients in the first row of A), its transpose the observable form, the modal form with the poles on the
diagonal, and any similarity transformation of these. They share the transfer function - the input-output
behaviour - and differ in what the state means. Numerically they are not equivalent: the companion forms are
badly conditioned for large order (their coefficients span many orders of magnitude), which is one reason
state-space computations prefer balanced or modal forms.

## Poles, zeros and the response they imply
"""),
    code(r"""for name, sys_ in (("DC motor", Gtf), ("(2s + 1)/((s + 1)(s + 2))", H), ("non-minimum phase (1 - s)/((s + 1)(s + 2))", sy.tf([-1.0, 1.0], [1.0, 3.0, 2.0]))):
    print(f"{name:<44}: poles {np.round(np.sort_complex(sys_.poles()), 3)}, zeros {np.round(sys_.zeros(), 3)}, dc gain {sys_.dcgain():.4f}")
fig, ax = plt.subplots(figsize=(8, 3.5))
for name, sys_ in (("minimum phase", H), ("non-minimum phase", sy.tf([-1.0, 1.0], [1.0, 3.0, 2.0]))):
    t, y = rs.step(sys_, 6.0)
    ax.plot(t, y, label=name)
ax.axhline(0, color="k", lw=0.5); ax.legend(); ax.set_xlabel("t"); ax.set_title("the right-half-plane zero makes the response start the wrong way"); plt.show()"""),
    md(r"""
The poles set the modes (e^{−t}, e^{−2t} here) and the zeros set how the modes are mixed. A zero in the right
half-plane is the signature of a non-minimum-phase system: its step response starts in the wrong direction
before recovering - a steam-engine's "inverse response", the reverse swing of a boat's stern when the rudder
turns - and it will limit how fast any controller can make the loop (notebook 14).

## Interconnections and hidden modes
"""),
    code(r"""P = sy.tf([1.0], [1.0, 1.0])                  # plant
C = sy.tf([2.0, 1.0], [1.0, 0.0])             # PI controller (2s + 1)/s
L = sy.series(C, P)                            # open loop C P
T = sy.feedback(L)                             # closed loop L / (1 + L)
print("L =", L, "\nT =", T, " poles", np.round(np.sort_complex(T.poles()), 4))
Tss = sy.feedback(sy.series(C.to_ss(), P.to_ss()))
print("the same loop in state space:", Tss, "poles", np.round(np.sort_complex(Tss.poles()), 4))
par = sy.parallel(H, sy.tf([1.0], [1.0, 1.0])).to_ss()
print(f"\n(2s+1)/((s+1)(s+2)) + 1/(s+1) realised naively has {par.n} states; minimal realisation: {sy.minimal(par).n} state,", sy.minimal(par).to_tf())"""),
    md(r"""
Series, parallel and feedback connections are polynomial algebra in the transfer-function world and block
matrices in state space; the library does both and they agree. The parallel example hides a trap: the naive
realisation of a sum has the states of both terms, but the sum (2s+1)/((s+1)(s+2)) + 1/(s+1) = 3/(s+2) has
only one - the (s+1) mode is uncontrollable or unobservable in the combination and cancels. Minimal
realisation removes such modes; notebook 08 shows why they matter (a cancelled unstable mode is a disaster).

## The course's plants
"""),
    code(r"""rows = []
for name in ("dc_motor", "mass_spring_damper", "two_tanks", "cruise_control", "magnetic_levitation", "cart_pendulum", "aircraft_short_period", "quarter_car", "distillation_2x2"):
    s_ = getattr(models, name)()["system"]
    lam = s_.poles()
    rows.append((name, s_.n, s_.m, s_.p, "unstable" if np.max(lam.real) > 1e-9 else "stable", f"{np.min(np.abs(lam)):.3g} .. {np.max(np.abs(lam)):.3g}"))
print(pd.DataFrame(rows, columns=["plant", "states", "inputs", "outputs", "stability", "|pole| range (rad/s)"]).to_string(index=False))"""),
    md(r"""
Nine plants with physical parameters and known properties, from a first-order cruise control to a 4-state,
2-input quarter car and an interacting 2 × 2 distillation column; two are unstable (the levitated ball and the
inverted pendulum) and need feedback before anything else. They return throughout the course.
"""),
]

NOTEBOOKS["01_time_response"] = [
    md(r"""
# 01 · Time response

The response of a linear system to an input is the convolution of the input with the impulse response, and
the state evolves by the matrix exponential: x(t) = e^{At} x₀ + ∫ e^{A(t−τ)} B u(τ) dτ. This notebook
computes responses exactly on a time grid, reads the step response's characteristics (rise, peak, overshoot,
settling), derives them for the second-order prototype, shows the dominant-pole approximation and the effect
of zeros, and connects steady-state error to system type.
"""),
    setup_cell(["from englincontrol import systems as sy, response as rs, models, linalg as la"]),
    md("## The exact response on a grid: the matrix exponential"),
    code(r"""A = np.array([[0.0, 1.0], [-4.0, -1.2]])          # the prototype wn = 2, zeta = 0.3
B = np.array([[0.0], [4.0]]); C = np.array([[1.0, 0.0]])
h = 0.01
Phi, Gamma = la.expm_with_integral(A, B, h)        # one matrix exponential of the augmented matrix gives both
print("Phi = e^{Ah} =\n", np.round(Phi, 6), "\nGamma = int_0^h e^{At} dt B =", np.round(Gamma.ravel(), 6))
x = np.zeros(2); ys = []
for k in range(1000):
    ys.append((C @ x)[0]); x = Phi @ x + Gamma.ravel() * 1.0       # unit step
t = np.arange(1000) * h
t2, y2 = rs.step(sy.ss(A, B, C, 0.0), 9.99, 1000)
print("hand-rolled recursion vs library step response:", f"{np.max(np.abs(np.array(ys) - y2)):.1e}")"""),
    md(r"""
For a piecewise-constant input the solution is *exact* at the sample instants: x_{k+1} = Φ x_k + Γ u_k with
Φ = e^{Ah} and Γ = ∫₀ʰ e^{At} dt B, both obtained from one exponential of the augmented matrix [[A, B], [0, 0]]
(Van Loan's trick). No ODE solver, no step-size error: the library's responses are computed this way, and the
same formulas are the zero-order-hold discretisation of notebook 12.

## The second-order prototype and its formulas
"""),
    code(r"""rows = []
for zeta in (0.1, 0.3, 0.5, 0.7, 1.0):
    G = sy.tf([4.0], [1.0, 4 * zeta, 4.0])
    t, y = rs.step(G, 12.0, 6001)
    info = rs.step_info(t, y, final=G.dcgain())
    th = rs.second_order(zeta, 2.0) if zeta < 1 else {"overshoot": 0.0, "peak_time": np.nan, "settling_time_2pct": np.nan}
    rows.append((zeta, info["overshoot"], th["overshoot"], info["peak_time"], th["peak_time"], info["settling_time"], th["settling_time_2pct"], info["rise_time"]))
    plt.plot(t, y, label=f"zeta = {zeta}")
plt.axhline(1, color="k", lw=0.5); plt.legend(); plt.xlabel("t"); plt.title("wn = 2: the damping ratio sets the shape"); plt.show()
print(pd.DataFrame(rows, columns=["zeta", "overshoot %", "exp(-pi z/sqrt(1-z^2)) %", "peak time", "pi/wd", "settling (2 %)", "4/(zeta wn)", "rise time"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
The prototype ω_n²/(s² + 2ζω_n s + ω_n²) is the whole vocabulary of transient response: the measured overshoot
equals exp(−πζ/√(1−ζ²)) and the peak time π/ω_d to all digits, while the "4/(ζω_n)" settling time is only an
estimate (it assumes the envelope alone matters; the actual 2 % settling time jumps between oscillation cycles,
as the ζ = 0.7 row shows). These formulas run the other way too: a specification of 10 % overshoot and 1 s
settling is a region of the s-plane (ζ ≥ 0.59, σ ≥ 4), the target of every placement design.

## Dominant poles and the effect of zeros
"""),
    code(r"""G3 = sy.zpk([], [-1 + 2j, -1 - 2j, -10.0], 50.0)              # a fast third pole
G2 = sy.zpk([], [-1 + 2j, -1 - 2j], 5.0)                        # the dominant pair alone
Gz = sy.zpk([-3.0], [-1 + 2j, -1 - 2j], 5.0 / 3)                # a zero near the pair
Gnmp = sy.zpk([3.0], [-1 + 2j, -1 - 2j], -5.0 / 3)              # a right-half-plane zero
fig, ax = plt.subplots(figsize=(8, 3.5))
for name, g in (("3rd order", G3), ("dominant pair", G2), ("with a zero at -3", Gz), ("with a zero at +3", Gnmp)):
    t, y = rs.step(g, 6.0)
    ax.plot(t, y, label=f"{name}: overshoot {rs.step_info(t, y, final=g.dcgain())['overshoot']:.0f} %")
ax.axhline(1, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8); plt.show()
print("dominant poles of G3:", np.round(rs.dominant_poles(G3), 3))"""),
    md(r"""
A pole ten times further from the imaginary axis than the dominant pair barely changes the response (its mode
decays in a tenth of the time), which is why second-order formulas serve for higher-order systems: the
*dominant* poles decide. Zeros are not so forgiving: a left-half-plane zero near the dominant poles increases
the overshoot (it adds a derivative of the response), and a right-half-plane zero produces the undershoot of
the non-minimum-phase system.

## Steady-state error and system type
"""),
    code(r"""P = sy.tf([1.0], [1.0, 3.0, 2.0])
for name, C in (("P control, K = 5 (type 0)", sy.tf([5.0], [1.0])), ("PI control (type 1)", sy.tf([5.0, 5.0], [1.0, 0.0]))):
    T = sy.feedback(sy.series(C, P))
    t, y = rs.step(T, 15.0)
    u_ramp = t
    _, y_ramp, _ = rs.lsim(T, u_ramp, t)
    print(f"{name:<28}: step error {1 - y[-1]:+.4f} (1/(1 + Kp) = {1 / (1 + 5 * P.dcgain()):.4f} for type 0), ramp error at t = 15: {u_ramp[-1] - y_ramp[-1]:.4f}")"""),
    md(r"""
With proportional control of a type-0 plant the step error is 1/(1 + K_p) - a finite offset - and a ramp runs
away. One integrator in the loop (type 1) drives the step error to zero and leaves a finite ramp error 1/K_v.
The system type counts the integrators in the open loop; the final-value theorem turns it into the steady-state
error for each test input. Integral action is the cure for offsets, at a cost in phase margin that notebook 06
measures.
"""),
]

NOTEBOOKS["02_frequency_response"] = [
    md(r"""
# 02 · Frequency response

A sinusoidal input to a stable linear system produces, after transients, a sinusoidal output of the same
frequency scaled by |G(jω)| and shifted by ∠G(jω). The function G(jω) - the frequency response - can be
measured on a real plant, plotted on Bode's logarithmic axes where products become sums, and read for stability
margins and bandwidth. This notebook establishes the sinusoidal steady state by simulation, builds Bode plots
from asymptotes, shows the Nyquist and Nichols representations, and computes the margins that the design
chapters will use.
"""),
    setup_cell(["from englincontrol import systems as sy, frequency as fr, response as rs, models"]),
    md("## The sinusoidal steady state"),
    code(r"""G = sy.tf([4.0], [1.0, 1.2, 4.0])
w = 3.0
t = np.linspace(0, 40, 16001)
_, y, _ = rs.lsim(G, np.sin(w * t), t)
tail = t > 30
M = np.column_stack([np.sin(w * t[tail]), np.cos(w * t[tail])])
a, b = np.linalg.lstsq(M, y[tail], rcond=None)[0]
print(f"simulated steady state: amplitude {np.hypot(a, b):.5f}, phase {np.degrees(np.arctan2(b, a)):.3f} deg")
Gjw = fr.freqresp(G, w)
print(f"G(3j) = {Gjw:.5f}: |G| = {abs(Gjw):.5f}, arg G = {np.degrees(np.angle(Gjw)):.3f} deg  (the simulation's phase also carries the half-sample delay of the zero-order hold, -w h/2 = {np.degrees(-w * (t[1] - t[0]) / 2):.3f} deg)")"""),
    md(r"""
The simulation agrees with G(jω) to the fifth digit in amplitude; the phase differs by exactly the half-sample
delay the zero-order hold imposes on the simulated input (notebook 12). G(jω) is the transfer function
evaluated on the imaginary axis - nothing more - and yet it contains the whole input-output behaviour.

## Bode plots from asymptotes
"""),
    code(r"""L = sy.zpk([-10.0], [0.0, -1.0, -50.0], 100.0)          # 100 (s + 10) / (s (s + 1)(s + 50))
w = np.logspace(-2, 3, 500)
w_, mag, ph = fr.bode(L, w)
asym = 20 * np.log10(100 * 10 / 50) - 20 * np.log10(w)                                  # low-frequency: integrator with gain K = 100*10/50
asym = asym - 20 * np.log10(np.maximum(w / 1.0, 1)) + 20 * np.log10(np.maximum(w / 10.0, 1)) - 20 * np.log10(np.maximum(w / 50.0, 1))
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].semilogx(w, mag, label="exact"); axes[0].semilogx(w, asym, "k--", lw=1, label="asymptotes (-20 dB/dec per pole, +20 per zero)"); axes[0].set_ylabel("|L| (dB)"); axes[0].legend(fontsize=8)
for c, lab in ((1, "pole"), (10, "zero"), (50, "pole")):
    axes[0].axvline(c, color="0.8")
axes[1].semilogx(w, ph); axes[1].set_ylabel("arg L (deg)"); axes[1].set_xlabel("w (rad/s)"); axes[1].set_yticks([-180, -135, -90, -45])
plt.show()
print(f"largest error of the straight-line asymptotes: {np.max(np.abs(mag - asym)):.2f} dB (3 dB at a real pole or zero's corner, as the theory says)")"""),
    md(r"""
On logarithmic axes each factor of the transfer function contributes additively: an integrator a −20 dB/decade
line through 0 dB at ω = K, each real pole a corner that bends the slope down by 20 dB/decade, each zero one
that bends it up, and each corner shifts the phase by 90° spread over two decades. The straight-line
asymptotes miss the exact curve by at most 3 dB at a corner (6 dB for a double pole, more for an underdamped
pair whose peak is 1/(2ζ)). Sketching them is still the quickest way to see what a controller will do.

## Nyquist and Nichols
"""),
    code(r"""w = np.logspace(-2, 2, 2000)
_, Lj = fr.nyquist(L, w)
ph_n, mag_n = fr.nichols(L, w)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(Lj.real, Lj.imag, label="w > 0"); axes[0].plot(Lj.real, -Lj.imag, "--", label="w < 0 (conjugate)"); axes[0].plot(-1, 0, "rx", ms=8)
axes[0].set_xlim(-3, 1); axes[0].set_ylim(-3, 3); axes[0].set_aspect("equal"); axes[0].legend(fontsize=8); axes[0].set_title("Nyquist plot of L(jw)")
axes[1].plot(ph_n, mag_n); axes[1].plot(-180, 0, "rx", ms=8); axes[1].set_xlabel("phase (deg)"); axes[1].set_ylabel("|L| (dB)"); axes[1].set_title("Nichols chart coordinates"); axes[1].set_xlim(-270, -60); axes[1].set_ylim(-40, 40)
plt.show()"""),
    md(r"""
The Nyquist plot draws G(jω) in the complex plane as ω runs from −∞ to ∞ (the negative half is the mirror
image); the point −1 is where the loop gain would make 1 + L = 0 and the closed loop sing. The Nichols chart
plots the same curve as magnitude in dB against phase, where the critical point is (−180°, 0 dB) and the
closed-loop magnitude can be read from overlaid contours. All three plots carry the same information; Bode is
best for design, Nyquist for stability proofs, Nichols for closed-loop peaks.

## Margins, bandwidth and resonance
"""),
    code(r"""m = fr.margins(L)
print(f"gain margin {m['gain_margin']:.3f} ({m['gain_margin_db']:.1f} dB) at the phase crossover w = {m['w_pc']:.3f} rad/s")
print(f"phase margin {m['phase_margin']:.1f} deg at the gain crossover w = {m['w_gc']:.3f} rad/s; delay margin {m['delay_margin'] * 1000:.0f} ms")
T = sy.feedback(L)
print(f"closed loop: -3 dB bandwidth {fr.bandwidth(T):.3f} rad/s (about the gain crossover), resonant peak {fr.resonant_peak(T)[0]:.3f} at {fr.resonant_peak(T)[1]:.3f} rad/s")
L2 = sy.tf([10.0], [1.0, 3.0, 2.0, 0.0])
m2 = fr.margins(L2)
print(f"\n10/(s(s+1)(s+2)): gain margin {m2['gain_margin']:.2f} < 1 and phase margin {m2['phase_margin']:.1f} deg < 0 -> the closed loop is unstable (poles {np.round(sy.feedback(L2).poles(), 3)})")"""),
    md(r"""
The gain margin is how much the loop gain can grow before the Nyquist curve passes through −1 (read at the
phase crossover, where the phase is −180°); the phase margin is how much phase lag can be added before it does
(read at the gain crossover, where |L| = 1); the delay margin converts the phase margin into a time, the delay
that would destabilise the loop. The library finds every crossing and reports the critical ones. The example
loop has an *infinite* gain margin - its phase approaches −180° from above and never crosses - yet it is not a
robust loop: a phase margin of 32° and a resonant peak of 1.9 say so, and 120 ms of delay would destabilise it.
One margin never tells the whole story. For 10/(s(s+1)(s+2)) both margins are negative and the closed loop
is indeed unstable; for a well-designed loop 45-60° and 2-3 (6-10 dB) are the usual targets. The closed-loop
bandwidth is close to the gain crossover frequency, and the resonant peak grows as the margins shrink - the
frequency-domain face of overshoot.
"""),
]

NOTEBOOKS["03_stability_tests"] = [
    md(r"""
# 03 · Stability tests

A linear system is asymptotically stable when every pole has a negative real part (every eigenvalue of A).
Before computers one had to decide that without finding the roots, and the classical tests still matter: the
Routh array and Hurwitz determinants decide stability from the polynomial's coefficients, give the number of
unstable roots, and - most usefully - give stability as a condition on a parameter such as a gain. Lyapunov's
method decides it from a quadratic form and extends to nonlinear systems. For discrete systems Jury's test does
the same inside the unit circle. This notebook runs each test and checks it against the roots.
"""),
    setup_cell(["from englincontrol import systems as sy, stability as st, polynomials as pl, linalg as la, models"]),
    md("## The Routh array"),
    code(r"""for c in ([1, 2, 3, 4], [1, 1, 2, 2, 3], [1, 2, 8, 12, 20, 16, 16]):
    r = st.routh(c)
    print(f"polynomial {c}:\n{np.round(r['table'], 4)}\n  sign changes in the first column: {r['rhp_roots']}   actual right-half-plane roots: {int(np.sum(pl.roots(c).real > 1e-9))}" + (f"   auxiliary polynomial {r['auxiliary']} (roots {np.round(pl.roots(r['auxiliary']), 3)})" if r["auxiliary"] is not None else "") + "\n")"""),
    md(r"""
The array's first column changes sign once for each root in the right half-plane. Two special cases need
care: a zero in the first column (the ε method replaces it by a small number and the signs are read in the
limit), and a whole row of zeros, which signals roots symmetric about the origin - on the imaginary axis or in
± pairs - and is handled by differentiating the auxiliary polynomial formed from the row above; its roots
(±j√2, ±2j in the last example) are the marginal ones.

## Stability as a condition on a gain
"""),
    code(r"""K = 1.0
G = sy.tf([1.0], pl.from_roots([0, -1, -2]))                       # 1 / (s (s + 1)(s + 2))
print("closed-loop polynomial s^3 + 3 s^2 + 2 s + K: Routh's condition is 3*2 > K, i.e. K < 6")
for K in (1.0, 5.9, 6.0, 6.1):
    r = st.routh(pl.add(G.den, K * G.num))
    print(f"   K = {K}: first column {np.round(r['table'][:, 0], 3)} -> {'stable' if r['stable'] else 'NOT stable'}" + (f" (auxiliary: roots at {np.round(pl.roots(r['auxiliary']), 3)})" if r["auxiliary"] is not None else ""))"""),
    md(r"""
Here the array decides more than stability: it gives the range of K for which the loop is stable (K < 6) and
the frequency at which it goes marginal (the auxiliary polynomial 3s² + 6 at K = 6 has roots ±j√2). Both
numbers reappear in the root locus (notebook 04) and in the gain margin (notebook 02): three views, one fact.

## Hurwitz, Lyapunov, Jury
"""),
    code(r"""c = [1, 2, 3, 4]
hz = st.hurwitz(c)
print("Hurwitz matrix:\n", hz["matrix"], "\nleading minors:", np.round(hz["minors"], 3), "-> stable:", hz["stable"])
A = models.mass_spring_damper()["system"].A
ly = st.lyapunov_test(A)
print("\nLyapunov: A^T P + P A = -I gives P =\n", np.round(ly["P"], 4), "\neigenvalues of P", np.round(ly["eigenvalues"], 4), "-> positive definite:", ly["stable"])
print("verification: A^T P + P A + I =", np.max(np.abs(A.T @ ly["P"] + ly["P"] @ A + np.eye(2))))
for cz in ([1, -1.5, 0.56], [1, 0.5, -0.5], [1, -0.3, 0.2, -0.1]):
    j = st.jury(cz)
    print(f"Jury for {cz}: conditions {['ok' if ok else 'FAIL' for ok in j['conditions']]} -> stable {j['stable']}; root moduli {np.round(np.abs(pl.roots(cz)), 3)}")"""),
    md(r"""
Hurwitz's criterion is the Routh array in determinant form: all leading principal minors of the Hurwitz matrix
positive. Lyapunov's is different in kind: if a positive-definite quadratic form V = xᵀPx decreases along every
trajectory (AᵀP + PA negative definite), the origin is stable - and for a linear system the equation
AᵀP + PA = −Q always has a positive-definite solution exactly when A is stable. The library solves it by
Bartels-Stewart on the real Schur form (notebook 18). Jury's test is the discrete-time counterpart: conditions
on the coefficients that place all roots inside the unit circle, which the moduli of the roots confirm.

## When the tests disagree with the roots: conditioning
"""),
    code(r"""roots_true = -np.arange(1, 13, dtype=float)                   # 12 well-separated stable roots
c = pl.from_roots(roots_true)
print(f"coefficients span {c.max():.2e} to {c.min():.2e}; Routh says {st.routh(c)['rhp_roots']} RHP roots, roots computed from the coefficients: {np.round(np.sort(pl.roots(c).real), 3)}")
c_pert = c * (1 + 1e-8 * np.array([1, -1] * 6 + [1]))         # relative perturbation of 1e-8 in the coefficients
print(f"after perturbing the coefficients by 1e-8: roots {np.round(np.sort(pl.roots(c_pert).real), 3)}")"""),
    md(r"""
Twelve roots at −1 … −12 give coefficients from 1 to 4.8×10⁸, and a relative perturbation of 10⁻⁸ in them -
the size of a measurement's rounding - moves the roots by tenths. Wilkinson's famous example in miniature:
polynomial coefficients are a poorly conditioned way to carry a high-order system, which is why the
state-space representation and eigenvalue computations on A are preferred numerically (notebook 18), and why
the classical tests are at their best for low-order systems and symbolic conditions.
"""),
]

NOTEBOOKS["04_the_root_locus"] = [
    md(r"""
# 04 · The root locus

Evans' root locus draws the closed-loop poles of 1 + K G(s) = 0 as the gain K runs from 0 to ∞: a picture of
everything a proportional gain can do. The poles start at the open-loop poles, end at the open-loop zeros (or
at infinity along asymptotes), stay on the real axis to the left of an odd number of real poles and zeros, and
leave it at breakaway points where dK/ds = 0. The rules let one sketch the locus by hand; the computation
confirms the sketch; and reading the locus against the damping-ratio lines of notebook 01 is the design method.
"""),
    setup_cell(["from englincontrol import systems as sy, stability as st, polynomials as pl, response as rs, frequency as fr"]),
    md("## The locus and its rules"),
    code(r"""G = sy.tf([1.0], pl.from_roots([0, -1, -2]))
rl = st.root_locus(G, np.logspace(-2, 2, 800))
fig, ax = plt.subplots(figsize=(6, 6))
for k in range(rl["branches"].shape[1]):
    ax.plot(rl["branches"][:, k].real, rl["branches"][:, k].imag, "b-", lw=1.2)
ax.plot(rl["poles"].real, rl["poles"].imag, "kx", ms=9, label="open-loop poles")
for ang in rl["asymptote_angles"]:
    ax.plot([rl["centroid"], rl["centroid"] + 6 * np.cos(np.radians(ang))], [0, 6 * np.sin(np.radians(ang))], "k:", lw=0.8)
ax.plot(rl["breakaway"], 0 * rl["breakaway"], "ro", label=f"breakaway at {rl['breakaway'][0]:.3f}")
ax.axhline(0, color="0.7"); ax.axvline(0, color="0.7"); ax.set_xlim(-4, 2); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_title("root locus of 1 / (s (s+1)(s+2))"); plt.show()
print(f"asymptotes from the centroid {rl['centroid']:.1f} at {rl['asymptote_angles']} degrees; real-axis segments {[(float(round(a, 2)), float(round(b, 2))) for a, b in rl['real_axis_segments']]}")
print(f"breakaway where dK/ds = 0: s = {rl['breakaway'][0]:.4f} = -1 + 1/sqrt(3)")"""),
    md(r"""
Three poles and no zeros: three branches, all going to infinity along asymptotes at 60°, 180° and 300° from
the centroid (0 − 1 − 2)/3 = −1. The real axis between 0 and −1 and to the left of −2 belongs to the locus
(an odd number of real poles to the right); the two branches starting at 0 and −1 meet at the breakaway point
−1 + 1/√3 = −0.423 (where N′D − ND′ = 0) and leave the axis as a complex pair that crosses into the right
half-plane. Every feature is computable from the rules before plotting a point.

## Reading the locus: gain for a damping ratio, and the stability limit
"""),
    code(r"""K_half, p_half = st.gain_for_damping(G, 0.5, np.logspace(-2, 1, 3000))
print(f"damping ratio 0.5 is reached at K = {K_half:.3f}, poles {np.round(p_half, 3)} (and its conjugate); the third pole is at {np.round(pl.roots(pl.add(G.den, K_half * G.num)).real.min(), 3)}")
K_marg = 6.0
print(f"the locus crosses the imaginary axis at K = {K_marg} (Routh's limit), poles {np.round(pl.roots(pl.add(G.den, K_marg * G.num)), 3)}")
fig, ax = plt.subplots(figsize=(8, 3.5))
for K in (K_half, 3.0, 5.9):
    T = sy.feedback(sy.tf(K * G.num, G.den))
    t, y = rs.step(T, 15.0)
    ax.plot(t, y, label=f"K = {K:.2f}: overshoot {rs.step_info(t, y, final=T.dcgain())['overshoot']:.0f} %")
ax.legend(); ax.set_xlabel("t"); ax.set_title("closed-loop step responses along the locus"); plt.show()"""),
    md(r"""
Design on the locus is reading it against specifications: the gain at which the complex branch crosses the
ζ = 0.5 line (K ≈ 1.04) gives about 16 % overshoot, but the third pole at −2.3 is not far enough to be ignored
(the actual overshoot is higher than the second-order formula's); at K = 6 the pair sits on the imaginary axis
and the loop oscillates. The locus also shows the structural fact that no gain can fix: a plant with three
poles and no zeros always goes unstable at high gain.

## Compensators change the locus
"""),
    code(r"""Glead = sy.series(sy.tf([1.0, 1.0], [1.0, 10.0]), G)              # lead: zero at -1 cancels... no: adds zero at -1, pole at -10
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, g) in zip(axes, (("plant alone", G), ("with lead (s + 1)/(s + 10)", Glead))):
    r = st.root_locus(g, np.logspace(-2, 3, 800))
    for k in range(r["branches"].shape[1]):
        ax.plot(r["branches"][:, k].real, r["branches"][:, k].imag, "b-", lw=1.2)
    ax.plot(r["poles"].real, r["poles"].imag, "kx", ms=9); ax.plot(r["zeros"].real, r["zeros"].imag, "ko", ms=8, mfc="none")
    ax.axhline(0, color="0.7"); ax.axvline(0, color="0.7"); ax.set_xlim(-11, 2); ax.set_ylim(-6, 6); ax.set_aspect("equal"); ax.set_title(name)
    print(f"{name:<28}: centroid {r['centroid']:.2f}, asymptotes {np.round(r['asymptote_angles'], 0)}, gain for zeta = 0.5: {st.gain_for_damping(g, 0.5, np.logspace(-2, 2, 3000))[0]:.2f}")
plt.show()"""),
    md(r"""
Adding a lead compensator's zero and pole moves the centroid left (to −4) and bends the locus away from
the right half-plane: a higher gain now reaches the same damping ratio, so the closed loop is faster with the
same overshoot. (The zero at −1 cancels the plant pole at −1 here - allowed for a stable pole, though the
cancelled mode remains in the disturbance response; cancelling an *unstable* pole is never allowed, as notebook
08 shows.) This is the root-locus view of the frequency-domain lead design of notebook 07.

## Departure angles and complex poles
"""),
    code(r"""Gc = sy.tf([1.0], pl.from_roots([-1 + 1j, -1 - 1j, -3.0]))
r = st.root_locus(Gc, np.logspace(-2, 2, 600))
print("departure angles from the complex poles:", {k: round(float(v), 2) for k, v in r["departure_angles"].items()})
print("the angle condition: 180 - (sum of angles from the other poles) = 180 - 90 - atan(1/2) =", round(180 - 90 - np.degrees(np.arctan(0.5)), 2), "degrees")
fig, ax = plt.subplots(figsize=(5, 5))
for k in range(3):
    ax.plot(r["branches"][:, k].real, r["branches"][:, k].imag, "b-")
ax.plot(r["poles"].real, r["poles"].imag, "kx", ms=9); ax.axhline(0, color="0.7"); ax.axvline(0, color="0.7"); ax.set_xlim(-4, 1); ax.set_ylim(-3, 3); ax.set_aspect("equal"); plt.show()"""),
    md(r"""
The angle condition, ∠G(s) = 180° on the locus, fixes the direction in which a branch leaves a complex pole;
here 63.4°, pointing up and to the right - towards instability. The rules are a complete algorithm: a reader
who can compute angles can draw the locus of any transfer function, which is the real reason the method
survives in an age when the roots can be computed in a millisecond.
"""),
]

NOTEBOOKS["05_the_nyquist_criterion"] = [
    md(r"""
# 05 · The Nyquist criterion

The Nyquist criterion decides closed-loop stability from the open-loop frequency response - a quantity one can
measure - and it works for unstable plants, for delays, and for anything without a rational transfer function.
It rests on Cauchy's argument principle: as s travels once around a contour enclosing the right half-plane,
the number of times L(s) encircles −1 equals the number of closed-loop poles minus the number of open-loop
poles inside. This notebook builds the contour, counts the encirclements, treats poles on the imaginary axis,
and shows the criterion at work on a plant that is only stable with feedback.
"""),
    setup_cell(["from englincontrol import systems as sy, frequency as fr, stability as st, polynomials as pl, response as rs"]),
    md("## Encirclements of −1: Z = N + P"),
    code(r"""def nyquist_figure(L, ax, lim=3.0, title=""):
    r = st.nyquist_criterion(L)
    Ls = r["L"]
    ax.plot(Ls.real, Ls.imag, lw=1.2); ax.plot(-1, 0, "rx", ms=9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axhline(0, color="0.7"); ax.axvline(0, color="0.7")
    ax.set_title(f"{title}\nN = {r['N_clockwise']} clockwise, P = {r['P']}, Z = N + P = {r['Z']}", fontsize=9)
    return r
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (L, lim, name) in zip(axes, ((sy.tf([1.0], [1.0, 3.0, 2.0, 0.0]), 1.5, "1/(s(s+1)(s+2)): stable loop"), (sy.tf([10.0], [1.0, 3.0, 2.0, 0.0]), 4.0, "10/(s(s+1)(s+2)): too much gain"), (sy.tf([2.0], [1.0, -1.0]), 3.0, "2/(s - 1): unstable plant, stabilised"))):
    r = nyquist_figure(L, ax, lim, name)
    print(f"{name:<40}: Z = {r['Z']} closed-loop RHP poles; from the characteristic polynomial: {r['closed_loop_rhp_poles']}")
plt.show()"""),
    md(r"""
Three loops, three verdicts. The stable loop with K = 1 leaves −1 outside the curve (N = 0, P = 0, Z = 0); with
K = 10 the curve swings round −1 twice clockwise (Z = 2: two unstable closed-loop poles, exactly those the
characteristic polynomial has); and the unstable plant 2/(s − 1) is stabilised by the loop: its curve encircles
−1 once *counterclockwise* (N = −1) which cancels the open-loop unstable pole (P = 1) to give Z = 0. The last
case is the one no Bode-plot rule of thumb handles, and the reason the criterion is worth its contour.

## The contour and the indentation around poles on the axis
"""),
    code(r"""L = sy.tf([1.0], [1.0, 3.0, 2.0, 0.0])
r = st.nyquist_criterion(L, eps=0.05, n=4000)
s = r["contour"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].plot(s.real, s.imag, ".", ms=2); axes[0].plot(0, 0, "kx", ms=9); axes[0].set_xlim(-0.1, 0.2); axes[0].set_ylim(-0.3, 0.3); axes[0].set_aspect("equal"); axes[0].set_title("the contour indented around the pole at the origin")
Ls = r["L"]
axes[1].plot(Ls.real, Ls.imag, lw=1); axes[1].plot(-1, 0, "rx", ms=9); axes[1].set_xlim(-2, 2); axes[1].set_ylim(-20, 20); axes[1].set_title("its image: the small semicircle maps to a large one (clockwise through infinity)")
plt.show()
print(f"with eps = 0.05 the image of the indentation reaches |L| = {np.max(np.abs(Ls)):.0f}; the count is unaffected: N = {r['N_clockwise']}, Z = {r['Z']}")"""),
    md(r"""
A pole on the imaginary axis - the integrator here - lies on the contour, where L is infinite; the contour
detours around it by a small semicircle into the right half-plane (so the pole is excluded from P), and L maps
that semicircle to a large arc at infinity that closes the curve. The library builds this contour explicitly,
logarithmically dense near the origin so that nothing is missed, and counts the phase change of 1 + L along
it. Poles at other points of the axis get the same treatment; closed-loop poles *on* the axis make the
criterion meaningless (1 + L vanishes on the contour) and are flagged as marginal.

## From the criterion to the margins
"""),
    code(r"""L = sy.zpk([-10.0], [0.0, -1.0, -50.0], 100.0)
w = np.logspace(-2, 3, 3000)
_, Lj = fr.nyquist(L, w)
m = fr.margins(L)
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(Lj.real, Lj.imag); ax.plot(Lj.real, -Lj.imag, "--", color="C0", lw=0.7)
th = np.linspace(0, 2 * np.pi, 200); ax.plot(np.cos(th), np.sin(th), "0.6", lw=0.7)
Lg = fr.freqresp(L, m["w_gc"]); ax.plot([0, Lg.real], [0, Lg.imag], "r-", lw=1); ax.plot(-1, 0, "rx", ms=9)
ax.set_xlim(-2, 1); ax.set_ylim(-2, 1.5); ax.set_aspect("equal"); ax.set_title(f"phase margin {m['phase_margin']:.1f} deg: the angle between the unit circle crossing and -1"); plt.show()
dist = np.min(np.abs(1 + Lj))
print(f"closest approach to -1: {dist:.3f} (= 1/Ms, the sensitivity peak {1 / dist:.2f}); gain margin {m['gain_margin']}, phase margin {m['phase_margin']:.1f} deg")"""),
    md(r"""
On the Nyquist plot the margins are distances from −1: the gain margin along the negative real axis, the phase
margin along the unit circle. The single number that captures both is the closest approach of the curve to
−1, which is 1/M_s with M_s the peak of the sensitivity function (notebook 14): this loop never crosses the
real axis left of the origin (infinite gain margin) but comes within 0.5 of −1, which is why its step response
rings. Margins are shorthand for that distance, and can mislead when the curve approaches −1 obliquely.

## A delay: where the criterion earns its keep
"""),
    code(r"""P = sy.tf([1.0], [1.0, 1.0])
w = np.logspace(-2, 2, 4000)
for Td in (0.0, 0.5, 1.0, 2.0):
    K = 2.0
    Lj = K * fr.freqresp(P, w, delay=Td)
    phase = np.degrees(np.unwrap(np.angle(Lj)))
    m = fr.margins(sy.tf([K], [1.0, 1.0]), delay=Td)
    print(f"delay {Td:.1f} s: gain margin {m['gain_margin']:.3f}, phase margin {m['phase_margin']:.1f} deg, closest approach to -1: {np.min(np.abs(1 + Lj)):.3f}")
Td = 2.0
print(f"\nwith a 2 s delay the loop K/(s+1) e^(-2s) is unstable at K = 2: the curve spirals around -1; the maximum stable gain is {fr.margins(sy.tf([1.0], [1.0, 1.0]), delay=2.0)['gain_margin']:.3f}")
t_sim = np.linspace(0, 30, 3001)"""),
    md(r"""
A time delay e^{−sT} has no poles or zeros and no rational transfer function, yet its frequency response - unit
magnitude, phase −ωT - is exact, and the Nyquist curve of K e^{−sT}/(s+1) spirals inward through every phase.
The criterion handles it directly: the gain margin shrinks as the delay grows (to 1.52 at T = 2 s) until K = 2
no longer stabilises the loop. Delays are the commonest reason control loops are slower than their plants
seem to allow, and notebook 14 quantifies the limit; the Padé approximation (`frequency.pade`) brings them
into the rational world when a state-space model is needed.
"""),
]

NOTEBOOKS["06_pid_control_and_tuning"] = [
    md(r"""
# 06 · PID control and tuning

Most industrial loops are closed with a PID controller: a proportional term for speed, an integral term for
zero offset, a derivative term for damping. This notebook takes the three terms apart on a third-order plant,
fits the first-order-plus-dead-time model that tuning rules want, applies Ziegler-Nichols, Cohen-Coon and
SIMC rules and compares them, measures the ultimate gain with a relay experiment instead of a model, and then
confronts the controller with reality: actuator saturation, integrator windup and the derivative kick.
"""),
    setup_cell(["from englincontrol import systems as sy, design as dg, response as rs, frequency as fr, models"]),
    md("## The three terms"),
    code(r"""P = sy.tf([1.0], [1.0, 3.0, 3.0, 1.0])                 # 1/(s + 1)^3
fig, ax = plt.subplots(figsize=(8, 3.8))
for name, (Kp, Ki, Kd) in (("P: Kp = 4", (4.0, 0.0, 0.0)), ("PI: Kp = 4, Ki = 2", (4.0, 2.0, 0.0)), ("PID: Kp = 4, Ki = 2, Kd = 3", (4.0, 2.0, 3.0))):
    C = dg.pid(Kp, Ki, Kd)
    cl = dg.closed_loop(P, C)
    t, y = rs.step(cl["T"], 20.0)
    m = fr.margins(cl["L"])
    ax.plot(t, y, label=f"{name}: final {y[-1]:.3f}, overshoot {rs.step_info(t, y, final=cl['T'].dcgain())['overshoot']:.0f} %, PM {m['phase_margin']:.0f} deg")
ax.axhline(1, color="k", lw=0.5); ax.legend(fontsize=8); ax.set_xlabel("t"); plt.show()"""),
    md(r"""
Proportional control alone leaves an offset (the final value 0.8 = K/(1 + K) for unit plant gain); the integral
term removes it but lowers the phase margin and raises the overshoot; the derivative term adds phase lead
near the crossover and recovers damping. The library's PID carries a filter on the derivative (time constant
K_d/(K_p N), N = 10) because an unfiltered derivative is improper and amplifies measurement noise without bound.

## A model for the tuning rules: FOPDT
"""),
    code(r"""t, y = rs.step(P, 15.0, 3001)
fo = dg.fopdt_from_step(t, y)
print(f"first-order-plus-dead-time fit of 1/(s+1)^3: K = {fo['K']:.3f}, tau = {fo['tau']:.3f}, L = {fo['L']:.3f}")
Pf = sy.tf([fo["K"]], [fo["tau"], 1.0])
t_after = t[t >= fo["L"]]
_, yf = rs.step(Pf, t_after[-1] - t_after[0], len(t_after))
plt.plot(t, y, label="plant (s+1)^-3"); plt.plot(t_after, yf, "--", label="FOPDT model (delayed)"); plt.legend(); plt.xlabel("t"); plt.show()
ug = dg.ultimate_gain(P)
print(f"ultimate gain of the plant: Ku = {ug['Ku']:.3f} at w = {ug['w_u']:.4f} (Pu = {ug['Pu']:.3f} s): the classical closed-loop ZN quantities")"""),
    md(r"""
The two-point method reads the times to 28 % and 63 % of the final value and returns K, τ and L; the
third-order plant's S-shaped response is approximated by a first-order lag of 1.8 s delayed by 1.3 s. The
ultimate gain and period are the other classical inputs, exact here from the phase crossover (K_u = 8 at
ω = √3 for (s+1)⁻³).

## Four tuning rules, one plant
"""),
    code(r"""rules = {"ZN step": dg.ziegler_nichols_step(fo["K"], fo["tau"], fo["L"])["PID"], "ZN ultimate": dg.ziegler_nichols_ultimate(ug["Ku"], ug["Pu"])["PID"],
         "Cohen-Coon": dg.cohen_coon(fo["K"], fo["tau"], fo["L"])["PID"], "SIMC (PI)": dg.imc_pid(fo["K"], fo["tau"], fo["L"])}
rows = []
fig, ax = plt.subplots(figsize=(8, 3.8))
for name, p_ in rules.items():
    C = dg.pid_standard(p_["Kp"], p_["Ti"], p_["Td"])
    cl = dg.closed_loop(P, C)
    t, y = rs.step(cl["T"], 25.0)
    info = rs.step_info(t, y, final=1.0)
    m = fr.margins(cl["L"])
    rows.append((name, p_["Kp"], p_["Ti"], p_["Td"], info["overshoot"], info["settling_time"], m["gain_margin"], m["phase_margin"]))
    ax.plot(t, y, label=name)
ax.axhline(1, color="k", lw=0.5); ax.legend(); ax.set_xlabel("t"); plt.show()
print(pd.DataFrame(rows, columns=["rule", "Kp", "Ti", "Td", "overshoot %", "settling (s)", "GM", "PM (deg)"]).to_string(index=False, float_format="%.2f"))"""),
    md(r"""
The Ziegler-Nichols ultimate-cycle rule was designed for quarter-amplitude decay - aggressive, with 40-60 %
overshoot and a phase margin around 30° - and Cohen-Coon is a milder relative; the ZN step rule happens to
come out gentle here because the FOPDT fit's long apparent delay makes it conservative (different rules read
the same plant differently). SIMC (Skogestad's simple IMC rule, λ = L) gives the calmest loop, with a 60°
margin at the cost of speed. There is no best rule, only a best trade-off for the
purpose: a tight temperature loop and a surge-prone level loop want different tunings, and the margins in the
table are the honest way to compare them.

## Tuning without a model: the relay experiment
"""),
    code(r"""rl = dg.relay_experiment(P, amplitude=1.0, dt=0.005, T=80.0)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(rl["t"], rl["y"], label="output"); ax.plot(rl["t"], 0.3 * rl["u"], label="relay (scaled)"); ax.set_xlim(40, 60); ax.legend(fontsize=8); ax.set_xlabel("t"); plt.show()
print(f"relay feedback: oscillation amplitude {rl['amplitude']:.4f}, period {rl['Pu']:.3f} s -> Ku = 4d/(pi a) = {rl['Ku']:.3f} (exact {ug['Ku']:.3f}), Pu (exact {ug['Pu']:.3f})")"""),
    md(r"""
Replacing the controller by a relay makes the loop oscillate at the ultimate frequency with an amplitude from
which the ultimate gain follows by the describing-function approximation K_u ≈ 4d/(πa). The estimates land
within a few percent of the exact values, and this is how autotuning buttons on industrial controllers work:
a few oscillation cycles, then a ZN-type rule. No model, no step test that disturbs the process.

## Reality: saturation, windup and the derivative kick
"""),
    code(r"""zn = rules["ZN ultimate"]
Kp, Ki, Kd = zn["Kp"], zn["Kp"] / zn["Ti"], zn["Kp"] * zn["Td"]
runs = {"no limits": dg.simulate_pid(P, Kp, Ki, Kd, 1.0, T=30.0), "limits +-1.5, no anti-windup": dg.simulate_pid(P, Kp, Ki, Kd, 1.0, T=30.0, u_limits=(-1.5, 1.5), antiwindup=None),
        "limits, clamping": dg.simulate_pid(P, Kp, Ki, Kd, 1.0, T=30.0, u_limits=(-1.5, 1.5), antiwindup="clamping"), "limits, back-calculation": dg.simulate_pid(P, Kp, Ki, Kd, 1.0, T=30.0, u_limits=(-1.5, 1.5), antiwindup="back-calculation")}
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for name, r in runs.items():
    axes[0].plot(r["t"], r["y"], label=f"{name}: overshoot {100 * (r['y'].max() - 1):.0f} %"); axes[1].plot(r["t"], r["u"])
axes[0].axhline(1, color="k", lw=0.5); axes[0].legend(fontsize=8); axes[0].set_ylabel("y"); axes[1].set_ylabel("u"); axes[1].set_xlabel("t"); plt.show()
kick = dg.simulate_pid(P, Kp, Ki, Kd, 1.0, T=2.0, derivative_on="error")
nokick = dg.simulate_pid(P, Kp, Ki, Kd, 1.0, T=2.0)
print(f"initial control move with the derivative on the error: {np.abs(kick['u'][:3]).max():.1f}; on the measurement: {np.abs(nokick['u'][:3]).max():.1f}")"""),
    md(r"""
With the actuator limited to ±1.5 the integrator keeps accumulating error while the output cannot follow -
*windup* - and the overshoot grows from 50 % to over 70 % with a long recovery. Clamping the integrator while
saturated helps; back-calculation (feeding the saturation excess back into the integrator) nearly restores the
unsaturated behaviour. The derivative kick is the other practical issue: differentiating the error turns a
setpoint step into an impulse in the control signal (an initial move of 100 against 5), which is why the
derivative is applied to the measurement in practice. These are not refinements; a PID without them does not
survive its first plant.
"""),
]

NOTEBOOKS["07_lead_lag_and_loop_shaping"] = [
    md(r"""
# 07 · Lead, lag and loop shaping

Classical design shapes the open-loop frequency response: high gain at low frequency for small errors, a
crossover at the frequency the specifications demand with enough phase margin for damping, and low gain at
high frequency where the model is wrong and the noise lives. A lead compensator adds phase near the crossover,
a lag adds low-frequency gain without disturbing the crossover, and the pair does both. This notebook
designs them on the Bode plot for the DC motor's position loop and a double-integrator-like plant, checks the
results in the time domain, and adds feedforward where feedback alone is not enough.
"""),
    setup_cell(["from englincontrol import systems as sy, design as dg, frequency as fr, response as rs, models, polynomials as pl"]),
    md("## The problem: a plant that needs phase"),
    code(r"""P = sy.tf([1.0], [1.0, 1.0, 0.0])                            # 1/(s (s + 1)): type 1, a servo-like plant
for K in (1.0, 10.0):
    m = fr.margins(sy.tf([K], P.den))
    t, y = rs.step(sy.feedback(sy.tf([K], P.den)), 15.0)
    print(f"proportional gain {K:4.0f}: crossover {m['w_gc']:.2f} rad/s, phase margin {m['phase_margin']:.1f} deg, overshoot {rs.step_info(t, y, final=1.0)['overshoot']:.0f} %")"""),
    md(r"""
Gain alone trades speed for damping: K = 10 moves the crossover to 3 rad/s but leaves 18° of phase margin and
60 % overshoot. The specification - crossover 3 rad/s *and* 50° margin - needs 32° of extra phase at 3 rad/s,
which is exactly what a lead compensator supplies.

## Lead design on the Bode plot
"""),
    code(r"""design = dg.design_lead(P, w_c=3.0, pm_target_deg=50.0)
C = design["controller"]
print(f"lead angle needed {design['lead_angle']:.1f} deg (50 - (180 + arg P(3j)) + 5 extra); compensator {C} with gain K = {design['K']:.2f}")
m = design["margins"]
print(f"result: crossover {m['w_gc']:.3f} rad/s, phase margin {m['phase_margin']:.1f} deg, gain margin inf (phase never reaches -180)")
w = np.logspace(-1, 2, 400)
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for name, sys_ in (("K P (K = 10)", sy.tf([10.0], P.den)), ("lead-compensated C P", sy.series(C, P))):
    w_, mag, ph = fr.bode(sys_, w)
    axes[0].semilogx(w, mag, label=name); axes[1].semilogx(w, ph)
axes[0].axhline(0, color="k", lw=0.5); axes[0].axvline(3, color="0.7"); axes[1].axhline(-130, color="0.7", ls="--"); axes[1].axvline(3, color="0.7"); axes[0].legend(); axes[0].set_ylabel("|L| dB"); axes[1].set_ylabel("deg"); axes[1].set_xlabel("w"); plt.show()
t, y = rs.step(sy.feedback(sy.series(C, P)), 6.0)
print(f"step response: overshoot {rs.step_info(t, y, final=1.0)['overshoot']:.1f} %, settling {rs.step_info(t, y, final=1.0)['settling_time']:.2f} s")"""),
    md(r"""
The procedure is the classical one: read the plant's phase at the desired crossover, compute the phase lead
still missing (plus a few degrees to absorb the lead's own gain shift), build the lead (1 + Ts)/(1 + αTs) with
its maximum phase at the crossover, then set the gain so that |C P| = 1 there. The crossover lands at 3.000
rad/s and the margin at 55°; the overshoot drops to about 17 %. In root-locus terms the lead's zero pulled the
locus left (notebook 04); on the Bode plot it lifted the phase where it mattered.

## Lag compensation for steady-state accuracy
"""),
    code(r"""P2 = sy.tf([1.0], [1.0, 3.0, 2.0])                          # type 0: a finite steady-state error under P control
K = 5.0
L0 = sy.tf([K], P2.den)
Clag = dg.design_lag(P2, 10.0, fr.margins(L0)["w_gc"])
L1 = sy.series(sy.tf([K], [1.0]), sy.series(Clag, P2))
for name, L_ in (("K P", L0), ("K lag P", L1)):
    m = fr.margins(L_)
    T_ = sy.feedback(L_)
    t, y = rs.step(T_, 40.0)
    print(f"{name:<8}: dc loop gain {L_.dcgain():.1f}, step error {1 - T_.dcgain():.4f}, crossover {m['w_gc']:.3f} rad/s, phase margin {m['phase_margin']:.1f} deg")"""),
    md(r"""
A lag compensator β(1 + Ts)/(1 + βTs) with β = 10 raises the loop gain tenfold at low frequency - the
steady-state error falls from 0.29 to 0.04 - while leaving the crossover and the phase margin almost untouched,
because its zero sits a decade below the crossover where its phase lag has faded to a few degrees. It is a slow integrator
that stops short of a pole at the origin; a true PI would drive the error to zero, at a larger cost in phase.

## Loop shaping as a whole, and the price of bandwidth
"""),
    code(r"""mot = models.dc_motor(position=True)
Pm = mot["system"].to_tf()
print("DC motor position:", Pm)
rows = []
for wc in (2.0, 5.0, 10.0, 20.0):
    d = dg.design_lead(Pm, w_c=wc, pm_target_deg=50.0)
    cl = dg.closed_loop(Pm, d["controller"])
    t, y = rs.step(cl["T"], 8.0 / wc * 3)
    tu, u = rs.step(cl["CS"], 8.0 / wc * 3)
    rows.append((wc, d["K"], d["lead_angle"], d["n_leads"], d["margins"]["phase_margin"], rs.step_info(t, y, final=1.0)["settling_time"], np.max(np.abs(u)), fr.bandwidth(cl["T"])))
print(pd.DataFrame(rows, columns=["target w_c", "gain K", "lead (deg)", "leads", "PM (deg)", "settling (s)", "peak |u| (V)", "bandwidth"]).to_string(index=False, float_format="%.2f"))"""),
    md(r"""
Pushing the crossover from 2 to 20 rad/s on the motor's position loop makes the response ten times faster
and asks for a gain twenty times larger and a peak control voltage a thousand times larger - and beyond
5 rad/s, where the plant's phase has passed −180°, one lead can no longer supply the phase and the design
cascades two. Loop shaping is this negotiation: bandwidth costs actuator effort, noise amplification (each
lead's high-frequency gain 1/α) and sensitivity to unmodelled dynamics such as the resonance the project's
servo turns out to have.

## Feedforward
"""),
    code(r"""P = sy.tf([2.0], [1.0, 3.0, 2.0])
C = dg.pid(3.0, 2.0, 0.5)
F = dg.feedforward_inverse(P, bandwidth_filter=20.0)
print("feedforward (approximate plant inverse):", F)
t = np.linspace(0, 8, 1601)
r_t = np.clip(t, 0, 1.0)                                                   # a ramp to 1 in 1 s, then hold
cl = dg.closed_loop(P, C)
_, y_fb, _ = rs.lsim(cl["T"], r_t, t)
# with feedforward u = C e + F r the reference-to-output map is P (C + F) / (1 + P C)
CF_P = sy.series(sy.parallel(C, F), P)
L_ = sy.series(C, P)
T_ff = sy.tf(pl.mul(CF_P.num, L_.den), pl.mul(CF_P.den, pl.add(L_.den, L_.num))).cancel()
_, y_ff, _ = rs.lsim(T_ff, r_t, t)
plt.plot(t, r_t, "k--", lw=1, label="reference"); plt.plot(t, y_fb, label=f"feedback only: tracking error rms {np.sqrt(np.mean((y_fb - r_t) ** 2)):.4f}"); plt.plot(t, y_ff, label=f"feedback + feedforward: {np.sqrt(np.mean((y_ff - r_t) ** 2)):.4f}"); plt.legend(fontsize=8); plt.xlabel("t"); plt.show()"""),
    md(r"""
Feedback reacts to errors after they appear; feedforward anticipates. Adding u = F r with F an approximate
inverse of the plant (made proper by a filter) lets the output follow a ramp almost without error, while the
feedback loop remains in charge of disturbances and model mismatch. For the tracking of known references -
machine tools, robots, batch recipes - feedforward does most of the work and feedback cleans up.
"""),
]

NOTEBOOKS["08_controllability_observability_and_pole_placement"] = [
    md(r"""
# 08 · Controllability, observability and pole placement

State feedback u = −Kx moves every closed-loop pole wherever one wants - if the system is controllable. The
dual property, observability, decides whether the state can be reconstructed from the output. This notebook
tests both (rank, singular values, the PBH test), shows what an uncontrollable or unobservable mode looks like
and why a cancelled unstable mode is a catastrophe, places poles by Ackermann's formula and by the Sylvester
method, and ends with the numerical warning every placement routine should carry.
"""),
    setup_cell(["from englincontrol import systems as sy, statespace as sp, response as rs, models, polynomials as pl"]),
    md("## The tests"),
    code(r"""cp = models.cart_pendulum()["system"]
c = sp.controllability(cp.A, cp.B); o = sp.observability(cp.A, cp.C); o_pos = sp.observability(cp.A, cp.C[[0]]); o_ang = sp.observability(cp.A, cp.C[[1]])
print(f"cart-pendulum: controllable from the force ({c['rank']} of 4; singular values {np.round(c['singular_values'], 2)})")
print(f"  observable from position and angle: {o['observable']}; from position alone: {o_pos['observable']}; from the angle alone: {o_ang['observable']} (rank {o_ang['rank']})")
A_un = np.array([[-1.0, 0.0], [0.0, -2.0]]); B_un = np.array([[1.0], [0.0]])
print(f"\ndiag(-1, -2) with B = [1, 0]: rank {sp.controllability(A_un, B_un)['rank']}; PBH test names the uncontrollable mode: {sp.pbh_test(A_un, B_un)['uncontrollable_modes']}")"""),
    md(r"""
The cart-pendulum is controllable from the single force and observable from both outputs - but not from the
angle alone: the cart's position never shows in the angle, so it cannot be estimated from it (and a controller
that only measures the angle cannot keep the cart in place). The PBH test gives more than a rank: it names the
mode that is out of reach, here the eigenvalue −2 whose eigenvector is orthogonal to B.

## What a hidden mode means: the cancelled unstable pole
"""),
    code(r"""P = sy.tf([1.0, -1.0], [1.0, 1.0, -2.0])                    # (s - 1) / ((s - 1)(s + 2)): the unstable pole cancels on paper
print("transfer function after cancellation:", P, "-> looks like a harmless 1/(s + 2)")
Pss = sy.ss(np.array([[1.0, 0.0], [0.0, -2.0]]), [[1.0], [1.0]], [[0.0, 3.0]], 0.0)      # a realisation that keeps both modes
print("its full realisation has poles", np.round(Pss.poles(), 3), "and an unobservable mode:", sp.observability(Pss.A, Pss.C)["rank"], "observable of 2")
t, y, x = rs.initial(Pss, [1e-6, 0.0], 15.0)
print(f"a 1e-6 perturbation of the hidden state grows to {x[-1, 0]:.0f} by t = 15 while the output stays {np.max(np.abs(y)):.1e}: the output sees nothing of the explosion")"""),
    md(r"""
On paper (s − 1)/((s − 1)(s + 2)) cancels to 1/(s + 2); in the physical system the unstable mode is still there,
unobservable from the output, growing from any perturbation while the output reports all is well. Cancelling an
unstable pole with a controller zero produces exactly this: internal instability behind a healthy-looking
transfer function. The rule is absolute: never cancel right-half-plane poles or zeros, and check minimality
of what remains.

## Pole placement: Ackermann and beyond
"""),
    code(r"""A, B, C = cp.A, cp.B, cp.C
desired = np.array([-2 + 2j, -2 - 2j, -5.0, -6.0])
K = sp.place(A, B, desired)
print("K =", np.round(K, 3), "\nclosed-loop poles:", np.round(np.sort_complex(np.linalg.eigvals(A - B @ K)), 6))
Acl = sy.ss(A - B @ K, B, C, np.zeros((2, 1)))
t, y, x = rs.initial(Acl, [0.0, 0.0, 0.2, 0.0], 5.0)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(t, np.degrees(x[:, 2]), label="pendulum angle (deg)"); axes[0].plot(t, 100 * x[:, 0], label="cart position (cm)"); axes[0].legend(); axes[0].set_xlabel("t")
axes[1].plot(t, (-x @ K.T)[:, 0]); axes[1].set_ylabel("force (N)"); axes[1].set_xlabel("t"); axes[1].set_title("control effort")
plt.show()
rng = np.random.default_rng(2)
A5, B5 = rng.normal(size=(5, 5)), rng.normal(size=(5, 2))
des5 = np.array([-1, -2, -3 + 2j, -3 - 2j, -5])
K5 = sp.place(A5, B5, des5)
print("\ntwo inputs, five states (Sylvester method): poles", np.round(np.sort_complex(np.linalg.eigvals(A5 - B5 @ K5)), 6), "- the gain is not unique; the method picks a small one")"""),
    md(r"""
With the four poles chosen, Ackermann's formula gives the single gain vector that produces them; the pendulum
released at 11° is brought upright in two seconds with a cart excursion of a few centimetres and a peak force
of a few newtons. With two inputs the gain matrix is no longer unique: the Sylvester-equation method chooses a
parameter matrix at random and solves for K, and the freedom can be spent on small gains or robust eigenvectors.

## The numerical warning
"""),
    code(r"""for n in (4, 8, 12, 16):
    A_ = np.diag(np.arange(1, n + 1, dtype=float))                     # distinct, spread eigenvalues
    B_ = np.ones((n, 1))
    des = -np.arange(1, n + 1, dtype=float)
    K_ = sp.acker(A_, B_, des)
    got = np.sort(np.linalg.eigvals(A_ - B_ @ K_).real)
    print(f"n = {n:2d}: controllability matrix condition number {np.linalg.cond(sp.ctrb(A_, B_)):.1e}; largest pole error {np.max(np.abs(got - np.sort(des))):.1e}; |K| {np.max(np.abs(K_)):.1e}")"""),
    md(r"""
Ackermann's formula inverts the controllability matrix, whose condition number grows exponentially with the
order: by n = 12 the placed poles are wrong in the first digits and by 16 the computation has lost its meaning,
with gains of 10¹³ that no actuator could deliver. The lesson is twofold: use Ackermann only for small
systems (the library does, and switches to the Sylvester method otherwise), and distrust any pole placement
whose gains are enormous - the specification was unreasonable, not the arithmetic. LQR (notebook 10) chooses
poles the actuator can afford.
"""),
]

NOTEBOOKS["09_observers_and_the_separation_principle"] = [
    md(r"""
# 09 · Observers and the separation principle

State feedback needs the state, and sensors measure only some of it. A Luenberger observer runs a copy of the
model driven by the same input and corrects it with the measured output: the estimation error obeys
e′ = (A − LC)e, and its poles are placed with the gain L exactly as the controller's are placed with K, by
duality. Combined, the controller and observer give a closed loop whose eigenvalues are simply those of A − BK
and A − LC together - the separation principle. This notebook builds the observer, verifies the principle,
tracks a reference with the right feedforward gain, and shows what an observer costs in robustness.
"""),
    setup_cell(["from englincontrol import systems as sy, statespace as sp, response as rs, frequency as fr, models"]),
    md("## An observer for the pendulum, from the angle and the position"),
    code(r"""cp = models.cart_pendulum()["system"]
A, B, C = cp.A, cp.B, cp.C
K = sp.place(A, B, [-2 + 2j, -2 - 2j, -5.0, -6.0])
L = sp.observer_gain(A, C, [-10.0, -11.0, -12.0, -13.0])              # observer poles 2-5 times faster than the controller's
print("observer gain L (4 x 2) =\n", np.round(L, 2))
print("eig(A - L C) =", np.round(np.sort(np.linalg.eigvals(A - L @ C).real), 6))
obs = sp.observer(cp, L)
# simulate plant + observer from different initial states, open loop (u = 0 small perturbation)
t = np.linspace(0, 3, 601)
x0 = np.array([0.0, 0.0, 0.05, 0.0]); xh0 = np.zeros(4)
_, y_plant, x_plant = rs.initial(cp, x0, 3.0, 601)
uy = np.column_stack([np.zeros(601), y_plant])                     # observer inputs: [u, y1, y2]
_, xh, _ = rs.lsim(obs, uy, t, x0=xh0)
err = x_plant - xh
plt.semilogy(t, np.abs(err[:, 2]) + 1e-12, label="|angle estimation error|"); plt.semilogy(t, np.abs(err[:, 3]) + 1e-12, label="|angle-rate error|"); plt.legend(); plt.xlabel("t"); plt.title("the estimate converges at the observer's rate (even though the plant is unstable)"); plt.show()"""),
    md(r"""
The observer starts from a wrong state (zero, while the pendulum is at 3°) and its error decays like e^{−10t},
the slowest observer pole - while the plant itself is running away. Nothing about the plant's stability enters
the observer's convergence: the error dynamics are A − LC, chosen freely because (A, C) is observable. The
observer gain is placed on the dual system (Aᵀ, Cᵀ), the same computation as pole placement.

## The separation principle
"""),
    code(r"""cl = sp.closed_loop_with_observer(cp, K, L)
print("closed-loop eigenvalues (state and estimation error):", np.round(np.sort_complex(cl.poles()), 4))
print("= controller poles", np.round(np.sort_complex(np.linalg.eigvals(A - B @ K)), 4), "+ observer poles", np.round(np.sort(np.linalg.eigvals(A - L @ C).real), 4))
reg = sp.regulator(cp, K, L)
print("\nthe compensator (from [r, y1, y2] to u) has order", reg.n, "- the observer's state; its own poles", np.round(np.sort_complex(reg.poles()), 3), "need not be stable (here they are)")"""),
    md(r"""
The closed loop in the coordinates (x, e = x − x̂) has a block-triangular matrix: the estimation error does not
depend on the control, and the control drives x with x̂ = x − e. Its eigenvalues are exactly the union of the
two designed sets - design the controller as if the state were known, design the observer as if there were no
controller, and the pair works. The compensator that results is a dynamic system of the observer's order, an
output-feedback controller like any classical one, and its own poles are not the closed-loop poles.

## Tracking a reference
"""),
    code(r"""mot = models.dc_motor(position=True)["system"]
Km = sp.place(mot.A, mot.B, [-8 + 6j, -8 - 6j, -20.0])
Lm = sp.observer_gain(mot.A, mot.C, [-40.0, -45.0, -50.0])
Nbar = sp.reference_gain(mot.A, mot.B, mot.C, Km)
cl_m = sp.closed_loop_with_observer(mot, Km, Lm, Nbar)
t, y = rs.step(cl_m, 1.5)
print(f"reference gain Nbar = {Nbar:.3f}: the step response settles at {y[-1]:.4f} (not 1 without it: {rs.step(sp.closed_loop_with_observer(mot, Km, Lm, 1.0), 1.5)[1][-1]:.4f})")
plt.plot(t, y); plt.axhline(1, color="k", lw=0.5); plt.xlabel("t"); plt.title("DC motor position with observer-based state feedback"); plt.show()"""),
    md(r"""
State feedback regulates to zero; to follow a reference the input must be scaled by N̄ = 1/(C(−A + BK)⁻¹B) so
that the closed loop's dc gain is one. This is an open-loop correction - exact for the model, off for the real
plant - which is why integral action (notebook 10's LQR with an integrator) is preferred when the gain is
uncertain.

## What an observer costs: the LQG robustness problem in miniature
"""),
    code(r"""Pm = mot.to_tf()
# loop transfer at the plant input with full state feedback vs with the observer in the loop
L_state = sy.ss(mot.A, mot.B, Km, np.zeros((1, 1))).to_tf()
comp = sy.ss(mot.A - mot.B @ Km - Lm @ mot.C, Lm, Km, np.zeros((1, 1)))     # the compensator from y to -u
L_obs = sy.series(Pm, comp.to_tf())
for name, L_ in (("full state feedback", L_state), ("with the observer in the loop", L_obs)):
    m = fr.margins(L_)
    print(f"{name:<32}: gain margin {m['gain_margin']:.2f}, phase margin {m['phase_margin']:.1f} deg")"""),
    md(r"""
With the true state fed back the loop has an infinite gain margin and a comfortable phase margin; once the
observer supplies the state the margins shrink, because the compensator's dynamics add phase in the loop.
Doyle's 1978 observation that LQG has *no* guaranteed margins is this effect in general: the separation
principle guarantees nominal stability, not robustness, and the observer's speed and noise properties must be
chosen with the loop's margins in view (loop transfer recovery, in notebook 11's exercises).
"""),
]

NOTEBOOKS["10_the_linear_quadratic_regulator"] = [
    md(r"""
# 10 · The linear-quadratic regulator

Pole placement asks where the poles should go and leaves the answer to taste; the linear-quadratic regulator
asks what the control should cost. Minimising ∫(xᵀQx + uᵀRu) dt over all state feedbacks gives u = −Kx with
K = R⁻¹BᵀP and P the stabilising solution of the algebraic Riccati equation - a controller that trades state
error against control effort by weights an engineer can argue about, that works for any number of inputs, and
that comes with guaranteed margins. This notebook solves the Riccati equation, verifies optimality, sweeps the
weights, adds integral action, and measures the margins on the cart-pendulum and the DC motor.
"""),
    setup_cell(["from englincontrol import systems as sy, statespace as sp, linalg as la, response as rs, frequency as fr, models"]),
    md("## The Riccati equation and the optimal gain"),
    code(r"""cp = models.cart_pendulum()["system"]
A, B, C = cp.A, cp.B, cp.C
Q = np.diag([10.0, 1.0, 100.0, 1.0])          # penalise cart position and pendulum angle
R = np.array([[0.1]])
r = sp.lqr(A, B, Q, R)
K, P = r["K"], r["P"]
print("K =", np.round(K, 3), "\nclosed-loop poles", np.round(np.sort_complex(r["closed_loop_poles"]), 3))
print("Riccati residual |A'P + PA - PBR^-1B'P + Q| =", np.linalg.norm(A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T) @ P + Q))
x0 = np.array([0.0, 0.0, 0.2, 0.0])
print(f"optimal cost from x0: x0' P x0 = {sp.cost_to_go(P, x0):.4f};  the integral of x'Qx + u'Ru along the closed loop: {sp.lyapunov_cost(A - B @ K, Q + K.T @ R @ K, x0):.4f}")"""),
    md(r"""
The Riccati equation is quadratic in P and has many solutions; the stabilising one is found from the stable
invariant subspace of the Hamiltonian matrix and refined by Newton's method (notebook 18). Its residual is at
rounding level, and its value x₀ᵀPx₀ equals the cost actually incurred along the closed-loop trajectory,
computed independently from a Lyapunov equation: the optimality is verified, not assumed. The poles that come
out were never specified - they are where the cost puts them.

## Sweeping the weights
"""),
    code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
rows = []
for rho in (10.0, 1.0, 0.1, 0.01):
    rr = sp.lqr(A, B, Q, np.array([[rho]]))
    Acl = sy.ss(A - B @ rr["K"], B, C, np.zeros((2, 1)))
    t, y, x = rs.initial(Acl, x0, 4.0)
    u = -(x @ rr["K"].T)[:, 0]
    axes[0].plot(t, np.degrees(x[:, 2]), label=f"R = {rho}"); axes[1].plot(t, u)
    rows.append((rho, np.max(np.abs(u)), t[np.argmax(np.abs(x[:, 2]) < 0.01 * 0.2)] if np.any(np.abs(x[:, 2]) < 0.002) else np.nan, np.max(np.abs(x[:, 0])) * 100, np.max(np.abs(rr["closed_loop_poles"].real))))
axes[0].set_ylabel("angle (deg)"); axes[0].legend(fontsize=8); axes[1].set_ylabel("force (N)"); axes[0].set_xlabel("t"); axes[1].set_xlabel("t"); plt.show()
print(pd.DataFrame(rows, columns=["R", "peak force (N)", "angle settled (s)", "cart excursion (cm)", "fastest pole |Re|"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
Lowering the control weight R buys speed with force: the pendulum settles faster, the peak force and the
fastest pole grow. The weights are the design knobs, and their effect is monotone and predictable - which is
why LQR is the method of choice when the actuator's limits are known: choose R so that the peak force fits
them. Bryson's rule (weight each variable by the inverse square of its allowed excursion) is the usual
starting point.

## Integral action
"""),
    code(r"""mot = models.dc_motor()["system"]
li = sp.lqr_integral(mot.A, mot.B, mot.C, np.diag([1.0, 10.0]), np.array([[0.01]]), qi=1e4)
print("closed-loop poles with integral action:", np.round(np.sort_complex(li["closed_loop_poles"]), 3))
Aa, Ba = li["Aa"], li["Ba"]
Ka = np.hstack([li["K"], li["Ki"]])
# closed loop of the augmented system: xa' = (Aa - Ba Ka) xa + [0; 1] r, y = C x
Acl = Aa - Ba @ Ka
Bcl = np.array([[0.0], [0.0], [1.0]])
Ccl = np.hstack([mot.C, np.zeros((1, 1))])
cl = sy.ss(Acl, Bcl, Ccl, 0.0)
t, y = rs.step(cl, 2.0)
# a model error: the real motor has 30 % more friction
mot_real = models.dc_motor(b=0.13)["system"]
Acl_r = np.block([[mot_real.A - mot_real.B @ li["K"], -mot_real.B @ li["Ki"]], [-mot_real.C, np.zeros((1, 1))]])
t2, y2 = rs.step(sy.ss(Acl_r, Bcl, np.hstack([mot_real.C, np.zeros((1, 1))]), 0.0), 2.0)
Km = sp.lqr(mot.A, mot.B, np.diag([1.0, 10.0]), np.array([[0.01]]))["K"]
Nbar = sp.reference_gain(mot.A, mot.B, mot.C, Km)
t3, y3 = rs.step(sy.ss(mot_real.A - mot_real.B @ Km, mot_real.B * Nbar, mot_real.C, 0.0), 2.0)
plt.plot(t, y, label="LQR + integral, nominal motor"); plt.plot(t2, y2, "--", label="LQR + integral, 30 % more friction"); plt.plot(t3, y3, ":", label="LQR + reference gain Nbar, 30 % more friction"); plt.axhline(1, color="k", lw=0.5); plt.legend(fontsize=8); plt.xlabel("t"); plt.title("speed control of the DC motor"); plt.show()
print(f"final values: {y[-1]:.4f}, {y2[-1]:.4f}, {y3[-1]:.4f}")"""),
    md(r"""
Augmenting the state with the integral of the tracking error and weighting it turns LQR into a controller with
integral action: the speed settles at the reference even when the motor's friction is 30 % off the model,
where the feedforward gain N̄ leaves a 23 % offset. The integral weight q_i sets how fast the offset is
removed; the trade-off with overshoot is the same as for any PI loop.

## Guaranteed margins
"""),
    code(r"""for name, (A_, B_, Q_, R_) in (("DC motor", (mot.A, mot.B, np.diag([1.0, 10.0]), np.array([[0.01]]))), ("cart-pendulum", (A, B, Q, R))):
    m = sp.lqr_margins(A_, B_, Q_, R_)
    gms = [float(round(g_, 3)) for g_ in m["gain_margins"]] or ["none (never crosses -180)"]
    print(f"{name:<14}: loop K (sI - A)^-1 B at the plant input: gain margins at the phase crossovers {gms} (guaranteed: can be reduced to 1/2 and raised without limit), phase margin {m['phase_margin']:.1f} deg (>= 60 guaranteed)")
L_lqr = sy.ss(A, B, K, np.zeros((1, 1))).to_tf()
w = np.logspace(-1, 2, 2000)
_, Lj = fr.nyquist(L_lqr, w)
print(f"return difference |1 + L| >= 1 at every frequency: minimum {np.min(np.abs(1 + Lj)):.4f} (Kalman's inequality)")"""),
    md(r"""
Kalman's return-difference inequality |1 + L(jω)| ≥ 1 says the LQR loop never comes closer than 1 to −1: a
gain margin from 1/2 to infinity and a phase margin of at least 60°, for any plant and any weights, at the
plant input. The motor's loop never reaches −180° (no gain crossover of the phase, hence no finite gain
margin); the pendulum's, being unstable in open loop, has a *downward* gain margin - the loop goes unstable if
the gain falls to 0.21, below the guaranteed 1/2 - and none upward. These are the margins that LQG loses when an observer is inserted (notebook 11) and that loop
transfer recovery tries to regain. They hold for the state-feedback loop; what the designer still has to
check is the actuator and the model's validity at the resulting bandwidth.
"""),
]

NOTEBOOKS["11_kalman_filtering_and_lqg"] = [
    md(r"""
# 11 · Kalman filtering and LQG

The Luenberger observer placed the estimator's poles by taste; the Kalman filter places them by the noise:
given the covariances of the process disturbance and the measurement noise, it is the estimator with the
smallest error variance, and its gain comes from a Riccati equation dual to the regulator's. Combined with LQR
it gives the LQG compensator, the optimal output-feedback controller for linear systems with Gaussian noise.
This notebook builds the filter in continuous and discrete time, runs it on noisy data, verifies its error
covariance against the Riccati prediction, assembles LQG, and shows the robustness problem Doyle found.
"""),
    setup_cell(["from englincontrol import systems as sy, statespace as sp, discrete as dc, response as rs, frequency as fr, models"]),
    md("## The steady-state Kalman filter"),
    code(r"""mot = models.dc_motor()["system"]
A, B, C = mot.A, mot.B, mot.C
Qn = np.diag([1e-4, 1e-2])       # process noise on current and speed
Rn = np.array([[0.25]])          # tachometer noise variance (0.5 rad/s rms)
kf = sp.kalman(A, C, Qn, Rn)
print("Kalman gain L =", np.round(kf["L"].ravel(), 4), "\nestimator poles", np.round(kf["estimator_poles"], 3), "\nerror covariance P =\n", np.round(kf["P"], 5))
print("duality: the Kalman gain is the LQR gain of (A^T, C^T) with weights (Qn, Rn), transposed:", np.allclose(kf["L"], sp.lqr(A.T, C.T, Qn, Rn)["K"].T))"""),
    md(r"""
The filter gain L = PCᵀR_n⁻¹ comes from the dual Riccati equation AP + PAᵀ − PCᵀR_n⁻¹CP + Q_n = 0, and P is the
steady-state covariance of the estimation error - a prediction of how good the estimate will be before any
data are seen. The duality with LQR is exact: the Kalman gain is the LQR gain of (Aᵀ, Cᵀ) transposed, with
the noise covariances in the role of the weights.

## The filter on data: discrete time
"""),
    code(r"""dt = 0.005
Gd = dc.c2d(mot, dt)
Ad, Bd, Cd = Gd.A, Gd.B, Gd.C
Qd, Rd = np.diag([1e-6, 1e-4]), np.array([[0.25]])
kd = sp.kalman(Ad, Cd, Qd, Rd, discrete=True)
rng = np.random.default_rng(1)
N = 2000
u = 6.0 * (1 + np.sign(np.sin(0.5 * np.arange(N) * dt)))                 # a square-wave voltage
x = np.zeros(2); X = np.zeros((N, 2)); Y = np.zeros(N)
for k in range(N):
    X[k] = x
    Y[k] = (Cd @ x)[0] + np.sqrt(Rd[0, 0]) * rng.normal()
    x = Ad @ x + Bd[:, 0] * u[k] + rng.normal(size=2) * np.sqrt(np.diag(Qd))
Xh = sp.kalman_filter_run(Ad, Bd, Cd, kd["L"], u, Y)
t = np.arange(N) * dt
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(t, Y, ".", ms=2, color="0.7", label="measured speed"); axes[0].plot(t, X[:, 1], "k", lw=1, label="true speed"); axes[0].plot(t, Xh[:, 1], "r", lw=1, label="Kalman estimate"); axes[0].legend(fontsize=8); axes[0].set_ylabel("rad/s")
axes[1].plot(t, X[:, 0], "k", lw=1, label="true current (not measured)"); axes[1].plot(t, Xh[:, 0], "r", lw=1, label="estimated current"); axes[1].legend(fontsize=8); axes[1].set_ylabel("A"); axes[1].set_xlabel("t")
plt.show()
e = Xh[200:] - X[200:]
print(f"error variance of the speed estimate {np.var(e[:, 1]):.5f} vs the Riccati prediction P_22 = {kd['P'][1, 1]:.5f}; measurement noise variance {Rd[0, 0]}: the filter removes {100 * (1 - np.var(e[:, 1]) / Rd[0, 0]):.0f} % of it")
print(f"the unmeasured current is estimated with error variance {np.var(e[:, 0]):.2e} (prediction {kd['P'][0, 0]:.2e})")"""),
    md(r"""
On noisy tachometer data the filter returns a speed estimate with a fraction of the measurement's variance,
and an estimate of the *unmeasured* current - the model's structure makes the current observable through the
speed. Both error variances are of the size the Riccati equation predicts (within the uncertainty of a 10 s
record whose errors are correlated in time), which is the test that the noise model was right; when it is
wrong the measured and predicted variances disagree by factors, and that disagreement is how a filter is
tuned in practice.

## LQG: the regulator on the estimate
"""),
    code(r"""cp = models.cart_pendulum()["system"]
Q = np.diag([10.0, 1.0, 100.0, 1.0]); R = np.array([[0.1]])
g = sp.lqg(cp, Q, R, 1e-3 * np.eye(4), 1e-4 * np.eye(2))
print("regulator poles", np.round(np.sort_complex(g["regulator_poles"]), 3), "\nestimator poles", np.round(np.sort_complex(g["estimator_poles"]), 3))
comp = g["compensator"]
Acl = np.block([[cp.A, cp.B @ comp.C], [comp.B @ cp.C, comp.A]])
print("closed-loop eigenvalues = regulator + estimator poles:", np.allclose(np.sort_complex(np.linalg.eigvals(Acl)), np.sort_complex(np.r_[g['regulator_poles'], g['estimator_poles']])))"""),
    md(r"""
LQG is the separation principle with optimal halves: the LQR gain on the Kalman estimate. The compensator is a
dynamic system of the plant's order, driven by the measurements, and the closed loop's eigenvalues are again
the two designed sets. It is optimal for the stated noise - and that is all it is optimal for.

## Doyle's warning: LQG has no guaranteed margins
"""),
    code(r"""from englincontrol import robust as rb
K_lqr = sp.lqr(cp.A, cp.B, Q, R)["K"]
L_state = sy.ss(cp.A, cp.B, K_lqr, np.zeros((1, 1))).to_tf()
print(f"state-feedback loop at the plant input: closest approach to -1 is {rb.nyquist_distance(L_state)['distance']:.3f} (Kalman's inequality: >= 1)")
rows = []
for q_scale in (1e-6, 1e-3, 1.0, 1e2):
    g_ = sp.lqg(cp, Q, R, q_scale * np.eye(4), 1e-4 * np.eye(2))
    comp = g_["compensator"]
    L_in = sy.series(cp, sy.ss(comp.A, comp.B, -comp.C, np.zeros((1, 2)))).to_tf()      # loop at the plant input, sign for negative feedback
    d = rb.nyquist_distance(L_in)
    rows.append((q_scale, f"{np.min(g_['estimator_poles'].real):.0f} .. {np.max(g_['estimator_poles'].real):.1f}", d["distance"], 1 / d["distance"]))
print(pd.DataFrame(rows, columns=["process noise scale", "estimator poles (Re)", "closest approach to -1", "Ms"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
The state-feedback loop keeps the Nyquist curve at distance 1 from −1; the same regulator driven by a Kalman
estimate comes within 0.21 of it (M_s = 4.7) - a loop that a 20 % gain error would destabilise. Raising the
assumed process noise speeds up the estimator and recovers part of the distance (loop transfer recovery,
Doyle and Stein's remedy), but it saturates at 0.5 with the estimator poles already at −1000: the cart's
position is a non-minimum-phase output of this plant, and recovery is known to fail for such outputs. (Pushing
the estimator further makes the loop's transfer function numerically unreliable - the next row of this table
differs between machines - which is its own lesson about computing with poles at −3000.) Doyle's 1978 paper gave an LQG design
with arbitrarily small margins; the general lesson is that optimality for a performance index is not
robustness, and the loop that will actually be closed must be checked. (With several crossovers and an
unstable plant the classical gain and phase margins become ambiguous, which is why the closest approach to
−1 is the measure used here.)
"""),
]

NOTEBOOKS["12_sampling_and_discretisation"] = [
    md(r"""
# 12 · Sampling and discretisation

A digital controller sees the plant through a sampler and acts through a hold: it works with sequences, and
its design works with discrete-time models. This notebook builds the exact zero-order-hold equivalent of a
continuous plant from the matrix exponential, compares it with Tustin's transform (and its prewarped
variant), matched mapping and Euler's rules, shows where the poles go (z = e^{sΔt}) and what the hold costs (a
half-sample delay), demonstrates aliasing, and gives the rules for choosing the sampling rate.
"""),
    setup_cell(["from englincontrol import systems as sy, discrete as dc, response as rs, frequency as fr, stability as st, models"]),
    md("## The zero-order-hold equivalent is exact"),
    code(r"""G = sy.tf([4.0], [1.0, 1.2, 4.0])
dt = 0.1
Gd = dc.c2d(G, dt)
print("continuous:", G, "\nZOH equivalent:", Gd)
print("poles map by z = e^{s dt}:", np.round(np.sort_complex(Gd.poles()), 5), "=", np.round(np.sort_complex(np.exp(G.poles() * dt)), 5))
t, y = rs.step(G, 5.0, 51); td, yd = rs.step(Gd, 5.0)
print(f"the discrete step response equals the continuous one at every sample: max difference {np.max(np.abs(y - yd)):.1e}")
Gss = G.to_ss()
print("d2c recovers A by the matrix logarithm:", np.max(np.abs(dc.d2c(dc.c2d(Gss, dt)).A - Gss.A)) < 1e-12)"""),
    md(r"""
With a zero-order hold the input is constant between samples, and then x_{k+1} = e^{AΔt} x_k + (∫₀^{Δt} e^{Aτ}dτ) B u_k
is exact - the same formulas as the time responses of notebook 01. The poles move to z = e^{sΔt}: the left
half-plane maps inside the unit circle, the imaginary axis onto it. The zeros do not map so simply: ZOH
discretisation adds sampling zeros, which can even fall outside the unit circle for fast sampling of
high-relative-degree plants - a discrete non-minimum-phase behaviour with no continuous counterpart.

## Tustin, prewarping, matched poles, Euler
"""),
    code(r"""w = np.logspace(-1, np.log10(np.pi / dt), 300)
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(w, 20 * np.log10(np.abs(fr.freqresp(G, w))), "k", lw=2, label="continuous")
for meth, kw in (("zoh", {}), ("tustin", {}), ("prewarp", {"w_prewarp": 2.0}), ("matched", {}), ("euler", {}), ("backward", {})):
    Gm = dc.c2d(G, dt, meth, **kw)
    ax.semilogx(w, 20 * np.log10(np.abs(fr.freqresp(Gm, w))), label=meth + (" at w = 2" if kw else ""))
ax.axvline(2.0, color="0.8"); ax.axvline(np.pi / dt, color="0.6", ls="--"); ax.set_xlabel("w (rad/s)"); ax.set_ylabel("|G| dB"); ax.legend(fontsize=8); ax.set_title("discretisations of the prototype at dt = 0.1 (Nyquist frequency dashed)"); plt.show()
for meth, kw in (("tustin", {}), ("prewarp", {"w_prewarp": 2.0})):
    Gm = dc.c2d(G, dt, meth, **kw)
    print(f"{meth:<8}: |G_d(e^(j 2 dt))| = {abs(fr.freqresp(Gm, 2.0)):.5f} vs |G(2j)| = {abs(fr.freqresp(G, 2.0)):.5f}" + ("" if kw else f"  (Tustin's warping: the discrete response at w equals the continuous one at (2/dt) tan(w dt/2) = {2 / dt * np.tan(2.0 * dt / 2):.4f})"))
print("forward Euler of a stable system can be unstable: poles of euler at dt = 1:", np.round(np.abs(dc.c2d(G, 1.0, 'euler').poles()), 3))"""),
    md(r"""
Tustin's bilinear transform s = (2/Δt)(z − 1)/(z + 1) maps the whole imaginary axis onto the unit circle, so it
preserves stability and the shape of the frequency response - compressed: the discrete response at ω equals
the continuous one at (2/Δt)tan(ωΔt/2), the warping that prewarping corrects at one chosen frequency (the
resonance, say). Matched mapping places poles and zeros at e^{sΔt} and fixes the dc gain. Forward Euler is
the simplest and the worst: it maps the left half-plane onto a half-plane, not the unit disc, and a stable
system can come out unstable at a large step. For plant models ZOH is the right choice (it *is* what the
hold does); for controllers Tustin is the usual one.

## What the hold costs: a half-sample delay
"""),
    code(r"""for dt_ in (0.01, 0.05, 0.1, 0.2):
    H = dc.zoh_frequency_response(2.0, dt_)
    print(f"dt = {dt_:4}: zero-order hold at w = 2: |H| = {abs(H):.5f}, phase {np.degrees(np.angle(H)):+.2f} deg = -w dt/2 = {np.degrees(-2.0 * dt_ / 2):+.2f} deg")
L = sy.tf([10.0], [1.0, 1.0, 0.0])
m0 = fr.margins(L)
print(f"\nthe loop 10/(s(s+1)) has phase margin {m0['phase_margin']:.1f} deg at w_gc = {m0['w_gc']:.2f};")
for dt_ in (0.02, 0.1, 0.3):
    m = fr.margins(L, delay=dt_ / 2)
    print(f"   sampled at dt = {dt_}: the hold's half-sample delay costs {m0['phase_margin'] - m['phase_margin']:.1f} deg of phase margin -> {m['phase_margin']:.1f} deg")"""),
    md(r"""
The zero-order hold's frequency response (1 − e^{−jωΔt})/(jωΔt) has unit magnitude at low frequency and a
phase of −ωΔt/2: it behaves as a delay of half a sample. In a loop with crossover ω_c that delay removes
ω_cΔt/2 radians of phase margin, which is where the sampling-rate rules come from: for the loss to stay
under 5-10°, ω_cΔt must be about 0.2-0.3.

## Aliasing and the sampling rate
"""),
    code(r"""fs_dt = 0.1
t_fine = np.linspace(0, 2, 2001)
w_true = 100.0                                                        # above the Nyquist frequency pi/dt = 31.4
x = np.sin(w_true * t_fine)
t_s = np.arange(0, 2, fs_dt)
plt.plot(t_fine, x, "0.7", lw=0.8, label=f"true signal, {w_true} rad/s"); plt.plot(t_s, np.sin(w_true * t_s), "ro-", label=f"samples at dt = {fs_dt}: look like {dc.alias(w_true, fs_dt):.2f} rad/s"); plt.legend(fontsize=8); plt.xlabel("t"); plt.show()
rules = dc.sample_rate_rules(5.0)
print(f"for a closed-loop bandwidth of 5 rad/s: dt between {rules['dt_min']:.3f} and {rules['dt_max']:.3f} s (w_b dt in 0.2-0.6), recommended {rules['dt_recommended']:.3f} s")
print(f"an anti-aliasing filter must then attenuate everything above the Nyquist frequency {dc.nyquist_frequency(rules['dt_recommended']):.0f} rad/s")"""),
    md(r"""
A 100 rad/s component sampled at Δt = 0.1 is indistinguishable from one at 25.7 rad/s: it folds back into the
band the controller acts on and appears as a slow disturbance that is not there. Hence the two rules of digital
control: sample fast enough for the loop (ω_bΔt ≈ 0.2-0.6, a compromise between the hold's delay and the
numerical problems of very fast sampling seen in notebook 15), and filter the measurement before sampling so
that nothing above the Nyquist frequency reaches the sampler.
"""),
]

NOTEBOOKS["13_digital_control_design"] = [
    md(r"""
# 13 · Digital control design

Two routes lead to a digital controller: design in continuous time and discretise the controller (emulation),
or discretise the plant and design in the z-domain directly. This notebook does both for the DC motor - a
Tustin-discretised lead and PID at several sampling rates, then discrete pole placement, dead-beat control,
discrete LQR and the discrete Kalman filter on the ZOH model - compares them against the continuous design,
and ends with the implementation details that decide whether the code on the microcontroller matches the
design: the computation delay, the integrator's discretisation, and the derivative filter.
"""),
    setup_cell(["from englincontrol import systems as sy, discrete as dc, design as dg, statespace as sp, response as rs, frequency as fr, stability as st, models"]),
    md("## Emulation: discretise a continuous design"),
    code(r"""mot = models.dc_motor(position=True)
P = mot["system"].to_tf()
d = dg.design_lead(P, w_c=5.0, pm_target_deg=50.0)
C = d["controller"]
print("continuous lead design: crossover 5 rad/s, phase margin", round(d["margins"]["phase_margin"], 1), "deg")
rows = []
fig, ax = plt.subplots(figsize=(8, 3.8))
t, y = rs.step(sy.feedback(sy.series(C, P)), 3.0); ax.plot(t, y, "k", lw=2, label="continuous design")
for dt in (0.02, 0.1, 0.25):
    Cd = dc.c2d(C, dt, "tustin")
    Pd = dc.c2d(P, dt)
    Ld = sy.series(Cd, Pd)
    m = fr.margins(Ld)
    td, yd = rs.step(sy.feedback(Ld), 3.0)
    ax.step(td, yd, where="post", label=f"Tustin controller at dt = {dt}: PM {m['phase_margin']:.0f} deg")
    rows.append((dt, 5.0 * dt, m["phase_margin"], rs.step_info(td, yd, final=1.0)["overshoot"]))
ax.axhline(1, color="k", lw=0.5); ax.legend(fontsize=8); ax.set_xlabel("t"); plt.show()
print(pd.DataFrame(rows, columns=["dt", "w_c dt", "phase margin (deg)", "overshoot %"]).to_string(index=False, float_format="%.2f"))"""),
    md(r"""
The continuous design survives emulation when ω_cΔt is small (0.1: the margin drops from 55° to 49°) and
degrades as the sampling slows: at ω_cΔt = 1.25 the hold's half-sample delay has eaten 35° of margin and the
response rings. Emulation is the quick route; its rule is the one of notebook 12, ω_cΔt ≤ 0.3, and its check
is a discrete analysis of the loop as it will really run - the ZOH plant with the discretised controller.

## Direct design on the ZOH model: pole placement and dead-beat
"""),
    code(r"""dt = 0.02
Gd = dc.c2d(mot["system"], dt)
Ad, Bd, Cd = Gd.A, Gd.B, Gd.C
s_poles = np.array([-8 + 6j, -8 - 6j, -20.0])
z_poles = np.exp(s_poles * dt)
Kz = sp.place(Ad, Bd, z_poles)
print("continuous targets", s_poles, "-> z targets", np.round(z_poles, 4), "\nplaced:", np.round(np.sort_complex(np.linalg.eigvals(Ad - Bd @ Kz)), 4))
Kdb = dc.deadbeat(Ad, Bd)
Acl_db = Ad - Bd @ Kdb
print(f"dead-beat gain {np.round(Kdb, 1)}: (A - BK)^3 = {np.max(np.abs(np.linalg.matrix_power(Acl_db, 3))):.1e} - any state is driven to zero in 3 samples")
x0 = np.array([0.0, 0.0, 0.5])      # 0.5 rad position error
for name, Kk in (("pole placement", Kz), ("dead-beat", Kdb)):
    x = x0.copy(); us = []
    for k in range(6):
        u = -(Kk @ x)[0]; us.append(u); x = Ad @ x + Bd[:, 0] * u
    print(f"{name:<15}: control sequence (V) {np.round(us, 1)}, position after 6 samples {x[2]:.2e}")"""),
    md(r"""
Direct design maps the continuous specification to z = e^{sΔt} and places the poles there; the result is the
continuous design's discrete twin. Dead-beat control, with all poles at z = 0, has no continuous counterpart:
it drives any state to zero in n samples exactly - and demands, for a 0.5 rad error at Δt = 0.02, a first
control move of several thousand volts. A finite settling time in n samples is a specification about samples,
not about physics; it is useful only when Δt is long enough for the actuator to deliver.

## Discrete LQR and Kalman filter
"""),
    code(r"""Qz, Rz = np.diag([1.0, 10.0, 1000.0]), np.array([[0.1]])
rd = sp.lqr(Ad, Bd, Qz, Rz, discrete=True)
rc = sp.lqr(mot["system"].A, mot["system"].B, Qz, Rz)
print("discrete LQR poles", np.round(np.sort_complex(rd["closed_loop_poles"]), 4), "\ncontinuous LQR poles mapped by e^{s dt}:", np.round(np.sort_complex(np.exp(rc["closed_loop_poles"] * dt)), 4), "(close, not equal: the discrete cost sums, the continuous one integrates)")
kd = sp.kalman(Ad, Cd, np.diag([1e-6, 1e-4, 1e-6]), np.array([[1e-4]]), discrete=True)
print("discrete Kalman gain (predictor form)", np.round(kd["L"].ravel(), 4), "estimator poles", np.round(np.abs(kd["estimator_poles"]), 4))"""),
    md(r"""
The discrete LQR minimises a sum instead of an integral; its poles sit near, but not exactly at, the images of
the continuous LQR's. The discrete Kalman filter is the natural form on sampled data (notebook 11 ran it); in
predictor form its gain multiplies the innovation y_k − Cx̂_k to produce x̂_{k+1}, so the estimate used for
control at step k + 1 is available before the measurement at that step is taken - a full sample of latency
that the current-estimator form avoids.

## Implementation details that change the loop
"""),
    code(r"""dt = 0.02
Cz_back = dc.pid_discrete(3.0, 2.0, 0.3, dt, method="backward")
Cz_tust = dc.pid_discrete(3.0, 2.0, 0.3, dt, method="tustin")
C_cont = dg.pid(3.0, 2.0, 0.3)
w = np.logspace(-1, np.log10(np.pi / dt), 300)
for name, Cz in (("backward-rectangle integral", Cz_back), ("Tustin integral", Cz_tust)):
    print(f"{name:<28}: |C_d - C| at w = 1: {abs(fr.freqresp(Cz, 1.0) - fr.freqresp(C_cont, 1.0)):.4f}, at w = 50: {abs(fr.freqresp(Cz, 50.0) - fr.freqresp(C_cont, 50.0)):.3f}")
Pd = dc.c2d(mot["system"].to_tf(), dt)
Cz_lead = dc.c2d(C, dt, "tustin")                                   # the lead design of the first section (crossover 5 rad/s)
for comp_delay in (0.0, 1.0):
    Lz = sy.series(Cz_lead, Pd)
    if comp_delay:
        Lz = sy.series(Lz, sy.tf([1.0], [1.0, 0.0], dt))           # one sample of computation delay: 1/z
    m = fr.margins(Lz)
    print(f"lead loop with computation delay {comp_delay:.0f} sample(s): phase margin {m['phase_margin']:.1f} deg at w_gc = {m['w_gc']:.2f} rad/s (one sample at w_gc costs w_gc dt = {np.degrees(m['w_gc'] * dt):.1f} deg)")"""),
    md(r"""
Two discretisations of the same PID differ by a fraction of a percent at the loop's frequencies and by more
near the Nyquist frequency; the choice matters little compared with the computation delay. If the control
computed from the sample at k is applied only at k + 1, a full sample of delay enters the loop - ω_cΔt radians,
here 6° of phase margin at Δt = 0.02 with a 5 rad/s crossover, and proportionally more at slower rates. Reading the sensor, computing and writing
the actuator within a small fraction of the sample period, in that order, is the implementation rule that
keeps the design valid.
"""),
]

NOTEBOOKS["14_robustness_and_limitations"] = [
    md(r"""
# 14 · Robustness and fundamental limitations

A controller designed on a model must work on the plant, which differs from the model. This notebook measures
robustness through the gang of four (S, T, PS, CS), the sensitivity peak M_s and the disk margin; shows that
sensitivity cannot be reduced everywhere (Bode's integral, the waterbed effect) and how unstable poles make it
worse; derives the bandwidth limits imposed by right-half-plane zeros, unstable poles and delays; and applies
the small-gain theorem to multiplicative model uncertainty. These limits do not depend on the controller: they
say what no controller can do.
"""),
    setup_cell(["from englincontrol import systems as sy, robust as rb, frequency as fr, design as dg, response as rs, polynomials as pl, models"]),
    md("## The gang of four"),
    code(r"""P = sy.tf([1.0], [1.0, 3.0, 2.0, 0.0])
C = dg.pid(2.0, 0.3, 1.0)
g4 = rb.gang_of_four(P, C)
w = np.logspace(-2, 2, 500)
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for ax, name, note in zip(axes.ravel(), ("S", "T", "PS", "CS"), ("output disturbance -> output: sensitivity", "reference -> output: complementary sensitivity", "input disturbance -> output: load sensitivity", "noise -> control: noise sensitivity")):
    ax.semilogx(w, 20 * np.log10(np.abs(fr.freqresp(g4[name], w)))); ax.set_title(f"{name}: {note}", fontsize=9); ax.axhline(0, color="k", lw=0.5)
plt.show()
pk = rb.sensitivity_peak(g4["L"])
print(f"Ms = {pk['Ms']:.3f} at w = {pk['w_Ms']:.2f} rad/s, Mt = {pk['Mt']:.3f}; Ms alone guarantees a gain margin of at least {pk['gm_lower_bound']:.2f} and a phase margin of at least {pk['pm_lower_bound_deg']:.1f} deg")
print(f"actual margins: {fr.margins(g4['L'])['gain_margin']:.2f} and {fr.margins(g4['L'])['phase_margin']:.1f} deg; disk margin alpha = {rb.disk_margin(g4['L'])['alpha']:.3f}")"""),
    md(r"""
Four transfer functions describe a loop completely: S for disturbances at the output and for how much the
loop reduces model errors (|S| < 1 is where feedback helps), T for reference tracking and for how
measurement noise reaches the output, PS for input disturbances, CS for how much noise the actuator sees.
S + T = 1 at every frequency, so both cannot be small at once: the loop chooses, by frequency. The sensitivity
peak M_s is the most useful single robustness number - the inverse of the Nyquist curve's closest approach to
−1 - and it bounds both classical margins; the disk margin generalises it to simultaneous gain and phase
changes.

## Bode's sensitivity integral: the waterbed
"""),
    code(r"""fig, ax = plt.subplots(figsize=(8, 3.8))
for name, L in (("stable loop", sy.tf([1.0], [1.0, 3.0, 2.0, 0.0])), ("stabilised unstable loop, pole at +1", sy.tf([30.0, 60.0], pl.from_roots([1, -5, -8])))):
    bi = rb.bode_sensitivity_integral(L)
    S = 1 / (1 + fr.freqresp(L, w))
    ax.semilogx(w, np.log(np.abs(S)), label=f"{name}: integral of ln|S| = {bi['integral']:.3f} (theory {bi['theory']:.3f})")
ax.axhline(0, color="k", lw=0.5); ax.fill_between(w, 0, 0, color="C0"); ax.legend(fontsize=8); ax.set_xlabel("w"); ax.set_ylabel("ln |S|"); plt.show()"""),
    md(r"""
∫₀^∞ ln|S(jω)| dω = π Σ Re(p_k) over the unstable open-loop poles, for loops of relative degree at least two:
the area below zero (where feedback helps) equals the area above (where it hurts) for a stable loop, and the
balance tips against the designer by π per unit of unstable pole. Push |S| down at low frequency and it pops up
elsewhere - the waterbed. The only escapes are a loop with relative degree one (an infinite bandwidth, which
no actuator offers) or accepting the peak where the model is least trusted.

## Right-half-plane zeros, unstable poles, delays
"""),
    code(r"""Pz = sy.tf([-1.0, 2.0], [1.0, 2.0, 1.0])                       # (2 - s)/(s + 1)^2: a RHP zero at 2
lim = rb.rhp_limits(Pz)
print(f"RHP zero at {lim['rhp_zeros'][0].real}: the crossover should stay below about {lim['w_c_max_from_zero']:.1f} rad/s")
fig, ax = plt.subplots(figsize=(8, 3.8))
for K in (0.3, 0.6, 1.0):
    C = sy.tf([K, 0.5 * K], [1.0, 0.0])                             # PI with the zero at -0.5
    L = sy.series(C, Pz)
    T = sy.feedback(L)
    t, y = rs.step(T, 15.0)
    m = fr.margins(L)
    ax.plot(t, y, label=f"K = {K}: crossover {m['w_gc']:.2f}, PM {m['phase_margin']:.0f} deg, undershoot {-min(y) * 100:.0f} %")
ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="k", lw=0.5); ax.legend(fontsize=8); ax.set_xlabel("t"); plt.show()
lim2 = rb.rhp_limits(sy.tf([1.0, -4.0], [1.0, -1.0, 0.0]))
print(f"\nunstable pole at 1 and RHP zero at 4: any stabilising controller has Ms >= |p + z|/|p - z| = {lim2['Ms_lower_bound']:.3f}; with the zero at 1.5 it would be {rb.rhp_limits(sy.tf([1.0, -1.5], [1.0, -1.0, 0.0]))['Ms_lower_bound']:.1f}")
print(f"a delay of 0.2 s limits the crossover to about {rb.rhp_limits(P, delay=0.2)['w_c_max_from_delay']:.0f} rad/s")"""),
    md(r"""
A right-half-plane zero at z bounds the achievable bandwidth to roughly z/2: pushing the crossover towards it
costs phase (the zero's phase goes the wrong way) and deepens the inverse response, and a RHP zero close to an
unstable pole makes the plant nearly uncontrollable in practice - M_s ≥ |p + z|/|p − z| grows without limit as
they approach. A delay L bounds the crossover to about 1/L. These are properties of the plant, and the right
response is to change the plant (move the sensor, remove the delay) rather than to fight them with a cleverer
controller.

## Model uncertainty and the small-gain theorem
"""),
    code(r"""P = sy.tf([1.0], [1.0, 3.0, 2.0, 0.0])
C = dg.pid(2.0, 0.3, 1.0)
L = sy.series(C, P)
W = sy.tf([0.5, 0.0], [1.0 / 20.0, 1.0])                     # multiplicative uncertainty: 0 at dc, 50 % at 10 rad/s, 1000 % at high frequency
r = rb.multiplicative_robustness(L, W)
print(f"robust stability for P (1 + W Delta), |Delta| <= 1: peak of |W T| = {r['peak_WT']:.3f} at w = {r['w_peak']:.2f} -> {'robustly stable' if r['robustly_stable'] else 'NOT guaranteed'}")
C2 = dg.pid(6.0, 1.0, 3.0)
r2 = rb.multiplicative_robustness(sy.series(C2, P), W)
print(f"a faster controller (3x the gains): peak |W T| = {r2['peak_WT']:.3f} -> {'robustly stable' if r2['robustly_stable'] else 'NOT guaranteed'}; its nominal margins: GM {fr.margins(sy.series(C2, P))['gain_margin']:.2f}, PM {fr.margins(sy.series(C2, P))['phase_margin']:.0f} deg")
sg = rb.small_gain(sy.tf([0.5], [1.0, 1.0]), 1.5)
print(f"\nsmall-gain theorem on M = 0.5/(s + 1) with |Delta| < 1.5: ||M||_inf = {sg['M_inf']:.3f}, product {sg['M_inf'] * 1.5:.3f} < 1 -> stable for every such Delta")"""),
    md(r"""
If the true plant is P(1 + WΔ) with an unknown stable Δ of gain at most one, the loop stays stable for every Δ
exactly when |W T| < 1 at all frequencies: the complementary sensitivity must roll off before the model's
uncertainty grows. The faster controller has good *nominal* margins and still fails the test, because its
bandwidth reaches into the frequencies where the model is 50 % wrong. The small-gain theorem is the general
statement: a loop of two stable systems is stable if the product of their gains is below one - the
foundation of the robust-control methods beyond this course.
"""),
]

NOTEBOOKS["15_system_identification"] = [
    md(r"""
# 15 · System identification

Models come from physics or from data, and usually from both: the structure from physics, the numbers from
experiments. This notebook identifies the course's DC motor from its measured step response and from a
pseudo-random binary excitation, in time and in frequency domain. It shows the two-point FOPDT fit and a
second-order fit, then the ARX least-squares model and its failure mode - bias from output noise, amplified by
fast sampling - and the two remedies: a slower identification rate and the output-error fit on the simulation
error. The residuals' whiteness and a frequency-response fit close the chapter.
"""),
    setup_cell(["from englincontrol import systems as sy, identify as idf, design as dg, discrete as dc, response as rs, frequency as fr, datasets, models"]),
    md("## Step-response data: cleaning and fitting"),
    code(r"""d = pd.read_csv(datasets.path("motor_step.csv"), comment="#")
print(f"{len(d)} samples at 1 kHz; {d.speed_rad_s.isna().sum()} dropped sample(s) -> interpolated")
d["speed_rad_s"] = d.speed_rad_s.interpolate()
t, V, y = d.t_s.to_numpy(), d.V.to_numpy(), d.speed_rad_s.to_numpy()
on = t >= 0.1
fo = dg.fopdt_from_step(t[on] - 0.1, y[on], u_amplitude=12.0)
so = idf.fit_second_order_step(t[on] - 0.1, y[on], u_amplitude=12.0)
true = models.dc_motor(R=2.0, L=0.1, Kt=0.05, Ke=0.05, J=0.002, b=0.005)["system"]
print(f"FOPDT fit: K = {fo['K']:.3f} rad/s/V, tau = {fo['tau']:.3f} s, L = {fo['L']:.3f} s    (true dc gain {true.dcgain()[0, 0]:.3f}, poles {np.round(true.poles().real, 2)})")
print(f"second-order fit: K = {so['K']:.3f}, zeta = {so['zeta']:.3f}, wn = {so['wn']:.3f} - a plant with two real poles has been given a complex pair: the noise peak above the final value was read as an overshoot")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(t, y, ".", ms=2, color="0.6", label="measured speed")
tt = t[on] - 0.1
Pf = sy.tf([fo["K"]], [fo["tau"], 1.0])
tf_, yf = rs.step(Pf, tt[-1] - fo["L"], int(np.sum(tt >= fo["L"])))
ax.plot(0.1 + fo["L"] + tf_, 12 * yf, "r", label="FOPDT model"); ax.plot(t, 12 * rs.lsim(true, V / 12.0, t)[1], "k--", lw=1, label="true model"); ax.legend(); ax.set_xlabel("t (s)"); ax.set_ylabel("rad/s"); plt.show()"""),
    md(r"""
A dropped sample is interpolated; the two-point method then returns a gain within a percent of the truth and
a time constant close to the motor's slow pole, with the fast electrical pole absorbed into an apparent delay
of a few milliseconds. A step response gives the gain and the dominant time constant reliably and little
else; the second-order fit does worse than fail - it reads the noise peak above the final value as an
overshoot and returns ζ = 0.72, a complex pole pair for a plant whose poles are real. A fit that returns a
plausible number is not evidence that its model structure is right. For the fast dynamics one needs a richer
excitation.

## PRBS data and ARX least squares
"""),
    code(r"""p = pd.read_csv(datasets.path("motor_prbs.csv"), comment="#")
p = p[p.t_s >= 2.0]                                                   # drop the start-up transient (the record begins at rest)
u = (p.V - p.V.mean()).to_numpy(); yv = (p.speed_rad_s - p.speed_rad_s.mean()).to_numpy()
dt = 0.005
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(p.t_s, p.V, lw=0.8, label="voltage (V)"); ax.plot(p.t_s, p.speed_rad_s, lw=0.8, label="speed (rad/s)"); ax.set_xlim(2, 8); ax.legend(fontsize=8); ax.set_xlabel("t (s)"); plt.show()
rows = []
for dec in (1, 4, 10):
    m = idf.arx(u[::dec], yv[::dec], 2, 2, 1, dt=dt * dec)
    true_z = np.sort(np.exp(true.poles().real * dt * dec))
    rows.append((dt * dec, np.round(np.sort(m["tf"].poles().real), 4).tolist(), np.round(true_z, 4).tolist(), m["tf"].dcgain(), m["fit_percent"], np.max(m["std_errors"][:2])))
print(pd.DataFrame(rows, columns=["dt", "ARX poles", "true poles", "dc gain", "fit %", "std err of a"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
Least squares on the ARX regression is exact for noise-free data and fast to compute, and at the full 200 Hz
it is badly wrong: the two lagged outputs are almost collinear when the poles sit near z = 1, the b
coefficients are tiny, and the tachometer noise - 0.3 rad/s on a 47 rad/s swing - biases the estimate by far
more than its own standard error admits (0.004, against an error of 0.5 in a pole). Decimating to 20 Hz
spreads the poles and the estimate becomes usable. The fit percentage is no guide: it measures one-step
prediction, which any model with the right recent outputs does well.

## The output-error fit, and residual analysis
"""),
    code(r"""oe = idf.output_error(u[::10], yv[::10], 2, dt * 10)
Gc = oe["tf"]
print(f"output-error model at 20 Hz: poles {np.round(np.sort(Gc.poles().real), 4)} (true {np.round(np.sort(np.exp(true.poles().real * 0.05)), 4)}), dc gain {Gc.dcgain():.3f} (true 4.000), simulation fit {oe['fit_percent']:.1f} %")
ls10 = idf.arx(u[::10], yv[::10], 2, 2, 1, dt=0.05)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
lags = np.arange(21)
axes[0].stem(lags, idf.autocorrelation(ls10["residuals"], 20)); axes[0].axhline(2 / np.sqrt(len(ls10["residuals"])), color="r", ls="--"); axes[0].axhline(-2 / np.sqrt(len(ls10["residuals"])), color="r", ls="--"); axes[0].set_title("ARX residual autocorrelation (should be white)")
axes[1].stem(lags, idf.autocorrelation(oe["residuals"], 20)); axes[1].axhline(2 / np.sqrt(len(oe["residuals"])), color="r", ls="--"); axes[1].axhline(-2 / np.sqrt(len(oe["residuals"])), color="r", ls="--"); axes[1].set_title("output-error residual autocorrelation")
plt.show()
t20 = np.arange(len(yv[::10])) * 0.05
_, y_sim, _ = rs.lsim(Gc, u[::10])
plt.plot(t20, yv[::10], ".", ms=2, color="0.6", label="measured"); plt.plot(t20, y_sim, "r", lw=1, label="output-error model simulation"); plt.xlim(0, 6); plt.legend(fontsize=8); plt.xlabel("t (s)"); plt.show()"""),
    md(r"""
The output-error estimator minimises the error of the *simulated* output, so the noisy measurements never
enter a regressor and the estimate is unbiased: both poles and the gain come back within a percent of the
truth. The price is a nonlinear least-squares problem (Levenberg-Marquardt, started from the ARX estimate).
The residual autocorrelations tell the two stories: the ARX residual is coloured, the signature of a noise
model that does not match the data, while the output-error residual is close to white. Whiteness of the
residuals is the test that nothing systematic has been left unmodelled.

## Frequency-domain identification
"""),
    code(r"""s_ = pd.read_csv(datasets.path("servo_frequency_response.csv"), comment="#")
w = s_.w_rad_s.to_numpy(); H = 10 ** (s_.gain_dB.to_numpy() / 20) * np.exp(1j * np.radians(s_.phase_deg.to_numpy()))
fit = idf.fit_frequency_response(w, H, 0, 3)
w_fine = np.logspace(-1, 2.3, 400)
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].semilogx(w, s_.gain_dB, "o", ms=4, label="measured"); axes[0].semilogx(w_fine, 20 * np.log10(np.abs(fr.freqresp(fit, w_fine))), label="3rd-order rational fit"); axes[0].set_ylabel("dB"); axes[0].legend()
axes[1].semilogx(w, s_.phase_deg, "o", ms=4); axes[1].semilogx(w_fine, np.degrees(np.unwrap(np.angle(fr.freqresp(fit, w_fine)))), label="fit"); axes[1].set_ylabel("deg"); axes[1].set_xlabel("w (rad/s)")
plt.show()
print("fitted model:", fit, "\npoles", np.round(fit.poles(), 3))
print("the fit finds the integrator (a pole near 0) but turns the motor's two real poles into a complex pair: the resonance and the delay in the data, absent from the model, pull the fit - the project's problem")"""),
    md(r"""
A measured frequency response (a sine sweep on the servo) can be fitted directly by a rational function:
Levy's linearised least squares, reweighted by the previous denominator (Sanathanan-Koerner) to undo its
high-frequency bias. A third-order fit recovers the integrator but not the motor's two real poles: the data
contain a resonance and a delay that the model structure cannot represent, and the least-squares fit
compensates by placing a complex pole pair where none exists. The lesson is the same as the step's: structure
first, then numbers. The project takes the resonance and the delay into the model explicitly.
"""),
]

NOTEBOOKS["16_model_reduction_and_realisation"] = [
    md(r"""
# 16 · Model reduction and realisation theory

A realisation of a transfer function is a state-space model with that input-output behaviour; the minimal
one has as many states as the system's order, and a *balanced* one orders those states by how much each
contributes to the input-output map, measured by the Hankel singular values. This notebook computes
controllability and observability Gramians, balances a model, reduces it by truncation with the error bound
that comes with it, reconstructs a model from impulse-response data by the eigensystem realisation algorithm
(Ho-Kalman), and shows how the same singular values decide the order of a model fitted to data.
"""),
    setup_cell(["from englincontrol import systems as sy, linalg as la, identify as idf, response as rs, frequency as fr, polynomials as pl, discrete as dc, models"]),
    md("## Gramians: what the input reaches and what the output sees"),
    code(r"""mot = models.dc_motor()["system"]
Wc, Wo = la.gramians(mot.A, mot.B, mot.C)
print("controllability Gramian Wc =\n", np.round(Wc, 6), "\nobservability Gramian Wo =\n", np.round(Wo, 6))
print("eigenvalues of Wc Wo (invariant under similarity):", np.round(np.sort(np.linalg.eigvals(Wc @ Wo).real), 8))
bal, hsv = sy.balanced_realisation(mot)
Wcb, Wob = la.gramians(bal.A, bal.B, bal.C)
print("balanced: both Gramians diagonal and equal to diag(Hankel singular values):", np.round(np.diag(Wcb), 6), np.round(np.diag(Wob), 6), "| HSV", np.round(hsv, 6))"""),
    md(r"""
The controllability Gramian W_c = ∫e^{At}BBᵀe^{Aᵀt}dt measures how much energy an input can put into each
state direction; the observability Gramian W_o measures how much each state direction shows in the output.
Each depends on the coordinates, but the eigenvalues of their product do not: their square roots, the Hankel
singular values, are the system's own numbers. In balanced coordinates both Gramians equal diag(σ₁, σ₂, …),
and a state with a small σ is both hard to excite and hard to see - a candidate for removal.

## Balanced truncation and its error bound
"""),
    code(r"""big = sy.zpk([-3.0, -30.0], [-1.0, -2.0, -15.0, -20.0, -100.0, -120.0], 2e5).to_ss()
bal, hsv = sy.balanced_realisation(big)
print("Hankel singular values of a 6th-order model:", np.round(hsv, 5))
w = np.logspace(-2, 3, 500)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.semilogx(w, 20 * np.log10(np.abs(fr.freqresp(big, w))), "k", lw=2, label="full, order 6")
rows = []
for k in (1, 2, 3, 4):
    red = idf.balanced_truncation(big, k)
    err = np.max(np.abs(fr.freqresp(big, w) - fr.freqresp(red["system"], w)))
    rows.append((k, err, red["error_bound"], err <= red["error_bound"]))
    ax.semilogx(w, 20 * np.log10(np.abs(fr.freqresp(red["system"], w))), label=f"order {k}")
ax.legend(fontsize=8); ax.set_xlabel("w"); ax.set_ylabel("|G| dB"); plt.show()
print(pd.DataFrame(rows, columns=["kept states", "max |G - G_r|", "bound 2 sum(discarded HSV)", "within bound"]).to_string(index=False, float_format="%.2e"))"""),
    md(r"""
Truncating the balanced realisation to its k largest singular values gives a stable reduced model whose
H∞ error is at most twice the sum of the discarded singular values (Glover 1984, Enns 1984) - a guarantee
before any response is computed, which a frequency sweep confirms. Two states capture this sixth-order model
to a few percent across the band; the fast modes at 100 rad/s contribute almost nothing to the input-output
map and are the first to go.

## Realisation from data: Ho-Kalman / ERA
"""),
    code(r"""Gd = dc.c2d(mot, 0.01)
t, ys = rs.step(Gd, 1.0)
markov = np.r_[ys[0], np.diff(ys)]                                # the impulse response = the Markov parameters h_0, h_1, ...
rng = np.random.default_rng(0)
for noise in (0.0, 1e-4):
    e = idf.era(markov + noise * rng.normal(size=len(markov)), 2, 0.01)
    print(f"noise {noise}: ERA poles {np.round(np.sort(e['system'].poles().real), 5)} (true {np.round(np.sort(Gd.poles().real), 5)}); first Hankel singular values {np.round(e['hankel_singular_values'][:5], 5)}")"""),
    md(r"""
From the Markov parameters (the impulse response) the block Hankel matrix has rank equal to the system order,
and its SVD yields a realisation directly: the eigensystem realisation algorithm. Without noise the third
singular value is zero and the poles are exact; with a little noise the singular values beyond the order
become small but nonzero, and reading the order off the gap is the practitioner's judgement. ERA is the
ancestor of the subspace identification methods, and the Hankel matrix is the bridge between realisation
theory and identification.

## Why representations matter numerically
"""),
    code(r"""rows = []
for n in (4, 8, 12):
    poles = -np.arange(1, n + 1, dtype=float)
    G = sy.zpk([], poles, 1.0)
    S_c = G.to_ss()                                                # controllable canonical
    S_b, _ = sy.balanced_realisation(S_c)
    rows.append((n, np.linalg.cond(S_c.A), np.linalg.cond(S_b.A), np.max(np.abs(np.sort(S_c.poles().real) - np.sort(poles))), np.max(np.abs(np.sort(S_b.poles().real) - np.sort(poles)))))
print(pd.DataFrame(rows, columns=["order", "cond(A) canonical", "cond(A) balanced", "pole error canonical", "pole error balanced"]).to_string(index=False, float_format="%.2e"))"""),
    md(r"""
The table does not say what one expects. The companion matrix of a 12th-order polynomial has a condition
number of 10¹⁰ against a few hundred for the balanced realisation - and yet its poles come out *more*
accurately (10⁻⁸ against 10⁻³). Two things are at work. The conditioning of a matrix is not the conditioning
of its eigenvalues, and the QR algorithm (with its own internal balancing) handles companion matrices well.
And the balanced realisation was *computed*, from the companion form through two Gramians and a
transformation containing the inverse fourth roots of Hankel singular values that span ten orders of
magnitude; the transformation is itself ill-conditioned and injects the error. A representation is only as
good as the arithmetic that produced it - notebook 18 pushes the order to where both routes fail. Transfer-function coefficients are a convenient notation and a poor numerical representation; the
library converts to state space for every computation that matters and keeps balanced or modal forms when it
can (notebook 18 continues the theme).
"""),
]

NOTEBOOKS["17_multivariable_basics"] = [
    md(r"""
# 17 · Multivariable basics

With several inputs and outputs the transfer function becomes a matrix, gain becomes direction-dependent, and
loops interact: moving one input disturbs every output. This notebook introduces the tools that make the
transition: the transfer matrix and its singular values, transmission zeros, the relative gain array that
measures interaction and recommends a pairing, static decoupling, and a decentralised PI design on the Wood-Berry
distillation column with and without a decoupler - where SISO intuition ends and the multivariable design
methods begin.
"""),
    setup_cell(["from englincontrol import systems as sy, mimo, frequency as fr, response as rs, design as dg, models"]),
    md("## The transfer matrix and direction-dependent gain"),
    code(r"""col = models.distillation_2x2()
G = col["system"]
TM = mimo.transfer_matrix(G)
for i in range(2):
    print(f"{col['outputs'][i]:<20}:", " | ".join(f"{col['inputs'][j]}: {TM[i][j]}" for j in range(2)))
w = np.logspace(-3, 0, 300)
sv = mimo.sigma(G, w)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.semilogx(w, 20 * np.log10(sv[:, 0]), label="largest singular value"); ax.semilogx(w, 20 * np.log10(sv[:, 1]), label="smallest singular value"); ax.legend(); ax.set_xlabel("w"); ax.set_ylabel("dB"); ax.set_title("the gain of G(jw) depends on the input direction"); plt.show()
print(f"at dc the gains range from {sv[0, 1]:.2f} to {sv[0, 0]:.2f}: condition number {mimo.condition_number(G, 0.0)[0]:.1f}")"""),
    md(r"""
Each entry of the transfer matrix is a SISO transfer function (reflux to top composition, steam to bottom
composition, and the two cross couplings), but the system's gain at a frequency is not a number: it lies
between the smallest and largest singular values of G(jω), depending on the direction of the input vector. A
condition number of 7.5 at dc means some input combinations move the outputs seven times more than others -
the column is moderately ill-conditioned, a fact that no single-loop analysis reveals.

## Interaction: the relative gain array
"""),
    code(r"""G0 = G.dcgain()
print("dc gain matrix:\n", np.round(G0, 2))
pair = mimo.rga_pairing(G0)
print("RGA:\n", np.round(pair["rga"], 3), "\nrecommended pairing (output, input):", pair["pairs"], "| any negative relative gains:", pair["negative_entries"])
D = mimo.static_decoupler(G0)
print("static decoupler D = G(0)^-1:\n", np.round(D, 4), "\nG(0) D =\n", np.round(G0 @ D, 10))
print("transmission zeros of the column (none in the right half-plane):", np.round(mimo.transmission_zeros(G), 4))"""),
    md(r"""
The relative gain λ_ij compares the gain from input j to output i with all other loops open and with them
closed; a value of 1 means the other loops do not interfere, 0.5 means they halve it, a negative value means
the sign reverses when the other loops close - a loop that would run away. The Wood-Berry column's diagonal
relative gains of 2.0 say the two loops fight each other (each loop sees half its gain once the other is
closed), and the pairing rule still picks the diagonal. A static decoupler inverts the dc gain so that each
input affects one output at steady state; dynamic interaction remains.

## Decentralised PI control, with and without a decoupler
"""),
    code(r"""def closed_loop_2x2(plant, controllers, decoupler=None):
    # u = D v, v_i = C_i (r_i - y_i): build the loop in state space
    C1, C2 = controllers
    Cs = sy.ss(np.block([[C1.to_ss().A, np.zeros((1, 1))], [np.zeros((1, 1)), C2.to_ss().A]]), np.block([[C1.to_ss().B, np.zeros((1, 1))], [np.zeros((1, 1)), C2.to_ss().B]]),
               np.block([[C1.to_ss().C, np.zeros((1, 1))], [np.zeros((1, 1)), C2.to_ss().C]]), np.block([[C1.to_ss().D, np.zeros((1, 1))], [np.zeros((1, 1)), C2.to_ss().D]]))
    if decoupler is not None:
        Cs = sy.ss(Cs.A, Cs.B, decoupler @ Cs.C, decoupler @ Cs.D)
    return sy.feedback(sy.series(Cs, plant))
# PI loops tuned by SIMC (lambda = 10 min) on the diagonal elements; with the decoupler the controllers see G D, whose
# dc gains are +1 on both channels, so they are retuned for unit gain (the raw steam loop's negative gain would become positive feedback)
C1 = dg.pid_standard(16.7 / (12.8 * 10.0), min(16.7, 40.0))                 # top composition from reflux
C2 = dg.pid_standard(-14.4 / (19.4 * 10.0), min(14.4, 40.0))                # bottom composition from steam (negative gain)
C1d = dg.pid_standard(16.7 / 10.0, min(16.7, 40.0))                          # the same rule on the decoupled channels
C2d = dg.pid_standard(14.4 / 10.0, min(14.4, 40.0))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (name, ctrls, dec) in zip(axes, (("decentralised PI", (C1, C2), None), ("PI + static decoupler", (C1d, C2d), D))):
    T = closed_loop_2x2(G, ctrls, dec)
    t, y = rs.step(T, 120.0, input_index=0)
    ax.plot(t, y[:, 0], label="top composition (setpoint step)"); ax.plot(t, y[:, 1], label="bottom composition (should stay 0)")
    ax.set_title(f"{name}: poles stable {np.max(T.poles().real) < 0}"); ax.legend(fontsize=8); ax.set_xlabel("t (min)")
    print(f"{name:<22}: closed-loop poles {np.round(np.sort_complex(T.poles())[:4], 3)} ... ; peak disturbance of the other output {np.max(np.abs(y[:, 1])):.3f}")
plt.show()"""),
    md(r"""
Two PI loops tuned on the diagonal elements as if they were alone are stable together here, but a setpoint
step on the top composition disturbs the bottom one by a quarter of the step - the interaction the RGA warned
of. The static decoupler removes the steady-state coupling and cuts the cross-disturbance to a third, with
faster loops - once the controllers are retuned for the plant they now see, G·D with unit positive gains (the
steam loop's original negative gain would otherwise become positive feedback and a slowly unstable loop, a
trap this notebook's first draft fell into). What remains is dynamic interaction (the cross terms have
different time constants), which only a dynamic decoupler or a full multivariable design (LQG, H∞) removes. The lesson of the chapter: MIMO plants need MIMO
analysis before any loop is closed, and the RGA is the first thing to compute.
"""),
]

NOTEBOOKS["18_numerics_of_control"] = [
    md(r"""
# 18 · The numerics of control

Control computations are matrix computations, and the textbook formulas are often not the way to do them.
This notebook opens the library's algorithms: the matrix exponential by scaling and squaring with a Padé
approximant (and why the Taylor series fails), the Lyapunov equation by the real Schur form, the Riccati
equation by the Hamiltonian's invariant subspace with Newton refinement, the conditioning of polynomial
coefficients against state-space representations, and the H∞ norm by bisection on a Hamiltonian. Each is
checked against SciPy, and each has a case where the naive method breaks.
"""),
    setup_cell(["from scipy import linalg as sla", "from englincontrol import linalg as la, systems as sy, polynomials as pl, frequency as fr, statespace as sp, models"]),
    md("## The matrix exponential: Taylor fails, Padé with scaling succeeds"),
    code(r"""def expm_taylor(A, terms=30):
    E = np.eye(len(A)); term = np.eye(len(A))
    for k in range(1, terms):
        term = term @ A / k; E = E + term
    return E
for scale in (1.0, 10.0, 30.0):
    A = scale * np.array([[-1.0, 20.0], [0.0, -2.0]])            # a non-normal matrix: large transient growth
    exact = sla.expm(A)
    print(f"|A| = {np.linalg.norm(A, 1):5.0f}: Taylor (30 terms) relative error {np.max(np.abs(expm_taylor(A) - exact)) / np.max(np.abs(exact)):.1e}, Pade 13 with scaling-and-squaring {np.max(np.abs(la.expm(A) - exact)) / np.max(np.abs(exact)):.1e}")
A = 30.0 * np.array([[-1.0, 20.0], [0.0, -2.0]])
print(f"\nfor |A| = 660 the library scales by 2^-{int(np.ceil(np.log2(np.linalg.norm(A, 1) / 5.37)))} before the Pade step and squares the result that many times")"""),
    md(r"""
The Taylor series of e^{A} converges for every A, in exact arithmetic; in floating point its terms grow to
10²⁰ before cancelling to a result of size 1, and the rounding of those terms destroys the answer. Higham's
algorithm scales A by a power of two until its norm is below 5.37, applies the [13/13] Padé approximant (exact
to double precision at that norm), and squares the result back - the method SciPy uses, and the library
reproduces it to the last digit.

## Lyapunov by the Schur form
"""),
    code(r"""rng = np.random.default_rng(0)
A = rng.normal(size=(12, 12)) - 4 * np.eye(12)
T, U = la.schur(A)
print(f"real Schur form: |U T U^T - A| = {np.max(np.abs(U @ T @ U.T - A)):.1e}, |U^T U - I| = {np.max(np.abs(U.T @ U - np.eye(12))):.1e}; T has {int(np.sum(np.abs(np.diag(T, -1)) > 1e-12))} 2 x 2 blocks (complex pairs)")
Q = np.eye(12)
P = la.lyapunov(A, Q)
print(f"Lyapunov A P + P A' + Q = 0: residual {np.linalg.norm(A @ P + P @ A.T + Q):.1e}; vs SciPy {np.max(np.abs(P - sla.solve_continuous_lyapunov(A, -Q))):.1e}")
n = 12
K = np.kron(np.eye(n), A) + np.kron(A, np.eye(n))
print(f"the Kronecker alternative solves a {n * n} x {n * n} system ({K.size * 8 / 1e6:.1f} MB, O(n^6) flops); Bartels-Stewart on the Schur form costs O(n^3)")"""),
    md(r"""
The Lyapunov equation is linear in P and can be written as an n²×n² system through the Kronecker product -
fine for n = 4, hopeless for n = 100. Bartels and Stewart reduce A to real Schur form (quasi-triangular, by
the QR algorithm) and then solve for P block by block by back-substitution, in O(n³). The library's Schur
decomposition is a shifted QR iteration with deflation written in NumPy; it agrees with LAPACK's to rounding.

## Riccati: the Hamiltonian's invariant subspace, then Newton
"""),
    code(r"""cp = models.cart_pendulum()["system"]
A, B = cp.A, cp.B
Q, R = np.diag([10.0, 1.0, 100.0, 1.0]), np.array([[0.1]])
S = B @ np.linalg.solve(R, B.T)
H = np.block([[A, -S], [-Q, -A.T]])
lam, V = np.linalg.eig(H)
print("Hamiltonian eigenvalues come in +- pairs:", np.round(np.sort_complex(lam), 3))
stable = V[:, lam.real < 0]
P_sub = np.real(np.linalg.solve(stable[:4].T, stable[4:].T).T)
resid = lambda P_: np.linalg.norm(A.T @ P_ + P_ @ A - P_ @ S @ P_ + Q)   # noqa: E731
P_lib = la.care(A, B, Q, R)
print(f"P from the stable subspace alone: residual {resid(P_sub):.1e}; after Newton-Kleinman refinement (library): {resid(P_lib):.1e}; vs SciPy: {np.max(np.abs(P_lib - sla.solve_continuous_are(A, B, Q, R))):.1e}")
Ad, Bd = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([[0.005], [0.1]])
Pd = la.dare(Ad, Bd, np.eye(2), np.array([[0.1]]))
print(f"discrete Riccati by the structure-preserving doubling algorithm: vs SciPy {np.max(np.abs(Pd - sla.solve_discrete_are(Ad, Bd, np.eye(2), np.array([[0.1]])))):.1e}")"""),
    md(r"""
The Hamiltonian matrix of the LQR problem has eigenvalues in ± pairs; the stabilising Riccati solution is
P = X₂X₁⁻¹ where [X₁; X₂] spans the stable half of its invariant subspace. Computing that subspace from
eigenvectors is accurate to the conditioning of the eigenvectors (here already 10⁻¹² in the residual, so the
Newton steps - each a Lyapunov equation - have little left to polish; for badly separated eigenvalues they
gain several digits, as the exercises show). (Production codes use the Schur form of
the Hamiltonian instead of eigenvectors, for the same reason as the Lyapunov solver.) The discrete equation is
solved by the doubling algorithm, which needs no inverse of A.

## Conditioning: coefficients versus states
"""),
    code(r"""rows = []
for n in (6, 10, 14, 18):
    poles = -np.arange(1, n + 1, dtype=float)
    c = pl.from_roots(poles)
    roots_from_coeffs = np.sort(pl.roots(c).real)
    G = sy.zpk([], poles, 1.0)
    bal, _ = sy.balanced_realisation(G.to_ss())
    rows.append((n, c.max(), np.max(np.abs(roots_from_coeffs - np.sort(poles))), np.max(np.abs(np.sort(bal.poles().real) - np.sort(poles))), np.max(np.abs(np.sort(G.to_ss().poles().real) - np.sort(poles)))))
print(pd.DataFrame(rows, columns=["order", "largest coefficient", "pole error from coefficients", "from the balanced ss", "from the companion ss"]).to_string(index=False, float_format="%.1e"))"""),
    md(r"""
Roots at −1, …, −18 give coefficients up to 10¹⁶; recovering the roots from them (the companion matrix's
eigenvalues - which is also what the companion-form state-space model carries) loses digits steadily, to
10⁻³ at order 18, and the balanced realisation computed from it through Gramians loses far more (its
transformation is ill-conditioned, as notebook 16 found). The accurate representation of such a system is
the modal one, built directly from the known poles, never from the coefficients. The practical rules: never
form a high-order transfer function as a coefficient list if the poles are known; build state-space models
from physics or from modal data; and when a transformation is needed, check its condition number before
trusting what comes out of it.

## The H-infinity norm by bisection
"""),
    code(r"""G = models.quarter_car()["system"]
norm, w_pk = fr.hinf_norm(G)
w = np.logspace(-1, 3, 50000)
sweep = np.max([np.linalg.norm(fr.freqresp(G, wk), 2) for wk in w[::10]])
print(f"H-inf norm of the quarter car (2 x 2) by Boyd-Balakrishnan bisection: {norm:.4f} (peak near {w_pk:.1f} rad/s); a 5000-point sweep gives {sweep:.4f}")"""),
    md(r"""
The peak gain over frequency is found without a sweep: γ exceeds ‖G‖∞ exactly when a Hamiltonian matrix built
from (A, B, C, D, γ) has no eigenvalue on the imaginary axis, so a bisection on γ with one eigenvalue
computation per step converges to the norm to any precision - the engine inside H∞ synthesis. A sweep can
miss a sharp peak between grid points; the bisection cannot.
"""),
]

NOTEBOOKS["19_project_a_position_servo_from_data"] = [
    md(r"""
# 19 · Project: a position servo from measured data to a digital controller

**Problem.** A position servo - the course's DC motor, a gearbox and a position sensor - is to be controlled
digitally. Three measurements exist: the motor's speed step response, a PRBS experiment, and a sine-sweep
frequency response of the whole servo, which shows something the motor model does not contain. The task:
identify the motor; find what else the servo contains and bound it as model uncertainty; design a controller
that meets a 2 % settling time of 0.5 s on a 10 rad step (about a turn and a half) without exceeding the 12 V
supply, with a phase margin of 45° and robust stability against the unmodelled part; implement it at a
sampling rate justified by the analysis; verify by simulation with the saturation; and report.
"""),
    setup_cell(["from scipy import optimize as sopt", "from englincontrol import systems as sy, identify as idf, design as dg, discrete as dc, frequency as fr, response as rs, robust as rb, stability as st, polynomials as pl, datasets, models"]),
    md("## 1. The motor from the PRBS data"),
    code(r"""p = pd.read_csv(datasets.path("motor_prbs.csv"), comment="#")
p = p[p.t_s >= 2.0]
u = (p.V - p.V.mean()).to_numpy(); yv = (p.speed_rad_s - p.speed_rad_s.mean()).to_numpy()
oe = idf.output_error(u[::10], yv[::10], 2, 0.05)
Gm_d = oe["tf"]
Gm = dc.d2c(Gm_d.to_ss())                                            # continuous motor model: voltage -> speed
Gm_tf = Gm.to_tf()
print("identified motor (voltage -> speed):", Gm_tf, "\n  poles", np.round(Gm_tf.poles().real, 3), " dc gain", round(Gm_tf.dcgain(), 3), "rad/s per V")
true = models.dc_motor(R=2.0, L=0.1, Kt=0.05, Ke=0.05, J=0.002, b=0.005)["system"].to_tf()
print("  (the plant that generated the data:", true, ")")"""),
    md("## 2. The servo's frequency response: what the motor model misses"),
    code(r"""s_ = pd.read_csv(datasets.path("servo_frequency_response.csv"), comment="#")
w = s_.w_rad_s.to_numpy(); H = 10 ** (s_.gain_dB.to_numpy() / 20) * np.exp(1j * np.radians(s_.phase_deg.to_numpy()))
P0 = sy.series(Gm_tf, sy.tf([1.0], [1.0, 0.0]))                       # motor speed -> position: an integrator
gear = np.median(np.abs(H[:10]) / np.abs(fr.freqresp(P0, w[:10])))    # overall gain (gearbox, sensor) from the low-frequency data
P0 = sy.tf(gear * P0.num, P0.den)
H0 = fr.freqresp(P0, w)
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].semilogx(w, s_.gain_dB, "o", ms=4, label="measured"); axes[0].semilogx(w, 20 * np.log10(np.abs(H0)), label="motor x integrator x gain"); axes[0].legend(); axes[0].set_ylabel("dB")
axes[1].semilogx(w, s_.phase_deg, "o", ms=4); axes[1].semilogx(w, np.degrees(np.unwrap(np.angle(H0)))); axes[1].set_ylabel("deg"); axes[1].set_xlabel("w (rad/s)")
plt.show()
print(f"overall gain {gear:.1f}; the measured/model ratio reaches {np.max(np.abs(H / H0)):.1f} near {w[np.argmax(np.abs(H / H0))]:.0f} rad/s (a resonance) and the phase falls below the model's by an amount growing with frequency (a delay)")
def mismatch(theta):                                                  # fit a resonance and a delay jointly to the ratio data/model
    wr, zr, L = theta
    model = fr.freqresp(sy.series(P0, sy.tf([wr**2], [1.0, 2 * zr * wr, wr**2])), w, delay=L)
    return np.r_[np.log(np.abs(H / model)), np.angle(H / model)]
fit = sopt.least_squares(mismatch, [60.0, 0.1, 0.002], bounds=([20, 0.01, 0], [200, 1.0, 0.05]))
wr, zr, delay = fit.x
P_nom = sy.series(P0, sy.tf([wr**2], [1.0, 2 * zr * wr, wr**2]))     # the nominal model: motor, integrator, gain, resonance (the delay kept separately)
resid = H / fr.freqresp(P_nom, w, delay=delay)
print(f"joint fit: resonance wr = {wr:.1f} rad/s, zeta = {zr:.3f}, delay = {delay * 1000:.1f} ms; residual gain error {np.max(np.abs(20 * np.log10(np.abs(resid)))):.2f} dB, phase error {np.max(np.abs(np.degrees(np.angle(resid)))):.1f} deg")"""),
    md(r"""
The motor model with an integrator and an overall gain of 50 follows the measured servo response up to
about 30 rad/s; beyond it two things appear that the motor tests could not show: a lightly damped resonance
at 80 rad/s (a structural mode of the gearbox and coupling) and a phase lag growing with frequency, a
transport delay of a few milliseconds from the sensor and the electronics. (Estimating the delay from the
phase below the resonance alone gives 5 ms; fitting resonance and delay together gives 4 ms, still a little
above the 3 ms the data were generated with, because the sweep's 1° phase scatter is of the same order as
the delay's phase at low frequency.) With both in the model the residual is a third of a decibel and 3°.

## 3. Uncertainty bound and the limits it imposes
"""),
    code(r"""W_data = np.abs(resid - 1)                                           # multiplicative error of the nominal model (with delay) against the data
W = sy.tf([0.5 / 50.0, 0.07], [1.0 / 400.0, 1.0])                   # a first-order envelope above it
wg = np.logspace(-1, 3, 2000)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.loglog(w, W_data, "o", ms=4, label="|P_measured / P_nominal - 1|"); ax.loglog(wg, np.abs(fr.freqresp(W, wg)), label="uncertainty weight W"); ax.axhline(1, color="k", lw=0.5); ax.legend(); ax.set_xlabel("w (rad/s)"); plt.show()
print(f"the weight lies above the data at every measured frequency: {bool(np.all(np.abs(fr.freqresp(W, w)) >= W_data))}; it crosses 1 at {wg[np.argmax(np.abs(fr.freqresp(W, wg)) > 1)]:.0f} rad/s")
lim = rb.rhp_limits(P_nom, delay=delay)
print(f"limits: the delay allows a crossover up to ~{lim['w_c_max_from_delay']:.0f} rad/s; the resonance (gain peak {1 / (2 * zr):.0f}x at {wr:.0f} rad/s) requires |L(j{wr:.0f})| << 1, i.e. a crossover well below {wr / 4:.0f} rad/s; the 0.5 s settling needs a dominant pole near sigma = 4/0.5 = 8 1/s")"""),
    md(r"""
With the resonance in the nominal model the remaining uncertainty is small - a few percent at low frequency,
rising above one beyond 100 rad/s where the model has not been measured - so robust stability (|WT| < 1,
notebook 14) will be easy to satisfy and the real limit is the resonance itself: the loop gain at 80 rad/s
must stay well below one, which confines the crossover to about 15 rad/s. The settling-time specification
needs closed-loop poles with σ ≈ 8 s⁻¹ - compatible, but with little room.

## 4. A classical design: lead compensation
"""),
    code(r"""def simulate_digital(controller_ss, dt, T=1.5, step=10.0, u_lim=12.0, observer=None, K=None, Nbar=None):
    # the plant (nominal model with delay) at 1 kHz, the controller discretised at dt: either an output-feedback
    # controller (controller_ss from the error e to u) or observer-based state feedback (u = -K xhat + Nbar r)
    Pz = dc.c2d(P_nom.to_ss(), 0.001)
    nd = int(round(delay / 0.001)); N = int(T / 0.001); sub = int(round(dt / 0.001))
    xp = np.zeros(Pz.n); uh = np.zeros(N + nd + 1); yh = np.zeros(N); ur = np.zeros(N)
    xc = np.zeros(controller_ss.n) if controller_ss is not None else None
    xo = np.zeros(observer.n) if observer is not None else None
    u = 0.0
    for k in range(N):
        y = (Pz.C @ xp)[0]; yh[k] = y
        if k % sub == 0:                                               # the controller runs at its own rate
            if observer is None:
                e = step - y
                u = float(np.clip((controller_ss.C @ xc + controller_ss.D @ [e])[0], -u_lim, u_lim))
                xc = controller_ss.A @ xc + controller_ss.B @ [e]
            else:
                u = float(np.clip((-K @ xo)[0] + Nbar * step, -u_lim, u_lim))
                xo = observer.A @ xo + observer.B @ [u, y]
        ur[k] = u; uh[k + nd] = u
        xp = Pz.A @ xp + Pz.B @ [uh[k]]
    t_ = np.arange(N) * 0.001
    return t_, yh, ur, rs.step_info(t_, yh, final=step)
rows = []
for wc, pm in ((12.0, 50.0), (15.0, 55.0), (20.0, 60.0)):
    d = dg.design_lead(P_nom, w_c=wc, pm_target_deg=pm + np.degrees(wc * delay))   # ask for the delay's phase in advance
    L = sy.series(d["controller"], P_nom)
    m = fr.margins(L, delay=delay)
    rob = rb.multiplicative_robustness(L, W)
    t_, yh, ur, info = simulate_digital(dc.c2d(d["controller"], 0.001, "tustin").to_ss(), 0.001)
    rows.append((wc, d["n_leads"], m["w_gc"], m["phase_margin"], m["gain_margin"], rb.sensitivity_peak(L)["Ms"], rob["peak_WT"], info["overshoot"], info["settling_time"], np.max(np.abs(ur))))
print(pd.DataFrame(rows, columns=["target w_c", "leads", "w_gc", "PM (deg)", "GM", "Ms", "|WT| peak", "overshoot %", "settling (s)", "peak V"]).to_string(index=False, float_format="%.2f"))"""),
    md(r"""
A lead design on the nominal model (asking for the delay's phase in advance) gives robust loops - M_s below
1.7, the resonance a factor of five below unity gain, |WT| far below one - and a 10 rad step that stays within
the 12 V supply. But none meets the 0.5 s settling: the two cascaded leads leave a slow closed-loop pole near
their zeros, and the response creeps to its final value for 0.6-0.7 s. Pushing the crossover does not help;
the structure does not fit the specification.

## 5. Observer-based state feedback on the identified model
"""),
    code(r"""from englincontrol import statespace as sp
Pss = P_nom.to_ss()                                                  # 5 states: motor (2), integrator, resonance (2)
A, B, C = Pss.A, Pss.B, Pss.C
r_lqr = sp.lqr(A, B, 10.0 * C.T @ C, np.array([[1.0]]))
K = r_lqr["K"]; Nbar = sp.reference_gain(A, B, C, K)
print("LQR on the position output: closed-loop poles", np.round(np.sort_complex(r_lqr["closed_loop_poles"]), 1))
rows = []
for scale in (1.0, 2.0, 4.0, 8.0):
    L_obs = sp.observer_gain(A, C, -scale * np.array([15.0, 16.0, 17.0, 18.0, 19.0]))
    comp = sy.ss(A - B @ K - L_obs @ C, L_obs, K, np.zeros((1, 1)))      # the compensator from y to -u
    L_in = sy.series(P_nom, comp.to_tf())                                 # the loop at the plant input
    dist = rb.nyquist_distance(L_in); m = fr.margins(L_in, delay=delay); rob = rb.multiplicative_robustness(L_in, W)
    obs_z = dc.c2d(sy.ss(A - L_obs @ C, np.hstack([B, L_obs]), np.eye(5), np.zeros((5, 2))), 0.001)
    t_, yh, ur, info = simulate_digital(None, 0.001, observer=obs_z, K=K, Nbar=Nbar)
    rows.append((f"-{15 * scale:.0f} .. -{19 * scale:.0f}", dist["distance"], 1 / dist["distance"], m["phase_margin"], m["gain_margin"], rob["peak_WT"], info["overshoot"], info["settling_time"], np.max(np.abs(ur))))
print(pd.DataFrame(rows, columns=["observer poles", "min |1 + L|", "Ms", "PM (deg)", "GM", "|WT| peak", "overshoot %", "settling (s)", "peak V"]).to_string(index=False, float_format="%.2f"))"""),
    md(r"""
State feedback designed by LQR on the identified five-state model, with the state reconstructed by an
observer from the position measurement, meets the settling time easily - and the observer's speed decides
the robustness, the other way round from the textbook rule: the *slowest* observer, with poles near the
regulator's, gives the largest distance from −1 (M_s = 1.56, 51° of phase margin), while observers three to
eight times faster push the loop gain up near the resonance and cut the margin to 20-30°. The rule "observer
poles 3-5 times faster than the controller's" assumes the plant is well modelled at those frequencies; here
it is not, and the data decide. The slow-observer design meets every specification: 0.27 s settling, 51°,
|WT| = 0.15, 12 V peak.

## 6. Digital implementation
"""),
    code(r"""L_obs = sp.observer_gain(A, C, -np.array([15.0, 16.0, 17.0, 18.0, 19.0]))
comp = sy.ss(A - B @ K - L_obs @ C, L_obs, K, np.zeros((1, 1)))
L_in = sy.series(P_nom, comp.to_tf())
T_cl = sy.feedback(L_in)
wb = fr.bandwidth(T_cl)
rules = dc.sample_rate_rules(wb)
print(f"closed-loop bandwidth {wb:.1f} rad/s -> the rules suggest dt between {rules['dt_min'] * 1000:.0f} and {rules['dt_max'] * 1000:.0f} ms; the resonance at {wr:.0f} rad/s must stay below the Nyquist frequency with margin: dt < {np.pi / wr / 2 * 1000:.0f} ms")
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for dt in (0.001, 0.005, 0.01):
    obs_z = dc.c2d(sy.ss(A - L_obs @ C, np.hstack([B, L_obs]), np.eye(5), np.zeros((5, 2))), dt)
    t_, yh, ur, info = simulate_digital(None, dt, observer=obs_z, K=K, Nbar=Nbar)
    axes[0].plot(t_, yh, label=f"dt = {dt * 1000:.0f} ms: settling {info['settling_time']:.2f} s, overshoot {info['overshoot']:.0f} %"); axes[1].plot(t_, ur, lw=0.8)
    print(f"dt = {dt * 1000:4.0f} ms: 10 rad step: overshoot {info['overshoot']:.1f} %, 2 % settling {info['settling_time']:.3f} s, peak {np.max(np.abs(ur)):.1f} V, saturated for {np.sum(np.abs(ur) >= 12.0):.0f} ms")
axes[0].axhline(10, color="k", lw=0.5); axes[0].axhline(10.2, color="0.7", ls="--"); axes[0].axhline(9.8, color="0.7", ls="--"); axes[0].legend(fontsize=8); axes[0].set_ylabel("position (rad)"); axes[1].axhline(12, color="r", ls="--"); axes[1].axhline(-12, color="r", ls="--"); axes[1].set_ylabel("voltage (V)"); axes[1].set_xlabel("t (s)")
plt.show()
dt = 0.005
info_final = simulate_digital(None, dt, observer=dc.c2d(sy.ss(A - L_obs @ C, np.hstack([B, L_obs]), np.eye(5), np.zeros((5, 2))), dt), K=K, Nbar=Nbar)[3]
m_final = fr.margins(L_in, delay=delay + dt / 2 + dt)                 # the hold's half sample and one sample of computation delay"""),
    md(r"""
The observer and gain are discretised (ZOH on the observer, the gain unchanged) and run at 1, 5 and 10 ms
against the plant with its delay and the 12 V limit. At 5 ms (ω_bΔt ≈ 0.1, and the 80 rad/s resonance well
inside the Nyquist band) the response is indistinguishable from the 1 ms one and meets the specification;
at 10 ms the hold's delay begins to show. 200 Hz it is, with an anti-aliasing filter above 150 rad/s, the
measurement read and the voltage written at the start of each period.

## 7. The report
"""),
    code(r"""report = f'''# Position servo: identification, design and digital implementation

Data: motor PRBS test (output-error fit at 20 Hz), servo sine sweep (60 frequencies).
Model: motor {Gm_tf}, servo gain {gear:.1f}, structural resonance {wr:.0f} rad/s (zeta {zr:.2f}), delay {delay * 1000:.1f} ms;
       residual uncertainty bounded by W = {W} (crosses 1 at ~100 rad/s).
Limits: the resonance confines the crossover to ~15 rad/s (the delay alone would allow {1 / delay:.0f}); the 0.5 s settling needs sigma ~ 8 1/s.
Classical design: cascaded leads at 12-20 rad/s are robust (Ms < 1.7) but settle in 0.6-0.7 s: a slow closed-loop pole from the lead pair.
Chosen design: LQR state feedback (q = 10 on the position) on the identified 5-state model with a Luenberger observer at poles -15 .. -19 -
       the slowest tried, and the most robust: min |1 + L| = {rb.nyquist_distance(L_in)['distance']:.2f} (Ms {rb.sensitivity_peak(L_in)['Ms']:.2f}), phase margin {m_final['phase_margin']:.0f} deg with hold and computation delay,
       |W T| peak {rb.multiplicative_robustness(L_in, W)['peak_WT']:.2f} < 1. Faster observers (x2 to x8) cut the margin to 20-30 deg: the resonance punishes high-frequency loop gain.
Implementation: 200 Hz (w_b dt = {wb * dt:.2f}), observer discretised by ZOH; 10 rad step: overshoot {info_final['overshoot']:.0f} %, settling {info_final['settling_time']:.2f} s (spec 0.5), peak 12 V (brief saturation).
Caveats: linear friction and a single resonance assumed; the uncertainty weight comes from one sweep; the model above 100 rad/s is uncertain and above 200 rad/s unmeasured, hence the anti-aliasing filter.
'''
print(report)"""),
    md(r"""
The report follows the course's logic: identify what the data support, model what they show and bound what
they do not, let the limits set the bandwidth, try the classical structure and recognise when it does not
fit, design in state space on the identified model and check the loop that is actually closed, discretise at
a rate the analysis justifies, and verify on the full model with the real actuator limit. Its central finding
- that the observer should be slow here - came from the data and the robustness check, not from a rule; every
number in it can be traced to a cell above, which is what a design document for a real servo should allow.
"""),
]

# =====================================================================================================
# Teaching layer
def _x(objectives, prereq, time, inside, exercises=(), implement=""):
    return dict(
        objectives=objectives, prereq=prereq, time=time, inside=inside, exercises=list(exercises), implement=implement
    )



EXTRAS["00_models_of_linear_systems"] = _x(
    ["write state-space models from physical laws and convert them to transfer functions and back", "recognise canonical, modal and minimal realisations",
     "read poles, zeros and dc gain and anticipate the response", "assemble series, parallel and feedback connections in both representations"],
    "linear ODEs, the Laplace transform, basic linear algebra", "60 min",
    [md(r"""
## Inside the algorithm: the transfer function of (A, B, C, D) by Leverrier-Faddeev
"""), code(r"""A, B, C = G.A, G.B, G.C
n = len(A); N = np.eye(n); den = [1.0]; num = []
for k in range(n):
    num.append((C @ N @ B)[0, 0])
    ck = -np.trace(A @ N) / (k + 1)
    den.append(ck); N = A @ N + ck * np.eye(n)
print("den by the recursion:", np.round(den, 6), "  library:", np.round(Gtf.den, 6))
print("num by the recursion:", np.round(num, 6), "  library:", np.round(Gtf.num, 6))""")],
    ["Derive the state-space model of the two-tank system from mass balances (A h' = q_in - q_out with q = h/R across each valve) and confirm `models.two_tanks()`'s matrices and dc gain R2.",
     "Realise (s + 3)/((s + 1)(s + 2)(s + 3)) in controllable canonical form; is it minimal? Use `systems.minimal` and explain which mode is lost and why it is harmless here."],
    "**Implement it yourself:** write the series connection of two state-space systems as block matrices (the output of the first feeds the second) and check your result against `systems.series` on the DC motor followed by a first-order sensor.")

EXTRAS["01_time_response"] = _x(
    ["compute exact responses with the matrix exponential", "read rise time, overshoot and settling time and derive them for the second-order prototype",
     "apply the dominant-pole approximation and anticipate the effect of zeros", "relate steady-state error to system type"],
    "notebook 00", "60 min",
    [md(r"""
## Inside the algorithm: the step response of the prototype in closed form
"""), code(r"""zeta, wn = 0.3, 2.0
wd = wn * np.sqrt(1 - zeta**2)
tt = np.linspace(0, 10, 1001)
y_formula = 1 - np.exp(-zeta * wn * tt) * (np.cos(wd * tt) + zeta / np.sqrt(1 - zeta**2) * np.sin(wd * tt))
_, y_lib = rs.step(sy.tf([4.0], [1.0, 1.2, 4.0]), 10.0, 1001)
print("closed-form step response vs the matrix-exponential computation:", f"{np.max(np.abs(y_formula - y_lib)):.1e}")""")],
    ["For the DC motor, find the voltage step response's rise time and settling time and relate them to the two time constants. Then double the inertia J and repeat: which time constant moved?",
     "A specification asks for overshoot < 10 % and 2 % settling < 1 s. Draw the allowed region of the s-plane (zeta and sigma bounds) and check it with three pole pairs of your choice by simulation."],
    "**Implement it yourself:** compute the impulse response of the prototype from the initial-condition response with x0 = B (no input) and compare with `response.impulse`; explain why they are the same.")

EXTRAS["02_frequency_response"] = _x(
    ["establish the sinusoidal steady state and compute G(jw)", "sketch Bode plots from asymptotes and read corners, slopes and phases",
     "interpret Nyquist and Nichols plots", "compute gain, phase and delay margins, bandwidth and resonant peak"],
    "notebooks 00 and 01", "75 min",
    [md(r"""
## Inside the algorithm: finding a gain crossover by bisection
"""), code(r"""f = lambda w_: np.log10(abs(fr.freqresp(L, w_)))     # noqa: E731
a, b = 0.1, 100.0
for _ in range(60):
    m_ = np.sqrt(a * b)
    if f(a) * f(m_) <= 0: b = m_
    else: a = m_
print(f"gain crossover by bisection: {a:.6f} rad/s; library: {fr.margins(L)['w_gc']:.6f}")""")],
    ["Add a pure delay of 0.1 s to L (`fr.freqresp(L, w, delay=0.1)` and `fr.margins(L, delay=0.1)`): how do the margins change, and does the gain margin or the phase margin react first? Compare with the delay margin computed before.",
     "Sketch by hand the Bode asymptotes of the DC motor's voltage-to-speed transfer function and check them against `fr.bode`. Where is its bandwidth, and what sets it?"],
    "**Implement it yourself:** compute the Nyquist curve of L from the state-space model with C (jwI - A)^-1 B + D (one linear solve per frequency) and compare with the transfer-function evaluation; which is better conditioned for a 10th-order system?")

EXTRAS["03_stability_tests"] = _x(
    ["build the Routh array including its special cases and count unstable roots", "use Hurwitz determinants and Lyapunov's equation",
     "derive stability conditions on parameters", "apply Jury's test to discrete systems and recognise the conditioning problem of polynomial coefficients"],
    "notebook 00", "60 min",
    [md(r"""
## Inside the algorithm: one Routh row by hand
"""), code(r"""c = [1, 2, 3, 4]
r1, r2 = np.array([c[0], c[2]], float), np.array([c[1], c[3]], float)
r3 = np.array([(r2[0] * r1[1] - r1[0] * r2[1]) / r2[0], 0.0])
r4 = np.array([(r3[0] * r2[1] - r2[0] * r3[1]) / r3[0], 0.0])
print("rows by hand:", r1, r2, r3, r4, "\nlibrary table:\n", st.routh(c)["table"])""")],
    ["The loop 1 + K (s + a)/(s^2 (s + 10)) = 0: use the Routh array symbolically (write the conditions on K and a) and confirm numerically for a = 1 and a = 5 the range of K for stability.",
     "For the levitated ball (`models.magnetic_levitation`) with proportional feedback u = -K x, show with the Routh array that no K stabilises it, and that u = -K x - Kd x' does for suitable Kd. Confirm with the eigenvalues."],
    "**Implement it yourself:** solve the Lyapunov equation A^T P + P A = -Q for a 3 x 3 A by the Kronecker product (vec form) and compare with `linalg.lyapunov`; then show that P fails to be positive definite when A has an unstable eigenvalue.")

EXTRAS["04_the_root_locus"] = _x(
    ["apply the root-locus rules: asymptotes, real-axis segments, breakaway points, departure angles", "read gains for a damping ratio and the stability limit from the locus",
     "anticipate how added poles and zeros reshape the locus", "relate the locus to Routh and to the gain margin"],
    "notebooks 01 and 03", "75 min",
    [md(r"""
## Inside the algorithm: the breakaway point from dK/ds = 0
"""), code(r"""N, D = G.num, G.den
dK = pl.sub(pl.mul(pl.derivative(N), D), pl.mul(N, pl.derivative(D)))      # d/ds (D/N) = 0  <=>  N' D - N D' = 0
cand = pl.roots(dK)
Kc = -pl.evaluate(D, cand) / pl.evaluate(N, cand)
print("candidates:", np.round(cand, 4), " gains:", np.round(Kc.real, 4), "-> the one with K > 0 on a locus segment is the breakaway point")""")],
    ["Draw the root locus of 1/(s (s + 4)(s^2 + 4s + 20)) by the rules first (asymptotes, breakaway, departure angles), then compute it. At what gain does it cross the imaginary axis, and does Routh agree?",
     "Add a zero to 1/(s (s + 1)(s + 2)) at -0.5, -1.5 and -3 in turn and describe how the locus and the maximum achievable damping change. Which position is best for a fast, well-damped loop?"],
    "**Implement it yourself:** compute a root locus by solving D(s) + K N(s) = 0 for 500 gains and matching each root to the nearest root at the previous gain (so that branches are continuous); plot it against `stability.root_locus`.")

EXTRAS["05_the_nyquist_criterion"] = _x(
    ["state the Nyquist criterion and count encirclements on a computed contour", "handle poles on the imaginary axis by indentation",
     "read gain and phase margins and the closest approach to -1 from the plot", "apply the criterion to loops with delays and unstable plants"],
    "notebooks 02 and 03", "75 min",
    [md(r"""
## Inside the algorithm: counting encirclements by the winding of 1 + L
"""), code(r"""r = st.nyquist_criterion(sy.tf([10.0], [1.0, 3.0, 2.0, 0.0]))
phase = np.unwrap(np.angle(1 + r["L"]))
print(f"total phase change of 1 + L along the contour: {np.degrees(phase[-1] - phase[0]):.1f} degrees = {(phase[-1] - phase[0]) / (2 * np.pi):.2f} turns (counterclockwise positive) -> N clockwise = {r['N_clockwise']}")""")],
    ["Apply the criterion to L = K/(s - 1)^2 for K = 0.5, 2 and 5 and compare with the closed-loop poles. Is there any K that stabilises it? Then add a zero, L = K (s + 1)/(s - 1)^2, and find the stabilising range.",
     "For P = 1/(s + 1)^3 with proportional gain K, find from the Nyquist curve the gain at which the loop goes unstable, and compare with the Routh limit and with the ultimate gain used in notebook 06."],
    "**Implement it yourself:** build the Nyquist contour for L = 1/(s^2 (s + 1)) with the indentation around the double pole, evaluate L along it and count the encirclements of -1; compare with `stability.nyquist_criterion` and with the closed-loop poles.")

EXTRAS["06_pid_control_and_tuning"] = _x(
    ["explain the role of each PID term and the need for a derivative filter", "fit a FOPDT model and apply Ziegler-Nichols, Cohen-Coon and SIMC rules",
     "run a relay autotuning experiment", "handle saturation, windup and the derivative kick in simulation"],
    "notebooks 01 and 02", "90 min",
    [md(r"""
## Inside the algorithm: one step of the discrete PID with back-calculation
"""), code(r"""Kp_, Ki_, Kd_, dt_, Tt_ = 2.0, 1.0, 0.5, 0.01, 0.5
I_, y_prev, e = 0.0, 0.0, 1.0                      # error 1 at the first sample
v = Kp_ * e + I_ - Kd_ * (0.0 - y_prev) / dt_      # unsaturated control
u_ = float(np.clip(v, -1.5, 1.5))
I_ += dt_ * (Ki_ * e + (u_ - v) / Tt_)             # the saturation excess (u - v) is fed back into the integrator
print(f"v = {v:.2f}, saturated u = {u_:.2f}, integrator after back-calculation {I_:.4f} (it would be {dt_ * Ki_ * e:.4f} without)")""")],
    ["Tune a PI controller for the thermal process `models.thermal_process()` (K = 2, tau = 50, L = 5) by ZN step, Cohen-Coon and SIMC with lambda = L and lambda = 3L; simulate with `design.simulate_pid(..., delay=5)` and compare overshoot, settling time and the control effort. Which would you install?",
     "Measure the ultimate gain of 1/(s + 1)^3 with the relay experiment using a hysteresis of 0.05 and relay amplitudes 0.5 and 2. How do the estimates change, and why does hysteresis bias the period?"],
    "**Implement it yourself:** write a discrete PID simulation with clamping anti-windup and a setpoint weighting b (proportional term on b r - y instead of r - y); show that b = 0.5 reduces the setpoint overshoot without changing the disturbance response.")

EXTRAS["07_lead_lag_and_loop_shaping"] = _x(
    ["design a lead compensator for a crossover and a phase margin", "design a lag compensator for steady-state accuracy",
     "read the cost of bandwidth in gain, control effort and noise", "add model-inverse feedforward for tracking"],
    "notebooks 02 and 06", "75 min",
    [md(r"""
## Inside the algorithm: the lead's alpha and T from the required phase
"""), code(r"""phi = np.radians(design["lead_angle"])
alpha = (1 - np.sin(phi)) / (1 + np.sin(phi)); T_ = 1 / (3.0 * np.sqrt(alpha))
print(f"alpha = (1 - sin phi)/(1 + sin phi) = {alpha:.4f}, T = 1/(w_c sqrt(alpha)) = {T_:.4f}: lead ({T_:.3f} s + 1)/({alpha * T_:.4f} s + 1); library: {dg.lead(design['lead_angle'], 3.0)}")""")],
    ["Design a lead-lag for 1/(s (s + 1)): crossover 3 rad/s, phase margin 50 degrees, and a velocity error constant Kv >= 50. Report the final margins and the ramp error.",
     "Repeat the lead design of the motor position loop with the sensor noise in mind: add white noise of 0.01 rad rms to the measurement in a simulation (use `response.lsim` on CS with a noise input) and tabulate the control-signal noise against the target crossover."],
    "**Implement it yourself:** write the lead design procedure (read the plant phase at w_c, compute the needed lead, build the compensator, set the gain) and reproduce `design.design_lead` for the DC motor position loop at w_c = 10 rad/s.")

EXTRAS["08_controllability_observability_and_pole_placement"] = _x(
    ["test controllability and observability by rank, singular values and the PBH test", "explain hidden modes and why unstable cancellations are forbidden",
     "place poles with Ackermann's formula and the Sylvester method", "recognise the ill-conditioning of pole placement in high order"],
    "notebooks 00 and 01", "75 min",
    [md(r"""
## Inside the algorithm: Ackermann's formula by hand
"""), code(r"""A2, B2 = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])     # double integrator
phi = pl.from_roots([-2.0, -3.0])                                             # desired (s + 2)(s + 3) = s^2 + 5 s + 6
phiA = A2 @ A2 + phi[1] * A2 + phi[2] * np.eye(2)
Wc = np.hstack([B2, A2 @ B2])
K_hand = np.array([0.0, 1.0]) @ np.linalg.inv(Wc) @ phiA
print("K by hand:", K_hand, " library:", sp.place(A2, B2, [-2.0, -3.0]).ravel(), " -> closed-loop poles", np.round(np.linalg.eigvals(A2 - B2 @ K_hand.reshape(1, 2)), 6))""")],
    ["For the cart-pendulum, place the poles at (-1 +- j, -2, -3), (-2 +- 2j, -5, -6) and (-4 +- 4j, -10, -12); simulate the recovery from 10 degrees and tabulate the settling time, the peak force and the cart excursion. Where does the actuator (say 10 N) rule out the design?",
     "The quarter car (`models.quarter_car`) has two inputs; is it controllable from the actuator force alone? From the road alone? Use the PBH test to name any mode that cannot be moved."],
    "**Implement it yourself:** place the poles of the 5-state random system by the Sylvester method yourself (choose G, solve A X - X Lambda = B G with `linalg.sylvester`, K = G X^-1) and verify the closed-loop eigenvalues; try two different G and compare the gains.")

EXTRAS["09_observers_and_the_separation_principle"] = _x(
    ["design a Luenberger observer by duality and interpret its error dynamics", "state and verify the separation principle",
     "add a reference gain for tracking", "measure the robustness cost of an observer in the loop"],
    "notebook 08", "60 min",
    [md(r"""
## Inside the algorithm: the error dynamics
"""), code(r"""E = A - L @ C
print("the estimation error obeys e' = (A - L C) e, independent of u and of x; its eigenvalues", np.round(np.sort(np.linalg.eigvals(E).real), 3))
print("block-triangular closed loop: the lower-left block is zero ->", np.max(np.abs(cl.A[4:, :4])))""")],
    ["Place the observer poles at 2, 5 and 20 times the controller's slowest pole and compare the step responses, the sensitivity to measurement noise (simulate with noise added to y) and the loop margins. Where is the sweet spot?",
     "Design a reduced-order observer for the DC motor that estimates only the current from the measured speed (the speed is measured, so only one state needs estimating). Compare its order and convergence with the full-order observer."],
    "**Implement it yourself:** build the observer-based compensator from (A, B, C, K, L) as a transfer function from y to u and close the loop with the plant in the transfer-function domain; confirm that the closed-loop poles are the union of the two designed sets.")

EXTRAS["10_the_linear_quadratic_regulator"] = _x(
    ["solve the Riccati equation and verify optimality", "choose weights and read their effect on speed and effort",
     "add integral action to LQR", "state and check the guaranteed margins"],
    "notebook 08", "75 min",
    [md(r"""
## Inside the algorithm: Newton-Kleinman iteration for the Riccati equation
"""), code(r"""S_ = B @ np.linalg.solve(R, B.T)
K0 = sp.place(A, B, [-1, -2, -3, -4])              # any stabilising gain to start
Pk = la.lyapunov((A - B @ K0).T, Q + K0.T @ R @ K0)
for it in range(6):
    Kk = np.linalg.solve(R, B.T @ Pk)
    Pk = la.lyapunov((A - B @ Kk).T, Q + Kk.T @ R @ Kk)
    print(f"iteration {it + 1}: |P_k - P| = {np.max(np.abs(Pk - P)):.2e}")""")],
    ["Apply Bryson's rule to the cart-pendulum: allowed excursions 0.1 m, 0.5 m/s, 0.05 rad, 0.5 rad/s and 5 N. Simulate the recovery from 10 degrees and check whether the force stays within 5 N; adjust R until it does.",
     "For the DC motor speed loop compare LQR with integral action (qi = 10, 100, 1000) against a PI controller tuned for the same settling time: overshoot, control effort, and response to a load torque step (an input disturbance)."],
    "**Implement it yourself:** compute the LQR cost J = int (x'Qx + u'Ru) dt by simulating the closed loop from x0 with `response.initial` and integrating numerically; compare with x0' P x0 and show that any other stabilising gain (e.g. from pole placement) gives a larger cost.")

EXTRAS["11_kalman_filtering_and_lqg"] = _x(
    ["derive the Kalman gain from the dual Riccati equation", "run a discrete Kalman filter and compare its error variance with the prediction",
     "assemble LQG and verify the separation", "recognise the loss of margins and loop transfer recovery"],
    "notebooks 09 and 10", "90 min",
    [md(r"""
## Inside the algorithm: one predictor step of the Kalman filter
"""), code(r"""xh = np.array([0.0, 10.0]); yk = 10.8; uk = 6.0
innovation = yk - (Cd @ xh)[0]
xh_next = Ad @ xh + Bd[:, 0] * uk + kd["L"][:, 0] * innovation
print(f"innovation {innovation:+.3f}; next estimate {np.round(xh_next, 4)}; library: {np.round(sp.kalman_filter_run(Ad, Bd, Cd, kd['L'], [uk, uk], [yk, yk])[1], 4)}")""")],
    ["Run the Kalman filter with a measurement noise variance ten times smaller than the true one (an over-confident filter) and ten times larger; compare the estimation error variances with the Riccati predictions. Which direction of mis-tuning is worse?",
     "Loop transfer recovery: for the DC motor LQG, scale the process noise covariance by 1, 10^2, 10^4, 10^6 and tabulate the margins at the plant input and the rms of the control signal in response to measurement noise. Where does the recovery stop paying?"],
    "**Implement it yourself:** write the time-varying discrete Kalman filter (the Riccati recursion P_{k+1} = A P A' - A P C'(C P C' + R)^-1 C P A' + Q together with the gain) and show that its gain converges to the steady-state one of `statespace.kalman`.")

EXTRAS["12_sampling_and_discretisation"] = _x(
    ["derive the zero-order-hold equivalent and its pole mapping", "compare Tustin (with prewarping), matched and Euler discretisations",
     "quantify the hold's half-sample delay", "explain aliasing and choose a sampling rate"],
    "notebook 01", "60 min",
    [md(r"""
## Inside the algorithm: Tustin's transform on the state-space matrices
"""), code(r"""Ac, Bc, Cc, Dc = Gss.A, Gss.B, Gss.C, Gss.D
k = 2 / dt
M = np.linalg.inv(k * np.eye(2) - Ac)
Ad_hand = M @ (k * np.eye(2) + Ac)
print("Tustin A by hand:\n", np.round(Ad_hand, 6), "\nlibrary:\n", np.round(dc.c2d(Gss, dt, 'tustin').A, 6))""")],
    ["Discretise the prototype G at dt = 0.1, 0.5, 1.0 and 1.5 by ZOH and by Tustin and compare the step responses with the continuous one at the samples. Which method is exact at the samples whatever dt, and which preserves stability whatever dt?",
     "The ZOH discretisation of 1/s^3 at small dt has zeros outside the unit circle (sampling zeros). Find them with `dc.c2d(sy.tf([1.0], [1, 0, 0, 0]), dt).zeros()` for dt = 0.1 and explain why a dead-beat or inverse-based digital controller would fail here."],
    "**Implement it yourself:** compute the ZOH matrices (Phi, Gamma) by the series expansion Phi = sum (A dt)^k/k!, Gamma = sum A^k dt^(k+1)/(k+1)! B truncated at 20 terms and compare with `linalg.expm_with_integral` for the DC motor at dt = 0.01 and at dt = 1.")

EXTRAS["13_digital_control_design"] = _x(
    ["discretise a continuous controller and check the loop it really closes", "place discrete poles and design dead-beat control",
     "use discrete LQR and the discrete Kalman filter", "account for computation delay, integrator discretisation and derivative filtering"],
    "notebooks 07, 08, 10 and 12", "90 min",
    [md(r"""
## Inside the algorithm: the dead-beat gain as a pole placement at z = 0
"""), code(r"""Kdb_hand = sp.acker(Ad, Bd, [0.0, 0.0, 0.0])
print("Ackermann with all poles at 0:", np.round(Kdb_hand, 1), " library:", np.round(Kdb, 1))""")],
    ["Implement the DC motor position loop as a discrete LQR with integral action at dt = 0.02 (augment the ZOH model with a discrete integrator of the error) and compare its tracking of a 1 rad step and its rejection of a load torque with the emulated lead design.",
     "Add a one-sample computation delay to the plant model (a state z^-1) and redesign the discrete pole placement for the delayed model. How does the achievable speed compare with the delay-free design at dt = 0.02 and at dt = 0.1?"],
    "**Implement it yourself:** write the discrete PID as a difference equation (u[k] from e[k], e[k-1], e[k-2] and u[k-1], u[k-2]) from the coefficients of `discrete.pid_discrete` and verify it against `response.step` of the controller transfer function.")

EXTRAS["14_robustness_and_limitations"] = _x(
    ["read the gang of four and the sensitivity peak", "state Bode's sensitivity integral and its consequences",
     "quantify the bandwidth limits of RHP zeros, unstable poles and delays", "apply the small-gain theorem to multiplicative uncertainty"],
    "notebooks 02, 05 and 07", "75 min",
    [md(r"""
## Inside the algorithm: Ms as the closest approach to -1
"""), code(r"""w_ = np.logspace(-2, 2, 20000)
Lj = fr.freqresp(g4["L"], w_)
print(f"min |1 + L| = {np.min(np.abs(1 + Lj)):.4f} -> Ms = {1 / np.min(np.abs(1 + Lj)):.4f}; library {rb.sensitivity_peak(g4['L'])['Ms']:.4f}")""")],
    ["For the levitated ball (unstable pole at 44 rad/s) design a lead controller with crossovers of 50, 100 and 200 rad/s and tabulate Ms, the margins and the peak |CS|. Confirm that the crossover must exceed about 2p for a decent Ms, and read the price in noise amplification.",
     "The plant (2 - s)/(s + 1)^2 under a PI controller: compute Ms as a function of the controller gain K and find the gain that minimises it; compare the corresponding crossover with the z/2 rule."],
    "**Implement it yourself:** verify Bode's integral numerically for the stabilised unstable loop of the notebook by integrating ln|S| on a fine grid up to 10^5 rad/s, and show that truncating the integral at 100 rad/s misses a visible fraction (the tail decays slowly).")

EXTRAS["15_system_identification"] = _x(
    ["fit FOPDT and second-order models to step responses and know their limits", "estimate ARX models by least squares and recognise the bias from output noise and fast sampling",
     "use the output-error method and residual whiteness", "fit a rational model to measured frequency-response data"],
    "notebooks 01, 02 and 12", "90 min",
    [md(r"""
## Inside the algorithm: the ARX regression matrix
"""), code(r"""na, nb, nk = 2, 2, 1
yy, uu = yv[::10], u[::10]
k0 = max(na, nb + nk - 1)
Phi = np.array([np.r_[-yy[k - na:k][::-1], uu[k - nk - nb + 1:k - nk + 1][::-1]] for k in range(k0, len(yy))])
theta = np.linalg.lstsq(Phi, yy[k0:], rcond=None)[0]
print("regression by hand: a =", np.round(theta[:na], 5), "b =", np.round(theta[na:], 5), " library:", np.round(ls10["a"], 5), np.round(ls10["b"], 5))""")],
    ["Repeat the ARX fit at 200 Hz with the instrumental-variable option (`identify.arx(..., method='iv')`) and with the record truncated to 10 s and 40 s. Does IV remove the bias, and how does its variance compare with the output-error fit?",
     "Fit the servo frequency response with orders (0, 3), (2, 5) and (2, 6) and compare the fits around the resonance; then add a Pade(2) delay factor to the model and fit again. Which model would you use for design, and why not the one with the best fit?"],
    "**Implement it yourself:** write the output-error cost (sum of squared simulation errors) as a function of the two poles and the gain of a first-order-plus-gain discrete model and minimise it with `scipy.optimize.minimize` on the PRBS data at 20 Hz; compare with `identify.output_error`.")

EXTRAS["16_model_reduction_and_realisation"] = _x(
    ["compute Gramians and Hankel singular values and interpret them", "balance a realisation and truncate it with the error bound",
     "realise a model from impulse-response data (ERA)", "explain why balanced or modal coordinates are numerically preferable"],
    "notebooks 00 and 08", "60 min",
    [md(r"""
## Inside the algorithm: balancing by the Cholesky-SVD route
"""), code(r"""Wc_, Wo_ = la.gramians(mot.A, mot.B, mot.C)
Lc = np.linalg.cholesky(Wc_)
U_, s_, Vt_ = np.linalg.svd(Lc.T @ Wo_ @ Lc)
T_ = Lc @ U_ @ np.diag(s_ ** -0.25)
balanced = mot.transform(T_)
Wcb_, Wob_ = la.gramians(balanced.A, balanced.B, balanced.C)
print("Gramians after the transformation:", np.round(np.diag(Wcb_), 6), np.round(np.diag(Wob_), 6), "- equal and diagonal; HSV = sqrt of the singular values:", np.round(np.sqrt(s_), 6))""")],
    ["Reduce the quarter-car model (4 states, 2 inputs, 2 outputs) to 2 states by balanced truncation and compare the singular-value plots of the full and reduced models; which mode was dropped, and at what frequency does the reduced model fail?",
     "Build the Hankel matrix from the motor's noisy impulse response with 10, 30 and 100 Markov parameters and tabulate the first five singular values; how does the gap between the second and third evolve with the record length and the noise?"],
    "**Implement it yourself:** write the ERA algorithm (block Hankel matrix, SVD, A = S^-1/2 U' H1 V S^-1/2, B and C from the factors) for the DC motor's discrete impulse response and compare the realised transfer function with the ZOH model.")

EXTRAS["17_multivariable_basics"] = _x(
    ["form the transfer matrix and read direction-dependent gain from singular values", "compute the RGA and choose a pairing",
     "apply static decoupling", "recognise when decentralised control fails and multivariable design is needed"],
    "notebooks 06 and 10", "75 min",
    [md(r"""
## Inside the algorithm: the RGA element by element
"""), code(r"""lam11_open = G0[0, 0]                                  # gain reflux -> top with the other loop open
Ginv = np.linalg.inv(G0)
lam11_closed = 1 / Ginv[0, 0]                           # gain with the other loop perfectly closed
print(f"open-loop gain {lam11_open:.2f}, closed-loop gain {lam11_closed:.2f}, ratio = relative gain {lam11_open / lam11_closed:.3f}; RGA[0,0] = {mimo.rga(G0)[0, 0]:.3f}")""")],
    ["Design the two PI loops of the column by SIMC with lambda = 5, 10 and 20 min, with and without the static decoupler, and tabulate the interaction (peak of the other output for a setpoint step) and the settling time. Does decoupling ever hurt?",
     "For the quarter car (inputs: road and actuator force; outputs: body acceleration and suspension travel) compute the singular values from 0.1 to 100 rad/s and explain the two peaks. Which input direction is the hardest to attenuate?"],
    "**Implement it yourself:** write a dynamic decoupler D(s) = G(s)^-1 G_diag(s) for the 2 x 2 column (each entry a transfer function) and show that G D is diagonal; is D proper and stable?")

EXTRAS["18_numerics_of_control"] = _x(
    ["explain scaling-and-squaring with Pade for the matrix exponential", "solve Lyapunov equations through the real Schur form",
     "solve Riccati equations by the Hamiltonian's invariant subspace and refine by Newton", "recognise the conditioning problems of polynomial representations and compute the H-infinity norm by bisection"],
    "notebooks 03, 10 and 16", "75 min",
    [md(r"""
## Inside the algorithm: one Newton-Kleinman step is a Lyapunov equation
"""), code(r"""Ak = A - S @ P_sub
dP = la.sylvester(Ak.T, Ak, -(A.T @ P_sub + P_sub @ A - P_sub @ S @ P_sub + Q))
print(f"after one Newton step from the subspace solution: residual {resid(P_sub + dP):.1e} (was {resid(P_sub):.1e})")""")],
    ["Time `linalg.expm`, the Taylor series and `scipy.linalg.expm` for random 10 x 10, 50 x 50 and 200 x 200 matrices of norm 1 and 100; where does the Taylor series lose accuracy, and how does the cost of the library's method scale?",
     "Compute the Riccati solution for the cart-pendulum with R from 1 to 1e-8: at what point does the subspace-only solution (before Newton) lose accuracy, and does Newton always recover it? Explain with the Hamiltonian's eigenvalue separation."],
    "**Implement it yourself:** write the bisection for the H-infinity norm of a SISO system (gamma is an upper bound iff the Hamiltonian has no imaginary eigenvalues) and reproduce `frequency.hinf_norm` for the prototype second-order system and for a lightly damped one with zeta = 0.01.")

EXTRAS["19_project_a_position_servo_from_data"] = _x(
    ["identify a plant from time- and frequency-domain data and model what the data show", "bound the remaining uncertainty and let the limits set the bandwidth",
     "compare classical and state-space designs against the full specification", "discretise, verify with saturation, and report"],
    "the whole course", "4 h",
    [md(r"""
## Inside the analysis: why the faster observer hurt
"""), code(r"""for scale in (1.0, 8.0):
    L_o = sp.observer_gain(A, C, -scale * np.array([15.0, 16.0, 17.0, 18.0, 19.0]))
    comp_ = sy.ss(A - B @ K - L_o @ C, L_o, K, np.zeros((1, 1)))
    L_i = sy.series(P_nom, comp_.to_tf())
    print(f"observer poles x{scale:.0f}: |L| at the resonance frequency {wr:.0f} rad/s = {abs(fr.freqresp(L_i, wr)):.3f}; at 2 x the crossover = {abs(fr.freqresp(L_i, 30.0)):.3f}")""")],
    ["Replace the LQR gain by a pole-placement gain with the same dominant poles and repeat the observer scan. Is the slow-observer advantage a property of LQR or of the plant?",
     "Add integral action to the state-space design (notebook 10's augmentation) so that a 10 % error in the servo gain leaves no position offset, and check what it costs in margins and settling time."],
    "**Implement it yourself:** write the complete digital controller as a difference-equation function `u = controller(y, r)` with its internal state (the discretised observer and the gain), and run it against the plant model at 200 Hz with the 12 V limit; the result must reproduce section 6's response.")


def apply_extras(name, cells):
    ex = EXTRAS[name]
    header = md(
        "**What you will learn**\n"
        + "\n".join(f"- {o}" for o in ex["objectives"])
        + f"\n\n**Before you start:** {ex['prereq']}  ·  **Time:** about {ex['time']}"
    )
    cells = [cells[0], header] + list(cells[1:])
    idx = [i for i, c in enumerate(cells) if c.cell_type == "markdown" and "## Exercises" in c.source]
    if idx:
        i = idx[0]
        n_items = len(re.findall(r"^\d+\. ", cells[i].source, flags=re.MULTILINE))
        cells[i] = md(cells[i].source.rstrip() + f"\n{n_items + 1}. {ex['implement']}")
        return cells[:i] + list(ex["inside"]) + cells[i:]
    items = ex["exercises"] + [ex["implement"]]
    exercises = md("## Exercises\n" + "\n".join(f"{k}. {t}" for k, t in enumerate(items, 1)))
    return cells + list(ex["inside"]) + [exercises]


def write_all(run: bool = True, only=()):
    import nbclient

    for name in sorted(NOTEBOOKS):
        if only and not name.startswith(tuple(only)):
            continue
        cells = apply_extras(name, NOTEBOOKS[name])
        nb = new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        if run:
            nbclient.NotebookClient(
                nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
            ).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    write_all(run="--no-run" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")])
