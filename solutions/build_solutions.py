"""Build the worked solutions to every exercise of the course, as executed notebooks.

The exercise texts are taken from ../notebooks/build_notebooks.py, so questions and solutions cannot
drift apart; the build fails if a notebook's number of solutions differs from its number of exercises.

    python solutions/build_solutions.py            # build and execute all solution notebooks
    python solutions/build_solutions.py 05 12      # only those whose names start with 05 or 12
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("course", HERE.parent / "notebooks" / "build_notebooks.py")
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
md, code = course.md, course.code
SOLUTIONS: dict[str, dict] = {}


def solutions(name, setup="", imports=()):
    """Register the solutions of one course notebook: a function returning one list of cells per exercise."""

    def deco(fn):
        SOLUTIONS[name] = {"setup": setup, "imports": list(imports), "answers": fn()}
        return fn

    return deco


def exercises(name) -> list[str]:
    cells = course.apply_extras(name, list(course.NOTEBOOKS[name]))
    text = [c.source for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source][0]
    items = re.split(r"\n(?=\d+\. )", text.split("## Exercises")[1].strip())
    return [re.sub(r"^\d+\.\s*", "", it).strip() for it in items]


def notebook_cells(name) -> list:
    title = re.sub(r"^#\s*", "", course.NOTEBOOKS[name][0].source.splitlines()[0])
    entry = SOLUTIONS[name]
    ex = exercises(name)
    if len(ex) != len(entry["answers"]):
        raise ValueError(f"{name}: {len(ex)} exercises but {len(entry['answers'])} solutions")
    cells = [
        md(f"""
# Solutions · {title}

Worked solutions to the exercises of [notebook {name[:2]}](../notebooks/{name}.ipynb). Try each exercise
yourself before reading its solution - the learning happens in the attempt. Solutions are one way to
answer each question; other correct approaches exist.
"""),
        course.setup_cell(entry["imports"]),
    ] + ([code(entry["setup"])] if entry["setup"] else [])
    for i, (text, answer) in enumerate(zip(ex, entry["answers"]), 1):
        cells.append(md(f"## Exercise {i}\n\n{text}"))
        cells.extend(answer)
    return cells


def write_all(only=()):
    import nbclient

    for name in SOLUTIONS:
        if only and not name.startswith(tuple(only)):
            continue
        nb = new_notebook(
            cells=notebook_cells(name),
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        nbclient.NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}
        ).execute()
        path = HERE / f"{name[:2]}_solutions.ipynb"
        nbformat.write(nb, path)
        print("wrote", path.name)




@solutions("00_models_of_linear_systems", imports=["from englincontrol import systems as sy, models, response as rs, statespace as sp"])
def _():
    return [
        [code(r"""A1, A2, R1, R2 = 1.0, 1.0, 2.0, 3.0
# mass balances: A1 h1' = q - (h1 - h2)/R1 ;  A2 h2' = (h1 - h2)/R1 - h2/R2
A = np.array([[-1 / (A1 * R1), 1 / (A1 * R1)], [1 / (A2 * R1), -1 / (A2 * R1) - 1 / (A2 * R2)]])
B = np.array([[1 / A1], [0.0]]); C = np.array([[0.0, 1.0]])
m = models.two_tanks(A1, A2, R1, R2)
print("A matches the library:", np.allclose(A, m["system"].A), "; B:", np.allclose(B, m["system"].B))
print(f"dc gain of level 2: {sy.ss(A, B, C, 0.0).dcgain()[0, 0]:.4f} = R2 (at steady state the whole inflow leaves through R2, so h2 = q R2)")"""),
         md(r"""
Each tank stores volume at the rate of its net flow; the flows through the valves are proportional to the
head across them. The resulting matrices are the library's, and the dc gain R₂ has a physical reading: at
steady state all the inflow passes through the outlet valve, so the second level is q R₂ whatever the first
valve does.
""")],
        [code(r"""G = sy.tf(np.polymul([1.0, 3.0], [1.0]), np.polymul(np.polymul([1.0, 1.0], [1.0, 2.0]), [1.0, 3.0]))
print("transfer function as written (no cancellation performed):", np.round(G.num, 3), "/", np.round(G.den, 3))
S_raw = sy.StateSpace(np.array([[-6.0, -11.0, -6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[1.0], [0.0], [0.0]]), np.array([[0.0, 1.0, 3.0]]), np.array([[0.0]]))
print(f"controllable canonical realisation: {S_raw.n} states; controllable: {sp.controllability(S_raw.A, S_raw.B)['controllable']}, observable: {sp.observability(S_raw.A, S_raw.C)['observable']}")
print(f"minimal realisation: {sy.minimal(S_raw).n} states:", sy.minimal(S_raw).to_tf())"""),
         md(r"""
The controllable canonical form is controllable by construction, but the mode at −3 is unobservable: the
numerator's zero at −3 cancels it in the output. The minimal realisation has two states and the transfer
function 1/((s+1)(s+2)). Losing a stable mode is harmless to the input-output behaviour (it decays on its own);
losing an unstable one would hide a runaway, as notebook 08 shows.
""")],
        [code(r"""motor = models.dc_motor()["system"]
sensor = sy.ss([[-50.0]], [[50.0]], [[1.0]], 0.0)             # first-order sensor with a 20 ms time constant
# series: output of the motor feeds the sensor
A = np.block([[motor.A, np.zeros((2, 1))], [sensor.B @ motor.C, sensor.A]])
B = np.vstack([motor.B, sensor.B @ motor.D]); C = np.hstack([sensor.D @ motor.C, sensor.C]); D = sensor.D @ motor.D
hand = sy.ss(A, B, C, D); lib = sy.series(motor, sensor)
print("poles by hand:", np.round(np.sort(hand.poles().real), 4), " library:", np.round(np.sort(lib.poles().real), 4))
print("transfer functions equal:", np.allclose(hand.to_tf().den, lib.to_tf().den) and np.allclose(hand.to_tf().num, lib.to_tf().num))"""),
         md(r"""
Stacking the states, the second system's input is the first's output: its B multiplies the first's C in the
lower-left block. The poles of the cascade are the union of the two sets, and the transfer function is the
product - the library's `series` does exactly this.
""")],
    ]


@solutions("01_time_response", imports=["from englincontrol import systems as sy, response as rs, models"])
def _():
    return [
        [code(r"""for J in (0.01, 0.02):
    m = models.dc_motor(J=J)
    G = m["system"]
    t, y = rs.step(G, 3.0, 3001)
    info = rs.step_info(t, y, final=G.dcgain()[0, 0])
    print(f"J = {J}: time constants {np.round(-1 / G.poles().real, 4)} s; rise time {info['rise_time']:.3f} s, settling {info['settling_time']:.3f} s")"""),
         md(r"""
The mechanical time constant (about J/b corrected by the back-emf) doubles with the inertia and the response
slows in proportion; the electrical one (L/R) is untouched. The rise and settling times follow the slow pole,
as the dominant-pole rule says.
""")],
        [code(r"""zeta_min = -np.log(0.10) / np.sqrt(np.pi**2 + np.log(0.10) ** 2)      # overshoot < 10 %
sigma_min = 4.0 / 1.0                                                   # 2 % settling < 1 s (approximation)
print(f"allowed region: zeta >= {zeta_min:.3f} (angle from the negative real axis <= {np.degrees(np.arccos(zeta_min)):.1f} deg) and Re(s) <= -{sigma_min}")
for p in (-5 + 5j, -4 + 7j, -2 + 2j):
    zeta, wn = -p.real / abs(p), abs(p)
    G = sy.tf([wn**2], [1.0, 2 * zeta * wn, wn**2])
    t, y = rs.step(G, 4.0, 4001)
    info = rs.step_info(t, y, final=1.0)
    ok = info["overshoot"] < 10 and info["settling_time"] < 1.0
    print(f"poles {p}: zeta {zeta:.3f}, sigma {-p.real}: overshoot {info['overshoot']:.1f} %, settling {info['settling_time']:.2f} s -> {'meets' if ok else 'fails'} the specification")"""),
         md(r"""
The specification is a wedge of the s-plane: left of σ = 4 and inside the lines at ±54° from the negative real
axis. The pair at −5 ± 5j lies inside and meets both; −4 ± 7j has too little damping (overshoot 17 %); −2 ± 2j is
well damped but too slow. The settling-time rule is approximate, so a pole pair near σ = 4 should be checked
by simulation, as here.
""")],
        [code(r"""G = sy.tf([4.0], [1.0, 1.2, 4.0])
S = G.to_ss()
t, y_init, _ = rs.initial(S, S.B[:, 0], 10.0, 1001)       # no input, x0 = B
t, y_imp = rs.impulse(G, 10.0, 1001)
print("initial-condition response from x0 = B vs impulse response:", f"{np.max(np.abs(y_init - y_imp)):.1e}")"""),
         md(r"""
An impulse at t = 0 transfers the state instantly from 0 to B (the integral of B δ(t)); afterwards the system
evolves freely. The impulse response is therefore the initial-condition response from x₀ = B, plus D δ(t) if
there is a direct term - exactly how the library computes it.
""")],
    ]


@solutions("02_frequency_response", imports=["from englincontrol import systems as sy, frequency as fr, models"])
def _():
    return [
        [code(r"""L = sy.zpk([-10.0], [0.0, -1.0, -50.0], 100.0)
m0 = fr.margins(L)
print(f"no delay: gain margin {m0['gain_margin']}, phase margin {m0['phase_margin']:.1f} deg, delay margin {m0['delay_margin'] * 1000:.0f} ms")
for Td in (0.05, 0.1, 0.12):
    m = fr.margins(L, delay=Td)
    print(f"delay {Td * 1000:4.0f} ms: gain margin {m['gain_margin']:.2f}, phase margin {m['phase_margin']:.1f} deg")"""),
         md(r"""
The delay subtracts ωT of phase: the phase margin falls first (by ω_gc T, 27° for 100 ms), while a finite gain
margin appears only once the phase reaches −180° somewhere. At the delay margin (120 ms) the phase margin
reaches zero - the delay margin computed without the delay predicted exactly where the loop would fail.
""")],
        [code(r"""G = models.dc_motor()["tf_speed"]
print("G =", G, "-> dc gain", round(G.dcgain(), 4), "= -20 dB and corners at the poles", np.round(-G.poles().real, 2), "rad/s")
w = np.logspace(-1, 3, 400)
_, mag, ph = fr.bode(G, w)
asym = 20 * np.log10(G.dcgain()) - 20 * np.log10(np.maximum(w / 2.0, 1)) - 20 * np.log10(np.maximum(w / 10.0, 1))
print(f"asymptotes: flat at {20 * np.log10(G.dcgain()):.1f} dB, then -20 dB/dec from 2 rad/s and -40 dB/dec from 10 rad/s; max deviation {np.max(np.abs(mag - asym)):.2f} dB")
print(f"bandwidth (-3 dB) {fr.bandwidth(G):.2f} rad/s: set by the mechanical pole at 2 rad/s (the slow one)")"""),
         md(r"""
Two real poles give two corners; the bandwidth is at the first of them, the mechanical time constant - the
motor's speed cannot follow voltage variations faster than about 2 rad/s, whatever the electrical pole does.
""")],
        [code(r"""n = 10
poles = -np.arange(1, n + 1, dtype=float)
G = sy.zpk([], poles, 1.0)
S = G.to_ss()
w = np.logspace(-1, 2, 200)
H_tf = fr.freqresp(G, w)
H_ss = fr.freqresp(S, w)
exact = np.prod([1 / (1j * w - p) for p in poles], axis=0)
print(f"10th order: transfer-function evaluation error {np.max(np.abs(H_tf - exact) / np.abs(exact)):.1e}, state-space (companion) error {np.max(np.abs(H_ss - exact) / np.abs(exact)):.1e}")
bal, _ = sy.balanced_realisation(S)
print(f"balanced state-space error {np.max(np.abs(fr.freqresp(bal, w) - exact) / np.abs(exact)):.1e}")"""),
         md(r"""
For a 10th-order system both the polynomial evaluation (Horner's scheme is stable for an imaginary argument)
and the companion-form linear solve are accurate to rounding - and the balanced realisation, computed from
the companion form through two Gramians, is wrong by a factor of 8: its transformation holds inverse fourth
roots of Hankel singular values spanning many orders of magnitude and injects the error (notebook 16 meets
this again). The representation one *computes into* matters as much as the one one evaluates in.
""")],
    ]


@solutions("03_stability_tests", imports=["from englincontrol import systems as sy, stability as st, polynomials as pl, linalg as la, models"])
def _():
    return [
        [code(r"""print("1 + K (s + a)/(s^2 (s + 10)) = 0  ->  s^3 + 10 s^2 + K s + K a = 0")
print("Routh: first column 1, 10, (10 K - K a)/10, K a  ->  stable iff K > 0, a > 0 and a < 10 (then K cancels from the third row's sign)")
for a in (1.0, 5.0, 12.0):
    verdicts = [st.routh([1.0, 10.0, K, K * a])["stable"] for K in (0.1, 1.0, 10.0, 100.0)]
    print(f"a = {a:4}: stable for K = 0.1, 1, 10, 100? {verdicts}")"""),
         md(r"""
The array gives the condition symbolically: with the zero inside the double-integrator plant's left half-plane
(0 < a < 10) every positive gain stabilises the loop; with a > 10 none does. A root locus would show the same
(the zero must lie between the double pole at the origin and the pole at −10 for the branches to bend left).
""")],
        [code(r"""ml = models.magnetic_levitation()["system"]
A, B = ml.A, ml.B
a, b = A[1, 0], B[1, 0]
print(f"x'' = {a:.1f} x + ({b:.2f}) u")
print(f"the input gain b is negative, so the stabilising sign is u = +K x: s^2 + 0 s + ({-b:.2f} K - {a:.1f}); the s coefficient is zero for every K, so the Routh array can never be all positive")
K = 100.0
print("   eigenvalues with u = K x (K = 100):", np.round(np.linalg.eigvals(A + B @ [[K, 0.0]]), 3), "(imaginary: marginal, never stable)")
Kd = 0.5
print(f"u = K x + Kd x' gives s^2 + ({-b:.2f} Kd) s + ({-b:.2f} K - {a:.1f}): positive coefficients for Kd > 0 and K > {a / -b:.2f}")
print("   eigenvalues with K = 100, Kd = 0.5:", np.round(np.linalg.eigvals(A + B @ [[K, Kd]]), 3))"""),
         md(r"""
Position feedback alone cannot add damping to a system without any: the characteristic polynomial lacks its
s-term and the Routh array says marginal at best (and with the gain too small, two real poles, one unstable).
Velocity feedback (a derivative term, or a lead) supplies the damping; this is why every levitation and every
inverted-pendulum controller contains a derivative. The sign of the feedback follows the sign of the input
gain - a detail that reverses stability if overlooked.
""")],
        [code(r"""rng = np.random.default_rng(0)
A = rng.normal(size=(3, 3)) - 3 * np.eye(3)
Q = np.eye(3)
K_ = np.kron(np.eye(3), A.T) + np.kron(A.T, np.eye(3))          # vec(A' P + P A) = (I x A' + A' x I) vec(P)
P_hand = np.linalg.solve(K_, -Q.reshape(-1, order="F")).reshape(3, 3, order="F")
print("Kronecker solution vs library:", f"{np.max(np.abs(P_hand - la.lyapunov(A.T, Q))):.1e}", "; eigenvalues of P:", np.round(np.linalg.eigvalsh(P_hand), 4))
A_un = A + 4 * np.eye(3)                                          # push an eigenvalue into the right half-plane
print("unstable A: eigenvalues", np.round(np.linalg.eigvals(A_un).real, 3), "-> P has eigenvalues", np.round(np.linalg.eigvalsh(la.lyapunov(A_un.T, Q)), 3), "(not all positive)")"""),
         md(r"""
The vec trick turns the matrix equation into an ordinary linear system of n² unknowns; the solution is
symmetric and positive definite exactly when A is stable, and acquires a negative eigenvalue as soon as A
has an unstable one - Lyapunov's theorem in a computation.
""")],
    ]


@solutions("04_the_root_locus", imports=["from englincontrol import systems as sy, stability as st, polynomials as pl"])
def _():
    return [
        [code(r"""G = sy.tf([1.0], pl.mul(pl.from_roots([0.0, -4.0]), [1.0, 4.0, 20.0]))
rl = st.root_locus(G, np.logspace(-1, 3, 1000))
print(f"rules: centroid {rl['centroid']:.2f}, asymptotes {rl['asymptote_angles']} deg, breakaway {np.round(rl['breakaway'], 3)}, departure angles {({k: round(float(v), 1) for k, v in rl['departure_angles'].items()})}")
# imaginary-axis crossing: Routh on s^4 + 8 s^3 + 36 s^2 + 80 s + K
Kc = None
for K in np.linspace(1, 400, 4000):
    if not st.routh(pl.add(G.den, K * G.num))["stable"]:
        Kc = K; break
print(f"Routh: unstable from K = {Kc:.0f} (closed form: 8*36*80 - 80^2 = 8^2 K -> K = {(8 * 36 * 80 - 80**2) / 64:.0f})")
cross = pl.roots(pl.add(G.den, (8 * 36 * 80 - 80**2) / 64 * G.num))
print("poles at that gain:", np.round(cross, 3))"""),
         md(r"""
Four poles, no zeros: asymptotes at ±45° and ±135° from the centroid −2, a breakaway between 0 and −4, and
departure angles from −2 ± 4j that point towards the right half-plane. The Routh array gives the crossing gain
260 and the frequency ±j√10 where the locus meets the imaginary axis; the computed locus confirms both.
""")],
        [code(r"""for z in (-0.5, -1.5, -3.0):
    Gz = sy.tf(pl.from_roots([z]), pl.from_roots([0.0, -1.0, -2.0]))
    gains = np.logspace(-2, 3, 3000)
    rl = st.root_locus(Gz, gains)
    best = None
    for K, r in zip(gains, rl["branches"]):
        dom = r[np.argmax(r.real)]                                   # the dominant (slowest) pole
        if abs(dom.imag) > 1e-9 and dom.real <= -0.5:                # the first gain at which the dominant pair is at least as fast as sigma = 0.5
            best = (K, dom, -dom.real / abs(dom)); break
    print(f"zero at {z:4}: centroid {rl['centroid']:.2f}, always stable: {np.all(rl['branches'].real < 1e-9)}; " + (f"dominant pair reaches sigma = 0.5 at K = {best[0]:.2f} with damping ratio {best[2]:.3f}" if best else "the dominant pair never reaches sigma = 0.5 (the vertical asymptotes sit to its right)"))"""),
         md(r"""
A zero on the real axis pulls the locus towards it and, by reducing the relative degree to two, makes the
asymptotes vertical at the centroid: all three loops are stable at every gain, but they differ in what the
dominant pole does. The zero at −0.5 captures the branch leaving the origin: the slowest closed-loop pole is
real and creeps towards −0.5, so the loop can never be faster than σ = 0.5 whatever the gain. The zero at
−1.5 lets the branches from 0 and −1 meet, break away and reach σ = 0.5 as a well-damped pair - the useful
case. The zero at −3 moves the centroid to the origin, and the complex pair heads up the imaginary axis without
ever reaching σ = 0.5. A zero near the dominant poles, but not between the slowest pole and the origin, is
the one that helps.
""")],
        [code(r"""G = sy.tf([1.0], pl.from_roots([0.0, -1.0, -2.0]))
gains = np.logspace(-2, 2, 500)
prev = None; branches = []
for K in gains:
    r = pl.roots(pl.add(G.den, K * G.num))
    if prev is not None:
        order, avail = [], list(range(3))
        for pk in prev:
            j = min(avail, key=lambda j: abs(r[j] - pk)); order.append(j); avail.remove(j)
        r = r[order]
    branches.append(r); prev = r
branches = np.array(branches)
lib = st.root_locus(G, gains)["branches"]
print("branches matched to the library's at every gain:", f"{np.max([np.max(np.abs(np.sort_complex(a) - np.sort_complex(b))) for a, b in zip(branches, lib)]):.1e}")
print("continuity: largest jump between consecutive gains along a branch", f"{np.max(np.abs(np.diff(branches, axis=0))):.3f}")"""),
         md(r"""
Solving the characteristic polynomial for each gain gives the points; matching each root to the nearest one at
the previous gain strings them into continuous branches (a greedy assignment that works when the gain steps
are small). The library does the same, which is why the branches, not just the point sets, agree.
""")],
    ]


@solutions("05_the_nyquist_criterion", imports=["from englincontrol import systems as sy, stability as st, frequency as fr, polynomials as pl, design as dg"])
def _():
    return [
        [code(r"""for K in (0.5, 2.0, 5.0):
    L = sy.tf([K], pl.from_roots([1.0, 1.0]))
    r = st.nyquist_criterion(L)
    print(f"K/(s - 1)^2, K = {K}: N_cw = {r['N_clockwise']}, P = {r['P']}, Z = {r['Z']}; closed-loop poles {np.round(sy.feedback(L).poles(), 3)}")
print("the Nyquist curve of K/(s-1)^2 starts at +K on the real axis and never encircles -1 counterclockwise twice: no K works")
for K in (1.0, 3.0, 10.0):
    L = sy.tf(K * np.array([1.0, 1.0]), pl.from_roots([1.0, 1.0]))
    r = st.nyquist_criterion(L)
    print(f"K (s + 1)/(s - 1)^2, K = {K}: Z = {r['Z']} (N_cw {r['N_clockwise']}); closed-loop poles {np.round(sy.feedback(L).poles(), 3)}")
print("Routh on s^2 + (K - 2) s + (K + 1): stable for K > 2")"""),
         md(r"""
A double unstable pole needs two counterclockwise encirclements of −1, which the curve of K/(s − 1)² (which
starts at +K and sweeps through the lower half-plane) never provides: every gain leaves Z = 2 or Z = 1.
Adding a zero at −1 turns the phase around; for K > 2 the curve encircles −1 twice counterclockwise and Z = 0
- the Routh array on s² + (K − 2)s + (K + 1) gives the same threshold.
""")],
        [code(r"""P = sy.tf([1.0], [1.0, 3.0, 3.0, 1.0])
w = np.logspace(-2, 2, 20000)
Pj = fr.freqresp(P, w)
i = np.argmin(np.abs(np.degrees(np.unwrap(np.angle(Pj))) + 180))
K_nyq = 1 / abs(Pj[i])
print(f"Nyquist: the curve crosses the negative real axis at w = {w[i]:.4f} with |P| = {abs(Pj[i]):.5f} -> the loop reaches -1 at K = {K_nyq:.3f}")
print(f"Routh on s^3 + 3 s^2 + 3 s + 1 + K: stable iff 3*3 > 1 + K, i.e. K < 8; ultimate gain of notebook 06: {dg.ultimate_gain(P)['Ku']:.3f}")"""),
         md(r"""
Three statements of one fact: the Nyquist curve reaches −1 at K = 8, the Routh array goes marginal at K = 8,
and the ultimate gain of the relay/ZN method is 8. The phase crossover at √3 rad/s is where (s + 1)³ has
phase −180° - three poles each contributing 60°.
""")],
        [code(r"""L = sy.tf([1.0], [1.0, 1.0, 0.0, 0.0])
eps, W = 1e-3, 1e4
parts = []
parts.append(1j * -np.logspace(np.log10(W), np.log10(eps), 4000))                       # down the negative axis to -j eps
th = np.linspace(-np.pi / 2, np.pi / 2, 400); parts.append(eps * np.exp(1j * th))        # the indentation around the double pole
parts.append(1j * np.logspace(np.log10(eps), np.log10(W), 4000))                         # up the positive axis
s = np.concatenate(parts)
Ls = pl.evaluate(L.num, s) / pl.evaluate(L.den, s)
phase = np.unwrap(np.angle(1 + Ls))
N_cw = -int(np.round((phase[-1] - phase[0]) / (2 * np.pi)))
print(f"encirclements by hand: N_cw = {N_cw}; library: {st.nyquist_criterion(L)['N_clockwise']}; closed-loop RHP poles: {int(np.sum(sy.feedback(L).poles().real > 0))}")"""),
         md(r"""
The double pole at the origin is excluded by the small semicircle, whose image is a large arc that sweeps
360° (twice the angle of a simple pole), and the count comes out as two clockwise encirclements: the loop
1/(s²(s + 1)) is unstable under unity feedback with two unstable closed-loop poles, as the characteristic
polynomial confirms.
""")],
    ]


@solutions("06_pid_control_and_tuning", imports=["from englincontrol import systems as sy, design as dg, response as rs, frequency as fr, models"])
def _():
    return [
        [code(r"""th = models.thermal_process()
K, tau, L = th["params"]["K"], th["params"]["tau"], th["params"]["L"]
rules = {"ZN step": dg.ziegler_nichols_step(K, tau, L)["PI"], "Cohen-Coon": dg.cohen_coon(K, tau, L)["PI"], "SIMC lambda = L": dg.imc_pid(K, tau, L, lam=L), "SIMC lambda = 3L": dg.imc_pid(K, tau, L, lam=3 * L)}
rows = []
for name, p_ in rules.items():
    sim = dg.simulate_pid(th["tf"], p_["Kp"], p_["Kp"] / p_["Ti"], 0.0, 1.0, dt=0.05, T=300.0, delay=L)
    info = rs.step_info(sim["t"], sim["y"], final=1.0)
    rows.append((name, p_["Kp"], p_["Ti"], info["overshoot"], info["settling_time"], np.max(np.abs(sim["u"]))))
print(pd.DataFrame(rows, columns=["rule", "Kp", "Ti", "overshoot %", "settling (s)", "peak |u|"]).to_string(index=False, float_format="%.2f"))"""),
         md(r"""
On a process with a 50 s time constant and a 5 s delay the Ziegler-Nichols and Cohen-Coon PI tunings are
fast and oscillatory with large control moves; SIMC with λ = L is brisk but smooth, and λ = 3L is gentle. For
a heater with a slow, well-behaved process and no tight constraint on speed the SIMC λ = L tuning is the one
to install: 10 % overshoot, half the peak heater power of ZN, and a 60°-class margin.
""")],
        [code(r"""P = sy.tf([1.0], [1.0, 3.0, 3.0, 1.0])
for d, h in ((0.5, 0.0), (2.0, 0.0), (0.5, 0.05), (2.0, 0.05)):
    r = dg.relay_experiment(P, amplitude=d, hysteresis=h, dt=0.005, T=80.0)
    print(f"amplitude {d}, hysteresis {h}: Ku = {r['Ku']:.3f} (exact 8), Pu = {r['Pu']:.3f} s (exact 3.628)")"""),
         md(r"""
Without hysteresis the estimates are independent of the relay amplitude (the loop is linear apart from the
relay, so the oscillation scales). With hysteresis the switching lags the zero crossing: the period lengthens
and the amplitude grows, so the ideal-relay formula underestimates K_u (4.5 and 6.3 against 8) and
overestimates P_u - the more so the smaller the relay amplitude relative to the hysteresis. The standard
correction treats the hysteresis as an extra phase in the describing function; in practice one keeps the
hysteresis just above the noise and the relay amplitude well above it.
""")],
        [code(r"""def pid_setpoint_weight(plant, Kp, Ki, Kd, b, dt=0.01, T=20.0, u_limits=(-3.0, 3.0), r_step=1.0, d_step=0.0, t_d=10.0):
    from englincontrol import linalg as la
    s = plant.to_ss(); Phi, Gam = la.expm_with_integral(s.A, s.B, dt)
    n = int(T / dt); x = np.zeros(s.n); I_ = 0.0; y_prev = 0.0; y = np.zeros(n); u = np.zeros(n)
    for k in range(n):
        t = k * dt; yk = (s.C @ x)[0]; y[k] = yk
        v = Kp * (b * r_step - yk) + I_ - Kd * (yk - y_prev) / dt      # proportional term on b r - y, derivative on -y
        uk = float(np.clip(v, *u_limits)); u[k] = uk
        if not (uk != v and np.sign(r_step - yk) == np.sign(v)):       # clamping anti-windup
            I_ += dt * Ki * (r_step - yk)
        y_prev = yk
        x = Phi @ x + Gam[:, 0] * (uk + (d_step if t >= t_d else 0.0))
    return np.arange(n) * dt, y, u
P = sy.tf([1.0], [1.0, 3.0, 3.0, 1.0])
Kp, Ki, Kd = 4.8, 4.8 / 1.81, 4.8 * 0.45
for b in (1.0, 0.5):
    t, y, u = pid_setpoint_weight(P, Kp, Ki, Kd, b, d_step=-0.5)
    print(f"b = {b}: setpoint overshoot {100 * (np.max(y[t < 10]) - 1):.1f} %; disturbance dip after t = 10: {np.min(y[t > 10]) - 1:+.3f} (the same for both)")"""),
         md(r"""
Setpoint weighting applies the proportional term to b·r − y instead of r − y: a setpoint step enters the
controller through the integral (and a fraction b of the proportional term) only, which removes the kick and
the overshoot, while the response to a disturbance - which enters through y - is unchanged. It is the
cheapest form of the two-degree-of-freedom controller, and most industrial PIDs have it.
""")],
    ]


@solutions("07_lead_lag_and_loop_shaping", imports=["from englincontrol import systems as sy, design as dg, frequency as fr, response as rs, models, polynomials as pl"])
def _():
    return [
        [code(r"""P = sy.tf([1.0], [1.0, 1.0, 0.0])
lead_d = dg.design_lead(P, w_c=3.0, pm_target_deg=50.0)
C_lead = lead_d["controller"]
Kv_now = sy.series(C_lead, P).num[-1] / sy.series(C_lead, P).den[-2]          # velocity constant = lim s L(s)
beta = 50.0 / Kv_now
C_lag = dg.lag(beta, 3.0 / 10.0)
C = sy.series(C_lag, C_lead)
L = sy.series(C, P)
m = fr.margins(L)
Kv = L.num[-1] / L.den[-2]
t = np.linspace(0, 20, 4001)
_, y_ramp, _ = rs.lsim(sy.feedback(L), t, t)
print(f"lead alone: Kv = {Kv_now:.2f}; lag with beta = {beta:.2f} (zero a decade below the crossover) -> Kv = {Kv:.1f}, crossover {m['w_gc']:.2f} rad/s, phase margin {m['phase_margin']:.1f} deg, ramp error {t[-1] - y_ramp[-1]:.4f} (1/Kv = {1 / Kv:.4f})")"""),
         md(r"""
The lead sets the crossover and the margin; the lag multiplies the low-frequency gain by β without moving
them, because its zero sits a decade below the crossover. The velocity error constant rises from 5 to 50 and
the ramp error falls to 1/K_v = 0.02, at a cost of a few degrees of phase margin - the classical lead-lag
recipe.
""")],
        [code(r"""mot = models.dc_motor(position=True)["system"].to_tf()
rng = np.random.default_rng(0)
t = np.linspace(0, 2, 2001)
noise = 0.01 * rng.normal(size=len(t))
rows = []
for wc in (2.0, 5.0, 10.0, 20.0):
    d = dg.design_lead(mot, w_c=wc, pm_target_deg=50.0)
    cl = dg.closed_loop(mot, d["controller"])
    _, u_noise, _ = rs.lsim(cl["CS"], noise, t)                      # measurement noise n enters the control through -CS n (sign irrelevant for the rms)
    rows.append((wc, d["K"], np.std(u_noise), np.max(np.abs(fr.freqresp(cl["CS"], np.logspace(-1, 4, 2000))))))
print(pd.DataFrame(rows, columns=["target w_c", "gain K", "control noise rms (V)", "peak |CS|"]).to_string(index=False, float_format="%.3f"))"""),
         md(r"""
Measurement noise reaches the actuator through CS, whose high-frequency gain is the controller's: each
doubling of the crossover multiplies the control-signal noise by about ten here (two cascaded leads, each
with a growing 1/α), until at 20 rad/s a 0.01 rad sensor noise produces hundreds of volts of actuator
chatter - far beyond the 12 V supply. Bandwidth is paid in noise amplification as
surely as in control effort; a lead's high-frequency gain 1/α is the price tag.
""")],
        [code(r"""P = models.dc_motor(position=True)["system"].to_tf()
w_c, pm_target, extra = 10.0, 50.0, 5.0
w_grid = np.logspace(np.log10(w_c) - 4, np.log10(w_c), 2000)
phase_now = np.degrees(np.unwrap(np.angle(fr.freqresp(P, w_grid))))[-1]
needed = pm_target - (180.0 + phase_now) + extra
n_leads = 1 if needed <= 65 else int(np.ceil(needed / 60.0))
phi = np.radians(needed / n_leads)
alpha = (1 - np.sin(phi)) / (1 + np.sin(phi)); T = 1 / (w_c * np.sqrt(alpha))
C = sy.tf([1.0], [1.0])
for _ in range(n_leads):
    C = sy.series(C, sy.tf([T, 1.0], [alpha * T, 1.0]))
K = 1 / abs(fr.freqresp(sy.series(C, P), w_c))
C = sy.tf(K * C.num, C.den)
m = fr.margins(sy.series(C, P))
lib = dg.design_lead(P, w_c, pm_target)
print(f"plant phase at {w_c} rad/s: {phase_now:.1f} deg -> lead needed {needed:.1f} deg in {n_leads} stage(s); result: crossover {m['w_gc']:.3f}, phase margin {m['phase_margin']:.1f} deg; library: {lib['margins']['w_gc']:.3f}, {lib['margins']['phase_margin']:.1f} deg")"""),
         md(r"""
The procedure in five lines: unwrap the plant's phase up to the target crossover, compute the missing lead
(with a few degrees of allowance), split it over two stages when one cannot supply it, build the leads with
their maximum phase at the crossover, and scale the gain to unity loop gain there. The result matches the
library's to the digit.
""")],
    ]


@solutions("08_controllability_observability_and_pole_placement", imports=["from englincontrol import systems as sy, statespace as sp, response as rs, models, linalg as la"])
def _():
    return [
        [code(r"""cp = models.cart_pendulum()["system"]
A, B, C = cp.A, cp.B, cp.C
rows = []
for des in ([-1 + 1j, -1 - 1j, -2.0, -3.0], [-2 + 2j, -2 - 2j, -5.0, -6.0], [-4 + 4j, -4 - 4j, -10.0, -12.0]):
    K = sp.place(A, B, des)
    Acl = sy.ss(A - B @ K, B, C, np.zeros((2, 1)))
    t, y, x = rs.initial(Acl, [0, 0, np.radians(10), 0], 6.0, 6001)
    u = -(x @ K.T)[:, 0]
    settled = t[np.max(np.flatnonzero(np.abs(x[:, 2]) > 0.02 * np.radians(10))) + 1]
    rows.append((str(np.round(des, 1)), settled, np.max(np.abs(u)), 100 * np.max(np.abs(x[:, 0]))))
print(pd.DataFrame(rows, columns=["poles", "angle settled (s)", "peak force (N)", "cart excursion (cm)"]).to_string(index=False, float_format="%.2f"))"""),
         md(r"""
Faster poles mean a faster recovery and a smaller cart excursion, bought with force: the fastest set asks
for about 20 N, twice a 10 N actuator's limit, and would saturate - after which the linear design means
nothing. The second set is the practical choice. Pole placement does not know about actuators; the
designer must check, which is why LQR's cost weights are the more natural language (notebook 10).
""")],
        [code(r"""qc = models.quarter_car()["system"]
for j, name in ((1, "actuator force alone"), (0, "road alone")):
    c = sp.controllability(qc.A, qc.B[:, [j]])
    pbh = sp.pbh_test(qc.A, qc.B[:, [j]])
    print(f"{name:<22}: rank {c['rank']} of 4, singular values {np.array2string(c['singular_values'], precision=2)}; uncontrollable modes {np.round(pbh['uncontrollable_modes'], 2)}")
print("both inputs together: rank", sp.controllability(qc.A, qc.B)["rank"])"""),
         md(r"""
The quarter car is controllable from the actuator force alone and from the road alone - every mode can be
moved by either - but the singular values of the controllability matrices show that the road reaches the
wheel-hop mode much more easily than the body mode and the actuator the reverse. Controllability is a yes/no
property; the singular values say how much control it will take.
""")],
        [code(r"""rng = np.random.default_rng(2)
A5, B5 = rng.normal(size=(5, 5)), rng.normal(size=(5, 2))
des5 = np.array([-1, -2, -3 + 2j, -3 - 2j, -5], complex)
Lam = sp._real_block_diag(des5)
for seed in (0, 1):
    G = np.random.default_rng(seed).normal(size=(2, 5))
    X = la.sylvester(A5, -Lam, B5 @ G)                              # A X - X Lam = B G
    K = G @ np.linalg.inv(X)
    print(f"G from seed {seed}: closed-loop poles {np.round(np.sort_complex(np.linalg.eigvals(A5 - B5 @ K)), 6)}, |K| = {np.linalg.norm(K):.2f}")"""),
         md(r"""
With two inputs the gain matrix is not unique: any parameter matrix G (with X invertible) gives a K that
places the poles exactly, and different choices give gains of different size. The freedom is what robust
pole placement spends on well-conditioned closed-loop eigenvectors; the library's choice is simply the
smallest-norm gain among a few random tries.
""")],
    ]


@solutions("09_observers_and_the_separation_principle", imports=["from englincontrol import systems as sy, statespace as sp, response as rs, frequency as fr, models"])
def _():
    return [
        [code(r"""mot = models.dc_motor(position=True)["system"]
K = sp.place(mot.A, mot.B, [-8 + 6j, -8 - 6j, -20.0])
slowest = 8.0
rng = np.random.default_rng(0)
rows = []
for factor in (2.0, 5.0, 20.0):
    L = sp.observer_gain(mot.A, mot.C, -factor * slowest * np.array([1.0, 1.1, 1.2]))
    cl = sp.closed_loop_with_observer(mot, K, L)
    t, y = rs.step(cl, 1.5)
    comp = sy.ss(mot.A - mot.B @ K - L @ mot.C, L, K, np.zeros((1, 1)))
    L_in = sy.series(mot.to_tf(), comp.to_tf())
    m = fr.margins(L_in)
    # noise: the estimate's response to measurement noise through the observer
    obs = sp.observer(mot, L)
    tn = np.linspace(0, 2, 2001)
    _, xh, _ = rs.lsim(obs, np.column_stack([np.zeros(2001), 0.01 * rng.normal(size=2001)]), tn)
    rows.append((factor, rs.step_info(t, y, final=1.0)["settling_time"], m["gain_margin"], m["phase_margin"], np.std(xh[:, 1])))
print(pd.DataFrame(rows, columns=["observer speed / controller", "settling (s)", "gain margin", "phase margin (deg)", "speed-estimate noise (rad/s)"]).to_string(index=False, float_format="%.3f"))"""),
         md(r"""
The step response does not change at all - the separation principle promises that - but two things do. The
noise in the speed estimate grows sixtyfold between the slowest and the fastest observer, and the loop's
margins *improve* with a faster observer: for this well-modelled motor the observer-based loop approaches
the full-state-feedback loop as the observer gets fast (loop transfer recovery). The sweet spot is where the
margins are acceptable and the noise still tolerable, a few times the controller's speed. The project's
servo, with an unmodelled resonance, turns the margin trend the other way round - which is why the choice
has to be made on the loop that is actually closed.
""")],
        [code(r"""mot = models.dc_motor()["system"]            # states [current, speed]; speed measured
A, B, C = mot.A, mot.B, mot.C
# reduced-order observer for the current (state 1) from the measured speed (state 2):
# with x = [i, w]: i' = A11 i + A12 w + B1 u ; w' = A21 i + A22 w + B2 u  ->  the second equation is a measurement of i: w' - A22 w - B2 u = A21 i
A11, A12, A21, A22 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]; B1, B2 = B[0, 0], B[1, 0]
pole = -60.0
Lr = (A11 - pole) / A21                              # observer gain: error dynamics e' = (A11 - Lr A21) e
# z = i_hat - Lr w obeys z' = (A11 - Lr A21) z + (A12 - Lr A22 + (A11 - Lr A21) Lr) w + (B1 - Lr B2) u
t = np.linspace(0, 1, 1001); u = 12.0 * np.ones_like(t)
_, y, x = rs.lsim(mot, u, t, x0=[0.5, 5.0])
z = 0.0; ihat = np.zeros_like(t); dt = t[1] - t[0]
Fz = A11 - Lr * A21; Gz = A12 - Lr * A22 + Fz * Lr; Hz = B1 - Lr * B2
for k in range(len(t)):
    ihat[k] = z + Lr * y[k]
    z = z + dt * (Fz * z + Gz * y[k] + Hz * u[k])
err = ihat - x[:, 0]
print(f"reduced-order observer (1 state, pole {pole}): current error {err[0]:+.3f} A at t = 0, {err[100]:+.1e} at 0.1 s, {err[-1]:+.1e} at 1 s")
print("a full-order observer would have 2 states and two poles to place; the reduced one estimates only what is not measured")"""),
         md(r"""
When part of the state is measured directly, only the rest needs estimating: the reduced-order observer has
as many states as unmeasured variables (one here, the current), uses the measured speed's derivative
implicitly through the change of variable z = î − L_r ω, and converges at its single pole (from a wild initial
value - the gain L_r is large - to 10⁻⁴ A within a second). It is smaller and
avoids filtering a measurement that is already clean; the full-order observer is simpler to derive and
smooths the measurement as a by-product.
""")],
        [code(r"""cp = models.cart_pendulum()["system"]
A, B, C = cp.A, cp.B, cp.C
K = sp.place(A, B, [-2 + 2j, -2 - 2j, -5.0, -6.0])
L = sp.observer_gain(A, C, [-10.0, -11.0, -12.0, -13.0])
comp = sy.ss(A - B @ K - L @ C, L, -K, np.zeros((1, 2)))        # y (2 outputs) -> u
# closed loop by state-space interconnection: u = comp(y), y = plant(u)
Acl = np.block([[A, B @ comp.C], [comp.B @ C, comp.A]])
print("closed-loop eigenvalues:", np.round(np.sort_complex(np.linalg.eigvals(Acl)), 4))
print("= union of the designed sets:", np.allclose(np.sort_complex(np.linalg.eigvals(Acl)), np.sort_complex(np.r_[np.linalg.eigvals(A - B @ K), np.linalg.eigvals(A - L @ C)])))"""),
         md(r"""
The compensator from y to u is the observer with u = −Kx̂ substituted; closing it around the plant gives an
8th-order system whose eigenvalues are the four controller poles and the four observer poles - the
separation principle holds in the compensator form too, as it must, since the two forms differ only by a
change of variables (x, x̂) → (x, x − x̂).
""")],
    ]


@solutions("10_the_linear_quadratic_regulator", imports=["from englincontrol import systems as sy, statespace as sp, linalg as la, response as rs, design as dg, frequency as fr, models"])
def _():
    return [
        [code(r"""cp = models.cart_pendulum()["system"]
A, B, C = cp.A, cp.B, cp.C
Q = np.diag(1 / np.array([0.1, 0.5, 0.05, 0.5]) ** 2)                 # Bryson: 1 / (allowed excursion)^2
R0 = 1 / 5.0**2
for R in (R0, 4 * R0, 16 * R0, 64 * R0):
    r = sp.lqr(A, B, Q, np.array([[R]]))
    Acl = sy.ss(A - B @ r["K"], B, C, np.zeros((2, 1)))
    t, y, x = rs.initial(Acl, [0, 0, np.radians(10), 0], 5.0, 5001)
    u = -(x @ r["K"].T)[:, 0]
    i_settle = min(np.max(np.flatnonzero(np.abs(x[:, 2]) > 0.002)) + 1, len(t) - 1)
    print(f"R = {R / R0:.0f} x Bryson: peak force {np.max(np.abs(u)):.2f} N, cart excursion {100 * np.max(np.abs(x[:, 0])):.1f} cm, angle settled in {t[i_settle]:.2f} s" + (" (not within 5 s)" if i_settle == len(t) - 1 else ""))"""),
         md(r"""
Bryson's rule gives a first weighting from the allowed excursions; the 10° recovery it produces asks for
30 N, six times the budget, and raising R - cheaper to tell the regulator that force is expensive than to
re-weight every state - brings the peak force down by about half per factor of four, under 5 N at 64 times
Bryson's R, at the cost of a larger cart excursion and a slower recovery. Three or four such iterations are
the normal LQR design loop; the rule is a starting point, not a result.
""")],
        [code(r"""mot = models.dc_motor()["system"]
A, B, C = mot.A, mot.B, mot.C
def recovery(td, yd):
    out = np.flatnonzero(np.abs(yd) > 0.02 * np.max(np.abs(yd)))
    return td[min(np.max(out) + 1, len(td) - 1)]
rows = []
for qi in (100.0, 1000.0, 10000.0):
    li = sp.lqr_integral(A, B, C, np.diag([1.0, 10.0]), np.array([[0.01]]), qi=qi)
    Aa, Ba = li["Aa"], li["Ba"]; Ka = np.hstack([li["K"], li["Ki"]])
    Acl = Aa - Ba @ Ka
    cl_r = sy.ss(Acl, np.array([[0.0], [0.0], [1.0]]), np.hstack([C, [[0.0]]]), 0.0)         # reference -> speed
    cl_d = sy.ss(Acl, np.vstack([B, [[0.0]]]), np.hstack([C, [[0.0]]]), 0.0)               # a disturbance at the plant input (voltage)
    t, y = rs.step(cl_r, 4.0); td, yd = rs.step(cl_d, 4.0)
    info = rs.step_info(t, y, final=1.0)
    rows.append((f"LQR+I qi={qi:.0f}", info["overshoot"], info["settling_time"], np.max(np.abs(yd)), recovery(td, yd)))
# a PI tuned by SIMC on the FOPDT fit of the motor
t_, y_ = rs.step(mot, 3.0, 3001)
fo = dg.fopdt_from_step(t_, y_)
pi = dg.imc_pid(fo["K"], fo["tau"], max(fo["L"], 0.02), lam=0.1)
cl = dg.closed_loop(mot.to_tf(), dg.pid_standard(pi["Kp"], pi["Ti"]))
t, y = rs.step(cl["T"], 4.0); info = rs.step_info(t, y, final=1.0)
td, yd = rs.step(cl["PS"], 4.0)                                                           # the same input disturbance, through PS
rows.append(("PI (SIMC, lambda = 0.1)", info["overshoot"], info["settling_time"], np.max(np.abs(yd)), recovery(td, yd)))
print(pd.DataFrame(rows, columns=["controller", "overshoot %", "settling (s)", "peak speed deviation (per V of disturbance)", "disturbance recovered (s)"]).to_string(index=False, float_format="%.3f"))"""),
         md(r"""
Raising the integral weight speeds up the rejection of an input disturbance and the removal of the offset,
with a growing overshoot on the setpoint - the same trade-off as a PI's integral time, but with the state
feedback on current and speed providing damping that a PI must obtain from its proportional gain alone. The
LQR-with-integral loop holds the speed deviation smaller and recovers sooner than the PI of comparable
settling time; it also uses the current measurement, which the PI does not.
""")],
        [code(r"""cp = models.cart_pendulum()["system"]
A, B, C = cp.A, cp.B, cp.C
Q, R = np.diag([10.0, 1.0, 100.0, 1.0]), np.array([[0.1]])
x0 = np.array([0.0, 0.0, 0.2, 0.0])
r = sp.lqr(A, B, Q, R)
def cost(K):
    Acl = sy.ss(A - B @ K, B, C, np.zeros((2, 1)))
    t, y, x = rs.initial(Acl, x0, 20.0, 20001)
    u = -(x @ K.T)
    integrand = np.einsum("ij,jk,ik->i", x, Q, x) + np.einsum("ij,jk,ik->i", u, R, u)
    return np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))
print(f"LQR gain: simulated cost {cost(r['K']):.4f} vs x0' P x0 = {sp.cost_to_go(r['P'], x0):.4f}")
for des in ([-2 + 2j, -2 - 2j, -5.0, -6.0], [-1 + 1j, -1 - 1j, -3.0, -4.0], [-4 + 4j, -4 - 4j, -10.0, -12.0]):
    Kp_ = sp.place(A, B, des)
    print(f"pole placement at {np.round(des, 1)}: cost {cost(Kp_):.4f} (larger, as optimality requires)")"""),
         md(r"""
The simulated cost of the LQR gain equals x₀ᵀPx₀ to the quadrature's accuracy, and every placed-pole gain
costs more - slower ones through a large state integral, faster ones through control effort. The Riccati
solution is the minimum over *all* stabilising gains, which is why it is called optimal.
""")],
    ]


@solutions("11_kalman_filtering_and_lqg", imports=["from englincontrol import systems as sy, statespace as sp, discrete as dc, models"])
def _():
    return [
        [code(r"""mot = models.dc_motor()["system"]
dt = 0.005
Gd = dc.c2d(mot, dt); Ad, Bd, Cd = Gd.A, Gd.B, Gd.C
Qd, Rd_true = np.diag([1e-6, 1e-4]), 0.25
rng = np.random.default_rng(3)
N = 6000
u = 6.0 * (1 + np.sign(np.sin(0.5 * np.arange(N) * dt)))
x = np.zeros(2); X = np.zeros((N, 2)); Y = np.zeros(N)
for k in range(N):
    X[k] = x; Y[k] = (Cd @ x)[0] + np.sqrt(Rd_true) * rng.normal()
    x = Ad @ x + Bd[:, 0] * u[k] + rng.normal(size=2) * np.sqrt(np.diag(Qd))
for factor, label in ((0.1, "over-confident (R/10)"), (1.0, "correct"), (10.0, "over-cautious (10 R)")):
    kd = sp.kalman(Ad, Cd, Qd, np.array([[Rd_true * factor]]), discrete=True)
    Xh = sp.kalman_filter_run(Ad, Bd, Cd, kd["L"], u, Y)
    e = Xh[500:] - X[500:]
    print(f"{label:<22}: speed error variance {np.var(e[:, 1]):.5f} (filter predicts {kd['P'][1, 1]:.5f}), current error variance {np.var(e[:, 0]):.2e} (predicts {kd['P'][0, 0]:.2e})")"""),
         md(r"""
An over-confident filter (told the measurement is cleaner than it is) weights it too much and lets noise
into the estimate: its actual error exceeds its own prediction several times over. An over-cautious one
trusts the model and lags the truth less visibly here, because this model is good. Measured error above the
predicted one is the signature of a filter that believes its sensor too much; it is the worse direction,
because the filter then also reports a confidence it does not have.
""")],
        [code(r"""from englincontrol import robust as rb, response as rs
cp = models.cart_pendulum()["system"]
Q, R = np.diag([10.0, 1.0, 100.0, 1.0]), np.array([[0.1]])
rng = np.random.default_rng(0)
K_lqr = sp.lqr(cp.A, cp.B, Q, R)["K"]
print(f"state feedback: closest approach to -1: {rb.nyquist_distance(sy.ss(cp.A, cp.B, K_lqr, np.zeros((1, 1))).to_tf())['distance']:.3f}")
rows = []
for scale in (1e-6, 1e-3, 1.0, 1e2):
    g = sp.lqg(cp, Q, R, scale * np.eye(4), 1e-4 * np.eye(2))
    comp = g["compensator"]
    L_in = sy.series(cp, sy.ss(comp.A, comp.B, -comp.C, np.zeros((1, 2)))).to_tf()
    d = rb.nyquist_distance(L_in)
    t = np.linspace(0, 1, 2001)
    _, u_noise, _ = rs.lsim(comp, 1e-3 * rng.normal(size=(2001, 2)), t)          # 1 mrad / 1 mm of measurement noise on both outputs
    rows.append((scale, f"{np.min(g['estimator_poles'].real):.0f} .. {np.max(g['estimator_poles'].real):.1f}", d["distance"], np.std(u_noise)))
print(pd.DataFrame(rows, columns=["process noise scale", "estimator poles (Re)", "closest approach to -1", "control noise rms (N)"]).to_string(index=False, float_format="%.3f"))"""),
         md(r"""
Scaling up the assumed process noise makes the Kalman filter trust the measurement more and its poles faster;
the loop's distance from −1 recovers from 0.21 towards the state-feedback value of 1 - loop transfer
recovery - but saturates at 0.5 while the control signal's response to measurement noise keeps growing, by
twentyfold across the sweep. The recovery stops paying at a process-noise scale of about 1 here; and it can
never complete, because the cart position is a non-minimum-phase output of this plant, for which the
recovery theorem does not hold.
""")],
        [code(r"""mot = models.dc_motor()["system"]
dt = 0.005
Gd = dc.c2d(mot, dt); Ad, Cd = Gd.A, Gd.C
Qd, Rd = np.diag([1e-6, 1e-4]), np.array([[0.25]])
P = np.zeros((2, 2)); gains = []
for k in range(400):
    L = Ad @ P @ Cd.T @ np.linalg.inv(Cd @ P @ Cd.T + Rd)
    P = Ad @ P @ Ad.T - Ad @ P @ Cd.T @ np.linalg.inv(Cd @ P @ Cd.T + Rd) @ Cd @ P @ Ad.T + Qd
    gains.append(L.ravel())
gains = np.array(gains)
Lss = sp.kalman(Ad, Cd, Qd, Rd, discrete=True)["L"].ravel()
print("gain after 50, 200, 400 steps:", np.round(gains[49], 5), np.round(gains[199], 5), np.round(gains[399], 5), " steady state:", np.round(Lss, 5))
print(f"difference at step 400: {np.max(np.abs(gains[399] - Lss)):.1e}")"""),
         md(r"""
The time-varying filter's Riccati recursion starts from any P₀ and converges to the steady-state solution of
the discrete algebraic Riccati equation; its gain converges with it, within a few hundred samples here. The
steady-state filter is this limit, and is what one implements when the transient of the first seconds does not
matter.
""")],
    ]


@solutions("12_sampling_and_discretisation", imports=["from englincontrol import systems as sy, discrete as dc, response as rs, linalg as la, models"])
def _():
    return [
        [code(r"""G = sy.tf([4.0], [1.0, 1.2, 4.0])
for dt in (0.1, 0.5, 1.0, 1.5):
    n = int(round(6.0 / dt)) + 1
    t, y = rs.step(G, 6.0, n)
    err = {}
    for meth in ("zoh", "tustin"):
        Gd = dc.c2d(G, dt, meth)
        td, yd = rs.step(Gd, 6.0)
        err[meth] = (np.max(np.abs(yd[: len(y)] - y[: len(yd)])), bool(np.all(np.abs(Gd.poles()) < 1)))
    print(f"dt = {dt}: ZOH error at the samples {err['zoh'][0]:.1e} (stable {err['zoh'][1]}), Tustin error {err['tustin'][0]:.2f} (stable {err['tustin'][1]})")"""),
         md(r"""
The zero-order-hold model is exact at the samples at *every* Δt, because that is what the hold does to a
step; Tustin's model is only approximate, and its error grows with Δt - but it stays stable whatever Δt,
because the bilinear map sends the left half-plane into the unit disc. ZOH for plants, Tustin for controllers
remains the rule.
""")],
        [code(r"""for dt in (0.1, 0.5, 1.0):
    Gd = dc.c2d(sy.tf([1.0], [1.0, 0.0, 0.0, 0.0]), dt)
    z = Gd.zeros()
    print(f"triple integrator at dt = {dt}: discrete zeros {np.round(z, 4)}, |z| = {np.round(np.abs(z), 4)}")"""),
         md(r"""
Sampling 1/s³ with a hold creates two zeros at z = −2 ± √3 - one at −3.73, outside the unit circle, for every
Δt (Åström, Hagander and Sternby 1984): a discrete non-minimum-phase zero of a plant that has none in
continuous time. Any controller that cancels or inverts the plant (dead-beat, model-inverse feedforward)
would have an unstable pole at that zero; the sampling zeros are why such designs fail at fast sampling for
plants of relative degree three or more.
""")],
        [code(r"""mot = models.dc_motor()["system"]
A, B = mot.A, mot.B
for dt in (0.01, 1.0):
    Phi = np.eye(2); Gam = np.zeros((2, 1)); term = np.eye(2)
    for k in range(1, 21):
        Gam = Gam + term @ B * dt**k / float(np.prod(np.arange(1, k + 1)))
        term = term @ A * dt / k
        Phi = Phi + term
    Phi_ref, Gam_ref = la.expm_with_integral(A, B, dt)
    print(f"dt = {dt}: series (20 terms) vs library: Phi error {np.max(np.abs(Phi - Phi_ref)):.1e}, Gamma error {np.max(np.abs(Gam - Gam_ref)):.1e}  (|A dt| = {np.linalg.norm(A * dt, 1):.1f})")"""),
         md(r"""
For a small step the truncated series is exact to rounding; at Δt = 1 the matrix AΔt has norm 27 and twenty
terms are nowhere near enough - the series is still growing where it should be cancelling. The library's
scaling-and-squaring approach (notebook 18) removes the dependence on the step size.
""")],
    ]


@solutions("13_digital_control_design", imports=["from englincontrol import systems as sy, discrete as dc, design as dg, statespace as sp, response as rs, frequency as fr, models"])
def _():
    return [
        [code(r"""mot = models.dc_motor(position=True)["system"]
dt = 0.02
Gd = dc.c2d(mot, dt); Ad, Bd, Cd = Gd.A, Gd.B, Gd.C
# augment with a discrete integrator of the tracking error: xi[k+1] = xi[k] + dt (r - y)
Aa = np.block([[Ad, np.zeros((3, 1))], [-dt * Cd, np.eye(1)]]); Ba = np.vstack([Bd, [[0.0]]])
Qa = np.diag([1.0, 10.0, 1000.0, 1e5]); Ra = np.array([[0.1]])
r = sp.lqr(Aa, Ba, Qa, Ra, discrete=True); Ka = r["K"]
Acl = Aa - Ba @ Ka
cl_r = sy.ss(Acl, np.array([[0.0], [0.0], [0.0], [dt]]), np.hstack([Cd, [[0.0]]]), 0.0, dt)
t, y = rs.step(cl_r, 3.0)
cl_d = sy.ss(Acl, np.vstack([Bd, [[0.0]]]), np.hstack([Cd, [[0.0]]]), 0.0, dt)     # a torque-like disturbance through the input
td, yd = rs.step(cl_d, 3.0)
info = rs.step_info(t, y, final=1.0)
print(f"discrete LQR + integral: step overshoot {info['overshoot']:.1f} %, settling {info['settling_time']:.2f} s, final {y[-1]:.4f}; disturbance step rejected to {yd[-1]:.1e} (integral action)")
C = dg.design_lead(mot.to_tf(), 5.0, 50.0)["controller"]; Cz = dc.c2d(C, dt, "tustin")
Pd = dc.c2d(mot.to_tf(), dt); Td = sy.feedback(sy.series(Cz, Pd)); PSd = sy.feedback(Pd, Cz)
t2, y2 = rs.step(Td, 3.0); td2, yd2 = rs.step(PSd, 3.0)
print(f"emulated lead: step overshoot {rs.step_info(t2, y2, final=1.0)['overshoot']:.1f} %, settling {rs.step_info(t2, y2, final=1.0)['settling_time']:.2f} s; disturbance leaves a steady offset {yd2[-1]:.3f} (no integral action)")"""),
         md(r"""
The discrete LQR with an integrator state removes the offset a load disturbance would leave (the lead
compensator leaves one, having no integrator beyond the plant's own, which is before the disturbance
enters). Direct discrete design on the ZOH model with the augmented state is the digital counterpart of
notebook 10's integral LQR, and needs no continuous design at all.
""")],
        [code(r"""mot = models.dc_motor(position=True)["system"]
for dt in (0.02, 0.1):
    Gd = dc.c2d(mot, dt); Ad, Bd = Gd.A, Gd.B
    # one sample of computation delay: u applied at k+1 -> add a state holding the last u
    Adel = np.block([[Ad, Bd], [np.zeros((1, 3)), np.zeros((1, 1))]]); Bdel = np.vstack([np.zeros((3, 1)), [[1.0]]])
    for s_poles in ([-8 + 6j, -8 - 6j, -20.0], [-16 + 12j, -16 - 12j, -40.0]):
        z_poles = np.exp(np.array(s_poles) * dt)
        K0 = sp.place(Ad, Bd, z_poles)
        Kd = sp.place(Adel, Bdel, np.r_[z_poles, 0.0])                   # the delay state's pole at 0 (dead-beat on the delay)
        ok0 = np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K0))) < 1
        print(f"dt = {dt}, target poles {np.round(s_poles, 0)}: delay-free gain |K| = {np.linalg.norm(K0):.1f}; with the delay state |K| = {np.linalg.norm(Kd):.1f}, max |z| = {np.max(np.abs(np.linalg.eigvals(Adel - Bdel @ Kd))):.4f}")"""),
         md(r"""
Modelling the computation delay as an extra state lets the pole placement account for it exactly, at the
price of larger gains, which grow with the target speed and with the sampling period: at Δt = 0.1 and the
faster targets the delayed design needs gains several times the delay-free ones. A delay of one sample is a
larger fraction of the loop's time scale when the samples are long - the same lesson as the half-sample delay
of the hold.
""")],
        [code(r"""dt = 0.02
Cz = dc.pid_discrete(3.0, 2.0, 0.3, dt)
b, a = Cz.num, Cz.den                                      # u[k] = (b0 e[k] + b1 e[k-1] + b2 e[k-2] - a1 u[k-1] - a2 u[k-2]) / a0
e = np.ones(200); u = np.zeros(200)
for k in range(200):
    acc = sum(b[i] * e[k - i] for i in range(len(b)) if k - i >= 0) - sum(a[i] * u[k - i] for i in range(1, len(a)) if k - i >= 0)
    u[k] = acc / a[0]
t, u_lib = rs.step(Cz, (200 - 1) * dt)
print(f"difference equation vs the library's step response of C(z): {np.max(np.abs(u - u_lib)):.1e}")
print("coefficients: b =", np.round(b, 4), " a =", np.round(a, 4))"""),
         md(r"""
The transfer function's coefficients are the difference equation's: the numerator weights the error's history,
the denominator the control's own. Four multiplications and a few additions per sample - which is what runs
on the microcontroller, and why the discretised controller, not the continuous design, is the one to test.
""")],
    ]


@solutions("14_robustness_and_limitations", imports=["from englincontrol import systems as sy, robust as rb, frequency as fr, design as dg, polynomials as pl, models"])
def _():
    return [
        [code(r"""ml = models.magnetic_levitation()
P = ml["system"].to_tf()
P = sy.tf(-P.num, P.den)                      # the current's effect on the gap is negative: flip the input sign so that the plant has positive gain (phase -180)
p_un = ml["unstable_pole"]
rows = []
for wc in (50.0, 100.0, 200.0):
    d = dg.design_lead(P, wc, 45.0)
    L = sy.series(d["controller"], P)
    g4 = rb.gang_of_four(P, d["controller"])
    w = np.logspace(0, 4, 3000)
    stable = bool(np.all(sy.feedback(L).poles().real < 0))
    rows.append((wc, wc / p_un, stable, rb.sensitivity_peak(L)["Ms"] if stable else np.nan, d["margins"]["phase_margin"], np.max(np.abs(fr.freqresp(g4["CS"], w)))))
print(pd.DataFrame(rows, columns=["crossover", "w_c / p", "closed loop stable", "Ms", "PM (deg)", "peak |CS|"]).to_string(index=False, float_format="%.2f"))"""),
         md(r"""
An unstable pole at p demands a crossover above about 2p: at ω_c = 50 (just above p = 44) the lead design
does not even stabilise the loop - the Nyquist curve fails to encircle −1 - while at 100 and 200 the loop is
comfortable (M_s = 1.4) - and the price shows in the peak of CS, the noise amplification, which quadruples
each time the crossover doubles. Unstable plants force bandwidth, and bandwidth costs noise: a levitation
sensor must be clean. (The plant's input gain is negative; the sign is flipped so that the design works with
the usual −180° convention, which `design_lead` expects.)
""")],
        [code(r"""Pz = sy.tf([-1.0, 2.0], [1.0, 2.0, 1.0])
rows = []
for K in np.linspace(0.1, 1.5, 15):
    C = sy.tf([K, 0.5 * K], [1.0, 0.0])
    L = sy.series(C, Pz)
    m = fr.margins(L)
    rows.append((K, rb.sensitivity_peak(L)["Ms"], m["w_gc"]))
rows = np.array(rows)
print("Ms rises monotonically with the gain:", np.round(rows[:, 1], 2))
i16 = np.argmax(rows[:, 1] > 1.6)
print(f"a design target Ms <= 1.6 allows K up to about {rows[i16 - 1, 0]:.2f}, where the crossover is {rows[i16 - 1, 2]:.3f} rad/s = z/{2.0 / rows[i16 - 1, 2]:.1f}; at the z/2 crossover (K = {rows[np.argmin(np.abs(rows[:, 2] - 1.0)), 0]:.2f}) Ms is already {rows[np.argmin(np.abs(rows[:, 2] - 1.0)), 1]:.2f}")"""),
         md(r"""
There is no interior optimum: M_s rises monotonically with the gain, because every increase pushes the
crossover towards the zero whose phase lag drives the curve towards −1. The z/2 rule is therefore a ceiling,
not a target - at a crossover of z/2 the loop already has M_s = 2.6 - and a robust design with M_s ≤ 1.6
sits at about a quarter of the zero. Right-half-plane zeros cost more bandwidth than the rule of thumb
suggests.
""")],
        [code(r"""L = sy.tf([30.0, 60.0], pl.from_roots([1, -5, -8]))
for w_max in (1e2, 1e3, 1e5):
    bi = rb.bode_sensitivity_integral(L, w_max=w_max, n=400000)
    print(f"integral of ln|S| up to {w_max:.0e} rad/s: {bi['integral']:.4f} (theory pi = {np.pi:.4f}, missing {100 * (1 - bi['integral'] / np.pi):.1f} %)")"""),
         md(r"""
ln|S| decays like −|L| at high frequency, and |L| of a relative-degree-2 loop decays like 1/ω²: the tail of
the integral is of order 1/ω_max and a cut-off at 100 rad/s misses a visible fraction. The integral is an
exact identity only over the whole axis - a numerical check must go far enough out.
""")],
    ]


@solutions("15_system_identification", imports=["from scipy import optimize as sopt", "from englincontrol import systems as sy, identify as idf, discrete as dc, response as rs, frequency as fr, datasets, models"])
def _():
    return [
        [code(r"""p = pd.read_csv(datasets.path("motor_prbs.csv"), comment="#")
p = p[p.t_s >= 2.0]
true = models.dc_motor(R=2.0, L=0.1, Kt=0.05, Ke=0.05, J=0.002, b=0.005)["system"]
u = (p.V - p.V.mean()).to_numpy(); y = (p.speed_rad_s - p.speed_rad_s.mean()).to_numpy()
true_z = np.sort(np.exp(true.poles().real * 0.005))
rows = []
for T in (10.0, 40.0):
    n = int(T / 0.005)
    for meth in ("ls", "iv"):
        m = idf.arx(u[:n], y[:n], 2, 2, 1, dt=0.005, method=meth)
        rows.append((T, meth, np.round(np.sort(m["tf"].poles().real), 4).tolist(), m["tf"].dcgain()))
oe = idf.output_error(u[::10], y[::10], 2, 0.05)
print(pd.DataFrame(rows, columns=["record (s)", "method", "poles", "dc gain"]).to_string(index=False, float_format="%.3f"))
print(f"true poles at 200 Hz: {np.round(true_z, 4)}; output-error at 20 Hz mapped to 200 Hz: {np.round(np.sort(np.exp(np.log(oe['tf'].poles().real) / 10)), 4)}")"""),
         md(r"""
The instrumental-variable estimate removes the systematic bias of least squares but at 200 Hz its variance is
enormous: the regressors are so nearly collinear that any consistent estimator needs far more data than 40 s
provides, and the estimates wander. The output-error fit at 20 Hz gets both poles to three digits from the
same record. Consistency is a statement about infinite data; the practical question is how fast the variance
falls, and here decimation does more than a better estimator.
""")],
        [code(r"""s_ = pd.read_csv(datasets.path("servo_frequency_response.csv"), comment="#")
w = s_.w_rad_s.to_numpy(); H = 10 ** (s_.gain_dB.to_numpy() / 20) * np.exp(1j * np.radians(s_.phase_deg.to_numpy()))
rows = []
for n_num, n_den in ((0, 3), (2, 5), (2, 6)):
    fit = idf.fit_frequency_response(w, H, n_num, n_den)
    err = np.abs(fr.freqresp(fit, w) / H)
    rows.append((f"({n_num}, {n_den})", np.max(np.abs(20 * np.log10(err))), np.round(fit.poles()[np.argsort(np.abs(fit.poles()))][-2:], 1).tolist(), bool(np.all(fit.poles().real < 1e-3))))
P3 = fr.pade(0.003, 2)
Hc = H / fr.freqresp(P3, w)                                      # divide the delay out before fitting
fit_d = idf.fit_frequency_response(w, Hc, 2, 5)
rows.append(("(2, 5) with Pade(2) delay", np.max(np.abs(20 * np.log10(np.abs(fr.freqresp(fit_d, w) * fr.freqresp(P3, w) / H)))), np.round(fit_d.poles()[np.argsort(np.abs(fit_d.poles()))][-2:], 1).tolist(), bool(np.all(fit_d.poles().real < 1e-3))))
print(pd.DataFrame(rows, columns=["orders (num, den)", "max gain error (dB)", "fastest poles", "stable"]).to_string(index=False, float_format="%.2f"))"""),
         md(r"""
None of the unstructured fits works: the third-order one misses the resonance by 48 dB, the higher orders
are unstable (Levy-type fits make no promise of stability) and no better at the peak, and dividing a Padé
delay out of the data does not rescue them - a lightly damped resonance three decades above the slowest
pole, plus a delay, is more than a free rational fit can resolve from 60 noisy points. The physically
structured model of notebook 19 - motor, integrator, one resonance, one delay - has four parameters that
mean something, fits to a third of a decibel, and generalises beyond the measured frequencies. Structure
first, then numbers.
""")],
        [code(r"""Gtrue = dc.c2d(sy.tf([2.0], [1.0, 3.0, 2.0]), 0.1)
rng = np.random.default_rng(0)
u = idf.prbs(2000, seed=1, hold=10); _, y, _ = rs.lsim(Gtrue, u); y = y + 0.01 * rng.normal(size=len(y))
def cost(theta):
    p1, p2, g = theta
    num = [0.0, g * (1 - p1) * (1 - p2)]                           # unit dc gain times g, one-sample delay
    G = sy.tf(num, np.polymul([1.0, -p1], [1.0, -p2]), dt=0.1)
    return np.sum((rs.lsim(G, u)[1] - y) ** 2)
res = sopt.minimize(cost, [0.5, 0.95, 1.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})
oe = idf.output_error(u, y, 2, 0.1)
print(f"Nelder-Mead on the simulation error: poles {np.round(np.sort(res.x[:2]), 5)}, gain {res.x[2]:.4f}; library output_error: poles {np.round(np.sort(oe['tf'].poles().real), 5)}, dc gain {oe['tf'].dcgain():.4f}; true poles {np.round(np.sort(Gtrue.poles().real), 5)}, gain 1.0000")"""),
         md(r"""
The output-error problem is a nonlinear least-squares problem in the model's parameters, and any minimiser
works given a reasonable start - but the three-parameter model (two poles and a gain, no zero) cannot
represent the ZOH model, which has a zero at −0.96, so its poles shift to compensate (0.767 and 0.914 against
0.819 and 0.905) while its simulation error is nearly as small. The library's four-parameter fit matches the
truth to three digits. A model class that cannot contain the truth gives biased parameters however good the
optimiser; the parametrisation by poles has the merit of staying stable by construction.
""")],
    ]


@solutions("16_model_reduction_and_realisation", imports=["from englincontrol import systems as sy, linalg as la, identify as idf, response as rs, frequency as fr, mimo, discrete as dc, models"])
def _():
    return [
        [code(r"""qc = models.quarter_car()["system"]
red = idf.balanced_truncation(qc, 2)
print("Hankel singular values:", np.round(red["hankel_singular_values"], 2), "-> the two largest carry one mode; the drop to the third is a factor of", round(red["hankel_singular_values"][1] / red["hankel_singular_values"][2], 1))
w = np.logspace(-1, 3, 400)
sv_full, sv_red = mimo.sigma(qc, w), mimo.sigma(red["system"], w)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.loglog(w, sv_full[:, 0], "k", label="full, sigma_max"); ax.loglog(w, sv_full[:, 1], "k--", label="full, sigma_min"); ax.loglog(w, sv_red[:, 0], "r", label="reduced (2 states), sigma_max"); ax.loglog(w, sv_red[:, 1], "r--"); ax.legend(fontsize=8); ax.set_xlabel("w (rad/s)"); plt.show()
print(f"the reduced model keeps the poles {np.round(red['system'].poles(), 2)} - the wheel-hop pair - and drops the body-bounce mode near {np.abs(qc.poles()).min():.0f} rad/s: it fails below about 20 rad/s")"""),
         md(r"""
Balanced truncation keeps the mode that dominates the input-output *energy*, and with road height as an
input and body acceleration as an output that is the wheel-hop mode at 64 rad/s, not the body bounce one
would have guessed: the reduced model is faithful above 20 rad/s and wrong at the body resonance, exactly
where a ride-comfort controller works. Reduction is relative to the inputs and outputs it is asked to
preserve; scaling them (acceleration in g, road in cm) or choosing frequency-weighted reduction changes the
answer. The Hankel singular values are the honest guide, but the designer must read them against the purpose.
""")],
        [code(r"""mot = models.dc_motor()["system"]
Gd = dc.c2d(mot, 0.01)
rng = np.random.default_rng(0)
t, ys = rs.step(Gd, 2.0)
markov_full = np.r_[ys[0], np.diff(ys)]
for n_mk in (10, 30, 100):
    sv = idf.era(markov_full[:n_mk] + 2e-5 * rng.normal(size=n_mk), 2, 0.01)["hankel_singular_values"][:5]
    print(f"{n_mk:3d} Markov parameters: singular values {np.round(sv, 5)}, gap sigma2/sigma3 = {sv[1] / sv[2]:.1f}")"""),
         md(r"""
With few Markov parameters the Hankel matrix is small and its noise-induced singular values are comparable
to the true second one: the order is unreadable. As the record grows the true singular values stabilise and
the noise ones shrink (they average down), and the gap between the second and third opens. Order
determination needs a record long enough for that gap to appear - and the gap, not a threshold, is the
criterion.
""")],
        [code(r"""Gd = dc.c2d(models.dc_motor()["system"], 0.01)
t, ys = rs.step(Gd, 1.0)
h = np.r_[ys[0], np.diff(ys)]                                     # Markov parameters h_0, h_1, ...
r = 40; c = len(h) - 1 - r
Hk = np.array([[h[i + j + 1] for j in range(c)] for i in range(r)])
Hk1 = np.array([[h[i + j + 2] for j in range(c)] for i in range(r)])
U, s, Vt = np.linalg.svd(Hk)
n = 2
Ur, Sr, Vr = U[:, :n], np.diag(s[:n]), Vt[:n].T
Sih = np.diag(s[:n] ** -0.5)
A = Sih @ Ur.T @ Hk1 @ Vr @ Sih; B = (np.sqrt(Sr) @ Vr.T)[:, :1]; C = (Ur @ np.sqrt(Sr))[:1, :]
era_sys = sy.ss(A, B, C, h[0], 0.01)
print("ERA by hand: poles", np.round(np.sort(era_sys.poles().real), 6), " ZOH model:", np.round(np.sort(Gd.poles().real), 6))
print("transfer functions:", era_sys.to_tf(), " vs ", Gd.to_tf())"""),
         md(r"""
The Hankel matrix factors as the product of the observability and controllability matrices; its SVD splits
them symmetrically, the shifted Hankel matrix yields A, and the first block column and row give B and C. The
realisation reproduces the ZOH model's poles and transfer function to rounding - Ho and Kalman's 1966
construction, which the library's `era` implements.
""")],
    ]


@solutions("17_multivariable_basics", imports=["from englincontrol import systems as sy, mimo, frequency as fr, response as rs, design as dg, models"])
def _():
    return [
        [code(r"""col = models.distillation_2x2()
G = col["system"]; G0 = G.dcgain(); D = mimo.static_decoupler(G0)
def closed_loop_2x2(plant, controllers, decoupler=None):
    C1, C2 = controllers
    s1, s2 = C1.to_ss(), C2.to_ss()
    Cs = sy.ss(np.block([[s1.A, np.zeros((s1.n, s2.n))], [np.zeros((s2.n, s1.n)), s2.A]]), np.block([[s1.B, np.zeros((s1.n, 1))], [np.zeros((s2.n, 1)), s2.B]]),
               np.block([[s1.C, np.zeros((1, s2.n))], [np.zeros((1, s1.n)), s2.C]]), np.block([[s1.D, np.zeros((1, 1))], [np.zeros((1, 1)), s2.D]]))
    if decoupler is not None:
        Cs = sy.ss(Cs.A, Cs.B, decoupler @ Cs.C, decoupler @ Cs.D)
    return sy.feedback(sy.series(Cs, plant))
rows = []
for lam in (5.0, 10.0, 20.0):
    C1 = dg.pid_standard(16.7 / (12.8 * lam), min(16.7, 4 * lam)); C2 = dg.pid_standard(-14.4 / (19.4 * lam), min(14.4, 4 * lam))
    C1d = dg.pid_standard(16.7 / lam, min(16.7, 4 * lam)); C2d = dg.pid_standard(14.4 / lam, min(14.4, 4 * lam))   # retuned for the decoupled plant G D (unit gains)
    for name, ctrls, dec in (("decentralised", (C1, C2), None), ("decoupled", (C1d, C2d), D)):
        T = closed_loop_2x2(G, ctrls, dec)
        t, y = rs.step(T, 150.0, input_index=0)
        info = rs.step_info(t, y[:, 0], final=1.0)
        rows.append((lam, name, bool(np.max(T.poles().real) < 0), info["settling_time"], np.max(np.abs(y[:, 1]))))
print(pd.DataFrame(rows, columns=["lambda (min)", "structure", "stable", "settling of y1 (min)", "peak |y2| (interaction)"]).to_string(index=False, float_format="%.3f"))"""),
         md(r"""
Decoupling cuts the interaction to a third at every tuning and makes the loops faster for the same λ - once
the controllers are retuned for the decoupled plant, whose dc gains are +1 on both channels (with the raw
controllers, the steam loop's negative gain turns into positive feedback through the decoupler and the loop
is slowly unstable). Static decoupling never hurt here, but it can: a decoupler inverts the dc gain, and when
that matrix is ill-conditioned the decoupler amplifies model errors (the column's condition number of 7.5 is
benign; the original Wood-Berry column with its delays is less so).
""")],
        [code(r"""qc = models.quarter_car()["system"]
w = np.logspace(-1, 2, 400)
sv = mimo.sigma(qc, w)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.loglog(w, sv[:, 0], label="sigma_max"); ax.loglog(w, sv[:, 1], label="sigma_min"); ax.legend(); ax.set_xlabel("w (rad/s)"); plt.show()
i1, i2 = np.argmax(sv[:, 0] * (w < 20)), np.argmax(sv[:, 0] * (w > 20))
print(f"peaks of the largest singular value near {w[i1]:.1f} rad/s (body bounce) and {w[i2]:.1f} rad/s (wheel hop)")
U, s, Vt = np.linalg.svd(fr.freqresp(qc, w[i2]))
print(f"at wheel hop the most amplified input direction is {np.round(np.abs(Vt[0]), 3)} in [road, actuator force] - the road input dominates: an actuator alone cannot cancel wheel hop")"""),
         md(r"""
The two peaks of the largest singular value are the suspension's two resonances; the right singular vector
at the wheel-hop peak points almost entirely along the road input, meaning the road excites that mode far
more than the actuator can act on it - the input direction that is hardest to attenuate. Singular values and
their vectors turn a 2 × 2 frequency response into a statement about which disturbances a controller can
fight.
""")],
        [code(r"""col = models.distillation_2x2()
TM = mimo.transfer_matrix(col["system"])
from englincontrol import polynomials as pl
N = [[TM[i][j].num for j in range(2)] for i in range(2)]; Dn = [[TM[i][j].den for j in range(2)] for i in range(2)]
# det G = (N11 N22 D12 D21 - N12 N21 D11 D22) / (D11 D22 D12 D21); D = G^-1 G_diag = adj(G) G_diag / det
det_num = pl.sub(pl.mul(pl.mul(N[0][0], N[1][1]), pl.mul(Dn[0][1], Dn[1][0])), pl.mul(pl.mul(N[0][1], N[1][0]), pl.mul(Dn[0][0], Dn[1][1])))
D11 = sy.tf(pl.mul(pl.mul(N[1][1], N[0][0]), pl.mul(Dn[0][1], Dn[1][0])), det_num).cancel()        # G22 G11 / det
D12 = sy.tf(-pl.mul(pl.mul(N[0][1], N[1][1]), pl.mul(Dn[0][0], Dn[1][0])), det_num).cancel()       # -G12 G22 / det
D21 = sy.tf(-pl.mul(pl.mul(N[1][0], N[0][0]), pl.mul(Dn[1][1], Dn[0][1])), det_num).cancel()       # -G21 G11 / det
D22 = sy.tf(pl.mul(pl.mul(N[0][0], N[1][1]), pl.mul(Dn[0][1], Dn[1][0])), det_num).cancel()        # G11 G22 / det
w = 0.05
Gw = fr.freqresp(col["system"], w)
Dw = np.array([[fr.freqresp(D11, w), fr.freqresp(D12, w)], [fr.freqresp(D21, w), fr.freqresp(D22, w)]])
print("G D at w = 0.05 (should be diagonal):\n", np.round(Gw @ Dw, 6))
print("proper:", all(len(d.num) <= len(d.den) for d in (D11, D12, D21, D22)), "| stable:", all(np.all(d.poles().real < 0) for d in (D11, D12, D21, D22)), "| poles of D are the zeros of det G:", np.round(pl.roots(det_num), 4))"""),
         md(r"""
The dynamic decoupler D = G⁻¹G_diag makes G D diagonal at every frequency, not just at dc; its poles are the
transmission zeros of G (the roots of det G), so it is stable only when the plant is minimum-phase, and it is
proper here because the entries share their relative degree. For plants with RHP transmission zeros or delays
an exact dynamic decoupler is unstable or non-causal, and one falls back to approximate or static decoupling.
""")],
    ]


@solutions("18_numerics_of_control", imports=["import time", "from scipy import linalg as sla", "from englincontrol import linalg as la, systems as sy, frequency as fr, models"])
def _():
    return [
        [code(r"""def expm_taylor(A, terms=40):
    E = np.eye(len(A)); term = np.eye(len(A))
    for k in range(1, terms):
        term = term @ A / k; E = E + term
    return E
rng = np.random.default_rng(0)
rows = []
for n in (10, 50, 200):
    for norm in (1.0, 100.0):
        A = rng.normal(size=(n, n)); A = A / np.linalg.norm(A, 1) * norm
        ref = sla.expm(A)
        t0 = time.perf_counter(); E1 = la.expm(A); t1 = time.perf_counter() - t0
        t0 = time.perf_counter(); E2 = expm_taylor(A); t2 = time.perf_counter() - t0
        t0 = time.perf_counter(); sla.expm(A); t3 = time.perf_counter() - t0
        rows.append((n, norm, np.max(np.abs(E1 - ref)) / np.max(np.abs(ref)), np.max(np.abs(E2 - ref)) / np.max(np.abs(ref)), t1 * 1e3, t2 * 1e3, t3 * 1e3))
print(pd.DataFrame(rows, columns=["n", "|A|", "library rel. error", "Taylor rel. error", "library (ms)", "Taylor (ms)", "scipy (ms)"]).to_string(index=False, float_format="%.2e"))"""),
         md(r"""
At norm 1 everything works; at norm 100 the Taylor series has lost all accuracy while the library's scaled
Padé keeps machine precision at every size. The cost of the library's method is a handful of matrix products
plus a solve and log₂(‖A‖/5.4) squarings - O(n³) with a modest constant - comparable with SciPy's, which is
the same algorithm in compiled code.
""")],
        [code(r"""cp = models.cart_pendulum()["system"]
A, B = cp.A, cp.B
Q = np.diag([10.0, 1.0, 100.0, 1.0])
for R in (1.0, 1e-2, 1e-4, 1e-6, 1e-8):
    S = B @ B.T / R
    H = np.block([[A, -S], [-Q, -A.T]])
    lam, V = np.linalg.eig(H)
    stable = V[:, lam.real < 0]
    P_sub = np.real(np.linalg.solve(stable[:4].T, stable[4:].T).T)
    P_lib = la.care(A, B, Q, np.array([[R]]))
    resid = lambda P_: np.linalg.norm(A.T @ P_ + P_ @ A - P_ @ S @ P_ + Q) / np.linalg.norm(Q)   # noqa: E731
    sep = np.min(np.abs(lam.real))
    print(f"R = {R:.0e}: eigenvalue separation from the axis {sep:.2e}; subspace residual {resid(P_sub):.1e} -> after Newton {resid(P_lib):.1e}; vs SciPy {np.max(np.abs(P_lib - sla.solve_continuous_are(A, B, Q, np.array([[R]])))) / np.max(np.abs(P_lib)):.1e}")"""),
         md(r"""
As R shrinks the cheap control pushes the closed-loop poles far out and the Hamiltonian's eigenvalues
spread over many orders of magnitude; the eigenvector-based subspace loses accuracy (the residual climbs
from 10⁻¹² to 10⁻⁴) and Newton's iterations recover it - until the problem is so badly scaled that a few
Newton steps are not enough. Production solvers use the Schur form and scaling of the Hamiltonian for this
reason; the library's Newton refinement is what makes its eigenvector route usable.
""")],
        [code(r"""def hinf_siso(G, tol=1e-8):
    s = G.to_ss(); A, B, C, D = s.A, s.B, s.C, s.D
    lo = max(np.max(np.abs(fr.freqresp(G, np.logspace(-2, 3, 500)))), abs(D[0, 0])) * 0.999; hi = 2 * lo + 1
    def imag_eig(g):
        R = g**2 - D.T @ D; Ri = np.linalg.inv(R)
        Ah = A + B @ Ri @ D.T @ C
        H = np.block([[Ah, B @ Ri @ B.T], [-C.T @ (np.eye(1) + D @ Ri @ D.T) @ C, -Ah.T]])
        ev = np.linalg.eigvals(H)
        return np.any(np.abs(ev.real) < 1e-7 * max(1, np.max(np.abs(ev))))
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        if imag_eig(mid): lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
for name, G in (("prototype zeta = 0.3", sy.tf([4.0], [1.0, 1.2, 4.0])), ("lightly damped zeta = 0.01", sy.tf([4.0], [1.0, 0.04, 4.0]))):
    print(f"{name:<28}: bisection {hinf_siso(G):.6f}, library {fr.hinf_norm(G.to_ss())[0]:.6f}, closed form 1/(2 zeta sqrt(1 - zeta^2)) = {1 / (2 * (0.3 if '0.3' in name else 0.01) * np.sqrt(1 - (0.3 if '0.3' in name else 0.01) ** 2)):.6f}")"""),
         md(r"""
The bisection needs only an eigenvalue test per step and converges to the peak gain without a frequency grid
- which matters for the lightly damped system, whose peak of 50 is a needle a sweep could straddle. Both the
hand-written bisection and the library agree with the closed form to six digits.
""")],
    ]


@solutions("19_project_a_position_servo_from_data", imports=["from scipy import optimize as sopt", "from englincontrol import systems as sy, identify as idf, design as dg, discrete as dc, frequency as fr, response as rs, robust as rb, statespace as sp, datasets, models"], setup=r'''p = pd.read_csv(datasets.path("motor_prbs.csv"), comment="#")
p = p[p.t_s >= 2.0]
u = (p.V - p.V.mean()).to_numpy(); yv = (p.speed_rad_s - p.speed_rad_s.mean()).to_numpy()
Gm_tf = dc.d2c(idf.output_error(u[::10], yv[::10], 2, 0.05)["tf"].to_ss()).to_tf()
s_ = pd.read_csv(datasets.path("servo_frequency_response.csv"), comment="#")
w = s_.w_rad_s.to_numpy(); H = 10 ** (s_.gain_dB.to_numpy() / 20) * np.exp(1j * np.radians(s_.phase_deg.to_numpy()))
P0 = sy.series(Gm_tf, sy.tf([1.0], [1.0, 0.0])); gear = np.median(np.abs(H[:10]) / np.abs(fr.freqresp(P0, w[:10]))); P0 = sy.tf(gear * P0.num, P0.den)
def mismatch(theta):
    wr, zr, L = theta
    model = fr.freqresp(sy.series(P0, sy.tf([wr**2], [1.0, 2 * zr * wr, wr**2])), w, delay=L)
    return np.r_[np.log(np.abs(H / model)), np.angle(H / model)]
wr, zr, delay = sopt.least_squares(mismatch, [60.0, 0.1, 0.002], bounds=([20, 0.01, 0], [200, 1.0, 0.05])).x
P_nom = sy.series(P0, sy.tf([wr**2], [1.0, 2 * zr * wr, wr**2]))
W = sy.tf([0.5 / 50.0, 0.07], [1.0 / 400.0, 1.0])
Pss = P_nom.to_ss(); A, B, C = Pss.A, Pss.B, Pss.C
K = sp.lqr(A, B, 10.0 * C.T @ C, np.array([[1.0]]))["K"]; Nbar = sp.reference_gain(A, B, C, K)

def simulate_digital(dt, K_, L_obs, Nbar_, T=1.5, step=10.0, u_lim=12.0, extra=None):
    Pz = dc.c2d(P_nom.to_ss(), 0.001)
    obs = dc.c2d(sy.ss(A - L_obs @ C, np.hstack([B, L_obs]), np.eye(len(A)), np.zeros((len(A), 2))), dt)
    nd = int(round(delay / 0.001)); N = int(T / 0.001); sub = int(round(dt / 0.001))
    xp = np.zeros(Pz.n); xo = np.zeros(obs.n); uh = np.zeros(N + nd + 1); yh = np.zeros(N); ur = np.zeros(N); u_ = 0.0; xi = 0.0
    for k in range(N):
        y = (Pz.C @ xp)[0]; yh[k] = y
        if k % sub == 0:
            u_ = float(np.clip((-K_ @ xo)[0] + Nbar_ * step + (extra(xi) if extra else 0.0), -u_lim, u_lim))
            xo = obs.A @ xo + obs.B @ [u_, y]; xi += dt * (step - y)
        ur[k] = u_; uh[k + nd] = u_
        xp = Pz.A @ xp + Pz.B @ [uh[k]]
    t_ = np.arange(N) * 0.001
    return t_, yh, ur, rs.step_info(t_, yh, final=step)''')
def _():
    return [
        [code(r"""alt = np.array([-8 + 6j, -8 - 6j, -20.0, -4.9 + 80.2j, -4.9 - 80.2j])   # a placed set with a different dominant pair (for one input the LQR poles themselves would give the same gain)
Kpp = sp.place(A, B, alt)
Nbar_pp = sp.reference_gain(A, B, C, Kpp)
rows = []
for name, K_, N_ in (("LQR", K, Nbar), ("pole placement (dominant pair -8 +- 6j)", Kpp, Nbar_pp)):
    for scale in (1.0, 4.0):
        L_obs = sp.observer_gain(A, C, -scale * np.array([15.0, 16.0, 17.0, 18.0, 19.0]))
        comp = sy.ss(A - B @ K_ - L_obs @ C, L_obs, K_, np.zeros((1, 1)))
        L_in = sy.series(P_nom, comp.to_tf())
        d = rb.nyquist_distance(L_in)
        t_, yh, ur, info = simulate_digital(0.005, K_, L_obs, N_)
        rows.append((name, f"x{scale:.0f}", d["distance"], info["settling_time"], np.max(np.abs(ur))))
print(pd.DataFrame(rows, columns=["gain", "observer speed", "min |1 + L|", "settling (s)", "peak V"]).to_string(index=False, float_format="%.3f"))"""),
         md(r"""
With a placed gain of comparable speed the pattern repeats: the slower observer gives the more robust loop.
The advantage belongs to the plant - its resonance punishes any compensator with high gain near 80 rad/s,
and a fast observer is such a compensator - not to the way the state feedback was chosen.
""")],
        [code(r"""Aa, Ba = sp.augment_integral(A, B, C)
Qa = np.zeros((6, 6)); Qa[:5, :5] = 10.0 * C.T @ C; Qa[5, 5] = 50.0
ra = sp.lqr(Aa, Ba, Qa, np.array([[1.0]]))
Kx, Ki = ra["K"][:, :5], ra["K"][:, 5:]
L_obs = sp.observer_gain(A, C, -np.array([15.0, 16.0, 17.0, 18.0, 19.0]))
comp = sy.ss(A - B @ Kx - L_obs @ C, L_obs, Kx, np.zeros((1, 1)))
L_in = sy.series(P_nom, comp.to_tf())
print(f"with integral action (the loop at the plant input ignores the integrator's path): min |1 + L| = {rb.nyquist_distance(L_in)['distance']:.3f}")
for gain_error in (1.0, 1.1):
    P_err = sy.tf(gain_error * P_nom.num, P_nom.den)
    # simulate with the integral state feeding u = -Kx xhat - Ki xi + ... on the perturbed plant
    Pz = dc.c2d(P_err.to_ss(), 0.001); obs = dc.c2d(sy.ss(A - L_obs @ C, np.hstack([B, L_obs]), np.eye(5), np.zeros((5, 2))), 0.005)
    nd = 4; N = 2000; xp = np.zeros(Pz.n); xo = np.zeros(5); uh = np.zeros(N + nd + 1); yh = np.zeros(N); xi = 0.0; u_ = 0.0
    for k in range(N):
        y = (Pz.C @ xp)[0]; yh[k] = y
        if k % 5 == 0:
            u_ = float(np.clip((-Kx @ xo)[0] - (Ki @ [xi])[0], -12, 12)); xo = obs.A @ xo + obs.B @ [u_, y]; xi += 0.005 * (10.0 - y)
        uh[k + nd] = u_; xp = Pz.A @ xp + Pz.B @ [uh[k]]
    t_ = np.arange(N) * 0.001
    info = rs.step_info(t_, yh, final=10.0)
    print(f"servo gain x{gain_error}: final position {yh[-1]:.3f} (reference 10), overshoot {info['overshoot']:.1f} %, settling {info['settling_time']:.2f} s")"""),
         md(r"""
The integrator removes the offset a 10 % gain error would leave with the reference-gain scheme, at a price:
a slower, more oscillatory response (0.4 s settling against 0.27, with more overshoot) and a slightly smaller
distance from −1. Integral action is insurance against model error, bought with bandwidth.
""")],
        [code(r"""dt = 0.005
L_obs = sp.observer_gain(A, C, -np.array([15.0, 16.0, 17.0, 18.0, 19.0]))
obs = dc.c2d(sy.ss(A - L_obs @ C, np.hstack([B, L_obs]), np.eye(5), np.zeros((5, 2))), dt)
class Controller:
    def __init__(self):
        self.xo = np.zeros(5)
    def __call__(self, y, r):
        u_ = float(np.clip((-K @ self.xo)[0] + Nbar * r, -12.0, 12.0))     # output first (the state was predicted last step)
        self.xo = obs.A @ self.xo + obs.B @ [u_, y]                           # then advance the observer
        return u_
ctrl = Controller()
Pz = dc.c2d(P_nom.to_ss(), 0.001); nd = int(round(delay / 0.001)); N = 1500
xp = np.zeros(Pz.n); uh = np.zeros(N + nd + 1); yh = np.zeros(N); u_ = 0.0
for k in range(N):
    y = (Pz.C @ xp)[0]; yh[k] = y
    if k % 5 == 0:
        u_ = ctrl(y, 10.0)
    uh[k + nd] = u_; xp = Pz.A @ xp + Pz.B @ [uh[k]]
t_ = np.arange(N) * 0.001
ref_t, ref_y, _, ref_info = simulate_digital(dt, K, L_obs, Nbar)
print(f"controller as a function: settling {rs.step_info(t_, yh, final=10.0)['settling_time']:.3f} s, overshoot {rs.step_info(t_, yh, final=10.0)['overshoot']:.1f} %; section 6's simulation: {ref_info['settling_time']:.3f} s, {ref_info['overshoot']:.1f} %; max difference {np.max(np.abs(yh - ref_y)):.1e}")"""),
         md(r"""
The deployable controller is a small object: five observer states, the gain and the reference gain, one
matrix-vector update per sample, the control computed before the observer is advanced so that it uses the
latest estimate. Run against the plant model it reproduces section 6's response exactly - the test to pass
before the same function goes onto the servo's microcontroller.
""")],
    ]


if __name__ == "__main__":
    write_all(only=[a for a in sys.argv[1:] if not a.startswith("--")])
