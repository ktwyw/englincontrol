"""README images: the hero animation and the gallery, computed with englincontrol. Run: python tools/make_images.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from englincontrol import discrete as dc  # noqa: E402
from englincontrol import frequency as fr  # noqa: E402
from englincontrol import (
    models,
)  # noqa: E402
from englincontrol import (
    polynomials as pl,
)
from englincontrol import (
    response as rs,
)
from englincontrol import (
    robust as rb,
)
from englincontrol import (
    stability as st,
)
from englincontrol import (
    statespace as sp,
)
from englincontrol import (
    systems as sy,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def hero():
    G = sy.tf([1.0], pl.from_roots([0.0, -1.0, -2.0]))
    gains = np.logspace(-1.5, np.log10(12.0), 70)
    rl = st.root_locus(G, gains)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=80)
    ax, ax2 = axes
    for k in range(3):
        ax.plot(rl["branches"][:, k].real, rl["branches"][:, k].imag, color="0.75", lw=1)
    ax.plot(rl["poles"].real, rl["poles"].imag, "kx", ms=9)
    ax.axhline(0, color="0.85")
    ax.axvline(0, color="0.85")
    ax.set_xlim(-3.5, 1.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_title("root locus of 1 / (s (s+1)(s+2))")
    ax.set_xlabel("Re s")
    ax.set_ylabel("Im s")
    (dots,) = ax.plot([], [], "o", color="C3", ms=7)
    ax2.set_xlim(0, 15)
    ax2.set_ylim(-0.2, 2.2)
    ax2.axhline(1, color="0.7", lw=0.8)
    ax2.set_xlabel("t")
    ax2.set_title("closed-loop step response")
    (line,) = ax2.plot([], [], color="C0")
    txt = ax2.text(0.98, 0.92, "", transform=ax2.transAxes, ha="right")
    t = np.linspace(0, 15, 601)

    def update(i):
        K = gains[i]
        r = rl["branches"][i]
        dots.set_data(r.real, r.imag)
        T = sy.feedback(sy.tf([K], G.den))
        _, y = rs.step(T, 15.0, 601)
        line.set_data(t, np.clip(y, -1, 3))
        txt.set_text(f"K = {K:.2f}" + ("  (unstable)" if K > 6 else ""))
        return dots, line, txt

    fig.tight_layout()
    FuncAnimation(fig, update, frames=len(gains), blit=True).save(OUT / "hero.gif", writer=PillowWriter(fps=10))
    plt.close(fig)


def bode_nyquist():
    L = sy.zpk([-10.0], [0.0, -1.0, -50.0], 100.0)
    m = fr.margins(L)
    w = np.logspace(-2, 3, 600)
    w_, mag, ph = fr.bode(L, w)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].semilogx(w, mag, label="|L| dB")
    axes[0].semilogx(w, ph / 10, "--", label="phase / 10 (deg)")
    axes[0].axvline(m["w_gc"], color="0.7")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("w (rad/s)")
    axes[0].set_title(f"Bode: phase margin {m['phase_margin']:.0f} deg at {m['w_gc']:.1f} rad/s")
    _, Lj = fr.nyquist(L, w)
    axes[1].plot(Lj.real, Lj.imag)
    axes[1].plot(Lj.real, -Lj.imag, "--", color="C0", lw=0.7)
    axes[1].plot(-1, 0, "rx", ms=9)
    th = np.linspace(0, 2 * np.pi, 200)
    axes[1].plot(np.cos(th), np.sin(th), "0.6", lw=0.6)
    axes[1].set_xlim(-2, 1)
    axes[1].set_ylim(-2, 1.5)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Nyquist: closest approach to -1 is {1 / rb.sensitivity_peak(L)['Ms']:.2f}")
    fig.tight_layout()
    fig.savefig(OUT / "bode_nyquist.png")
    plt.close(fig)


def lqr_pendulum():
    cp = models.cart_pendulum()["system"]
    A, B, C = cp.A, cp.B, cp.C
    Q = np.diag([10.0, 1.0, 100.0, 1.0])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for R in (1.0, 0.1, 0.01):
        K = sp.lqr(A, B, Q, np.array([[R]]))["K"]
        t, y, x = rs.initial(sy.ss(A - B @ K, B, C, np.zeros((2, 1))), [0, 0, 0.2, 0], 4.0)
        axes[0].plot(t, np.degrees(x[:, 2]), label=f"R = {R}")
        axes[1].plot(t, -(x @ K.T)[:, 0])
    axes[0].set_ylabel("pendulum angle (deg)")
    axes[0].set_xlabel("t")
    axes[0].legend()
    axes[0].set_title("LQR: the cart-pendulum recovers from 11 degrees")
    axes[1].set_ylabel("force (N)")
    axes[1].set_xlabel("t")
    axes[1].set_title("the price in control effort")
    fig.tight_layout()
    fig.savefig(OUT / "lqr_pendulum.png")
    plt.close(fig)


def kalman():
    mot = models.dc_motor()["system"]
    dt = 0.005
    Gd = dc.c2d(mot, dt)
    Ad, Bd, Cd = Gd.A, Gd.B, Gd.C
    Qd, Rd = np.diag([1e-6, 1e-4]), np.array([[0.25]])
    kd = sp.kalman(Ad, Cd, Qd, Rd, discrete=True)
    rng = np.random.default_rng(1)
    N = 1600
    u = 6.0 * (1 + np.sign(np.sin(0.5 * np.arange(N) * dt)))
    x = np.zeros(2)
    X = np.zeros((N, 2))
    Y = np.zeros(N)
    for k in range(N):
        X[k] = x
        Y[k] = (Cd @ x)[0] + 0.5 * rng.normal()
        x = Ad @ x + Bd[:, 0] * u[k] + rng.normal(size=2) * np.sqrt(np.diag(Qd))
    Xh = sp.kalman_filter_run(Ad, Bd, Cd, kd["L"], u, Y)
    t = np.arange(N) * dt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(t, Y, ".", ms=2, color="0.7", label="measured speed")
    ax.plot(t, X[:, 1], "k", lw=1, label="true")
    ax.plot(t, Xh[:, 1], "r", lw=1, label="Kalman estimate")
    ax.legend(fontsize=8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("rad/s")
    ax.set_title("the steady-state Kalman filter on tachometer noise")
    fig.tight_layout()
    fig.savefig(OUT / "kalman.png")
    plt.close(fig)


def discretisation():
    G = sy.tf([4.0], [1.0, 1.2, 4.0])
    dt = 0.1
    w = np.logspace(-1, np.log10(np.pi / dt), 300)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogx(w, 20 * np.log10(np.abs(fr.freqresp(G, w))), "k", lw=2, label="continuous")
    for meth in ("zoh", "tustin", "euler"):
        ax.semilogx(w, 20 * np.log10(np.abs(fr.freqresp(dc.c2d(G, dt, meth), w))), label=meth)
    ax.axvline(np.pi / dt, color="0.6", ls="--")
    ax.set_ylim(-50, 15)
    ax.legend(fontsize=8)
    ax.set_xlabel("w (rad/s)")
    ax.set_ylabel("|G| dB")
    ax.set_title("discretisations at dt = 0.1 (Nyquist frequency dashed)")
    fig.tight_layout()
    fig.savefig(OUT / "discretisation.png")
    plt.close(fig)


def waterbed():
    w = np.logspace(-2, 2, 500)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, L in (
        ("stable loop", sy.tf([1.0], [1.0, 3.0, 2.0, 0.0])),
        ("stabilised loop with a pole at +1", sy.tf([30.0, 60.0], pl.from_roots([1, -5, -8]))),
    ):
        S = 1 / (1 + fr.freqresp(L, w))
        bi = rb.bode_sensitivity_integral(L)
        ax.semilogx(w, np.log(np.abs(S)), label=f"{name}: integral {bi['integral']:.2f} (theory {bi['theory']:.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    ax.set_xlabel("w (rad/s)")
    ax.set_ylabel("ln |S|")
    ax.set_title("Bode's sensitivity integral: the waterbed")
    fig.tight_layout()
    fig.savefig(OUT / "waterbed.png")
    plt.close(fig)


def project():
    import pandas as pd
    from scipy import optimize as sopt

    from englincontrol import datasets
    from englincontrol import identify as idf

    p = pd.read_csv(datasets.path("motor_prbs.csv"), comment="#")
    p = p[p.t_s >= 2.0]
    u = (p.V - p.V.mean()).to_numpy()
    yv = (p.speed_rad_s - p.speed_rad_s.mean()).to_numpy()
    Gm = dc.d2c(idf.output_error(u[::10], yv[::10], 2, 0.05)["tf"].to_ss()).to_tf()
    s_ = pd.read_csv(datasets.path("servo_frequency_response.csv"), comment="#")
    w = s_.w_rad_s.to_numpy()
    H = 10 ** (s_.gain_dB.to_numpy() / 20) * np.exp(1j * np.radians(s_.phase_deg.to_numpy()))
    P0 = sy.series(Gm, sy.tf([1.0], [1.0, 0.0]))
    gear = np.median(np.abs(H[:10]) / np.abs(fr.freqresp(P0, w[:10])))
    P0 = sy.tf(gear * P0.num, P0.den)

    def mismatch(theta):
        wr, zr, L = theta
        model = fr.freqresp(sy.series(P0, sy.tf([wr**2], [1.0, 2 * zr * wr, wr**2])), w, delay=L)
        return np.r_[np.log(np.abs(H / model)), np.angle(H / model)]

    wr, zr, delay = sopt.least_squares(mismatch, [60.0, 0.1, 0.002], bounds=([20, 0.01, 0], [200, 1.0, 0.05])).x
    P_nom = sy.series(P0, sy.tf([wr**2], [1.0, 2 * zr * wr, wr**2]))
    wf = np.logspace(-1, 2.3, 400)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogx(w, s_.gain_dB, "o", ms=4, label="measured servo response")
    ax.semilogx(wf, 20 * np.log10(np.abs(fr.freqresp(P0, wf))), "--", label="identified motor x integrator")
    ax.semilogx(
        wf,
        20 * np.log10(np.abs(fr.freqresp(P_nom, wf, delay=delay))),
        label=f"+ resonance {wr:.0f} rad/s, delay {delay * 1000:.0f} ms",
    )
    ax.legend(fontsize=8)
    ax.set_xlabel("w (rad/s)")
    ax.set_ylabel("dB")
    ax.set_title("project: what the servo data show that the motor tests could not")
    fig.tight_layout()
    fig.savefig(OUT / "project_servo.png")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    bode_nyquist()
    lqr_pendulum()
    kalman()
    discretisation()
    waterbed()
    project()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:<22} {f.stat().st_size / 1024:7.1f} kB")
