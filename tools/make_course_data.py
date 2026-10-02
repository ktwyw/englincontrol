"""Write the course data files into src/englincontrol/data (seed 2026): a DC motor speed step test with
tachometer noise and a dropped sample, a PRBS experiment on the same motor, and the measured frequency response of
a position servo with an unmodelled resonance and a delay. Run: python tools/make_course_data.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from englincontrol import discrete as dc  # noqa: E402
from englincontrol import frequency as fr  # noqa: E402
from englincontrol import identify as idf  # noqa: E402
from englincontrol import models  # noqa: E402
from englincontrol import response as rs
from englincontrol import systems as sy

OUT = Path(__file__).resolve().parents[1] / "src" / "englincontrol" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

# the "true" motor of the course data: R = 2.0 ohm, L = 0.1 H, Kt = Ke = 0.05, J = 0.002, b = 0.005
motor = models.dc_motor(R=2.0, L=0.1, Kt=0.05, Ke=0.05, J=0.002, b=0.005)
G = motor["system"]

# 1. step test: 12 V step at t = 0.1 s, 1 kHz, 1.5 s, tachometer noise 1 rad/s rms, one dropped sample (NaN)
dt = 0.001
t = np.arange(0, 1.5, dt)
u = np.where(t >= 0.1, 12.0, 0.0)
_, y, _ = rs.lsim(G, u, t)
y = y + rng.normal(0, 1.0, len(t))
y[742] = np.nan
lines = [
    "# DC motor speed response to a 12 V armature voltage step applied at t = 0.1 s (1 kHz samples, 1.5 s).",
    "# Tachometer noise about 1 rad/s rms; the sample at t = 0.742 s was dropped by the logger (NaN).",
    "# Generated from englincontrol.models.dc_motor(R=2.0, L=0.1, Kt=0.05, Ke=0.05, J=0.002, b=0.005). Columns: t_s, V, speed_rad_s.",
    "t_s,V,speed_rad_s",
]
lines += [f"{tk:.3f},{uk:.1f},{yk:.3f}" if np.isfinite(yk) else f"{tk:.3f},{uk:.1f}," for tk, uk, yk in zip(t, u, y)]
(OUT / "motor_step.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

# 2. PRBS experiment: +-6 V around 6 V, 200 Hz, clock 150 ms (30 samples), 40 s; tachometer noise 0.3 rad/s
dt2 = 0.005
N = 8000
u2 = 6.0 + 6.0 * idf.prbs(N, seed=3, hold=30)
Gd = dc.c2d(G, dt2)
_, y2, _ = rs.lsim(Gd, u2)
y2 = y2 + rng.normal(0, 0.3, N)
t2 = np.arange(N) * dt2
lines = [
    "# DC motor PRBS experiment: armature voltage switching between 0 and 12 V (clock period 150 ms) and the measured speed, 200 Hz, 40 s.",
    "# Same motor as motor_step.csv; speed measured with a better tachometer (0.3 rad/s rms). Columns: t_s, V, speed_rad_s.",
    "t_s,V,speed_rad_s",
]
lines += [f"{tk:.3f},{uk:.1f},{yk:.3f}" for tk, uk, yk in zip(t2, u2, y2)]
(OUT / "motor_prbs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

# 3. servo frequency response: motor + gearbox position (integrator), a structural resonance at 80 rad/s (zeta 0.05) and 3 ms delay
servo = sy.series(motor["system"].to_tf(), sy.tf([1.0], [1.0, 0.0]))
servo = sy.series(servo, sy.tf([80.0**2], [1.0, 2 * 0.05 * 80.0, 80.0**2]))
w = np.logspace(-1, 2.3, 60)
H = fr.freqresp(servo, w, delay=0.003) * 50.0  # gear ratio and sensor scaling
H = H * (1 + 0.02 * rng.normal(size=60)) * np.exp(1j * np.radians(rng.normal(0, 1.0, 60)))
lines = [
    "# Measured frequency response of a position servo (voltage to shaft angle): sine sweep, 60 frequencies.",
    "# Gain in dB and phase in degrees (unwrapped); about 2 % gain and 1 degree phase scatter. The servo contains the motor of",
    "# motor_step.csv, a gearbox and position sensor (overall gain 50), a structural resonance and a transport delay.",
    "# Columns: w_rad_s, gain_dB, phase_deg.",
    "w_rad_s,gain_dB,phase_deg",
]
ph = np.degrees(np.unwrap(np.angle(H)))
lines += [f"{wk:.4f},{20 * np.log10(abs(hk)):.3f},{pk:.2f}" for wk, hk, pk in zip(w, H, ph)]
(OUT / "servo_frequency_response.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
for f in sorted(OUT.glob("*.csv")):
    print(f.name, len(f.read_text().splitlines()), "lines")
