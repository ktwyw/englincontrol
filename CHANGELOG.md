# Changelog

## 0.1.0 - 2026-10-01

First public release.

- **Library** (`englincontrol`, 14 modules, NumPy only): systems and realisations, exact time responses,
  frequency response with margins and the H-infinity norm, stability tests (Routh, Hurwitz, Jury, Lyapunov, root
  locus, the Nyquist criterion), PID with tuning rules and anti-windup, lead/lag design, pole placement,
  observers, LQR, Kalman filter, LQG, discretisation and digital design, robustness measures and limitations,
  identification (ARX/IV/output-error, frequency fits, ERA, balanced truncation), MIMO basics, ten example plants
  and three course data files.
- **Course**: 20 executed notebooks with learning objectives, "Inside the algorithm" sections and exercises, and
  20 notebooks of worked solutions (60 exercises), ending in a servo-design project from measured data.
- **Validation**: 204 checks against closed forms, SciPy/LAPACK, python-control and the theory, run with warnings
  as errors.
