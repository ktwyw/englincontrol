"""englincontrol: linear control theory from scratch.

polynomials  polynomial algebra for transfer functions
linalg       matrix exponential, Schur, Lyapunov/Sylvester, continuous and discrete Riccati, Gramians
systems      StateSpace, TransferFunction, conversions, canonical and balanced forms, interconnections, minimality
response     exact time responses, step/impulse/initial, step characteristics, second-order formulas
frequency    frequency response, Bode/Nyquist/Nichols, margins, bandwidth, Pade delay, H-infinity norm
stability    Routh-Hurwitz, Jury, Lyapunov test, root locus with its rules, the Nyquist criterion
design       PID forms and tuning rules, relay autotuning, saturation and anti-windup simulation, lead/lag, feedforward
statespace   controllability/observability, pole placement, observers, integral action, LQR, Kalman filter, LQG
discrete     ZOH/Tustin/matched/Euler discretisation, d2c, aliasing, discrete PID, dead-beat control, sample rates
robust       gang of four, Ms/Mt, Bode's sensitivity integral, RHP limits, disk margin, small gain
identify     FOPDT/second-order fits, ARX least squares, frequency-response fits, ERA, balanced truncation, PRBS
mimo         transfer matrices, singular values, transmission zeros, RGA and pairing, static decoupling
models       example plants (DC motor, tanks, thermal, cart-pendulum, aircraft, quarter car, distillation ...)
datasets     the course data files (motor step and PRBS tests, a servo frequency response)
"""

from . import (
    datasets,
    design,
    discrete,
    frequency,
    identify,
    linalg,
    mimo,
    models,
    polynomials,
    response,
    robust,
    stability,
    statespace,
    systems,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "datasets",
    "design",
    "discrete",
    "frequency",
    "identify",
    "linalg",
    "mimo",
    "models",
    "polynomials",
    "response",
    "robust",
    "stability",
    "statespace",
    "systems",
]
