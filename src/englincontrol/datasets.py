"""Bundled course data: measurements generated from stated models with documented noise and faults.

>>> from englincontrol import datasets
>>> datasets.available()
['motor_prbs.csv', 'motor_step.csv', 'servo_frequency_response.csv']
"""

from __future__ import annotations

from importlib import resources

INFO = {
    "motor_step.csv": "DC motor speed response to a 12 V step, 1 kHz samples with tachometer noise (identification chapter).",
    "motor_prbs.csv": "DC motor PRBS experiment: voltage and measured speed, 200 Hz, for ARX identification (identification chapter and project).",
    "servo_frequency_response.csv": "Measured frequency response (gain in dB, phase in degrees) of a position servo with a resonance and a delay (project).",
}


def path(name: str):
    """Path-like object of a bundled data file (pass it to pandas.read_csv with comment='#')."""
    if name not in INFO:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(sorted(INFO))}")
    return resources.files("englincontrol.data") / name


def available() -> list[str]:
    """Names of the bundled data files."""
    return sorted(INFO)


def describe(name: str) -> str:
    """One-line description of a bundled data file."""
    return INFO[name]
