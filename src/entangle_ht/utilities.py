from __future__ import annotations

import math


def validate_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1), got {value!r}")


def wrap_phase(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def phase_error(estimate: float, target: float) -> float:
    # Phases are periodic, so raw subtraction can make two nearby phases look
    # far apart when they lie on opposite sides of a 2*pi boundary.
    return wrap_phase(estimate - target)
