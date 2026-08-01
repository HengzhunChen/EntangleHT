from __future__ import annotations

import math

from .resources import ResourceModel, restarts_from_shots
from .utilities import validate_probability


def standard_hadamard_shots(
    *,
    epsilon: float,
    delta: float,
    p_fail: float,
) -> int:
    """Shot bound for one-quadrature standard HT in the exact case."""

    validate_probability(p_fail, "p_fail")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon!r}")
    if not 0.0 <= delta < math.pi / 2:
        raise ValueError(f"delta must lie in [0, pi/2), got {delta!r}")

    root = math.sqrt(max(0.0, math.cos(delta) ** 2 + epsilon**2))
    r_star = epsilon * (root - epsilon * math.sin(delta)) / (1.0 + epsilon**2)
    return int(
        math.ceil(
            (2.0 / r_star**2) * math.log(2.0 / p_fail)
        )
    )


def two_quadrature_standard_shots(
    *,
    epsilon: float,
    rho0: float,
    p_fail: float,
) -> int:
    """Total shots for the two-quadrature imperfect-eigenstate baseline."""

    validate_probability(p_fail, "p_fail")
    if not 0.0 < epsilon < math.pi / 2:
        raise ValueError(f"epsilon must lie in (0, pi/2), got {epsilon!r}")
    if not 0.0 < rho0 <= 1.0:
        raise ValueError(f"rho0 must lie in (0, 1], got {rho0!r}")

    shots_per_quadrature = math.ceil(
        (4.0 / (rho0**2 * math.sin(epsilon) ** 2))
        * math.log(4.0 / p_fail)
    )
    return 2 * shots_per_quadrature


def standard_restarts(shots: int, resources: ResourceModel) -> int:
    return restarts_from_shots(shots, resources.packing_capacity(1))
