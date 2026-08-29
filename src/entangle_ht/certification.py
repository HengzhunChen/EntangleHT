"""Certified bias, accuracy, shot, and failure-probability formulas."""

from __future__ import annotations

import math

from .utilities import validate_probability


def amplitude_bias_bound(m: int, delta_t: float, rho0: float) -> float:
    """Certified amplitude-bias bound B(m; Delta_t, rho0)."""
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m!r}")
    if delta_t <= 0.0:
        raise ValueError(f"delta_t must be positive, got {delta_t!r}")
    if not 0.0 < rho0 <= 1.0:
        raise ValueError(f"rho0 must lie in (0, 1], got {rho0!r}")

    angle = m * delta_t
    if angle >= math.pi / 2:
        return math.inf

    return ((1.0 - rho0**m) / m) * math.tan(angle)


def optimized_inversion_radius(
    m: int,
    delta: float,
    epsilon_stat: float,
) -> float:
    """Return the optimized inverse-sine radius ``r_star``.

    With ``q = m*epsilon_stat``, ``s = sin(m*delta)``, and
    ``c = cos(m*delta)``, the original form is

        r_star = q * (sqrt(c**2 + q**2) - q*s) / (1 + q**2).

    Rationalizing the numerator gives the equivalent form used below:

        r_star = q * c**2 / (sqrt(c**2 + q**2) + q*s).

    The second form avoids subtracting the nearby positive terms.
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m!r}")
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta!r}")
    if epsilon_stat <= 0.0:
        raise ValueError(f"epsilon_stat must be positive, got {epsilon_stat!r}")

    if m * delta >= math.pi / 2:
        raise ValueError(
            "The optimized high-probability shot rule requires "
            "m * delta < pi/2."
        )

    q = m * epsilon_stat
    c = math.cos(m * delta)
    s = math.sin(m * delta)
    r_star = q * c**2 / (math.sqrt(c**2 + q**2) + q * s)

    if r_star <= 0.0:
        raise ValueError(
            "Optimized inversion radius is non-positive. "
            "Retune epsilon_stat, delta, or the amplification schedule."
        )

    return r_star


def shots_for_round(
    *,
    p_fail: float,
    inverse_radius: float,
) -> int:
    """Certified number of shots for a per-round failure probability."""
    validate_probability(p_fail, "p_fail")
    if inverse_radius <= 0.0:
        raise ValueError(f"inverse_radius must be positive, got {inverse_radius!r}")
    return int(
        math.ceil(
            (2.0 / inverse_radius**2) * math.log(2.0 / p_fail)
        )
    )


def per_round_p_fail(
    total_success_probability: float,
    num_rounds: int,
) -> float:
    """Uniform conditional failure allocation from the paper."""
    validate_probability(total_success_probability, "total_success_probability")
    if num_rounds < 0:
        raise ValueError(f"num_rounds must be non-negative, got {num_rounds!r}")
    if num_rounds == 0:
        return 0.0
    return 1.0 - total_success_probability ** (1.0 / num_rounds)
