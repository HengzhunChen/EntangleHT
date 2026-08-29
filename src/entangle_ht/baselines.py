"""Shot and resource formulas for comparison methods."""

from __future__ import annotations

import math

from .certification import (
    amplitude_bias_bound,
    optimized_inversion_radius,
    shots_for_round,
)
from .resources import ResourceModel
from .utilities import validate_probability


# -----------------------------------------------------------------------------
# Fixed-m Entangled HT
# -----------------------------------------------------------------------------

def restart_optimal_fixed_amplification(
    *,
    initial_bound: float,
    branch_margin: float,
    hardware_amplification_cap: int,
    resources: ResourceModel,
) -> int:
    """Fixed amplification minimizing the asymptotic certified restart cost.

    For small target accuracy, the shot count scales as
    ``1 / (m**2 * cos(m * initial_bound)**2)``. Accounting for parallel
    packing therefore amounts to maximizing that denominator times
    ``resources.packing_capacity(m)`` over amplifications satisfying
    ``m * initial_bound <= branch_margin`` and the hardware-width cap.
    """

    if not 0.0 < branch_margin < math.pi / 2:
        raise ValueError(
            "branch_margin must lie in (0, pi/2), "
            f"got {branch_margin!r}"
        )
    if not 0.0 < initial_bound <= branch_margin:
        raise ValueError(
            "initial_bound must lie in (0, branch_margin], "
            f"got {initial_bound!r}"
        )

    branch_cap = math.floor(branch_margin / initial_bound)
    max_amplification = min(
        branch_cap,
        resources.effective_m_hw(hardware_amplification_cap),
    )
    return max(
        range(1, max_amplification + 1),
        key=lambda amplification: (
            resources.packing_capacity(amplification)
            * amplification**2
            * math.cos(amplification * initial_bound) ** 2
        ),
    )


def fixed_amplification_hadamard_shots(
    *,
    epsilon: float,
    delta: float,
    amplification: int,
    p_fail: float,
) -> int:
    """Shot bound for exact-state HT at one fixed amplification."""

    inverse_radius = optimized_inversion_radius(
        m=amplification,
        delta=delta,
        epsilon_stat=epsilon,
    )
    return shots_for_round(
        p_fail=p_fail,
        inverse_radius=inverse_radius,
    )


def imperfect_fixed_amplification_hadamard_shots(
    *,
    epsilon: float,
    delta: float,
    amplification: int,
    contrast_lower_bound: float,
    p_fail: float,
) -> int:
    """Shot bound for Fixed-m EHT with certified amplitude bias."""

    amplitude_bias = amplitude_bias_bound(
        amplification,
        delta,
        contrast_lower_bound,
    )
    statistical_accuracy = epsilon - amplitude_bias
    if statistical_accuracy <= 0.0:
        raise ValueError(
            "epsilon must exceed the fixed-amplification amplitude-bias "
            f"bound {amplitude_bias:.10g}"
        )
    return fixed_amplification_hadamard_shots(
        epsilon=statistical_accuracy,
        delta=delta,
        amplification=amplification,
        p_fail=p_fail,
    )

# -----------------------------------------------------------------------------
# Standard HT
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Two-quadrature HT
# -----------------------------------------------------------------------------

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
