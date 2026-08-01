"""Validation and optimization of one schedule round."""

from __future__ import annotations

import math
from typing import Sequence

from ..certification import (
    contrast_bias_bound,
    optimized_inversion_radius,
    shots_for_round,
)
from ..resources import ResourceModel, restarts_from_shots
from ..records import EstimationConfig, RoundPlan


def _round_restart_score(
    *,
    restarts: int,
    amplification: int,
) -> tuple[int, int]:
    """Minimize restarts, then prefer smaller amplification."""
    return restarts, amplification


def _select_round(
    *,
    bound_before: float,
    bound_after: float,
    config: EstimationConfig,
    p_fail: float,
    resources: ResourceModel,
    omegas: Sequence[float | None],
) -> RoundPlan:
    """Enumerate all feasible ``(omega, m)`` pairs and select the best."""
    hardware_cap = resources.effective_m_hw(
        config.hardware_amplification_cap
    )
    branch_cap = int(math.floor(config.branch_margin / bound_before))
    search_cap = min(hardware_cap, branch_cap)
    if search_cap < 1:
        raise ValueError("no branch- and width-feasible amplification")

    best: tuple[tuple[int, int], RoundPlan] | None = None
    for omega in omegas:
        if omega is None:
            statistical_accuracy = bound_after
            bias_budget = 0.0
        else:
            if not 0.0 < omega < 1.0:
                continue
            statistical_accuracy = omega * bound_after
            bias_budget = (1.0 - omega) * bound_after

        for amplification in range(1, search_cap + 1):
            if amplification * bound_before > config.branch_margin:
                continue
            if omega is not None and contrast_bias_bound(
                amplification,
                bound_before,
                config.contrast_lower_bound,
            ) > bias_budget:
                continue

            inverse_radius = optimized_inversion_radius(
                m=amplification,
                delta=bound_before,
                epsilon_stat=statistical_accuracy,
            )
            shots = shots_for_round(
                p_fail=p_fail,
                inverse_radius=inverse_radius,
            )
            restarts = restarts_from_shots(
                shots,
                resources.packing_capacity(amplification),
            )
            round_plan = RoundPlan(
                round_index=0,
                bound_before=bound_before,
                bound_after=bound_after,
                amplification=amplification,
                statistical_accuracy=statistical_accuracy,
                bias_budget=bias_budget,
                shots=shots,
                restarts=restarts,
                inverse_radius=inverse_radius,
                p_fail=p_fail,
                omega=omega,
            )
            score = _round_restart_score(
                restarts=restarts,
                amplification=amplification,
            )
            if best is None or score < best[0]:
                best = score, round_plan

    if best is None:
        state = "imperfect" if omegas != (None,) else "exact"
        raise ValueError(f"no feasible {state} round")
    return best[1]


def select_exact_round(
    *,
    bound_before: float,
    bound_after: float,
    config: EstimationConfig,
    p_fail: float,
    resources: ResourceModel,
) -> RoundPlan:
    """Select the restart-minimizing feasible exact-state round."""
    return _select_round(
        bound_before=bound_before,
        bound_after=bound_after,
        config=config,
        p_fail=p_fail,
        resources=resources,
        omegas=(None,),
    )


def select_imperfect_round(
    *,
    bound_before: float,
    bound_after: float,
    config: EstimationConfig,
    p_fail: float,
    resources: ResourceModel,
    omega_grid: Sequence[float] | None = None,
    fixed_omega: float | None = None,
) -> RoundPlan:
    """Select the restart-minimizing feasible imperfect-state round."""
    if (omega_grid is None) == (fixed_omega is None):
        raise ValueError("provide exactly one of omega_grid or fixed_omega")
    omegas: tuple[float | None, ...]
    if fixed_omega is not None:
        omegas = (fixed_omega,)
    else:
        omegas = tuple(omega_grid or ())
        if not omegas:
            raise ValueError("omega_grid must not be empty")
    return _select_round(
        bound_before=bound_before,
        bound_after=bound_after,
        config=config,
        p_fail=p_fail,
        resources=resources,
        omegas=omegas,
    )
