"""Optimization of geometric error-bound schedules."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Sequence

from ..certification import per_round_p_fail
from ..resources import ResourceModel
from ..records import EstimationConfig, RoundPlan, ScheduleResult, TrialPlan
from .round_selection import (
    select_exact_round,
    select_imperfect_round,
)


def compute_num_rounds(delta_0: float, epsilon: float, gamma: float) -> int:
    """Return the rounds needed for ``delta_t = gamma**t * delta_0``."""
    if delta_0 <= 0.0:
        raise ValueError(f"delta_0 must be positive, got {delta_0!r}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon!r}")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie in (0, 1), got {gamma!r}")

    if epsilon >= delta_0:
        return 0

    return int(math.ceil(math.log(delta_0 / epsilon) / math.log(1.0 / gamma)))


def _geometric_bounds(
    config: EstimationConfig,
    gamma: float,
) -> tuple[float, ...]:
    """Build the decreasing error-bound sequence for a geometric schedule."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    rounds = compute_num_rounds(
        delta_0=config.initial_bound,
        epsilon=config.target_accuracy,
        gamma=gamma,
    )
    bounds = [config.initial_bound]
    for _ in range(rounds):
        bounds.append(max(config.target_accuracy, gamma * bounds[-1]))
    bounds[-1] = config.target_accuracy
    return tuple(bounds)


def _plan_geometric(
    *,
    config: EstimationConfig,
    gamma: float,
    edge_selector: Callable[[float, float, float], RoundPlan],
    label: str,
) -> TrialPlan:
    """Construct a trial plan for a geometric sequence of error bounds."""
    bounds = _geometric_bounds(config, gamma)
    num_rounds = len(bounds) - 1
    p_fail = per_round_p_fail(
        config.total_success_probability,
        num_rounds,
    )
    rounds = tuple(
        replace(
            edge_selector(bounds[index], bounds[index + 1], p_fail),
            round_index=index,
        )
        for index in range(num_rounds)
    )
    return TrialPlan(label=label, rounds=rounds)


def _plan_restart_score(plan: TrialPlan) -> tuple[int, int, int]:
    """Minimize total restarts, then width and round count."""
    return (
        plan.total_restarts,
        plan.max_amplification,
        len(plan.rounds),
    )


def optimize_exact_geometric(
    *,
    base_config: EstimationConfig,
    gamma_grid: Sequence[float],
    resources: ResourceModel,
    label: str = "exact_entangled_geometric",
) -> ScheduleResult:
    """Find the restart-minimizing exact geometric schedule."""
    best: tuple[tuple[int, int, int], float, TrialPlan] | None = None
    evaluated = 0
    for gamma in gamma_grid:
        try:
            plan = _plan_geometric(
                config=base_config,
                gamma=gamma,
                edge_selector=lambda before, after, p_fail: select_exact_round(
                    bound_before=before,
                    bound_after=after,
                    config=base_config,
                    p_fail=p_fail,
                    resources=resources,
                ),
                label=label,
            )
        except ValueError:
            continue
        evaluated += 1
        candidate = _plan_restart_score(plan), gamma, plan
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("no feasible gamma found for exact geometric schedule")
    _, gamma, plan = best
    return ScheduleResult(
        config=base_config,
        plan=plan,
        method="geometric",
        evaluated_candidates=evaluated,
        gamma=gamma,
    )


def optimize_imperfect_geometric(
    *,
    base_config: EstimationConfig,
    gamma_grid: Sequence[float],
    omega_grid: Sequence[float],
    resources: ResourceModel,
    label: str = "imperfect_entangled_geometric",
) -> ScheduleResult:
    """Find the restart-minimizing imperfect geometric schedule."""
    best: tuple[tuple[int, int, int], float, float, TrialPlan] | None = None
    evaluated = 0
    for gamma in gamma_grid:
        for omega in omega_grid:
            try:
                plan = _plan_geometric(
                    config=base_config,
                    gamma=gamma,
                    edge_selector=lambda before, after, p_fail, omega=omega: (
                        select_imperfect_round(
                            bound_before=before,
                            bound_after=after,
                            config=base_config,
                            p_fail=p_fail,
                            resources=resources,
                            fixed_omega=omega,
                        )
                    ),
                    label=label,
                )
            except ValueError:
                continue
            evaluated += 1
            candidate = _plan_restart_score(plan), gamma, omega, plan
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise ValueError("no feasible gamma/omega pair found")
    _, gamma, omega, plan = best
    return ScheduleResult(
        config=base_config,
        plan=plan,
        method="geometric",
        evaluated_candidates=evaluated,
        gamma=gamma,
        omega=omega,
    )
