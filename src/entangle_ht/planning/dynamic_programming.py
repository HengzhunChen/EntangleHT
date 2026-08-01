"""Dynamic-programming optimization over an error-bound grid."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from ..certification import per_round_p_fail
from ..resources import ResourceModel
from ..records import EstimationConfig, RoundPlan, ScheduleResult, TrialPlan
from .round_selection import (
    _round_restart_score,
    select_exact_round,
    select_imperfect_round,
)


def build_delta_grid(
    *,
    initial_bound: float,
    target_accuracy: float,
    grid_size: int,
) -> tuple[float, ...]:
    """Build the decreasing logarithmic DP grid from Delta_0 to epsilon."""
    if not 0.0 < target_accuracy < initial_bound:
        raise ValueError("target_accuracy must lie in (0, initial_bound)")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    values = [
        float(value)
        for value in np.geomspace(initial_bound, target_accuracy, grid_size)
    ]
    values[0] = float(initial_bound)
    values[-1] = float(target_accuracy)
    return tuple(values)


def _optimize_dp(
    *,
    config: EstimationConfig,
    grid_size: int,
    max_rounds: int,
    edge_selector: Callable[[float, float, float], RoundPlan],
    label: str,
    imperfect: bool,
) -> ScheduleResult:
    """Find the best path through a fixed logarithmic error-bound grid."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    deltas = build_delta_grid(
        initial_bound=config.initial_bound,
        target_accuracy=config.target_accuracy,
        grid_size=grid_size,
    )
    target = len(deltas) - 1
    best: tuple[tuple[int, tuple[int, ...], int], TrialPlan] | None = None

    for num_rounds in range(1, max_rounds + 1):
        if num_rounds > target:
            continue
        p_fail = per_round_p_fail(
            config.total_success_probability,
            num_rounds,
        )
        # Each state stores cumulative restarts and its amplification sequence.
        # Usage: states[index] = (total_restarts, amplification_sequence)
        states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
        predecessors: list[dict[int, tuple[int, RoundPlan]]] = []
        edge_cache: dict[tuple[int, int], RoundPlan | None] = {}

        for step in range(num_rounds):
            next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
            step_predecessors: dict[int, tuple[int, RoundPlan]] = {}
            remaining = num_rounds - step - 1
            for from_index, state in states.items():
                furthest = target - remaining
                for to_index in range(from_index + 1, furthest + 1):
                    if remaining == 0 and to_index != target:
                        continue
                    key = from_index, to_index
                    if key not in edge_cache:
                        try:
                            edge_cache[key] = edge_selector(
                                deltas[from_index],
                                deltas[to_index],
                                p_fail,
                            )
                        except ValueError:
                            edge_cache[key] = None
                    edge = edge_cache[key]
                    if edge is None:
                        continue
                    edge_restarts, _ = _round_restart_score(
                        restarts=edge.restarts,
                        amplification=edge.amplification,
                    )
                    candidate = (
                        state[0] + edge_restarts,
                        state[1] + (edge.amplification,),
                    )
                    current = next_states.get(to_index)
                    if current is None or candidate < current:
                        next_states[to_index] = candidate
                        step_predecessors[to_index] = (from_index, edge)
            states = next_states
            predecessors.append(step_predecessors)
            if not states:
                break

        final_state = states.get(target)
        if final_state is None or len(predecessors) != num_rounds:
            continue
        reversed_rounds: list[RoundPlan] = []
        index = target
        for step in range(num_rounds - 1, -1, -1):
            previous, edge = predecessors[step][index]
            reversed_rounds.append(replace(edge, round_index=step))
            index = previous
        rounds = tuple(reversed(reversed_rounds))
        plan = TrialPlan(label=label, rounds=rounds)
        score = final_state + (num_rounds,)
        if best is None or score < best[0]:
            best = score, plan

    if best is None:
        state = "imperfect" if imperfect else "exact"
        raise ValueError(f"no feasible {state} DP schedule found")
    _, plan = best
    return ScheduleResult(
        config=config,
        plan=plan,
        method="dp",
        evaluated_candidates=max_rounds,
        grid_points=len(deltas),
    )


def optimize_exact_dp(
    *,
    base_config: EstimationConfig,
    resources: ResourceModel,
    grid_size: int,
    max_rounds: int,
    label: str = "exact_entangled_dp",
) -> ScheduleResult:
    """Find the restart-minimizing exact DP schedule."""
    return _optimize_dp(
        config=base_config,
        grid_size=grid_size,
        max_rounds=max_rounds,
        edge_selector=lambda before, after, p_fail: select_exact_round(
            bound_before=before,
            bound_after=after,
            config=base_config,
            p_fail=p_fail,
            resources=resources,
        ),
        label=label,
        imperfect=False,
    )


def optimize_imperfect_dp(
    *,
    base_config: EstimationConfig,
    resources: ResourceModel,
    omega_grid: Sequence[float],
    grid_size: int,
    max_rounds: int,
    label: str = "imperfect_entangled_dp",
) -> ScheduleResult:
    """Find the restart-minimizing imperfect DP schedule."""
    return _optimize_dp(
        config=base_config,
        grid_size=grid_size,
        max_rounds=max_rounds,
        edge_selector=lambda before, after, p_fail: select_imperfect_round(
            bound_before=before,
            bound_after=after,
            config=base_config,
            p_fail=p_fail,
            resources=resources,
            omega_grid=omega_grid,
        ),
        label=label,
        imperfect=True,
    )
