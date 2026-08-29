"""Immutable configuration, planning, and simulation records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EstimationConfig:
    """Physical and statistical inputs shared by all schedule planners.

    ``phase`` is the eigenphase in the exact case and the effective overlap
    phase in the imperfect case.  Search parameters such as ``gamma`` and
    ``omega`` deliberately do not live here; they describe a scheduler, not
    the estimation problem.
    """

    phase: float = 0.35
    initial_reference: float = 0.20
    initial_bound: float = 0.20
    target_accuracy: float = 0.01
    total_success_probability: float = 0.95
    branch_margin: float = 0.3
    hardware_amplification_cap: int = 100
    contrast: float = 1.0
    contrast_lower_bound: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_accuracy < self.initial_bound:
            raise ValueError("target_accuracy must lie in (0, initial_bound)")
        if not 0.0 < self.total_success_probability < 1.0:
            raise ValueError("total_success_probability must lie in (0, 1)")
        if not 0.0 < self.branch_margin < math.pi / 2:
            raise ValueError("branch_margin must lie in (0, pi/2)")
        if self.initial_bound > self.branch_margin:
            raise ValueError("initial_bound must not exceed branch_margin")
        if self.hardware_amplification_cap < 1:
            raise ValueError("hardware_amplification_cap must be at least 1")
        if not 0.0 < self.contrast_lower_bound <= 1.0:
            raise ValueError("contrast_lower_bound must lie in (0, 1]")


@dataclass(frozen=True)
class RoundPlan:
    """A fully reproducible, certified transition between two error bounds."""

    round_index: int
    bound_before: float
    bound_after: float
    amplification: int
    statistical_accuracy: float
    amplitude_bias_budget: float
    shots: int
    restarts: int
    inverse_radius: float
    p_fail: float
    omega: float | None = None

    @property
    def queries(self) -> int:
        return self.amplification * self.shots


@dataclass(frozen=True)
class TrialPlan:
    label: str
    rounds: tuple[RoundPlan, ...]

    @property
    def total_shots(self) -> int:
        return sum(round_plan.shots for round_plan in self.rounds)

    @property
    def total_restarts(self) -> int:
        return sum(round_plan.restarts for round_plan in self.rounds)

    @property
    def total_queries(self) -> int:
        return sum(round_plan.queries for round_plan in self.rounds)

    @property
    def max_amplification(self) -> int:
        return max((round_plan.amplification for round_plan in self.rounds), default=1)


@dataclass(frozen=True)
class ScheduleResult:
    config: EstimationConfig
    plan: TrialPlan
    method: Literal["geometric", "dp"]
    evaluated_candidates: int
    gamma: float | None = None
    omega: float | None = None
    grid_points: int | None = None

    @property
    def rounds(self) -> int:
        return len(self.plan.rounds)

    @property
    def max_amplification(self) -> int:
        return self.plan.max_amplification

    @property
    def restarts(self) -> int:
        return self.plan.total_restarts


@dataclass(frozen=True)
class EstimateResult:
    estimate: float
    effective_error: float
    target_error: float


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    reference: float
    plan: RoundPlan
    empirical_signal: float
    clipped_signal: float
    estimate: float


@dataclass(frozen=True)
class TrialResult:
    label: str
    seed: int
    final_estimate: float
    final_error: float
    plan: TrialPlan
    rounds: tuple[RoundRecord, ...]
