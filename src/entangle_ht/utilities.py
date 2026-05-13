from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DemoConfig:
    rho: float = 0.97
    phi_true: float = 0.35
    theta_0: float = 0.20
    Delta_0: float = 0.20
    epsilon: float = 0.01
    p_total: float = 0.95
    rho0: float = 0.95
    m_hw: int = 100
    gamma: float = 0.80
    omega: float = 0.60
    base_seed: int = 1234


@dataclass(frozen=True)
class AlgorithmParameters:
    m_start: int
    gamma: float
    omega: float
    c_stat: float
    c_bias: float
    num_rounds: int
    p_round: float


@dataclass(frozen=True)
class RoundPlan:
    round_index: int
    delta_bound: float
    amplification: int
    shots: int
    s_star: float
    r_star: float
    p_round: float


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    theta_ref: float
    delta_bound: float
    amplification: int
    shots: int
    s_star: float
    r_star: float
    p_round: float
    signal_theory: float
    signal_empirical: float
    clipped_signal: float
    estimate: float


@dataclass(frozen=True)
class TrialPlan:
    label: str
    total_shots: int
    rounds: List[RoundPlan]


@dataclass(frozen=True)
class TrialResult:
    label: str
    seed: int
    final_estimate: float
    final_error: float
    total_shots: int
    rounds: List[RoundRecord]


def validate_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1), got {value!r}")


def wrap_phase(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def phase_error(estimate: float, target: float) -> float:
    # Phases are periodic, so raw subtraction can make two nearby phases look
    # far apart when they lie on opposite sides of a 2*pi boundary.
    return wrap_phase(estimate - target)
