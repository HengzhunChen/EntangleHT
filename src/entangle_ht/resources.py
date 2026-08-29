from __future__ import annotations

import math
from dataclasses import dataclass

from .records import TrialPlan


@dataclass(frozen=True)
class ResourceModel:
    """Device-width model used to convert shots into packed restarts."""

    device_qubits: int = 2500
    system_qubits: int = 1

    def __post_init__(self) -> None:
        if self.system_qubits < 1:
            raise ValueError("system_qubits must be at least 1")
        if self.device_qubits < self.instance_width:
            raise ValueError(
                "device_qubits must fit at least one standard Hadamard-test instance"
            )

    @property
    def instance_width(self) -> int:
        return self.system_qubits + 1

    def packing_capacity(self, amplification: int = 1) -> int:
        if amplification < 1:
            raise ValueError(f"amplification must be at least 1, got {amplification!r}")
        return self.device_qubits // (self.instance_width * amplification)

    def max_amplification(self) -> int:
        return max(1, self.device_qubits // self.instance_width)

    def effective_m_hw(self, m_hw: int) -> int:
        """Limit a requested amplification cap to what fits on the device."""
        if m_hw < 1:
            raise ValueError(f"m_hw must be at least 1, got {m_hw!r}")
        return min(m_hw, self.max_amplification())


def restarts_from_shots(shots: int, packing_capacity: int) -> int:
    if shots < 0:
        raise ValueError(f"shots must be non-negative, got {shots!r}")
    if packing_capacity < 1:
        raise ValueError(
            f"packing_capacity must be at least 1, got {packing_capacity!r}"
        )
    return int(math.ceil(shots / packing_capacity))


def plan_restarts(plan: TrialPlan, resources: ResourceModel) -> int:
    computed = sum(
        restarts_from_shots(
            round_plan.shots,
            resources.packing_capacity(round_plan.amplification),
        )
        for round_plan in plan.rounds
    )
    if computed != plan.total_restarts:
        raise ValueError("plan restart counts are inconsistent with the resource model")
    return computed
