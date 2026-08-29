from __future__ import annotations

import math
from typing import Any

import numpy as np

from .circuit_models.common import run_round
from .circuit_models.exact import build_compiled_exact_phase_ghz_circuit
from .circuit_models.imperfect import (
    build_compiled_imperfect_phase_ghz_circuit,
    build_imperfect_cosine_circuit,
    build_imperfect_sine_circuit,
    imperfect_overlap,
)
from .records import EstimateResult, EstimationConfig, TrialPlan
from .utilities import phase_error


def clipped_arcsin_estimate(theta_ref: float, signal: float, amplification: int) -> float:
    clipped_signal = float(np.clip(signal, -1.0, 1.0))
    return theta_ref + math.asin(clipped_signal) / amplification


def standard_ht_infinite_shot_amplitude_bias(
    *,
    contrast: float,
    effective_phase: float,
    theta_ref: float,
) -> float:
    """Return the limiting amplitude bias of amplitude-unaware 1Q HT."""
    residual = phase_error(effective_phase, theta_ref)
    expected_signal = contrast * math.sin(residual)
    limiting_estimate = clipped_arcsin_estimate(
        theta_ref,
        expected_signal,
        amplification=1,
    )
    return abs(phase_error(limiting_estimate, effective_phase))


def run_exact_plan(
    *,
    config: EstimationConfig,
    plan: TrialPlan,
    simulator: Any,
    seed: int,
) -> EstimateResult:
    theta_ref = config.initial_reference
    for round_plan in plan.rounds:
        round_seed = seed + 1009 * (round_plan.round_index + 1)
        circuit = build_compiled_exact_phase_ghz_circuit(
            m=round_plan.amplification,
            theta_ref=theta_ref,
            theta_true=config.phase,
            simulator=simulator,
            seed=round_seed,
        )
        signal = run_round(
            compiled_circuit=circuit,
            shots=round_plan.shots,
            simulator=simulator,
            seed=round_seed,
        )
        theta_ref = clipped_arcsin_estimate(
            theta_ref,
            signal,
            round_plan.amplification,
        )

    error = abs(phase_error(theta_ref, config.phase))
    return EstimateResult(
        estimate=theta_ref,
        effective_error=error,
        target_error=error,
    )


def run_imperfect_plan(
    *,
    config: EstimationConfig,
    plan: TrialPlan,
    theta_target: float,
    eta: float,
    simulator: Any,
    seed: int,
) -> EstimateResult:
    """Run an iterative plan for the imperfect-eigenstate model."""
    _, theta_psi, _ = imperfect_overlap(
        theta_target=theta_target,
        eta=eta,
    )
    theta_ref = config.initial_reference
    for round_plan in plan.rounds:
        round_seed = seed + 1009 * (round_plan.round_index + 1)
        circuit = build_compiled_imperfect_phase_ghz_circuit(
            m=round_plan.amplification,
            theta_ref=theta_ref,
            theta_target=theta_target,
            eta=eta,
            simulator=simulator,
            seed=round_seed,
        )
        signal = run_round(
            compiled_circuit=circuit,
            shots=round_plan.shots,
            simulator=simulator,
            seed=round_seed,
        )
        theta_ref = clipped_arcsin_estimate(
            theta_ref,
            signal,
            round_plan.amplification,
        )

    return EstimateResult(
        estimate=theta_ref,
        effective_error=abs(phase_error(theta_ref, theta_psi)),
        target_error=abs(phase_error(theta_ref, theta_target)),
    )


def run_one_quadrature_exact_standard(
    *,
    theta_true: float,
    theta_ref: float,
    shots: int,
    simulator: Any,
    seed: int,
) -> EstimateResult:
    circuit = build_compiled_exact_phase_ghz_circuit(
        m=1,
        theta_ref=theta_ref,
        theta_true=theta_true,
        simulator=simulator,
        seed=seed,
    )
    signal = run_round(
        compiled_circuit=circuit,
        shots=shots,
        simulator=simulator,
        seed=seed,
    )
    estimate = clipped_arcsin_estimate(theta_ref, signal, 1)
    error = abs(phase_error(estimate, theta_true))
    return EstimateResult(
        estimate=estimate,
        effective_error=error,
        target_error=error,
    )


def run_one_quadrature_imperfect_standard(
    *,
    theta_target: float,
    eta: float,
    theta_ref: float,
    shots: int,
    simulator: Any,
    seed: int,
) -> EstimateResult:
    """Run the amplitude-unaware 1Q baseline on an imperfect state."""
    _, theta_psi, _ = imperfect_overlap(
        theta_target=theta_target,
        eta=eta,
    )
    circuit = build_compiled_imperfect_phase_ghz_circuit(
        m=1,
        theta_ref=theta_ref,
        theta_target=theta_target,
        eta=eta,
        simulator=simulator,
        seed=seed,
    )
    signal = run_round(
        compiled_circuit=circuit,
        shots=shots,
        simulator=simulator,
        seed=seed,
    )
    estimate = clipped_arcsin_estimate(theta_ref, signal, 1)
    return EstimateResult(
        estimate=estimate,
        effective_error=abs(phase_error(estimate, theta_psi)),
        target_error=abs(phase_error(estimate, theta_target)),
    )


def run_two_quadrature_imperfect_standard(
    *,
    theta_target: float,
    eta: float,
    shots_per_quadrature: int,
    simulator: Any,
    seed: int,
) -> EstimateResult:
    """Run the two-quadrature baseline for the imperfect-eigenstate model."""
    from qiskit import transpile

    _, theta_psi, _ = imperfect_overlap(
        theta_target=theta_target,
        eta=eta,
    )

    sine_circuit, cosine_circuit = transpile(
        [
            build_imperfect_sine_circuit(theta_target, eta),
            build_imperfect_cosine_circuit(theta_target, eta),
        ],
        simulator,
        optimization_level=1,
        seed_transpiler=seed,
    )
    sine_signal = run_round(
        compiled_circuit=sine_circuit,
        shots=shots_per_quadrature,
        simulator=simulator,
        seed=seed,
    )

    cosine_seed = seed + 17
    cosine_signal = run_round(
        compiled_circuit=cosine_circuit,
        shots=shots_per_quadrature,
        simulator=simulator,
        seed=cosine_seed,
    )
    estimate = math.atan2(sine_signal, cosine_signal)

    return EstimateResult(
        estimate=estimate,
        effective_error=abs(phase_error(estimate, theta_psi)),
        target_error=abs(phase_error(estimate, theta_target)),
    )


def require_qiskit_aer() -> None:
    try:
        import qiskit_aer  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Circuit simulation requires qiskit-aer. Install the project "
            "simulation dependencies before running this example."
        ) from exc


def build_simulator(
    noise_settings: dict[str, float | str] | None = None,
) -> Any:
    """Build the simulator used by one error-decay experiment."""
    if noise_settings is not None:
        from .circuit_models.simulators import build_noisy_simulator

        return build_noisy_simulator(**noise_settings)

    from .circuit_models.simulators import build_simulator as build_ideal_simulator

    return build_ideal_simulator(method="statevector")
