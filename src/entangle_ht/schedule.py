from __future__ import annotations

import math
from typing import Any, List

import numpy as np

from .circuits import run_round, theoretical_parity_mean
from .utilities import (
    AlgorithmParameters,
    DemoConfig,
    RoundRecord,
    TrialResult,
    phase_error,
)


def f_t(m: int, delta_t: float, rho0: float) -> float:
    angle = m * delta_t
    if angle <= 0.0:
        return 0.0
    if angle >= math.pi / 2:
        return math.inf
    return (1.0 - rho0**m) * math.tan(angle) / angle


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1), got {value!r}")


def max_feasible_amplification(delta_t: float, c_bias: float, rho0: float, m_hw: int) -> int:
    # Binary search for the largest m such that
    #   f_t(m, delta_t, rho0) <= c_bias,
    # subject to m <= m_hw and the safety condition m * delta_t < pi/2.
    if delta_t <= 0.0:
        raise ValueError(f"delta_t must be positive, got {delta_t!r}")
    if not 0.0 < c_bias < 1.0:
        raise ValueError(f"c_bias must lie in (0, 1), got {c_bias!r}")
    if not 0.0 < rho0 <= 1.0:
        raise ValueError(f"rho0 must lie in (0, 1], got {rho0!r}")
    if m_hw < 1:
        raise ValueError(f"m_hw must be at least 1, got {m_hw!r}")
    
    if rho0 == 1.0:
        m_max = m_hw
    else:
        m_max = max(1, int(math.floor(math.log(1.0 - c_bias) / math.log(rho0))))

    local_cap = int(math.ceil(math.pi / (2*delta_t)) - 1)
    upper = min(m_hw, m_max, local_cap)

    if f_t(1, delta_t, rho0) > c_bias:
        raise ValueError(
            "Current schedule is infeasible: F_t(1) exceeds c_bias. "
            "Retune the design parameters before running the algorithm."
        )
    if upper < 1:
        raise ValueError(
            "Current schedule is infeasible: the safety condition "
            "m_t * Delta_t < 1 fails even for m_t=1."
        )

    left, right = 1, upper
    while left < right:
        mid = (left + right + 1) // 2
        if f_t(mid, delta_t, rho0) <= c_bias:
            left = mid
        else:
            right = mid - 1
    return left


def optimized_inversion_parameters(
    m_t: int,
    delta_t: float,
    c_stat: float,
) -> tuple[float, float]:
    """Return the optimized (s_t^*, r_t^*) from the high-probability bound."""
    if m_t < 1:
        raise ValueError(f"m_t must be at least 1, got {m_t!r}")
    if delta_t <= 0.0:
        raise ValueError(f"delta_t must be positive, got {delta_t!r}")
    if c_stat <= 0.0:
        raise ValueError(f"c_stat must be positive, got {c_stat!r}")
    if m_t * delta_t >= (math.pi / 2):
        raise ValueError(
            "The optimized high-probability shot rule requires m_t * Delta_t < pi/2."
        )

    a_t = math.sin(m_t * delta_t)
    q_t = m_t * c_stat * delta_t
    root = math.sqrt(max(0.0, 1.0 - a_t**2 + q_t**2))
    s_star = (a_t + q_t * root) / (1.0 + q_t**2)
    r_star = s_star - a_t
    if r_star <= 0.0:
        raise ValueError(
            "Optimized inversion radius is non-positive. "
            "Retune c_stat, Delta_t, or the amplification schedule."
        )
    return s_star, r_star


def num_shots_for_round(m_t: int, delta_t: float, c_stat: float, p_round: float) -> int:
    _validate_probability(p_round, "p_round")
    _, r_star = optimized_inversion_parameters(
        m_t=m_t,
        delta_t=delta_t,
        c_stat=c_stat,
    )
    return int(math.ceil((2.0 / r_star**2) * math.log(2.0 / (1.0 - p_round))))


def compute_num_rounds(delta_0: float, epsilon: float, gamma: float) -> int:
    if delta_0 <= 0.0:
        raise ValueError(f"delta_0 must be positive, got {delta_0!r}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon!r}")
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie in (0, 1), got {gamma!r}")
    if epsilon >= delta_0:
        return 0
    return int(math.ceil(math.log(delta_0 / epsilon) / math.log(1.0 / gamma)))


def per_round_success_probability(p_total: float, num_rounds: int) -> float:
    _validate_probability(p_total, "p_total")
    if num_rounds < 0:
        raise ValueError(f"num_rounds cannot be negative, got {num_rounds!r}")
    if num_rounds == 0:
        return p_total
    return p_total ** (1.0 / num_rounds)


def verify_initial_feasibility(delta_0: float, c_bias: float, rho0: float) -> None:
    if delta_0 >= (math.pi / 2):
        raise ValueError(
            "Initial classical schedule is infeasible: Delta_0 must be less than 1 "
            "to satisfy the safety condition m_t * Delta_t < pi/2 for m_t=1."
        )
    if f_t(1, delta_0, rho0) > c_bias:
        raise ValueError(
            "Initial classical schedule is infeasible: F_0(1) exceeds c_bias. "
            "Retune the design parameters before running the algorithm."
        )


def max_starting_amplification(delta_0: float, rho0: float, m_hw: int) -> int:
    if delta_0 <= 0.0:
        raise ValueError(f"delta_0 must be positive, got {delta_0!r}")
    if delta_0 >= (math.pi / 2):
        raise ValueError(
            "Initial classical schedule is infeasible: Delta_0 must be less than pi/2."
        )
    upper = min(m_hw, int(math.ceil(math.pi / (2*delta_0))) - 1)
    if upper < 1:
        raise ValueError(
            "No starting amplification satisfies the safety condition "
            "m_start * Delta_0 < 1."
        )
    feasible = None
    for candidate in range(1, upper + 1):
        if f_t(candidate, delta_0, rho0) < 1.0:
            feasible = candidate
    if feasible is None:
        raise ValueError(
            "No starting amplification satisfies F_0(m_start) < 1. "
            "Retune Delta_0 or rho0."
        )
    return feasible


def design_algorithm_parameters(config: DemoConfig) -> AlgorithmParameters:
    m_start = max_starting_amplification(config.Delta_0, config.rho0, config.m_hw)
    
    gamma = config.gamma
    omega = config.omega
    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie in (0, 1), got {gamma!r}")
    if not 0.0 < omega < 1.0:
        raise ValueError(f"omega must lie in (0, 1), got {omega!r}")

    c_stat = omega * gamma
    c_bias = (1 - omega) * gamma
    verify_initial_feasibility(config.Delta_0, c_bias, config.rho0)
    
    num_rounds = compute_num_rounds(config.Delta_0, config.epsilon, gamma)
    p_round = per_round_success_probability(config.p_total, num_rounds)
    
    return AlgorithmParameters(
        m_start=m_start,
        gamma=gamma,
        omega=omega,
        c_stat=c_stat,
        c_bias=c_bias,
        num_rounds=num_rounds,
        p_round=p_round,
    )


def run_trial(
    config: DemoConfig,
    algorithm: AlgorithmParameters,
    simulator: Any,
    seed: int,
    label: str,
) -> TrialResult:
    alpha = config.rho * np.exp(1j * config.phi_true)
    theta_t = config.theta_0
    delta_t = config.Delta_0
    rounds: List[RoundRecord] = []
    total_shots = 0

    for round_index in range(algorithm.num_rounds):
        m_t = max_feasible_amplification(
            delta_t=delta_t,
            c_bias=algorithm.c_bias,
            rho0=config.rho0,
            m_hw=config.m_hw,
        )

        s_star, r_star = optimized_inversion_parameters(
            m_t=m_t,
            delta_t=delta_t,
            c_stat=algorithm.c_stat,
        )
        shots = num_shots_for_round(
            m_t=m_t,
            delta_t=delta_t,
            c_stat=algorithm.c_stat,
            p_round=algorithm.p_round,
        )
        signal_empirical = run_round(
            m=m_t,
            theta_ref=theta_t,
            shots=shots,
            alpha=alpha,
            simulator=simulator,
            seed=seed + 1009 * (round_index + 1),
        )
        clipped_signal = float(np.clip(signal_empirical, -1.0, 1.0))
        estimate = theta_t + math.asin(clipped_signal) / m_t
        rounds.append(
            RoundRecord(
                round_index=round_index,
                theta_ref=theta_t,
                delta_bound=delta_t,
                amplification=m_t,
                shots=shots,
                s_star=s_star,
                r_star=r_star,
                p_round=algorithm.p_round,
                signal_theory=theoretical_parity_mean(
                    rho=config.rho,
                    phi_true=config.phi_true,
                    theta_ref=theta_t,
                    m=m_t,
                ),
                signal_empirical=signal_empirical,
                clipped_signal=clipped_signal,
                estimate=estimate,
            )
        )
        theta_t = estimate
        delta_t *= algorithm.gamma
        total_shots += shots

    return TrialResult(
        label=label,
        seed=seed,
        final_estimate=theta_t,
        final_error=phase_error(theta_t, config.phi_true),
        total_shots=total_shots,
        rounds=rounds,
    )


def print_run_summary(config: DemoConfig, trial: TrialResult) -> None:
    algorithm = design_algorithm_parameters(config)

    print("Single adaptive entangle_ht run")
    print(
        "Setup: "
        f"rho={config.rho}, rho0={config.rho0}, "
        f"phi_true={config.phi_true}, theta_0={config.theta_0}, "
        f"Delta_0={config.Delta_0}, target_RMSE={config.epsilon}, "
        f"p_total={config.p_total}, m_hw={config.m_hw}"
    )
    print(
        "Algorithm: "
        f"gamma={algorithm.gamma:.6f}, omega={algorithm.omega:.6f}, "
        f"c_stat={algorithm.c_stat:.6f}, c_bias={algorithm.c_bias:.6f}, "
        f"rounds={algorithm.num_rounds}, p_round={algorithm.p_round:.8f}"
    )
    print()
    print("round\ttheta_ref\tDelta_t\tm_t\tr_star\tshots\tsignal\testimate")
    for round_record in trial.rounds:
        print(
            f"{round_record.round_index}\t"
            f"{round_record.theta_ref:.8f}\t"
            f"{round_record.delta_bound:.8f}\t"
            f"{round_record.amplification}\t"
            f"{round_record.r_star:.8f}\t"
            f"{round_record.shots}\t"
            f"{round_record.signal_empirical:.8f}\t"
            f"{round_record.estimate:.8f}"
        )

    print()
    print(f"Final estimate: {trial.final_estimate:.10f}")
    print(f"True phase:      {config.phi_true:.10f}")
    print(f"Final error:     {trial.final_error:.10f}")
    print(f"Total shots:     {trial.total_shots}")
