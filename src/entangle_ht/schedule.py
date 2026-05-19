from __future__ import annotations

import math
from typing import Any, List

import numpy as np

from .circuits import (
    build_compiled_sine_ghz_circuit, 
    run_round, 
)
from .utilities import (
    AlgorithmParameters,
    DemoConfig,
    RoundPlan,
    RoundRecord,
    TrialPlan,
    TrialResult,
    phase_error,
    validate_probability,
)


# ----------------------------------------------------------------------------
# Shot-optimized amplification
# ----------------------------------------------------------------------------

def contrast_bias_bound(m: int, delta_t: float, rho0: float) -> float:
    """Certified bias bound B(m; Delta_t, rho0)"""
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


def shot_optimized_amplification(
    delta_t: float,
    epsilon_stat: float,
    epsilon_bias: float,
    rho0: float,
    m_hw: int,
) -> tuple[int, float]:
    """Choose the feasible amplification with the largest statistical radius.

    This implements only for the iterative algorithm, where
    ``epsilon_stat`` is the constant budget c_stat * Delta_t.
    """
    if delta_t <= 0.0:
        raise ValueError(f"delta_t must be positive, got {delta_t!r}")
    if epsilon_bias <= 0.0:
        raise ValueError(f"epsilon_bias must be positive, got {epsilon_bias!r}")
    if not 0.0 < rho0 <= 1.0:
        raise ValueError(f"rho0 must lie in (0, 1], got {rho0!r}")
    if m_hw < 1:
        raise ValueError(f"m_hw must be at least 1, got {m_hw!r}")

    # necessary condition for feasibility
    if rho0 == 1.0 or epsilon_bias >= delta_t:
        m_nec = m_hw
    else:
        m_nec = int(math.floor(math.log(1.0 - epsilon_bias / delta_t) / math.log(rho0)))

    local_cap = int(math.ceil(math.pi / (2 * delta_t)) - 1)
    search_cap = min(m_hw, m_nec, local_cap)

    if contrast_bias_bound(1, delta_t, rho0) > epsilon_bias:
        raise ValueError(
            "Current schedule is infeasible: B(1; Delta_t, rho0) exceeds epsilon_bias. "
            "Retune the design parameters before running the algorithm."
        )
    if search_cap < 1:
        raise ValueError(
            "Current schedule is infeasible: the safety condition "
            "m_t * Delta_t < pi/2 fails even for m_t=1."
        )

    left, right = 1, search_cap
    while left < right:
        mid = (left + right + 1) // 2
        if contrast_bias_bound(mid, delta_t, rho0) <= epsilon_bias:
            left = mid
        else:
            right = mid - 1

    max_feasible = left

    m_best = 1
    r_best = -math.inf
    for candidate in range(1, max_feasible + 1):
        r_star = optimized_inversion_radius(
            m_t=candidate,
            delta_t=delta_t,
            epsilon_stat=epsilon_stat,
        )
        if r_star > r_best:
            m_best = candidate
            r_best = r_star

    return m_best, r_best


# ----------------------------------------------------------------------------
# Optimized inversion radius
# ----------------------------------------------------------------------------

def optimized_inversion_radius(
    m_t: int,
    delta_t: float,
    epsilon_stat: float,
) -> float:
    """Return the optimized r_t^* from the high-probability bound."""
    if m_t < 1:
        raise ValueError(f"m_t must be at least 1, got {m_t!r}")
    if delta_t <= 0.0:
        raise ValueError(f"delta_t must be positive, got {delta_t!r}")
    if epsilon_stat <= 0.0:
        raise ValueError(f"epsilon_stat must be positive, got {epsilon_stat!r}")
    if m_t * delta_t >= (math.pi / 2):
        raise ValueError(
            "The optimized high-probability shot rule requires "
            "m_t * Delta_t < pi/2."
        )

    a_t = math.sin(m_t * delta_t)
    q_t = m_t * epsilon_stat
    root = math.sqrt(max(0.0, 1.0 - a_t**2 + q_t**2))
    r_star = (a_t + q_t * root) / (1.0 + q_t**2) - a_t

    if r_star <= 0.0:
        raise ValueError(
            "Optimized inversion radius is non-positive. "
            "Retune c_stat, Delta_t, or the amplification schedule."
        )

    return r_star


# ----------------------------------------------------------------------------
# Shot count calculation
# ----------------------------------------------------------------------------

def num_shots_for_round(
    m_t: int,
    delta_t: float,
    p_round: float,
    epsilon_stat: float | None = None,
    r_star: float | None = None,
) -> int:
    validate_probability(p_round, "p_round")

    if r_star is None:
        if epsilon_stat is None:
            raise ValueError("epsilon_stat must be provided when r_star is not precomputed")
        r_star = optimized_inversion_radius(
            m_t=m_t,
            delta_t=delta_t,
            epsilon_stat=epsilon_stat,
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
    validate_probability(p_total, "p_total")
    if num_rounds < 0:
        raise ValueError(f"num_rounds must be non-negative, got {num_rounds!r}")
    if num_rounds == 0:
        return 1.0
    return p_total ** (1.0 / num_rounds)


# ---------------------------------------------------------------------------
# Initial checks
# ---------------------------------------------------------------------------

def verify_initial_feasibility(delta_0: float, c_bias: float, rho0: float) -> None:
    if delta_0 >= (math.pi / 2):
        raise ValueError(
            "Initial classical schedule is infeasible: Delta_0 must be less than "
            "pi/2 to satisfy the safety condition m_t * Delta_t < pi/2 for m_t=1."
        )

    if contrast_bias_bound(1, delta_0, rho0) > c_bias * delta_0:
        raise ValueError(
            "Initial classical schedule is infeasible: B(1; Delta_0, rho0) exceeds "
            "c_bias * Delta_0. "
            "Retune the design parameters before running the algorithm."
        )


# ---------------------------------------------------------------------------
# Parameters setup
# ---------------------------------------------------------------------------

def design_algorithm_parameters(config: DemoConfig) -> AlgorithmParameters:
    gamma = config.gamma
    omega = config.omega

    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must lie in (0, 1), got {gamma!r}")
    if not 0.0 < omega < 1.0:
        raise ValueError(f"omega must lie in (0, 1), got {omega!r}")

    c_stat = omega * gamma
    c_bias = (1.0 - omega) * gamma

    verify_initial_feasibility(
        delta_0=config.Delta_0,
        c_bias=c_bias,
        rho0=config.rho0,
    )

    num_rounds = compute_num_rounds(
        delta_0=config.Delta_0,
        epsilon=config.epsilon,
        gamma=gamma,
    )
    p_round = per_round_success_probability(
        p_total=config.p_total,
        num_rounds=num_rounds,
    )

    return AlgorithmParameters(
        gamma=gamma,
        omega=omega,
        c_stat=c_stat,
        c_bias=c_bias,
        num_rounds=num_rounds,
        p_round=p_round,
    )


# ---------------------------------------------------------------------------
# Classical planning layer
# ---------------------------------------------------------------------------

def plan_trial(
    config: DemoConfig,
    algorithm: AlgorithmParameters,
    label: str,
) -> TrialPlan:
    """Compute the full classical schedule without running any Qiskit simulation.

    This determines:
      - delta bound for each round,
      - amplification m_t for each round,
      - optimized inversion parameters,
      - shots for each round,
      - total shots.

    Those require actual simulation data and are handled by run_trial().
    """
    delta_t = config.Delta_0
    rounds: List[RoundPlan] = []
    total_shots = 0

    for round_index in range(algorithm.num_rounds):
        epsilon_stat_t = algorithm.c_stat * delta_t
        epsilon_bias_t = algorithm.c_bias * delta_t
        m_t, r_star = shot_optimized_amplification(
            delta_t=delta_t,
            epsilon_stat=epsilon_stat_t,
            epsilon_bias=epsilon_bias_t,
            rho0=config.rho0,
            m_hw=config.m_hw,
        )

        shots = num_shots_for_round(
            m_t=m_t,
            delta_t=delta_t,
            p_round=algorithm.p_round,
            r_star=r_star,
        )

        rounds.append(
            RoundPlan(
                round_index=round_index,
                delta_bound=delta_t,
                amplification=m_t,
                shots=shots,
                r_star=r_star,
                p_round=algorithm.p_round,
            )
        )

        total_shots += shots
        delta_t *= algorithm.gamma

    return TrialPlan(
        label=label,
        total_shots=total_shots,
        rounds=rounds,
    )


# ---------------------------------------------------------------------------
# Quantum execution layer
# ---------------------------------------------------------------------------

def run_trial(
    config: DemoConfig,
    algorithm: AlgorithmParameters,
    simulator: Any,
    seed: int,
    label: str,
    plan: TrialPlan | None = None,
) -> TrialResult:
    """
    Run the adaptive algorithm using a precomputed classical schedule.
    If plan is not provided, it is computed internally.
    """
    if plan is None:
        plan = plan_trial(
            config=config,
            algorithm=algorithm,
            label=label,
        )

    alpha = config.rho * np.exp(1j * config.phi_true)
    theta_t = config.theta_0
    rounds: List[RoundRecord] = []

    for round_plan in plan.rounds:
        round_seed = seed + 1009 * (round_plan.round_index + 1)

        compiled_circuit = build_compiled_sine_ghz_circuit(
            m=round_plan.amplification,
            theta_ref=theta_t,
            alpha=alpha,
            simulator=simulator,
            seed=round_seed,
        )
        signal_empirical = run_round(
            compiled_circuit=compiled_circuit,
            shots=round_plan.shots,
            simulator=simulator,
            seed=round_seed,
        )

        clipped_signal = float(np.clip(signal_empirical, -1.0, 1.0))
        estimate = theta_t + math.asin(clipped_signal) / round_plan.amplification

        rounds.append(
            RoundRecord(
                round_index=round_plan.round_index,
                theta_ref=theta_t,
                delta_bound=round_plan.delta_bound,
                amplification=round_plan.amplification,
                shots=round_plan.shots,
                r_star=round_plan.r_star,
                p_round=round_plan.p_round,
                signal_empirical=signal_empirical,
                clipped_signal=clipped_signal,
                estimate=estimate,
            )
        )

        theta_t = estimate

    return TrialResult(
        label=label,
        seed=seed,
        final_estimate=theta_t,
        final_error=phase_error(theta_t, config.phi_true),
        total_shots=plan.total_shots,
        rounds=rounds,
    )


# ---------------------------------------------------------------------------
# Print helper functions
# ---------------------------------------------------------------------------

def print_trial_plan_summary(
    config: DemoConfig,
    algorithm: AlgorithmParameters,
    plan: TrialPlan,
) -> None:
    """Print the classical schedule without requiring simulation results."""
    print("Classical adaptive entangle_ht schedule")
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
    print("round\tDelta_t\tm_t\tr_star\tshots")
    for round_plan in plan.rounds:
        print(
            f"{round_plan.round_index}\t"
            f"{round_plan.delta_bound:.8f}\t"
            f"{round_plan.amplification}\t"
            f"{round_plan.r_star:.8f}\t"
            f"{round_plan.shots}"
        )

    print()
    print(f"Total shots: {plan.total_shots}")


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
