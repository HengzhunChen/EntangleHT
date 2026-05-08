#!/usr/bin/env python3
"""
Demonstrate the measurement reduction from adaptive amplification.

This script supports two workflows:

1. Classical planning only:
   - Computes planned shot counts, rounds, and max amplification.
   - Does not build or run a Qiskit simulator.
   - Useful for fast measurement-cost estimates.

2. Full Qiskit simulation:
   - Reuses the classical plans.
   - Runs Qiskit circuits to estimate empirical errors.
   - Useful for validating the observed accuracy behavior.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from entangle_ht.circuits import build_simulator, run_round
from entangle_ht.schedule import (
    design_algorithm_parameters,
    plan_trial,
    run_trial,
)
from entangle_ht.utilities import DemoConfig, phase_error


# ----------------------------------------------------------
# For test simulation error
BASE_CONFIG = replace(
    DemoConfig(),
    rho=0.995,
    rho0=0.97,
    phi_true=0.35,
    theta_0=0.20,
    Delta_0=0.20,
    p_total=0.95,
    gamma=0.8,
    omega=0.6,
    m_hw=100,
    base_seed=20260416,
)
# EPSILON_GRID = (0.08, 0.07, 0.06, 0.05, 0.04)  # Quick run test
EPSILON_GRID = (0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125)
OUTPUT_DIR = Path("outputs")
# ----------------------------------------------------------

# # ----------------------------------------------------------
# # For test measurement plan scaling
# BASE_CONFIG = replace(
#     DemoConfig(),
#     rho=0.999,
#     rho0=0.99,
#     phi_true=0.35,
#     theta_0=0.20,
#     Delta_0=0.20,
#     p_total=0.95,
#     gamma=0.85,
#     omega=0.65,
#     m_hw=100,
#     base_seed=20260416,
# )
# EPSILON_GRID = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4)
# OUTPUT_DIR = Path("temp")
# # ----------------------------------------------------------


def configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(".cache/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())


def save_figure(fig, output_path: Path, *, dpi: int = 180) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi)
    return png_path


def num_shot_standard_hadamard_test(epsilon: float, p_success: float) -> int:
    # Ideal m=1, unit-contrast with the same Chebyshev success-probability.
    return int(math.ceil(1.0 / ((1.0 - p_success) * epsilon**2)))


def run_standard_hadamard(
    config: DemoConfig,
    simulator,
    seed: int,
) -> Dict[str, float]:
    shots = num_shot_standard_hadamard_test(
        epsilon=config.epsilon,
        p_success=config.p_total,
    )

    alpha = config.rho * complex(
        math.cos(config.phi_true),
        math.sin(config.phi_true),
    )

    signal_empirical = run_round(
        m=1,
        theta_ref=config.theta_0,
        shots=shots,
        alpha=alpha,
        simulator=simulator,
        seed=seed,
    )

    clipped_signal = float(min(1.0, max(-1.0, signal_empirical)))
    estimate = config.theta_0 + math.asin(clipped_signal)

    # Even with infinite shots, the fixed-reference m=1 Hadamard test inverts
    # rho * sin(phi - theta_ref), so rho < 1 leaves an irreducible bias floor.
    signal_limit = config.rho * math.sin(config.phi_true - config.theta_0)
    clipped_limit = float(min(1.0, max(-1.0, signal_limit)))
    limit_estimate = config.theta_0 + math.asin(clipped_limit)

    return {
        "shots": float(shots),
        "actual_error": abs(phase_error(estimate, config.phi_true)),
        "bias_floor": abs(phase_error(limit_estimate, config.phi_true)),
    }


# ---------------------------------------------------------------------------
# Planning-only experiment
# ---------------------------------------------------------------------------

def plan_accuracy_grid() -> List[Dict[str, float]]:
    """Compute measurement costs without running Qiskit simulation.

    Returned rows contain only deterministic, classical planning data:
      - shot counts,
      - number of rounds,
      - max amplification.
    """
    rows: List[Dict[str, float]] = []

    for epsilon in EPSILON_GRID:
        config = replace(BASE_CONFIG, epsilon=epsilon)

        algorithm = design_algorithm_parameters(config)
        entangled_plan = plan_trial(
            config=config,
            algorithm=algorithm,
            label="entangle",
        )

        one_register_config = replace(config, m_hw=1)
        one_register_algorithm = design_algorithm_parameters(one_register_config)
        one_register_plan = plan_trial(
            config=one_register_config,
            algorithm=one_register_algorithm,
            label="one_register",
        )

        standard_shots = num_shot_standard_hadamard_test(
            epsilon=config.epsilon,
            p_success=config.p_total,
        )

        max_m_used = max(
            round_plan.amplification for round_plan in entangled_plan.rounds
        )

        rows.append(
            {
                "epsilon": epsilon,
                "entangle_shots": float(entangled_plan.total_shots),
                "one_register_shots": float(one_register_plan.total_shots),
                "standard_hadamard_shots": float(standard_shots),
                "max_m_used": float(max_m_used),
                "entangle_rounds": float(algorithm.num_rounds),
                "one_register_rounds": float(one_register_algorithm.num_rounds),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Full Qiskit simulation experiment
# ---------------------------------------------------------------------------

def simulate_accuracy_grid(
    planning_rows: Sequence[Dict[str, float]] | None = None,
) -> List[Dict[str, float]]:
    """Run Qiskit simulations and append empirical error data.

    This function reuses the same classical schedule logic as plan_accuracy_grid().
    The returned rows include both planning data and simulation-derived errors.
    """
    simulator = build_simulator()
    rows: List[Dict[str, float]] = []

    if planning_rows is None:
        planning_rows = plan_accuracy_grid()

    for index, planning_row in enumerate(planning_rows):
        epsilon = planning_row["epsilon"]
        config = replace(BASE_CONFIG, epsilon=epsilon)

        algorithm = design_algorithm_parameters(config)
        entangled_plan = plan_trial(
            config=config,
            algorithm=algorithm,
            label="entangle",
        )
        entangled_trial = run_trial(
            config=config,
            algorithm=algorithm,
            simulator=simulator,
            seed=config.base_seed + 97 * index,
            label="entangle",
            plan=entangled_plan,
        )

        one_register_config = replace(config, m_hw=1)
        one_register_algorithm = design_algorithm_parameters(one_register_config)
        one_register_plan = plan_trial(
            config=one_register_config,
            algorithm=one_register_algorithm,
            label="one_register",
        )
        one_register_trial = run_trial(
            config=one_register_config,
            algorithm=one_register_algorithm,
            simulator=simulator,
            seed=config.base_seed + 10_000 + 97 * index,
            label="one_register",
            plan=one_register_plan,
        )

        standard_hadamard = run_standard_hadamard(
            config=config,
            simulator=simulator,
            seed=config.base_seed + 20_000 + 97 * index,
        )

        rows.append(
            {
                **planning_row,
                "entangle_estimate": entangled_trial.final_estimate,
                "entangle_error": abs(entangled_trial.final_error),
                "one_register_estimate": one_register_trial.final_estimate,
                "one_register_error": abs(one_register_trial.final_error),
                "standard_hadamard_error": standard_hadamard["actual_error"],
                "standard_hadamard_bias_floor": standard_hadamard["bias_floor"],
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_planning_table(rows: Sequence[Dict[str, float]]) -> None:
    print(
        "epsilon\tIterative entangled HT shots\t"
        "Iterative non-entangled HT shots\tStandard HT shots\t"
        "max_m\tIterative entangled HT rounds\t"
        "Iterative non-entangled HT rounds"
    )

    for row in rows:
        print(
            f"{row['epsilon']:.6g}\t"
            f"{int(row['entangle_shots'])}\t"
            f"{int(row['one_register_shots'])}\t"
            f"{int(row['standard_hadamard_shots'])}\t"
            f"{int(row['max_m_used'])}\t"
            f"{int(row['entangle_rounds'])}\t"
            f"{int(row['one_register_rounds'])}"
        )


def print_measurement_ratio_table(rows: Sequence[Dict[str, float]]) -> None:
    print()
    print("Measurement ratios")
    print(
        "epsilon\tIterative non-entangled HT / Iterative entangled HT\t"
        "Standard HT / Iterative entangled HT"
    )

    for row in rows:
        non_entangled_ratio = row["one_register_shots"] / row["entangle_shots"]
        standard_ratio = row["standard_hadamard_shots"] / row["entangle_shots"]

        print(
            f"{row['epsilon']:.6g}\t"
            f"{non_entangled_ratio:.4f}\t"
            f"{standard_ratio:.4f}"
        )


def print_error_table(rows: Sequence[Dict[str, float]]) -> None:
    print()
    print("Simulation errors")
    print(
        "epsilon\tIterative entangled HT error\t"
        "Iterative non-entangled HT error\tStandard HT error\t"
        "Standard HT bias floor"
    )

    for row in rows:
        print(
            f"{row['epsilon']:.6g}\t"
            f"{row['entangle_error']:.8f}\t"
            f"{row['one_register_error']:.8f}\t"
            f"{row['standard_hadamard_error']:.8f}\t"
            f"{row['standard_hadamard_bias_floor']:.8f}"
        )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_measurement_counts(
    rows: Sequence[Dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot planned measurement counts.

    This plot only needs classical planning rows. It does not require Qiskit
    simulation results.
    """
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilon_values = [row["epsilon"] for row in rows]
    entangle_shots = [row["entangle_shots"] for row in rows]
    one_register_shots = [row["one_register_shots"] for row in rows]
    standard_shots = [row["standard_hadamard_shots"] for row in rows]

    non_entangled_ratios = [
        row["one_register_shots"] / row["entangle_shots"] for row in rows
    ]
    standard_ratios = [
        row["standard_hadamard_shots"] / row["entangle_shots"] for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epsilon_values,
        standard_shots,
        marker="D",
        linestyle="-.",
        linewidth=2.2,
        color="tab:orange",
        label="Standard HT",
    )
    ax.plot(
        epsilon_values,
        one_register_shots,
        marker="^",
        linestyle=":",
        linewidth=2.2,
        color="tab:green",
        label="Iterative non-entangled HT",
    )
    ax.plot(
        epsilon_values,
        entangle_shots,
        marker="o",
        linewidth=2.2,
        color="tab:blue",
        label="Iterative entangled HT",
    )

    # Annotate max(m_t) on entangled curve.
    for row in rows:
        ax.annotate(
            f"max(m_t)={int(row['max_m_used'])}",
            xy=(row["epsilon"], row["entangle_shots"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:blue",
        )

    # Annotate ratios on the two higher-shot methods.
    for eps, shots, ratio in zip(
        epsilon_values,
        one_register_shots,
        non_entangled_ratios,
    ):
        ax.annotate(
            f"{ratio:.2f}x",
            xy=(eps, shots),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:green",
        )

    for eps, shots, ratio in zip(
        epsilon_values,
        standard_shots,
        standard_ratios,
    ):
        ax.annotate(
            f"{ratio:.2f}x",
            xy=(eps, shots),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:orange",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(epsilon_values), min(epsilon_values))
    ax.set_xlabel(r"Target accuracy")
    ax.set_ylabel("Number of measurements")
    ax.set_title("Planned measurement count")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, output_dir / "amplification_demo_shots.png", dpi=180)
    plt.close(fig)


def plot_error_comparison(
    rows: Sequence[Dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot empirical errors from Qiskit simulation.

    This plot requires simulation rows produced by simulate_accuracy_grid().
    """
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilon_values = [row["epsilon"] for row in rows]

    entangle_errors = [max(row["entangle_error"], 1e-16) for row in rows]
    one_register_errors = [max(row["one_register_error"], 1e-16) for row in rows]
    standard_errors = [
        max(row["standard_hadamard_error"], 1e-16) for row in rows
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        epsilon_values,
        standard_errors,
        marker="D",
        linestyle="-.",
        linewidth=2.2,
        color="tab:orange",
        label="Standard HT",
    )
    ax.plot(
        epsilon_values,
        one_register_errors,
        marker="^",
        linestyle=":",
        linewidth=2.2,
        color="tab:green",
        label="Iterative non-entangled HT",
    )
    ax.plot(
        epsilon_values,
        entangle_errors,
        marker="o",
        linewidth=2.2,
        color="tab:blue",
        label="Iterative entangled HT",
    )
    ax.plot(
        epsilon_values,
        epsilon_values,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        color="black",
        label=r"Target accuracy",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(epsilon_values), min(epsilon_values))
    ax.set_xlabel(r"Target accuracy")
    ax.set_ylabel("Actual absolute error")
    ax.set_title("Observed error from Qiskit simulation")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, output_dir / "amplification_demo_errors.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def main_planning_only() -> None:
    """Fast mode: only compute and plot planned measurement counts."""
    planning_rows = plan_accuracy_grid()

    print_planning_table(planning_rows)
    print_measurement_ratio_table(planning_rows)
    plot_measurement_counts(planning_rows, OUTPUT_DIR)


def main_full_simulation() -> None:
    """Full mode: compute planned counts, then run Qiskit simulations."""
    planning_rows = plan_accuracy_grid()

    print_planning_table(planning_rows)
    print_measurement_ratio_table(planning_rows)
    plot_measurement_counts(planning_rows, OUTPUT_DIR)

    simulation_rows = simulate_accuracy_grid(planning_rows)

    print_error_table(simulation_rows)
    plot_error_comparison(simulation_rows, OUTPUT_DIR)


def main() -> None:
    # Choose one of the two modes:
    #
    # 1. Fast planning-only mode:
    # main_planning_only()
    #
    # 2. Full simulation mode:
    main_full_simulation()


if __name__ == "__main__":
    main()