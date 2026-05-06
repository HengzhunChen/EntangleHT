#!/usr/bin/env python3
"""
Demonstrate the measurement reduction from adaptive amplification.

This script runs the Qiskit circuit for several target accuracies and compares
three different versions of Hadamard test.
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
    run_trial,
)
from entangle_ht.utilities import DemoConfig, phase_error


BASE_CONFIG = replace(
    DemoConfig(),
    rho=0.99,
    rho0=0.95,
    phi_true=0.35,
    theta_0=0.20,
    Delta_0=0.20,
    p_total=0.95,
    gamma=0.85,
    omega=0.6,
    m_hw=8,
    base_seed=20260416,
)
# EPSILON_GRID = (0.08, 0.06, 0.05, 0.04, 0.03, 0.02) # For quick run test
EPSILON_GRID = (0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001)
OUTPUT_PATH = Path("outputs/amplification_effect_demo.png")


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
    shots = num_shot_standard_hadamard_test(config.epsilon, config.p_total)
    alpha = config.rho * complex(math.cos(config.phi_true), math.sin(config.phi_true))
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


def run_accuracy_grid() -> List[Dict[str, float]]:
    simulator = build_simulator()
    rows: List[Dict[str, float]] = []

    for index, epsilon in enumerate(EPSILON_GRID):
        config = replace(BASE_CONFIG, epsilon=epsilon)
        algorithm = design_algorithm_parameters(config)
        entangled_trial = run_trial(
            config=config,
            algorithm=algorithm,
            simulator=simulator,
            seed=config.base_seed + 97 * index,
            label="entangle",
        )

        one_register_config = replace(config, m_hw=1)
        one_register_algorithm = design_algorithm_parameters(one_register_config)
        one_register_trial = run_trial(
            config=one_register_config,
            algorithm=one_register_algorithm,
            simulator=simulator,
            seed=config.base_seed + 10_000 + 97 * index,
            label="one_register",
        )

        standard_hadamard = run_standard_hadamard(
            config=config,
            simulator=simulator,
            seed=config.base_seed + 20_000 + 97 * index,
        )

        max_m_used = max(round_record.amplification for round_record in entangled_trial.rounds)
        rows.append(
            {
                "epsilon": epsilon,
                "entangle_shots": float(entangled_trial.total_shots),
                "one_register_shots": float(one_register_trial.total_shots),
                "standard_hadamard_shots": standard_hadamard["shots"],
                "max_m_used": float(max_m_used),
                "entangle_rounds": float(algorithm.num_rounds),
                "one_register_rounds": float(one_register_algorithm.num_rounds),
                "entangle_estimates": entangled_trial.final_estimate,
                "entangle_errors": abs(entangled_trial.final_error),
                "one_register_errors": abs(one_register_trial.final_error),
                "standard_hadamard_errors": standard_hadamard["actual_error"],
                "standard_hadamard_bias_floor": standard_hadamard["bias_floor"],
            }
        )

    return rows


def print_table(rows: Sequence[Dict[str, float]]) -> None:
    print(
        "epsilon\tIterative entangled HT shots\t"
        "Iterative non-entangled HT shots\tStandard HT shots\t"
        "max_m\tIterative entangled HT rounds\t"
        "Iterative non-entangled HT rounds\tIterative entangled HT errors\t"
        "Iterative non-entangled HT errors\tStandard HT errors\t"
        "Standard HT bias floor"
    )
    for row in rows:
        print(
            f"{row['epsilon']:.6g}\t"
            f"{int(row['entangle_shots'])}\t"
            f"{int(row['one_register_shots'])}\t"
            f"{int(row['standard_hadamard_shots'])}\t"
            f"{int(row['max_m_used'])}\t"
            f"{int(row['entangle_rounds'])}\t"
            f"{int(row['one_register_rounds'])}\t"
            f"{row['entangle_errors']:.8f}\t"
            f"{row['one_register_errors']:.8f}\t"
            f"{row['standard_hadamard_errors']:.8f}\t"
            f"{row['standard_hadamard_bias_floor']:.8f}"
        )


def print_measurement_ratio_table(rows: Sequence[Dict[str, float]]) -> None:
    print()
    print("Measurement ratios")
    print(
        "epsilon\tIterative non-entangled HT / Iterative entangled HT\t"
        "Standard HT / Iterative entangled HT\t"
    )
    for row in rows:
        non_entangled_ratio = row["one_register_shots"] / row["entangle_shots"]
        standard_ratio = row["standard_hadamard_shots"] / row["entangle_shots"]
        print(
            f"{row['epsilon']:.6g}\t"
            f"{non_entangled_ratio:.4f}\t"
            f"{standard_ratio:.4f}\t"
        )


def plot_amplification_effect(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon_values = [row["epsilon"] for row in rows]
    entangle_shots = [row["entangle_shots"] for row in rows]
    one_register_shots = [row["one_register_shots"] for row in rows]
    standard_shots = [row["standard_hadamard_shots"] for row in rows]
    entangle_errors = [max(row["entangle_errors"], 1e-16) for row in rows]
    one_register_errors = [max(row["one_register_errors"], 1e-16) for row in rows]
    standard_hadamard_errors = [max(row["standard_hadamard_errors"], 1e-16) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(
        epsilon_values,
        standard_shots,
        marker="x",
        linestyle="-.",
        linewidth=2.2,
        color="tab:orange",
        label=rf"Standard HT",
    )
    axes[0].plot(
        epsilon_values,
        one_register_shots,
        marker="^",
        linestyle=":",
        linewidth=2.2,
        color="tab:green",
        label="Iterative non-entangled HT",
    )
    axes[0].plot(
        epsilon_values,
        entangle_shots,
        marker="o",
        linewidth=2.2,
        color="tab:blue",
        label="Iterative entangled HT",
    )

    for row in rows:
        axes[0].annotate(
            f"max(m_t)={int(row['max_m_used'])}",
            xy=(row["epsilon"], row["entangle_shots"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:blue",
        )

    axes[1].plot(
        epsilon_values,
        standard_hadamard_errors,
        marker="x",
        linestyle="-.",
        linewidth=2.2,
        color="tab:orange",
        label="Standard HT",
    )
    axes[1].plot(
        epsilon_values,
        one_register_errors,
        marker="^",
        linestyle=":",
        linewidth=2.2,
        color="tab:green",
        label="Iterative non-entangled HT",
    )
    axes[1].plot(
        epsilon_values,
        entangle_errors,
        marker="o",
        linewidth=2.2,
        color="tab:blue",
        label="Iterative entangled HT",
    )
    axes[1].plot(
        epsilon_values,
        epsilon_values,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        color="black",
        label=r"Target accuracy",
    )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Number of measurements")
    axes[0].set_title("Amplification effect on measurement count")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(max(epsilon_values), min(epsilon_values))
    axes[1].set_xlabel(r"Target accuracy")
    axes[1].set_ylabel("Actual absolute error")
    axes[1].set_title("Observed error for each Qiskit run")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = run_accuracy_grid()
    print_table(rows)
    print_measurement_ratio_table(rows)
    plot_amplification_effect(rows, OUTPUT_PATH)


if __name__ == "__main__":
    main()
