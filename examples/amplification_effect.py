#!/usr/bin/env python3
"""
Demonstrate the measurement reduction from adaptive amplification.

This script runs the Qiskit circuit for several target accuracies and compares
the total adaptive measurements against an ideal standard Hadamard-test scaling.
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

from entangle_ht.circuits import build_simulator
from entangle_ht.schedule import (
    design_algorithm_parameters,
    run_trial,
)
from entangle_ht.utilities import DemoConfig


BASE_CONFIG = replace(
    DemoConfig(),
    rho=0.995,
    rho0=0.95,
    phi_true=0.35,
    theta_0=0.20,
    Delta_0=0.20,
    p_total=0.95,
    m_hw=8,
    base_seed=20260416,
)
# EPSILON_GRID = (0.08, 0.06, 0.05, 0.04, 0.03, 0.02)
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


def ideal_standard_hadamard_measurements(epsilon: float, p_success: float) -> int:
    # Ideal m=1, unit-contrast, unit-slope scaling with the same Chebyshev
    # success-probability convention used in the updated note.
    return int(math.ceil(1.0 / ((1.0 - p_success) * epsilon**2)))


def run_accuracy_grid() -> List[Dict[str, float]]:
    simulator = build_simulator()
    rows: List[Dict[str, float]] = []

    for index, epsilon in enumerate(EPSILON_GRID):
        config = replace(BASE_CONFIG, epsilon=epsilon)
        algorithm = design_algorithm_parameters(config)
        adaptive_trial = run_trial(
            config=config,
            algorithm=algorithm,
            simulator=simulator,
            seed=config.base_seed + 97 * index,
            label="adaptive",
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
        max_m_used = max(round_record.amplification for round_record in adaptive_trial.rounds)
        rows.append(
            {
                "epsilon": epsilon,
                "adaptive_measurements": float(adaptive_trial.total_shots),
                "one_register_measurements": float(one_register_trial.total_shots),
                "ideal_hadamard_measurements": float(
                    ideal_standard_hadamard_measurements(epsilon, config.p_total)
                ),
                "max_m_used": float(max_m_used),
                "rounds": float(algorithm.num_rounds),
                "one_register_rounds": float(one_register_algorithm.num_rounds),
                "final_estimate": adaptive_trial.final_estimate,
                "final_error": adaptive_trial.final_error,
                "actual_error": abs(adaptive_trial.final_error),
                "one_register_final_error": one_register_trial.final_error,
                "one_register_actual_error": abs(one_register_trial.final_error),
            }
        )

    return rows


def print_table(rows: Sequence[Dict[str, float]]) -> None:
    print(
        "epsilon\tadaptive_measurements\tone_register_measurements\t"
        "ideal_hadamard_measurements\tone_register/adaptive\tideal/adaptive\t"
        "max_m\trounds\tone_register_rounds\tactual_error\tone_register_actual_error"
    )
    for row in rows:
        practical_ratio = row["one_register_measurements"] / row["adaptive_measurements"]
        ideal_ratio = row["ideal_hadamard_measurements"] / row["adaptive_measurements"]
        print(
            f"{row['epsilon']:.6g}\t"
            f"{int(row['adaptive_measurements'])}\t"
            f"{int(row['one_register_measurements'])}\t"
            f"{int(row['ideal_hadamard_measurements'])}\t"
            f"{practical_ratio:.4f}\t"
            f"{ideal_ratio:.4f}\t"
            f"{int(row['max_m_used'])}\t"
            f"{int(row['rounds'])}\t"
            f"{int(row['one_register_rounds'])}\t"
            f"{row['actual_error']:.8f}\t"
            f"{row['one_register_actual_error']:.8f}"
        )


def plot_amplification_effect(rows: Sequence[Dict[str, float]], output_path: Path) -> None:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon_values = [row["epsilon"] for row in rows]
    adaptive_measurements = [row["adaptive_measurements"] for row in rows]
    one_register_measurements = [row["one_register_measurements"] for row in rows]
    ideal_measurements = [row["ideal_hadamard_measurements"] for row in rows]
    ideal_ratios = [
        row["ideal_hadamard_measurements"] / row["adaptive_measurements"]
        for row in rows
    ]
    practical_ratios = [
        row["one_register_measurements"] / row["adaptive_measurements"]
        for row in rows
    ]
    actual_errors = [max(row["actual_error"], 1e-16) for row in rows]
    one_register_actual_errors = [
        max(row["one_register_actual_error"], 1e-16)
        for row in rows
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    axes[0].plot(
        epsilon_values,
        adaptive_measurements,
        marker="o",
        linewidth=2.2,
        label="Adaptive amplified circuit",
    )
    axes[0].plot(
        epsilon_values,
        ideal_measurements,
        marker="s",
        linestyle="--",
        linewidth=2.2,
        label=rf"Ideal standard Hadamard test, $p={BASE_CONFIG.p_total:.2f}$",
    )
    axes[0].plot(
        epsilon_values,
        one_register_measurements,
        marker="^",
        linestyle=":",
        linewidth=2.2,
        label="Practical one-register circuit (m_hw = 1)",
    )

    for row in rows:
        axes[0].annotate(
            f"max(m_t)={int(row['max_m_used'])}",
            xy=(row["epsilon"], row["adaptive_measurements"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:blue",
        )

    axes[1].plot(
        epsilon_values,
        practical_ratios,
        marker="d",
        linewidth=2.2,
        color="tab:green",
        label="One-register/adaptive",
    )
    axes[1].plot(
        epsilon_values,
        ideal_ratios,
        marker="s",
        linestyle="--",
        linewidth=2.2,
        color="tab:olive",
        label="Ideal Hadamard/adaptive",
    )
    for epsilon, practical_ratio, ideal_ratio in zip(
        epsilon_values,
        practical_ratios,
        ideal_ratios,
    ):
        axes[1].annotate(
            f"{practical_ratio:.1f}x",
            xy=(epsilon, practical_ratio),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:green",
        )
        axes[1].annotate(
            f"{ideal_ratio:.1f}x",
            xy=(epsilon, ideal_ratio),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="tab:olive",
        )
    axes[2].plot(
        epsilon_values,
        actual_errors,
        marker="o",
        linewidth=2.2,
        color="tab:red",
        label="Adaptive actual error",
    )
    axes[2].plot(
        epsilon_values,
        one_register_actual_errors,
        marker="^",
        linestyle=":",
        linewidth=2.2,
        color="tab:orange",
        label="One-register actual error",
    )
    axes[2].plot(
        epsilon_values,
        epsilon_values,
        marker="s",
        linestyle="--",
        linewidth=2.0,
        color="black",
        label=r"Target RMSE $\epsilon$",
    )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Number of measurements")
    axes[0].set_title("Amplification effect on measurement count")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_xlim(max(epsilon_values), min(epsilon_values))
    axes[1].set_ylabel("Measurement ratio")
    axes[1].set_title("Amplification advantage curve")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend()

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlim(max(epsilon_values), min(epsilon_values))
    axes[2].set_xlabel(r"Target RMSE $\epsilon$")
    axes[2].set_ylabel("Actual absolute error")
    axes[2].set_title("Observed error for each Qiskit run")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend()

    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = run_accuracy_grid()
    print_table(rows)
    plot_amplification_effect(rows, OUTPUT_PATH)


if __name__ == "__main__":
    main()
