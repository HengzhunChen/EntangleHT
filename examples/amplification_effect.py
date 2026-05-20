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
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from entangle_ht.circuits import (
    build_simulator, 
    build_compiled_sine_ghz_circuit, 
    run_round,
)
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
#     rho=0.999999,
#     rho0=0.9999,
#     phi_true=0.35,
#     theta_0=0.20,
#     Delta_0=0.20,
#     p_total=0.95,
#     gamma=0.6,
#     omega=0.6,
#     m_hw=100,
#     base_seed=20260416,
# )
# EPSILON_GRID = (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6)
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


# ----------------------------------------------------------------------------
# Standard Hadamard test for comparison
# ----------------------------------------------------------------------------

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

    compiled_circuit = build_compiled_sine_ghz_circuit(
        m=1,
        theta_ref=config.theta_0,
        alpha=alpha,
        simulator=simulator,
        seed=seed,
    )

    signal_empirical = run_round(
        compiled_circuit=compiled_circuit,
        shots=shots,
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
        "estimate": estimate,
        "actual_error": abs(phase_error(estimate, config.phi_true)),
        "bias_floor": abs(phase_error(limit_estimate, config.phi_true)),
    }


# ---------------------------------------------------------------------------
# Planning-only experiment
# ---------------------------------------------------------------------------

def plan_accuracy_grid() -> tuple[
    List[Dict[str, float]],
    Dict[float, Dict[str, object]],
]:    
    """
    Compute measurement costs without running Qiskit simulation.
    """
    plan_data: List[Dict[str, float]] = []
    plan_records: Dict[float, Dict[str, object]] = {}

    for epsilon in EPSILON_GRID:
        config = replace(BASE_CONFIG, epsilon=epsilon)

        entangled_algorithm = design_algorithm_parameters(config)
        entangled_plan = plan_trial(
            config=config,
            algorithm=entangled_algorithm,
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

        plan_data.append(
            {
                "epsilon": epsilon,
                "entangle_shots": float(entangled_plan.total_shots),
                "one_register_shots": float(one_register_plan.total_shots),
                "standard_hadamard_shots": float(standard_shots),
                "max_m_used": float(max_m_used),
                "entangle_rounds": float(entangled_algorithm.num_rounds),
                "one_register_rounds": float(one_register_algorithm.num_rounds),
            }
        )

        plan_records[epsilon] = {
            "config": config,
            "entangled_algorithm": entangled_algorithm,
            "entangled_plan": entangled_plan,
            "one_register_config": one_register_config,
            "one_register_algorithm": one_register_algorithm,
            "one_register_plan": one_register_plan,
        }

    return plan_data, plan_records


# ---------------------------------------------------------------------------
# Full Qiskit simulation experiment
# ---------------------------------------------------------------------------

def simulate_accuracy_grid(
    plan_data: Sequence[Dict[str, float]],
    plan_records: Dict[float, Dict[str, object]],
    simulator = None,
) -> List[Dict[str, float]]:
    """Run Qiskit simulations and append empirical error data.

    The returned rows include both planning data and simulation-derived errors.
    """
    if simulator is None:
        simulator = build_simulator()
        
    simulation_data: List[Dict[str, float]] = []

    for index, planning_row in enumerate(plan_data):
        epsilon_start = time.perf_counter()
        epsilon = planning_row["epsilon"]
        plan_record = plan_records[epsilon]

        config = plan_record["config"]
        entangled_algorithm = plan_record["entangled_algorithm"]
        entangled_plan = plan_record["entangled_plan"]

        one_register_config = plan_record["one_register_config"]
        one_register_algorithm = plan_record["one_register_algorithm"]
        one_register_plan = plan_record["one_register_plan"]

        entangled_start = time.perf_counter()
        entangled_trial = run_trial(
            config=config,
            algorithm=entangled_algorithm,
            simulator=simulator,
            seed=config.base_seed + 97 * index,
            label="entangle",
            plan=entangled_plan,
        )
        entangled_seconds = time.perf_counter() - entangled_start

        one_register_start = time.perf_counter()
        one_register_trial = run_trial(
            config=one_register_config,
            algorithm=one_register_algorithm,
            simulator=simulator,
            seed=config.base_seed + 10_000 + 97 * index,
            label="one_register",
            plan=one_register_plan,
        )
        one_register_seconds = time.perf_counter() - one_register_start

        standard_start = time.perf_counter()
        standard_hadamard = run_standard_hadamard(
            config=config,
            simulator=simulator,
            seed=config.base_seed + 20_000 + 97 * index,
        )
        standard_seconds = time.perf_counter() - standard_start
        simulation_seconds = time.perf_counter() - epsilon_start

        simulation_data.append(
            {
                **planning_row,
                "entangle_estimate": entangled_trial.final_estimate,
                "entangle_error": abs(entangled_trial.final_error),
                "one_register_estimate": one_register_trial.final_estimate,
                "one_register_error": abs(one_register_trial.final_error),
                "standard_hadamard_error": standard_hadamard["actual_error"],
                "standard_hadamard_bias_floor": standard_hadamard["bias_floor"],
                "entangle_seconds": entangled_seconds,
                "one_register_seconds": one_register_seconds,
                "standard_hadamard_seconds": standard_seconds,
                "simulation_seconds": simulation_seconds,
            }
        )

        print(
            f"[simulation] epsilon={epsilon:.6g} "
            f"entangled={entangled_seconds:.3f}s "
            f"one_register={one_register_seconds:.3f}s "
            f"standard={standard_seconds:.3f}s "
            f"total={simulation_seconds:.3f}s"
        )

    return simulation_data


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_planning_table(plan_data: Sequence[Dict[str, float]]) -> None:
    print(
        "epsilon\tEIHT shots\tIHT shots\tStdHT shots\t"
        "max_m\tEIHT rounds\tIHT rounds"
    )

    for row in plan_data:
        print(
            f"{row['epsilon']:.6g}\t"
            f"{int(row['entangle_shots'])}\t"
            f"{int(row['one_register_shots'])}\t"
            f"{int(row['standard_hadamard_shots'])}\t"
            f"{int(row['max_m_used'])}\t"
            f"{int(row['entangle_rounds'])}\t"
            f"{int(row['one_register_rounds'])}"
        )


def print_measurement_ratio_table(plan_data: Sequence[Dict[str, float]]) -> None:
    print()
    print("Measurement ratios")
    print("epsilon\tIHT / EIHT\tStdHT / EIHT")

    for row in plan_data:
        non_entangled_ratio = row["one_register_shots"] / row["entangle_shots"]
        standard_ratio = row["standard_hadamard_shots"] / row["entangle_shots"]

        print(
            f"{row['epsilon']:.6g}\t"
            f"{non_entangled_ratio:.4f}\t"
            f"{standard_ratio:.4f}"
        )


def print_error_table(simulation_data: Sequence[Dict[str, float]]) -> None:
    print()
    print("Simulation errors")
    print("epsilon\tEIHT error\tIHT error\tStdHT error\tStdHT bias floor")

    for row in simulation_data:
        print(
            f"{row['epsilon']:.6g}\t"
            f"{row['entangle_error']:.8f}\t"
            f"{row['one_register_error']:.8f}\t"
            f"{row['standard_hadamard_error']:.8f}\t"
            f"{row['standard_hadamard_bias_floor']:.8f}"
        )


def print_simulation_timing_table(
    simulation_data: Sequence[Dict[str, float]],
) -> None:
    print()
    print("Simulation timing")
    print("epsilon\tEIHT seconds\tIHT seconds\tStdHT seconds\ttotal seconds")

    for row in simulation_data:
        print(
            f"{row['epsilon']:.6g}\t"
            f"{row['entangle_seconds']:.6f}\t"
            f"{row['one_register_seconds']:.6f}\t"
            f"{row['standard_hadamard_seconds']:.6f}\t"
            f"{row['simulation_seconds']:.6f}"
        )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_measurement_counts(
    plan_data: Sequence[Dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot planned measurement counts.

    This plot only needs classical planning rows. It does not require Qiskit
    simulation results.
    """
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilon_values = [row["epsilon"] for row in plan_data]
    entangle_shots = [row["entangle_shots"] for row in plan_data]
    one_register_shots = [row["one_register_shots"] for row in plan_data]
    standard_shots = [row["standard_hadamard_shots"] for row in plan_data]

    non_entangled_ratios = [
        row["one_register_shots"] / row["entangle_shots"] for row in plan_data
    ]
    standard_ratios = [
        row["standard_hadamard_shots"] / row["entangle_shots"] for row in plan_data
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
    for row in plan_data:
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


def plot_round_shots_by_epsilon(
    plan_records: Dict[float, Dict[str, object]],
    output_dir: Path,
) -> None:
    """Plot per-round shot counts for each target accuracy.

    A heatmap is easier to read than many overlaid curves because schedules for
    nearby epsilon values often share the same early-round shot counts.
    """
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt
    import numpy as np

    epsilon_values = sorted(plan_records.keys(), reverse=True)
    max_rounds = max(
        (
            len(plan_records[epsilon]["entangled_plan"].rounds)
            for epsilon in epsilon_values
        ),
        default=0,
    )

    if max_rounds == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No shots are required for the configured epsilon grid.",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        save_figure(fig, output_dir / "amplification_demo_round_shots.png", dpi=180)
        plt.close(fig)
        return

    shot_grid = np.full((len(epsilon_values), max_rounds), np.nan)
    amplification_grid: List[List[int | None]] = [
        [None for _ in range(max_rounds)] for _ in epsilon_values
    ]
    shot_label_grid: List[List[str | None]] = [
        [None for _ in range(max_rounds)] for _ in epsilon_values
    ]

    def format_shots(shots: int) -> str:
        if shots >= 1_000_000_000_000:
            return f"{shots / 1_000_000_000_000:.1f}T"
        if shots >= 1_000_000_000:
            return f"{shots / 1_000_000_000:.1f}B"
        if shots >= 1_000_000:
            return f"{shots / 1_000_000:.1f}M"
        if shots >= 1_000:
            return f"{shots / 1_000:.1f}k"
        return str(shots)

    for row_index, epsilon in enumerate(epsilon_values):
        rounds = plan_records[epsilon]["entangled_plan"].rounds
        for round_plan in rounds:
            col_index = round_plan.round_index
            shot_grid[row_index, col_index] = math.log10(round_plan.shots)
            amplification_grid[row_index][col_index] = round_plan.amplification
            shot_label_grid[row_index][col_index] = format_shots(round_plan.shots)

    masked_shot_grid = np.ma.masked_invalid(shot_grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")

    fig_width = max(8, 0.45 * max_rounds + 3.5)
    fig_height = max(4, 0.45 * len(epsilon_values) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(
        masked_shot_grid,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    def text_color_for_cell(value: float) -> str:
        r, g, b, _ = cmap(image.norm(value))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.5 else "white"

    for row_index, epsilon in enumerate(epsilon_values):
        for col_index in range(max_rounds):
            amplification = amplification_grid[row_index][col_index]
            shot_label = shot_label_grid[row_index][col_index]
            if amplification is None:
                continue
            ax.text(
                col_index,
                row_index,
                f"{shot_label}\nm={amplification}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color_for_cell(shot_grid[row_index, col_index]),
            )

    ax.set_xticks(range(max_rounds))
    ax.set_xticklabels(range(max_rounds))
    ax.set_yticks(range(len(epsilon_values)))
    ax.set_yticklabels([f"{epsilon:.1e}" for epsilon in epsilon_values])
    ax.set_xlabel("Planned round")
    ax.set_ylabel(r"Target accuracy")
    ax.set_title("Per-round planned shots for entangled HT")
    ax.grid(False)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}(\mathrm{shots})$")

    fig.tight_layout()
    save_figure(fig, output_dir / "amplification_demo_round_shots.png", dpi=180)
    plt.close(fig)


def plot_error_comparison(
    simulation_data: Sequence[Dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot empirical errors from Qiskit simulation.

    This plot requires simulation rows produced by simulate_accuracy_grid().
    """
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilon_values = [row["epsilon"] for row in simulation_data]

    entangle_errors = [max(row["entangle_error"], 1e-16) for row in simulation_data]
    one_register_errors = [max(row["one_register_error"], 1e-16) for row in simulation_data]
    standard_errors = [
        max(row["standard_hadamard_error"], 1e-16) for row in simulation_data
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
    plan_data, plan_records = plan_accuracy_grid()

    print_planning_table(plan_data)
    print_measurement_ratio_table(plan_data)
    plot_measurement_counts(plan_data, OUTPUT_DIR)
    plot_round_shots_by_epsilon(plan_records, OUTPUT_DIR)


def main_full_simulation() -> None:
    """Full mode: compute planned counts, then run Qiskit simulations."""
    plan_data, plan_records = plan_accuracy_grid()

    print_planning_table(plan_data)
    print_measurement_ratio_table(plan_data)
    plot_measurement_counts(plan_data, OUTPUT_DIR)
    plot_round_shots_by_epsilon(plan_records, OUTPUT_DIR)

    simulation_data = simulate_accuracy_grid(plan_data, plan_records)

    print_error_table(simulation_data)
    print_simulation_timing_table(simulation_data)
    plot_error_comparison(simulation_data, OUTPUT_DIR)


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
