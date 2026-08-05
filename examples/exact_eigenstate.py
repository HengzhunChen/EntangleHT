#!/usr/bin/env python3
"""Compare Standard HT, Fixed-m EHT, and Iterative EHT for an exact state.

Iterative EHT uses the geometric schedule defined by ``GAMMA_GRID``. CSV tables
are always written under ``--output-dir``; use ``--no-plots`` to skip figures.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from entangle_ht.baselines import (
    fixed_amplification_hadamard_shots,
    restart_optimal_fixed_amplification,
    standard_hadamard_shots,
)
from entangle_ht.planning import (
    optimize_exact_geometric,
    select_exact_round,
)
from entangle_ht.records import EstimationConfig, ScheduleResult
from entangle_ht.resources import ResourceModel, restarts_from_shots
from example_utils import (
    METHOD_LABELS,
    METHOD_STYLES,
    configure_matplotlib_cache,
    format_count,
    format_grid_value,
    save_csv_rows,
    save_figure,
)


# *****************************************************************************
# Experiment settings
# *****************************************************************************

THETA_TARGET = 2.0
INITIAL_REFERENCE = 1.8
INITIAL_BOUND = 0.2
P_SUCCESS_TOTAL = 0.95
BRANCH_MARGIN = math.pi / 4
M_HW = 100
RESOURCES = ResourceModel(device_qubits=2500, system_qubits=1)
FIXED_AMPLIFICATION = restart_optimal_fixed_amplification(
    initial_bound=INITIAL_BOUND,
    branch_margin=BRANCH_MARGIN,
    hardware_amplification_cap=M_HW,
    resources=RESOURCES,
)

GAMMA_GRID = (
    0.005,
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.35,
    0.40,
    0.55,
    0.70,
    0.85,
    0.95,
)
SINGLE_ROUND_EPSILON = 1e-4
# Sample the logarithmic plot regularly by halving the bound at each point.
REFERENCE_BOUND_GRID = tuple(INITIAL_BOUND / (2**level) for level in range(7))
EPSILON_GRID = (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4)
# Accuracy used for the schedule-structure plot.
TRAJECTORY_EPSILON = 0.001
OUTPUT_DIR = Path("outputs/exact_eigenstate")
EXACT_BASE_CONFIG = EstimationConfig(
    phase=THETA_TARGET,
    initial_reference=INITIAL_REFERENCE,
    initial_bound=INITIAL_BOUND,
    target_accuracy=EPSILON_GRID[0],
    total_success_probability=P_SUCCESS_TOTAL,
    branch_margin=BRANCH_MARGIN,
    hardware_amplification_cap=M_HW,
    contrast=1.0,
    contrast_lower_bound=1.0,
)


# *****************************************************************************
# Schedule construction
# *****************************************************************************

def exact_config(epsilon: float) -> EstimationConfig:
    return replace(EXACT_BASE_CONFIG, target_accuracy=epsilon)


def plan_schedule(epsilon: float) -> ScheduleResult:
    config = exact_config(epsilon)
    return optimize_exact_geometric(
        base_config=config,
        gamma_grid=GAMMA_GRID,
        resources=RESOURCES,
    )


# *****************************************************************************
# Result generation
# *****************************************************************************

def single_round_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for delta in REFERENCE_BOUND_GRID:
        round_plan = select_exact_round(
            bound_before=delta,
            bound_after=SINGLE_ROUND_EPSILON,
            config=exact_config(SINGLE_ROUND_EPSILON),
            p_fail=1.0 - P_SUCCESS_TOTAL,
            resources=RESOURCES,
        )
        m_t = round_plan.amplification
        entangled_shots = round_plan.shots
        standard_shots = standard_hadamard_shots(
            epsilon=SINGLE_ROUND_EPSILON,
            delta=delta,
            p_fail=1.0 - P_SUCCESS_TOTAL,
        )
        entangled_restarts = round_plan.restarts
        standard_restart_count = restarts_from_shots(
            standard_shots,
            RESOURCES.packing_capacity(1),
        )
        rows.append(
            {
                "delta": delta,
                "amplification": float(m_t),
                "entangled_shots": float(entangled_shots),
                "standard_shots": float(standard_shots),
                "entangled_queries": float(m_t * entangled_shots),
                "standard_queries": float(standard_shots),
                "entangled_restarts": float(entangled_restarts),
                "standard_restarts": float(standard_restart_count),
                "shot_ratio": standard_shots / entangled_shots,
                "restart_ratio": standard_restart_count / entangled_restarts,
            }
        )
    return rows


def iterative_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for epsilon in EPSILON_GRID:
        result = plan_schedule(epsilon)
        p_fail = 1.0 - result.config.total_success_probability
        standard_shots = standard_hadamard_shots(
            epsilon=epsilon,
            delta=result.config.initial_bound,
            p_fail=p_fail,
        )
        fixed_m_shots = fixed_amplification_hadamard_shots(
            epsilon=epsilon,
            delta=result.config.initial_bound,
            amplification=FIXED_AMPLIFICATION,
            p_fail=p_fail,
        )
        standard_restart_count = restarts_from_shots(
            standard_shots,
            RESOURCES.packing_capacity(1),
        )
        fixed_m_restarts = restarts_from_shots(
            fixed_m_shots,
            RESOURCES.packing_capacity(FIXED_AMPLIFICATION),
        )
        schedule_label = f"gamma={format_grid_value(result.gamma or 0.0)}"
        rows.append(
            {
                "epsilon": epsilon,
                "entangled_shots": float(result.plan.total_shots),
                "standard_shots": float(standard_shots),
                "fixed_m_shots": float(fixed_m_shots),
                "entangled_queries": float(result.plan.total_queries),
                "standard_queries": float(standard_shots),
                "fixed_m_queries": float(
                    FIXED_AMPLIFICATION * fixed_m_shots
                ),
                "entangled_restarts": float(result.restarts),
                "standard_restarts": float(standard_restart_count),
                "fixed_m_restarts": float(fixed_m_restarts),
                "max_m": float(result.max_amplification),
                "rounds": float(result.rounds),
                "schedule": schedule_label,
                "shot_ratio": standard_shots / result.plan.total_shots,
                "restart_ratio": standard_restart_count / result.restarts,
            }
        )
    return rows


# *****************************************************************************
# Console output
# *****************************************************************************

def print_single_round_table(rows: Sequence[dict[str, float]]) -> None:
    print("Single-round exact-eigenstate comparison")
    print(
        "Delta\tm\tEHT shots\tStandard shots\tEHT queries\t"
        "Standard/EHT shots\tEHT restarts\tStandard restarts\t"
        "Standard/EHT restarts"
    )
    for row in rows:
        print(
            f"{row['delta']:.6g}\t"
            f"{int(row['amplification'])}\t"
            f"{format_count(row['entangled_shots'])}\t"
            f"{format_count(row['standard_shots'])}\t"
            f"{format_count(row['entangled_queries'])}\t"
            f"{row['shot_ratio']:.3f}\t"
            f"{format_count(row['entangled_restarts'])}\t"
            f"{format_count(row['standard_restarts'])}\t"
            f"{row['restart_ratio']:.3f}"
        )


def print_iterative_table(
    rows: Sequence[dict[str, float | str]],
) -> None:
    print()
    print("Iterative exact-eigenstate comparison")
    print(
        "epsilon\tIterative EHT shots\tIterative EHT queries\tStandard shots\t"
        "Fixed-m EHT shots\tIterative EHT restarts\tStandard restarts\t"
        "Fixed-m EHT restarts\tmax_m\trounds\tschedule\t"
        "Standard/Iterative EHT restarts"
    )
    for row in rows:
        print(
            f"{float(row['epsilon']):.6g}\t"
            f"{format_count(float(row['entangled_shots']))}\t"
            f"{format_count(float(row['entangled_queries']))}\t"
            f"{format_count(float(row['standard_shots']))}\t"
            f"{format_count(float(row['fixed_m_shots']))}\t"
            f"{format_count(float(row['entangled_restarts']))}\t"
            f"{format_count(float(row['standard_restarts']))}\t"
            f"{format_count(float(row['fixed_m_restarts']))}\t"
            f"{int(float(row['max_m']))}\t"
            f"{int(float(row['rounds']))}\t"
            f"{row['schedule']}\t"
            f"{float(row['restart_ratio']):.3f}"
        )


# *****************************************************************************
# CSV output
# *****************************************************************************

def write_csv_results(
    *,
    single_rows: Sequence[dict[str, float]],
    iterative_rows: Sequence[dict[str, float | str]],
    output_dir: Path,
) -> list[Path]:
    paths = [
        save_csv_rows(
            single_rows,
            output_dir / "single_round_comparison.csv",
        ),
        save_csv_rows(
            iterative_rows,
            output_dir / "iterative_comparison.csv",
        ),
    ]
    for path in paths:
        print(f"[csv] {path}")
    return paths


# *****************************************************************************
# Plotting
# *****************************************************************************

def plot_single_round_restarts(
    rows: Sequence[dict[str, float]],
    output_dir: Path,
) -> Path:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    deltas = [row["delta"] for row in rows]
    entangled = [row["entangled_restarts"] for row in rows]
    standard = [row["standard_restarts"] for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        deltas,
        standard,
        linewidth=2.0,
        label=METHOD_LABELS["standard"],
        **METHOD_STYLES["standard"],
    )
    ax.plot(
        deltas,
        entangled,
        linewidth=2.0,
        label=METHOD_LABELS["eht"],
        **METHOD_STYLES["entangled"],
    )
    for row in rows:
        ax.annotate(
            f"m={int(row['amplification'])}",
            xy=(row["delta"], row["entangled_restarts"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(deltas), min(deltas))
    ax.set_xlabel(r"reference bound $\Delta$")
    ax.set_ylabel("device restarts")
    ax.set_title("Single-round exact-eigenstate restart count")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    path = save_figure(fig, output_dir / "single_round_restarts_vs_delta.png")
    plt.close(fig)
    return path


def plot_iterative_restarts(
    rows: Sequence[dict[str, float | str]],
    output_dir: Path,
) -> Path:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilons = [float(row["epsilon"]) for row in rows]
    standard = [float(row["standard_restarts"]) for row in rows]
    fixed_m = [float(row["fixed_m_restarts"]) for row in rows]
    entangled = [float(row["entangled_restarts"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        epsilons,
        standard,
        linewidth=2.0,
        label=METHOD_LABELS["standard"],
        **METHOD_STYLES["standard"],
    )
    ax.plot(
        epsilons,
        fixed_m,
        linewidth=2.0,
        label=rf"{METHOD_LABELS['fixed_m']} ($m={FIXED_AMPLIFICATION}$)",
        **METHOD_STYLES["fixed_m"],
    )
    ax.plot(
        epsilons,
        entangled,
        linewidth=2.0,
        label=METHOD_LABELS["entangled"],
        **METHOD_STYLES["entangled"],
    )
    for row in rows:
        epsilon = float(row["epsilon"])
        horizontal_alignment = (
            "left"
            if epsilon == max(epsilons)
            else "right" if epsilon == min(epsilons) else "center"
        )
        ax.annotate(
            rf"$m\leq {int(float(row['max_m']))}$",
            xy=(epsilon, float(row["entangled_restarts"])),
            xytext=(0, -14),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="top",
            fontsize=8,
            color=METHOD_STYLES["entangled"]["color"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.12 * max(epsilons), min(epsilons) / 1.12)
    ax.set_xlabel(r"target accuracy $\epsilon$")
    ax.set_ylabel("device restarts")
    ax.set_title("Iterative exact-eigenstate restart count")
    ax.margins(y=0.15)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    path = save_figure(
        fig,
        output_dir / "iterative_restarts.png",
    )
    plt.close(fig)
    return path


def plot_iterative_restart_ratio(
    rows: Sequence[dict[str, float | str]],
    output_dir: Path,
) -> Path:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilons = [float(row["epsilon"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        epsilons,
        [float(row["restart_ratio"]) for row in rows],
        linewidth=2.2,
        **METHOD_STYLES["entangled"],
    )
    for row in rows:
        epsilon = float(row["epsilon"])
        horizontal_alignment = (
            "left"
            if epsilon == max(epsilons)
            else "right" if epsilon == min(epsilons) else "center"
        )
        ax.annotate(
            rf"$m\leq {int(float(row['max_m']))}$",
            xy=(epsilon, float(row["restart_ratio"])),
            xytext=(0, -16),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="top",
            fontsize=8,
            color=METHOD_STYLES["entangled"]["color"],
        )
    ax.set_xscale("log")
    ax.set_xlim(1.12 * max(epsilons), min(epsilons) / 1.12)
    ax.set_xlabel(r"target accuracy $\epsilon$")
    ax.set_ylabel("Standard HT restarts / Iterative EHT restarts")
    ax.set_title("Iterative exact-eigenstate restart saving")
    ax.margins(y=0.20)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = save_figure(
        fig,
        output_dir / "iterative_restart_ratio.png",
    )
    plt.close(fig)
    return path


def plot_schedule_structure(
    result: ScheduleResult,
    output_dir: Path,
) -> Path:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    rounds = [round_plan.round_index for round_plan in result.plan.rounds]
    deltas = [round_plan.bound_before for round_plan in result.plan.rounds]
    amplifications = [round_plan.amplification for round_plan in result.plan.rounds]

    fig, ax_delta = plt.subplots(figsize=(7.2, 4.8))
    ax_m = ax_delta.twinx()
    ax_delta.plot(
        rounds,
        deltas,
        marker="o",
        linewidth=2.0,
        color="tab:blue",
        label=r"$\Delta_t$",
    )
    ax_m.step(
        rounds,
        amplifications,
        where="mid",
        marker="s",
        linewidth=2.0,
        color="tab:orange",
        label=r"$m_t$",
    )
    for round_index, amplification in zip(rounds, amplifications):
        ax_m.annotate(
            f"{amplification}",
            xy=(round_index, amplification),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )
    ax_delta.axhline(
        result.config.target_accuracy,
        color="tab:blue",
        linestyle=":",
        linewidth=1.2,
        label=r"target $\epsilon$",
    )
    ax_delta.set_yscale("log")
    ax_delta.set_xlabel(r"round $t$")
    ax_delta.set_ylabel(r"certified bound $\Delta_t$")
    ax_m.set_ylabel(r"amplification $m_t$")
    ax_delta.set_title(
        rf"Exact-eigenstate schedule structure, "
        rf"$\epsilon={result.config.target_accuracy:g}$"
    )
    ax_delta.grid(alpha=0.3, which="both")

    lines_delta, labels_delta = ax_delta.get_legend_handles_labels()
    lines_m, labels_m = ax_m.get_legend_handles_labels()
    ax_delta.legend(
        lines_delta + lines_m,
        labels_delta + labels_m,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout()
    path = save_figure(
        fig,
        output_dir / "schedule_structure.png",
    )
    plt.close(fig)
    return path


def write_plots(
    *,
    single_rows: Sequence[dict[str, float]],
    iterative_rows: Sequence[dict[str, float | str]],
    output_dir: Path,
) -> list[Path]:
    try:
        configure_matplotlib_cache()
        import matplotlib.pyplot  # noqa: F401

        paths = [
            plot_single_round_restarts(single_rows, output_dir),
            plot_iterative_restarts(
                iterative_rows,
                output_dir,
            ),
            plot_iterative_restart_ratio(iterative_rows, output_dir),
            plot_schedule_structure(
                plan_schedule(TRAJECTORY_EPSILON),
                output_dir,
            ),
        ]
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            print("[plot skip] matplotlib is not installed.")
            return []
        raise
    for path in paths:
        print(f"[plot] {path}")
    return paths


# *****************************************************************************
# Experiment entry point
# *****************************************************************************

def run_planning(
    *,
    output_dir: Path,
    make_plots: bool,
) -> None:
    print("Exact-eigenstate experiment")
    print(
        f"theta_target={THETA_TARGET:.10f}, "
        f"initial_reference={INITIAL_REFERENCE:.10f}, "
        f"initial_bound={INITIAL_BOUND:.6g}"
    )
    print(
        f"resources: Q={RESOURCES.device_qubits}, "
        f"n_sys={RESOURCES.system_qubits}, "
        f"kappa(1)={RESOURCES.packing_capacity(1)}, "
        f"m_hw={RESOURCES.effective_m_hw(M_HW)}"
    )
    print(
        "restart-optimal fixed-amplification baseline: "
        f"m={FIXED_AMPLIFICATION}, "
        f"kappa(m)={RESOURCES.packing_capacity(FIXED_AMPLIFICATION)}"
    )
    print("schedule=geometric")
    single_rows = single_round_rows()
    iter_rows = iterative_rows()

    print_single_round_table(single_rows)
    print_iterative_table(iter_rows)
    write_csv_results(
        single_rows=single_rows,
        iterative_rows=iter_rows,
        output_dir=output_dir,
    )
    if make_plots:
        write_plots(
            single_rows=single_rows,
            iterative_rows=iter_rows,
            output_dir=output_dir,
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Standard HT, Fixed-m EHT, and Iterative EHT "
            "for an exact eigenstate."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for generated CSV files and plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="print tables and write CSV files without plot files",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_planning(
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
