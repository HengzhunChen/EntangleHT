#!/usr/bin/env python3
"""Planning tables for the exact-eigenstate experiments.

Command-line options:
  -h, --help                 Show the command-line help and exit.
  --schedule {geometric,dp}  Select the primary optimizer
                             (default: geometric).
  --comparison-curves CURVE  Select one or more comparison-plot curves from
                             one-quadrature, two-quadrature, geometric, and dp
                             (default: all four).
  --output-dir PATH          Set the figure directory
                             (default: outputs/exact_eigenstate).
  --no-plots                 Print tables without generating figures.

With no options, the script uses the geometric schedule, prints the single-round
table and an iterative table for every plotted optimizer, and writes all figures
to outputs/exact_eigenstate. Any optimizer included in an iterative plot is also
included in the console output.

Examples:
  python exact_eigenstate.py --no-plots
  python exact_eigenstate.py --schedule dp --output-dir outputs/exact_dp
  python exact_eigenstate.py --comparison-curves one-quadrature dp
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from entangle_ht.baselines import (
    standard_hadamard_shots,
    standard_restarts,
    two_quadrature_standard_shots,
)
from entangle_ht.planning import (
    optimize_exact_dp,
    optimize_exact_geometric,
    select_exact_round,
)
from entangle_ht.records import EstimationConfig, ScheduleResult
from entangle_ht.resources import ResourceModel
from example_utils import (
    configure_matplotlib_cache,
    format_count,
    format_grid_value,
    save_figure,
)


# *****************************************************************************
# Experiment settings
# *****************************************************************************

PHI_TRUE = 0.35
THETA_0 = 0.20
DELTA_0 = 0.20
P_SUCCESS_TOTAL = 0.95
BRANCH_MARGIN = math.pi / 10
M_HW = 100
RESOURCES = ResourceModel(device_qubits=2500, system_qubits=1)

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
DP_DELTA_GRID_SIZE = 50
DP_MAX_ROUNDS = 8

SCHEDULE_METHOD = "geometric"
COMPARISON_CURVE_CHOICES = (
    "one-quadrature",
    "two-quadrature",
    "geometric",
    "dp",
)
BASELINE_COLORS = {
    "one-quadrature": "#6A3D9A",
    "two-quadrature": "#009E73",
}
SCHEDULE_COLORS = {
    "geometric": "#0072B2",
    "dp": "#D55E00",
}

SINGLE_ROUND_EPSILON = 1e-4
# Sample the logarithmic plot regularly by halving the bound at each point.
REFERENCE_BOUND_GRID = tuple(DELTA_0 / (2**level) for level in range(7))
EPSILON_GRID = (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4)
TRAJECTORY_EPSILON = 0.001
OUTPUT_DIR = Path("outputs/exact_eigenstate")


# *****************************************************************************
# Schedule construction
# *****************************************************************************

def exact_config(epsilon: float) -> EstimationConfig:
    return EstimationConfig(
        phase=PHI_TRUE,
        initial_reference=THETA_0,
        initial_bound=DELTA_0,
        target_accuracy=epsilon,
        total_success_probability=P_SUCCESS_TOTAL,
        branch_margin=BRANCH_MARGIN,
        hardware_amplification_cap=M_HW,
        contrast=1.0,
        contrast_lower_bound=1.0,
    )


def plan_schedule(epsilon: float, *, schedule: str) -> ScheduleResult:
    config = exact_config(epsilon)
    if schedule == "geometric":
        return optimize_exact_geometric(
            base_config=config,
            gamma_grid=GAMMA_GRID,
            resources=RESOURCES,
        )
    if schedule == "dp":
        return optimize_exact_dp(
            base_config=config,
            resources=RESOURCES,
            grid_size=DP_DELTA_GRID_SIZE,
            max_rounds=DP_MAX_ROUNDS,
        )
    raise ValueError(f"unknown schedule method {schedule!r}")


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
        baseline_restarts = standard_restarts(standard_shots, RESOURCES)
        rows.append(
            {
                "delta": delta,
                "amplification": float(m_t),
                "entangled_shots": float(entangled_shots),
                "standard_shots": float(standard_shots),
                "entangled_queries": float(m_t * entangled_shots),
                "standard_queries": float(standard_shots),
                "entangled_restarts": float(entangled_restarts),
                "standard_restarts": float(baseline_restarts),
                "shot_ratio": standard_shots / entangled_shots,
                "restart_ratio": baseline_restarts / entangled_restarts,
            }
        )
    return rows


def iterative_rows(*, schedule: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for epsilon in EPSILON_GRID:
        result = plan_schedule(epsilon, schedule=schedule)
        baseline_shots = standard_hadamard_shots(
            epsilon=epsilon,
            delta=result.config.initial_bound,
            p_fail=(
                1.0 - result.config.total_success_probability
            ),
        )
        baseline_restarts = standard_restarts(baseline_shots, RESOURCES)
        two_quadrature_shots = two_quadrature_standard_shots(
            epsilon=epsilon,
            rho0=1.0,
            p_fail=(
                1.0 - result.config.total_success_probability
            ),
        )
        two_quadrature_restarts = standard_restarts(
            two_quadrature_shots,
            RESOURCES,
        )
        if result.method == "geometric":
            schedule_label = f"gamma={format_grid_value(result.gamma or 0.0)}"
        else:
            schedule_label = f"DP[{result.grid_points}]"
        rows.append(
            {
                "epsilon": epsilon,
                "entangled_shots": float(result.plan.total_shots),
                "standard_shots": float(baseline_shots),
                "two_quadrature_shots": float(two_quadrature_shots),
                "entangled_queries": float(result.plan.total_queries),
                "standard_queries": float(baseline_shots),
                "two_quadrature_queries": float(two_quadrature_shots),
                "entangled_restarts": float(result.restarts),
                "standard_restarts": float(baseline_restarts),
                "two_quadrature_restarts": float(two_quadrature_restarts),
                "max_m": float(result.max_amplification),
                "rounds": float(result.rounds),
                "schedule": schedule_label,
                "shot_ratio": baseline_shots / result.plan.total_shots,
                "restart_ratio": baseline_restarts / result.restarts,
            }
        )
    return rows


def iterative_comparison_rows(
    *,
    selected_schedule: str,
    selected_rows: Sequence[dict[str, float | str]],
    schedules: Sequence[str],
) -> dict[str, list[dict[str, float | str]]]:
    comparison: dict[str, list[dict[str, float | str]]] = {}
    for schedule in schedules:
        comparison[schedule] = (
            list(selected_rows)
            if schedule == selected_schedule
            else iterative_rows(schedule=schedule)
        )
    return comparison


def trajectory_result(*, schedule: str) -> ScheduleResult:
    return plan_schedule(TRAJECTORY_EPSILON, schedule=schedule)


# *****************************************************************************
# Console output
# *****************************************************************************

def print_single_round_table(rows: Sequence[dict[str, float]]) -> None:
    print("Single-round exact-eigenstate comparison")
    print(
        "Delta\tm\tEnt shots\tStd shots\tEnt queries\tStd/Ent shots\t"
        "Ent restarts\tStd restarts\tStd/Ent restarts"
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
    *,
    method: str,
) -> None:
    print()
    print(f"Iterative exact-eigenstate comparison ({method})")
    print(
        "epsilon\tEnt shots\tEnt queries\t1Q shots\t2Q shots\tEnt restarts\t"
        "1Q restarts\t2Q restarts\tmax_m\trounds\tschedule\t1Q/Ent restarts"
    )
    for row in rows:
        print(
            f"{float(row['epsilon']):.6g}\t"
            f"{format_count(float(row['entangled_shots']))}\t"
            f"{format_count(float(row['entangled_queries']))}\t"
            f"{format_count(float(row['standard_shots']))}\t"
            f"{format_count(float(row['two_quadrature_shots']))}\t"
            f"{format_count(float(row['entangled_restarts']))}\t"
            f"{format_count(float(row['standard_restarts']))}\t"
            f"{format_count(float(row['two_quadrature_restarts']))}\t"
            f"{int(float(row['max_m']))}\t"
            f"{int(float(row['rounds']))}\t"
            f"{row['schedule']}\t"
            f"{float(row['restart_ratio']):.3f}"
        )


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
        marker="D",
        linewidth=2.0,
        linestyle="-.",
        label="standard HT",
    )
    ax.plot(
        deltas,
        entangled,
        marker="o",
        linewidth=2.0,
        label="entangled HT",
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
    reference_rows: Sequence[dict[str, float | str]],
    rows_by_schedule: dict[str, Sequence[dict[str, float | str]]],
    output_dir: Path,
    curves: Sequence[str],
) -> Path:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilons = [float(row["epsilon"]) for row in reference_rows]
    standard = [float(row["standard_restarts"]) for row in reference_rows]
    two_quadrature = [
        float(row["two_quadrature_restarts"]) for row in reference_rows
    ]
    labels = {"geometric": "geometric schedule", "dp": "DP schedule"}
    offsets = {"geometric": (0, 12), "dp": (0, -16)}
    vertical_alignment = {"geometric": "bottom", "dp": "top"}
    selected_curves = set(curves)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if "one-quadrature" in selected_curves:
        ax.plot(
            epsilons,
            standard,
            marker="D",
            linewidth=2.0,
            linestyle="-.",
            color=BASELINE_COLORS["one-quadrature"],
            label="one-quadrature standard HT",
        )
    if "two-quadrature" in selected_curves:
        ax.plot(
            epsilons,
            two_quadrature,
            marker="P",
            linewidth=1.9,
            linestyle=":",
            color=BASELINE_COLORS["two-quadrature"],
            label=r"two-quadrature standard HT",
        )
    for schedule in ("geometric", "dp"):
        if schedule not in rows_by_schedule:
            continue
        rows = rows_by_schedule[schedule]
        ax.plot(
            [float(row["epsilon"]) for row in rows],
            [float(row["entangled_restarts"]) for row in rows],
            marker="o" if schedule == "geometric" else "s",
            linewidth=2.0,
            color=SCHEDULE_COLORS[schedule],
            linestyle="-" if schedule == "geometric" else "--",
            label=labels[schedule],
        )
        for row in rows:
            ax.annotate(
                rf"$m\leq {int(float(row['max_m']))}$",
                xy=(float(row["epsilon"]), float(row["entangled_restarts"])),
                xytext=offsets[schedule],
                textcoords="offset points",
                ha="center",
                va=vertical_alignment[schedule],
                fontsize=8,
                color=SCHEDULE_COLORS[schedule],
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
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
        output_dir / "iterative_restarts_comparison_restarts.png",
    )
    plt.close(fig)
    return path


def plot_iterative_restart_ratio(
    rows_by_schedule: dict[str, Sequence[dict[str, float | str]]],
    output_dir: Path,
) -> Path:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    reference_rows = next(iter(rows_by_schedule.values()))
    epsilons = [float(row["epsilon"]) for row in reference_rows]
    labels = {"geometric": "geometric schedule", "dp": "DP schedule"}
    offsets = {"geometric": (0, -16), "dp": (0, 12)}
    vertical_alignment = {"geometric": "top", "dp": "bottom"}

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for schedule in ("geometric", "dp"):
        if schedule not in rows_by_schedule:
            continue
        rows = rows_by_schedule[schedule]
        ax.plot(
            [float(row["epsilon"]) for row in rows],
            [float(row["restart_ratio"]) for row in rows],
            marker="o" if schedule == "geometric" else "s",
            linewidth=2.2,
            color=SCHEDULE_COLORS[schedule],
            linestyle="-" if schedule == "geometric" else "--",
            label=labels[schedule],
        )
        for row in rows:
            ax.annotate(
                rf"$m\leq {int(float(row['max_m']))}$",
                xy=(float(row["epsilon"]), float(row["restart_ratio"])),
                xytext=offsets[schedule],
                textcoords="offset points",
                ha="center",
                va=vertical_alignment[schedule],
                fontsize=8,
                color=SCHEDULE_COLORS[schedule],
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
            )
    ax.set_xscale("log")
    ax.set_xlim(1.12 * max(epsilons), min(epsilons) / 1.12)
    ax.set_xlabel(r"target accuracy $\epsilon$")
    ax.set_ylabel("standard restarts / entangled restarts")
    ax.set_title("Iterative exact-eigenstate restart saving")
    ax.margins(y=0.20)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    path = save_figure(
        fig,
        output_dir / "iterative_restart_ratio_comparison_restarts.png",
    )
    plt.close(fig)
    return path


def plot_schedule_structure(
    result: ScheduleResult,
    output_dir: Path,
    *,
    schedule: str,
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
        output_dir / f"schedule_structure_{schedule}_restarts.png",
    )
    plt.close(fig)
    return path


def write_plots(
    *,
    single_rows: Sequence[dict[str, float]],
    reference_rows: Sequence[dict[str, float | str]],
    rows_by_schedule: dict[str, Sequence[dict[str, float | str]]],
    comparison_curves: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    try:
        configure_matplotlib_cache()
        import matplotlib.pyplot  # noqa: F401

        paths = [
            plot_single_round_restarts(single_rows, output_dir),
            plot_iterative_restarts(
                reference_rows,
                rows_by_schedule,
                output_dir,
                comparison_curves,
            ),
        ]
        if rows_by_schedule:
            paths.append(
                plot_iterative_restart_ratio(
                    rows_by_schedule,
                    output_dir,
                )
            )
        for plotted_schedule in ("geometric", "dp"):
            if plotted_schedule in rows_by_schedule:
                paths.append(
                    plot_schedule_structure(
                        trajectory_result(schedule=plotted_schedule),
                        output_dir,
                        schedule=plotted_schedule,
                    )
                )
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
    schedule: str,
    comparison_curves: Sequence[str],
    output_dir: Path,
    make_plots: bool,
) -> None:
    print("Exact-eigenstate experiment")
    print(
        f"resources: Q={RESOURCES.device_qubits}, n_sys={RESOURCES.system_qubits}, "
        f"kappa(1)={RESOURCES.packing_capacity(1)}, m_hw={RESOURCES.effective_m_hw(M_HW)}"
    )
    print(f"primary schedule={schedule}")
    single_rows = single_round_rows()
    iter_rows = iterative_rows(schedule=schedule)

    plotted_schedules = []
    if make_plots:
        plotted_schedules = [
            curve for curve in comparison_curves if curve in ("geometric", "dp")
        ]
    printed_schedules = list(dict.fromkeys([schedule, *plotted_schedules]))
    rows_by_schedule = iterative_comparison_rows(
        selected_schedule=schedule,
        selected_rows=iter_rows,
        schedules=printed_schedules,
    )

    print_single_round_table(single_rows)
    for printed_schedule in printed_schedules:
        print_iterative_table(
            rows_by_schedule[printed_schedule],
            method=printed_schedule,
        )
    if make_plots:
        plotted_rows = {
            plotted_schedule: rows_by_schedule[plotted_schedule]
            for plotted_schedule in plotted_schedules
        }
        write_plots(
            single_rows=single_rows,
            reference_rows=iter_rows,
            rows_by_schedule=plotted_rows,
            comparison_curves=comparison_curves,
            output_dir=output_dir,
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-eigenstate planning experiments for entangled HT.",
    )
    parser.add_argument(
        "--schedule",
        choices=("geometric", "dp"),
        default=SCHEDULE_METHOD,
        help="trust-radius schedule optimizer",
    )
    parser.add_argument(
        "--comparison-curves",
        nargs="+",
        choices=COMPARISON_CURVE_CHOICES,
        default=COMPARISON_CURVE_CHOICES,
        metavar="CURVE",
        help="curves included in the restart comparison plots",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for generated plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="print tables without writing plot files",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_planning(
        schedule=args.schedule,
        comparison_curves=args.comparison_curves,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
