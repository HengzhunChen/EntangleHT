#!/usr/bin/env python3
"""Planning tables for the imperfect-eigenstate experiments.

The exact and imperfect models use the same phase gate and target eigenstate;
only the state preparation differs.

Command-line options:
  -h, --help                 Show the command-line help and exit.
  --schedule {geometric,dp}  Select the primary optimizer
                             (default: geometric).
  --comparison-curves CURVE  Select one or more comparison-plot curves from
                             two-quadrature, geometric, and dp
                             (default: all three).
  --output-dir PATH          Set the figure directory
                             (default: outputs/imperfect_eigenstate).
  --no-plots                 Print tables without generating figures.

With no options, the script prints an iterative table for every plotted
optimizer and writes all figures to outputs/imperfect_eigenstate. Any optimizer
included in a plot is also included in the console output.

Examples:
  python imperfect_eigenstate.py --no-plots
  python imperfect_eigenstate.py --schedule dp --output-dir outputs/imperfect_dp
  python imperfect_eigenstate.py --comparison-curves two-quadrature dp
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

from entangle_ht.baselines import standard_restarts, two_quadrature_standard_shots
from entangle_ht.circuit_models.imperfect import (
    imperfect_contrast_lower_bound,
    imperfect_overlap,
)
from entangle_ht.planning import (
    optimize_imperfect_dp,
    optimize_imperfect_geometric,
)
from entangle_ht.records import EstimationConfig, ScheduleResult
from entangle_ht.resources import ResourceModel
from entangle_ht.utilities import phase_error
from example_utils import (
    configure_matplotlib_cache,
    format_count,
    format_grid_value,
    save_figure,
)


# *****************************************************************************
# Experiment settings
# *****************************************************************************

THETA_TARGET = 0.35
ETA = 0.01
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
OMEGA_GRID = (
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
DP_DELTA_GRID_SIZE = 50
DP_MAX_ROUNDS = 8

SCHEDULE_METHOD = "geometric"
COMPARISON_CURVE_CHOICES = (
    "two-quadrature",
    "geometric",
    "dp",
)
BASELINE_COLORS = {"two-quadrature": "#009E73"}
SCHEDULE_COLORS = {
    "geometric": "#0072B2",
    "dp": "#D55E00",
}

EPSILON_GRID = (
    5e-2,
    2e-2,
    1e-2,
    5e-3,
    2e-3,
    1e-3,
)
OUTPUT_DIR = Path("outputs/imperfect_eigenstate")


# *****************************************************************************
# Schedule construction
# *****************************************************************************

def model_parameters() -> tuple[float, float, float, float]:
    contrast, effective_phase, _ = imperfect_overlap(THETA_TARGET, ETA)
    contrast_lower_bound = imperfect_contrast_lower_bound(ETA)
    preparation_floor = abs(phase_error(effective_phase, THETA_TARGET))
    return contrast, effective_phase, contrast_lower_bound, preparation_floor


def imperfect_config(epsilon: float) -> EstimationConfig:
    contrast, effective_phase, contrast_lower_bound, _ = model_parameters()
    return EstimationConfig(
        phase=effective_phase,
        initial_reference=THETA_0,
        initial_bound=DELTA_0,
        target_accuracy=epsilon,
        total_success_probability=P_SUCCESS_TOTAL,
        branch_margin=BRANCH_MARGIN,
        hardware_amplification_cap=M_HW,
        contrast=contrast,
        contrast_lower_bound=contrast_lower_bound,
    )


def plan_schedule(epsilon: float, *, schedule: str) -> ScheduleResult:
    config = imperfect_config(epsilon)
    if schedule == "geometric":
        return optimize_imperfect_geometric(
            base_config=config,
            gamma_grid=GAMMA_GRID,
            omega_grid=OMEGA_GRID,
            resources=RESOURCES,
        )
    if schedule == "dp":
        return optimize_imperfect_dp(
            base_config=config,
            resources=RESOURCES,
            omega_grid=OMEGA_GRID,
            grid_size=DP_DELTA_GRID_SIZE,
            max_rounds=DP_MAX_ROUNDS,
        )
    raise ValueError(f"unknown schedule method {schedule!r}")


# *****************************************************************************
# Result generation
# *****************************************************************************

def accuracy_rows(*, schedule: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    _, _, _, prep_floor = model_parameters()

    for epsilon in EPSILON_GRID:
        result = plan_schedule(epsilon, schedule=schedule)
        baseline_shots = two_quadrature_standard_shots(
            epsilon=epsilon,
            rho0=result.config.contrast_lower_bound,
            p_fail=(
                1.0 - result.config.total_success_probability
            ),
        )
        baseline_restarts = standard_restarts(baseline_shots, RESOURCES)
        if result.method == "geometric":
            schedule_label = (
                f"gamma={format_grid_value(result.gamma or 0.0)}, "
                f"omega={format_grid_value(result.omega or 0.0)}"
            )
        else:
            schedule_label = f"DP[{result.grid_points}]"
        rows.append(
            {
                "epsilon": epsilon,
                "entangled_shots": float(result.plan.total_shots),
                "standard_shots": float(baseline_shots),
                "entangled_queries": float(result.plan.total_queries),
                "standard_queries": float(baseline_shots),
                "entangled_restarts": float(result.restarts),
                "standard_restarts": float(baseline_restarts),
                "max_m": float(result.max_amplification),
                "rounds": float(result.rounds),
                "schedule": schedule_label,
                "shot_ratio": baseline_shots / result.plan.total_shots,
                "restart_ratio": baseline_restarts / result.restarts,
                "target_floor": prep_floor,
            }
        )
    return rows


def comparison_rows(
    *,
    selected_schedule: str,
    selected_rows: Sequence[dict[str, float | str]],
    schedules: Sequence[str],
) -> dict[str, list[dict[str, float | str]]]:
    rows_by_schedule: dict[str, list[dict[str, float | str]]] = {}
    for schedule in schedules:
        rows_by_schedule[schedule] = (
            list(selected_rows)
            if schedule == selected_schedule
            else accuracy_rows(schedule=schedule)
        )
    return rows_by_schedule


# *****************************************************************************
# Console output
# *****************************************************************************

def print_setup() -> None:
    rho, theta_psi, rho0, prep_floor = model_parameters()
    print("Imperfect-eigenstate experiment")
    print(
        f"theta_target={THETA_TARGET:.10f}, theta_psi={theta_psi:.10f}, "
        f"rho={rho:.10f}, rho0={rho0:.10f}, eta={ETA:.6g}"
    )
    print(f"target-phase preparation floor={prep_floor:.6g}")
    print(
        f"resources: Q={RESOURCES.device_qubits}, n_sys={RESOURCES.system_qubits}, "
        f"kappa(1)={RESOURCES.packing_capacity(1)}, m_hw={RESOURCES.effective_m_hw(M_HW)}"
    )


def print_accuracy_table(
    rows: Sequence[dict[str, float | str]],
    *,
    method: str,
) -> None:
    print()
    print(f"Iterative entangled HT vs two-quadrature standard HT ({method})")
    print(
        "epsilon\tEnt shots\tEnt queries\t2Q shots\tEnt restarts\t"
        "2Q restarts\tmax_m\trounds\tschedule\t2Q/Ent restarts"
    )
    for row in rows:
        print(
            f"{float(row['epsilon']):.6g}\t"
            f"{format_count(float(row['entangled_shots']))}\t"
            f"{format_count(float(row['entangled_queries']))}\t"
            f"{format_count(float(row['standard_shots']))}\t"
            f"{format_count(float(row['entangled_restarts']))}\t"
            f"{format_count(float(row['standard_restarts']))}\t"
            f"{int(float(row['max_m']))}\t"
            f"{int(float(row['rounds']))}\t"
            f"{row['schedule']}\t"
            f"{float(row['restart_ratio']):.3f}"
        )


# *****************************************************************************
# Plotting
# *****************************************************************************

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
    labels = {"geometric": "geometric schedule", "dp": "DP schedule"}
    offsets = {"geometric": (0, 12), "dp": (0, -16)}
    vertical_alignment = {"geometric": "bottom", "dp": "top"}
    selected_curves = set(curves)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if "two-quadrature" in selected_curves:
        ax.plot(
            epsilons,
            standard,
            marker="D",
            linewidth=2.0,
            linestyle="-.",
            color=BASELINE_COLORS["two-quadrature"],
            label="two-quadrature standard HT",
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
    ax.set_xlabel(r"target effective-phase accuracy $\epsilon$")
    ax.set_ylabel("device restarts")
    ax.set_title("Imperfect-eigenstate restart count")
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
    ax.set_xlabel(r"target effective-phase accuracy $\epsilon$")
    ax.set_ylabel("2Q standard restarts / entangled restarts")
    ax.set_title("Imperfect-eigenstate restart saving")
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


def write_plots(
    *,
    reference_rows: Sequence[dict[str, float | str]],
    rows_by_schedule: dict[str, Sequence[dict[str, float | str]]],
    comparison_curves: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    try:
        configure_matplotlib_cache()
        import matplotlib.pyplot  # noqa: F401

        paths = [
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
    print_setup()
    print(f"primary schedule={schedule}")
    rows = accuracy_rows(schedule=schedule)

    plotted_schedules = []
    if make_plots:
        plotted_schedules = [
            curve for curve in comparison_curves if curve in ("geometric", "dp")
        ]
    printed_schedules = list(dict.fromkeys([schedule, *plotted_schedules]))
    rows_by_schedule = comparison_rows(
        selected_schedule=schedule,
        selected_rows=rows,
        schedules=printed_schedules,
    )

    for printed_schedule in printed_schedules:
        print_accuracy_table(
            rows_by_schedule[printed_schedule],
            method=printed_schedule,
        )
    if make_plots:
        plotted_rows = {
            plotted_schedule: rows_by_schedule[plotted_schedule]
            for plotted_schedule in plotted_schedules
        }
        write_plots(
            reference_rows=rows,
            rows_by_schedule=plotted_rows,
            comparison_curves=comparison_curves,
            output_dir=output_dir,
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Imperfect-eigenstate planning experiments for entangled HT.",
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
