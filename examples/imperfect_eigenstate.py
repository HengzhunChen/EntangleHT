#!/usr/bin/env python3
"""Compare Standard HT, Fixed-m EHT, and Adaptive EHT for an imperfect state.

The exact and imperfect models use the same phase gate and target eigenstate;
only the state preparation differs. Adaptive EHT uses the geometric schedule
defined by ``GAMMA_GRID`` and ``OMEGA_GRID``. CSV tables are always written
under ``--output-dir``; use ``--no-plots`` to skip figures.
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
    imperfect_fixed_amplification_hadamard_shots,
    standard_hadamard_shots,
)
from entangle_ht.certification import contrast_bias_bound
from entangle_ht.circuit_models.imperfect import (
    imperfect_model_parameters,
)
from entangle_ht.planning import optimize_imperfect_geometric
from entangle_ht.records import EstimationConfig, ScheduleResult
from entangle_ht.resources import ResourceModel, restarts_from_shots
from example_utils import (
    ANNOTATION_FONT_SIZE,
    METHOD_LABELS,
    METHOD_STYLES,
    REFERENCE_COLOR,
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
ETA = 0.01
INITIAL_REFERENCE = 1.8
INITIAL_BOUND = 0.2
P_SUCCESS_TOTAL = 0.95
BRANCH_MARGIN = math.pi / 4
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

EPSILON_GRID = (
    3e-2,
    2e-2,
    1.5e-2,
    1.1e-2,
    9e-3,
    7e-3,
    5.5e-3,
    4.5e-3,
    3.5e-3,
)
FIXED_AMPLIFICATION = 2
MODEL_PARAMETERS = imperfect_model_parameters(THETA_TARGET, ETA)
# The Standard HT contrast-bias bound is B(1; Delta_0, rho_0). Its statistical
# budget is positive only when the target accuracy is greater than this bound.
STANDARD_HT_BIAS_BOUND = contrast_bias_bound(
    1,
    INITIAL_BOUND,
    MODEL_PARAMETERS.contrast_lower_bound,
)
FIXED_M_BIAS_BOUND = contrast_bias_bound(
    FIXED_AMPLIFICATION,
    INITIAL_BOUND,
    MODEL_PARAMETERS.contrast_lower_bound,
)
OUTPUT_DIR = Path("outputs/imperfect_eigenstate")
IMPERFECT_BASE_CONFIG = EstimationConfig(
    phase=MODEL_PARAMETERS.effective_phase,
    initial_reference=INITIAL_REFERENCE,
    initial_bound=INITIAL_BOUND,
    target_accuracy=EPSILON_GRID[0],
    total_success_probability=P_SUCCESS_TOTAL,
    branch_margin=BRANCH_MARGIN,
    hardware_amplification_cap=M_HW,
    contrast=MODEL_PARAMETERS.contrast,
    contrast_lower_bound=MODEL_PARAMETERS.contrast_lower_bound,
)


# *****************************************************************************
# Schedule construction
# *****************************************************************************

def imperfect_config(epsilon: float) -> EstimationConfig:
    return replace(IMPERFECT_BASE_CONFIG, target_accuracy=epsilon)


def plan_schedule(epsilon: float) -> ScheduleResult:
    config = imperfect_config(epsilon)
    return optimize_imperfect_geometric(
        base_config=config,
        gamma_grid=GAMMA_GRID,
        omega_grid=OMEGA_GRID,
        resources=RESOURCES,
    )


# *****************************************************************************
# Result generation
# *****************************************************************************

def accuracy_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    for epsilon in EPSILON_GRID:
        result = plan_schedule(epsilon)
        p_fail = 1.0 - result.config.total_success_probability
        standard_bias_bound = contrast_bias_bound(
            1,
            result.config.initial_bound,
            result.config.contrast_lower_bound,
        )
        standard_statistical_accuracy = epsilon - standard_bias_bound
        if standard_statistical_accuracy > 0.0:
            standard_shots = standard_hadamard_shots(
                epsilon=standard_statistical_accuracy,
                delta=result.config.initial_bound,
                p_fail=p_fail,
            )
            standard_restart_count = restarts_from_shots(
                standard_shots,
                RESOURCES.packing_capacity(1),
            )
        else:
            standard_statistical_accuracy = math.nan
            standard_shots = math.nan
            standard_restart_count = math.nan
        fixed_m_eht_bias_bound = contrast_bias_bound(
            FIXED_AMPLIFICATION,
            result.config.initial_bound,
            result.config.contrast_lower_bound,
        )
        fixed_m_eht_statistical_accuracy = epsilon - fixed_m_eht_bias_bound
        if fixed_m_eht_statistical_accuracy > 0.0:
            fixed_m_eht_shots = imperfect_fixed_amplification_hadamard_shots(
                epsilon=epsilon,
                delta=result.config.initial_bound,
                amplification=FIXED_AMPLIFICATION,
                contrast_lower_bound=result.config.contrast_lower_bound,
                p_fail=p_fail,
            )
            fixed_m_eht_restarts = restarts_from_shots(
                fixed_m_eht_shots,
                RESOURCES.packing_capacity(FIXED_AMPLIFICATION),
            )
        else:
            fixed_m_eht_shots = math.nan
            fixed_m_eht_restarts = math.nan
        schedule_label = (
            f"gamma={format_grid_value(result.gamma or 0.0)}, "
            f"omega={format_grid_value(result.omega or 0.0)}"
        )
        rows.append(
            {
                "epsilon": epsilon,
                "adaptive_eht_shots": float(result.plan.total_shots),
                "standard_shots": float(standard_shots),
                "adaptive_eht_queries": float(result.plan.total_queries),
                "standard_queries": float(standard_shots),
                "adaptive_eht_restarts": float(result.restarts),
                "standard_restarts": float(standard_restart_count),
                "fixed_m_eht_shots": float(fixed_m_eht_shots),
                "fixed_m_eht_restarts": float(fixed_m_eht_restarts),
                "fixed_m_eht_statistical_accuracy": (
                    fixed_m_eht_statistical_accuracy
                    if fixed_m_eht_statistical_accuracy > 0.0
                    else math.nan
                ),
                "standard_bias_bound": standard_bias_bound,
                "fixed_m_eht_bias_bound": fixed_m_eht_bias_bound,
                "standard_statistical_accuracy": standard_statistical_accuracy,
                "max_m": float(result.max_amplification),
                "rounds": float(result.rounds),
                "schedule": schedule_label,
                "shot_ratio": (
                    standard_shots / result.plan.total_shots
                    if math.isfinite(standard_shots)
                    else math.nan
                ),
                "restart_ratio": (
                    standard_restart_count / result.restarts
                    if math.isfinite(standard_restart_count)
                    else math.nan
                ),
                "target_floor": MODEL_PARAMETERS.preparation_floor,
            }
        )
    return rows


# *****************************************************************************
# Console output
# *****************************************************************************

def print_setup() -> None:
    print("Imperfect-eigenstate experiment")
    print(
        f"theta_target={THETA_TARGET:.10f}, "
        f"theta_psi={MODEL_PARAMETERS.effective_phase:.10f}, "
        f"rho={MODEL_PARAMETERS.contrast:.10f}, "
        f"rho0={MODEL_PARAMETERS.contrast_lower_bound:.10f}, "
        f"eta={ETA:.6g}"
    )
    print(
        f"initial_reference={INITIAL_REFERENCE:.10f}, "
        f"initial_bound={INITIAL_BOUND:.6g}"
    )
    print(
        "target-phase preparation floor="
        f"{MODEL_PARAMETERS.preparation_floor:.6g}"
    )
    print(
        "SHT contrast-bias bound="
        f"{STANDARD_HT_BIAS_BOUND:.10g}; "
        f"epsilon grid={EPSILON_GRID}"
    )
    print(
        f"EHT (m={FIXED_AMPLIFICATION}) contrast-bias bound="
        f"{FIXED_M_BIAS_BOUND:.10g}; "
        f"kappa(m)={RESOURCES.packing_capacity(FIXED_AMPLIFICATION)}"
    )
    print(
        f"resources: Q={RESOURCES.device_qubits}, "
        f"n_sys={RESOURCES.system_qubits}, "
        f"kappa(1)={RESOURCES.packing_capacity(1)}, "
        f"m_hw={RESOURCES.effective_m_hw(M_HW)}"
    )


def print_accuracy_table(
    rows: Sequence[dict[str, float | str]],
) -> None:
    print()
    print("Adaptive imperfect-eigenstate comparison")
    print(
        "epsilon\tSHT stat eps\tEHT stat eps\t"
        "AEHT shots\tAEHT queries\tSHT shots\t"
        "EHT shots\tAEHT restarts\tSHT restarts\t"
        "EHT restarts\tmax_m\trounds\tschedule\t"
        "SHT/AEHT"
    )
    for row in rows:
        standard_statistical_accuracy = float(
            row["standard_statistical_accuracy"]
        )
        standard_shots = float(row["standard_shots"])
        standard_restarts = float(row["standard_restarts"])
        restart_ratio = float(row["restart_ratio"])
        fixed_m_eht_statistical_accuracy = float(
            row["fixed_m_eht_statistical_accuracy"]
        )
        fixed_m_eht_shots = float(row["fixed_m_eht_shots"])
        fixed_m_eht_restarts = float(row["fixed_m_eht_restarts"])
        standard_statistical_text = (
            f"{standard_statistical_accuracy:.6g}"
            if math.isfinite(standard_statistical_accuracy)
            else "--"
        )
        standard_shots_text = (
            format_count(standard_shots)
            if math.isfinite(standard_shots)
            else "--"
        )
        standard_restarts_text = (
            format_count(standard_restarts)
            if math.isfinite(standard_restarts)
            else "--"
        )
        restart_ratio_text = (
            f"{restart_ratio:.3f}"
            if math.isfinite(restart_ratio)
            else "--"
        )
        fixed_m_eht_statistical_text = (
            f"{fixed_m_eht_statistical_accuracy:.6g}"
            if math.isfinite(fixed_m_eht_statistical_accuracy)
            else "--"
        )
        fixed_m_eht_shots_text = (
            format_count(fixed_m_eht_shots)
            if math.isfinite(fixed_m_eht_shots)
            else "--"
        )
        fixed_m_eht_restarts_text = (
            format_count(fixed_m_eht_restarts)
            if math.isfinite(fixed_m_eht_restarts)
            else "--"
        )
        print(
            f"{float(row['epsilon']):.6g}\t"
            f"{standard_statistical_text}\t"
            f"{fixed_m_eht_statistical_text}\t"
            f"{format_count(float(row['adaptive_eht_shots']))}\t"
            f"{format_count(float(row['adaptive_eht_queries']))}\t"
            f"{standard_shots_text}\t"
            f"{fixed_m_eht_shots_text}\t"
            f"{format_count(float(row['adaptive_eht_restarts']))}\t"
            f"{standard_restarts_text}\t"
            f"{fixed_m_eht_restarts_text}\t"
            f"{int(float(row['max_m']))}\t"
            f"{int(float(row['rounds']))}\t"
            f"{row['schedule']}\t"
            f"{restart_ratio_text}"
        )


# *****************************************************************************
# CSV output
# *****************************************************************************

def write_csv_results(
    rows: Sequence[dict[str, float | str]],
    output_dir: Path,
) -> Path:
    path = save_csv_rows(
        rows,
        output_dir / "iterative_comparison.csv",
    )
    print(f"[csv] {path}")
    return path


# *****************************************************************************
# Plotting
# *****************************************************************************

def plot_iterative_restarts(
    rows: Sequence[dict[str, float | str]],
    output_dir: Path,
) -> Path:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilons = [float(row["epsilon"]) for row in rows]
    standard_rows = [
        row
        for row in rows
        if math.isfinite(float(row["standard_restarts"]))
    ]
    fixed_m_eht_rows = [
        row
        for row in rows
        if math.isfinite(float(row["fixed_m_eht_restarts"]))
    ]
    adaptive_eht = [float(row["adaptive_eht_restarts"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        [float(row["epsilon"]) for row in standard_rows],
        [float(row["standard_restarts"]) for row in standard_rows],
        linewidth=2.0,
        label=METHOD_LABELS["standard"],
        **METHOD_STYLES["standard"],
    )
    ax.plot(
        [float(row["epsilon"]) for row in fixed_m_eht_rows],
        [float(row["fixed_m_eht_restarts"]) for row in fixed_m_eht_rows],
        linewidth=2.0,
        label=rf"{METHOD_LABELS['fixed_m_eht']} ($m={FIXED_AMPLIFICATION}$)",
        **METHOD_STYLES["fixed_m_eht"],
    )
    ax.plot(
        epsilons,
        adaptive_eht,
        linewidth=2.0,
        label=METHOD_LABELS["adaptive_eht"],
        **METHOD_STYLES["adaptive_eht"],
    )
    ax.axvline(
        STANDARD_HT_BIAS_BOUND,
        color=REFERENCE_COLOR,
        linestyle=":",
        linewidth=1.8,
        label="SHT bias bound",
    )
    ax.axvline(
        FIXED_M_BIAS_BOUND,
        color=REFERENCE_COLOR,
        linestyle="-.",
        linewidth=1.8,
        label=rf"{METHOD_LABELS['fixed_m_eht']} bias bound "
        rf"($m={FIXED_AMPLIFICATION}$)",
    )
    for row in rows:
        epsilon = float(row["epsilon"])
        horizontal_alignment = (
            "left"
            if epsilon == max(epsilons)
            else "right" if epsilon == min(epsilons) else "center"
        )
        horizontal_offset = 10 if epsilon == min(epsilons) else 0
        ax.annotate(
            rf"$m\leq {int(float(row['max_m']))}$",
            xy=(epsilon, float(row["adaptive_eht_restarts"])),
            xytext=(horizontal_offset, -12),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            color=METHOD_STYLES["adaptive_eht"]["color"],
            bbox={
                "boxstyle": "round,pad=0.1",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    lower_limit = min(min(epsilons), STANDARD_HT_BIAS_BOUND) / 1.20
    ax.set_xlim(1.12 * max(epsilons), lower_limit)
    ax.set_xticks((3e-2, 1e-2, 3e-3))
    ax.set_xticklabels(
        (r"$3\times10^{-2}$", r"$10^{-2}$", r"$3\times10^{-3}$")
    )
    ax.set_xlabel(r"target effective-phase accuracy")
    ax.set_ylabel("device restarts")
    ax.margins(y=0.15)
    ax.grid(alpha=0.28, which="major")
    ax.grid(alpha=0.12, which="minor")
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
    ratio_rows = [
        row
        for row in rows
        if math.isfinite(float(row["restart_ratio"]))
    ]
    ratio_epsilons = [float(row["epsilon"]) for row in ratio_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        ratio_epsilons,
        [float(row["restart_ratio"]) for row in ratio_rows],
        linewidth=2.2,
        **METHOD_STYLES["adaptive_eht"],
    )
    ax.axvline(
        STANDARD_HT_BIAS_BOUND,
        color=REFERENCE_COLOR,
        linestyle=":",
        linewidth=1.8,
        label="SHT bias bound",
    )
    for row in ratio_rows:
        epsilon = float(row["epsilon"])
        horizontal_alignment = (
            "left"
            if epsilon == max(ratio_epsilons)
            else "right" if epsilon == min(ratio_epsilons) else "center"
        )
        ax.annotate(
            rf"$m\leq {int(float(row['max_m']))}$",
            xy=(epsilon, float(row["restart_ratio"])),
            xytext=(0, -16),
            textcoords="offset points",
            ha=horizontal_alignment,
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            color=METHOD_STYLES["adaptive_eht"]["color"],
        )
    ax.set_xscale("log")
    lower_limit = min(min(epsilons), STANDARD_HT_BIAS_BOUND) / 1.12
    ax.set_xlim(1.12 * max(epsilons), lower_limit)
    ax.set_xlabel(r"target effective-phase accuracy")
    ax.set_ylabel("SHT restarts / AEHT restarts")
    ax.margins(y=0.20)
    ax.grid(alpha=0.28, which="major")
    ax.grid(alpha=0.12, which="minor")
    ax.legend()
    fig.tight_layout()
    path = save_figure(
        fig,
        output_dir / "iterative_restart_ratio.png",
    )
    plt.close(fig)
    return path


def write_plots(
    *,
    rows: Sequence[dict[str, float | str]],
    output_dir: Path,
) -> list[Path]:
    try:
        configure_matplotlib_cache()
        import matplotlib.pyplot  # noqa: F401

        paths = [
            plot_iterative_restarts(
                rows,
                output_dir,
            ),
            plot_iterative_restart_ratio(rows, output_dir),
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
    print_setup()
    print("schedule=geometric")
    rows = accuracy_rows()

    print_accuracy_table(rows)
    write_csv_results(rows, output_dir)
    if make_plots:
        write_plots(
            rows=rows,
            output_dir=output_dir,
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Standard HT, Fixed-m EHT, and Adaptive EHT "
            "for an imperfect state."
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
