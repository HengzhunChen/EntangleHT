#!/usr/bin/env python3
"""Noiseless circuit-sampled error decay for exact and imperfect states.

Commands and options:
  run                         Simulate both state models and append CSV rows.
    --epsilons VALUE [...]    Set the target-accuracy grid.
    --repetitions N           Set the number of independent sampled estimates.
    --output PATH             Set the output CSV file.
    --force                   Replace the output CSV before running.
    --plot                    Plot the results produced by this run.
    --output-dir PATH         Set the figure directory used by --plot.
  plot                        Print selected rows and generate decay figures.
    --input PATH              Read results from this CSV file.
    --output-dir PATH         Set the figure directory.
    --comparison-curves CURVE Select one or more curves from entangled,
                              one-quadrature, and two-quadrature.

The exact-state comparison uses the one-quadrature standard HT baseline; the
imperfect-state comparison uses the two-quadrature standard HT baseline. With
no curve option, ``plot`` prints and plots every applicable method.

Examples:
  python error_decay.py run --force --plot
  python error_decay.py run --epsilons 0.04 0.02 --repetitions 10
  python error_decay.py plot --comparison-curves entangled two-quadrature
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from entangle_ht.baselines import (
    standard_hadamard_shots,
    two_quadrature_standard_shots,
)
from entangle_ht.circuit_models.imperfect import (
    imperfect_contrast_lower_bound,
    imperfect_overlap,
)
from entangle_ht.planning import optimize_exact_geometric, optimize_imperfect_geometric
from entangle_ht.records import EstimationConfig
from entangle_ht.resources import ResourceModel
from entangle_ht.simulation import (
    require_qiskit_aer,
    run_exact_plan,
    run_imperfect_plan,
    run_one_quadrature_exact_standard,
    run_two_quadrature_imperfect_standard,
    simulator_factory,
    summarize_errors,
)
from entangle_ht.utilities import phase_error
from example_utils import configure_matplotlib_cache, save_figure


# *****************************************************************************
# Experiment settings
# *****************************************************************************

PHI_TRUE = 0.35
EXACT_THETA_0 = 0.20
EXACT_DELTA_0 = 0.20
THETA_TARGET = 0.35
ETA = 0.02
IMPERFECT_THETA_0 = 0.20
IMPERFECT_DELTA_0 = 0.20
P_SUCCESS_TOTAL = 0.95
BRANCH_MARGIN = math.pi / 10
RESOURCES = ResourceModel(device_qubits=2500, system_qubits=1)

# Circuit simulation at m=100 is intractable on a classical statevector.
# This simulation-only cap is separate from the Q=2500 planning experiment.
SIMULATION_HARDWARE_CAP = 10
GAMMA_GRID = (0.45, 0.55, 0.65, 0.75, 0.85)
OMEGA_GRID = (0.45, 0.60, 0.75, 0.90)
EPSILON_GRID = (0.04, 0.03, 0.02, 0.015, 0.01)
REPETITIONS = 1
BASE_SEED = 20260727
OUTPUT_DIR = Path("outputs/error_decay")
CSV_PATH = OUTPUT_DIR / "noiseless_error_decay.csv"

COMPARISON_CURVE_CHOICES = (
    "entangled",
    "one-quadrature",
    "two-quadrature",
)
# Preserve existing CSV method values while exposing consistent curve names.
CURVE_BY_METHOD = {
    "entangled": "entangled",
    "standard": "one-quadrature",
    "two_quadrature": "two-quadrature",
}
CURVE_LABELS = {
    "entangled": "entangled HT",
    "one-quadrature": "one-quadrature standard HT",
    "two-quadrature": "two-quadrature standard HT",
}
CURVE_STYLES = {
    "entangled": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "one-quadrature": {"color": "#6A3D9A", "linestyle": "-.", "marker": "D"},
    "two-quadrature": {"color": "#009E73", "linestyle": ":", "marker": "P"},
}


# *****************************************************************************
# Model configuration
# *****************************************************************************

def exact_config(epsilon: float) -> EstimationConfig:
    return EstimationConfig(
        phase=PHI_TRUE,
        initial_reference=EXACT_THETA_0,
        initial_bound=EXACT_DELTA_0,
        target_accuracy=epsilon,
        total_success_probability=P_SUCCESS_TOTAL,
        branch_margin=BRANCH_MARGIN,
        hardware_amplification_cap=SIMULATION_HARDWARE_CAP,
        contrast=1.0,
        contrast_lower_bound=1.0,
    )


def model_parameters() -> tuple[float, float, float, float]:
    contrast, effective_phase, _ = imperfect_overlap(THETA_TARGET, ETA)
    contrast_lower_bound = imperfect_contrast_lower_bound(ETA)
    preparation_floor = abs(phase_error(effective_phase, THETA_TARGET))
    return contrast, effective_phase, contrast_lower_bound, preparation_floor


def imperfect_config(epsilon: float) -> EstimationConfig:
    contrast, effective_phase, contrast_lower_bound, _ = model_parameters()
    return EstimationConfig(
        phase=effective_phase,
        initial_reference=IMPERFECT_THETA_0,
        initial_bound=IMPERFECT_DELTA_0,
        target_accuracy=epsilon,
        total_success_probability=P_SUCCESS_TOTAL,
        branch_margin=BRANCH_MARGIN,
        hardware_amplification_cap=SIMULATION_HARDWARE_CAP,
        contrast=contrast,
        contrast_lower_bound=contrast_lower_bound,
    )


# *****************************************************************************
# CSV storage
# *****************************************************************************

FIELDNAMES = [
    "model",
    "epsilon",
    "method",
    "one_qubit_error_rate",
    "two_qubit_error_rate",
    "readout_error_rate",
    "shots",
    "queries",
    "rounds",
    "max_m",
    "repetitions",
    "mean_effective_error",
    "median_effective_error",
    "rmse_effective_error",
    "std_effective_error",
    "ci95_effective_error",
    "mean_target_error",
    "median_target_error",
    "rmse_target_error",
    "std_target_error",
    "ci95_target_error",
    "seconds",
]


def append_rows(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = set(FIELDNAMES) - {"model", "method"}
    return [
        {key: (float(value) if key in numeric else value) for key, value in row.items()}
        for row in rows
    ]


# *****************************************************************************
# Circuit simulation
# *****************************************************************************

def simulate_exact_epsilon(
    epsilon: float,
    repetitions: int,
    simulator_kwargs: dict[str, float | str] | None = None,
) -> list[dict[str, float | str]]:
    config = exact_config(epsilon)
    result = optimize_exact_geometric(
        base_config=config,
        gamma_grid=GAMMA_GRID,
        resources=RESOURCES,
    )
    standard_shots = standard_hadamard_shots(
        epsilon=epsilon,
        delta=config.initial_bound,
        p_fail=1.0 - config.total_success_probability,
    )

    ent_effective_errors: list[float] = []
    ent_target_errors: list[float] = []
    std_effective_errors: list[float] = []
    std_target_errors: list[float] = []
    start = time.perf_counter()
    noisy = simulator_kwargs is not None
    factory = simulator_factory(noisy, **(simulator_kwargs or {}))

    for repetition in range(repetitions):
        seed = BASE_SEED + 10_000 * repetition + int(round(epsilon * 1_000_000))
        simulator = factory()
        entangled = run_exact_plan(
            config=result.config,
            plan=result.plan,
            simulator=simulator,
            seed=seed,
        )
        standard = run_one_quadrature_exact_standard(
            theta_true=config.phase,
            theta_ref=config.initial_reference,
            shots=standard_shots,
            simulator=simulator,
            seed=seed + 503,
        )
        ent_effective_errors.append(entangled.effective_error)
        ent_target_errors.append(entangled.target_error)
        std_effective_errors.append(standard.effective_error)
        std_target_errors.append(standard.target_error)

    seconds = time.perf_counter() - start
    return [
        result_row(
            model="exact",
            epsilon=epsilon,
            method="entangled",
            shots=result.plan.total_shots,
            queries=result.plan.total_queries,
            rounds=result.rounds,
            max_m=result.max_amplification,
            repetitions=repetitions,
            effective_errors=ent_effective_errors,
            target_errors=ent_target_errors,
            seconds=seconds,
            simulator_kwargs=simulator_kwargs,
        ),
        result_row(
            model="exact",
            epsilon=epsilon,
            method="standard",
            shots=standard_shots,
            queries=standard_shots,
            rounds=1,
            max_m=1,
            repetitions=repetitions,
            effective_errors=std_effective_errors,
            target_errors=std_target_errors,
            seconds=seconds,
            simulator_kwargs=simulator_kwargs,
        ),
    ]


def simulate_imperfect_epsilon(
    epsilon: float,
    repetitions: int,
    simulator_kwargs: dict[str, float | str] | None = None,
) -> list[dict[str, float | str]]:
    config = imperfect_config(epsilon)
    result = optimize_imperfect_geometric(
        base_config=config,
        gamma_grid=GAMMA_GRID,
        omega_grid=OMEGA_GRID,
        resources=RESOURCES,
    )
    standard_total_shots = two_quadrature_standard_shots(
        epsilon=epsilon,
        rho0=config.contrast_lower_bound,
        p_fail=1.0 - config.total_success_probability,
    )
    shots_per_quadrature = standard_total_shots // 2

    ent_effective_errors: list[float] = []
    ent_target_errors: list[float] = []
    std_effective_errors: list[float] = []
    std_target_errors: list[float] = []
    start = time.perf_counter()
    noisy = simulator_kwargs is not None
    factory = simulator_factory(noisy, **(simulator_kwargs or {}))

    for repetition in range(repetitions):
        seed = BASE_SEED + 20_000 * repetition + int(round(epsilon * 1_000_000))
        simulator = factory()
        entangled = run_imperfect_plan(
            config=result.config,
            plan=result.plan,
            theta_target=THETA_TARGET,
            eta=ETA,
            simulator=simulator,
            seed=seed,
        )
        standard = run_two_quadrature_imperfect_standard(
            theta_target=THETA_TARGET,
            eta=ETA,
            shots_per_quadrature=shots_per_quadrature,
            simulator=simulator,
            seed=seed + 503,
        )
        ent_effective_errors.append(entangled.effective_error)
        ent_target_errors.append(entangled.target_error)
        std_effective_errors.append(standard.effective_error)
        std_target_errors.append(standard.target_error)

    seconds = time.perf_counter() - start
    return [
        result_row(
            model="imperfect",
            epsilon=epsilon,
            method="entangled",
            shots=result.plan.total_shots,
            queries=result.plan.total_queries,
            rounds=result.rounds,
            max_m=result.max_amplification,
            repetitions=repetitions,
            effective_errors=ent_effective_errors,
            target_errors=ent_target_errors,
            seconds=seconds,
            simulator_kwargs=simulator_kwargs,
        ),
        result_row(
            model="imperfect",
            epsilon=epsilon,
            method="two_quadrature",
            shots=standard_total_shots,
            queries=standard_total_shots,
            rounds=1,
            max_m=1,
            repetitions=repetitions,
            effective_errors=std_effective_errors,
            target_errors=std_target_errors,
            seconds=seconds,
            simulator_kwargs=simulator_kwargs,
        ),
    ]


def result_row(
    *,
    model: str,
    epsilon: float,
    method: str,
    shots: int,
    queries: int,
    rounds: int,
    max_m: int,
    repetitions: int,
    effective_errors: list[float],
    target_errors: list[float],
    seconds: float,
    simulator_kwargs: dict[str, float | str] | None = None,
) -> dict[str, float | str]:
    effective = summarize_errors(effective_errors)
    target = summarize_errors(target_errors)
    noise = simulator_kwargs or {}
    return {
        "model": model,
        "epsilon": epsilon,
        "method": method,
        "one_qubit_error_rate": float(noise.get("one_qubit_error_rate", 0.0)),
        "two_qubit_error_rate": float(noise.get("two_qubit_error_rate", 0.0)),
        "readout_error_rate": float(noise.get("readout_error_rate", 0.0)),
        "shots": float(shots),
        "queries": float(queries),
        "rounds": float(rounds),
        "max_m": float(max_m),
        "repetitions": float(repetitions),
        "mean_effective_error": effective["mean"],
        "median_effective_error": effective["median"],
        "rmse_effective_error": effective["rmse"],
        "std_effective_error": effective["standard_deviation"],
        "ci95_effective_error": effective["rmse_ci95_half_width"],
        "mean_target_error": target["mean"],
        "median_target_error": target["median"],
        "rmse_target_error": target["rmse"],
        "std_target_error": target["standard_deviation"],
        "ci95_target_error": target["rmse_ci95_half_width"],
        "seconds": seconds,
    }


# *****************************************************************************
# Experiment execution and console output
# *****************************************************************************

def run_simulations(
    *,
    epsilons: Sequence[float],
    repetitions: int,
    output_path: Path,
    force: bool,
) -> list[dict[str, float | str]]:
    require_qiskit_aer()
    if force and output_path.exists():
        output_path.unlink()

    run_rows: list[dict[str, float | str]] = []
    for epsilon in epsilons:
        rows = simulate_exact_epsilon(epsilon, repetitions)
        rows += simulate_imperfect_epsilon(epsilon, repetitions)
        append_rows(output_path, rows)
        run_rows.extend(rows)
        print_rows(rows)
        print(f"[saved] {output_path}", flush=True)
    return run_rows


def print_rows(rows: Sequence[dict[str, float | str]]) -> None:
    print(
        "model\tepsilon\tmethod\tshots\tqueries\trounds\tmax_m\t"
        "rmse_eff\trmse_target\tseconds"
    )
    for row in rows:
        method = str(row["method"])
        curve = CURVE_BY_METHOD.get(method, method)
        print(
            f"{row['model']}\t"
            f"{float(row['epsilon']):.6g}\t"
            f"{curve}\t"
            f"{int(float(row['shots']))}\t"
            f"{int(float(row['queries']))}\t"
            f"{int(float(row['rounds']))}\t"
            f"{int(float(row['max_m']))}\t"
            f"{float(row['rmse_effective_error']):.6g}\t"
            f"{float(row['rmse_target_error']):.6g}\t"
            f"{float(row['seconds']):.2f}"
        )


# *****************************************************************************
# Plotting
# *****************************************************************************

def plot_rows(
    rows: Sequence[dict[str, float | str]],
    output_dir: Path,
    *,
    experiment_label: str = "Noiseless",
    filename_prefix: str = "noiseless",
    comparison_curves: Sequence[str] = COMPARISON_CURVE_CHOICES,
) -> None:
    if not rows:
        raise ValueError("no rows to plot")
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for model in ("exact", "imperfect"):
        model_rows = [
            row
            for row in rows
            if row["model"] == model
            and CURVE_BY_METHOD.get(str(row["method"])) in comparison_curves
        ]
        if not model_rows:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for method in CURVE_BY_METHOD:
            curve = CURVE_BY_METHOD[method]
            if curve not in comparison_curves:
                continue
            method_rows = sorted(
                [row for row in model_rows if row["method"] == method],
                key=lambda row: float(row["epsilon"]),
                reverse=True,
            )
            if not method_rows:
                continue
            ax.errorbar(
                [float(row["epsilon"]) for row in method_rows],
                [max(float(row["rmse_target_error"]), 1e-16) for row in method_rows],
                yerr=[float(row["ci95_target_error"]) for row in method_rows],
                linewidth=2.0,
                capsize=3,
                label=CURVE_LABELS[curve],
                **CURVE_STYLES[curve],
            )
        epsilons = sorted({float(row["epsilon"]) for row in model_rows}, reverse=True)
        ax.plot(
            epsilons,
            epsilons,
            color="#4D4D4D",
            linestyle="--",
            label="target epsilon",
        )
        if model == "imperfect":
            _, _, _, floor = model_parameters()
            ax.axhline(floor, color="#D55E00", linestyle=":", label="target floor")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if len(epsilons) > 1:
            ax.set_xlim(max(epsilons), min(epsilons))
        ax.set_xlabel("target accuracy epsilon")
        ax.set_ylabel("RMSE target-phase error")
        ax.set_title(f"{experiment_label} sampled error decay: {model}")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        path = save_figure(
            fig,
            output_dir / f"{filename_prefix}_{model}_error_decay.png",
        )
        plt.close(fig)
        print(f"[plot] {path}")


# *****************************************************************************
# Command-line interface
# *****************************************************************************

def parse_float_list(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Noiseless sampled error-decay experiments for entangled HT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="simulate both state models and append CSV rows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--epsilons",
        nargs="+",
        default=[str(value) for value in EPSILON_GRID],
        help="target-accuracy values",
    )
    run_parser.add_argument(
        "--repetitions",
        type=int,
        default=REPETITIONS,
        help="independent estimates per accuracy and method",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=CSV_PATH,
        help="CSV file to append",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="replace the output CSV before running",
    )
    run_parser.add_argument(
        "--plot",
        action="store_true",
        help="plot only the results produced by this run",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for plots generated by --plot",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="print selected rows and generate decay figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot_parser.add_argument(
        "--input",
        type=Path,
        default=CSV_PATH,
        help="CSV file to read",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for generated plots",
    )
    plot_parser.add_argument(
        "--comparison-curves",
        nargs="+",
        choices=COMPARISON_CURVE_CHOICES,
        default=COMPARISON_CURVE_CHOICES,
        metavar="CURVE",
        help="curves included in the printed table and decay plots",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        try:
            rows = run_simulations(
                epsilons=parse_float_list(args.epsilons),
                repetitions=args.repetitions,
                output_path=args.output,
                force=args.force,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.plot:
            plot_rows(rows, args.output_dir)
        return
    if args.command == "plot":
        rows = load_rows(args.input)
        rows = [
            row
            for row in rows
            if CURVE_BY_METHOD.get(str(row["method"])) in args.comparison_curves
        ]
        print_rows(rows)
        plot_rows(
            rows,
            args.output_dir,
            comparison_curves=args.comparison_curves,
        )
        return
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
