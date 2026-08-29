#!/usr/bin/env python3
"""Noiseless error decay for Standard HT and Adaptive EHT.

Commands and options:
  run                         Simulate both state models and append CSV rows.
    --epsilons VALUE [...]    Set the target-accuracy grid.
    --output PATH             Set the output CSV file.
    --force                   Replace the output CSV before running.
    --plot                    Plot the results produced by this run.
    --output-dir PATH         Set the figure directory used by --plot.
  plot                        Print selected rows and generate decay figures.
    --input PATH              Read results from this CSV file.
    --output-dir PATH         Set the figure directory.

The imperfect-state comparison reports effective-phase error and includes a
contrast-unaware Standard HT baseline so its nonzero bias can be seen.

Examples:
  python error_decay.py run --force --plot
  python error_decay.py run --epsilons 0.04 0.02
  python error_decay.py plot
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from entangle_ht.baselines import standard_hadamard_shots
from entangle_ht.certification import contrast_bias_bound
from entangle_ht.circuit_models.imperfect import (
    imperfect_model_parameters,
)
from entangle_ht.planning import (
    optimize_exact_geometric,
    optimize_imperfect_geometric,
)
from entangle_ht.records import EstimationConfig
from entangle_ht.resources import ResourceModel, restarts_from_shots
from entangle_ht.simulation import (
    build_simulator,
    require_qiskit_aer,
    run_exact_plan,
    run_imperfect_plan,
    run_one_quadrature_exact_standard,
    run_one_quadrature_imperfect_standard,
)
from example_utils import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_STYLES,
    REFERENCE_COLOR,
    configure_matplotlib_cache,
    save_figure,
)


# *****************************************************************************
# Experiment settings
# *****************************************************************************

THETA_TARGET = 2.0
# This reference keeps the imperfect effective phase inside INITIAL_BOUND.
INITIAL_REFERENCE = 1.8
INITIAL_BOUND = 0.2
ETA = 0.01
P_SUCCESS_TOTAL = 0.95
BRANCH_MARGIN = math.pi / 4
RESOURCES = ResourceModel(device_qubits=2500, system_qubits=1)

# Circuit simulation at m=100 is intractable on a classical statevector.
# This simulation-only cap is separate from the Q=2500 planning experiment.
SIMULATION_HARDWARE_CAP = 8
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
    5e-2, 3e-2, 2e-2, 1e-2, 8e-3, 6e-3, 4e-3, 2e-3, 1e-3, 5e-4,
)
BASE_SEED = 20260727
OUTPUT_DIR = Path("outputs/error_decay")
CSV_PATH = OUTPUT_DIR / "noiseless_error_decay.csv"


METHODS = ("adaptive_eht", "standard")
RESTART_ACCURACY_STYLES = {
    ("standard", "target"): {
        "color": METHOD_COLORS["standard"],
        "linestyle": "--",
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": METHOD_COLORS["standard"],
        "markeredgewidth": 1.5,
    },
    ("adaptive_eht", "target"): {
        "color": METHOD_COLORS["adaptive_eht"],
        "linestyle": "--",
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": METHOD_COLORS["adaptive_eht"],
        "markeredgewidth": 1.5,
    },
    ("standard", "actual"): {
        "color": METHOD_COLORS["standard"],
        "linestyle": "-",
        "marker": "D",
        "markerfacecolor": METHOD_COLORS["standard"],
    },
    ("adaptive_eht", "actual"): {
        "color": METHOD_COLORS["adaptive_eht"],
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": METHOD_COLORS["adaptive_eht"],
    },
}


# *****************************************************************************
# Model configuration
# *****************************************************************************

MODEL_PARAMETERS = imperfect_model_parameters(THETA_TARGET, ETA)
EXACT_BASE_CONFIG = EstimationConfig(
    phase=THETA_TARGET,
    initial_reference=INITIAL_REFERENCE,
    initial_bound=INITIAL_BOUND,
    target_accuracy=EPSILON_GRID[0],
    total_success_probability=P_SUCCESS_TOTAL,
    branch_margin=BRANCH_MARGIN,
    hardware_amplification_cap=SIMULATION_HARDWARE_CAP,
    contrast=1.0,
    contrast_lower_bound=1.0,
)
IMPERFECT_BASE_CONFIG = replace(
    EXACT_BASE_CONFIG,
    phase=MODEL_PARAMETERS.effective_phase,
    contrast=MODEL_PARAMETERS.contrast,
    contrast_lower_bound=MODEL_PARAMETERS.contrast_lower_bound,
)
STANDARD_HT_BIAS_BOUND = contrast_bias_bound(
    1,
    INITIAL_BOUND,
    MODEL_PARAMETERS.contrast_lower_bound,
)


def exact_config(epsilon: float) -> EstimationConfig:
    return replace(EXACT_BASE_CONFIG, target_accuracy=epsilon)


def imperfect_config(epsilon: float) -> EstimationConfig:
    return replace(IMPERFECT_BASE_CONFIG, target_accuracy=epsilon)


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
    "restarts",
    "rounds",
    "max_m",
    "effective_error",
    "target_error",
    "seconds",
]


def append_rows(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if not write_header:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != FIELDNAMES:
            raise ValueError(
                f"CSV schema mismatch for {path}; rerun with --force"
            )
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                f"CSV schema mismatch for {path}; rerun the simulation "
                "with --force"
            )
        rows = list(reader)
    numeric = set(FIELDNAMES) - {"model", "method"}
    converted_rows = [
        {key: (float(value) if key in numeric else value) for key, value in row.items()}
        for row in rows
    ]
    for row in converted_rows:
        if row["method"] == "entangled":
            row["method"] = "adaptive_eht"
    return converted_rows


# *****************************************************************************
# Circuit simulation
# *****************************************************************************

def simulate_exact(
    config: EstimationConfig,
    simulator_kwargs: dict[str, float | str] | None = None,
    *,
    resources: ResourceModel = RESOURCES,
    gamma_grid: Sequence[float] = GAMMA_GRID,
    base_seed: int = BASE_SEED,
) -> list[dict[str, float | str]]:
    epsilon = config.target_accuracy
    result = optimize_exact_geometric(
        base_config=config,
        gamma_grid=gamma_grid,
        resources=resources,
    )
    standard_shots = standard_hadamard_shots(
        epsilon=epsilon,
        delta=config.initial_bound,
        p_fail=1.0 - config.total_success_probability,
    )

    seed = base_seed + int(round(epsilon * 1_000_000))
    simulator = build_simulator(simulator_kwargs)

    start = time.perf_counter()
    adaptive_eht = run_exact_plan(
        config=result.config,
        plan=result.plan,
        simulator=simulator,
        seed=seed,
    )
    adaptive_eht_seconds = time.perf_counter() - start

    start = time.perf_counter()
    standard = run_one_quadrature_exact_standard(
        theta_true=config.phase,
        theta_ref=config.initial_reference,
        shots=standard_shots,
        simulator=simulator,
        seed=seed + 503,
    )
    standard_seconds = time.perf_counter() - start
    return [
        result_row(
            model="exact",
            epsilon=epsilon,
            method="adaptive_eht",
            shots=result.plan.total_shots,
            restarts=result.restarts,
            rounds=result.rounds,
            max_m=result.max_amplification,
            effective_error=adaptive_eht.effective_error,
            target_error=adaptive_eht.target_error,
            seconds=adaptive_eht_seconds,
            simulator_kwargs=simulator_kwargs,
        ),
        result_row(
            model="exact",
            epsilon=epsilon,
            method="standard",
            shots=standard_shots,
            restarts=restarts_from_shots(
                standard_shots,
                resources.packing_capacity(1),
            ),
            rounds=1,
            max_m=1,
            effective_error=standard.effective_error,
            target_error=standard.target_error,
            seconds=standard_seconds,
            simulator_kwargs=simulator_kwargs,
        ),
    ]


def simulate_imperfect(
    config: EstimationConfig,
    simulator_kwargs: dict[str, float | str] | None = None,
    *,
    theta_target: float = THETA_TARGET,
    eta: float = ETA,
    resources: ResourceModel = RESOURCES,
    gamma_grid: Sequence[float] = GAMMA_GRID,
    omega_grid: Sequence[float] = OMEGA_GRID,
    base_seed: int = BASE_SEED,
) -> list[dict[str, float | str]]:
    epsilon = config.target_accuracy
    result = optimize_imperfect_geometric(
        base_config=config,
        gamma_grid=gamma_grid,
        omega_grid=omega_grid,
        resources=resources,
    )
    # Standard HT spends epsilon only on statistical error and ignores contrast
    # bias. Decreasing epsilon therefore exposes its fixed infinite-shot bias.
    standard_shots = standard_hadamard_shots(
        epsilon=epsilon,
        delta=config.initial_bound,
        p_fail=1.0 - config.total_success_probability,
    )

    seed = base_seed + int(round(epsilon * 1_000_000))
    simulator = build_simulator(simulator_kwargs)

    start = time.perf_counter()
    adaptive_eht = run_imperfect_plan(
        config=result.config,
        plan=result.plan,
        theta_target=theta_target,
        eta=eta,
        simulator=simulator,
        seed=seed,
    )
    adaptive_eht_seconds = time.perf_counter() - start

    start = time.perf_counter()
    standard = run_one_quadrature_imperfect_standard(
        theta_target=theta_target,
        eta=eta,
        theta_ref=config.initial_reference,
        shots=standard_shots,
        simulator=simulator,
        seed=seed + 503,
    )
    standard_seconds = time.perf_counter() - start
    return [
        result_row(
            model="imperfect",
            epsilon=epsilon,
            method="adaptive_eht",
            shots=result.plan.total_shots,
            restarts=result.restarts,
            rounds=result.rounds,
            max_m=result.max_amplification,
            effective_error=adaptive_eht.effective_error,
            target_error=adaptive_eht.target_error,
            seconds=adaptive_eht_seconds,
            simulator_kwargs=simulator_kwargs,
        ),
        result_row(
            model="imperfect",
            epsilon=epsilon,
            method="standard",
            shots=standard_shots,
            restarts=restarts_from_shots(
                standard_shots,
                resources.packing_capacity(1),
            ),
            rounds=1,
            max_m=1,
            effective_error=standard.effective_error,
            target_error=standard.target_error,
            seconds=standard_seconds,
            simulator_kwargs=simulator_kwargs,
        ),
    ]


def result_row(
    *,
    model: str,
    epsilon: float,
    method: str,
    shots: int,
    restarts: int,
    rounds: int,
    max_m: int,
    effective_error: float,
    target_error: float,
    seconds: float,
    simulator_kwargs: dict[str, float | str] | None = None,
) -> dict[str, float | str]:
    noise = simulator_kwargs or {}
    return {
        "model": model,
        "epsilon": epsilon,
        "method": method,
        "one_qubit_error_rate": float(noise.get("one_qubit_error_rate", 0.0)),
        "two_qubit_error_rate": float(noise.get("two_qubit_error_rate", 0.0)),
        "readout_error_rate": float(noise.get("readout_error_rate", 0.0)),
        "shots": float(shots),
        "restarts": float(restarts),
        "rounds": float(rounds),
        "max_m": float(max_m),
        "effective_error": effective_error,
        "target_error": target_error,
        "seconds": seconds,
    }


# *****************************************************************************
# Experiment execution and console output
# *****************************************************************************

def run_simulations(
    *,
    epsilons: Sequence[float],
    output_path: Path,
    force: bool,
) -> list[dict[str, float | str]]:
    require_qiskit_aer()
    if force and output_path.exists():
        output_path.unlink()

    run_rows: list[dict[str, float | str]] = []
    for epsilon in epsilons:
        rows = simulate_exact(exact_config(epsilon))
        rows += simulate_imperfect(imperfect_config(epsilon))
        append_rows(output_path, rows)
        run_rows.extend(rows)
        print_rows(rows)
        print(f"[saved] {output_path}", flush=True)
    return run_rows


def print_rows(rows: Sequence[dict[str, float | str]]) -> None:
    print(
        "model\tepsilon\tmethod\tshots\trestarts\trounds\tmax_m\t"
        "effective_error\ttarget_error\tseconds"
    )
    for row in rows:
        method = str(row["method"])
        method_label = METHOD_LABELS.get(method, method)
        print(
            f"{row['model']}\t"
            f"{float(row['epsilon']):.6g}\t"
            f"{method_label}\t"
            f"{int(float(row['shots']))}\t"
            f"{int(float(row['restarts']))}\t"
            f"{int(float(row['rounds']))}\t"
            f"{int(float(row['max_m']))}\t"
            f"{float(row['effective_error']):.6g}\t"
            f"{float(row['target_error']):.6g}\t"
            f"{float(row['seconds']):.2f}"
        )


# *****************************************************************************
# Plotting
# *****************************************************************************

def error_plot_semantics(
    model: str,
    experiment_label: str,
) -> tuple[str, str, str, str]:
    """Return the error field and labels appropriate for a state model."""
    if model == "exact":
        return (
            "target_error",
            r"target phase accuracy",
            "absolute phase error",
            f"{experiment_label} exact-eigenstate phase error decay",
        )
    if model == "imperfect":
        return (
            "effective_error",
            r"target effective-phase accuracy",
            "absolute effective-phase error",
            (
                f"{experiment_label} imperfect-eigenstate "
                "effective-phase error decay"
            ),
        )
    raise ValueError(f"unknown state model {model!r}")


def plot_rows(
    rows: Sequence[dict[str, float | str]],
    output_dir: Path,
    *,
    experiment_label: str = "Noiseless",
    filename_prefix: str = "noiseless",
    standard_bias_bound: float,
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
            and str(row["method"]) in METHODS
        ]
        if not model_rows:
            continue

        error_field, x_label, y_label, _title = error_plot_semantics(
            model,
            experiment_label,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in METHODS:
            method_rows = sorted(
                [row for row in model_rows if row["method"] == method],
                key=lambda row: float(row["epsilon"]),
                reverse=True,
            )
            if not method_rows:
                continue
            label = METHOD_LABELS[method]
            ax.plot(
                [float(row["epsilon"]) for row in method_rows],
                [
                    max(float(row[error_field]), 1e-16)
                    for row in method_rows
                ],
                linewidth=2.0,
                label=label,
                **METHOD_STYLES[method],
            )
        epsilons = sorted({float(row["epsilon"]) for row in model_rows}, reverse=True)
        ax.plot(
            epsilons,
            epsilons,
            color=REFERENCE_COLOR,
            linestyle="--",
            label=r"target accuracy",
        )
        if model == "imperfect":
            ax.axhline(
                standard_bias_bound,
                color=REFERENCE_COLOR,
                linestyle=":",
                label="SHT bias bound",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        if len(epsilons) > 1:
            ax.set_xlim(max(epsilons), min(epsilons))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.28, which="major")
        ax.grid(alpha=0.12, which="minor")
        ax.legend()
        fig.tight_layout()
        path = save_figure(
            fig,
            output_dir / f"{filename_prefix}_{model}_error_decay.png",
        )
        plt.close(fig)
        print(f"[plot] {path}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for method in METHODS:
            method_rows = sorted(
                [row for row in model_rows if row["method"] == method],
                key=lambda row: float(row["restarts"]),
            )
            if not method_rows:
                continue
            label = METHOD_LABELS[method]
            restarts = [float(row["restarts"]) for row in method_rows]
            target_tolerances = [float(row["epsilon"]) for row in method_rows]
            if model == "imperfect" and method == "standard":
                target_tolerances = [
                    epsilon + standard_bias_bound
                    for epsilon in target_tolerances
                ]
            target_style = RESTART_ACCURACY_STYLES[(method, "target")]
            ax.plot(
                restarts,
                target_tolerances,
                linewidth=2.0,
                label=rf"{label} target tolerance",
                **target_style,
            )
            actual_style = RESTART_ACCURACY_STYLES[(method, "actual")]
            ax.plot(
                restarts,
                [max(float(row[error_field]), 1e-16) for row in method_rows],
                linewidth=2.0,
                label=f"{label} actual error",
                **actual_style,
            )
        if model == "imperfect":
            ax.axhline(
                standard_bias_bound,
                color=REFERENCE_COLOR,
                linestyle=":",
                label="SHT bias bound",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("device restarts required for target accuracy")
        ax.set_ylabel(restart_accuracy_ylabel(model))
        ax.grid(alpha=0.28, which="major")
        ax.grid(alpha=0.12, which="minor")
        ax.legend()
        fig.tight_layout()
        path = save_figure(
            fig,
            output_dir / f"{filename_prefix}_{model}_restart_accuracy.png",
        )
        plt.close(fig)
        print(f"[plot] {path}")


def restart_accuracy_ylabel(model: str) -> str:
    if model == "exact":
        return "phase error magnitude"
    if model == "imperfect":
        return "effective-phase error magnitude"
    raise ValueError(f"unknown state model {model!r}")


def restart_accuracy_title(
    model: str,
    experiment_label: str,
) -> str:
    if model == "exact":
        return f"{experiment_label} exact-eigenstate accuracy versus restarts"
    if model == "imperfect":
        return (
            f"{experiment_label} imperfect-eigenstate effective accuracy "
            "versus restarts"
        )
    raise ValueError(f"unknown state model {model!r}")


# *****************************************************************************
# Command-line interface
# *****************************************************************************

def parse_float_list(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Noiseless error decay for Standard HT and Adaptive EHT.",
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        try:
            rows = run_simulations(
                epsilons=parse_float_list(args.epsilons),
                output_path=args.output,
                force=args.force,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.plot:
            plot_rows(
                rows,
                args.output_dir,
                standard_bias_bound=STANDARD_HT_BIAS_BOUND,
            )
        return
    if args.command == "plot":
        rows = load_rows(args.input)
        rows = [row for row in rows if str(row["method"]) in METHODS]
        print_rows(rows)
        plot_rows(
            rows,
            args.output_dir,
            standard_bias_bound=STANDARD_HT_BIAS_BOUND,
        )
        return
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
