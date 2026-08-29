#!/usr/bin/env python3
"""Noisy error decay for Standard HT and Adaptive EHT.

Commands and options:
  run                              Simulate selected state models and append CSV rows.
    --models MODEL [...]           Select imperfect (default), exact, or both.
    --epsilons VALUE [...]         Set the target-accuracy grid.
    --output PATH                  Set the output CSV file.
    --force                        Replace the output CSV before running.
    --one-qubit-error-rate RATE    Set the one-qubit depolarizing error rate.
    --two-qubit-error-rate RATE    Set the two-qubit depolarizing error rate.
    --readout-error-rate RATE      Set the symmetric readout error rate.
    --plot                         Plot the results produced by this run.
    --output-dir PATH              Set the figure directory used by --plot.
  plot                             Print selected rows and generate decay figures.
    --input PATH                   Read results from this CSV file.
    --models MODEL [...]           Select state models to plot.
    --output-dir PATH              Set the figure directory.

The imperfect-state plot uses effective-phase error. It compares Adaptive EHT
with a fixed-reference Standard HT baseline.
With no model option, only the imperfect-state experiment is run and plotted.

Examples:
  python noisy_error_decay.py run --force --plot
  python noisy_error_decay.py run --epsilons 0.04 0.02
  python noisy_error_decay.py run --models exact imperfect
  python noisy_error_decay.py plot
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from error_decay import (
    METHODS,
    append_rows,
    load_rows,
    parse_float_list,
    plot_rows,
    print_rows,
    simulate_exact,
    simulate_imperfect,
)
from entangle_ht.certification import contrast_bias_bound
from entangle_ht.circuit_models.imperfect import (
    imperfect_model_parameters,
)
from entangle_ht.records import EstimationConfig
from entangle_ht.resources import ResourceModel
from entangle_ht.simulation import require_qiskit_aer


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
    5e-2, 3e-2, 2e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3, 8e-4
)
BASE_SEED = 20260812

NOISY_OUTPUT_DIR = Path("outputs/noisy_error_decay")
NOISY_CSV_PATH = NOISY_OUTPUT_DIR / "noisy_error_decay.csv"
ONE_QUBIT_ERROR_RATE = 1e-4
TWO_QUBIT_ERROR_RATE = 1e-3
READOUT_ERROR_RATE = 1e-2
MODEL_CHOICES = ("imperfect", "exact")
DEFAULT_MODELS = ("imperfect",)


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
# Noisy simulation
# *****************************************************************************

def run_noisy_simulations(
    *,
    epsilons: Sequence[float],
    output_path: Path,
    force: bool,
    one_qubit_error_rate: float,
    two_qubit_error_rate: float,
    readout_error_rate: float,
    models: Sequence[str] = DEFAULT_MODELS,
) -> list[dict[str, float | str]]:
    require_qiskit_aer()
    if force and output_path.exists():
        output_path.unlink()

    noise = {
        "method": "automatic",
        "one_qubit_error_rate": one_qubit_error_rate,
        "two_qubit_error_rate": two_qubit_error_rate,
        "readout_error_rate": readout_error_rate,
    }

    run_rows: list[dict[str, float | str]] = []
    for epsilon in epsilons:
        rows: list[dict[str, float | str]] = []
        if "exact" in models:
            rows.extend(
                simulate_exact(
                    exact_config(epsilon),
                    simulator_kwargs=noise,
                    resources=RESOURCES,
                    gamma_grid=GAMMA_GRID,
                    base_seed=BASE_SEED,
                )
            )
        if "imperfect" in models:
            rows.extend(
                simulate_imperfect(
                    imperfect_config(epsilon),
                    simulator_kwargs=noise,
                    theta_target=THETA_TARGET,
                    eta=ETA,
                    resources=RESOURCES,
                    gamma_grid=GAMMA_GRID,
                    omega_grid=OMEGA_GRID,
                    base_seed=BASE_SEED,
                )
            )
        append_rows(output_path, rows)
        run_rows.extend(rows)
        print_rows(rows)
        print(f"[saved] {output_path}", flush=True)
    return run_rows


# *****************************************************************************
# Command-line interface
# *****************************************************************************

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Noisy error decay for Standard HT and Adaptive EHT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="simulate selected state models with circuit noise and append CSV rows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help="state models to simulate",
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
        default=NOISY_CSV_PATH,
        help="CSV file to append",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="replace the output CSV before running",
    )
    run_parser.add_argument(
        "--one-qubit-error-rate",
        type=float,
        default=ONE_QUBIT_ERROR_RATE,
        help="one-qubit depolarizing error rate",
    )
    run_parser.add_argument(
        "--two-qubit-error-rate",
        type=float,
        default=TWO_QUBIT_ERROR_RATE,
        help="two-qubit depolarizing error rate",
    )
    run_parser.add_argument(
        "--readout-error-rate",
        type=float,
        default=READOUT_ERROR_RATE,
        help="symmetric readout error rate",
    )
    run_parser.add_argument(
        "--plot",
        action="store_true",
        help="plot only the results produced by this run",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=NOISY_OUTPUT_DIR,
        help="directory for plots generated by --plot",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="print selected rows and generate noisy decay figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot_parser.add_argument(
        "--input",
        type=Path,
        default=NOISY_CSV_PATH,
        help="CSV file to read",
    )
    plot_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help="state models to plot",
    )
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=NOISY_OUTPUT_DIR,
        help="directory for generated plots",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        try:
            rows = run_noisy_simulations(
                epsilons=parse_float_list(args.epsilons),
                output_path=args.output,
                force=args.force,
                one_qubit_error_rate=args.one_qubit_error_rate,
                two_qubit_error_rate=args.two_qubit_error_rate,
                readout_error_rate=args.readout_error_rate,
                models=args.models,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.plot:
            plot_rows(
                rows,
                args.output_dir,
                experiment_label="Noisy",
                filename_prefix="noisy",
                standard_bias_bound=STANDARD_HT_BIAS_BOUND,
            )
        return
    if args.command == "plot":
        rows = load_rows(args.input)
        rows = [
            row
            for row in rows
            if row["model"] in args.models
            and str(row["method"]) in METHODS
        ]
        print_rows(rows)
        plot_rows(
            rows,
            args.output_dir,
            experiment_label="Noisy",
            filename_prefix="noisy",
            standard_bias_bound=STANDARD_HT_BIAS_BOUND,
        )
        return
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
