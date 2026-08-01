#!/usr/bin/env python3
"""Noisy circuit-sampled error decay for exact and imperfect states.

Commands and options:
  run                              Simulate both state models and append CSV rows.
    --epsilons VALUE [...]         Set the target-accuracy grid.
    --repetitions N                Set the number of independent sampled estimates.
    --output PATH                  Set the output CSV file.
    --force                        Replace the output CSV before running.
    --one-qubit-error-rate RATE    Set the one-qubit depolarizing error rate.
    --two-qubit-error-rate RATE    Set the two-qubit depolarizing error rate.
    --readout-error-rate RATE      Set the symmetric readout error rate.
    --plot                         Plot the results produced by this run.
    --output-dir PATH              Set the figure directory used by --plot.
  plot                             Print selected rows and generate decay figures.
    --input PATH                   Read results from this CSV file.
    --output-dir PATH              Set the figure directory.
    --comparison-curves CURVE      Select one or more curves from entangled,
                                   one-quadrature, and two-quadrature.

The plots use the same curve names, ordering, colors, and styles as the
noiseless experiment. With no curve option, every applicable method is printed
and plotted.

Examples:
  python noisy_error_decay.py run --force --plot
  python noisy_error_decay.py run --epsilons 0.04 0.02 --repetitions 3
  python noisy_error_decay.py plot --comparison-curves entangled one-quadrature
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from error_decay import (
    COMPARISON_CURVE_CHOICES,
    CURVE_BY_METHOD,
    EPSILON_GRID,
    REPETITIONS,
    append_rows,
    load_rows,
    parse_float_list,
    plot_rows,
    print_rows,
    simulate_exact_epsilon,
    simulate_imperfect_epsilon,
)
from entangle_ht.simulation import require_qiskit_aer


# *****************************************************************************
# Experiment settings
# *****************************************************************************

NOISY_OUTPUT_DIR = Path("outputs/noisy_error_decay")
NOISY_CSV_PATH = NOISY_OUTPUT_DIR / "noisy_error_decay.csv"
ONE_QUBIT_ERROR_RATE = 1e-4
TWO_QUBIT_ERROR_RATE = 1e-3
READOUT_ERROR_RATE = 1e-2


# *****************************************************************************
# Noisy simulation
# *****************************************************************************

def run_noisy_simulations(
    *,
    epsilons: Sequence[float],
    repetitions: int,
    output_path: Path,
    force: bool,
    one_qubit_error_rate: float,
    two_qubit_error_rate: float,
    readout_error_rate: float,
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
        rows = simulate_exact_epsilon(epsilon, repetitions, simulator_kwargs=noise)
        rows += simulate_imperfect_epsilon(epsilon, repetitions, simulator_kwargs=noise)
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
        description="Noisy sampled error-decay experiments for entangled HT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="simulate both state models with circuit noise and append CSV rows",
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
        "--output-dir",
        type=Path,
        default=NOISY_OUTPUT_DIR,
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
            rows = run_noisy_simulations(
                epsilons=parse_float_list(args.epsilons),
                repetitions=args.repetitions,
                output_path=args.output,
                force=args.force,
                one_qubit_error_rate=args.one_qubit_error_rate,
                two_qubit_error_rate=args.two_qubit_error_rate,
                readout_error_rate=args.readout_error_rate,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.plot:
            plot_rows(
                rows,
                args.output_dir,
                experiment_label="Noisy",
                filename_prefix="noisy",
            )
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
            experiment_label="Noisy",
            filename_prefix="noisy",
            comparison_curves=args.comparison_curves,
        )
        return
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
