#!/usr/bin/env python3
"""
Run noisy Qiskit simulations one accuracy at a time.

The noisy runs can be expensive, so this script writes one CSV result file per
epsilon. Use ``run`` or ``run-all`` to produce results, then ``plot`` to read 
the saved files without rerunning simulation.

Common usage:

    # Run one target accuracy and save one CSV result file.
    python examples/noise_simulation.py run --epsilon 0.01

    # Rerun even if the matching CSV result file already exists.
    python examples/noise_simulation.py run --epsilon 0.01 --force

    # Run the default small epsilon grid.
    python examples/noise_simulation.py run-all

    # Run a custom epsilon grid.
    python examples/noise_simulation.py run-all --epsilons 0.04 0.02 0.01

    # Plot all saved CSV result files in the output directory.
    python examples/noise_simulation.py plot

Useful options:

    --output-dir outputs/noise_simulation
    --one-qubit-error-rate 1e-4
    --two-qubit-error-rate 1e-3
    --readout-error-rate 1e-2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from amplification_effect import (
    configure_matplotlib_cache,
    num_shot_standard_hadamard_test,
    run_standard_hadamard,
    save_figure,
)
from entangle_ht.circuits import build_noisy_simulator
from entangle_ht.schedule import design_algorithm_parameters, plan_trial, run_trial
from entangle_ht.utilities import DemoConfig


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Keep the default grid small. For long cluster runs, pass one epsilon at a time:
#   python examples/noise_simulation.py run --epsilon 0.01
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
    m_hw=9,  # for memory and time constraints of noisy simulation
    base_seed=20260416,
)
EPSILON_GRID = (0.04, 0.02, 0.01)
OUTPUT_DIR = Path("outputs/noise_simulation")

ONE_QUBIT_ERROR_RATE = 1e-4
TWO_QUBIT_ERROR_RATE = 1e-3
READOUT_ERROR_RATE = 1e-2

FIELDNAMES = [
    "epsilon",
    "one_qubit_error_rate",
    "two_qubit_error_rate",
    "readout_error_rate",
    "seed",
    "eiht_shots",
    "stdht_shots",
    "max_m",
    "eiht_rounds",
    "eiht_estimate",
    "stdht_estimate",
    "eiht_error",
    "stdht_error",
    "stdht_bias_floor",
    "eiht_seconds",
    "stdht_seconds",
    "total_seconds",
]


# ---------------------------------------------------------------------------
# Result file helpers
# ---------------------------------------------------------------------------

def epsilon_tag(epsilon: float) -> str:
    return f"{epsilon:.12g}".replace("-", "m").replace(".", "p")


def value_tag(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def result_path(
    output_dir: Path,
    *,
    epsilon: float,
    one_qubit_error_rate: float,
    two_qubit_error_rate: float,
    readout_error_rate: float,
) -> Path:
    filename = (
        f"eps_{epsilon_tag(epsilon)}"
        f"_p1_{value_tag(one_qubit_error_rate)}"
        f"_p2_{value_tag(two_qubit_error_rate)}"
        f"_ro_{value_tag(readout_error_rate)}.csv"
    )
    return output_dir / filename


def write_result(path: Path, row: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def read_result(path: Path) -> Dict[str, float]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    return {key: float(value) for key, value in row.items()}


def result_files(output_dir: Path) -> List[Path]:
    return sorted(output_dir.glob("eps_*.csv"))


# ---------------------------------------------------------------------------
# Per-epsilon noisy simulation
# ---------------------------------------------------------------------------

def simulate_epsilon(
    epsilon: float,
    *,
    output_dir: Path,
    one_qubit_error_rate: float,
    two_qubit_error_rate: float,
    readout_error_rate: float,
    force: bool,
) -> Dict[str, float]:
    path = result_path(
        output_dir,
        epsilon=epsilon,
        one_qubit_error_rate=one_qubit_error_rate,
        two_qubit_error_rate=two_qubit_error_rate,
        readout_error_rate=readout_error_rate,
    )
    if path.exists() and not force:
        # Long runs are resumable: completed epsilon/noise combinations are reused.
        print(f"[skip] epsilon={epsilon:.6g} existing={path}", flush=True)
        return read_result(path)

    config = replace(BASE_CONFIG, epsilon=epsilon)
    seed = config.base_seed + int(round(epsilon * 1_000_000_000))

    algorithm = design_algorithm_parameters(config)
    plan = plan_trial(config=config, algorithm=algorithm, label="EIHT")
    standard_shots = num_shot_standard_hadamard_test(
        epsilon=config.epsilon,
        p_success=config.p_total,
    )
    max_m = max(round_plan.amplification for round_plan in plan.rounds)

    # The same noisy backend is used for EIHT and StdHT so their runtime and
    # accuracy are compared under identical hardware-noise parameters.
    simulator = build_noisy_simulator(
        method="statevector",
        one_qubit_error_rate=one_qubit_error_rate,
        two_qubit_error_rate=two_qubit_error_rate,
        readout_error_rate=readout_error_rate,
    )

    print(
        f"[run] epsilon={epsilon:.6g} "
        f"EIHT_shots={plan.total_shots} StdHT_shots={standard_shots} "
        f"max_m={max_m} rounds={algorithm.num_rounds}",
        flush=True,
    )

    total_start = time.perf_counter()

    eiht_start = time.perf_counter()
    eiht_trial = run_trial(
        config=config,
        algorithm=algorithm,
        simulator=simulator,
        seed=seed,
        label="EIHT",
        plan=plan,
    )
    eiht_seconds = time.perf_counter() - eiht_start

    stdht_start = time.perf_counter()
    stdht = run_standard_hadamard(
        config=config,
        simulator=simulator,
        seed=seed + 20_000,
    )
    stdht_seconds = time.perf_counter() - stdht_start
    total_seconds = time.perf_counter() - total_start

    row = {
        "epsilon": epsilon,
        "one_qubit_error_rate": one_qubit_error_rate,
        "two_qubit_error_rate": two_qubit_error_rate,
        "readout_error_rate": readout_error_rate,
        "seed": float(seed),
        "eiht_shots": float(plan.total_shots),
        "stdht_shots": float(standard_shots),
        "max_m": float(max_m),
        "eiht_rounds": float(algorithm.num_rounds),
        "eiht_estimate": eiht_trial.final_estimate,
        "stdht_estimate": stdht["estimate"],
        "eiht_error": abs(eiht_trial.final_error),
        "stdht_error": stdht["actual_error"],
        "stdht_bias_floor": stdht["bias_floor"],
        "eiht_seconds": eiht_seconds,
        "stdht_seconds": stdht_seconds,
        "total_seconds": total_seconds,
    }
    # Save immediately after each epsilon so interrupted batch jobs keep results.
    write_result(path, row)

    print(
        "epsilon\tEIHT error\tStdHT error\tEIHT seconds\tStdHT seconds\ttotal seconds",
        flush=True,
    )
    print(
        f"{epsilon:.6g}\t"
        f"{row['eiht_error']:.8f}\t"
        f"{row['stdht_error']:.8f}\t"
        f"{eiht_seconds:.3f}\t"
        f"{stdht_seconds:.3f}\t"
        f"{total_seconds:.3f}",
        flush=True,
    )
    print(f"[saved] {path}", flush=True)

    return row


# ---------------------------------------------------------------------------
# Result loading and printing
# ---------------------------------------------------------------------------

def load_results(output_dir: Path) -> List[Dict[str, float]]:
    files = result_files(output_dir)
    rows = [read_result(path) for path in files]
    rows.sort(key=lambda row: row["epsilon"], reverse=True)
    return rows


def print_results_table(rows: Sequence[Dict[str, float]]) -> None:
    print(
        "epsilon\tEIHT shots\tStdHT shots\tmax_m\tEIHT rounds\t"
        "EIHT error\tStdHT error\tEIHT seconds\tStdHT seconds\ttotal seconds"
    )
    for row in rows:
        print(
            f"{row['epsilon']:.6g}\t"
            f"{int(row['eiht_shots'])}\t"
            f"{int(row['stdht_shots'])}\t"
            f"{int(row['max_m'])}\t"
            f"{int(row['eiht_rounds'])}\t"
            f"{row['eiht_error']:.8f}\t"
            f"{row['stdht_error']:.8f}\t"
            f"{row['eiht_seconds']:.3f}\t"
            f"{row['stdht_seconds']:.3f}\t"
            f"{row['total_seconds']:.3f}"
        )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_results(rows: Sequence[Dict[str, float]], output_dir: Path) -> None:
    if not rows:
        raise ValueError(f"No result files found in {output_dir}")

    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    epsilon_values = [row["epsilon"] for row in rows]
    eiht_errors = [max(row["eiht_error"], 1e-16) for row in rows]
    stdht_errors = [max(row["stdht_error"], 1e-16) for row in rows]
    eiht_seconds = [max(row["eiht_seconds"], 1e-16) for row in rows]
    stdht_seconds = [max(row["stdht_seconds"], 1e-16) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epsilon_values,
        eiht_errors,
        marker="o",
        linewidth=2.2,
        color="tab:blue",
        label="Iterative entangled HT",
    )
    ax.plot(
        epsilon_values,
        stdht_errors,
        marker="D",
        linestyle="-.",
        linewidth=2.2,
        color="tab:orange",
        label="Standard HT",
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
    ax.set_title("Noisy simulation error")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "noise_simulation_errors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epsilon_values,
        eiht_seconds,
        marker="o",
        linewidth=2.2,
        color="tab:blue",
        label="Iterative entangled HT",
    )
    ax.plot(
        epsilon_values,
        stdht_seconds,
        marker="D",
        linestyle="-.",
        linewidth=2.2,
        color="tab:orange",
        label="Standard HT",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(epsilon_values), min(epsilon_values))
    ax.set_xlabel(r"Target accuracy")
    ax.set_ylabel("Simulation seconds")
    ax.set_title("Noisy simulation timing")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "noise_simulation_timing.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_epsilon_list(values: Iterable[str]) -> List[float]:
    return [float(value) for value in values]


def add_common_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--one-qubit-error-rate", type=float, default=ONE_QUBIT_ERROR_RATE)
    parser.add_argument("--two-qubit-error-rate", type=float, default=TWO_QUBIT_ERROR_RATE)
    parser.add_argument("--readout-error-rate", type=float, default=READOUT_ERROR_RATE)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="simulate one epsilon")
    run_parser.add_argument("--epsilon", type=float, required=True)
    add_common_run_options(run_parser)

    run_all_parser = subparsers.add_parser("run-all", help="simulate the default grid")
    run_all_parser.add_argument(
        "--epsilons",
        nargs="+",
        default=[str(epsilon) for epsilon in EPSILON_GRID],
        help="epsilon values to simulate",
    )
    add_common_run_options(run_all_parser)

    plot_parser = subparsers.add_parser("plot", help="plot saved result files")
    plot_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        simulate_epsilon(
            args.epsilon,
            output_dir=args.output_dir,
            one_qubit_error_rate=args.one_qubit_error_rate,
            two_qubit_error_rate=args.two_qubit_error_rate,
            readout_error_rate=args.readout_error_rate,
            force=args.force,
        )
        return

    if args.command == "run-all":
        rows = []
        for epsilon in parse_epsilon_list(args.epsilons):
            rows.append(
                simulate_epsilon(
                    epsilon,
                    output_dir=args.output_dir,
                    one_qubit_error_rate=args.one_qubit_error_rate,
                    two_qubit_error_rate=args.two_qubit_error_rate,
                    readout_error_rate=args.readout_error_rate,
                    force=args.force,
                )
            )
        print()
        print_results_table(rows)
        return

    if args.command == "plot":
        rows = load_results(args.output_dir)
        print_results_table(rows)
        plot_results(rows, args.output_dir)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
