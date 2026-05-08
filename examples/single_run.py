#!/usr/bin/env python3
"""
Run one adaptive entangle_ht trial on the Qiskit simulator.

The constants below intentionally use a loose target accuracy so the first
end-to-end circuit run is quick. Edit them later for your application.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from entangle_ht.circuits import build_simulator
from entangle_ht.schedule import (
    design_algorithm_parameters,
    print_run_summary,
    run_trial,
)
from entangle_ht.utilities import (
    AlgorithmParameters,
    DemoConfig,
    TrialResult,
    phase_error,
)


RHO = 0.99
RHO0 = 0.95
PHI_TRUE = 0.35
THETA_0 = 0.20
DELTA_0 = 0.20
TARGET_RMSE = 0.001
P_TOTAL = 0.95
GAMMA = 0.85
OMEGA = 0.6
M_HW = 100
SEED = 20260415
OUTPUT_PATH = Path("outputs/single_run_schedule.png")


def build_example_config() -> DemoConfig:
    return replace(
        DemoConfig(),
        rho=RHO,
        rho0=RHO0,
        phi_true=PHI_TRUE,
        theta_0=THETA_0,
        Delta_0=DELTA_0,
        epsilon=TARGET_RMSE,
        p_total=P_TOTAL,
        gamma=GAMMA,
        omega=OMEGA,
        m_hw=M_HW,
        base_seed=SEED,
    )


def configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(".cache/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())


def plot_schedule_example(
    config: DemoConfig,
    algorithm: AlgorithmParameters,
    trial: TrialResult,
    output_path: Path,
) -> None:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rounds = trial.rounds
    x = np.arange(len(rounds))
    theta_ref = np.asarray([round_data.theta_ref for round_data in rounds], dtype=float)
    estimates = np.asarray([round_data.estimate for round_data in rounds], dtype=float)
    deltas = np.asarray([round_data.delta_bound for round_data in rounds], dtype=float)
    actual_errors = np.asarray(
        [abs(phase_error(round_data.estimate, config.phi_true)) for round_data in rounds],
        dtype=float,
    )
    actual_errors = np.maximum(actual_errors, 1e-16)
    amplifications = np.asarray([round_data.amplification for round_data in rounds], dtype=float)
    shots = np.asarray([round_data.shots for round_data in rounds], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(x, theta_ref, marker="o", label=r"$\vartheta_t$ reference")
    axes[0].plot(x, estimates, marker="s", label=r"$\hat{\phi}_t$ estimate")
    axes[0].axhline(config.phi_true, color="black", linestyle="--", label=r"$\phi_{\mathrm{true}}$")
    axes[0].set_ylabel("Phase")
    axes[0].set_title(
        "Adaptive schedule example "
        f"(gamma={algorithm.gamma:.3f}, p_total={config.p_total:.2f})"
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].step(x, amplifications, where="mid", label=r"$m_t$", linewidth=2.0)
    axes[1].set_ylabel("Amplification")
    axes[1].grid(alpha=0.3)
    shots_axis = axes[1].twinx()
    shots_axis.plot(x, shots, color="tab:red", marker="d", label=r"$N_t$")
    shots_axis.set_ylabel("Shots")
    shots_axis.set_yscale("log")
    lines_a, labels_a = axes[1].get_legend_handles_labels()
    lines_b, labels_b = shots_axis.get_legend_handles_labels()
    axes[1].legend(lines_a + lines_b, labels_a + labels_b, loc="upper left")

    axes[2].plot(
        x,
        actual_errors,
        marker="o",
        linewidth=2.0,
        label=r"$|\hat{\phi}_t-\phi_{\mathrm{true}}|$ actual error",
    )
    axes[2].plot(
        x,
        deltas,
        linestyle="--",
        color="black",
        label=rf"$\Delta_t=\Delta_0\gamma^t$ reference",
    )
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Absolute phase error")
    axes[2].set_yscale("log")
    axes[2].set_title("Decay of actual phase-estimation error")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_example() -> None:
    config = build_example_config()
    simulator = build_simulator()
    algorithm = design_algorithm_parameters(config)
    trial = run_trial(
        config=config,
        algorithm=algorithm,
        simulator=simulator,
        seed=config.base_seed,
        label="adaptive",
    )
    print_run_summary(config, trial)
    plot_schedule_example(
        config=config,
        algorithm=algorithm,
        trial=trial,
        output_path=OUTPUT_PATH,
    )
    print(f"Saved schedule plot to {OUTPUT_PATH}")


def main() -> None:
    run_example()


if __name__ == "__main__":
    main()
