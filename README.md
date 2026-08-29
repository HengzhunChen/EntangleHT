# EntangleHT

> **Status:** This research repository is under active development. The APIs,
> experiment settings, and documentation may change.

## Project overview

EntangleHT implements adaptive entanglement-assisted Hadamard testing for
quantum phase estimation. Starting from a certified coarse reference phase,
the iterative protocol refines the reference over multiple rounds and uses
progressively stronger GHZ-based phase amplification. The repository includes
experiments for both accurately prepared eigenstates and imperfect state
preparation, as can arise in quantum algorithms such as VQE.

The numerical studies compare Standard Hadamard Test (SHT),
fixed-amplification Entangled Hadamard Test (EHT), and Adaptive
Entangled Hadamard Test (AEHT). The main resource metric is the number
of packed device restarts, which accounts for the number of independent
circuit instances that can be executed in parallel on a finite-width
quantum device.

## Installation

Python 3.10 or later is recommended. From the repository root, create a virtual
environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include NumPy, Matplotlib, Qiskit, and Qiskit Aer. Qiskit Aer
is required for the circuit-sampled noiseless and noisy experiments.

## Quick start

Run the certified resource-planning experiments from the `examples` directory:

```bash
cd examples
python exact_eigenstate.py
python imperfect_eigenstate.py
```

Plot the circuit-simulation results already stored in `examples/outputs/`:

```bash
python error_decay.py plot
python noisy_error_decay.py plot --models exact imperfect
```

To regenerate the noiseless and noisy circuit samples before plotting them:

```bash
python error_decay.py run --force --plot
python noisy_error_decay.py run --models exact imperfect --force --plot
```

The full circuit simulations, especially the high-accuracy noisy cases, can be
computationally expensive. Use each command's `--help` option to select a
smaller accuracy grid or a different output location.
