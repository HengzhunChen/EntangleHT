from __future__ import annotations

from typing import Any


def build_simulator(method: str = "statevector") -> Any:
    from qiskit_aer import AerSimulator

    return AerSimulator(
        method=method,
        max_parallel_threads=0,
    )


def build_noisy_simulator(
    method: str = "automatic",
    one_qubit_error_rate: float = 1e-4,
    two_qubit_error_rate: float = 1e-3,
    readout_error_rate: float = 1e-2,
) -> Any:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    basis_gates = ("id", "rz", "sx", "x", "cx")

    for name, value in (
        ("one_qubit_error_rate", one_qubit_error_rate),
        ("two_qubit_error_rate", two_qubit_error_rate),
        ("readout_error_rate", readout_error_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value!r}")

    noise_model = NoiseModel()

    one_qubit_noise_gates = ("id", "rz", "sx", "x")
    if one_qubit_error_rate > 0.0 and one_qubit_noise_gates:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(one_qubit_error_rate, 1),
            one_qubit_noise_gates,
        )

    two_qubit_noise_gates = ("cx",)
    if two_qubit_error_rate > 0.0 and two_qubit_noise_gates:
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(two_qubit_error_rate, 2),
            two_qubit_noise_gates,
        )

    if readout_error_rate > 0.0:
        readout_error = ReadoutError(
            [
                [1.0 - readout_error_rate, readout_error_rate],
                [readout_error_rate, 1.0 - readout_error_rate],
            ]
        )
        noise_model.add_all_qubit_readout_error(readout_error)

    return AerSimulator(
        method=method,
        noise_model=noise_model,
        basis_gates=list(basis_gates),
        max_parallel_threads=0,
    )

