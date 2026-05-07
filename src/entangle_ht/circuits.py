from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


# Demo circuit model:
# - Each system register is a single qubit initialized in |0>.
# - We choose a simple effective 1-qubit unitary U so that <0|U|0> = alpha,
#   where alpha = rho * exp(i * phi_true) is the overlap whose phase we want
#   to estimate and rho = |alpha| is the contrast.
# - Writing
#       U = [[alpha, -beta], [beta, conj(alpha)]],
#   with beta = sqrt(1 - |alpha|^2), makes U unitary and keeps the measured
#   overlap exactly equal to alpha on the |0> input state.
# - For amplification m, the circuit uses m ancilla qubits prepared in a GHZ
#   state and m system qubits prepared in |0>^{\otimes m}. Each ancilla then
#   controls one copy of the effective U on one system qubit.

def build_simulator() -> Any:
    from qiskit_aer import AerSimulator
    return AerSimulator()


def build_effective_u(alpha: complex) -> np.ndarray:
    alpha_abs = abs(alpha)
    if alpha_abs > 1.0 + 1e-12:
        raise ValueError(f"|alpha| must be <= 1, got {alpha_abs:.6f}")
    beta = math.sqrt(max(0.0, 1.0 - alpha_abs**2))
    return np.asarray(
        [[alpha, -beta], [beta, np.conjugate(alpha)]],
        dtype=complex,
    )


def effective_u_parameters(alpha: complex) -> tuple[float, float]:
    # Write alpha = r * exp(i * phase), so beta = sqrt(1 - r^2). Then
    #
    #   U = [[alpha, -beta], [beta, conj(alpha)]]
    #
    # can be written in Qiskit's single-qubit U form with an extra global phase as
    #   e^{i gamma} [
    #     [cos(theta/2),              -e^{i lambda} sin(theta/2)],
    #     [e^{i phi} sin(theta/2),   e^{i(phi + lambda)} cos(theta/2)]
    #   ]
    # by choosing
    #
    #   theta = 2 arccos(r),  phi = -phase,  lambda = -phase,  gamma = phase.
    #
    # In build_sine_ghz_circuit() we therefore use
    #
    #   circuit.cu(theta, -phase, -phase, phase, control, target),
    #
    # which reproduces the effective matrix exactly.
    alpha_abs = abs(alpha)
    if alpha_abs > 1.0 + 1e-12:
        raise ValueError(f"|alpha| must be <= 1, got {alpha_abs:.6f}")
    theta = 2.0 * math.acos(min(1.0, max(0.0, alpha_abs)))
    phase = math.atan2(alpha.imag, alpha.real)
    return theta, phase


def theoretical_parity_mean(rho: float, phi_true: float, theta_ref: float, m: int) -> float:
    return (rho**m) * math.sin(m * (phi_true - theta_ref))


def parity_mean_from_counts(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Counts dictionary is empty.")
    signed_total = 0
    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        signed_total += count if parity == 0 else -count
    return signed_total / total


def build_sine_ghz_circuit(m: int, theta_ref: float, alpha: complex) -> Any:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2 * m, m)
    ancillas = list(range(m))
    systems = list(range(m, 2 * m))

    circuit.h(ancillas[0])
    for qubit in ancillas[1:]:
        circuit.cx(ancillas[0], qubit)

    theta_u, phase_u = effective_u_parameters(alpha)
    for ancilla, system in zip(ancillas, systems):
        circuit.cu(theta_u, -phase_u, -phase_u, phase_u, ancilla, system)

    # On the GHZ support, these gates apply the required phase
    # -i * exp(-i * m * theta_ref) to |1...1>.
    for ancilla in ancillas:
        circuit.p(-theta_ref, ancilla)  # phase gate
    circuit.sdg(ancillas[0])  # inverse of S gate, to get the -i factor right.

    for ancilla in ancillas:
        circuit.h(ancilla)
    circuit.measure(ancillas, list(range(m)))

    return circuit


def run_round(
    m: int,
    theta_ref: float,
    shots: int,
    alpha: complex,
    simulator: Any,
    seed: int,
) -> float:
    from qiskit import transpile

    circuit = build_sine_ghz_circuit(m=m, theta_ref=theta_ref, alpha=alpha)
    compiled = transpile(
        circuit,
        simulator,
        optimization_level=1,
        seed_transpiler=seed,
    )
    result = simulator.run(compiled, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    return parity_mean_from_counts(counts)
