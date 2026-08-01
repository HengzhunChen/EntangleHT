"""Circuit model for exact-eigenstate phase estimation."""

from __future__ import annotations

from typing import Any


def build_exact_phase_ghz_circuit(
    m: int,
    theta_ref: float,
    theta_true: float,
) -> Any:
    """Build the GHZ Hadamard test for U=diag(1, exp(i theta_true)), |phi>=|1>."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2 * m, m)
    ancillas = list(range(m))
    systems = list(range(m, 2 * m))

    # Prepare the ancilla GHZ state
    circuit.h(ancillas[0])
    for qubit in ancillas[1:]:
        circuit.cx(ancillas[0], qubit)

    # Prepare m copies the exact eigenstate
    for system in systems:
        circuit.x(system)

    # Apply the controlled phase gate
    for ancilla, system in zip(ancillas, systems):
        circuit.cp(theta_true, ancilla, system)

    # Subtract the reference phase
    for ancilla in ancillas:
        circuit.p(-theta_ref, ancilla)
    # Select the sine quadrature
    circuit.sdg(ancillas[0])

    # Rotate into the measurement basis
    for ancilla in ancillas:
        circuit.h(ancilla)
    circuit.measure(ancillas, list(range(m)))

    return circuit


def build_compiled_exact_phase_ghz_circuit(
    m: int,
    theta_ref: float,
    theta_true: float,
    simulator: Any,
    seed: int,
) -> Any:
    from qiskit import transpile

    circuit = build_exact_phase_ghz_circuit(
        m=m,
        theta_ref=theta_ref,
        theta_true=theta_true,
    )

    return transpile(
        circuit,
        simulator,
        optimization_level=1,
        seed_transpiler=seed,
    )
