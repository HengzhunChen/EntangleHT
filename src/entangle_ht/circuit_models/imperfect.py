"""Circuit model for imperfect-eigenstate phase estimation."""

from __future__ import annotations

import math
from typing import Any


def imperfect_overlap(
    theta_target: float,
    eta: float,
) -> tuple[float, float, complex]:
    """Return the polar parameters of alpha=<psi|U|psi>.

    Define rho=|alpha| and theta_psi=arg(alpha), so that
    alpha=rho*exp(i*theta_psi).  The returned tuple is
    (rho, theta_psi, alpha).

    Here U=diag(1, exp(i*theta_target)) and the imperfect preparation is
    sqrt(eta)|0> + sqrt(1 - eta)|1>.
    """
    if not 0.0 <= eta < 0.5:
        raise ValueError(f"eta must lie in [0, 1/2), got {eta!r}")

    alpha = eta + (1.0 - eta) * complex(
        math.cos(theta_target),
        math.sin(theta_target),
    )
    rho = abs(alpha)
    theta_psi = math.atan2(alpha.imag, alpha.real)
    return rho, theta_psi, alpha


def imperfect_contrast_lower_bound(eta: float) -> float:
    if not 0.0 <= eta < 0.5:
        raise ValueError(f"eta must lie in [0, 1/2), got {eta!r}")
    return 1.0 - 2.0 * eta


def build_imperfect_phase_ghz_circuit(
    m: int,
    theta_ref: float,
    theta_target: float,
    eta: float,
) -> Any:
    """Build the sine-quadrature imperfect-state GHZ test.

    This uses the same U=diag(1, exp(i*theta_target)) as the exact model,
    but prepares sqrt(eta)|0> + sqrt(1 - eta)|1> instead of |1>.
    """
    if m < 1:
        raise ValueError(f"m must be at least 1, got {m!r}")
    if not 0.0 <= eta < 0.5:
        raise ValueError(f"eta must lie in [0, 1/2), got {eta!r}")

    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2 * m, m)
    ancillas = list(range(m))
    systems = list(range(m, 2 * m))

    circuit.h(ancillas[0])
    for qubit in ancillas[1:]:
        circuit.cx(ancillas[0], qubit)

    # Prepare m copies of the imperfect state
    prep_angle = 2.0 * math.acos(math.sqrt(eta))
    for system in systems:
        circuit.ry(prep_angle, system)

    for ancilla, system in zip(ancillas, systems):
        circuit.cp(theta_target, ancilla, system)

    for ancilla in ancillas:
        circuit.p(-theta_ref, ancilla)
    circuit.sdg(ancillas[0])

    for ancilla in ancillas:
        circuit.h(ancilla)
    circuit.measure(ancillas, list(range(m)))

    return circuit


def build_compiled_imperfect_phase_ghz_circuit(
    m: int,
    theta_ref: float,
    theta_target: float,
    eta: float,
    simulator: Any,
    seed: int,
) -> Any:
    from qiskit import transpile

    circuit = build_imperfect_phase_ghz_circuit(
        m=m,
        theta_ref=theta_ref,
        theta_target=theta_target,
        eta=eta,
    )

    return transpile(
        circuit,
        simulator,
        optimization_level=1,
        seed_transpiler=seed,
    )


def build_imperfect_cosine_circuit(theta_target: float, eta: float) -> Any:
    """Build the reference-free cosine circuit for the standard baseline."""
    if not 0.0 <= eta < 0.5:
        raise ValueError(f"eta must lie in [0, 1/2), got {eta!r}")

    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.ry(2.0 * math.acos(math.sqrt(eta)), 1)
    circuit.cp(theta_target, 0, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def build_imperfect_sine_circuit(theta_target: float, eta: float) -> Any:
    """Build the reference-free sine circuit for the standard baseline."""
    if not 0.0 <= eta < 0.5:
        raise ValueError(f"eta must lie in [0, 1/2), got {eta!r}")

    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.ry(2.0 * math.acos(math.sqrt(eta)), 1)
    circuit.cp(theta_target, 0, 1)
    circuit.sdg(0)
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit
