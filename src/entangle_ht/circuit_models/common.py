from __future__ import annotations

from typing import Any, Dict


def parity_mean_from_counts(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Counts dictionary is empty.")
    signed_total = 0
    for bitstring, count in counts.items():
        parity = bitstring.count("1") % 2
        signed_total += count if parity == 0 else -count
    return signed_total / total


def run_round(
    compiled_circuit: Any,
    shots: int,
    simulator: Any,
    seed: int,
) -> float:
    result = simulator.run(
        compiled_circuit,
        shots=shots,
        seed_simulator=seed,
    ).result()

    counts = result.get_counts()
    return parity_mean_from_counts(counts)

