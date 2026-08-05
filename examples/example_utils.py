"""Shared presentation helpers for the runnable examples."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping, Sequence


METHOD_LABELS = {
    "standard": "Standard HT",
    "eht": "EHT",
    "fixed_m": r"Fixed-$m$ EHT",
    "entangled": "Iterative EHT",
}
METHOD_STYLES = {
    "standard": {
        "color": "#6A3D9A",
        "linestyle": "-.",
        "marker": "D",
    },
    "entangled": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
    },
    "fixed_m": {
        "color": "#E69F00",
        "linestyle": "--",
        "marker": "^",
    },
}


def format_count(value: float) -> str:
    return f"{int(round(value)):,}"


def format_grid_value(value: float) -> str:
    return f"{value:.3g}"


def configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(".cache/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())


def save_figure(fig, output_path: Path, *, dpi: int = 180) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def save_csv_rows(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Path:
    """Overwrite a CSV file with a nonempty sequence of result rows."""
    if not rows:
        raise ValueError("rows must not be empty")

    fieldnames = list(rows[0])
    expected_fields = set(fieldnames)
    for index, row in enumerate(rows):
        if set(row) != expected_fields:
            raise ValueError(
                f"row {index} has fields that do not match the CSV schema"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return output_path
