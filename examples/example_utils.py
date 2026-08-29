"""Shared presentation helpers for the runnable examples."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping, Sequence


METHOD_LABELS = {
    "standard": "SHT",
    "single_round_eht": "EHT",
    "fixed_m_eht": "EHT",
    "adaptive_eht": "AEHT",
}
METHOD_COLORS = {
    "standard": "#EE6677",
    "single_round_eht": "#228833",
    "fixed_m_eht": "#228833",
    "adaptive_eht": "#4477AA",
}
REFERENCE_COLOR = "#666666"
METHOD_STYLES = {
    "standard": {
        "color": METHOD_COLORS["standard"],
        "linestyle": "-",
        "marker": "D",
    },
    "single_round_eht": {
        "color": METHOD_COLORS["single_round_eht"],
        "linestyle": "-",
        "marker": "^",
    },
    "adaptive_eht": {
        "color": METHOD_COLORS["adaptive_eht"],
        "linestyle": "-",
        "marker": "o",
    },
    "fixed_m_eht": {
        "color": METHOD_COLORS["fixed_m_eht"],
        "linestyle": "-",
        "marker": "^",
    },
}
ANNOTATION_FONT_SIZE = 12


def format_count(value: float) -> str:
    return f"{int(round(value)):,}"


def format_grid_value(value: float) -> str:
    return f"{value:.3g}"


def configure_matplotlib_cache() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path(".cache/matplotlib")
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir.resolve())

    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "mathtext.fontset": "stix",
        }
    )


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
