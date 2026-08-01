from __future__ import annotations

import os
from pathlib import Path


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
