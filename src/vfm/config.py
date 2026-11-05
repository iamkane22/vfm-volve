"""Paths and project-wide constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Phase = Literal["oil", "gas", "water"]
PHASES: tuple[Phase, ...] = ("oil", "gas", "water")

#: Target column for each phase, in standard m3 per producing day.
TARGETS: dict[Phase, str] = {"oil": "oil_rate", "gas": "gas_rate", "water": "water_rate"}


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. The data directory is configurable via ``VFM_DATA_DIR``."""

    data_dir: Path
    reports_dir: Path

    @property
    def raw_xlsx(self) -> Path:
        return self.data_dir / "raw" / "volve_production.xlsx"

    @property
    def daily_parquet(self) -> Path:
        return self.data_dir / "processed" / "daily.parquet"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


def default_paths() -> Paths:
    return Paths(
        data_dir=Path(os.environ.get("VFM_DATA_DIR", "data")),
        reports_dir=Path(os.environ.get("VFM_REPORTS_DIR", "reports")),
    )
