"""From the raw daily allocation table to model-ready rows.

Every rule is recorded in a :class:`CleaningReport` so the effect of the cleaning can be
inspected and quoted instead of being hidden inside a notebook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import pandas as pd

from vfm.data import schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningConfig:
    #: Daily averages from partially producing days are dominated by start-up and
    #: shut-in transients, so such days are excluded.
    min_on_stream_hrs: float = 12.0
    #: Rows below this oil rate (Sm3/d) are near shut-in and carry no metering signal.
    min_oil_rate: float = 1.0


@dataclass
class CleaningReport:
    rows_in: int
    steps: list[tuple[str, int]] = field(default_factory=list)

    def record(self, rule: str, frame: pd.DataFrame) -> None:
        self.steps.append((rule, len(frame)))
        logger.debug("%-40s %6d rows", rule, len(frame))

    def to_frame(self) -> pd.DataFrame:
        remaining = [self.rows_in, *(n for _, n in self.steps)]
        return pd.DataFrame(
            {
                "rule": [rule for rule, _ in self.steps],
                "rows_remaining": remaining[1:],
                "rows_removed": [a - b for a, b in pairwise(remaining)],
            }
        )


def read_workbook(path: Path) -> pd.DataFrame:
    """Read the daily sheet and map it onto canonical column names."""
    raw = pd.read_excel(path, sheet_name=schema.SHEET_DAILY)
    missing = schema.REQUIRED_RAW_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Workbook is missing columns: {sorted(missing)}")
    return raw[list(schema.RAW_TO_CANONICAL)].rename(columns=schema.RAW_TO_CANONICAL)


def clean(
    df: pd.DataFrame, config: CleaningConfig | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Filter to producing days and turn allocated volumes into daily rates.

    Returns one row per well and day with sensor columns (missing values as NaN) and
    ``oil_rate``, ``gas_rate`` and ``water_rate`` in Sm3 per producing day.
    """
    config = config or CleaningConfig()
    report = CleaningReport(rows_in=len(df))

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    out = out[(out["flow_kind"] == "production") & (out["well_type"] == "OP")]
    report.record("oil producers only", out)

    out = out[out["on_stream_hrs"] >= config.min_on_stream_hrs]
    report.record(f"on stream >= {config.min_on_stream_hrs:g} h", out)

    volume_cols = list(schema.VOLUME_COLUMNS)
    out = out[out[volume_cols].ge(0).all(axis=1)]
    report.record("allocated volumes present and >= 0", out)

    out = out.assign(
        **{
            rate: out[vol] * 24.0 / out["on_stream_hrs"].clip(upper=24.0)
            for vol, rate in schema.VOLUME_COLUMNS.items()
        }
    )
    out = out[out["oil_rate"] >= config.min_oil_rate]
    report.record(f"oil rate >= {config.min_oil_rate:g} Sm3/d", out)

    gauges = list(schema.GAUGES_ZERO_IS_MISSING)
    out[gauges] = out[gauges].where(out[gauges] > 0)
    # The source reports dp_tubing = p_downhole - p_wellhead, but keeps a stale value when
    # the downhole gauge has failed. Re-deriving it makes those rows honestly missing.
    out["dp_tubing"] = out["p_downhole"] - out["p_wellhead"]

    columns = [
        "date",
        "well",
        "on_stream_hrs",
        *schema.SENSOR_COLUMNS,
        *schema.VOLUME_COLUMNS.values(),
    ]
    out = out[columns].sort_values(["well", "date"]).reset_index(drop=True)
    out["well"] = out["well"].astype("category")
    return out, report
