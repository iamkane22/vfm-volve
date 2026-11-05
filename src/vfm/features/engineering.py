"""Features available at prediction time, i.e. without looking at measured rates."""

from __future__ import annotations

import pandas as pd

from vfm.data.schema import SENSOR_COLUMNS

#: Numeric model inputs. ``well`` is passed separately as a categorical.
FEATURES: tuple[str, ...] = (
    *SENSOR_COLUMNS,
    "days_on_production",
    "wellhead_to_downhole",
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived inputs to a cleaned daily frame (returns a copy).

    ``days_on_production`` counts calendar days since the first producing day of each
    well. It captures reservoir depletion and is known in advance, so it does not leak
    the target.
    """
    out = df.copy()
    first_day = out.groupby("well", observed=True)["date"].transform("min")
    out["days_on_production"] = (out["date"] - first_day).dt.days.astype(float)
    out["wellhead_to_downhole"] = out["p_wellhead"] / out["p_downhole"]
    return out
