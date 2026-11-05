"""Time-aware evaluation protocols.

Two questions are asked of every meter:

* **Walk-forward** (primary): the meter is recalibrated every ``horizon_days`` with all
  data up to that day and then used blind for the next ``horizon_days``. This mirrors a
  field VFM that is re-tuned after each monthly well test.
* **Hold-out** (stress test): the meter is calibrated once and used for the last part of
  every well's life, typically more than a year, without any recalibration.

A random split is never used: neighbouring days are almost identical, so it would let
any model look good by interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    cutoff: pd.Timestamp
    train: np.ndarray
    test: np.ndarray


def holdout(df: pd.DataFrame, test_fraction: float = 0.2) -> list[Fold]:
    """Single fold holding out the last ``test_fraction`` of each well's producing days."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    by_well = df.groupby("well", observed=True)["date"]
    position = (by_well.rank(method="first") - 1) / by_well.transform("size")
    test = (position >= 1.0 - test_fraction).to_numpy()
    return [Fold(cutoff=df.loc[test, "date"].min(), train=~test, test=test)]


def walk_forward(
    df: pd.DataFrame,
    horizon_days: int = 30,
    n_folds: int = 24,
    min_history_days: int = 30,
) -> list[Fold]:
    """Expanding-window folds covering the last ``n_folds * horizon_days`` of the data.

    Test rows of wells with fewer than ``min_history_days`` of calibration data are
    dropped: no meter can be tuned for a well that has barely produced.
    """
    if horizon_days <= 0 or n_folds <= 0:
        raise ValueError("horizon_days and n_folds must be positive")
    dates = df["date"]
    wells = df["well"].astype(str)
    horizon = pd.Timedelta(days=horizon_days)
    first_cutoff = dates.max() + pd.Timedelta(days=1) - n_folds * horizon

    folds = []
    for k in range(n_folds):
        cutoff = first_cutoff + k * horizon
        train = (dates < cutoff).to_numpy()
        history = wells[train].value_counts()
        calibrated = wells.map(history).fillna(0).to_numpy() >= min_history_days
        test = ((dates >= cutoff) & (dates < cutoff + horizon)).to_numpy() & calibrated
        if test.any():
            folds.append(Fold(cutoff=cutoff, train=train, test=test))
    return folds
