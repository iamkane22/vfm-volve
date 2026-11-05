from __future__ import annotations

import pandas as pd
import pytest

from vfm.evaluation.splits import holdout, walk_forward


def test_holdout_takes_the_tail_of_every_well(daily: pd.DataFrame) -> None:
    (fold,) = holdout(daily, test_fraction=0.25)
    for _, rows in daily.groupby("well", observed=True):
        test_dates = rows.loc[fold.test[rows.index], "date"]
        train_dates = rows.loc[fold.train[rows.index], "date"]
        assert len(test_dates) == pytest.approx(0.25 * len(rows), abs=1)
        assert train_dates.max() < test_dates.min()


def test_walk_forward_never_trains_on_the_future(daily: pd.DataFrame) -> None:
    folds = walk_forward(daily, horizon_days=30, n_folds=6)
    assert len(folds) == 6
    for fold in folds:
        assert daily.loc[fold.train, "date"].max() < fold.cutoff
        test_dates = daily.loc[fold.test, "date"]
        assert test_dates.min() >= fold.cutoff
        assert (test_dates - fold.cutoff).dt.days.max() < 30


def test_walk_forward_skips_wells_without_calibration_history(daily: pd.DataFrame) -> None:
    late_start = daily[(daily["well"] == "W-2") & (daily["date"] < "2015-12-15")].index
    data = daily.drop(late_start).reset_index(drop=True)
    folds = walk_forward(data, horizon_days=30, n_folds=3, min_history_days=30)
    first = folds[0]
    tested_wells = set(data.loc[first.test, "well"].astype(str))
    assert "W-2" not in tested_wells


def test_rejects_invalid_arguments(daily: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        holdout(daily, test_fraction=1.0)
    with pytest.raises(ValueError):
        walk_forward(daily, horizon_days=0)
