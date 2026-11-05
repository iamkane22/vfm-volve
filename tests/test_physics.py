from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import TRUE_CHOKE_EXPONENT, TRUE_DP_EXPONENT
from vfm.features import add_features
from vfm.models.physics import TEXTBOOK, ChokeExponents, ChokeMeter, fit_within_blocks


def _liquid(daily: pd.DataFrame) -> np.ndarray:
    return (daily["oil_rate"] + daily["water_rate"]).to_numpy()


def test_within_estimator_recovers_exponents_despite_depletion(daily: pd.DataFrame) -> None:
    well = daily[daily["well"] == "W-1"]
    block = ((well["date"] - well["date"].min()).dt.days // 30).to_numpy()
    fitted = fit_within_blocks(
        well["choke_pct"].to_numpy(), well["dp_choke"].to_numpy(), _liquid(well), block
    )
    assert fitted.choke == pytest.approx(TRUE_CHOKE_EXPONENT, abs=0.1)
    assert fitted.dp == pytest.approx(TRUE_DP_EXPONENT, abs=0.1)


def test_naive_regression_is_biased_by_depletion(daily: pd.DataFrame) -> None:
    """The reason the within estimator exists: one block spanning the whole history."""
    well = daily[daily["well"] == "W-1"]
    naive = fit_within_blocks(
        well["choke_pct"].to_numpy(),
        well["dp_choke"].to_numpy(),
        _liquid(well),
        np.zeros(len(well)),
    )
    assert naive.choke < TRUE_CHOKE_EXPONENT - 0.3


def test_unphysical_estimates_fall_back() -> None:
    assert not ChokeExponents(choke=-0.2, dp=0.5).is_physical
    assert TEXTBOOK.is_physical


def test_ratio_meter_tracks_choke_changes(daily: pd.DataFrame) -> None:
    frame = add_features(daily)
    train = frame[frame["date"] < "2016-01-01"]
    meter = ChokeMeter().fit(train)
    last = train.groupby("well", observed=True).tail(1)
    opened = last.assign(choke_pct=last["choke_pct"] * 1.5)
    ratio = (
        meter.predict(opened)["oil_rate"].to_numpy() / meter.predict(last)["oil_rate"].to_numpy()
    )
    assert np.all(ratio > 1.2)


def test_as_of_predictions_do_not_look_ahead(daily: pd.DataFrame) -> None:
    frame = add_features(daily)
    meter = ChokeMeter().fit(frame)
    before = meter.as_of_predictions(frame)

    tampered = frame.copy()
    late = tampered["date"] >= "2015-12-01"
    tampered.loc[late, ["oil_rate", "gas_rate", "water_rate"]] *= 10.0
    after = meter.as_of_predictions(tampered)

    # Rows whose 30-day block starts before the tampered period only see earlier data.
    early = frame["date"] < "2015-11-01"
    pd.testing.assert_frame_equal(before[early], after[early])
