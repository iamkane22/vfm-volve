from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vfm.config import TARGETS
from vfm.evaluation.benchmark import run_benchmark
from vfm.features import add_features
from vfm.models import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_meter_returns_aligned_non_negative_rates(name: str, daily: pd.DataFrame) -> None:
    frame = add_features(daily)
    train, test = frame.iloc[::2], frame.iloc[1::2]
    predicted = build_model(name).fit(train).predict(test)
    assert list(predicted.columns) == list(TARGETS.values())
    assert predicted.index.equals(test.index)
    assert np.isfinite(predicted.to_numpy()).all()
    assert (predicted.to_numpy() >= 0).all()


def test_meters_handle_failed_gauges(daily: pd.DataFrame) -> None:
    frame = add_features(daily)
    test = frame.tail(20).copy()
    test[["choke_pct", "dp_choke", "p_downhole"]] = np.nan
    for name in ("physics", "hybrid"):
        predicted = build_model(name).fit(frame.head(-20)).predict(test)
        assert np.isfinite(predicted.to_numpy()).all()


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("crystal_ball")


def test_physics_beats_sensor_blind_baselines_walk_forward(daily: pd.DataFrame) -> None:
    result = run_benchmark(daily, models=("well_mean", "last_test", "physics"))
    oil = result.summary().loc["oil", "wmape"]
    assert oil["physics"] < oil["last_test"] < oil["well_mean"]


def test_hybrid_response_to_the_choke_is_monotone(daily: pd.DataFrame) -> None:
    """Opening the choke or raising its pressure drop must never lower any phase rate."""
    frame = add_features(daily)
    meter = build_model("hybrid").fit(frame)
    for _, rows in frame.groupby("well", observed=True):
        last = rows.tail(1)
        for column, grid in (
            ("choke_pct", np.linspace(5, 100, 40)),
            ("dp_choke", np.linspace(1, 30, 40)),
        ):
            sweep = pd.concat([last] * len(grid), ignore_index=True).assign(**{column: grid})
            rates = meter.predict(sweep).to_numpy()
            assert np.all(np.diff(rates, axis=0) >= -1e-9)
