from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vfm.data.cleaning import CleaningConfig, clean


def _raw(**overrides: list[object]) -> pd.DataFrame:
    base: dict[str, list[object]] = {
        "date": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04", "2015-01-05"],
        "well": ["A", "A", "A", "A", "B"],
        "flow_kind": ["production"] * 4 + ["injection"],
        "well_type": ["OP"] * 4 + ["WI"],
        "on_stream_hrs": [24.0, 12.0, 6.0, 24.0, 24.0],
        "p_downhole": [250.0, 0.0, 250.0, 249.0, 300.0],
        "t_downhole": [100.0, 100.0, 100.0, 100.0, 50.0],
        "dp_tubing": [200.0, 33.0, 200.0, 199.0, 0.0],
        "p_annulus": [0.0, 10.0, 10.0, 10.0, 0.0],
        "choke_pct": [50.0, 50.0, 50.0, 0.0, 100.0],
        "p_wellhead": [50.0, 50.0, 50.0, 50.0, 200.0],
        "t_wellhead": [80.0, 80.0, 80.0, 80.0, 20.0],
        "dp_choke": [10.0, 10.0, 10.0, 10.0, 0.0],
        "oil_vol": [1000.0, 500.0, 250.0, 900.0, 0.0],
        "gas_vol": [1.5e5, 7.5e4, 3.0e4, 1.4e5, 0.0],
        "water_vol": [100.0, 50.0, 20.0, -5.0, 0.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_keeps_only_full_producing_days() -> None:
    out, _ = clean(_raw())
    # injector, the 6 h day and the negative-water day are gone
    assert list(out["date"].dt.day) == [1, 2]
    assert set(out["well"].astype(str)) == {"A"}


def test_volumes_are_normalised_to_a_full_day() -> None:
    out, _ = clean(_raw())
    half_day = out.loc[out["date"].dt.day == 2].iloc[0]
    assert half_day["oil_rate"] == pytest.approx(1000.0)
    assert half_day["water_rate"] == pytest.approx(100.0)


def test_zero_gauge_readings_become_missing_and_dp_tubing_is_rederived() -> None:
    out, _ = clean(_raw())
    failed = out.loc[out["date"].dt.day == 2].iloc[0]
    assert np.isnan(failed["p_downhole"])
    # the stale 33 bar in the source must not survive a failed downhole gauge
    assert np.isnan(failed["dp_tubing"])
    healthy = out.loc[out["date"].dt.day == 1].iloc[0]
    assert healthy["dp_tubing"] == pytest.approx(200.0)
    # zero annulus pressure is physically possible and is kept
    assert healthy["p_annulus"] == 0.0


def test_report_accounts_for_every_removed_row() -> None:
    raw = _raw()
    out, report = clean(raw, CleaningConfig(min_on_stream_hrs=12.0))
    table = report.to_frame()
    assert table["rows_removed"].sum() == len(raw) - len(out)
    assert table["rows_remaining"].iloc[-1] == len(out)
