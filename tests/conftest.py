"""Synthetic wells with known physics, so tests do not depend on the real workbook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TRUE_CHOKE_EXPONENT = 0.8
TRUE_DP_EXPONENT = 0.5


def make_daily(
    n_days: int = 420,
    wells: tuple[str, ...] = ("W-1", "W-2"),
    seed: int = 0,
    noise: float = 0.03,
) -> pd.DataFrame:
    """Cleaned-format daily frame generated from the choke equation.

    The choke is opened step by step while the reservoir depletes, the same confounding
    pattern as in the real field: rate falls over time although the choke opens.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i, well in enumerate(wells):
        t = np.arange(n_days, dtype=float)
        trend = 30.0 + 60.0 * t / n_days
        choke = np.clip(trend + 8.0 * rng.standard_normal(n_days), 5.0, 100.0)
        dp_choke = np.clip(6.0 + 2.0 * rng.standard_normal(n_days), 1.0, None)
        productivity = (1500.0 + 500.0 * i) * np.exp(-t / 250.0)
        liquid = (
            productivity
            * (choke / 100.0) ** TRUE_CHOKE_EXPONENT
            * dp_choke**TRUE_DP_EXPONENT
            * np.exp(noise * rng.standard_normal(n_days))
        )
        water_cut = 0.1 + 0.5 * t / n_days
        p_downhole = 250.0 - 0.1 * t + rng.standard_normal(n_days)
        p_wellhead = 40.0 + dp_choke + rng.standard_normal(n_days)
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.date_range("2015-01-01", periods=n_days, freq="D"),
                    "well": well,
                    "on_stream_hrs": 24.0,
                    "p_downhole": p_downhole,
                    "t_downhole": 100.0 + rng.standard_normal(n_days),
                    "dp_tubing": p_downhole - p_wellhead,
                    "p_annulus": 20.0 + rng.standard_normal(n_days),
                    "choke_pct": choke,
                    "p_wellhead": p_wellhead,
                    "t_wellhead": 80.0 + rng.standard_normal(n_days),
                    "dp_choke": dp_choke,
                    "oil_rate": liquid * (1.0 - water_cut),
                    "gas_rate": liquid * (1.0 - water_cut) * 150.0,
                    "water_rate": liquid * water_cut,
                }
            )
        )
    daily = pd.concat(frames, ignore_index=True)
    daily["well"] = daily["well"].astype("category")
    return daily


@pytest.fixture
def daily() -> pd.DataFrame:
    return make_daily()
