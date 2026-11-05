"""Choke-ratio meter: carry the last test forward, scaled by the choke equation.

Flow through a restriction follows the orifice equation ``q = C * A(u) * sqrt(dp / rho)``.
Writing the effective area as a power of the choke opening ``u`` and letting the pressure
exponent float (to absorb multiphase and compressibility effects) gives a choke
*capacity*

    k(u, dp) = (u / 100) ** a * dp ** b

Between two well tests the fluid properties and the reservoir change slowly, so the rate
on any day is the rate of the last test scaled by how much the capacity changed:

    q(t) = q_test * k(u_t, dp_t) / k(u_test, dp_test)

The exponents ``a`` and ``b`` must describe the short-term hydraulic response, not the
long-term decline. Over a well's life operators open the choke as the reservoir
depletes, so a naive regression of rate on choke opening confounds the two and can even
return a negative exponent. We therefore use a *within* estimator: log quantities are
demeaned inside 30-day blocks of each well before the least-squares fit, which removes
anything that changes slowly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from vfm.config import TARGETS

_INPUTS = ("choke_pct", "dp_choke")


@dataclass(frozen=True)
class ChokeExponents:
    choke: float
    dp: float

    def capacity(self, choke_pct: np.ndarray, dp_choke: np.ndarray) -> np.ndarray:
        capacity: np.ndarray = np.power(choke_pct / 100.0, self.choke) * np.power(dp_choke, self.dp)
        return capacity

    @property
    def is_physical(self) -> bool:
        # Opening the choke or raising the differential pressure must increase flow.
        return self.choke > 0.0 and self.dp > 0.0


#: Textbook values for a sub-critical orifice: flow area proportional to opening and
#: rate proportional to the square root of the pressure drop.
TEXTBOOK = ChokeExponents(choke=1.0, dp=0.5)


def fit_within_blocks(
    choke_pct: np.ndarray,
    dp_choke: np.ndarray,
    rate: np.ndarray,
    block: np.ndarray,
    min_block_size: int = 5,
) -> ChokeExponents:
    """Estimate the choke exponents from variation *within* blocks of days.

    Inputs must be strictly positive. Blocks smaller than ``min_block_size`` carry too
    little variation and are ignored.
    """
    frame = pd.DataFrame(
        {
            "log_u": np.log(choke_pct / 100.0),
            "log_dp": np.log(dp_choke),
            "log_q": np.log(rate),
            "block": block,
        }
    )
    sizes = frame.groupby("block")["log_q"].transform("size")
    frame = frame[sizes >= min_block_size]
    if len(frame) < 3:
        return TEXTBOOK
    centred = frame[["log_u", "log_dp", "log_q"]] - frame.groupby("block")[
        ["log_u", "log_dp", "log_q"]
    ].transform("mean")
    coef, *_ = np.linalg.lstsq(
        centred[["log_u", "log_dp"]].to_numpy(), centred["log_q"].to_numpy(), rcond=None
    )
    return ChokeExponents(choke=float(coef[0]), dp=float(coef[1]))


class ChokeMeter:
    name = "physics"

    def __init__(
        self,
        reference_days: int = 7,
        block_days: int = 30,
        min_rows_per_well: int = 60,
        max_ratio: float = 3.0,
    ) -> None:
        self.reference_days = reference_days
        self.block_days = block_days
        self.min_rows_per_well = min_rows_per_well
        #: Guards against single-day gauge spikes multiplying the reference rate.
        self.max_ratio = max_ratio
        self.exponents: dict[str, ChokeExponents] = {}
        self.pooled: ChokeExponents = TEXTBOOK
        self._operating_point: pd.DataFrame | None = None
        self._reference: pd.DataFrame | None = None
        self._reference_date: pd.Series | None = None

    # -- fitting ---------------------------------------------------------------------
    def fit(self, frame: pd.DataFrame) -> Self:
        data = frame.assign(
            well=frame["well"].astype(str),
            liquid=frame["oil_rate"] + frame["water_rate"],
        )
        valid = data.dropna(subset=list(_INPUTS))
        valid = valid[(valid["choke_pct"] > 0) & (valid["dp_choke"] > 0) & (valid["liquid"] > 0)]
        if valid.empty:
            raise ValueError("No rows with positive choke opening, choke dp and liquid rate")

        block = (
            valid["well"]
            + ":"
            + ((valid["date"] - valid["date"].min()).dt.days // self.block_days).astype(str)
        )

        def estimate(rows: pd.DataFrame) -> ChokeExponents:
            return fit_within_blocks(
                rows["choke_pct"].to_numpy(),
                rows["dp_choke"].to_numpy(),
                rows["liquid"].to_numpy(),
                block.loc[rows.index].to_numpy(),
            )

        pooled = estimate(valid)
        self.pooled = pooled if pooled.is_physical else TEXTBOOK
        self.exponents = {}
        for well, rows in valid.groupby("well"):
            if len(rows) < self.min_rows_per_well:
                continue
            fitted = estimate(rows)
            self.exponents[str(well)] = fitted if fitted.is_physical else self.pooled

        self._operating_point = valid.groupby("well")[list(_INPUTS)].median()

        recent = data.sort_values("date").groupby("well").tail(self.reference_days)
        recent = recent.assign(capacity=self._capacity(recent))
        self._reference = recent.groupby("well")[["capacity", *TARGETS.values()]].mean()
        self._reference_date = recent.groupby("well")["date"].max()
        return self

    def as_of_predictions(self, frame: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
        """Predictions for the fitting data *as the meter would have made them at the time*.

        History is cut into ``horizon_days`` blocks ending at the last date. Every block is
        predicted from the ``reference_days`` producing days just before it, exactly as a
        recalibrated meter is used. Rows without earlier data get NaN. The result carries a
        ``days_since_reference`` column. Used to train corrections without look-ahead.
        """
        if self._operating_point is None:
            raise RuntimeError("Model is not fitted")
        data = frame.assign(well=frame["well"].astype(str), capacity=self._capacity(frame))
        end = data["date"].max()
        block = (end - data["date"]).dt.days // horizon_days
        block_start = end - pd.to_timedelta((block + 1) * horizon_days - 1, unit="D")
        data = data.assign(block_start=block_start)

        columns = [*TARGETS.values(), "days_since_reference"]
        out = pd.DataFrame(np.nan, index=frame.index, columns=columns)
        for _, rows in data.sort_values("date").groupby("well"):
            for start, members in rows.groupby("block_start"):
                history = rows[rows["date"] < pd.Timestamp(str(start))].tail(self.reference_days)
                if history.empty:
                    continue
                ratio = np.clip(
                    members["capacity"].to_numpy() / history["capacity"].mean(),
                    1.0 / self.max_ratio,
                    self.max_ratio,
                )
                for target in TARGETS.values():
                    out.loc[members.index, target] = history[target].mean() * ratio
                out.loc[members.index, "days_since_reference"] = (
                    members["date"] - history["date"].max()
                ).dt.days.to_numpy()
        return out

    def days_since_reference(self, frame: pd.DataFrame) -> np.ndarray:
        """Calendar days between each row and the meter's last calibration day."""
        if self._reference_date is None:
            raise RuntimeError("Model is not fitted")
        reference = frame["well"].astype(str).map(self._reference_date)
        reference = reference.fillna(self._reference_date.max())
        return (frame["date"] - reference).dt.days.to_numpy(dtype=float)

    # -- prediction ------------------------------------------------------------------
    def _capacity(self, frame: pd.DataFrame) -> np.ndarray:
        if self._operating_point is None:
            raise RuntimeError("Model is not fitted")
        wells = frame["well"].astype(str).to_numpy()
        inputs: dict[str, np.ndarray] = {}
        for col in _INPUTS:
            # A failed gauge is replaced by the well's typical operating point, which
            # makes the ratio fall back to "no change" for that input.
            typical = (
                pd.Series(wells)
                .map(self._operating_point[col])
                .fillna(self._operating_point[col].median())
                .to_numpy()
            )
            observed = frame[col].to_numpy(dtype=float)
            inputs[col] = np.where(observed > 0, observed, typical)

        out = np.empty(len(frame), dtype=float)
        for well in np.unique(wells):
            rows = wells == well
            exponents = self.exponents.get(well, self.pooled)
            out[rows] = exponents.capacity(inputs["choke_pct"][rows], inputs["dp_choke"][rows])
        return out

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._reference is None:
            raise RuntimeError("Model is not fitted")
        reference = self._reference.reindex(frame["well"].astype(str).to_numpy())
        reference = reference.fillna(self._reference.median())
        ratio = np.clip(
            self._capacity(frame) / reference["capacity"].to_numpy(),
            1.0 / self.max_ratio,
            self.max_ratio,
        )
        return pd.DataFrame(
            {target: reference[target].to_numpy() * ratio for target in TARGETS.values()},
            index=frame.index,
        )
