"""Model interfaces and reference baselines.

A *meter* predicts all three phase rates for a frame of well-days. Single-phase
regressors are lifted to meters with :class:`PerPhase`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, Self

import numpy as np
import pandas as pd

from vfm.config import PHASES, TARGETS


class RateModel(Protocol):
    """Single-phase regressor: ``X`` has a ``well`` column plus feature columns."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class Meter(Protocol):
    """Multiphase model, fitted on frames that contain the target columns."""

    name: str

    def fit(self, frame: pd.DataFrame) -> Self: ...

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``oil_rate``, ``gas_rate`` and ``water_rate`` (Sm3/d) aligned with ``frame``."""
        ...


class PerPhase:
    """Fit an independent single-phase model for every phase."""

    def __init__(self, name: str, factory: Callable[[], RateModel]) -> None:
        self.name = name
        self._factory = factory
        self._models: dict[str, RateModel] = {}

    def fit(self, frame: pd.DataFrame) -> Self:
        self._models = {
            TARGETS[phase]: self._factory().fit(frame, frame[TARGETS[phase]]) for phase in PHASES
        }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._models:
            raise RuntimeError("Model is not fitted")
        return pd.DataFrame(
            {target: model.predict(frame) for target, model in self._models.items()},
            index=frame.index,
        )


class _ConstantPerWell:
    """Shared logic for baselines that predict one constant rate per well."""

    name = ""

    def __init__(self) -> None:
        self._per_well: pd.DataFrame | None = None
        self._fallback: pd.Series | None = None

    def _levels(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, frame: pd.DataFrame) -> Self:
        self._per_well = self._levels(frame)
        self._fallback = self._per_well.mean()
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._per_well is None or self._fallback is None:
            raise RuntimeError("Model is not fitted")
        wells = frame["well"].astype(str).to_numpy()
        out = self._per_well.reindex(wells).fillna(self._fallback)
        out.index = frame.index
        return out


class WellMean(_ConstantPerWell):
    """Mean rate of each well over the training data. Ignores sensors and time."""

    name = "well_mean"

    def _levels(self, frame: pd.DataFrame) -> pd.DataFrame:
        targets = list(TARGETS.values())
        return frame.groupby(frame["well"].astype(str))[targets].mean()


class LastTest(_ConstantPerWell):
    """Carry forward the rates of the most recent calibration window.

    This is what an operator does without a virtual meter: assume the well keeps
    producing what it produced in its last well test. A useful VFM has to beat it.
    """

    name = "last_test"

    def __init__(self, window_days: int = 7) -> None:
        super().__init__()
        self.window_days = window_days

    def _levels(self, frame: pd.DataFrame) -> pd.DataFrame:
        targets = list(TARGETS.values())
        recent = frame.sort_values("date").groupby("well", observed=True).tail(self.window_days)
        return recent.groupby(recent["well"].astype(str))[targets].mean()
