"""Gradient-boosted trees on sensor features (pure data-driven model)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from vfm.features import FEATURES

DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
}


class BoostedRegressor:
    """``HistGradientBoostingRegressor`` with the well as a native categorical feature.

    Missing sensor values are routed by the trees themselves, so no imputation is needed.
    With ``log_target`` the model is trained on ``log1p(y)``: errors then scale with the
    rate, which matches how metering accuracy is specified (relative error).
    """

    def __init__(
        self,
        features: Sequence[str] = FEATURES,
        *,
        log_target: bool = True,
        random_state: int = 0,
        **params: Any,
    ) -> None:
        self.features = tuple(features)
        self.log_target = log_target
        self.random_state = random_state
        self.params = {**DEFAULT_PARAMS, **params}
        self._wells: list[str] = []
        self._model: HistGradientBoostingRegressor | None = None

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        design = X[list(self.features)].astype(float)
        design["well"] = pd.Categorical(X["well"].astype(str), categories=self._wells)
        return design

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self._wells = sorted(X["well"].astype(str).unique())
        target = np.log1p(y.to_numpy()) if self.log_target else y.to_numpy()
        self._model = HistGradientBoostingRegressor(
            categorical_features="from_dtype", random_state=self.random_state, **self.params
        )
        self._model.fit(self._design(X), target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted")
        raw: np.ndarray = self._model.predict(self._design(X))
        return np.clip(np.expm1(raw), 0.0, None) if self.log_target else raw
