"""Physics prior with a learned correction (grey-box meter).

The work is split by what each part is good at:

* the choke-ratio meter owns the response to the choke opening and the pressure drop
  across it, which follows the orifice equation;
* boosted trees learn a multiplicative correction for everything else (decline since the
  last test, water breakthrough, gauge behaviour), separately per phase:

      log1p(q) = log1p(q_physics) + f(other sensors, well, days since test)

The correction never sees the choke inputs. Because the physics prior increases with
choke opening and pressure drop and the correction does not depend on them, the meter's
response to a choke move is monotone by construction: opening the choke can never
lower the predicted rate.

The correction is trained on *as-of* physics predictions: every 30-day block of the
history is predicted from the test just before it, the way the meter is used in
operation. Training against today's calibration instead would ask the trees to explain
years of depletion they can never see at prediction time.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from vfm.config import TARGETS
from vfm.features import FEATURES
from vfm.models.boosting import BoostedRegressor
from vfm.models.physics import ChokeMeter

_AGE = "days_since_reference"
#: Inputs owned by the physics prior and hidden from the correction.
HYDRAULIC_INPUTS: tuple[str, ...] = ("choke_pct", "dp_choke")
CORRECTION_FEATURES: tuple[str, ...] = (
    *(f for f in FEATURES if f not in HYDRAULIC_INPUTS),
    _AGE,
)


class HybridMeter:
    name = "hybrid"

    def __init__(
        self, physics: ChokeMeter | None = None, horizon_days: int = 30, random_state: int = 0
    ) -> None:
        self.physics = physics or ChokeMeter()
        self.horizon_days = horizon_days
        self.random_state = random_state
        self._corrections: dict[str, BoostedRegressor] = {}

    def _correction_model(self) -> BoostedRegressor:
        # Strongly regularised: it only has to explain the residual of a good prior.
        return BoostedRegressor(
            CORRECTION_FEATURES,
            log_target=False,
            random_state=self.random_state,
            learning_rate=0.03,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=5.0,
        )

    def fit(self, frame: pd.DataFrame) -> Self:
        self.physics.fit(frame)
        as_of = self.physics.as_of_predictions(frame, self.horizon_days)
        usable = as_of[_AGE].notna().to_numpy()
        train, prior = frame[usable], as_of[usable]
        design = train.assign(**{_AGE: prior[_AGE].to_numpy()})

        self._corrections = {}
        for target in TARGETS.values():
            log_prior = np.log1p(prior[target].to_numpy())
            residual = pd.Series(np.log1p(train[target].to_numpy()) - log_prior, index=train.index)
            self._corrections[target] = self._correction_model().fit(design, residual)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._corrections:
            raise RuntimeError("Model is not fitted")
        prior = self.physics.predict(frame)
        design = frame.assign(**{_AGE: self.physics.days_since_reference(frame)})
        out = {}
        for target, correction in self._corrections.items():
            log_prior = np.log1p(prior[target].to_numpy())
            out[target] = np.clip(np.expm1(log_prior + correction.predict(design)), 0.0, None)
        return pd.DataFrame(out, index=frame.index)
