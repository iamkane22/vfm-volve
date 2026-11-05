"""Run every meter through an evaluation protocol and score it."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from vfm.config import PHASES, TARGETS
from vfm.evaluation import splits
from vfm.evaluation.metrics import regression_metrics
from vfm.features import add_features
from vfm.models import MODEL_NAMES, build_model

logger = logging.getLogger(__name__)

Protocol = Literal["walk-forward", "holdout"]


@dataclass(frozen=True)
class BenchmarkResult:
    protocol: str
    #: One row per (phase, model, scope); scope is "all" or a well name.
    metrics: pd.DataFrame
    #: Out-of-sample predictions in long format.
    predictions: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        overall = self.metrics[self.metrics["scope"] == "all"]
        return overall.set_index(["phase", "model"])[["wmape", "r2", "bias_pct", "mae", "n"]]


def _folds(frame: pd.DataFrame, protocol: Protocol) -> list[splits.Fold]:
    if protocol == "walk-forward":
        return splits.walk_forward(frame)
    if protocol == "holdout":
        return splits.holdout(frame)
    raise ValueError(f"Unknown protocol {protocol!r}")


def run_benchmark(
    daily: pd.DataFrame,
    protocol: Protocol = "walk-forward",
    models: Sequence[str] = MODEL_NAMES,
) -> BenchmarkResult:
    frame = add_features(daily)
    folds = _folds(frame, protocol)
    logger.info(
        "%s: %d folds, %d test rows", protocol, len(folds), sum(f.test.sum() for f in folds)
    )

    pieces: list[pd.DataFrame] = []
    for fold in folds:
        train, test = frame[fold.train], frame[fold.test]
        for name in models:
            predicted = build_model(name).fit(train).predict(test)
            for phase in PHASES:
                target = TARGETS[phase]
                pieces.append(
                    pd.DataFrame(
                        {
                            "cutoff": fold.cutoff,
                            "date": test["date"].to_numpy(),
                            "well": test["well"].astype(str).to_numpy(),
                            "model": name,
                            "phase": phase,
                            "actual": test[target].to_numpy(),
                            "predicted": predicted[target].to_numpy(),
                        }
                    )
                )
    predictions = pd.concat(pieces, ignore_index=True)
    return BenchmarkResult(protocol, _score(predictions), predictions)


def _score(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (phase, model), group in predictions.groupby(["phase", "model"], sort=False):
        scopes = [("all", group), *((str(w), g) for w, g in group.groupby("well"))]
        for scope, part in scopes:
            rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "scope": scope,
                    "n": len(part),
                    **regression_metrics(part["actual"].to_numpy(), part["predicted"].to_numpy()),
                }
            )
    return pd.DataFrame(rows)
