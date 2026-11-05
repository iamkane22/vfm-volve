"""Meters and a small registry to build them by name."""

from __future__ import annotations

from collections.abc import Callable

from vfm.models.base import LastTest, Meter, PerPhase, RateModel, WellMean
from vfm.models.boosting import BoostedRegressor
from vfm.models.hybrid import HybridMeter
from vfm.models.physics import ChokeMeter

_REGISTRY: dict[str, Callable[[], Meter]] = {
    "well_mean": WellMean,
    "last_test": LastTest,
    "physics": ChokeMeter,
    "gbm": lambda: PerPhase("gbm", BoostedRegressor),
    "hybrid": HybridMeter,
}

MODEL_NAMES: tuple[str, ...] = tuple(_REGISTRY)

LABELS: dict[str, str] = {
    "well_mean": "Well mean (no sensors)",
    "last_test": "Last test carried forward",
    "physics": "Choke-ratio physics",
    "gbm": "Gradient boosting",
    "hybrid": "Hybrid (physics + learned correction)",
}


def build_model(name: str) -> Meter:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}") from None


__all__ = [
    "LABELS",
    "MODEL_NAMES",
    "BoostedRegressor",
    "ChokeMeter",
    "HybridMeter",
    "LastTest",
    "Meter",
    "PerPhase",
    "RateModel",
    "WellMean",
    "build_model",
]
