from __future__ import annotations

import numpy as np
import pytest

from vfm.evaluation.metrics import regression_metrics


def test_known_values() -> None:
    m = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert m["mae"] == pytest.approx(15.0)
    assert m["wmape"] == pytest.approx(30.0 / 300.0)
    assert m["bias_pct"] == pytest.approx(100.0 * -10.0 / 300.0)


def test_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["r2"] == pytest.approx(1.0)
    assert m["wmape"] == 0.0


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        regression_metrics(np.ones(3), np.ones(2))
