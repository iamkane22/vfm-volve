"""Accuracy measures used in flow metering."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, R2, wMAPE and volume bias.

    * ``wmape`` (sum of absolute errors over sum of actuals) is the headline number: unlike
      plain MAPE it is not blown up by near-zero rates.
    * ``bias_pct`` compares cumulative predicted and actual volume, which is what matters
      for production allocation and reporting.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and of equal shape")

    error = y_pred - y_true
    total = float(np.sum(np.abs(y_true)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "wmape": float(np.sum(np.abs(error))) / total if total > 0 else float("nan"),
        "bias_pct": 100.0 * float(np.sum(error)) / total if total > 0 else float("nan"),
    }
