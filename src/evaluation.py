"""Error metrics. The assessment does not name one, so all are reported."""
from __future__ import annotations

import numpy as np


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": float(np.abs(err).mean()),
        "MAPE_%": float((np.abs(err) / y_true).mean() * 100),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "bias_%": float((err / y_true).mean() * 100),   
    }
