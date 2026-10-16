"""Cleaning functions for the freight rate data.

Two kinds of steps, kept separate on purpose:

* Row-preserving steps (``clean_features``) are applied identically to
  train_test.csv, validation.csv and the December chart inputs. They never drop
  rows, because every validation load needs a prediction.
* Label-outlier removal (``flag_label_outliers``) needs ``posted_rate`` and is
  therefore only applied to the labeled development data.

Anything learned from data (the weight median, the robust fit) is computed on
train_test.csv only and stored in ``cleaning_params.json`` so the same values
are reused at prediction time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

STRING_COLUMNS = ["load_id", "pickup", "delivery", "equipment"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path: Path) -> pd.DataFrame:
    """Read a CSV and parse the date column."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Row-preserving cleaning (train, validation, December)
# --------------------------------------------------------------------------- #
def fit_params(train: pd.DataFrame) -> dict:
    """Learn the values used for imputation from the training data only."""
    return {"weight_median": float(train["weight"].abs().median())}


def clean_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Row-preserving cleaning of the feature columns.

    * strip whitespace in text columns
    * weight: negative values are sign errors -> absolute value;
      missing values -> train median plus a ``weight_missing`` flag
    * drop market_index and quote_signal (see config.DROPPED_FEATURES)
    """
    out = df.copy()

    for col in STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    if "weight" in out.columns:
        out["weight_missing"] = out["weight"].isna().astype(int)
        out["weight"] = out["weight"].abs().fillna(params["weight_median"])

    return out.drop(columns=[c for c in config.DROPPED_FEATURES if c in out.columns])


# --------------------------------------------------------------------------- #
# Label outliers (labeled data only)
# --------------------------------------------------------------------------- #
def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    """log(distance) plus one intercept per equipment type."""
    cols = [(df["equipment"] == e).astype(float).to_numpy() for e in config.EQUIPMENT_TYPES]
    cols.append(np.log(df["distance"].to_numpy()))
    return np.column_stack(cols)


def fit_robust_rate_line(df: pd.DataFrame, n_iter: int = 4, k: float = 4.0) -> dict:
    """Trimmed least squares: log(rate) ~ equipment + log(distance).

    Each pass refits on the rows within ``k`` robust standard deviations (MAD)
    of the previous fit, so the corrupted rates cannot pull the line.
    """
    X = _design_matrix(df)
    y = np.log(df[config.TARGET].to_numpy())
    keep = np.ones(len(df), dtype=bool)
    for _ in range(n_iter):
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        resid = y - X @ beta
        center = np.median(resid[keep])
        mad_std = 1.4826 * np.median(np.abs(resid[keep] - center))
        keep = np.abs(resid - center) < k * mad_std
    return {"beta": beta.tolist(), "center": float(center), "mad_std": float(mad_std)}


def log_rate_residual(df: pd.DataFrame, line: dict) -> pd.Series:
    """Distance of each row's log-rate from the robust line."""
    pred = _design_matrix(df) @ np.asarray(line["beta"]) + line["center"]
    return pd.Series(np.log(df[config.TARGET].to_numpy()) - pred, index=df.index)


def flag_label_outliers(df: pd.DataFrame, line: dict, threshold: float | None = None) -> pd.Series:
    """True where posted_rate is a corrupted label (far from the robust line)."""
    threshold = config.OUTLIER_LOG_THRESHOLD if threshold is None else threshold
    return log_rate_residual(df, line).abs() > threshold


# --------------------------------------------------------------------------- #
# Checks and persistence
# --------------------------------------------------------------------------- #
def assert_clean(df: pd.DataFrame, labeled: bool = True) -> None:
    """Fail loudly if a cleaned frame still has a known problem."""
    assert df["load_id"].is_unique, "duplicate load_id"
    assert df["date"].notna().all(), "invalid dates"
    assert df["weight"].notna().all() and (df["weight"] > 0).all(), "bad weight"
    assert (df["distance"] > 0).all(), "bad distance"
    assert set(df["equipment"].unique()) <= set(config.EQUIPMENT_TYPES), "unknown equipment"
    for col in config.DROPPED_FEATURES:
        assert col not in df.columns, f"{col} should have been dropped"
    core = [c for c in df.columns]
    assert not df[core].isna().any().any(), "missing values remain"
    if labeled:
        assert (df[config.TARGET] > 0).all(), "bad target"


def save_params(params: dict, path: Path = config.CLEANING_PARAMS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))


def load_params(path: Path = config.CLEANING_PARAMS) -> dict:
    return json.loads(path.read_text())
