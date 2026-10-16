"""Time-based splits.

The real task trains on Jan-Oct 2025 and predicts Nov-Dec 2025, so every
validation scheme here trains on the past and tests on the next two months.
No random shuffling: it would let the model see loads from the same days it is
asked to predict.
"""
from __future__ import annotations

import pandas as pd

from src import config

# (name, test_start, test_end_exclusive); the training set is everything before test_start
EXPANDING_FOLDS = [
    ("A", "2025-05-01", "2025-07-01"),
    ("B", "2025-07-01", "2025-09-01"),
    ("C", "2025-09-01", "2025-11-01"),   # = the main holdout
]


def fold_masks(df: pd.DataFrame, test_start: str, test_end: str) -> tuple[pd.Series, pd.Series]:
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    return df["date"] < start, (df["date"] >= start) & (df["date"] < end)


def time_holdout(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Main holdout: train Jan-Aug, test Sep-Oct."""
    start = pd.Timestamp(config.HOLDOUT_START)
    return df["date"] < start, df["date"] >= start


def assert_no_leakage(df: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series) -> None:
    train, test = df[train_mask], df[test_mask]
    assert not (train_mask & test_mask).any(), "rows in both train and test"
    assert train["date"].max() < test["date"].min(), "training data is not strictly before test data"
    assert not set(train["load_id"]) & set(test["load_id"]), "load_id shared by train and test"
