"""Feature engineering shared by train, validation and the December chart.

``build_features`` takes a frame that already went through
``preprocessing.clean_features`` (so ``weight`` is positive and complete and
``weight_missing`` exists) and returns the model matrix.

Feature groups
--------------
core    distance, log_distance, weight, weight_missing, equipment one-hot (always on)
dow     day of week (0 = Monday)
coords  pickup / delivery latitude and longitude
month, t, doy   time features that were TESTED and rejected (see notebook 02);
                kept here only so the experiment can be reproduced.

The December chart file has no coordinates, so ``add_coordinates`` fills them
from a city table built from the data files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

COORD_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
TIME_ORIGIN = pd.Timestamp("2025-01-01")


def build_city_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) from any frames that have pickup/delivery coordinates."""
    parts = []
    for f in frames:
        for side in ("pickup", "delivery"):
            parts.append(
                f[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
            )
    table = pd.concat(parts).drop_duplicates()
    assert table["city"].is_unique, "a city has more than one coordinate pair"
    return table.set_index("city")


def add_coordinates(df: pd.DataFrame, city_table: pd.DataFrame) -> pd.DataFrame:
    """Fill the four coordinate columns from the city names (used for December)."""
    out = df.copy()
    for side in ("pickup", "delivery"):
        missing = set(out[side]) - set(city_table.index)
        if missing:
            raise KeyError(f"no coordinates for {sorted(missing)}")
        out[f"{side}_lat"] = out[side].map(city_table["lat"])
        out[f"{side}_lon"] = out[side].map(city_table["lon"])
    return out


def build_features(df: pd.DataFrame, groups=config.FINAL_FEATURE_GROUPS) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"].astype(float)
    X["log_distance"] = np.log(df["distance"].astype(float))
    X["weight"] = df["weight"].astype(float)
    X["weight_missing"] = df["weight_missing"].astype(int)
    for equipment in config.EQUIPMENT_TYPES:
        X[f"eq_{equipment.replace(' ', '_')}"] = (df["equipment"] == equipment).astype(int)

    if "dow" in groups:
        X["dow"] = df["date"].dt.dayofweek
    if "month" in groups:
        X["month"] = df["date"].dt.month
    if "t" in groups:
        X["t"] = (df["date"] - TIME_ORIGIN).dt.days
    if "doy" in groups:
        angle = 2 * np.pi * df["date"].dt.dayofyear / 365.25
        X["doy_sin"], X["doy_cos"] = np.sin(angle), np.cos(angle)
    if "coords" in groups:
        missing = [c for c in COORD_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"coordinate columns missing: {missing}; call add_coordinates first")
        for c in COORD_COLUMNS:
            X[c] = df[c].astype(float)

    assert not X.isna().any().any(), "feature matrix contains missing values"
    return X
