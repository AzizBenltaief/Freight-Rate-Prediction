"""Model persistence and single-load prediction for the web app.

The CLI entry point (train_predict.py) retrains from scratch on every run,
which is fine for batch scoring. Serving a UI needs fast answers, so the
fitted model and the city lookup table are cached in models/freight_model.joblib
on first use and loaded afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src import features as F
from src import preprocessing as pp
from src.train_predict import build_model, read_clean

_EQUIPMENT_EQ = {e: e.replace(" ", "_") for e in config.EQUIPMENT_TYPES}


def _build_bundle() -> dict:
    """Train the final model exactly as train_predict.train_final_model does."""
    train = read_clean(config.TRAIN_CLEAN)
    params = pp.load_params()

    train_raw = pp.load_raw(config.TRAIN_RAW)
    validation_raw = pp.load_raw(config.VALIDATION_RAW)
    city_table = F.build_city_table(train_raw, validation_raw)

    model = build_model()
    model.fit(F.build_features(train, config.FINAL_FEATURE_GROUPS), np.log(train[config.TARGET]))
    return {"model": model, "city_table": city_table, "feature_groups": config.FINAL_FEATURE_GROUPS}


def load_bundle(refresh: bool = False) -> dict:
    from joblib import dump, load

    if refresh and config.MODEL_PATH.exists():
        config.MODEL_PATH.unlink()
    if not refresh and config.MODEL_PATH.exists():
        bundle = load(config.MODEL_PATH)
    else:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        bundle = _build_bundle()
        dump(bundle, config.MODEL_PATH)
    return bundle


# --------------------------------------------------------------------------- #
# Single-load prediction
# --------------------------------------------------------------------------- #
def predict_load(
    bundle: dict,
    date: str,
    distance_miles: float,
    weight_lbs: float | None,
    equipment: str,
    pickup_city: str,
    delivery_city: str,
) -> float:
    """Predict a rate for one load using the same cleaning + feature pipeline
    as the batch entry point, so results match the CLI outputs."""
    row = {
        "load_id": "APP-000001",
        "pickup": pickup_city,
        "delivery": delivery_city,
        "distance": float(distance_miles),
        "equipment": equipment,
        "weight": np.nan if weight_lbs in (None, "", 0) else abs(float(weight_lbs)),
        "date": pd.Timestamp(date),
    }
    df = pd.DataFrame([row])
    df["date"] = pd.to_datetime(df["date"])

    params = pp.load_params()
    cleaned = pp.clean_features(df, params)
    cleaned = F.add_coordinates(cleaned, bundle["city_table"])
    X = F.build_features(cleaned, bundle["feature_groups"])
    return float(np.exp(bundle["model"].predict(X)[0]))


def december_chart(bundle: dict) -> pd.DataFrame:
    """The 31 daily rates for the fixed December lane (Lexington -> Fort Wayne)."""
    params = pp.load_params()
    december_raw = pp.load_raw(config.DECEMBER_INPUTS)
    december = F.add_coordinates(pp.clean_features(december_raw, params), bundle["city_table"])
    X = F.build_features(december, bundle["feature_groups"])
    out = december_raw.copy()
    out["predicted_rate"] = np.exp(bundle["model"].predict(X))
    return out
