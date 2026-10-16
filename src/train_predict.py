"""Train the final freight-rate model and write assessment predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src import features as F
from src import preprocessing as pp


def read_clean(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def build_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**config.LGBM_FINAL_PARAMS)


def train_final_model() -> tuple[
    lgb.LGBMRegressor, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    train = read_clean(config.TRAIN_CLEAN)
    validation = read_clean(config.VALIDATION_CLEAN)
    params = pp.load_params()

    december_raw = pp.load_raw(config.DECEMBER_INPUTS)
    train_raw = pp.load_raw(config.TRAIN_RAW)
    validation_raw = pp.load_raw(config.VALIDATION_RAW)
    city_table = F.build_city_table(train_raw, validation_raw)
    december = F.add_coordinates(pp.clean_features(december_raw, params), city_table)

    X_train = F.build_features(train, config.FINAL_FEATURE_GROUPS)
    X_validation = F.build_features(validation, config.FINAL_FEATURE_GROUPS)
    X_december = F.build_features(december, config.FINAL_FEATURE_GROUPS)
    if list(X_train.columns) != list(X_validation.columns) or list(X_train.columns) != list(X_december.columns):
        raise ValueError("train, validation, and December feature columns differ")

    model = build_model()
    model.fit(X_train, np.log(train[config.TARGET]))
    return model, validation, december_raw, X_validation, X_december


def write_outputs(
    model: lgb.LGBMRegressor,
    validation: pd.DataFrame,
    december_raw: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_december: pd.DataFrame,
    predictions_path: Path,
    december_path: Path,
) -> None:
    validation_prediction = np.exp(model.predict(X_validation))
    december_prediction = np.exp(model.predict(X_december))

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"load_id": validation["load_id"], "predicted_rate": validation_prediction}
    ).to_csv(predictions_path, index=False)

    december_output = december_raw.copy()
    december_output["predicted_rate"] = december_prediction
    december_output.to_csv(december_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=config.ROOT / "validation_predictions.csv",
    )
    parser.add_argument(
        "--december-predictions",
        type=Path,
        default=config.ROOT / "december_predictions.csv",
    )
    args = parser.parse_args()

    model, validation, december_raw, X_validation, X_december = train_final_model()
    write_outputs(
        model,
        validation,
        december_raw,
        X_validation,
        X_december,
        args.predictions,
        args.december_predictions,
    )
    print(f"trained rows: {len(pd.read_csv(config.TRAIN_CLEAN)):,}")
    print(f"validation predictions: {args.predictions}")
    print(f"December predictions: {args.december_predictions}")


if __name__ == "__main__":
    main()
