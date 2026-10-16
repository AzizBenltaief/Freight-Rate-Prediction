"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

TRAIN_RAW = DATA_DIR / "train_test.csv"
VALIDATION_RAW = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUTS = DATA_DIR / "december_chart_inputs.csv"

TRAIN_CLEAN = PROCESSED_DIR / "train_test_clean.csv"
LABEL_OUTLIERS = PROCESSED_DIR / "train_label_outliers.csv"
VALIDATION_CLEAN = PROCESSED_DIR / "validation_clean.csv"
CLEANING_PARAMS = PROCESSED_DIR / "cleaning_params.json"

TARGET = "posted_rate"

# Features we deliberately do not use (decision: keep the model simple and make
# the December chart possible, since that file has no market_index/quote_signal).
DROPPED_FEATURES = ["market_index", "quote_signal"]

# A row is a corrupted label when its log-rate is further than this from the
# robust log-linear fit (log distance + equipment). In train_test.csv there are
# no rows between 0.3 and 0.7, so 0.5 sits inside an empty gap.
OUTLIER_LOG_THRESHOLD = 0.5

EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]
RANDOM_STATE = 42

# Expanding-window, forward-in-time validation folds on the cleaned development data.
# Each fold trains on every day up to `train_end` and tests on the next two months,
# which is the same horizon as the real task (train ends Oct 31, predict Nov-Dec).
CV_FOLDS = [
    # (name,         train_end,    test_start,   test_end)
    ("F1 May-Jun", "2025-04-30", "2025-05-01", "2025-06-30"),
    ("F2 Jul-Aug", "2025-06-30", "2025-07-01", "2025-08-31"),
    ("F3 Sep-Oct", "2025-08-31", "2025-09-01", "2025-10-31"),
]
# The fold that most resembles the real task (most recent data, same 2-month horizon).
MAIN_HOLDOUT = "F3 Sep-Oct"

# --- Phase 2: features and time-based validation ---------------------------
# Feature groups used by the final model (see src/features.py).
FINAL_FEATURE_GROUPS = ("dow", "coords")

# First day of the development holdout (train Jan-Aug, test Sep-Oct): the same
# two-month horizon as the real task (train Jan-Oct, predict Nov-Dec).
HOLDOUT_START = "2025-09-01"

# Fast fixed LightGBM settings used only to compare feature sets in notebook 02.
# They are not tuned; phase 3 tunes the final model.
LGBM_QUICK_PARAMS = dict(
    objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=31,
    min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
    verbose=-1, random_state=RANDOM_STATE,
)

# Final phase-3 model settings. The target is log(posted_rate); validation is
# performed before this model is refit on all clean Jan-Oct rows.
LGBM_FINAL_PARAMS = dict(
    objective="l1", n_estimators=900, learning_rate=0.035, num_leaves=31,
    min_child_samples=40, subsample=0.85, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=0.2, verbose=-1, random_state=RANDOM_STATE,
)
