# Freight Rate Prediction

Predict `posted_rate` for freight loads. See the assessment PDF for the full instructions.

## Project structure

```
freight-rate-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── train_test.csv                     # labeled development data (Jan-Oct 2025)
│   ├── validation.csv                     # 12,000 loads to predict (Nov-Dec 2025)
│   ├── validation_predictions_template.csv
│   ├── december_chart_inputs.csv
│   └── processed/                         # written by notebook 01
│       ├── train_test_clean.csv           # cleaned training data, ready for modeling
│       ├── train_label_outliers.csv       # corrupted-rate rows removed from training
│       ├── validation_clean.csv           # validation features, same cleaning, no rows dropped
│       └── cleaning_params.json           # values learned on train (weight median, rate line)
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb          # phase 1: explore + clean            (done)
│   ├── 02_features_and_split.ipynb        # phase 2: features + time-based validation (done)
│   └── 03_modeling.ipynb                  # phase 3: baseline vs gradient boosting
├── src/
│   ├── config.py                          # paths and constants
│   ├── preprocessing.py                   # cleaning (shared by train / validation / December)
│   ├── features.py                        # feature builder (same function for all three inputs)
│   ├── splits.py                          # time-based folds + leakage assertions
│   ├── evaluation.py                      # MAE / MAPE / RMSE / bias
│   └── train_predict.py                   # phase 3: final fit + predictions
├── scorer/score.py                        # provided by Spotter
├── reports/
│   ├── figures/                           # charts used in the report
│   ├── phase2_ablation.csv                # every feature set x fold result
│   └── phase2_decisions.json              # final feature set, target, split
├── validation_predictions.csv             # final deliverable (phase 3)
└── scorer_results/candidate_december.png  # produced by score.py (phase 3)
```

## How to run

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_and_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_features_and_split.ipynb
```

Notebook 01 writes `data/processed/`. Notebook 02 takes about one minute and writes `reports/`.

To train the final model and validate the required output files:

```powershell
.\.venv\Scripts\python.exe -m src.train_predict
.\.venv\Scripts\python.exe scorer\score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

This writes `validation_predictions.csv`, `december_predictions.csv`, and the December chart under `scorer_results/`.

## Cleaning decisions (notebook 01)

| Issue | Fix |
|---|---|
| 292 negative weights (sign errors) | absolute value |
| 300 missing weights | train median + `weight_missing` flag |
| 677 corrupted `posted_rate` values (1.4%) | removed from training, saved separately |
| `market_index`, `quote_signal` | not used (simplicity; December inputs do not contain them) |
| 8 cities only in validation | no city/lane identity features |

## Validation and features (notebook 02)

The real task trains on Jan-Oct 2025 and predicts Nov-Dec 2025, so every check trains on the past and tests on the
next two months. Three expanding windows: A (train Jan-Apr, test May-Jun), B (Jan-Jun, Jul-Aug) and
C (Jan-Aug, Sep-Oct), with C as the main holdout. `src/splits.py` asserts that each test period is strictly later
than its training data. Final model: fit on all of Jan-Oct.

Features: `distance`, `log_distance`, `weight`, `weight_missing`, equipment one-hot, day of week, and the four
pickup/delivery coordinates. Target: `log(posted_rate)`.

| Result (LightGBM, untuned) | MAE | MAPE |
|---|---|---|
| Naive $/mile x distance, fold C | $175 | 8.1% |
| Chosen features, fold C (clean rows) | $44.7 | 2.0% |
| Chosen features, mean of folds A, B, C | $81.9 | 3.4% |

Not used, and why: month / trend / seasonal terms (they help one fold and hurt another, because a tree freezes at the
last level it saw), city or lane identity (8 validation cities are unseen; coordinates generalise instead).
