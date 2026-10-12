"""Train the demand model, check how good it is, and save it.

Run from the project folder:
    uv run python -m nz_electricity_demand_api.train --note "What changed in this run"

Each run appends a row of results to EXPERIMENTS.md.
"""

import argparse
import logging
from datetime import date

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from nz_electricity_demand_api.data import FEATURES, PROJECT_ROOT, TARGET, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "demand_model.joblib"
EXPERIMENTS_PATH = PROJECT_ROOT / "EXPERIMENTS.md"


def time_split(df: pd.DataFrame, test_from: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on everything before `test_from`, test on everything after.

    For time series you must NOT split randomly: a random split lets the
    model peek at the days either side of each test day, so the score looks
    far better than it will be on genuinely future dates.
    """
    cutoff = pd.Timestamp(test_from)
    return df[df["trading_date"] < cutoff], df[df["trading_date"] >= cutoff]


def report(name: str, y_true: pd.Series, y_pred) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    log.info("%-28s MAE = %4.0f MW   MAPE = %.1f%%", name, mae, mape)
    return mae, mape


def log_experiment(change: str, baseline_mae: float, model_mae: float, model_mape: float) -> None:
    """Append one row to the results table in EXPERIMENTS.md.

    The Notes column is left blank: fill in what you learned by hand.
    """
    row = (
        f"| {date.today()} | {change} | {baseline_mae:.0f} | {model_mae:.0f} "
        f"| {model_mape:.1f}% | |\n"
    )
    with EXPERIMENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(row)
    log.info("Logged results to %s", EXPERIMENTS_PATH.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the demand model.")
    parser.add_argument("--note", required=True, help="What changed in this run")
    args = parser.parse_args()

    df = load_dataset()
    log.info(
        "Loaded %s half-hours from %s to %s",
        f"{len(df):,}",
        df["trading_date"].min().date(),
        df["trading_date"].max().date(),
    )

    train, test = time_split(df, test_from="2025-01-01")
    log.info("Train: %s rows | Test (2025): %s rows", f"{len(train):,}", f"{len(test):,}")

    # 1. Baseline: "predict the average for that month, weekday and half-hour".
    #    Any real model has to beat this, or it isn't learning anything useful.
    keys = ["month", "day_of_week", "trading_period"]
    avg = train.groupby(keys)[TARGET].mean().rename("pred")
    baseline_pred = test.join(avg, on=keys)["pred"].fillna(train[TARGET].mean())
    baseline_mae, _ = report("Baseline (historical avg)", test[TARGET], baseline_pred)

    # 2. The model
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train[TARGET])
    model_mae, model_mape = report("HistGradientBoosting", test[TARGET], model.predict(test[FEATURES]))
    log_experiment(args.note, baseline_mae, model_mae, model_mape)

    # 3. Refit on ALL the data for the version we actually use, then save it.
    #    (The test score above is our honest estimate of how good it is.)
    model.fit(df[FEATURES], df[TARGET])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info("Saved model to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
