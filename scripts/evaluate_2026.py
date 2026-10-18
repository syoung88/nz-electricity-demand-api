"""Score the saved model on 2026 data it has never seen.

Put the 2026 *_Grid_export.csv files in data/new/ (NOT data/raw/), then run:
    uv run python scripts/evaluate_2026.py
"""

import joblib

from nz_electricity_demand_api.data import FEATURES, PROJECT_ROOT, TARGET, load_dataset
from nz_electricity_demand_api.train import MODEL_PATH, log, report

NEW_DIR = PROJECT_ROOT / "data" / "new"


def main() -> None:
    history = load_dataset()  # 2023–2025: what the saved model was trained on
    new = load_dataset(raw_dir=NEW_DIR)  # 2026: never seen

    # Guard against the classic mistake: test data overlapping training data
    assert new["trading_date"].min() > history["trading_date"].max(), "2026 data overlaps training data"
    log.info("Scoring on %s half-hours from %s to %s", f"{len(new):,}",
             new["trading_date"].min().date(), new["trading_date"].max().date())

    # Baseline, built from the same 2023–2025 history the model learned from
    keys = ["month", "day_of_week", "trading_period"]
    avg = history.groupby(keys)[TARGET].mean().rename("pred")
    baseline_pred = new.join(avg, on=keys)["pred"].fillna(history[TARGET].mean())
    report("Baseline (historical avg)", new[TARGET], baseline_pred)

    # The saved model: load it, don't retrain it
    model = joblib.load(MODEL_PATH)
    new["pred"] = model.predict(new[FEATURES])
    report("Saved model", new[TARGET], new["pred"])

    # Error by month: shows whether one period (e.g. winter) is much worse
    new["abs_error"] = (new[TARGET] - new["pred"]).abs()
    log.info("MAE by month:\n%s", new.groupby("month")["abs_error"].mean().round(0).to_string())


if __name__ == "__main__":
    main()