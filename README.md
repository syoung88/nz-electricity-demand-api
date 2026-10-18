# NZ Electricity Demand API

A machine learning model that predicts New Zealand's national electricity demand for any half-hour of any date, served as a web API with FastAPI and packaged with Docker.

Example request and response:

```
GET /predict?trading_date=2026-01-15&trading_period=20
```

```json
{
  "trading_date": "2026-01-15",
  "trading_period": 20,
  "demand_mw": 4158.86
}
```

## How it works

- **Data:** the Electricity Authority's monthly `Grid_export` files (January 2023 to December 2025), https://www.ea.govt.nz/data-and-insights/datasets/Datasets/Wholesale/Metered_data/Grid_export/. Each row is one connection point on one day, with the energy used in each half-hour "trading period". These are summed into one national demand series in MW.
- **Tiwai Point is excluded.** The aluminium smelter is a single large customer that can be switched down at short notice, so its demand isn't driven by time of day or season.
- **Features:** calendar information only (trading period, day of week, month, day of year, weekend, NZ public holiday), so the model can predict any future date without needing recent data.
- **Model:** scikit-learn `HistGradientBoostingRegressor`, compared against a baseline that predicts the historical average for that month, weekday and half-hour.

## Results

Target: national half-hourly electricity demand (MW), excluding Tiwai Point. All splits are time-based, not random, so the model can't peek at neighbouring days. Lower is better.

| Evaluation | Trained on | Baseline MAE | Model MAE | MAPE |
|---|---|---|---|---|
| Test year 2025 | 2023–2024 | 175 MW | 161 MW | 4.2% |
| Unseen Jan–Sep 2026 (final model, scored once) | 2023–2025 | 194 MW | 183 MW | 4.8% |

The baseline predicts the historical average for the same month, weekday and half-hour. The saved model used by the API is the one trained on 2023–2025. On 2026 the model's error is lowest in summer and autumn (138–172 MW) and highest in winter (200–250 MW), which points to temperature as the main missing input. Full experiment log: [EXPERIMENTS.md](EXPERIMENTS.md).

**Limitations:** there is no weather data yet, and only calendar features are used, so the model can't know about a cold snap or a heatwave.

## Run it with Docker

You only need [Docker](https://docs.docker.com/get-started/get-docker/) installed.

```
git clone https://github.com/syoung88/nz-electricity-demand-api
cd nz-electricity-demand-api
docker build -t nz-electricity-api .
docker run --publish 8000:8000 nz-electricity-api
```

Then open http://localhost:8000/docs to try the API in your browser.

## Run it without Docker

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run uvicorn nz_electricity_demand_api.api:app --reload
```

## Retrain the model

The raw data isn't stored in the repo. Download the monthly `Grid_export` CSVs from the Electricity Authority into `data/raw/`, then run:

```
uv run python -m nz_electricity_demand_api.train --note "What changed in this run"
```

This prints the results, adds a row to `EXPERIMENTS.md`, and saves a new model to `models/demand_model.joblib`. Rebuild the Docker image afterwards so it picks up the new model.

## Project layout

```
src/nz_electricity_demand_api/
    data.py     load the CSVs, build national demand and features
    train.py    train, evaluate and save the model
    api.py      the FastAPI app
models/         the saved model the API loads
scripts/        data exploration and test on unseen 2026 data
tests/          tests (run with `uv run pytest`)
Dockerfile      how to build the API image
```
