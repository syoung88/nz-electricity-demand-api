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

- **Data:** the Electricity Authority's monthly `Grid_export` files (January 2023 to December 2025). Each row is one connection point on one day, with the energy used in each half-hour "trading period". These are summed into one national demand series in MW.
- **Tiwai Point is excluded.** The aluminium smelter is a single large customer that can be switched down at short notice, so its demand isn't driven by time of day or season.
- **Features:** calendar information only (trading period, day of week, month, day of year, weekend, NZ public holiday), so the model can predict any future date without needing recent data.
- **Model:** scikit-learn `HistGradientBoostingRegressor`, compared against a baseline that predicts the historical average for that month, weekday and half-hour.

## Results

Trained on 2023 to 2024 and tested on 2025 (a time-based split, not a random one, so the model can't peek at neighbouring days). Lower is better. Full history in [EXPERIMENTS.md](EXPERIMENTS.md).

| Model | MAE (MW) | MAPE |
|---|---|---|
| Baseline (historical average) | 175 | n/a |
| HistGradientBoosting with holiday feature | 161 | 4.2% |

The saved model used by the API is refit on all of 2023 to 2025.

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
scripts/        data exploration
tests/          tests (run with `uv run pytest`)
Dockerfile      how to build the API image
```
