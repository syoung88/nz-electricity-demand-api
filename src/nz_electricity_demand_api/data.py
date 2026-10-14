"""Load the Electricity Authority Grid_export CSVs and turn them into a
national half-hourly demand series, plus the features the model uses.

Lives in the package (not in a script) so that training AND the API
can import the same code. If the API built features differently from
training, predictions would silently be wrong.
"""

from pathlib import Path

import pandas as pd
import holidays

# Project root = three levels up from this file (src/nz_electricity_demand_api/data.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Tiwai Point aluminium smelter. A single industrial customer using ~14% of
# national demand, on a contract that can switch it down at short notice.
# It isn't driven by time of day or season, so we leave it out.
TIWAI_PREFIX = "TWI"

TRADING_PERIODS = [f"TP{i}" for i in range(1, 51)]

# The columns the model is trained on. The API will need to build exactly these.
FEATURES = ["trading_period", "day_of_week", "month", "day_of_year", "is_weekend", "is_holiday"]
TARGET = "demand_mw"


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read every monthly Grid_export CSV into one wide DataFrame.

    Each row is one connection point (POC) on one day, with columns
    TP1..TP50 holding the kWh used in each half-hour trading period.
    """
    files = sorted(raw_dir.glob("*_Grid_export.csv"))
    if not files:
        raise FileNotFoundError(f"No *_Grid_export.csv files found in {raw_dir}")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def national_demand(raw: pd.DataFrame) -> pd.DataFrame:
    """Sum all connection points (except Tiwai) into one national series.

    Returns one row per (trading_date, trading_period) with demand in MW.
    """
    raw = raw[~raw["POC"].str.startswith(TIWAI_PREFIX)]

    # Wide -> long: one row per date per trading period
    daily = raw.groupby("Trading_Date")[TRADING_PERIODS].sum(min_count=1)
    long = daily.reset_index().melt(
        id_vars="Trading_Date", var_name="tp", value_name="kwh"
    )

    # Normal days have 48 periods. Daylight-saving days have 46 (spring
    # forward) or 50 (fall back), so the unused columns are empty -> drop them.
    long = long.dropna(subset=["kwh"])

    long["trading_date"] = pd.to_datetime(long["Trading_Date"])
    long["trading_period"] = long["tp"].str[2:].astype(int)
    # kWh in a half hour -> average MW over that half hour: kWh * 2 / 1000
    long[TARGET] = long["kwh"] * 2 / 1000

    return (
        long[["trading_date", "trading_period", TARGET]]
        .sort_values(["trading_date", "trading_period"])
        .reset_index(drop=True)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features from trading_date and trading_period.

    Only uses things you know in advance for any future date, so the API
    can predict for a date someone asks about without needing recent data.
    """
    df = df.copy()
    dates = pd.to_datetime(df["trading_date"])
    df["day_of_week"] = dates.dt.dayofweek  # Monday=0 ... Sunday=6
    df["month"] = dates.dt.month
    df["day_of_year"] = dates.dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    nz_holidays = holidays.country_holidays("NZ", years=dates.dt.year.unique())
    df["is_holiday"] = dates.dt.date.isin(nz_holidays).astype(int)
    return df


def load_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Everything in one call: raw CSVs -> national demand with features."""
    return add_features(national_demand(load_raw(raw_dir)))
