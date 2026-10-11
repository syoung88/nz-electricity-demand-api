"""Explore the national demand data before modelling.

Run from the project folder:
    uv run python scripts/explore.py

Prints a summary and saves charts to reports/figures/.
"""

import matplotlib

matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt

from nz_electricity_demand_api.data import PROJECT_ROOT, load_dataset, load_raw

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. The raw data -------------------------------------------------------
raw = load_raw()
print("=== Raw data ===")
print(f"Rows: {len(raw):,}  (one per connection point per day)")
print(f"Connection points (POCs): {raw['POC'].nunique()}")
print(f"Dates: {raw['Trading_Date'].min()} to {raw['Trading_Date'].max()}")
print(raw.head(3).iloc[:, :12])  # first 12 columns so it fits on screen

# --- 2. National half-hourly demand ----------------------------------------
df = load_dataset()
print("\n=== National demand (excluding Tiwai) ===")
print(f"Half-hour rows: {len(df):,}")
print(df["demand_mw"].describe().round(0))

periods_per_day = df.groupby("trading_date").size()
print("\nTrading periods per day (48 normal, 46/50 = daylight saving):")
print(periods_per_day.value_counts())

# --- 3. Charts ---------------------------------------------------------------
# a) Daily average demand over time: shows the seasonal (winter) peak
daily = df.groupby("trading_date")["demand_mw"].mean()
fig, ax = plt.subplots(figsize=(10, 4))
daily.plot(ax=ax)
ax.set(title="Daily average NZ demand (excl. Tiwai)", xlabel="", ylabel="MW")
fig.tight_layout()
fig.savefig(FIG_DIR / "daily_demand.png", dpi=120)

# b) Average shape of a day, weekday vs weekend
shape = df.groupby(["is_weekend", "trading_period"])["demand_mw"].mean().unstack(0)
shape.columns = ["Weekday", "Weekend"]
fig, ax = plt.subplots(figsize=(10, 4))
shape.plot(ax=ax)
ax.set(
    title="Average demand by half-hour",
    xlabel="Trading period (1 = 00:00-00:30, 48 = 23:30-24:00)",
    ylabel="MW",
)
fig.tight_layout()
fig.savefig(FIG_DIR / "daily_shape.png", dpi=120)

# c) Average by month
monthly = df.groupby("month")["demand_mw"].mean()
fig, ax = plt.subplots(figsize=(8, 4))
monthly.plot.bar(ax=ax)
ax.set(title="Average demand by month", xlabel="Month", ylabel="MW")
fig.tight_layout()
fig.savefig(FIG_DIR / "monthly_demand.png", dpi=120)

print(f"\nCharts saved to {FIG_DIR}")
