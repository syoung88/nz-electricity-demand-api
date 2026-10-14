"""Tests for data.py.

Run with:  uv run pytest -v
"""

import pandas as pd

from nz_electricity_demand_api.data import TRADING_PERIODS, add_features, national_demand


def make_raw(rows):
    """Build a fake Grid_export table.

    rows: list of (poc, date, kwh_list). Periods past the end of kwh_list are
    left empty, the same way the real CSVs leave them blank on short days.
    """
    records = []
    for poc, date, kwh in rows:
        record = {"POC": poc, "Trading_Date": date}
        for i, tp in enumerate(TRADING_PERIODS):
            record[tp] = kwh[i] if i < len(kwh) else float("nan")
        records.append(record)
    return pd.DataFrame(records)


def test_tiwai_is_excluded():
    # Arrange: one normal day with two connection points.
    # ABC0011 uses 500 kWh every half hour. 500 kWh in half an hour = 1.0 MW
    # (kWh * 2 / 1000), so if Tiwai is dropped, every period should be 1.0 MW.
    # Tiwai gets a huge number so it's obvious if it sneaks into the total.
    raw = make_raw([
        ("ABC0011", "2024-01-15", [500.0] * 48),
        ("TWI2201", "2024-01-15", [99_999.0] * 48),
    ])

    # Act: run the function we're testing.
    result = national_demand(raw)

    # Assert: check what came out.
    assert len(result) == 48                     # one row per trading period
    assert (result["demand_mw"] == 1.0).all()    # only ABC0011's demand, no Tiwai


def test_daylight_saving_day_has_46_periods():
    # TODO (Sam): build a raw table for one day with only 46 values,
    # run national_demand, and assert there are 46 rows and the

    raw = make_raw([
            ("ABC0011", "2024-01-15", [500.0] * 46),
        ])

    result = national_demand(raw)
    assert len(result) == 46
    assert result["trading_period"].max() == 46
    # highest trading_period is 46.
    # Bonus: add a 50-period day too and check each date separately.
    ...


def test_calendar_features():
    # TODO (Sam): add_features only needs a trading_date column, e.g.
    df = pd.DataFrame({"trading_date": ["2024-02-05", "2024-02-10", "2024-06-03"]})
    # Pick a plain weekday, a Saturday and a public holiday (2024-02-06 is
    # Waitangi Day), then assert is_weekend and is_holiday are what you expect.

    result = add_features(df)
    assert result["is_weekend"].tolist() == [0,1,0]
    assert result["is_holiday"].tolist() == [0,0,1]



    
