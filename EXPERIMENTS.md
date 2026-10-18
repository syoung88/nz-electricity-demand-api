# Experiments

Test set: 2025 (trained on 2023–2024). Lower MAE is better.

| Date | Change | Baseline MAE | Model MAE | MAPE | Notes |
|---|---|---|---|---|---|
| 2026-10-12 | First model: trading_period, day_of_week, month, day_of_year, is_weekend | 175 | 166 | 4.4% | Beats baseline by ~5%. Needs better features. |
| 2026-10-12 | Add is_holiday | 175 | 161 | 4.2% | |

## Final test: 2026 (never seen)

The saved model (`models/demand_model.joblib`, trained on 2023–2025 with features: trading_period, day_of_week, month, day_of_year, is_weekend, is_holiday) was scored once on Jan–Sep 2026 with `scripts/evaluate_2026.py`. This data played no part in any modelling decision. The baseline here is the month × weekday × half-hour average over 2023–2025, so it isn't directly comparable to the baseline column above.

| Date | Period | Half-hours | Baseline MAE | Model MAE | MAPE |
|---|---|---|---|---|---|
| 2026-10-18 | 2026-01-01 to 2026-09-30 | 13,104 | 194 | 183 | 4.8% |

MAE by month (MW):

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|
| 152 | 138 | 143 | 172 | 139 | 204 | 247 | 250 | 200 |

Notes: Worse than the 2025 test (161 MW), as expected for a model scored further from its training data and not tuned to this period. It still beats the baseline by about 6%. Error is much higher in winter (Jun–Sep: 200–250 MW) than Jan–May (138–172 MW). Winter demand depends on heating, and the model has no temperature input, so adding temperature is the obvious next feature.
