# Experiments

Test set: 2025 (trained on 2023–2024). Lower MAE is better.

| Date | Change | Baseline MAE | Model MAE | MAPE | Notes |
|---|---|---|---|---|---|
| 2026-10-12 | First model: trading_period, day_of_week, month, day_of_year, is_weekend | 175 | 166 | 4.4% | Beats baseline by ~5%. Needs better features. |
| 2026-10-12 | Add is_holiday | 175 | 161 | 4.2% | |
