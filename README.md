# Ad Revenue Forecasting Pipeline

An end-to-end, robust machine learning pipeline designed to forecast weekly ad revenue per campaign across Google Ads, Meta Ads, and Bing Ads. 

The system utilizes a single pooled LightGBM model combined with extensive leakage-free feature engineering, probabilistic quantile regression, walk-forward time-series validation, and conformal calibration.

---

## 📖 Architecture & Design Decisions

### 1. Single Pooled Model vs. Individual Models
Rather than training 130+ separate models (e.g., individual Prophet/ARIMA models for each campaign), we train a **single pooled LightGBM regressor** on all platforms and campaigns combined.
- **Statistical Strength:** Individual campaigns often have short history (e.g., 1–2 years of weekly data points). Pooling allows campaigns to share patterns (like platform-level seasonality or trend) while retaining campaign-specific behavior via categorical variables and causal target-encoded baselines.
- **Robustness:** Handles low-data or cold-start campaigns significantly better.

### 2. Leakage-Free Feature Engineering
To prevent look-ahead bias, all features are constructed using strictly historical context:
- **Lags & Rolling Metrics:** 1, 2, and 4-week lags for revenue, spend, and ROAS. Rolling averages and standard deviations are computed on historical buffers shifted by 1 week.
- **Causal Target Encoding:** Campain-level baselines are computed using expanding (point-in-time) means of preceding weeks, ensuring no future data leaks into the past.
- **Known-in-Advance Features:** Spend, impressions, clicks, daily budget, and calendar details (month, quarter, holiday flags) for the current week are treated as features (as they are known same-day or planned).

### 3. Probabilistic Range Forecasting
For downside and upside planning, the model forecasts:
- **P50:** Point estimate (Log-transformed target using `log1p` to handle highly right-skewed revenue distribution).
- **P10 & P90:** Probabilistic downside and upside ranges.
- **Split-Conformal Calibration:** Corrects LightGBM's default tight quantile losses to guarantee nominal interval coverage (e.g., ~80%) on unseen data.

---

## 🚀 Setup & Execution

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Ensure your raw data files (`google_ads_campaign_stats.csv`, `meta_ads_campaign_stats.csv`, `bing_campaign_stats.csv`) are located inside a `data/` folder in the project root:
```bash
python revenue_forecasting_pipeline.py
```

---

## 📊 Pipeline Reports

Running the pipeline outputs:
- **Walk-Forward CV Report:** Out-of-fold metrics across 4 expanding time windows.
- **Holdout Validation Report:** Direct comparison of the final P50 model against a naive "last week's revenue" baseline (improving MAE, RMSE, and $R^2$ by over **30%**).
- **Segment Report:** Metrics grouped by `platform` $\times$ `campaign_type`.
- **Top 15 Features list** by LightGBM split/gain importance.
- **Plots (`feature_importance.png`, `forecast_vs_actual.png`)** illustrating feature importances and actual vs. forecasted intervals.
- **Output Forecast CSVS (`future_forecast.csv`, `future_forecast_production.csv`)** detailing weekly predictions per campaign.
