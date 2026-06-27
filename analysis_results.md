# Ad Revenue Forecasting Pipeline Results

We have set up the environment, copied the raw advertising dataset files from the Downloads folder to the local workspace (`d:\AdsML\data`), installed all the packages specified in `requirements.txt` (including `lightgbm`, `scikit-learn`, `seaborn`, etc.), and executed the forecasting pipeline script (`revenue_forecasting_pipeline.py`).

Below is the summary of the setup and execution results.

---

## 1. Data Summary
The data spans from **January 29, 2024** to **June 01, 2026**. 
- **Total daily-grain rows loaded:** 25,562
- **Daily records by Platform:**
  - Google: 19,272
  - Meta: 3,417
  - Bing: 2,873
- **Daily records by Campaign Type:**
  - `PERFORMANCE_MAX`: 15,367
  - `SEARCH`: 5,431
  - `PROSPECTING`: 1,588
  - `REMARKETING`: 1,566
  - `VIDEO`: 476
  - `DEMAND_GEN`: 368
  - `SHOPPING`: 354
  - `GENERIC`: 263
  - `DISPLAY`: 83
  - `AUDIENCE`: 66

After converting the daily data to a **weekly aggregation**, the pipeline processed **3,820 weekly grain rows** (with 3,278 model-ready rows after constructing features with historical lags).

---

## 2. Model Performance & Evaluation

The pipeline trains a single pooled LightGBM model across all campaigns and platforms. To ensure robustness, a **Walk-Forward Time-Series Cross Validation (4 folds)** was performed, followed by evaluation on a final **12-week holdout window** (the most recent 12 weeks of data).

### Walk-Forward Cross-Validation (P50 Point Model)
| Fold | Train Size | Val Size | Val Weeks Range | MAE | RMSE | $R^2$ | WAPE | Best Iter |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 1,184 | 463 | 2025-04-14 $\rightarrow$ 2025-06-30 | 867.43 | 2,019.65 | 0.5402 | 52.05% | 100 |
| **Fold 1** | 1,647 | 373 | 2025-07-07 $\rightarrow$ 2025-09-22 | 426.30 | 846.55 | 0.7506 | 41.08% | 153 |
| **Fold 2** | 2,020 | 466 | 2025-09-29 $\rightarrow$ 2025-12-15 | 2,962.83 | 10,744.22 | 0.5184 | 53.64% | 115 |
| **Fold 3** | 2,486 | 458 | 2025-12-22 $\rightarrow$ 2026-03-09 | 902.58 | 2,039.19 | 0.6102 | 50.18% | 112 |
| **Mean** | **-** | **-** | **-** | **1,289.78** | **3,912.40** | **0.6048** | **49.24%** | **-** |
| **Std** | **-** | **-** | **-** | **1,136.22** | **4,588.56** | **0.1048** | **5.62%** | **-** |

### Headline Metrics on Final Holdout vs. Naive Baseline
The model was tested against a naive baseline model (which simply predicts the last week's revenue as the forecast). The results demonstrate a significant improvement:

| Metric | LightGBM P50 Point Model | Naive Baseline (Last Week's Revenue) | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE** | **1,389.64** | 1,993.84 | **+30.3%** |
| **RMSE** | **3,007.06** | 4,363.29 | **+31.1%** |
| **$R^2$** | **0.6567** | 0.2772 | **+136.9%** |
| **WAPE** | **47.85%** | 68.66% | **+30.3%** |

> [!NOTE]
> **Conformal Calibration:** The LightGBM probabilistic quantiles (P10/P90) were calibrated using split-conformal prediction. A conformal widening factor of **0.08** was applied, achieving a **73.1%** interval coverage on the holdout set (against an 80% nominal coverage target).

---

## 3. Performance by Platform and Campaign Segment
Evaluating performance at a segment level prevents high-volume campaigns (like Performance Max) from hiding low performance elsewhere:

| Platform | Campaign Type | N | MAE | WAPE | $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Google** | PERFORMANCE_MAX | 147 | 1,168.79 | 47.47% | 0.3254 |
| **Google** | SEARCH | 65 | 1,656.01 | 45.98% | 0.2975 |
| **Bing** | PERFORMANCE_MAX | 38 | 37.43 | 91.40% | 0.1404 |
| **Bing** | SEARCH | 27 | 265.25 | 86.90% | 0.1094 |
| **Meta** | PROSPECTING | 24 | 1,545.21 | 46.50% | 0.3565 |
| **Google** | SHOPPING | 12 | 9,785.70 | 45.66% | 0.2181 |
| **Meta** | REMARKETING | 12 | 1,758.10 | 78.75% | 0.2071 |
| **Bing** | SHOPPING | 9 | 54.56 | 105.47% | -0.1120 |

---

## 4. Visualizations

Here is a look at the feature importance list and the actual vs. forecast timeline for the largest campaign.

### Feature Importance (Top 15 Features)
The engineered features and current-week variables that most impact the prediction model:

![Feature Importance](file:///C:/Users/Admins/.gemini/antigravity/brain/5351e4a2-3c0b-4388-9440-2160a4b6e07c/artifacts/feature_importance.png)

### Forecast vs. Actual (Holdout Set)
A visual comparison of the P50 forecast, the actual values, and the P10-P90 prediction band for the highest volume campaign in the holdout window:

![Forecast vs Actual](file:///C:/Users/Admins/.gemini/antigravity/brain/5351e4a2-3c0b-4388-9440-2160a4b6e07c/artifacts/forecast_vs_actual.png)

---

## 5. Production Forecast (Next 8 Weeks)
The final production models were retrained using **all available historical data** up to June 01, 2026. The predicted weekly aggregated revenues across all 26 active campaigns are as follows:

| Week Commencing | P10 (Downside Plan) | P50 (Expected Revenue) | P90 (Upside Plan) |
| :---: | :---: | :---: | :---: |
| **2026-06-08** | $6,278 | $19,136 | $63,194 |
| **2026-06-15** | $6,155 | $19,947 | $65,582 |
| **2026-06-22** | $5,808 | $20,138 | $64,491 |
| **2026-06-29** | $4,558 | $19,005 | $60,434 |
| **2026-07-06** | $4,219 | $18,911 | $63,073 |
| **2026-07-13** | $3,769 | $18,329 | $62,570 |
| **2026-07-20** | $3,269 | $19,098 | $62,422 |
| **2026-07-27** | $3,449 | $18,175 | $62,081 |

> [!TIP]
> The detailed per-campaign, per-week forecasts can be found in [future_forecast_production.csv](file:///d:/AdsML/future_forecast_production.csv).
