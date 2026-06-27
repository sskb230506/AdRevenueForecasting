# Model Tuning & Comparison: Tweedie Objective Experiments

This document records the baseline model results and the results of hyperparameter tuning experiments to evaluate the **Tweedie objective** (`objective="tweedie"`) with different `tweedie_variance_power` values.

---

## 1. Reference Baseline Model
- **Target:** `log_revenue` (`log1p(revenue)`)
- **Objective:** `regression` (L2/MSE loss)
- **Parameters:**
  - `n_estimators`: 2,000
  - `learning_rate`: 0.03
  - `num_leaves`: 31
  - `min_child_samples`: 15
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `reg_alpha`: 0.1
  - `reg_lambda`: 0.1

### Baseline 4-Fold CV Metrics
- **Mean CV $R^2$:** `0.6048`
- **Mean CV RMSE:** `3,912.40`
- **Mean CV MAE:** `1,289.78`
- **Mean CV WAPE:** `49.24%`

---

## 2. Tweedie Objective Tuning Results

We tested `objective="tweedie"` across nine different `tweedie_variance_power` values (`1.05` to `1.50`). 

Tweedie was evaluated in two setups:
1. **Experiment A:** Tweedie on the Log-transformed target (`log_revenue`).
2. **Experiment B:** Tweedie on the Raw target (`revenue`).

### Experiment A: Tweedie on Log Target (`log_revenue`)
> [!NOTE]
> Training Tweedie on log-transformed values does not natively leverage the zero-inflation benefits of Tweedie and generally underperforms compared to L2 regression.

| Tweedie Variance Power | Mean CV $R^2$ | Mean CV RMSE | Mean CV MAE | Mean CV WAPE |
| :---: | :---: | :---: | :---: | :---: |
| 1.05 | 0.6090 | 4,017.75 | 1,324.37 | 49.64% |
| 1.10 | 0.5943 | 4,094.79 | 1,344.43 | 50.20% |
| 1.15 | 0.5837 | 4,088.56 | 1,346.86 | 50.58% |
| 1.20 | 0.1044 | 4,987.52 | 1,437.16 | 53.84% |
| 1.25 | 0.5638 | 4,318.09 | 1,410.46 | 51.52% |
| 1.30 | 0.4529 | 4,662.74 | 1,477.15 | 53.86% |
| 1.35 | 0.3984 | 4,776.47 | 1,481.30 | 54.21% |
| 1.40 | 0.4996 | 4,664.72 | 1,517.27 | 54.79% |
| 1.50 | 0.4557 | 4,893.15 | 1,610.41 | 57.27% |

### Experiment B: Tweedie on Raw Target (`revenue`)
> [!IMPORTANT]
> Training Tweedie directly on raw revenue values provides a **massive performance boost** across all metrics, as it directly models the point mass at zero and positive continuous right-skewness.

| Tweedie Variance Power | Mean CV $R^2$ | Mean CV RMSE | Mean CV MAE | Mean CV WAPE |
| :---: | :---: | :---: | :---: | :---: |
| **1.05 (Best)** | **0.6981** | **2,954.33** | **1,203.40** | **47.85%** |
| 1.10 | 0.6862 | 3,033.83 | 1,216.18 | 48.21% |
| 1.15 | 0.6842 | 3,055.96 | 1,216.83 | 48.29% |
| 1.20 | 0.6889 | 3,022.41 | 1,218.37 | 48.30% |
| 1.25 | 0.6856 | 3,065.02 | 1,214.82 | 48.13% |
| 1.30 | 0.6860 | 3,086.23 | 1,211.43 | 47.90% |
| 1.35 | 0.6867 | 3,115.47 | 1,218.40 | 48.03% |
| 1.40 | 0.6838 | 3,144.98 | 1,233.38 | 48.53% |
| 1.50 | 0.6818 | 3,197.48 | 1,229.63 | 48.44% |

---

## 🏆 Summary & Recommendation

1. **Raw Target is Superior:** When using `objective="tweedie"`, we should fit the model directly on raw `revenue` (Experiment B), not log-transformed `log_revenue`.
2. **Performance Gains:** The best Tweedie configuration (**variance power = 1.05**) outperforms the baseline L2 log-target model significantly:
   - **$R^2$ increases** from `0.6048` to `0.6981` (**+15.4% relative improvement** / **+9.3% absolute**).
   - **RMSE decreases** from `3,912.40` to `2,954.33` (**-24.5% error reduction**).
   - **MAE decreases** from `1,289.78` to `1,203.40` (**-6.7% error reduction**).
   - **WAPE decreases** from `49.24%` to `47.85%` (**-2.8% relative error reduction**).

3. **Optimal Hyperparameter:** `tweedie_variance_power = 1.05`.
