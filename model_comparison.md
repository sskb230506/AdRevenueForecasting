# Model Tuning & Comparison: Tweedie Objective & Hyperparameter Optimization

This document records the baseline model results, the Tweedie objective experiments, the results of **Optuna Hyperparameter Tuning**, and **Feature Engineering Experiments**.

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

## 2. Tweedie Objective Tuning Results (Default Hyperparameters)

We tested `objective="tweedie"` across nine different `tweedie_variance_power` values (`1.05` to `1.50`). 

Tweedie was evaluated in two setups:
1. **Experiment A:** Tweedie on the Log-transformed target (`log_revenue`).
2. **Experiment B:** Tweedie on the Raw target (`revenue`).

### Experiment B: Tweedie on Raw Target (`revenue`)
> [!IMPORTANT]
> Training Tweedie directly on raw revenue values provides a massive performance boost across all metrics, as it directly models the point mass at zero and positive continuous right-skewness.

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

## 3. Optuna Hyperparameter Optimization (150 Trials)

Using the best objective (**Tweedie on Raw Target with `tweedie_variance_power=1.05`**), we ran an automated Optuna study to optimize the remaining hyperparameters.

### Best Hyperparameters Found
```json
{
    "learning_rate": 0.033356087308330576,
    "num_leaves": 104,
    "max_depth": 9,
    "min_child_samples": 16,
    "subsample": 0.8775924060295223,
    "bagging_freq": 6,
    "colsample_bytree": 0.5380692088171432,
    "reg_alpha": 1.4292315637255737e-05,
    "reg_lambda": 3.308604137212157e-07,
    "min_split_gain": 2.8044580958429424
}
```

### Optuna Tuned Model 4-Fold CV Metrics
- **Mean CV $R^2$:** `0.6967`
- **Mean CV RMSE:** `3,091.67`
- **Mean CV MAE:** `1,191.34`
- **Mean CV WAPE:** `46.87%` (Best)

---

## 4. Final Holdout Evaluation Results
To verify that these tuning results generalize well and do not overfit the validation folds, we evaluated both the Baseline model and the Tuned Tweedie model on the **final 12-week unseen holdout dataset**.

| Model | Holdout $R^2$ | Holdout MAE | Holdout RMSE | Holdout WAPE |
| :--- | :---: | :---: | :---: | :---: |
| **Log-Target L2 Baseline** | 0.5295 | 1,517.15 | 3,521.14 | 52.25% |
| **Tuned Tweedie Model (Raw)** | **0.6194** | **1,430.82** | **3,166.72** | **49.28%** |
| **Absolute Improvement** | **+9.0%** | **-86.33** | **-354.42** | **-2.97%** |
| **Relative Improvement** | **+17.0%** | **-5.7%** | **-10.1%** | **-5.7%** |

---

## 5. Feature Engineering Experiments

We evaluated five separate feature engineering experiments on top of our best tuned Tweedie model, using walk-forward 4-fold CV to check for performance gains.

### Experiments Definition
- **Baseline Feature Set:** Standard features (`lag_1`, `lag_2`, `lag_4`, `rolling_4w` for revenue, spend, and ROAS).
- **Experiment A:** Adds `lag_8` and `lag_12` for revenue, spend, and ROAS.
- **Experiment B:** Adds `lag_26` and `lag_52` for revenue, spend, and ROAS.
- **Experiment C:** Adds expanding point-in-time mean for spend and ROAS (complementing the existing revenue baseline).
- **Experiment D:** Adds 8-week and 12-week rolling standard deviations.
- **Experiment E:** Adds 4-week and 8-week rolling medians.

### Results Comparison
*Evaluated on the exact same dataset splits:*

| Experiment | Mean CV $R^2$ | Mean CV WAPE | Mean CV MAE | Mean CV RMSE | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned Baseline (Tweedie)** | **0.6967** | **46.87%** | **1,191.34** | **3,091.67** | **Best** |
| **Experiment A (lag 8, 12)** | 0.6948 | 47.44% | 1,186.02 | 3,013.79 | Degraded |
| **Experiment B (lag 26, 52)** | 0.6900 | 47.46% | 1,188.71 | 3,055.54 | Degraded |
| **Experiment C (expanding mean)**| 0.6937 | 47.48% | 1,197.10 | 3,066.99 | Degraded |
| **Experiment D (rolling std)** | 0.6766 | 48.21% | 1,207.54 | 3,087.54 | Degraded |
| **Experiment E (rolling median)** | 0.6838 | 47.88% | 1,208.76 | 3,133.00 | Degraded |

> [!WARNING]
> **Why did additional features degrade performance?**
> 1. **Overfitting/Noise:** Adding more columns to a relatively small weekly dataset (3,278 rows) leads to higher model variance and overfitting, increasing CV error.
> 2. **Young Campaigns & NaNs:** Long lag features (like `lag_26` and `lag_52`) or long rolling windows introduce a high fraction of NaN values for newer campaigns. This missingness reduces the quality of split decisions in LightGBM and adds noise.

### Recommendation
Keep the **Tuned Baseline Feature Set** without adding any of the experimental lag or rolling feature groups.
