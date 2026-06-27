import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

from revenue_forecasting_pipeline import (
    CONFIG, load_raw, standardize, engineer_features, prepare_model_table,
    make_holdout_split, walk_forward_splits, regression_report,
    CATEGORICAL_FEATURES, FEATURES, TARGET, LGBM_PARAMS
)

def run_cv_for_params(train_full, cv_splits, model_params, target_col="log_revenue", is_log_target=True):
    fold_reports = []
    for tr_idx, va_idx in cv_splits:
        X_tr = train_full.loc[tr_idx, FEATURES]
        y_tr = train_full.loc[tr_idx, target_col]
        X_va = train_full.loc[va_idx, FEATURES]
        y_va = train_full.loc[va_idx, target_col]

        model = LGBMRegressor(**model_params, verbosity=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        
        preds_raw = model.predict(X_va, num_iteration=model.best_iteration_)
        if is_log_target:
            preds = np.expm1(preds_raw)
            actual = np.expm1(y_va)
        else:
            preds = np.clip(preds_raw, a_min=0, a_max=None)
            actual = y_va
            
        report = regression_report(actual, preds)
        fold_reports.append(report)
        
    df_report = pd.DataFrame(fold_reports)
    return {
        "MAE": df_report["MAE"].mean(),
        "RMSE": df_report["RMSE"].mean(),
        "R2": df_report["R2"].mean(),
        "WAPE": df_report["WAPE"].mean()
    }

if __name__ == "__main__":
    print("Loading and preparing data...")
    google, meta, bing = load_raw(CONFIG)
    combined = standardize(google, meta, bing)
    weekly = engineer_features(combined)
    data = prepare_model_table(weekly)
    
    train_full, holdout = make_holdout_split(data, CONFIG["test_weeks"])
    cv_splits = walk_forward_splits(train_full, CONFIG["n_cv_folds"], CONFIG["test_weeks"])
    
    # 1. Baseline Model (Regression on Log Target)
    print("\nRunning Baseline CV...")
    baseline_params = dict(LGBM_PARAMS)
    baseline_results = run_cv_for_params(train_full, cv_splits, baseline_params, target_col="log_revenue", is_log_target=True)
    print(f"Baseline Results: R2={baseline_results['R2']:.4f}, RMSE={baseline_results['RMSE']:.2f}, MAE={baseline_results['MAE']:.2f}, WAPE={baseline_results['WAPE']:.4%}")
    
    tweedie_powers = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5]
    
    # Experiment A: Tweedie on Log Target
    print("\n--- Experiment A: Tweedie on Log Target (log1p(revenue)) ---")
    log_tweedie_results = []
    for power in tweedie_powers:
        print(f"Running Tweedie CV (Log Target) with variance_power={power}...")
        params = dict(LGBM_PARAMS)
        params.update(objective="tweedie", tweedie_variance_power=power)
        res = run_cv_for_params(train_full, cv_splits, params, target_col="log_revenue", is_log_target=True)
        res["variance_power"] = power
        log_tweedie_results.append(res)
        print(f"  R2={res['R2']:.4f}, RMSE={res['RMSE']:.2f}, MAE={res['MAE']:.2f}, WAPE={res['WAPE']:.4%}")
        
    df_log_tweedie = pd.DataFrame(log_tweedie_results)
    
    # Experiment B: Tweedie on Raw Target (revenue)
    print("\n--- Experiment B: Tweedie on Raw Target (revenue) ---")
    raw_tweedie_results = []
    for power in tweedie_powers:
        print(f"Running Tweedie CV (Raw Target) with variance_power={power}...")
        params = dict(LGBM_PARAMS)
        params.update(objective="tweedie", tweedie_variance_power=power)
        res = run_cv_for_params(train_full, cv_splits, params, target_col="revenue", is_log_target=False)
        res["variance_power"] = power
        raw_tweedie_results.append(res)
        print(f"  R2={res['R2']:.4f}, RMSE={res['RMSE']:.2f}, MAE={res['MAE']:.2f}, WAPE={res['WAPE']:.4%}")
        
    df_raw_tweedie = pd.DataFrame(raw_tweedie_results)
    
    # Save the output results to CSV for record keeping
    df_log_tweedie.to_csv("tweedie_log_target_results.csv", index=False)
    df_raw_tweedie.to_csv("tweedie_raw_target_results.csv", index=False)
    
    print("\n=== Experiment A: Tweedie on Log Target Summary ===")
    print(df_log_tweedie[["variance_power", "R2", "RMSE", "MAE", "WAPE"]].to_string(index=False))
    
    print("\n=== Experiment B: Tweedie on Raw Target Summary ===")
    print(df_raw_tweedie[["variance_power", "R2", "RMSE", "MAE", "WAPE"]].to_string(index=False))
