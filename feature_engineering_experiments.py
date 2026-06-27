import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
import json
import warnings
warnings.filterwarnings("ignore")

from revenue_forecasting_pipeline import (
    CONFIG, load_raw, standardize, engineer_features, prepare_model_table,
    make_holdout_split, walk_forward_splits, regression_report,
    CATEGORICAL_FEATURES, FEATURES, TARGET
)

def run_cv_for_features(train_full, cv_splits, model_params, features_list, target_col="revenue"):
    fold_reports = []
    for tr_idx, va_idx in cv_splits:
        X_tr = train_full.loc[tr_idx, features_list]
        y_tr = train_full.loc[tr_idx, target_col]
        X_va = train_full.loc[va_idx, features_list]
        y_va = train_full.loc[va_idx, target_col]

        model = LGBMRegressor(**model_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            categorical_feature=[c for c in CATEGORICAL_FEATURES if c in features_list],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        
        preds_raw = model.predict(X_va, num_iteration=model.best_iteration_)
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
    print("Loading optimized hyperparameters...")
    with open("best_tweedie_params.json", "r") as f:
        best_optuna_params = json.load(f)
        
    model_params = {
        "n_estimators": 2000,
        "objective": "tweedie",
        "tweedie_variance_power": 1.05,
        "random_state": 42,
        "verbosity": -1,
        "learning_rate": best_optuna_params["learning_rate"],
        "num_leaves": best_optuna_params["num_leaves"],
        "max_depth": best_optuna_params["max_depth"],
        "min_child_samples": best_optuna_params["min_data_in_leaf"],
        "subsample": best_optuna_params["bagging_fraction"],
        "bagging_freq": best_optuna_params["bagging_freq"],
        "colsample_bytree": best_optuna_params["feature_fraction"],
        "reg_alpha": best_optuna_params["reg_alpha"],
        "reg_lambda": best_optuna_params["reg_lambda"],
        "min_split_gain": best_optuna_params["min_gain_to_split"],
    }
    
    print("Preparing base data...")
    google, meta, bing = load_raw(CONFIG)
    combined = standardize(google, meta, bing)
    weekly_base = engineer_features(combined)
    data_base = prepare_model_table(weekly_base)
    
    # We lock the dataset row indices using the baseline model ready dataset.
    # Any new features will be merged/added to this exact index alignment.
    base_indices = data_base.index
    
    # Run Baseline (Tuned Tweedie on Baseline Features)
    train_full_base, holdout_base = make_holdout_split(data_base, CONFIG["test_weeks"])
    cv_splits = walk_forward_splits(train_full_base, CONFIG["n_cv_folds"], CONFIG["test_weeks"])
    
    print("\nRunning Baseline Feature Set CV...")
    baseline_res = run_cv_for_features(train_full_base, cv_splits, model_params, FEATURES)
    print(f"Baseline Results: R2={baseline_res['R2']:.4f}, WAPE={baseline_res['WAPE']:.4%}")
    
    experiments = {}
    
    # ----------------------------------------------------
    # Experiment A: lag 8 + lag 12
    # ----------------------------------------------------
    print("\nRunning Experiment A (+ lag 8, lag 12)...")
    weekly_A = weekly_base.copy()
    g_A = weekly_A.groupby("campaign_id")
    new_feats_A = []
    for lag in [8, 12]:
        for col in ["revenue", "spend", "roas"]:
            name = f"lag_{lag}_{col}"
            weekly_A[name] = g_A[col].shift(lag)
            new_feats_A.append(name)
            
    data_A = prepare_model_table(weekly_A).loc[base_indices]
    train_full_A, holdout_A = make_holdout_split(data_A, CONFIG["test_weeks"])
    res_A = run_cv_for_features(train_full_A, cv_splits, model_params, FEATURES + new_feats_A)
    experiments["Experiment A (lag 8, 12)"] = res_A
    print(f"Experiment A Results: R2={res_A['R2']:.4f}, WAPE={res_A['WAPE']:.4%}")

    # ----------------------------------------------------
    # Experiment B: lag 26 + lag 52
    # ----------------------------------------------------
    print("\nRunning Experiment B (+ lag 26, lag 52)...")
    weekly_B = weekly_base.copy()
    g_B = weekly_B.groupby("campaign_id")
    new_feats_B = []
    for lag in [26, 52]:
        for col in ["revenue", "spend", "roas"]:
            name = f"lag_{lag}_{col}"
            weekly_B[name] = g_B[col].shift(lag)
            new_feats_B.append(name)
            
    data_B = prepare_model_table(weekly_B).loc[base_indices]
    train_full_B, holdout_B = make_holdout_split(data_B, CONFIG["test_weeks"])
    res_B = run_cv_for_features(train_full_B, cv_splits, model_params, FEATURES + new_feats_B)
    experiments["Experiment B (lag 26, 52)"] = res_B
    print(f"Experiment B Results: R2={res_B['R2']:.4f}, WAPE={res_B['WAPE']:.4%}")

    # ----------------------------------------------------
    # Experiment C: expanding mean of spend and roas
    # ----------------------------------------------------
    print("\nRunning Experiment C (+ expanding mean of spend/roas)...")
    weekly_C = weekly_base.copy().sort_values(["campaign_id", "week"]).reset_index(drop=True)
    new_feats_C = []
    for col in ["spend", "roas"]:
        name = f"campaign_baseline_{col}"
        weekly_C[name] = weekly_C.groupby("campaign_id")[col].transform(lambda s: s.shift(1).expanding().mean())
        new_feats_C.append(name)
        
    data_C = prepare_model_table(weekly_C).loc[base_indices]
    train_full_C, holdout_C = make_holdout_split(data_C, CONFIG["test_weeks"])
    res_C = run_cv_for_features(train_full_C, cv_splits, model_params, FEATURES + new_feats_C)
    experiments["Experiment C (expanding mean spend/roas)"] = res_C
    print(f"Experiment C Results: R2={res_C['R2']:.4f}, WAPE={res_C['WAPE']:.4%}")

    # ----------------------------------------------------
    # Experiment D: rolling standard deviation (8w, 12w)
    # ----------------------------------------------------
    print("\nRunning Experiment D (+ rolling std 8w, 12w)...")
    weekly_D = weekly_base.copy()
    g_D = weekly_D.groupby("campaign_id")
    new_feats_D = []
    for window in [8, 12]:
        for col in ["revenue", "spend", "roas"]:
            name = f"rolling_{window}w_{col}_std"
            weekly_D[name] = g_D[col].transform(lambda s: s.shift(1).rolling(window).std())
            new_feats_D.append(name)
            
    data_D = prepare_model_table(weekly_D).loc[base_indices]
    train_full_D, holdout_D = make_holdout_split(data_D, CONFIG["test_weeks"])
    res_D = run_cv_for_features(train_full_D, cv_splits, model_params, FEATURES + new_feats_D)
    experiments["Experiment D (rolling std 8w, 12w)"] = res_D
    print(f"Experiment D Results: R2={res_D['R2']:.4f}, WAPE={res_D['WAPE']:.4%}")

    # ----------------------------------------------------
    # Experiment E: rolling median (4w, 8w)
    # ----------------------------------------------------
    print("\nRunning Experiment E (+ rolling median 4w, 8w)...")
    weekly_E = weekly_base.copy()
    g_E = weekly_E.groupby("campaign_id")
    new_feats_E = []
    for window in [4, 8]:
        for col in ["revenue", "spend", "roas"]:
            name = f"rolling_{window}w_{col}_median"
            weekly_E[name] = g_E[col].transform(lambda s: s.shift(1).rolling(window).median())
            new_feats_E.append(name)
            
    data_E = prepare_model_table(weekly_E).loc[base_indices]
    train_full_E, holdout_E = make_holdout_split(data_E, CONFIG["test_weeks"])
    res_E = run_cv_for_features(train_full_E, cv_splits, model_params, FEATURES + new_feats_E)
    experiments["Experiment E (rolling median 4w, 8w)"] = res_E
    print(f"Experiment E Results: R2={res_E['R2']:.4f}, WAPE={res_E['WAPE']:.4%}")

    # Save results to JSON
    summary_results = {
        "Baseline": baseline_res,
        "Experiment A (lag 8, 12)": res_A,
        "Experiment B (lag 26, 52)": res_B,
        "Experiment C (expanding mean spend/roas)": res_C,
        "Experiment D (rolling std 8w, 12w)": res_D,
        "Experiment E (rolling median 4w, 8w)": res_E,
    }
    with open("feature_engineering_results.json", "w") as f:
        json.dump(summary_results, f, indent=4)
        
    print("\n=== Feature Engineering Experiments Summary ===")
    print(f"{'Experiment':<45} | {'R2':<8} | {'WAPE':<8} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 90)
    print(f"{'Baseline':<45} | {baseline_res['R2']:.4f}   | {baseline_res['WAPE']:.4%} | {baseline_res['MAE']:.2f}  | {baseline_res['RMSE']:.2f}")
    for name, r in experiments.items():
        diff_r2 = r['R2'] - baseline_res['R2']
        diff_wape = r['WAPE'] - baseline_res['WAPE']
        print(f"{name:<45} | {r['R2']:.4f} ({diff_r2:+.4f}) | {r['WAPE']:.4%} ({diff_wape:+.4%}) | {r['MAE']:.2f} | {r['RMSE']:.2f}")
