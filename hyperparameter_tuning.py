import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
import optuna
import json
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from revenue_forecasting_pipeline import (
    CONFIG, load_raw, standardize, engineer_features, prepare_model_table,
    make_holdout_split, walk_forward_splits, regression_report,
    CATEGORICAL_FEATURES, FEATURES, TARGET, LGBM_PARAMS
)

def run_cv_for_params(train_full, cv_splits, model_params, target_col="revenue", is_log_target=False):
    fold_reports = []
    for tr_idx, va_idx in cv_splits:
        X_tr = train_full.loc[tr_idx, FEATURES]
        y_tr = train_full.loc[tr_idx, target_col]
        X_va = train_full.loc[va_idx, FEATURES]
        y_va = train_full.loc[va_idx, target_col]

        model = LGBMRegressor(**model_params)
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
    
    # 1. Baseline Tweedie Model with Default Hyperparameters
    print("\nRunning Default Tweedie CV...")
    default_tweedie_params = dict(LGBM_PARAMS)
    default_tweedie_params.update(objective="tweedie", tweedie_variance_power=1.05, verbosity=-1)
    default_res = run_cv_for_params(train_full, cv_splits, default_tweedie_params, target_col="revenue", is_log_target=False)
    print(f"Default Tweedie Results: R2={default_res['R2']:.4f}, RMSE={default_res['RMSE']:.2f}, MAE={default_res['MAE']:.2f}, WAPE={default_res['WAPE']:.4%}")
    
    # 2. Optuna Hyperparameter Optimization
    print("\nStarting Optuna Hyperparameter Tuning (150 trials)...")
    
    def objective(trial):
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_data_in_leaf", 5, 100),
            "subsample": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "colsample_bytree": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_gain_to_split", 0.0, 15.0),
            "objective": "tweedie",
            "tweedie_variance_power": 1.05,
            "random_state": 42,
            "verbosity": -1,
        }
        res = run_cv_for_params(train_full, cv_splits, params, target_col="revenue", is_log_target=False)
        # Minimize WAPE
        return res["WAPE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=150)
    
    print("\nTuning Complete!")
    print(f"Best Trial WAPE: {study.best_value:.4%}")
    print("Best Hyperparameters:")
    print(json.dumps(study.best_params, indent=4))
    
    # Save the best parameters
    with open("best_tweedie_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    # Evaluate best parameters on CV
    best_params = {
        "n_estimators": 2000,
        "objective": "tweedie",
        "tweedie_variance_power": 1.05,
        "random_state": 42,
        "verbosity": -1,
    }
    # Map trial suggestions to LGBMRegressor parameters
    best_params.update(study.best_params)
    # Rename parameter names back to lightgbm scikit-learn standard
    best_params["min_child_samples"] = best_params.pop("min_data_in_leaf")
    best_params["subsample"] = best_params.pop("bagging_fraction")
    best_params["colsample_bytree"] = best_params.pop("feature_fraction")
    best_params["min_split_gain"] = best_params.pop("min_gain_to_split")
    best_params["reg_alpha"] = best_params.pop("reg_alpha")
    best_params["reg_lambda"] = best_params.pop("reg_lambda")

    best_cv_res = run_cv_for_params(train_full, cv_splits, best_params, target_col="revenue", is_log_target=False)
    print(f"\nTuned Tweedie CV Results: R2={best_cv_res['R2']:.4f}, RMSE={best_cv_res['RMSE']:.2f}, MAE={best_cv_res['MAE']:.2f}, WAPE={best_cv_res['WAPE']:.4%}")
    
    # Evaluate on Holdout set
    print("\nEvaluating on Holdout Set...")
    # Baseline Model Holdout
    baseline_params = dict(LGBM_PARAMS)
    baseline_model = LGBMRegressor(**baseline_params, verbosity=-1)
    # Train on full train set
    baseline_model.fit(
        train_full[FEATURES], train_full["log_revenue"],
        categorical_feature=CATEGORICAL_FEATURES
    )
    baseline_holdout_preds = np.expm1(baseline_model.predict(holdout[FEATURES]))
    baseline_holdout_res = regression_report(holdout[TARGET].values, baseline_holdout_preds)
    
    # Tuned Tweedie Model Holdout
    tuned_model = LGBMRegressor(**best_params)
    tuned_model.fit(
        train_full[FEATURES], train_full["revenue"],
        categorical_feature=CATEGORICAL_FEATURES
    )
    tuned_holdout_preds = np.clip(tuned_model.predict(holdout[FEATURES]), a_min=0, a_max=None)
    tuned_holdout_res = regression_report(holdout[TARGET].values, tuned_holdout_preds)
    
    print("\nHoldout Comparison:")
    print(f"  Baseline: R2={baseline_holdout_res['R2']:.4f}, WAPE={baseline_holdout_res['WAPE']:.4%}")
    print(f"  Tuned Tweedie: R2={tuned_holdout_res['R2']:.4f}, WAPE={tuned_holdout_res['WAPE']:.4%}")
    
    # Save comparison data to json/txt
    results_comparison = {
        "baseline_cv": {
            "R2": 0.6048,
            "RMSE": 3912.40,
            "MAE": 1289.78,
            "WAPE": 0.4924
        },
        "default_tweedie_cv": default_res,
        "tuned_tweedie_cv": best_cv_res,
        "baseline_holdout": baseline_holdout_res,
        "tuned_tweedie_holdout": tuned_holdout_res
    }
    with open("results_comparison.json", "w") as f:
        json.dump(results_comparison, f, indent=4)
        
    print("\nResults comparison saved to results_comparison.json")
