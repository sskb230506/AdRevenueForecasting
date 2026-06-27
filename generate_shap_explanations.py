import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
import shap
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Force matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

from revenue_forecasting_pipeline import (
    CONFIG, load_raw, standardize, engineer_features, prepare_model_table,
    make_holdout_split, CATEGORICAL_FEATURES, FEATURES, TARGET
)

if __name__ == "__main__":
    print("Loading data...")
    google, meta, bing = load_raw(CONFIG)
    combined = standardize(google, meta, bing)
    weekly = engineer_features(combined)
    data = prepare_model_table(weekly)
    
    train_full, holdout = make_holdout_split(data, CONFIG["test_weeks"])
    
    print("Loading optimized hyperparameters...")
    with open("best_tweedie_params.json", "r") as f:
        best_params = json.load(f)
        
    # Reconstruct parameter names back to lightgbm standard
    params = {
        "n_estimators": 2000,
        "objective": "tweedie",
        "tweedie_variance_power": 1.05,
        "random_state": 42,
        "verbosity": -1,
        "learning_rate": best_params["learning_rate"],
        "num_leaves": best_params["num_leaves"],
        "max_depth": best_params["max_depth"],
        "min_child_samples": best_params["min_data_in_leaf"],
        "subsample": best_params["bagging_fraction"],
        "bagging_freq": best_params["bagging_freq"],
        "colsample_bytree": best_params["feature_fraction"],
        "reg_alpha": best_params["reg_alpha"],
        "reg_lambda": best_params["reg_lambda"],
        "min_split_gain": best_params["min_gain_to_split"],
    }
    
    print("Training final tuned Tweedie model on full train set...")
    model = LGBMRegressor(**params)
    model.fit(
        train_full[FEATURES], train_full["revenue"],
        categorical_feature=CATEGORICAL_FEATURES
    )
    
    X_explain = holdout[FEATURES]
    print(f"Calculating SHAP values for holdout set ({X_explain.shape[0]} rows)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_explain)
    
    # 1. SHAP Summary Plot (dot plot)
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, show=False)
    plt.title("SHAP Feature Importance (Summary Plot)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.close()
    
    # 2. SHAP Bar Plot
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance (Bar Plot)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("shap_bar.png", dpi=150)
    plt.close()
    
    # 3. SHAP Dependence Plots
    # Identify the top 3 features based on mean absolute SHAP values
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1][:3]
    top_features = [FEATURES[idx] for idx in top_indices]
    print(f"Top 3 features identified by SHAP: {top_features}")
    
    for i, feature in enumerate(top_features):
        print(f"Generating SHAP dependence plot for {feature}...")
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature, shap_values.values, X_explain, show=False)
        plt.title(f"SHAP Dependence Plot: {feature}", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{feature}.png", dpi=150)
        plt.close()
        
    print("\nSHAP explanation plots successfully generated!")
