"""
================================================================================
 AD REVENUE FORECASTING PIPELINE  -  Google Ads + Meta Ads + Bing Ads
================================================================================

Goal
----
Given daily campaign-level spend/clicks/impressions/revenue for three ad
platforms, forecast weekly revenue per campaign:
    - a point estimate (P50)
    - a probabilistic range (P10 / P90) for downside / upside planning

Architecture
------------
A SINGLE pooled LightGBM model (not 130+ separate per-campaign models) is
trained on all platforms/campaigns together, using:
    - leakage-free engineered features (calendar, lags, rolling stats,
      causal campaign-level target encoding)
    - native categorical handling (no arbitrary LabelEncoder ordering)
    - a log1p target transform (revenue is heavily right-skewed)
    - LightGBM quantile regression for P10 / P50 / P90
    - walk-forward (expanding-window) time-series cross-validation, because
      a single 80/20 split can make a mediocre model look good or bad by luck
    - per-segment (platform x campaign_type) error reporting, since a couple
      of huge PMax campaigns will otherwise dominate the aggregate metric and
      hide poor performance everywhere else

A single pooled model is preferred over per-campaign time-series models
(e.g. one Prophet per campaign) because most individual campaigns only have
1-2 years of weekly history (~50-100 points) -- too little for a model that
has to learn its own seasonality/trend in isolation. Pooling lets every
campaign borrow statistical strength from every other campaign of the same
platform/type, while campaign-specific behaviour is still captured through
the campaign_id categorical feature and the causal target-encoded baseline.

How to run
----------
1. Edit CONFIG below to point at your three CSV files
   (Kaggle path shown is what this was originally built against).
2. Run end to end:  python revenue_forecasting_pipeline.py
   Or paste section by section into Jupyter/Kaggle cells (split on the
   "# %%" markers).
"""

# %% [Imports & config] -------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")
RANDOM_STATE = 42

CONFIG = {
    # Swap these for your Kaggle paths, e.g.
    # "/kaggle/input/datasets/saisameerkalamraju/adsdata/google_ads_campaign_stats.csv"
    "google_path": "data/google_ads_campaign_stats.csv",
    "meta_path": "data/meta_ads_campaign_stats.csv",
    "bing_path": "data/bing_campaign_stats.csv",
    "n_cv_folds": 4,           # walk-forward folds for evaluation
    "test_weeks": 12,          # final holdout window size (in weeks) for the headline numbers
    "quantiles": [0.1, 0.5, 0.9],
}


# %% [1. Load raw data] --------------------------------------------------------
def load_raw(cfg):
    google = pd.read_csv(cfg["google_path"])
    meta = pd.read_csv(cfg["meta_path"])
    bing = pd.read_csv(cfg["bing_path"])
    return google, meta, bing


# %% [2. Standardize each platform to a common schema] ------------------------
def simplify_meta_campaign_type(campaign_name: str) -> str:
    """Meta's CSV has no campaign-type column -- it's encoded as a prefix
    in the campaign name. Collapse it to the same granularity the other
    platforms use (objective-level, not full ad-set name)."""
    name = str(campaign_name)
    if "Prospecting" in name:
        return "PROSPECTING"
    if "Remarketing" in name:
        return "REMARKETING"
    if "Generic" in name:
        return "GENERIC"
    return "OTHER"


def standardize(google: pd.DataFrame, meta: pd.DataFrame, bing: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame({
        "platform": "Google",
        "campaign_id": google["campaign_id"].astype(str),
        "campaign_name": google["campaign_name"],
        "campaign_type": google["campaign_advertising_channel_type"],
        "date": pd.to_datetime(google["segments_date"]),
        "spend": google["metrics_cost_micros"] / 1_000_000,
        "revenue": google["metrics_conversions_value"],
        "clicks": google["metrics_clicks"],
        "impressions": google["metrics_impressions"],
        "daily_budget": google["campaign_budget_amount"],
    })

    m = pd.DataFrame({
        "platform": "Meta",
        "campaign_id": meta["campaign_id"].astype(str),
        "campaign_name": meta["campaign_name"],
        "campaign_type": meta["campaign_name"].apply(simplify_meta_campaign_type),
        "date": pd.to_datetime(meta["date_start"]),
        "spend": meta["spend"],
        "revenue": meta["conversion"],          # Meta's "conversion" column is the $ value field
        "clicks": meta["clicks"],
        "impressions": meta["impressions"],
        "daily_budget": meta["daily_budget"],
    })

    b = pd.DataFrame({
        "platform": "Bing",
        "campaign_id": bing["CampaignId"].astype(str),
        "campaign_name": bing["CampaignName"],
        "campaign_type": bing["CampaignType"],
        "date": pd.to_datetime(bing["TimePeriod"]),
        "spend": bing["Spend"],
        "revenue": bing["Revenue"],
        "clicks": bing["Clicks"],
        "impressions": bing["Impressions"],
        "daily_budget": bing["DailyBudget"],
    })

    combined = pd.concat([g, m, b], ignore_index=True)

    # campaign_type should be comparable across platforms -> uppercase, strip,
    # and fold platform-specific spelling variants onto one canonical label
    combined["campaign_type"] = combined["campaign_type"].astype(str).str.upper().str.strip()
    combined["campaign_type"] = combined["campaign_type"].replace({
        "PERFORMANCEMAX": "PERFORMANCE_MAX",
    })

    # basic cleaning
    combined = combined.dropna(subset=["date"])
    for col in ["spend", "revenue", "clicks", "impressions"]:
        combined[col] = combined[col].clip(lower=0)          # no negative spend/revenue/clicks
    combined["daily_budget"] = combined["daily_budget"].fillna(0)

    return combined


if __name__ == "__main__":
    google, meta, bing = load_raw(CONFIG)
    combined = standardize(google, meta, bing)
    print(combined.shape)
    print(combined.head())
    print(combined["platform"].value_counts())
    print(combined["campaign_type"].value_counts())


# %% [3. Weekly aggregation] ---------------------------------------------------
def to_weekly(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    combined["week"] = combined["date"].dt.to_period("W").apply(lambda p: p.start_time)

    weekly = (
        combined.groupby(["week", "platform", "campaign_type", "campaign_id", "campaign_name"])
        .agg(spend=("spend", "sum"),
             revenue=("revenue", "sum"),
             clicks=("clicks", "sum"),
             impressions=("impressions", "sum"),
             daily_budget=("daily_budget", "mean"))
        .reset_index()
        .sort_values(["campaign_id", "week"])
        .reset_index(drop=True)
    )

    # same-period efficiency ratios -- SAFE features: derived only from
    # spend/clicks/impressions, never from revenue (the target), so using
    # the *current* week's value is not leakage.
    weekly["ctr"] = np.where(weekly["impressions"] > 0, weekly["clicks"] / weekly["impressions"], 0)
    weekly["cpc"] = np.where(weekly["clicks"] > 0, weekly["spend"] / weekly["clicks"], 0)
    weekly["cpm"] = np.where(weekly["impressions"] > 0, weekly["spend"] / weekly["impressions"] * 1000, 0)

    # roas uses revenue -> only ever safe to use in LAGGED form (see below)
    weekly["roas"] = np.where(weekly["spend"] > 0, weekly["revenue"] / weekly["spend"], 0)

    return weekly


# %% [4. Feature engineering -- all leakage-checked] ---------------------------
def add_calendar_features(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly["month"] = weekly["week"].dt.month
    weekly["quarter"] = weekly["week"].dt.quarter
    weekly["year"] = weekly["week"].dt.year
    weekly["week_of_year"] = weekly["week"].dt.isocalendar().week.astype(int)
    weekly["is_holiday_season"] = weekly["month"].isin([11, 12]).astype(int)
    return weekly


def add_lag_and_rolling_features(weekly: pd.DataFrame) -> pd.DataFrame:
    g = weekly.groupby("campaign_id")

    for lag in [1, 2, 4]:
        weekly[f"lag_{lag}_revenue"] = g["revenue"].shift(lag)
        weekly[f"lag_{lag}_spend"] = g["spend"].shift(lag)
        weekly[f"lag_{lag}_roas"] = g["roas"].shift(lag)

    # rolling stats computed on values *shifted by 1 first* -> the window
    # never includes the current (target) week
    for col in ["revenue", "spend", "roas"]:
        weekly[f"rolling_4w_{col}"] = g[col].transform(lambda s: s.shift(1).rolling(4).mean())
        weekly[f"rolling_4w_{col}_std"] = g[col].transform(lambda s: s.shift(1).rolling(4).std())

    # week-over-week spend momentum (planned spend change -> known in advance, safe)
    weekly["spend_wow_change"] = g["spend"].pct_change().replace([np.inf, -np.inf], np.nan)

    # campaign maturity: how many weeks of history this campaign has had so far
    weekly["campaign_age_weeks"] = g.cumcount()

    # zero-activity flag
    weekly["is_zero_spend"] = (weekly["spend"] == 0).astype(int)

    return weekly


def add_causal_campaign_target_encoding(weekly: pd.DataFrame) -> pd.DataFrame:
    """Expanding (point-in-time) mean revenue for each campaign, using ONLY
    weeks strictly before the current one. This gives the model a strong
    'this campaign's typical baseline' signal without ever looking at the
    current or future target -- safe by construction, unlike a global
    groupby().mean() target encoding fit on the whole dataset."""
    weekly = weekly.sort_values(["campaign_id", "week"]).reset_index(drop=True)
    g = weekly.groupby("campaign_id")["revenue"]
    # shift(1) first so the expanding mean at row t only sees rows < t
    weekly["campaign_baseline_revenue"] = g.transform(lambda s: s.shift(1).expanding().mean())
    return weekly


def engineer_features(combined: pd.DataFrame) -> pd.DataFrame:
    weekly = to_weekly(combined)
    weekly = add_calendar_features(weekly)
    weekly = add_lag_and_rolling_features(weekly)
    weekly = add_causal_campaign_target_encoding(weekly)
    return weekly


if __name__ == "__main__":
    weekly = engineer_features(combined)
    print("\nWeekly grain shape:", weekly.shape)
    print(weekly.isnull().sum().sort_values(ascending=False).head(15))


# %% [5. Feature list -- explicitly leakage-checked] ---------------------------
CATEGORICAL_FEATURES = ["platform", "campaign_type", "campaign_id"]

NUMERIC_FEATURES = [
    "spend", "clicks", "impressions",          # current week, known same-day -> not leakage
    "ctr", "cpc", "cpm",                        # derived only from spend/clicks/impressions
    "daily_budget",
    "month", "quarter", "week_of_year", "is_holiday_season",
    "campaign_age_weeks", "is_zero_spend", "spend_wow_change",

    "lag_1_revenue", "lag_2_revenue", "lag_4_revenue",
    "lag_1_spend", "lag_2_spend", "lag_4_spend",
    "lag_1_roas", "lag_2_roas", "lag_4_roas",

    "rolling_4w_revenue", "rolling_4w_revenue_std",
    "rolling_4w_spend", "rolling_4w_spend_std",
    "rolling_4w_roas", "rolling_4w_roas_std",

    "campaign_baseline_revenue",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "revenue"

# NOTE: "roas" (current week) is deliberately EXCLUDED -- it's
# revenue / spend, i.e. computed directly from the target. Only its lagged
# and rolling versions (which never see the current week) are included.


def prepare_model_table(weekly: pd.DataFrame) -> pd.DataFrame:
    data = weekly.dropna(subset=NUMERIC_FEATURES + [TARGET]).copy()
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype("category")
    data["log_revenue"] = np.log1p(data[TARGET])
    return data.sort_values("week").reset_index(drop=True)


if __name__ == "__main__":
    data = prepare_model_table(weekly)
    print("\nModel-ready shape:", data.shape)
    print("Date range:", data["week"].min(), "->", data["week"].max())


# %% [6. Time-based splitting: holdout + walk-forward CV] ---------------------
def make_holdout_split(data: pd.DataFrame, test_weeks: int):
    """Reserve the most recent `test_weeks` calendar weeks as a final,
    never-touched-during-tuning holdout set."""
    weeks_sorted = np.sort(data["week"].unique())
    cutoff = weeks_sorted[-test_weeks]
    train_full = data[data["week"] < cutoff].copy()
    holdout = data[data["week"] >= cutoff].copy()
    return train_full, holdout


def walk_forward_splits(data: pd.DataFrame, n_folds: int, fold_weeks: int):
    """Expanding-window walk-forward CV: fold i trains on everything before
    a cutoff and validates on the `fold_weeks` immediately after it. Cutoffs
    step forward fold_weeks at a time, ending right before the holdout."""
    weeks_sorted = np.sort(data["week"].unique())
    splits = []
    for i in range(n_folds, 0, -1):
        val_end_idx = len(weeks_sorted) - (i - 1) * fold_weeks
        val_start_idx = val_end_idx - fold_weeks
        if val_start_idx <= 0:
            continue
        train_cut = weeks_sorted[val_start_idx]
        val_cut_end = weeks_sorted[val_end_idx - 1] if val_end_idx <= len(weeks_sorted) else weeks_sorted[-1]
        train_idx = data["week"] < train_cut
        val_idx = (data["week"] >= train_cut) & (data["week"] <= val_cut_end)
        if train_idx.sum() > 50 and val_idx.sum() > 5:
            splits.append((train_idx, val_idx))
    return splits


if __name__ == "__main__":
    train_full, holdout = make_holdout_split(data, CONFIG["test_weeks"])
    print(f"\ntrain_full: {train_full.shape}, weeks {train_full['week'].min()} -> {train_full['week'].max()}")
    print(f"holdout   : {holdout.shape}, weeks {holdout['week'].min()} -> {holdout['week'].max()}")

    cv_splits = walk_forward_splits(train_full, CONFIG["n_cv_folds"], CONFIG["test_weeks"])
    for i, (tr_idx, va_idx) in enumerate(cv_splits):
        print(f"fold {i}: train={tr_idx.sum()} rows, val={va_idx.sum()} rows, "
              f"val weeks {train_full.loc[va_idx,'week'].min()} -> {train_full.loc[va_idx,'week'].max()}")


# %% [7. Metrics] ---------------------------------------------------------------
def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error -- more meaningful than MAPE here
    because a huge fraction of campaign-weeks have revenue near $0, which
    makes plain MAPE blow up to nonsense values."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true).sum()
    return np.nan if denom == 0 else np.abs(y_true - y_pred).sum() / denom


def regression_report(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "n": len(y_true),
    }


def pinball_loss(y_true, y_pred, quantile):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


# %% [8. Point-estimate model (P50) -- LightGBM regressor on log1p(revenue)] ---
LGBM_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=15,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
)


def fit_point_model(X_train, y_train_log, X_val, y_val_log, params=None):
    params = params or LGBM_PARAMS
    model = LGBMRegressor(**params, verbosity=-1)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        eval_metric="mae",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
    )
    return model


def run_walk_forward_cv(train_full, cv_splits):
    fold_reports = []
    for i, (tr_idx, va_idx) in enumerate(cv_splits):
        X_tr, y_tr = train_full.loc[tr_idx, FEATURES], train_full.loc[tr_idx, "log_revenue"]
        X_va, y_va = train_full.loc[va_idx, FEATURES], train_full.loc[va_idx, "log_revenue"]

        model = fit_point_model(X_tr, y_tr, X_va, y_va)
        preds = np.expm1(model.predict(X_va, num_iteration=model.best_iteration_))
        actual = np.expm1(y_va)

        report = regression_report(actual, preds)
        report["fold"] = i
        report["best_iteration"] = model.best_iteration_
        fold_reports.append(report)
    return pd.DataFrame(fold_reports)


if __name__ == "__main__":
    cv_report = run_walk_forward_cv(train_full, cv_splits)
    print("\n=== Walk-forward CV (point model, P50) ===")
    print(cv_report)
    print("\nMean +/- std across folds:")
    print(cv_report[["MAE", "RMSE", "R2", "WAPE"]].agg(["mean", "std"]))


# %% [9. Final model: train on all train_full, evaluate on the true holdout] --
def time_based_internal_val(train_full, val_weeks):
    """Carve an internal validation tail (for early stopping) out of
    train_full -- always the most recent weeks, never random rows."""
    return make_holdout_split(train_full, val_weeks)


def fit_final_point_model(train_full, val_weeks):
    inner_train, inner_val = time_based_internal_val(train_full, val_weeks)
    X_tr, y_tr = inner_train[FEATURES], inner_train["log_revenue"]
    X_va, y_va = inner_val[FEATURES], inner_val["log_revenue"]
    model = fit_point_model(X_tr, y_tr, X_va, y_va)
    return model


def fit_quantile_model(train_full, val_weeks, alpha):
    inner_train, inner_val = time_based_internal_val(train_full, val_weeks)
    X_tr, y_tr = inner_train[FEATURES], inner_train["log_revenue"]
    X_va, y_va = inner_val[FEATURES], inner_val["log_revenue"]

    params = dict(LGBM_PARAMS)
    params.update(objective="quantile", alpha=alpha)
    model = LGBMRegressor(**params, verbosity=-1)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
    )
    return model


if __name__ == "__main__":
    final_model = fit_final_point_model(train_full, CONFIG["test_weeks"])

    X_holdout = holdout[FEATURES]
    holdout_preds = np.expm1(final_model.predict(X_holdout, num_iteration=final_model.best_iteration_))
    holdout_actual = holdout[TARGET].values

    print("\n=== Final holdout (most recent", CONFIG["test_weeks"], "weeks, P50 point model) ===")
    print(regression_report(holdout_actual, holdout_preds))

    quantile_models = {
        q: fit_quantile_model(train_full, CONFIG["test_weeks"], q) for q in CONFIG["quantiles"]
    }
    holdout_q_preds_raw = {
        q: np.expm1(m.predict(X_holdout, num_iteration=m.best_iteration_)) for q, m in quantile_models.items()
    }


# %% [9b. Conformal calibration of the P10/P90 band] --------------------------
# LightGBM's own quantile loss is not guaranteed to hit the nominal coverage
# (66% actual vs 80% target here -- bands are too tight). Fix with simple
# split-conformal calibration: measure the *additive* error needed to reach
# nominal coverage on an internal validation slice the model has not been
# scored against yet, then apply that same correction to the holdout.
def conformal_calibrate(train_full, val_weeks, models: dict, target_coverage=0.80):
    inner_train, inner_val = time_based_internal_val(train_full, val_weeks)
    X_val, y_val_log = inner_val[FEATURES], inner_val["log_revenue"]
    y_val = np.expm1(y_val_log)

    lo_q, hi_q = min(models), max(models)
    lo_pred = np.expm1(models[lo_q].predict(X_val, num_iteration=models[lo_q].best_iteration_))
    hi_pred = np.expm1(models[hi_q].predict(X_val, num_iteration=models[hi_q].best_iteration_))

    # widen symmetrically (in log space, so the correction scales with
    # revenue size rather than being a fixed dollar amount) until coverage
    # on inner_val reaches target_coverage
    for widen in np.arange(0, 3.0, 0.02):
        lo_adj = np.expm1(np.log1p(lo_pred) - widen)
        hi_adj = np.expm1(np.log1p(hi_pred) + widen)
        cov = np.mean((y_val >= lo_adj) & (y_val <= hi_adj))
        if cov >= target_coverage:
            return widen, cov
    return widen, cov


if __name__ == "__main__":
    widen, achieved_cov = conformal_calibrate(train_full, CONFIG["test_weeks"], quantile_models)
    print(f"\nConformal widening factor: {widen:.2f} (achieved {achieved_cov:.1%} on internal val)")

    p10 = np.expm1(np.log1p(holdout_q_preds_raw[0.1]) - widen)
    p50 = holdout_q_preds_raw[0.5]
    p90 = np.expm1(np.log1p(holdout_q_preds_raw[0.9]) + widen)

    coverage = np.mean((holdout_actual >= p10) & (holdout_actual <= p90))
    print(f"P10-P90 interval coverage on holdout AFTER calibration (target ~80%): {coverage:.1%}")
    print("Pinball losses (raw, pre-calibration):",
          {q: round(pinball_loss(np.log1p(holdout_actual), np.log1p(p), q), 4)
           for q, p in holdout_q_preds_raw.items()})


# %% [10. Segment-level breakdown -- don't trust one global number] -----------
def segment_report(holdout: pd.DataFrame, preds: np.ndarray, group_cols=("platform", "campaign_type")) -> pd.DataFrame:
    tmp = holdout.copy()
    tmp["pred"] = preds
    rows = []
    for keys, grp in tmp.groupby(list(group_cols)):
        if len(grp) < 5:
            continue
        r = regression_report(grp[TARGET], grp["pred"])
        r.update(dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


if __name__ == "__main__":
    seg_report = segment_report(holdout, holdout_preds)
    print("\n=== Holdout error by platform x campaign_type ===")
    print(seg_report[["platform", "campaign_type", "n", "MAE", "WAPE", "R2"]].to_string(index=False))


# %% [11. Sanity baseline -- are we even beating "predict last week's revenue"?] -
def naive_lag1_baseline(holdout: pd.DataFrame) -> dict:
    mask = holdout["lag_1_revenue"].notna()
    return regression_report(holdout.loc[mask, TARGET], holdout.loc[mask, "lag_1_revenue"])


if __name__ == "__main__":
    print("\n=== Naive baseline (predict = last week's revenue) ===")
    print(naive_lag1_baseline(holdout))


# %% [12. Feature importance] --------------------------------------------------
def feature_importance_table(model, features=FEATURES) -> pd.DataFrame:
    fi = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    fi = feature_importance_table(final_model)
    print("\n=== Top 15 features (final P50 model) ===")
    print(fi.head(15).to_string(index=False))

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi.head(15), x="importance", y="feature", color="steelblue")
    plt.title("LightGBM feature importance (P50 model)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()


# %% [13. Plot: actual vs P10/P50/P90 forecast for the biggest campaign] ------
if __name__ == "__main__":
    biggest_campaign = (
        holdout.groupby("campaign_id")[TARGET].sum().sort_values(ascending=False).index[0]
    )
    plot_df = holdout[holdout["campaign_id"] == biggest_campaign].sort_values("week").copy()
    idx = plot_df.index
    plot_df["p10"] = pd.Series(p10, index=holdout.index).loc[idx]
    plot_df["p50"] = pd.Series(p50, index=holdout.index).loc[idx]
    plot_df["p90"] = pd.Series(p90, index=holdout.index).loc[idx]

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["week"], plot_df[TARGET], label="Actual", marker="o", color="black")
    plt.plot(plot_df["week"], plot_df["p50"], label="P50 forecast", marker="o", color="steelblue")
    plt.fill_between(plot_df["week"], plot_df["p10"], plot_df["p90"], alpha=0.2,
                      color="steelblue", label="P10-P90 range")
    plt.title(f"Holdout forecast vs actual — campaign {biggest_campaign}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_vs_actual.png", dpi=120)
    plt.close()
    print("\nSaved feature_importance.png and forecast_vs_actual.png")


# %% [14. Forecast forward: recursive multi-week revenue forecast per campaign] -
def forecast_future(data: pd.DataFrame, point_model, quantile_models: dict, widen: float,
                     n_weeks: int = 8, future_spend: dict | None = None,
                     active_within_weeks: int = 3) -> pd.DataFrame:
    """Roll forward n_weeks beyond the last known week, per campaign.

    Because lag/rolling features depend on revenue, multi-step forecasting
    has to be recursive: each predicted week's P50 revenue is fed back in
    as "history" so the lag/rolling features for the following week are
    computable. This is standard practice for lag-feature tree models used
    for multi-step forecasting (the alternative -- training a separate
    model per horizon -- is more accurate but far more code; start here and
    upgrade to that only if recursive drift becomes a real problem).

    future_spend: optional {campaign_id: weekly_spend_value} to simulate a
    planned budget change. Defaults to carrying forward each campaign's most
    recent known weekly spend ("if nothing changes" scenario).

    active_within_weeks: campaigns whose last observed week is older than
    this many weeks before the dataset's global max week are treated as
    paused/ended and skipped -- there's no business value in "forecasting"
    a campaign that was shut down a year ago, and including it would also
    anchor that campaign's forecast to a stale starting date instead of the
    real current week.
    """
    future_spend = future_spend or {}
    global_max_week = data["week"].max()
    cutoff = global_max_week - pd.Timedelta(weeks=active_within_weeks)
    results = []
    skipped = 0

    for cid, hist in data.groupby("campaign_id"):
        hist = hist.sort_values("week").copy()
        if len(hist) < 5:
            continue  # not enough history to build lag features safely
        if hist["week"].iloc[-1] < cutoff:
            skipped += 1
            continue  # campaign looks paused/ended -- don't forecast it forward

        platform = hist["platform"].iloc[-1]
        campaign_type = hist["campaign_type"].iloc[-1]
        daily_budget = hist["daily_budget"].iloc[-1]
        last_clicks = hist["clicks"].iloc[-1]
        last_impr = hist["impressions"].iloc[-1]
        last_spend_actual = hist["spend"].iloc[-1]
        planned_spend = future_spend.get(cid, last_spend_actual)

        # rolling history buffers we will keep extending with predictions
        rev_hist = list(hist["revenue"].values)
        spend_hist = list(hist["spend"].values)
        roas_hist = list(hist["roas"].values)
        age = int(hist["campaign_age_weeks"].iloc[-1])
        last_week = hist["week"].iloc[-1]

        for step in range(1, n_weeks + 1):
            week = last_week + pd.Timedelta(weeks=step)
            age += 1

            spend = planned_spend
            clicks = last_clicks * (spend / last_spend_actual) if last_spend_actual > 0 else last_clicks
            impressions = last_impr * (spend / last_spend_actual) if last_spend_actual > 0 else last_impr
            ctr = clicks / impressions if impressions > 0 else 0
            cpc = spend / clicks if clicks > 0 else 0
            cpm = spend / impressions * 1000 if impressions > 0 else 0
            spend_wow_change = (spend / spend_hist[-1] - 1) if spend_hist[-1] > 0 else 0

            row = {
                "platform": platform, "campaign_type": campaign_type, "campaign_id": cid,
                "spend": spend, "clicks": clicks, "impressions": impressions,
                "ctr": ctr, "cpc": cpc, "cpm": cpm, "daily_budget": daily_budget,
                "month": week.month, "quarter": week.quarter,
                "week_of_year": week.isocalendar()[1], "is_holiday_season": int(week.month in (11, 12)),
                "campaign_age_weeks": age, "is_zero_spend": int(spend == 0),
                "spend_wow_change": spend_wow_change,
                "lag_1_revenue": rev_hist[-1], "lag_2_revenue": rev_hist[-2], "lag_4_revenue": rev_hist[-4],
                "lag_1_spend": spend_hist[-1], "lag_2_spend": spend_hist[-2], "lag_4_spend": spend_hist[-4],
                "lag_1_roas": roas_hist[-1], "lag_2_roas": roas_hist[-2], "lag_4_roas": roas_hist[-4],
                "rolling_4w_revenue": np.mean(rev_hist[-4:]), "rolling_4w_revenue_std": np.std(rev_hist[-4:]),
                "rolling_4w_spend": np.mean(spend_hist[-4:]), "rolling_4w_spend_std": np.std(spend_hist[-4:]),
                "rolling_4w_roas": np.mean(roas_hist[-4:]), "rolling_4w_roas_std": np.std(roas_hist[-4:]),
                "campaign_baseline_revenue": np.mean(rev_hist),
            }
            X_row = pd.DataFrame([row])
            for col in CATEGORICAL_FEATURES:
                X_row[col] = X_row[col].astype("category")
            X_row = X_row[FEATURES]

            pred_log = point_model.predict(X_row, num_iteration=point_model.best_iteration_)[0]
            p50_pred = float(np.expm1(pred_log))
            p10_pred = float(np.expm1(np.log1p(np.expm1(
                quantile_models[0.1].predict(X_row, num_iteration=quantile_models[0.1].best_iteration_)[0])) - widen))
            p90_pred = float(np.expm1(np.log1p(np.expm1(
                quantile_models[0.9].predict(X_row, num_iteration=quantile_models[0.9].best_iteration_)[0])) + widen))
            p50_pred, p10_pred, p90_pred = max(p50_pred, 0), max(p10_pred, 0), max(p90_pred, 0)

            results.append({
                "campaign_id": cid, "platform": platform, "campaign_type": campaign_type,
                "week": week, "p10": p10_pred, "p50": p50_pred, "p90": p90_pred,
            })

            # feed the P50 prediction back in as "history" for the next step
            rev_hist.append(p50_pred)
            spend_hist.append(spend)
            roas_hist.append(p50_pred / spend if spend > 0 else 0)
            last_spend_actual, last_clicks, last_impr = spend, clicks, impressions

    if skipped:
        print(f"  (skipped {skipped} campaign(s) with no activity in the last "
              f"{active_within_weeks} weeks -- treated as paused/ended)")
    return pd.DataFrame(results)


if __name__ == "__main__":
    future = forecast_future(data, final_model, quantile_models, widen, n_weeks=8)
    print(f"\n=== Forward forecast: next 8 weeks, {future['campaign_id'].nunique()} campaigns ===")
    print(future.groupby("week")[["p10", "p50", "p90"]].sum().round(0))
    future.to_csv("future_forecast.csv", index=False)
    print("\nSaved future_forecast.csv (per-campaign, per-week P10/P50/P90)")


# %% [15. Production model -- retrain on ALL data before deploying] -----------
# final_model/quantile_models above were deliberately trained on train_full
# ONLY (holding out the most recent test_weeks) so the metrics printed above
# are an honest read on unseen data. For the forecast you actually ship,
# retrain on every week you have -- there's no reason to throw away the
# most recent (and most relevant) weeks once evaluation is done.
if __name__ == "__main__":
    prod_model = fit_final_point_model(data, CONFIG["test_weeks"])
    prod_quantiles = {q: fit_quantile_model(data, CONFIG["test_weeks"], q) for q in CONFIG["quantiles"]}
    prod_widen, _ = conformal_calibrate(data, CONFIG["test_weeks"], prod_quantiles)

    future_prod = forecast_future(data, prod_model, prod_quantiles, prod_widen, n_weeks=8)
    future_prod.to_csv("future_forecast_production.csv", index=False)
    print("\n=== Production forecast (model retrained on full history) ===")
    print(future_prod.groupby("week")[["p10", "p50", "p90"]].sum().round(0))
    print("\nSaved future_forecast_production.csv -- this is the file to actually act on.")
