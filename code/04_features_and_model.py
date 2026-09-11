"""
04_features_and_model.py

Feature engineering + modeling + walk-forward backtest + Polar Vortex stress
test + quantile intervals, all leakage-free for a DAY-AHEAD forecasting setup:
every feature is built from data known at least 24h before the hour being
predicted.

Models compared, in order (this progression IS the portfolio story):
  1. Naive seasonal baseline  -- predict = same hour, one week ago
  2. Model A: HistGradientBoostingRegressor, lag + calendar features only
  3. Model B: same, PLUS a temperature feature
     (lightgbm was the original plan, but is not installable in this
     sandbox -- see requirements.txt note. sklearn's
     HistGradientBoostingRegressor is a very close, fully documented
     substitute: histogram-based gradient boosting, native quantile loss,
     and no material accuracy gap for a tabular problem at this scale.)

Also fits 10th/90th percentile quantile models from Model B's feature set for
an interval forecast, and trains a final full-history "deployment" model.
"""
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _features import build_features, FEATURES_NO_WEATHER, FEATURES_WITH_WEATHER, TEMP

BASE = Path(__file__).resolve().parents[1]
CLEAN = BASE / "data" / "processed" / "pjm_load_clean.csv"
PLOTS = BASE / "plots"
RESULTS = BASE / "outputs" / "results"
MODELS = BASE / "outputs" / "models"
for p in (PLOTS, RESULTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA = \
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
})

TEST_YEARS = [2013, 2014, 2015, 2016, 2017, 2018]
VORTEX_START, VORTEX_END = "2014-01-05", "2014-01-09"


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def walk_forward_backtest(feat_df: pd.DataFrame):
    rows = []
    all_preds = {}  # year -> dataframe of Datetime, actual, naive, model_a, model_b

    for year in TEST_YEARS:
        train = feat_df[feat_df["Datetime"] < f"{year}-01-01"].dropna(subset=FEATURES_WITH_WEATHER + ["PJME_MW"])
        test = feat_df[(feat_df["Datetime"] >= f"{year}-01-01") & (feat_df["Datetime"] < f"{year + 1}-01-01")].dropna(
            subset=FEATURES_WITH_WEATHER + ["PJME_MW"])
        if len(train) < 24 * 30 or len(test) == 0:
            continue

        model_a = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.08, max_iter=250, random_state=42)
        model_a.fit(train[FEATURES_NO_WEATHER], train["PJME_MW"])

        model_b = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.08, max_iter=250, random_state=42)
        model_b.fit(train[FEATURES_WITH_WEATHER], train["PJME_MW"])

        pred_naive = test["lag_168"].values
        pred_a = model_a.predict(test[FEATURES_NO_WEATHER])
        pred_b = model_b.predict(test[FEATURES_WITH_WEATHER])
        actual = test["PJME_MW"].values

        rows.append({
            "year": year, "n_test_hours": len(test),
            "naive_mae": mae(actual, pred_naive), "naive_mape": mape(actual, pred_naive),
            "model_a_mae": mae(actual, pred_a), "model_a_mape": mape(actual, pred_a),
            "model_b_mae": mae(actual, pred_b), "model_b_mape": mape(actual, pred_b),
        })

        all_preds[year] = pd.DataFrame({
            "Datetime": test["Datetime"].values, "actual": actual,
            "naive": pred_naive, "model_a": pred_a, "model_b": pred_b,
        })
        print(f"[{year}] naive MAPE={rows[-1]['naive_mape']:.2f}%  "
              f"model_a MAPE={rows[-1]['model_a_mape']:.2f}%  model_b MAPE={rows[-1]['model_b_mape']:.2f}%")

    return pd.DataFrame(rows), all_preds


def vortex_stress_test(all_preds: dict):
    if 2014 not in all_preds:
        return None
    df2014 = all_preds[2014]
    df2014["Datetime"] = pd.to_datetime(df2014["Datetime"])
    vortex = df2014[(df2014["Datetime"] >= VORTEX_START) & (df2014["Datetime"] < VORTEX_END)]
    rest = df2014[~((df2014["Datetime"] >= VORTEX_START) & (df2014["Datetime"] < VORTEX_END))]

    def block(name, d):
        return {
            "period": name,
            "naive_mape": mape(d["actual"], d["naive"]),
            "model_a_mape": mape(d["actual"], d["model_a"]),
            "model_b_mape": mape(d["actual"], d["model_b"]),
        }

    return pd.DataFrame([block("Polar Vortex (Jan 5-9, 2014)", vortex),
                          block("Rest of 2014", rest)])


def plot_backtest_by_year(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(metrics))
    w = 0.25
    ax.bar(x - w, metrics["naive_mape"], width=w, color="#c3c2b7", label="Naive (same hour, last week)")
    ax.bar(x, metrics["model_a_mape"], width=w, color=C_BLUE, label="Model A (lag + calendar)")
    ax.bar(x + w, metrics["model_b_mape"], width=w, color=C_ORANGE, label="Model B (+ temperature)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["year"].astype(str))
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Walk-forward backtest: MAPE by test year", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison_by_year.png", dpi=140)
    plt.close(fig)


def plot_sample_week(all_preds: dict):
    df = all_preds[2018]
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    week = df[(df["Datetime"] >= "2018-03-05") & (df["Datetime"] < "2018-03-12")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(week["Datetime"], week["actual"], color=INK, lw=1.8, label="Actual")
    ax.plot(week["Datetime"], week["naive"], color="#c3c2b7", lw=1.2, linestyle="--", label="Naive")
    ax.plot(week["Datetime"], week["model_b"], color=C_ORANGE, lw=1.4, label="Model B (best model)")
    ax.set_ylabel("Load (MW)")
    ax.set_title("Sample week, held-out 2018 test set: actual vs. forecast", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS / "actual_vs_predicted_sample_week.png", dpi=140)
    plt.close(fig)


def plot_vortex_stress(stress_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(stress_df))
    w = 0.25
    ax.bar(x - w, stress_df["naive_mape"], width=w, color="#c3c2b7", label="Naive")
    ax.bar(x, stress_df["model_a_mape"], width=w, color=C_BLUE, label="Model A (no weather)")
    ax.bar(x + w, stress_df["model_b_mape"], width=w, color=C_ORANGE, label="Model B (+ temperature)")
    ax.set_xticks(x)
    ax.set_xticklabels(stress_df["period"], fontsize=9)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Extreme-event stress test: Polar Vortex week vs. rest of 2014", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "polar_vortex_stress_test.png", dpi=140)
    plt.close(fig)


def plot_feature_importance(model, X_sample, y_sample, feature_names):
    # HistGradientBoostingRegressor has no built-in feature_importances_, so
    # use permutation importance (model-agnostic: shuffle each feature and
    # measure the drop in accuracy) on a held-out-ish sample for speed.
    result = permutation_importance(model, X_sample, y_sample, n_repeats=5,
                                     random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error")
    importances = result.importances_mean
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[order], np.array(importances)[order], color=C_AQUA)
    ax.set_xlabel("Increase in MAE when feature is shuffled (permutation importance)")
    ax.set_title("Model B feature importance (final full-history fit)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_importance.png", dpi=140)
    plt.close(fig)


def plot_quantile_interval(feat_df, q10_model, q90_model, point_model):
    week = feat_df[(feat_df["Datetime"] >= "2014-01-01") & (feat_df["Datetime"] < "2014-01-15")].dropna(
        subset=FEATURES_WITH_WEATHER + ["PJME_MW"])
    X = week[FEATURES_WITH_WEATHER]
    lo = q10_model.predict(X)
    hi = q90_model.predict(X)
    point = point_model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(week["Datetime"], lo, hi, color=C_BLUE, alpha=0.18, label="10th-90th percentile interval")
    ax.plot(week["Datetime"], week["PJME_MW"], color=INK, lw=1.6, label="Actual")
    ax.plot(week["Datetime"], point, color=C_ORANGE, lw=1.2, linestyle="--", label="Point forecast (Model B)")
    ax.axvspan(pd.Timestamp(VORTEX_START), pd.Timestamp(VORTEX_END), color=C_YELLOW, alpha=0.15)
    ax.set_ylabel("Load (MW)")
    ax.set_title("Quantile interval forecast, including the Jan 2014 Polar Vortex", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS / "quantile_interval_forecast.png", dpi=140)
    plt.close(fig)

    coverage = float(np.mean((week["PJME_MW"] >= lo) & (week["PJME_MW"] <= hi)) * 100)
    return coverage


def main():
    df = pd.read_csv(CLEAN, parse_dates=["Datetime"])
    has_weather = TEMP.exists()
    feat_df = build_features(df, with_weather=has_weather)

    metrics, all_preds = walk_forward_backtest(feat_df)
    metrics.to_csv(RESULTS / "backtest_metrics_by_year.csv", index=False)
    plot_backtest_by_year(metrics)
    plot_sample_week(all_preds)

    stress_df = vortex_stress_test(all_preds)
    if stress_df is not None:
        stress_df.to_csv(RESULTS / "polar_vortex_stress_test.csv", index=False)
        plot_vortex_stress(stress_df)

    # --- Final deployment-ready model: fit on ALL available history ---
    full_train = feat_df.dropna(subset=FEATURES_WITH_WEATHER + ["PJME_MW"])
    final_model_b = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.08, max_iter=300, random_state=42)
    final_model_b.fit(full_train[FEATURES_WITH_WEATHER], full_train["PJME_MW"])
    joblib.dump(final_model_b, MODELS / "model_b_final.joblib")
    sample = full_train.sample(n=min(8000, len(full_train)), random_state=42)
    plot_feature_importance(final_model_b, sample[FEATURES_WITH_WEATHER], sample["PJME_MW"], FEATURES_WITH_WEATHER)

    q10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, max_depth=8,
                                         learning_rate=0.08, max_iter=250, random_state=42)
    q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_depth=8,
                                         learning_rate=0.08, max_iter=250, random_state=42)
    q10.fit(full_train[FEATURES_WITH_WEATHER], full_train["PJME_MW"])
    q90.fit(full_train[FEATURES_WITH_WEATHER], full_train["PJME_MW"])
    joblib.dump(q10, MODELS / "model_q10.joblib")
    joblib.dump(q90, MODELS / "model_q90.joblib")
    coverage = plot_quantile_interval(feat_df, q10, q90, final_model_b)

    overall = {
        "avg_naive_mape": float(metrics["naive_mape"].mean()),
        "avg_model_a_mape": float(metrics["model_a_mape"].mean()),
        "avg_model_b_mape": float(metrics["model_b_mape"].mean()),
        "model_b_improvement_vs_naive_pct": float(
            100 * (1 - metrics["model_b_mape"].mean() / metrics["naive_mape"].mean())),
        "model_b_improvement_vs_model_a_pct": float(
            100 * (1 - metrics["model_b_mape"].mean() / metrics["model_a_mape"].mean())),
        "quantile_10_90_coverage_pct_jan2014_window": coverage,
    }
    with open(RESULTS / "overall_summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    print("\n=== OVERALL SUMMARY ===")
    for k, v in overall.items():
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    main()
