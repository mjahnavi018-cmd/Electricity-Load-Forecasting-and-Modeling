"""
05_export_dashboard_data.py

Exports everything the interactive HTML dashboard needs as one compact JSON
file: headline metrics, backtest-by-year, the Polar Vortex stress test, a
sample forecast week, the quantile interval week, feature importance, and a
daily-aggregated 16-year overview series (kept small for the browser).
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _features import build_features, FEATURES_NO_WEATHER, FEATURES_WITH_WEATHER, TEMP
from sklearn.inspection import permutation_importance
import joblib

BASE = Path(__file__).resolve().parents[1]
CLEAN = BASE / "data" / "processed" / "pjm_load_clean.csv"
RESULTS = BASE / "outputs" / "results"
MODELS = BASE / "outputs" / "models"
DASH_DATA = BASE / "outputs" / "dashboard_data.json"


def round_list(arr, nd=1):
    return [None if pd.isna(v) else round(float(v), nd) for v in arr]


def main():
    df = pd.read_csv(CLEAN, parse_dates=["Datetime"])
    feat_df = build_features(df, with_weather=TEMP.exists())

    # --- daily overview (16 years hourly -> daily mean) ---
    daily = df.set_index("Datetime")["PJME_MW"].resample("D").mean().dropna()
    overview = {"dates": daily.index.strftime("%Y-%m-%d").tolist(), "values": round_list(daily.values, 0)}

    metrics = pd.read_csv(RESULTS / "backtest_metrics_by_year.csv")
    backtest_by_year = {
        "year": metrics["year"].astype(int).tolist(),
        "naive_mape": round_list(metrics["naive_mape"], 2),
        "model_a_mape": round_list(metrics["model_a_mape"], 2),
        "model_b_mape": round_list(metrics["model_b_mape"], 2),
    }

    stress = pd.read_csv(RESULTS / "polar_vortex_stress_test.csv")
    stress_test = {
        "period": stress["period"].tolist(),
        "naive_mape": round_list(stress["naive_mape"], 2),
        "model_a_mape": round_list(stress["model_a_mape"], 2),
        "model_b_mape": round_list(stress["model_b_mape"], 2),
    }

    with open(RESULTS / "overall_summary.json") as f:
        overall = json.load(f)

    quality_text = (BASE / "outputs" / "data_quality_report.md").read_text()

    # --- sample week (held-out 2018 test period) ---
    model_b = joblib.load(MODELS / "model_b_final.joblib")
    week = feat_df[(feat_df["Datetime"] >= "2018-03-05") & (feat_df["Datetime"] < "2018-03-12")].dropna(
        subset=FEATURES_WITH_WEATHER + ["PJME_MW"])
    sample_week = {
        "dates": week["Datetime"].dt.strftime("%Y-%m-%dT%H:%M").tolist(),
        "actual": round_list(week["PJME_MW"], 0),
        "naive": round_list(week["lag_168"], 0),
        "model_b": round_list(model_b.predict(week[FEATURES_WITH_WEATHER]), 0),
    }

    # --- quantile interval week (Jan 2014, includes the Polar Vortex) ---
    q10 = joblib.load(MODELS / "model_q10.joblib")
    q90 = joblib.load(MODELS / "model_q90.joblib")
    vweek = feat_df[(feat_df["Datetime"] >= "2014-01-01") & (feat_df["Datetime"] < "2014-01-15")].dropna(
        subset=FEATURES_WITH_WEATHER + ["PJME_MW"])
    quantile_week = {
        "dates": vweek["Datetime"].dt.strftime("%Y-%m-%dT%H:%M").tolist(),
        "actual": round_list(vweek["PJME_MW"], 0),
        "point": round_list(model_b.predict(vweek[FEATURES_WITH_WEATHER]), 0),
        "lo": round_list(q10.predict(vweek[FEATURES_WITH_WEATHER]), 0),
        "hi": round_list(q90.predict(vweek[FEATURES_WITH_WEATHER]), 0),
        "vortex_start": "2014-01-05T00:00",
        "vortex_end": "2014-01-09T00:00",
    }

    # --- feature importance ---
    full_train = feat_df.dropna(subset=FEATURES_WITH_WEATHER + ["PJME_MW"])
    sample = full_train.sample(n=min(6000, len(full_train)), random_state=42)
    imp = permutation_importance(model_b, sample[FEATURES_WITH_WEATHER], sample["PJME_MW"],
                                  n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error")
    order = np.argsort(imp.importances_mean)[::-1]
    feature_importance = {
        "features": [FEATURES_WITH_WEATHER[i] for i in order],
        "importance": round_list(imp.importances_mean[order], 1),
    }

    payload = {
        "overview": overview,
        "backtest_by_year": backtest_by_year,
        "stress_test": stress_test,
        "overall": overall,
        "sample_week": sample_week,
        "quantile_week": quantile_week,
        "feature_importance": feature_importance,
        "data_quality_text": quality_text,
        "date_range": {"start": str(df["Datetime"].min()), "end": str(df["Datetime"].max()),
                        "n_rows": int(len(df))},
    }
    with open(DASH_DATA, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {DASH_DATA} ({DASH_DATA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
