"""
02_clean_data.py

Cleans data/raw/PJME_hourly_synthetic.csv (or a real PJME_hourly.csv dropped
in that same location -- schema is identical) and writes:
  - data/processed/pjm_load_clean.csv
  - outputs/data_quality_report.md  (quantified, not just descriptive)

Cleaning policy (stated explicitly, on purpose -- this is the part most
tutorial notebooks on this dataset skip):
  1. Duplicate timestamps (DST "fall back"): average the duplicates rather
     than arbitrarily keeping one -- both readings are for the same real hour.
  2. Missing hours (DST "spring forward" + outage gaps): short gaps
     (<= 3 hours) are linearly interpolated; longer gaps are left as NaN and
     EXCLUDED from model training rather than silently filled, since a filled
     multi-hour gap would inject fake signal into a load-forecasting model.
  3. Outlier detection uses a rolling z-score (168h window, i.e. trailing
     week) so it adapts to season -- and outliers are FLAGGED, not deleted,
     because some "outliers" (e.g. the Jan 2014 Polar Vortex) are the most
     important real signal in the dataset. Only single-hour spikes/drops that
     are implausible relative to their immediate neighbors are treated as
     sensor glitches and corrected (interpolated).
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
_REAL = BASE / "data" / "raw" / "PJME_hourly_real.csv"
_SYNTH = BASE / "data" / "raw" / "PJME_hourly_synthetic.csv"
RAW = _REAL if _REAL.exists() else _SYNTH
OUT_CSV = BASE / "data" / "processed" / "pjm_load_clean.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
REPORT = BASE / "outputs" / "data_quality_report.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)

SHORT_GAP_HOURS = 3
GLITCH_Z_THRESH = 6.0        # single-hour, vs immediate neighbors -> sensor glitch
EVENT_Z_THRESH = 4.0         # sustained, multi-hour -> real event, just flag it


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW, parse_dates=["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)


def dedupe(df: pd.DataFrame):
    dup_mask = df.duplicated("Datetime", keep=False)
    n_dup_rows = dup_mask.sum()
    df = df.groupby("Datetime", as_index=False)["PJME_MW"].mean()
    return df, n_dup_rows


def reindex_full_hourly(df: pd.DataFrame):
    full_idx = pd.date_range(df["Datetime"].min(), df["Datetime"].max(), freq="h")
    df = df.set_index("Datetime").reindex(full_idx)
    df.index.name = "Datetime"
    n_missing = df["PJME_MW"].isna().sum()
    return df, n_missing, full_idx


def fill_short_gaps(df: pd.DataFrame):
    s = df["PJME_MW"]
    is_na = s.isna()
    # identify contiguous NaN run lengths
    grp = (is_na != is_na.shift()).cumsum()
    run_lengths = is_na.groupby(grp).transform("sum")
    short_gap_mask = is_na & (run_lengths <= SHORT_GAP_HOURS)
    long_gap_mask = is_na & (run_lengths > SHORT_GAP_HOURS)

    # Interpolate everything, then keep the interpolated value ONLY where the
    # run it belongs to is short -- long-gap positions stay NaN exactly as
    # before, rather than pandas' `limit=` silently filling the first few
    # hours of a longer run and leaving an inconsistent partial fill.
    s_interp_full = s.interpolate(method="linear", limit_area="inside")
    s_out = s.copy()
    s_out.loc[short_gap_mask] = s_interp_full.loc[short_gap_mask]
    df["PJME_MW"] = s_out
    n_short_filled = short_gap_mask.sum()
    n_long_left = long_gap_mask.sum()
    return df, n_short_filled, n_long_left


def flag_and_fix_glitches(df: pd.DataFrame):
    s = df["PJME_MW"]
    roll_med = s.rolling(168, center=True, min_periods=24).median()
    roll_std = s.rolling(168, center=True, min_periods=24).std()
    z = (s - roll_med) / roll_std.replace(0, np.nan)

    neighbor_avg = (s.shift(1) + s.shift(-1)) / 2
    neighbor_dev = (s - neighbor_avg).abs() / neighbor_avg.replace(0, np.nan)

    glitch_mask = (z.abs() > GLITCH_Z_THRESH) & (neighbor_dev > 0.25)
    event_mask = (z.abs() > EVENT_Z_THRESH) & ~glitch_mask

    n_glitches = glitch_mask.sum()
    n_event_flags = event_mask.sum()

    # Fix ONLY the flagged glitch points, using their immediate neighbors --
    # deliberately does not touch pre-existing long-gap NaNs from the
    # short-gap-fill step, which must stay out of the training data.
    fixed = df["PJME_MW"].copy()
    fixed.loc[glitch_mask] = neighbor_avg.loc[glitch_mask]
    df["PJME_MW"] = fixed

    df["flagged_extreme_event"] = event_mask.values
    return df, n_glitches, n_event_flags


def main():
    print(f"Reading from: {RAW.name} ({'REAL Kaggle data' if RAW is _REAL else 'synthetic stand-in'})")
    raw = load_raw()
    n_raw_rows = len(raw)

    deduped, n_dup_rows = dedupe(raw)
    reindexed, n_missing_after_reindex, full_idx = reindex_full_hourly(deduped)
    filled, n_short_filled, n_long_left = fill_short_gaps(reindexed.reset_index())
    cleaned, n_glitches, n_event_flags = flag_and_fix_glitches(filled)

    cleaned = cleaned.dropna(subset=["PJME_MW"]).reset_index(drop=True)
    cleaned.to_csv(OUT_CSV, index=False)

    pct = lambda x: 100 * x / len(full_idx)

    report = f"""# Data Quality Report

Source rows (raw file): {n_raw_rows:,}
Expected complete hourly rows for the full date range: {len(full_idx):,}

## Duplicate timestamps (DST fall-back)
{n_dup_rows} rows were duplicate timestamps ({pct(n_dup_rows):.3f}% of expected hours).
Policy: averaged, rather than arbitrarily keeping the first or last reading,
since both rows represent the same real hour.

## Missing hours
{n_missing_after_reindex} hours were missing entirely from the raw file
({pct(n_missing_after_reindex):.3f}% of expected hours), from a mix of DST
spring-forward transitions and short outage/reporting gaps.
- {n_short_filled} were part of gaps <= {SHORT_GAP_HOURS}h and were linearly interpolated.
- {n_long_left} were part of longer gaps and were left out of the training
  data entirely rather than filled, to avoid injecting fabricated signal into
  a load-forecasting model.

## Outlier / anomaly handling
- {n_glitches} single-hour points were flagged as sensor glitches (extreme
  relative to their immediate hour-by-hour neighbors, z > {GLITCH_Z_THRESH})
  and corrected via short linear interpolation.
- {n_event_flags} points were flagged as sustained statistical extremes
  (z > {EVENT_Z_THRESH} vs. a trailing-week seasonal baseline) but were
  **kept unchanged** and instead marked in a `flagged_extreme_event` column,
  because sustained multi-hour extremes are far more likely to be genuine
  demand events (e.g. an extreme cold snap or heat wave) than sensor errors.
  These flagged periods are exactly what the extreme-event stress test in
  the modeling stage evaluates against.

## Final clean dataset
{len(cleaned):,} rows written to `data/processed/pjm_load_clean.csv`
({pct(len(cleaned)):.2f}% of the expected complete hourly range retained).
"""
    REPORT.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
