"""
00_load_real_data.py

Loads the REAL PJM data the user uploaded (est_hourly.paruqet -- the actual
Kaggle "hourly-energy-consumption" merged-region file) using the
dependency-free _mini_parquet reader (no pyarrow/fastparquet available in
this sandbox), extracts the PJME (PJM East) column, and writes it out in the
exact same schema as the real standalone PJME_hourly.csv: `Datetime`, `PJME_MW`.

Also (re)generates a synthetic hourly temperature series aligned to this
exact real date range, used later as a proxy weather feature -- there's no
network access in this sandbox to pull real historical weather, so this one
piece stays synthetic. Everything else downstream (cleaning, EDA, modeling,
backtest) now runs on 100% real demand data.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _mini_parquet as mp

BASE = Path(__file__).resolve().parents[1]
UPLOADED = BASE / "data" / "uploaded" / "est_hourly.paruqet"
RAW_DIR = BASE / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


def load_real_pjme():
    cols = mp.load_parquet_columns(str(UPLOADED), wanted_columns=["Datetime", "PJME"])
    df = pd.DataFrame({
        "Datetime": pd.to_datetime(cols["Datetime"], unit="ms"),
        "PJME_MW": cols["PJME"],
    })
    df = df.dropna(subset=["PJME_MW"]).sort_values("Datetime").reset_index(drop=True)
    return df


def synthetic_temperature(idx: pd.DatetimeIndex) -> np.ndarray:
    """Same generator as the earlier synthetic-data script, aligned to the
    real date range. This is a PROXY feature only -- swap in real historical
    weather (e.g. NOAA / Open-Meteo, Philadelphia area for PJM East) for a
    production version; same `Datetime`, `temperature_f` schema."""
    doy = idx.dayofyear.values.astype(float)
    hour = idx.hour.values.astype(float)
    year_frac = (idx.year.values - idx.year.values.min())
    annual = 55 - 25 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    diurnal = 6 * np.sin(2 * np.pi * (hour - 9) / 24)
    multi_year = 3 * np.sin(2 * np.pi * year_frac / 5.3 + 1.0)
    noise = RNG.normal(0, 3.5, size=len(idx))
    temp = annual + diurnal + multi_year + noise
    vortex_mask = (idx >= "2014-01-05") & (idx < "2014-01-09")
    temp[vortex_mask] -= RNG.uniform(30, 42, size=vortex_mask.sum())
    return temp


def main():
    df = load_real_pjme()
    out_path = RAW_DIR / "PJME_hourly_real.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote REAL PJME data: {len(df):,} rows to {out_path}")
    print(f"Range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"Value range: {df['PJME_MW'].min():.0f} - {df['PJME_MW'].max():.0f} MW")
    print(f"Duplicate timestamps: {df['Datetime'].duplicated().sum()}")

    full_idx = pd.date_range(df["Datetime"].min(), df["Datetime"].max(), freq="h")
    temp = synthetic_temperature(full_idx)
    temp_df = pd.DataFrame({"Datetime": full_idx, "temperature_f": np.round(temp, 1)})
    temp_df.to_csv(RAW_DIR / "temperature_synthetic.csv", index=False)
    print(f"Wrote proxy temperature series: {len(temp_df):,} rows")


if __name__ == "__main__":
    main()
