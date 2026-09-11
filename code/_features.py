"""Shared feature-engineering code used by both the modeling script (04) and
the dashboard data export (05), so the two never drift out of sync."""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TEMP = BASE / "data" / "raw" / "temperature_synthetic.csv"

FEATURES_NO_WEATHER = ["lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_std_24",
                        "roll_mean_168", "hour", "dow", "month", "is_weekend", "is_holiday",
                        "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURES_WITH_WEATHER = FEATURES_NO_WEATHER + ["temperature_f"]


def us_federal_holidays(years):
    dates = set()
    for y in years:
        dates.add(pd.Timestamp(y, 1, 1))
        dates.add(pd.Timestamp(y, 7, 4))
        dates.add(pd.Timestamp(y, 12, 25))
        jan1 = pd.Timestamp(y, 1, 1)
        dates.add(jan1 + pd.Timedelta(days=(7 - jan1.dayofweek) % 7) + pd.Timedelta(weeks=2))
        feb1 = pd.Timestamp(y, 2, 1)
        dates.add(feb1 + pd.Timedelta(days=(7 - feb1.dayofweek) % 7) + pd.Timedelta(weeks=2))
        may31 = pd.Timestamp(y, 5, 31)
        dates.add(may31 - pd.Timedelta(days=may31.dayofweek))
        sep1 = pd.Timestamp(y, 9, 1)
        dates.add(sep1 + pd.Timedelta(days=(7 - sep1.dayofweek) % 7))
        nov1 = pd.Timestamp(y, 11, 1)
        dates.add(nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7) + pd.Timedelta(weeks=3))
    return dates


def build_features(df: pd.DataFrame, with_weather: bool) -> pd.DataFrame:
    d = df.copy()
    s = d["PJME_MW"]
    shifted = s.shift(24)

    d["lag_24"] = shifted
    d["lag_48"] = s.shift(48)
    d["lag_168"] = s.shift(168)
    d["roll_mean_24"] = shifted.rolling(24).mean()
    d["roll_std_24"] = shifted.rolling(24).std()
    d["roll_mean_168"] = shifted.rolling(168).mean()

    d["hour"] = d["Datetime"].dt.hour
    d["dow"] = d["Datetime"].dt.dayofweek
    d["month"] = d["Datetime"].dt.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)

    holidays = us_federal_holidays(range(d["Datetime"].dt.year.min(), d["Datetime"].dt.year.max() + 2))
    d["is_holiday"] = d["Datetime"].dt.normalize().isin(holidays).astype(int)

    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    doy = d["Datetime"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if with_weather:
        temp = pd.read_csv(TEMP, parse_dates=["Datetime"])
        d = d.merge(temp, on="Datetime", how="left")

    return d
