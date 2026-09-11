"""
03_eda.py

Exploratory analysis on the cleaned PJM load series. Produces:
  - plots/seasonal_decomposition.png   (manual trend/seasonal/residual split;
                                         statsmodels isn't installable in this
                                         sandbox, so decomposition is hand-rolled
                                         via rolling means -- documented below)
  - plots/load_duration_curve.png
  - plots/weekday_weekend_profile.png
  - plots/monthly_boxplot.png
  - plots/polar_vortex_case_study.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CLEAN = BASE / "data" / "processed" / "pjm_load_clean.csv"
PLOTS = BASE / "plots"
PLOTS.parent.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# dataviz-skill categorical palette (validated, colorblind-safe order)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
})


def load():
    df = pd.read_csv(CLEAN, parse_dates=["Datetime"]).set_index("Datetime")
    return df


def seasonal_decomposition(df):
    s = df["PJME_MW"]
    trend = s.rolling(24 * 30, center=True, min_periods=24 * 7).mean()
    detrended = s - trend
    # seasonal = average detrended value by (day-of-week, hour) -- captures the
    # weekly + daily cycle jointly, then tiled back across the full series
    keys = pd.DataFrame({"dow": s.index.dayofweek, "hour": s.index.hour}, index=s.index)
    seasonal = detrended.groupby([keys["dow"], keys["hour"]]).transform("mean")
    residual = s - trend - seasonal

    fig, axes = plt.subplots(4, 1, figsize=(11, 10))
    axes[0].plot(s.index, s.values, color=C_BLUE, lw=0.5)
    axes[0].set_title("Observed hourly load (MW)", loc="left", fontsize=10)
    axes[1].plot(trend.index, trend.values, color=C_ORANGE, lw=1.2)
    axes[1].set_title("Trend (30-day centered rolling mean)", loc="left", fontsize=10)
    axes[0].sharex(axes[1])
    axes[3].plot(residual.index, residual.values, color=MUTED, lw=0.4)
    axes[3].set_title("Residual", loc="left", fontsize=10)
    axes[1].sharex(axes[3])

    # Seasonal panel uses its OWN x-axis (hours into a 3-week window), since
    # this component repeats identically every week and plotting it against
    # the full 2002-2018 date range would compress it into an unreadable smear.
    seasonal_snippet = seasonal.values[:24 * 21]
    axes[2].plot(np.arange(len(seasonal_snippet)), seasonal_snippet, color=C_AQUA, lw=0.9)
    axes[2].set_title("Seasonal component (day-of-week x hour-of-day; 3-week repeating pattern shown on its own hour axis)",
                       loc="left", fontsize=10)
    axes[2].set_xlabel("Hours into a 3-week window")
    for i, ax in enumerate(axes):
        if i != 2:
            ax.margins(x=0.01)
    fig.suptitle("Manual seasonal decomposition (trend / seasonal / residual)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOTS / "seasonal_decomposition.png", dpi=140)
    plt.close(fig)


def load_duration_curve(df):
    sorted_load = np.sort(df["PJME_MW"].values)[::-1]
    pct_time = np.linspace(0, 100, len(sorted_load))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(pct_time, sorted_load, color=C_BLUE, lw=1.6)
    ax.fill_between(pct_time, sorted_load, sorted_load.min(), color=C_BLUE, alpha=0.08)
    ax.set_xlabel("% of hours load is at or above this level")
    ax.set_ylabel("Load (MW)")
    ax.set_title("Load duration curve", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOTS / "load_duration_curve.png", dpi=140)
    plt.close(fig)


def weekday_weekend_profile(df):
    d = df.copy()
    d["hour"] = d.index.hour
    d["is_weekend"] = d.index.dayofweek >= 5
    prof = d.groupby(["is_weekend", "hour"])["PJME_MW"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(prof.index, prof[False], color=C_BLUE, lw=2, label="Weekday")
    ax.plot(prof.index, prof[True], color=C_ORANGE, lw=2, label="Weekend")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average load (MW)")
    ax.set_title("Average daily load profile: weekday vs. weekend", loc="left", fontsize=12)
    ax.legend(frameon=False)
    ax.set_xticks(range(0, 24, 3))
    fig.tight_layout()
    fig.savefig(PLOTS / "weekday_weekend_profile.png", dpi=140)
    plt.close(fig)


def monthly_boxplot(df):
    d = df.copy()
    d["month"] = d.index.month
    data = [d.loc[d["month"] == m, "PJME_MW"].values for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                     medianprops=dict(color=INK, linewidth=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BLUE)
        patch.set_alpha(0.55)
        patch.set_edgecolor(C_BLUE)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Load (MW)")
    ax.set_title("Monthly load distribution (dual seasonal peaks: winter heating + summer cooling)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "monthly_boxplot.png", dpi=140)
    plt.close(fig)


def polar_vortex_case_study(df):
    window = df.loc["2014-01-01":"2014-01-15"]
    baseline_years = pd.concat([
        df.loc["2013-01-01":"2013-01-15", "PJME_MW"].reset_index(drop=True),
        df.loc["2015-01-01":"2015-01-15", "PJME_MW"].reset_index(drop=True),
    ], axis=1).mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(range(len(window)), window["PJME_MW"].values, color=C_ORANGE, lw=1.4,
            label="Jan 2014 (Polar Vortex)")
    ax.plot(range(len(baseline_years)), baseline_years.values, color=C_BLUE, lw=1.2,
            linestyle="--", label="Avg of Jan 2013 & Jan 2015 (typical years)")
    ax.axvspan(4 * 24, 8 * 24, color=C_YELLOW, alpha=0.15, label="Coldest stretch")
    ax.set_xlabel("Hours into January 1-15")
    ax.set_ylabel("Load (MW)")
    ax.set_title("Case study: Jan 2014 Polar Vortex vs. a typical mid-January", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "polar_vortex_case_study.png", dpi=140)
    plt.close(fig)

    peak_vortex = window["PJME_MW"].max()
    peak_typical = baseline_years.max()
    print(f"Polar Vortex peak: {peak_vortex:.0f} MW vs. typical-year peak: {peak_typical:.0f} MW "
          f"(+{100*(peak_vortex/peak_typical - 1):.1f}%)")


def main():
    df = load()
    seasonal_decomposition(df)
    load_duration_curve(df)
    weekday_weekend_profile(df)
    monthly_boxplot(df)
    polar_vortex_case_study(df)
    print(f"Saved 5 plots to {PLOTS}")


if __name__ == "__main__":
    main()
