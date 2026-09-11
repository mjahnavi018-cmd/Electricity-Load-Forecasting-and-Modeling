# PJM-East-Time-series-Forecasting

### Can a model beat "same hour, last week"?

A complete **day-ahead electricity demand forecasting project** using real **PJM East (PJME) hourly load data from 2002–2018**. The project compares a machine-learning model against the simple seasonal-naive approach of predicting demand using the same hour from the previous week.

## Problem

Electricity demand changes with time, season, weekdays, holidays, and extreme events. The objective was to develop a forecasting model that could provide more accurate and robust day-ahead predictions than the traditional **same-hour-last-week** baseline.

## Project Workflow

```text
Raw PJM Data
     ↓
Data Extraction & Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Baseline Forecast
     ↓
Gradient Boosting Model
     ↓
Walk-Forward Backtesting
     ↓
Polar Vortex Stress Test
     ↓
Uncertainty Analysis
     ↓
Interactive Dashboard
```

## Data & Preprocessing

* Real PJM East hourly electricity demand data.
* Period: **2002-01-01 to 2018-08-03**
* 145,366 raw observations.
* Duplicate timestamps and missing hours were identified and handled.
* Missing gaps of up to 3 hours were linearly interpolated.
* Final cleaned dataset: **145,392 hourly observations**.
* A 168-hour rolling z-score method was used for anomaly detection.

## Modelling

Three forecasting approaches were evaluated:

1. **Seasonal-naive baseline** — same hour one week earlier.
2. **Model A** — `HistGradientBoostingRegressor` using lag, rolling, calendar and cyclical features.
3. **Model B** — Model A with an additional synthetic temperature feature.

The models were evaluated using **walk-forward backtesting** on six independent years (2013–2018), ensuring that future data was not used during training.

## Results

| Model          | Average MAPE |
| -------------- | -----------: |
| Naive baseline |   **10.29%** |
| Model A        |    **5.43%** |
| Model B        |    **5.43%** |

The gradient-boosted model reduced forecasting error by approximately **47%** compared with the naive baseline.

### Polar Vortex Stress Test

During the **January 2014 Polar Vortex**:

| Model          |       MAPE |
| -------------- | ---------: |
| Naive baseline | **16.96%** |
| Model A        |  **9.10%** |
| Model B        |  **9.18%** |

The model remained substantially more robust than the naive baseline during the extreme-demand period.

## Uncertainty Analysis

10th and 90th percentile quantile models were developed to estimate forecast uncertainty. During the Polar Vortex, the interval achieved **67% coverage** against an 80% target, showing that uncertainty was underestimated during the extreme event.

## Dashboard

A self-contained interactive dashboard was developed to present:

* historical electricity demand,
* model predictions,
* yearly backtesting results,
* actual vs. predicted demand,
* Polar Vortex stress testing,
* uncertainty intervals, and
* feature importance.

## Key Limitation

The temperature feature used in Model B is **synthetic**, not historical weather data. It produced essentially no improvement over Model A. Using real historical weather data is therefore the most important next improvement.

## Project Structure

```text
pjm-load-forecast/
├── data/
├── scripts/
├── plots/
├── outputs/
└── artifact/
    └── dashboard.html
```

The complete pipeline covers **data extraction → cleaning → EDA → feature engineering → modelling → validation → stress testing → uncertainty analysis → dashboard development**.

## Conclusion

The project demonstrates that gradient-boosted machine learning can significantly outperform the **same-hour-last-week** baseline for PJM East day-ahead electricity-load forecasting, while also providing a more robust forecast during an extreme event such as the January 2014 Polar Vortex.
