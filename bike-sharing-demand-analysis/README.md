# 🚲 Bike-Sharing Demand Analysis — Washington, D.C.

Statistical analysis and predictive modeling of Capital Bikeshare hourly rental demand, using two years of data (2011–2012) from the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

This project analyzes how **temporal patterns**, **weather conditions**, and **user type** (casual vs. registered) drive urban micro-mobility demand, and builds a leak-safe forecasting pipeline that predicts hourly rental volume for the last third of every month using only the first two-thirds as training history.

📄 **[Read the full technical memorandum](reports/Bike_Sharing_Demand_Analysis_Memo.docx)**
📓 **[View the analysis notebook](notebooks/bike_sharing_analysis.ipynb)**

---

## Highlights

- **17,379 hourly records**, Jan 2011 – Dec 2012, no missing values or duplicates
- Rigorous statistical workflow: correlation/VIF screening, ANOVA, non-linearity checks, Welch's t-tests, OLS vs. Poisson vs. Negative Binomial regression
- **Leak-safe temporal forecasting pipeline** with a custom train/test partition (days 1–20 train, 21–end test, per month) and a recursive multi-step prediction loop that never touches held-out targets
- **XGBoost vs. statistical baseline**: gradient-boosted trees roughly halve the RMSE of a Negative Binomial GLM (RMSE 59.5 vs. 110.8, R² 0.89 vs. 0.63)

## Key Results

| Model / Feature Set | MAE | RMSE | R² |
|---|---|---|---|
| **Rolling-168h mean (best)** | 37.54 | **59.52** | **0.892** |
| Rolling-24h mean | 36.84 | 59.80 | 0.891 |
| XGBoost, no lag features | 36.98 | 59.88 | 0.891 |
| All rolling features | 37.93 | 60.18 | 0.890 |
| Lag-168h only | 39.85 | 69.05 | 0.855 |
| Lag-24h only | 44.23 | 78.53 | 0.812 |
| Negative Binomial GLM (baseline) | 68.17 | 110.80 | 0.626 |

Full regression tables, ANOVA results, and diagnostic reasoning are in the [memo](reports/Bike_Sharing_Demand_Analysis_Memo.docx).

---

## Project Structure

```
bike-sharing-demand-analysis/
├── notebooks/
│   └── bike_sharing_analysis.ipynb   # Full analysis: EDA, stats, regression, forecasting
├── reports/
│   └── Bike_Sharing_Demand_Analysis_Memo.docx   # Technical memorandum
├── images/                            # Key figures referenced below
├── requirements.txt
└── README.md
```

## Methodology

### 1. Descriptive Statistics & Exploratory Analysis
- Distribution of hourly demand (right-skewed, log-transform applied)
- Casual vs. registered user behavior — distinct commute vs. leisure signatures
- Temporal patterns (hour, weekday, month, season) and weather effects

<p align="center">
  <img src="images/01_demand_distribution.png" width="700" alt="Demand distribution">
  <br><em>Hourly rental demand is right-skewed; log transform symmetrizes it.</em>
</p>

<p align="center">
  <img src="images/02_casual_vs_registered.png" width="700" alt="Casual vs registered">
  <br><em>Registered users show a commute double-peak; casual users peak mid-afternoon.</em>
</p>

<p align="center">
  <img src="images/03_hourly_patterns.png" width="700" alt="Hourly patterns">
  <br><em>Working-day vs. non-working-day demand shapes, and average demand by weekday.</em>
</p>

<p align="center">
  <img src="images/04_weather_effects.png" width="700" alt="Weather effects">
  <br><em>Demand declines as weather worsens; rises with temperature.</em>
</p>

### 2. Statistical & Regression Analysis
- Pearson/Spearman correlation + VIF for multicollinearity screening (temp/atemp near-collinear)
- One-way ANOVA on categorical drivers (hour, weekday, month, season, weather — all p < 0.001)
- Non-linearity checks via quantile-binned linear vs. quadratic fits
- OLS baseline → residual diagnostics reveal heteroscedasticity and non-normality
- Poisson GLM rejected (deviance/df ≈ 117, severe overdispersion) in favor of **Negative Binomial GLM**, interpreted via incidence rate ratios (IRR)

<p align="center">
  <img src="images/05_correlation_vif.png" width="700" alt="Correlation and VIF">
</p>

<p align="center">
  <img src="images/06_nonlinearity.png" width="700" alt="Non-linearity check">
</p>

<p align="center">
  <img src="images/07_ols_diagnostics.png" width="600" alt="OLS diagnostics">
  <br><em>OLS residuals show heteroscedasticity and heavy tails — motivating the NB-GLM.</em>
</p>

### 3. Demand Forecasting & Predictive Modeling
- **Partitioning:** days 1–20 of each month → train; days 21–end → test (applied across both years)
- **Leakage control:** temporal (lag/rolling) features built from a target series with test-period values masked to `NaN`; predictions generated with a **recursive multi-step forecast** so later test-hour features are built only from real training history or the model's own prior predictions — never from held-out ground truth
- **Models compared:** Negative Binomial GLM baseline vs. XGBoost (500 trees) across 7 feature-set experiments (no lags, 24h/168h lags, 24h/168h rolling means, combinations)

<p align="center">
  <img src="images/08_forecast_results.png" width="700" alt="Forecast results">
  <br><em>Actual vs. top-3 model predictions on a held-out test block (recursive, leak-safe).</em>
</p>

---

## Getting Started

```bash
git clone https://github.com/<your-username>/bike-sharing-demand-analysis.git
cd bike-sharing-demand-analysis
pip install -r requirements.txt
jupyter notebook notebooks/bike_sharing_analysis.ipynb
```

The dataset (`hour.csv`, `day.csv`) is not included — download it from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) and point the notebook's `DATA_PATH` at the folder containing it.

## Tech Stack

`pandas` · `numpy` · `matplotlib` / `seaborn` · `statsmodels` (OLS, Poisson & Negative Binomial GLM, VIF) · `scipy.stats` (ANOVA, Welch's t-test) · `scikit-learn` (metrics) · `xgboost`

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
