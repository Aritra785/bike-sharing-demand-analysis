# Bike-Sharing Demand Analysis: Temporal, Weather, and User-Type Drivers in Washington, D.C.

*A statistical and machine-learning investigation of Capital Bikeshare hourly rental data (2011–2012)*

📄 [Full technical memorandum](reports/Bike_Sharing_Demand_Analysis_Memo.docx) · 📓 [Analysis notebook](notebooks/bike_sharing_analysis.ipynb)

---

# Capital Bikeshare Demand Analysis & Forecasting

## Technical Assessment — Part 1

An end-to-end analysis of hourly Capital Bikeshare demand combining **exploratory data analysis, statistical inference, regression diagnostics, leakage-aware temporal feature engineering, and machine-learning-based demand forecasting**.

The project investigates how temporal, calendar, weather, and historical-demand patterns influence hourly bicycle rentals and evaluates forecasting models under a chronological train/test protocol.

---

## Project Overview

This project uses the **UCI Bike Sharing Dataset** to analyze hourly bicycle rental demand over a two-year period.

The analysis is structured around three objectives:

1. **Descriptive & Exploratory Analysis**

   * Examine the distribution of hourly rental demand.
   * Compare casual and registered users.
   * Identify hourly, weekly, monthly, seasonal, and yearly patterns.
   * Investigate the relationship between rental demand and weather conditions.

2. **Statistical & Regression Analysis**

   * Assess relationships between continuous variables and rental demand.
   * Investigate multicollinearity using correlation analysis and VIF.
   * Test categorical predictors using one-way ANOVA.
   * Evaluate binary predictors using Welch's t-test.
   * Examine non-linearity using linear vs. quadratic models.
   * Diagnose OLS and Poisson regression assumptions.
   * Establish a Negative Binomial GLM as a count-regression baseline.

3. **Demand Forecasting & Predictive Modeling**

   * Construct a leakage-aware temporal train/test split.
   * Engineer historical demand features.
   * Compare XGBoost models using different temporal feature sets.
   * Evaluate models using MAE, RMSE, and R².
   * Perform recursive multi-step forecasting without using future test-set target values.

---

# Dataset

The project uses the **UCI Bike Sharing Dataset**, specifically the hourly rental data.

The hourly dataset contains:

* **17,379 observations**
* **17 original fields**
* Data spanning **2011-01-01 to 2012-12-31**
* No missing values
* No duplicate rows

The target variable is:

```text
cnt = casual + registered
```

### Main variable groups

| Group     | Variables                                         |
| --------- | ------------------------------------------------- |
| Temporal  | `dteday`, `hr`, `weekday`, `mnth`, `yr`           |
| Calendar  | `season`, `holiday`, `workingday`                 |
| Weather   | `weathersit`, `temp`, `atemp`, `hum`, `windspeed` |
| User type | `casual`, `registered`                            |
| Target    | `cnt`                                             |

The project verifies that the total rental count is exactly the sum of casual and registered rentals.

---

# Workflow

```text
Raw Bike Sharing Dataset
          │
          ▼
Data Loading & Quality Checks
          │
          ▼
Descriptive / Exploratory Analysis
          │
          ├── Demand Distribution
          ├── User Behaviour
          ├── Temporal Patterns
          └── Weather Patterns
          │
          ▼
Statistical Analysis
          │
          ├── Pearson / Spearman Correlation
          ├── VIF Multicollinearity
          ├── ANOVA
          ├── Welch's t-test
          └── Non-linearity Analysis
          │
          ▼
Regression Diagnostics
          │
          ├── OLS
          ├── Poisson
          └── Negative Binomial GLM
          │
          ▼
Leakage-Aware Temporal Split
          │
          ├── Days 1–20 → Training
          └── Days 21–end → Testing
          │
          ▼
Temporal Feature Engineering
          │
          ├── 24-hour lag
          ├── 168-hour lag
          ├── 24-hour rolling mean
          └── 168-hour rolling mean
          │
          ▼
XGBoost Forecasting
          │
          ├── Base features
          ├── Lag experiments
          └── Rolling-window experiments
          │
          ▼
Recursive Multi-Step Forecasting
          │
          ▼
MAE / RMSE / R² Evaluation
          │
          ▼
Actual vs. Predicted Analysis
```

---

# 1. Data Loading and Quality Assessment

The analysis begins by loading the hourly and daily datasets and inspecting their structure.

The following checks are performed:

* Dataset dimensions
* Column names
* Data types
* Missing values
* Duplicate observations
* Descriptive statistics
* Date range

The date field is converted to a proper datetime representation before temporal analysis.

---

# 2. Exploratory Data Analysis

## 2.1 Distribution of Rental Demand

The distribution of `cnt` is examined using both the original and log-transformed rental counts.

The original hourly rental demand is strongly right-skewed. A `log1p` transformation reduces the skewness but does not completely normalize the distribution.

This provides an early indication that a simple linear model may not adequately represent the response distribution.

### Figure to add

**Figure 1 — Hourly rental demand — raw vs. log-transformed**

```text
![Hourly rental demand](images/01_demand_distribution.png)
```

---

## 2.2 Casual vs. Registered Users

The analysis separates demand into:

* Casual users
* Registered users

Average hourly demand reveals substantially different behavioral patterns.

Registered-user demand shows a pronounced bimodal pattern around typical commuting hours, while casual-user demand follows a more gradual leisure-oriented pattern.

The analysis also compares the two user groups between working and non-working days.

### Figure to add

**Figure 2 — Average hourly rentals by user type and total rentals by user type**

```text
![Casual vs registered users](images/casual-vs-registered.png)
```

---

# 3. Temporal Demand Patterns

Several temporal dimensions are investigated:

* Hour of day
* Working-day status
* Day of week
* Month
* Season
* Year

## Hourly Pattern

Working days show a distinct bimodal demand structure, while non-working days show a broader midday peak.

## Monthly and Seasonal Pattern

Average demand generally increases toward the warmer months and decreases toward the end of the year.

Seasonal demand also differs substantially, with higher demand observed during summer and fall relative to spring.

## Yearly Trend

Daily aggregate demand shows an overall upward trend with recurring seasonal variation. Demand in 2012 is also generally higher than in 2011.

### Figures to add

**Figure 3 — Hourly demand by working-day status and average demand by weekday**

```text
![Temporal hourly patterns](images/workingday-weekday-patterns.png)
```

**Figure 4 — Average demand by month and rental demand by season**

```text
![Monthly and seasonal demand](images/month-season-patterns.png)
```

**Figure 5 — Daily total rental demand and demand by year**

```text
![Long-term demand trend](images/daily-yearly-demand.png)
```

---

# 4. Weather Effects

The following weather variables are investigated:

* Weather situation
* Temperature
* Humidity
* Windspeed

Rental demand decreases across increasingly adverse weather conditions.

Temperature shows a positive but nonlinear relationship with demand, while humidity exhibits a negative relationship.

Windspeed shows comparatively weak visual association with rental demand.

### Figures to add

**Figure 6 — Rental demand by weather condition and temperature**

```text
![Weather and temperature](images/weather-temperature.png)
```

**Figure 7 — Rental demand vs. humidity and windspeed**

```text
![Humidity and windspeed](images/humidity-windspeed.png)
```

---

# 5. Statistical Analysis

## 5.1 Correlation and Multicollinearity

Both **Pearson** and **Spearman** correlations are calculated for:

```text
temp
atemp
hum
windspeed
cnt
```

Variance Inflation Factor (VIF) is then used to investigate multicollinearity.

A major finding is the extremely high correlation between:

```text
temp
atemp
```

with a correlation of approximately **0.99**.

Both variables also produce very high VIF values.

Therefore, `atemp` is excluded from the forecasting feature set because it carries almost redundant information with `temp`.

### Figure to add

**Figure 8 — Pearson correlation, Spearman correlation, and VIF diagnostics**

```text
![Correlation and VIF diagnostics](images/correlation-vif.png)
```

---

# 6. Categorical Variable Analysis

One-way ANOVA is used to determine whether mean rental demand differs significantly across categories.

| Variable          | F-statistic | p-value |
| ----------------- | ----------: | ------: |
| Hour of Day       |      759.09 | < 0.001 |
| Weekday           |        3.49 | < 0.001 |
| Month             |      128.10 | < 0.001 |
| Season            |      409.18 | < 0.001 |
| Weather Situation |      127.17 | < 0.001 |

All evaluated categorical variables show statistically significant differences in rental demand.

The largest between-group variation occurs for:

* Hour of day
* Season

while weekday shows a comparatively smaller effect.

---

# 7. Non-Linearity Analysis

The continuous weather variables are evaluated using both linear and quadratic regression models.

| Variable    | Linear R² | Quadratic R² | Linear AIC | Quadratic AIC |
| ----------- | --------: | -----------: | ---------: | ------------: |
| Temperature |    0.1638 |       0.1640 |  226976.45 |     226974.74 |
| Humidity    |    0.1043 |       0.1054 |  228172.44 |     228152.01 |
| Windspeed   |    0.0087 |       0.0122 |  229934.45 |     229875.31 |

The weather variables have relatively weak standalone explanatory power.

Although quadratic terms provide small improvements in R² and AIC, the improvement is marginal.

This indicates that nonlinear transformations of weather variables alone are unlikely to explain the majority of demand variation.

### Figure to add

**Figure 9 — Mean rental count across temperature, humidity, and windspeed bins**

```text
![Non-linearity analysis](images/nonlinearity-analysis.png)
```

---

# 8. Binary Predictor Analysis

Welch's t-test is used for:

* `workingday`
* `holiday`
* `yr`

The results show statistically significant differences in mean rental demand between the corresponding groups.

These variables are therefore retained as candidate predictors for forecasting.

---

# 9. Regression Diagnostics

Three statistical modeling perspectives are considered:

### Ordinary Least Squares

OLS is initially evaluated as a conventional regression baseline.

Residual diagnostics indicate:

* Heteroscedasticity
* Non-normal residual behavior

Therefore, the assumptions required for conventional OLS inference are not adequately satisfied.

### Poisson Regression

Because `cnt` is a count response, a Poisson model is also considered.

However, the response exhibits substantial overdispersion, making the standard Poisson variance assumption inappropriate.

### Negative Binomial GLM

A **Negative Binomial GLM** is therefore used as the statistical count-regression baseline.

The model uses the base temporal, calendar, and weather features.

### Figure to add

**Figure 10 — OLS residuals vs. fitted values and Q-Q plot**

```text
![Regression diagnostics](images/ols-diagnostics.png)
```

---

# 10. Forecasting Feature Selection

The forecasting target is:

```python
target = "cnt"
```

The base feature set is:

```python
base_features = [
    "hr",
    "workingday",
    "weekday",
    "mnth",
    "season",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
    "holiday",
    "yr"
]
```

The following variables are excluded:

```python
excluded_features = [
    "casual",
    "registered",
    "atemp"
]
```

### Why?

`casual` and `registered` are components of the target:

```text
cnt = casual + registered
```

Using them as predictors would directly expose the target components and create target leakage.

`atemp` is excluded because of its extremely high multicollinearity with `temp`.

---

# 11. Leakage-Aware Temporal Train/Test Split

A chronological partition is used rather than a random train/test split.

For **every month**:

```text
Days 1–20  → Training
Days 21–end → Testing
```

The resulting dataset contains approximately:

```text
Training observations: 11,460
Testing observations:   5,919
```

This setup evaluates the models on later observations while preserving the temporal structure of the forecasting problem.

Importantly, the test period is never randomly mixed into the training period.

---

# 12. Temporal Feature Engineering

Historical demand features are introduced to provide the model with information about recent and weekly demand behavior.

### 24-hour lag

```text
lag_24h
```

Represents rental demand at the same hour on the previous day.

### 168-hour lag

```text
lag_168h
```

Represents rental demand at the same hour one week earlier.

### 24-hour rolling mean

```text
roll_mean_24h
```

Represents the average demand over the preceding 24 hours.

### 168-hour rolling mean

```text
roll_mean_168h
```

Represents the average demand over the preceding seven days.

---

# 13. Leakage Prevention During Temporal Feature Construction

A key part of the forecasting pipeline is preventing future information from entering the model.

Training-side temporal features are constructed only from observations that are available before the prediction point.

During test forecasting:

* Actual test-period target values are not used as future inputs.
* Previously predicted values are recursively inserted into the demand history.
* Each subsequent forecast therefore depends only on information available up to that point.

Conceptually:

```text
Known historical observations
          │
          ▼
Forecast t
          │
          ▼
Insert prediction at t
          │
          ▼
Forecast t+1
          │
          ▼
Insert prediction at t+1
          │
          ▼
Continue recursively
```

This produces a more realistic multi-step forecasting evaluation than using actual future test targets to construct lag features.

---

# 14. Models

## Negative Binomial GLM

The Negative Binomial model serves as the statistical baseline.

It uses the base feature set and is trained only on the training observations.

---

## XGBoost Regressor

The predictive forecasting model is an `XGBRegressor`.

Configuration:

```python
n_estimators = 500
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
objective = "reg:squarederror"
random_state = 42
n_jobs = -1
```

The same XGBoost configuration is used across the temporal-feature experiments.

---

# 15. Forecasting Experiments

Seven XGBoost feature configurations are evaluated:

| Experiment       | Additional Temporal Features      |
| ---------------- | --------------------------------- |
| XGBoost Base     | None                              |
| Lag-24 only      | `lag_24h`                         |
| Lag-168 only     | `lag_168h`                        |
| Rolling-24 only  | `roll_mean_24h`                   |
| Rolling-168 only | `roll_mean_168h`                  |
| All Lags         | `lag_24h`, `lag_168h`             |
| All Rolling      | `roll_mean_24h`, `roll_mean_168h` |

This experiment design allows the contribution of short-term and weekly historical information to be evaluated independently.

---

# 16. Evaluation Metrics

Three complementary metrics are used.

### Mean Absolute Error

```text
MAE = mean(|y - ŷ|)
```

MAE measures the average absolute forecasting error in rental units.

### Root Mean Squared Error

```text
RMSE = sqrt(mean((y - ŷ)²))
```

RMSE gives greater weight to larger forecasting errors.

### R²

```text
R² = 1 - SS_res / SS_tot
```

R² measures the proportion of variance explained by the predictions.

---

# 17. Results

The final test-set results are:

| Model / Feature Set   |         MAE |        RMSE |         R² |
| --------------------- | ----------: | ----------: | ---------: |
| **Rolling-168 only**  | **37.5429** | **59.5189** | **0.8920** |
| Rolling-24 only       |     36.8419 |     59.8032 |     0.8909 |
| XGBoost Base          |     36.9810 |     59.8848 |     0.8906 |
| All Rolling (24+168)  |     37.9289 |     60.1755 |     0.8896 |
| Lag-168 only          |     39.8493 |     69.0463 |     0.8546 |
| Lag-24 only           |     44.2258 |     78.5319 |     0.8119 |
| All Lags (24+168)     |     46.5679 |     81.8832 |     0.7956 |
| Negative Binomial GLM |     68.1717 |    110.8033 |     0.6256 |

### Main findings

The forecasting experiments show several important patterns:

1. **XGBoost substantially improves upon the Negative Binomial GLM baseline.**

2. The best RMSE is obtained by the **168-hour rolling feature**, with:

```text
MAE  = 37.5429
RMSE = 59.5189
R²   = 0.8920
```

3. The base XGBoost model is already highly competitive:

```text
MAE  = 36.9810
RMSE = 59.8848
R²   = 0.8906
```

4. Rolling-window features perform better than single-point lag features in terms of RMSE.

5. Adding both 24-hour and 168-hour lag features does not improve forecasting performance. In fact, the combined lag model produces substantially larger errors.

6. The relatively small difference between the base XGBoost model and the rolling-feature models suggests that much of the predictable demand structure is already captured by the hour, calendar, weather, and year variables.

---

# 18. Actual vs. Predicted Forecasts

The top three models are visualized against the actual demand for the first test block.

The comparison uses:

* Actual rental demand
* Rolling-168 XGBoost
* Rolling-24 XGBoost
* Base XGBoost

The visualization uses the recursive forecasting predictions rather than predictions generated using future test targets.

### Figure to add

**Figure 11 — Actual vs. predicted hourly counts for the top-3 models**

```text
![Actual vs predicted](images/actual-vs-predicted-top3.png)
```

---

# 19. Key Technical Decisions

| Decision                                | Rationale                                                         |
| --------------------------------------- | ----------------------------------------------------------------- |
| Use `cnt` as target                     | Direct measure of total hourly demand                             |
| Exclude `casual` and `registered`       | They directly constitute the target                               |
| Exclude `atemp`                         | Highly redundant with `temp`                                      |
| Preserve temporal ordering              | Forecasting requires future-aware evaluation                      |
| Days 1–20 → train                       | Follows the assessment specification                              |
| Days 21–end → test                      | Provides a held-out future-like period                            |
| Use ANOVA for multi-group variables     | Tests differences across categorical groups                       |
| Use Welch's t-test for binary variables | Avoids equal-variance assumption                                  |
| Check Pearson + Spearman                | Screens both linear and monotonic relationships                   |
| Check VIF                               | Detects multicollinearity                                         |
| Compare linear and quadratic models     | Investigates potential nonlinear effects                          |
| Use Negative Binomial GLM               | Addresses overdispersed count response                            |
| Use XGBoost                             | Captures nonlinear interactions and complex feature relationships |
| Use lag/rolling features                | Represents historical demand information                          |
| Recursive test forecasting              | Prevents future target leakage                                    |

---

# 20. Reproducibility

The project uses a fixed random seed where stochastic modeling is involved:

```python
random_state = 42
```

The XGBoost implementation also uses:

```python
n_jobs = -1
```

to utilize available CPU resources.

The complete analysis, statistical tests, feature engineering, forecasting experiments, and evaluation are implemented in the accompanying Jupyter notebook.

---

# 21. Project Structure

A recommended repository structure is:

```text
capital-bikeshare-demand-forecasting/
│
├── README.md
│
├── notebooks/
│   └── part-01-code-aritra-sarkar.ipynb
│
├── images/
│   ├── hourly-demand-distribution.png
│   ├── casual-vs-registered.png
│   ├── workingday-weekday-patterns.png
│   ├── month-season-patterns.png
│   ├── daily-yearly-demand.png
│   ├── weather-temperature.png
│   ├── humidity-windspeed.png
│   ├── correlation-vif.png
│   ├── nonlinearity-analysis.png
│   ├── ols-diagnostics.png
│   └── actual-vs-predicted-top3.png
│
└── requirements.txt
```

---

# 22. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Statsmodels
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

# 23. Conclusion

This project develops a complete demand-analysis and forecasting workflow for hourly bike-sharing demand.

The analysis first establishes the statistical characteristics of the response, identifies important temporal and weather patterns, investigates multicollinearity and nonlinear relationships, and evaluates appropriate count-regression assumptions.

For forecasting, a strict temporal partition is combined with leakage-safe historical-demand features and recursive multi-step prediction.

The final experiments demonstrate that tree-based forecasting substantially outperforms the Negative Binomial statistical baseline on the held-out test period. The strongest RMSE is obtained using the 168-hour rolling-demand feature, although the base XGBoost model performs nearly as well, indicating that the temporal and calendar variables already capture a large portion of the predictable demand structure.

The complete implementation is available in the accompanying notebook.

---

## Author

**Aritra Sarkar**
BSc in Electrical & Electronic Engineering, CUET

Research interests include:

* Statistical Machine Learning
* Computational Imaging
* Biomedical Signal Processing
* Machine Learning for Healthcare
* Biomedical Sensing

---

## Reference

Dataset:
The raw data (`hour.csv`, `day.csv`) is not redistributed in this repository — download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) and update the `DATA_PATH` variable at the top of the notebook to point to the folder containing it.

## Reproducing This Analysis

```bash
git clone https://github.com/<your-username>/bike-sharing-demand-analysis.git
cd bike-sharing-demand-analysis
pip install -r requirements.txt
jupyter notebook notebooks/bike_sharing_analysis.ipynb
```



## Tech Stack

`pandas`, `numpy` — data handling · `matplotlib`, `seaborn` — visualization · `scipy.stats` — ANOVA, Welch's t-test · `statsmodels` — OLS, Poisson & Negative Binomial GLM, VIF · `scikit-learn` — evaluation metrics · `xgboost` — gradient-boosted forecasting models

## License

Released under the [MIT License](LICENSE).
