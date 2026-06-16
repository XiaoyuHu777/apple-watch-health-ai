# apple-watch-health-ai

This project is a personal wearable health analytics prototype.
It is not a medical diagnostic system.

## Overview

`apple-watch-health-ai` is a Python project for analyzing Apple Watch and
Apple Health data at a daily level. The project parses Apple Health XML exports,
constructs daily sleep, HRV, resting heart rate, heart rate, steps, active
energy, and exercise features, builds a transparent rule-based personal recovery
trend score, and evaluates whether those features contain predictive signal for
next-day recovery prediction.

The project is designed as an end-to-end wearable health analytics and machine
learning baseline evaluation workflow. It emphasizes data privacy, time-based
validation, baseline comparison, and conservative interpretation.

## Project Objectives

- Parse Apple Health XML data locally without exposing raw health records.
- Convert irregular Apple Health records into daily aggregate features.
- Build an interpretable rule-based personal recovery trend score.
- Visualize recovery trends and score coverage over time.
- Construct an ML-ready dataset for next-day recovery prediction.
- Compare machine learning models against simple time-series baselines.
- Document limitations clearly and avoid clinical or diagnostic claims.

## Project Modules

1. **Apple Health XML parsing**
   - Script: `src/parse_apple_health.py`
   - Parses `data/raw/export.xml` using streaming XML processing.
   - Extracts selected Apple Health record types into processed CSV files.
   - Prints only aggregate parsing summaries, not raw health records.

2. **Daily health feature construction**
   - Script: `src/build_daily_features.py`
   - Aggregates parsed records into one row per date.
   - Produces daily sleep, HRV, resting heart rate, heart rate, steps, active
     energy, and exercise features.

3. **Rule-based recovery score**
   - Script: `src/recovery_score.py`
   - Builds a transparent 0-100 personal recovery trend score.
   - Uses fixed component weights and personal z-score normalization.
   - This logic is intentionally separate from the ML extension.

4. **Recovery trend visualization**
   - Script: `src/visualize.py`
   - Notebook: `notebooks/03_recovery_analysis.ipynb`
   - Creates clean and portfolio-style recovery trend charts.

5. **Date-filtered analysis: Jun-Dec 2025**
   - Uses the 2025-06-01 to 2025-12-31 period for focused trend analysis and
     ML evaluation.
   - Uses a chronological train/test split when this window is available.

6. **ML extension: next-day recovery prediction**
   - Feature script: `src/ml_features.py`
   - Training script: `src/train_recovery_ml.py`
   - Visualization script: `src/plot_ml_results.py`
   - Notebook: `notebooks/04_ml_recovery_prediction.ipynb`
   - Compares ML models against persistence and 7-day rolling mean baselines.

7. **Limitations and future work**
   - Documents sample-size limits, missing data, target definition constraints,
     and the need for stronger data quality before further modeling.

## Repository Structure

```text
apple-watch-health-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_explore_export_xml.ipynb
│   ├── 02_daily_health_features.ipynb
│   ├── 03_recovery_analysis.ipynb
│   └── 04_ml_recovery_prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── parse_apple_health.py
│   ├── build_daily_features.py
│   ├── recovery_score.py
│   ├── ml_features.py
│   ├── train_recovery_ml.py
│   ├── plot_ml_results.py
│   ├── visualize.py
│   └── utils.py
└── reports/
    └── figures/
```

## Data Privacy

Apple Health export data is sensitive personal data.

- `data/raw/` is ignored by Git.
- `data/processed/` is ignored by Git.
- `reports/figures/` is ignored by Git because figures may reveal personal
  time patterns.
- `reports/ml_predictions.csv` is ignored by Git because it contains per-date
  prediction outputs.
- Raw Apple Health XML files should never be committed, uploaded, or shared.
- Generated CSV files should be reviewed before sharing because dates, device
  patterns, and health metrics can be identifying.

The repository is intended to contain source code, documentation, notebooks
without executed outputs, and non-sensitive aggregate metric summaries only.

## Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`:

- pandas
- numpy
- matplotlib
- seaborn
- lxml
- jupyter
- scikit-learn

## Workflow

### 1. Parse Apple Health XML

Place the Apple Health export file locally at:

```text
data/raw/export.xml
```

Run:

```bash
python src/parse_apple_health.py \
  --input data/raw/export.xml \
  --output data/processed
```

Generated processed files include:

- `record_type_summary.csv`
- `heart_rate.csv`
- `resting_heart_rate.csv`
- `hrv.csv`
- `sleep_analysis.csv`
- `steps.csv`
- `active_energy.csv`
- `exercise_time.csv`

The parser uses streaming XML processing and prints only aggregate summaries.

### 2. Build Daily Health Features

Run:

```bash
python src/build_daily_features.py \
  --input-dir data/processed \
  --output data/processed/daily_health_features.csv
```

Optionally specify an IANA timezone for daily boundaries:

```bash
python src/build_daily_features.py \
  --input-dir data/processed \
  --output data/processed/daily_health_features.csv \
  --timezone Asia/Shanghai
```

Output columns include:

- `heart_rate_mean`, `heart_rate_min`, `heart_rate_max`
- `resting_heart_rate_mean`
- `hrv_mean`, `hrv_median`
- `steps_total`
- `active_energy_total`
- `exercise_minutes_total`
- `sleep_hours`

The output covers a continuous date range. Missing observations are preserved
as missing values rather than being filled with artificial values.

### 3. Build Rule-Based Recovery Score

Run:

```bash
python src/recovery_score.py \
  --input data/processed/daily_health_features.csv \
  --output data/processed/daily_recovery_score.csv \
  --figure-dir reports/figures
```

The rule-based score uses fixed component weights:

| Component | Weight | Direction |
|---|---:|---|
| Sleep duration | 30% | Higher is better |
| HRV | 30% | Higher is better |
| Resting heart rate | 20% | Higher is worse |
| Previous-day exercise load | 10% | Excess load is penalized |
| Previous-day active energy load | 10% | Excess load is penalized |

Component values are converted to personal z-scores, clipped to `[-3, 3]`, and
mapped to 0-100 component scores. Missing components are not imputed. Available
weights are re-normalized for each date, and the final score is left missing
when coverage is insufficient.

This score is a personal recovery trend indicator, not a clinical score.

### 4. Create Recovery Trend Visualizations

The recovery score script exports clean and portfolio-style charts:

- `reports/figures/recovery_score_trend_clean.png`
- `reports/figures/recovery_score_trend_clean.svg`
- `reports/figures/recovery_score_trend_portfolio.png`
- `reports/figures/recovery_score_trend_portfolio.svg`

The charting helper is implemented in `src/visualize.py`. The notebook
`notebooks/03_recovery_analysis.ipynb` reuses the same plotting function for
coverage checks, trend review, and component-level exploration.

### 5. Build ML Recovery Dataset

Run:

```bash
python src/ml_features.py \
  --features data/processed/daily_health_features.csv \
  --scores data/processed/daily_recovery_score.csv \
  --output data/processed/ml_recovery_dataset.csv
```

The ML dataset is built by merging daily health features and recovery scores by
`date`. It creates two targets:

```text
target_recovery_next_day = next day's recovery_score
low_recovery_next_day = 1 if next day's recovery_score < 40 else 0
```

The final date is dropped because it has no next-day target.

Feature engineering includes:

- Lag-1 features for recovery, sleep, HRV, resting heart rate, exercise,
  active energy, and steps.
- 3-day and 7-day rolling mean and rolling standard deviation.
- 7-day trend features for HRV, resting heart rate, and recovery score.
- Missingness indicators for core health fields.
- Calendar features: `day_of_week`, `is_weekend`, and `month`.

Rolling features use only the current date and past dates. They do not use
future information.

### 6. Train and Evaluate ML Models

Run:

```bash
python src/train_recovery_ml.py \
  --input data/processed/ml_recovery_dataset.csv \
  --reports-dir reports
```

The training script performs two tasks:

1. Regression: predict `target_recovery_next_day`.
2. Classification: predict `low_recovery_next_day`.

The script uses chronological splitting rather than random splitting. If the
dataset contains the full 2025-06-01 to 2025-12-31 window, the split is:

```text
Train: 2025-06-01 to 2025-10-31
Test:  2025-11-01 to 2025-12-31
```

Otherwise, the last 20% of target-valid dates are used as the test set.

Regression baselines:

- Persistence baseline: tomorrow's recovery equals today's recovery.
- 7-day rolling mean baseline: tomorrow's recovery equals the current 7-day
  rolling mean recovery score.

Regression models:

- Ridge Regression
- RandomForestRegressor
- HistGradientBoostingRegressor

Classification models:

- Logistic Regression
- RandomForestClassifier
- HistGradientBoostingClassifier

All ML models use `random_state=42`. Linear models use a median imputer,
standard scaler, and model inside a sklearn pipeline. Tree models use a median
imputer and model inside a sklearn pipeline.

Outputs:

- `reports/ml_regression_metrics.csv`
- `reports/ml_classification_metrics.csv`
- `reports/ml_predictions.csv`

`reports/ml_predictions.csv` is ignored by Git because it contains per-date
prediction outputs.

### 7. Visualize ML Results

Run:

```bash
python src/plot_ml_results.py \
  --predictions reports/ml_predictions.csv \
  --regression-metrics reports/ml_regression_metrics.csv \
  --classification-metrics reports/ml_classification_metrics.csv \
  --ml-dataset data/processed/ml_recovery_dataset.csv \
  --output-dir reports/figures
```

Generated figures:

- `ml_predicted_vs_actual.png` and `.svg`
- `ml_recovery_prediction_timeline.png` and `.svg`
- `ml_residuals.png` and `.svg`
- `ml_confusion_matrix.png` and `.svg`
- `ml_feature_importance.png` and `.svg`

The notebook `notebooks/04_ml_recovery_prediction.ipynb` organizes the ML
analysis into:

1. Project goal
2. Dataset overview
3. Train/test split explanation
4. Baseline comparison
5. Regression results
6. Classification results
7. Feature importance
8. Limitations
9. Conclusion

## Interpreting the ML Results

The central evaluation question is not which model is more complex. The central
question is whether the ML model improves over simple baselines.

For the current dataset:

- Best regression model: `RandomForestRegressor`
- Best classification model: `Logistic Regression`
- Train period: 2025-06-01 to 2025-10-31
- Test period: 2025-11-01 to 2025-12-31

Regression results show that the best ML model improves over the persistence
baseline, but it does not consistently improve over the 7-day rolling mean
baseline on common test samples.

On common test dates:

| Method | Common MAE |
|---|---:|
| Persistence baseline | 10.536 |
| 7-day rolling mean baseline | 7.841 |
| RandomForestRegressor | 7.999 |

The regression R2 remains below zero, and the classification task has very few
positive low-recovery examples in the test set. These results indicate limited
incremental predictive signal beyond rolling trends.

## Feature Importance

For the current best regression model, the top Random Forest importance features
include:

- `resting_heart_rate_mean_roll3_mean`
- `exercise_minutes_total_roll7_mean`
- `resting_heart_rate_mean`
- `exercise_minutes_total`
- `hrv_mean_roll3_std`

These should be interpreted as model-specific associations with next-day
recovery prediction. They are not causal explanations.

## Project Conclusion

This project demonstrates a complete wearable health analytics pipeline, from
local Apple Health XML parsing to daily feature construction, rule-based
personal recovery trend scoring, ML feature engineering, time-based validation,
baseline evaluation, and result visualization.

The current ML extension does not provide stable improvement over a simple
7-day rolling mean baseline. This is a valid project conclusion: the current
feature set and target-valid sample size are not yet sufficient for reliable
next-day recovery prediction.

## Limitations and Future Work

The next priority should be improving labels and data quality before changing
models.

Recommended next steps:

1. Replace the fixed `<40` low-recovery label with a personal percentile label,
   such as the lowest 25% of personal recovery scores.
2. Add a data coverage score to distinguish complete data days from days with
   missing sleep, HRV, resting heart rate, or activity data.
3. Add component-level attribution to explain whether low recovery scores are
   mainly associated with HRV, resting heart rate, sleep, or exercise load.
4. Evaluate on a longer target-valid time series.
5. Reserve an untouched future time window for final validation.
6. Investigate device source differences, missingness mechanisms, and temporal
   drift.

## Notes for Local Use

In VS Code:

1. Open the project root directory.
2. Select the `.venv/bin/python` interpreter.
3. Open the integrated terminal and activate the virtual environment if needed:

```bash
source .venv/bin/activate
```

4. Run the pipeline commands from the project root.

When opening notebooks, select the same `.venv` Python kernel.
