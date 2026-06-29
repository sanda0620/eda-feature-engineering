# Advanced EDA & Feature Engineering

A step-by-step Python pipeline that transforms raw, messy data 
into a mathematically clean dataset ready for machine learning.

## Pipeline

| Script | What it does |
|--------|-------------|
| `src/01_load_data.py` | Generates synthetic dataset with missing values and outliers |
| `src/02_eda.py` | Exploratory analysis — distributions, skewness, correlations |
| `src/03_imputation.py` | Missing data handling — Mean, Median, KNN |
| `src/04_outliers.py` | Outlier detection and capping — Z-Score and IQR |
| `src/05_feature_engineering.py` | Engineers 6 new predictive features |

## Key Findings

- Income skewness reduced from **16.4 → 0.92** after outlier removal
- Log transform reduced income skewness further to **-0.14**
- High-value customers spend **2.5x more** than standard customers
  despite having similar income levels
- KNN imputation used for skewed columns (income, spend) to avoid
  mean bias from outliers

## Engineered Features

| Feature | Method | Purpose |
|---------|--------|---------|
| `spend_to_income_ratio` | spend ÷ income | Spending behaviour |
| `customer_value_score` | Weighted composite | Overall customer worth |
| `log_income` | log1p transform | Fixes right skew |
| `age_group` | Binning | Generational segments |
| `tenure_years` | Days ÷ 365 | Human-readable tenure |
| `high_value_flag` | Top 25% by score | ML target variable |

## Tools
Python · Pandas · NumPy · Scikit-learn · SciPy · Matplotlib

## How to Run

```bash
pip install -r requirements.txt
python src/01_load_data.py
python src/02_eda.py
python src/03_imputation.py
python src/04_outliers.py
python src/05_feature_engineering.py
```

## Results
- Input: 300 rows, 8 columns, 86 missing values, 3 extreme outliers
- Output: 300 rows, 14 columns, 0 missing values, 0 outliers