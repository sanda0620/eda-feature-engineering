import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_data.csv")

print("Print data loaded")
print(f" Shape:  {df.shape}")


print("\n ---Descriptive statistics---")
print(df.describe().round(2))


print("\n---Missing Value Audit---")
missing_count = df.isnull().sum()
missing_pct   = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "missing_count": missing_count,
    "missing_%":     missing_pct
})
print(missing_df[missing_df["missing_count"] > 0])


print("\n── Numerical Column Analysis ──")
num_cols = ["age", "income", "spend", "visits", "satisfaction", "tenure_days"]
for col in num_cols:
    clean = df[col].dropna()
    skewness = clean.skew()

    if abs(skewness) < 0.5:
        shape = "roughly symmetric"
    elif skewness > 0.5:
        shape = "right-skewed (long tail on right)"
    else:
        shape = "left-skewed (long tail on left)"

    print(f"\n{col}:")
    print(f"  mean={clean.mean():.2f}, median={clean.median():.2f}, std={clean.std():.2f}")
    print(f"  skewness={skewness:.3f} → {shape}")


print("\n── Categorical Column Analysis ──")
cat_cols = ["region", "is_member"]
for col in cat_cols:
    print(f"\n{col}:")
    counts = df[col].value_counts()
    pcts   = df[col].value_counts(normalize=True).mul(100).round(1)

    summary = pd.DataFrame({
        "count": counts,
        "%":     pcts
    })
    print(summary)


print("\n── Correlation Matrix ──")
corr = df[num_cols].corr().round(3)
print(corr)


print("\n── Strongest Correlations ──")
# Get upper triangle only (avoid duplicates)
corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
)
corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
corr_pairs["abs_corr"] = corr_pairs["correlation"].abs()
corr_pairs = corr_pairs.sort_values("abs_corr", ascending=False)

print(corr_pairs.head(10).to_string(index=False))


print("\n── EDA Summary ──")
print(f"Total rows       : {len(df)}")
print(f"Total columns    : {df.shape[1]}")
print(f"Numeric columns  : {len(num_cols)}")
print(f"Categorical cols : {len(cat_cols)}")
print(f"Total missing    : {df.isnull().sum().sum()}")
print(f"Missing columns  : {df.isnull().any().sum()}")
print(f"\nOutlier suspects (check max vs 75%):")
print(df[num_cols].describe().loc[["75%","max"]].round(0))