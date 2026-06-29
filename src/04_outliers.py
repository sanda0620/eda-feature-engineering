import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("data/imputed_data.csv")

print(f"Loaded: {df.shape}")
numeric_cols = ["age", "income", "spend", "visits", "satisfaction", "tenure_days"]

print("\n── Outlier Suspects (before) ──")
print(df[numeric_cols].describe().loc[["25%", "75%", "max"]].round(2))


print("\n── Method 1: Z-Score Detection (threshold = 3) ──")
z_scores = pd.DataFrame(
    np.abs(stats.zscore(df[numeric_cols])),
    columns=numeric_cols
)
print("Outliers detected per column:")
outlier_counts = (z_scores > 3).sum()
print(outlier_counts)


print("\n── Capping Outliers via Z-Score ──")
for col in numeric_cols:
    col_z = z_scores[col]
    outlier_mask = col_z > 3

    if outlier_mask.any():
        # Find the highest NON-outlier value as the cap
        upper_cap = df.loc[~outlier_mask, col].max()
        lower_cap = df.loc[~outlier_mask, col].min()

        n_capped = outlier_mask.sum()

        df.loc[outlier_mask & (df[col] > upper_cap), col] = upper_cap
        df.loc[outlier_mask & (df[col] < lower_cap), col] = lower_cap

        print(f"  {col}: capped {n_capped} outlier(s) → upper={upper_cap:.2f}")


print("\n── Method 2: IQR Detection & Capping ──")
for col in ["income", "spend"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower_fence) | (df[col] > upper_fence))

    print(f"\n  {col}:")
    print(f"    Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}")
    print(f"    Fences → [{lower_fence:.0f}, {upper_fence:.0f}]")
    print(f"    Outliers found: {outliers.sum()}")

    df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)
    print(f"    Clipped to fences ✓")


print("\n── After Outlier Handling ──")
print(df[numeric_cols].describe().loc[["25%", "75%", "max"]].round(2))
print("\n── Key changes ──")
print(f"  age max:    {df['age'].max():.1f}   (was 140.0)")
print(f"  income max: {df['income'].max():.0f}  (was 9,500,000)")
print(f"  spend max:  {df['spend'].max():.0f}   (was 850,000)")



print("\n── Skewness Before vs After ──")
raw = pd.read_csv("data/raw_data.csv")
for col in ["age", "income", "spend", "tenure_days"]:
    before = raw[col].dropna().skew()
    after  = df[col].skew()
    print(f"  {col:15} before={before:7.3f}  →  after={after:7.3f}")
    

df.to_csv("data/clean_data.csv", index=False)
print(f"\n✓ Clean dataset saved → data/clean_data.csv")
print(f"  Shape: {df.shape}")