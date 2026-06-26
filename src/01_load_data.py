import pandas as pd
import numpy as np

np.random.seed(42)
n=300

df = pd.DataFrame({
    "age":          np.random.normal(38, 12, n).clip(18, 75),
    "income":       np.random.lognormal(10.5, 0.6, n),
    "spend":        np.random.lognormal(7.5, 0.8, n),
    "visits":       np.random.poisson(5, n).astype(float),
    "satisfaction": np.random.choice([1, 2, 3, 4, 5], n).astype(float),
    "tenure_days":  np.random.exponential(365, n),
    "region":       np.random.choice(["North", "South", "East", "West"], n),
    "is_member":    np.random.choice([0, 1], n),
})


missing_mask = {
    "age":          0.08,
    "income":       0.10,
    "spend":        0.06,
    "satisfaction": 0.05,
}

for col, pct in missing_mask.items():
    missing_indices = np.random.choice(n, size=int(n * pct), replace=False)
    df.loc[missing_indices, col] = np.nan

df.loc[5,  "income"] = 9_500_000
df.loc[12, "spend"]  = 850_000
df.loc[22, "age"]    = 140

df.to_csv("data/raw_data.csv", index=False)
print("Dataset saved → data/raw_data.csv")

print("\n── First 5 rows ──")
print(df.head())

print("\n── Shape ──")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n── Data Types ──")
print(df.dtypes)

print("\n── Missing Value Count ──")
print(df.isnull().sum())