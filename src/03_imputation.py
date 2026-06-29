import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.read_csv("data/raw_data.csv")

print(f"Loaded: {df.shape}")
print(f"\nMissing before imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])


print("\n── Method 1: Mean Imputation (age) ──")
age_mean = df["age"].mean()
print(f"  Mean age (excluding NaN): {age_mean:.2f}")
df["age"] = df["age"].fillna(age_mean)
print(f"  Missing after: {df['age'].isnull().sum()}")


print("\n── Method 2: Median Imputation (satisfaction) ──")
sat_median = df["satisfaction"].median()
print(f"  Median satisfaction: {sat_median:.1f}")
df["satisfaction"] = df["satisfaction"].fillna(sat_median)
print(f"  Missing after: {df['satisfaction'].isnull().sum()}")


print("\n── Method 3: KNN Imputation (income, spend) ──")
print(f"  Missing income before: {df['income'].isnull().sum()}")
print(f"  Missing spend before:  {df['spend'].isnull().sum()}")
knn_cols = ["age", "income", "spend", "visits", "satisfaction", "tenure_days"]
knn_imputer = KNNImputer(n_neighbors=5)
df[knn_cols] = knn_imputer.fit_transform(df[knn_cols])
print(f"\n  Missing income after: {df['income'].isnull().sum()}")
print(f"  Missing spend after:  {df['spend'].isnull().sum()}")


print("\n── Verification ──")
print(f"Total missing after all imputation: {df.isnull().sum().sum()}")
print("\nColumn-wise check:")
print(df[["age", "income", "spend", "satisfaction"]].isnull().sum())
print("\nDescriptive stats after imputation:")
print(df[["age", "income", "spend", "satisfaction"]].describe().round(2))


df.to_csv("data/imputed_data.csv", index=False)
print("\n✓ Imputed dataset saved → data/imputed_data.csv")