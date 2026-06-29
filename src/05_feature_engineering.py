import pandas as pd
import numpy as np

df = pd.read_csv("data/clean_data.csv")

print(f"Loaded: {df.shape}")
print(f"Columns: {list(df.columns)}")



print("\n── Feature 1: spend_to_income_ratio ──")
df["spend_to_income_ratio"] = df["spend"] / (df["income"] + 1)

print(f"  Formula: spend / (income + 1)")
print(f"  Mean:  {df['spend_to_income_ratio'].mean():.4f}")
print(f"  Min:   {df['spend_to_income_ratio'].min():.4f}")
print(f"  Max:   {df['spend_to_income_ratio'].max():.4f}")

print("\n  Sample interpretation:")
sample = df[["income", "spend", "spend_to_income_ratio"]].head(5)
print(sample.round(3))



print("\n── Feature 2: customer_value_score ──")
def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

spend_norm        = min_max_norm(df["spend"])
visits_norm       = min_max_norm(df["visits"])
satisfaction_norm = min_max_norm(df["satisfaction"])
tenure_norm       = min_max_norm(df["tenure_days"])

df["customer_value_score"] = (
    0.4 * spend_norm        +
    0.3 * visits_norm       +
    0.2 * satisfaction_norm +
    0.1 * tenure_norm
)

print(f"  Weights: spend=40%, visits=30%, satisfaction=20%, tenure=10%")
print(f"  Range:   {df['customer_value_score'].min():.4f} to {df['customer_value_score'].max():.4f}")
print(f"  Mean:    {df['customer_value_score'].mean():.4f}")



print("\n── Feature 3: log_income ──")
df["log_income"] = np.log1p(df["income"])
print(f"  Formula: log1p(income)  i.e. log(income + 1)")
print(f"  Income skewness:     {df['income'].skew():.3f}")
print(f"  Log income skewness: {df['log_income'].skew():.3f}")
print(f"  Income range:     {df['income'].min():.0f} to {df['income'].max():.0f}")
print(f"  Log income range: {df['log_income'].min():.3f} to {df['log_income'].max():.3f}")



print("\n── Feature 4: age_group ──")
bins   = [0, 25, 35, 50, 65, 100]
labels = ["Gen-Z", "Young-Adult", "Mid-Career", "Senior", "Retired"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
print("  Distribution:")
print(df["age_group"].value_counts().sort_index())



print("\n── Feature 5: tenure_years ──")
df["tenure_years"] = (df["tenure_days"] / 365).round(2)
print(f"  Mean tenure:  {df['tenure_years'].mean():.2f} years")
print(f"  Max tenure:   {df['tenure_years'].max():.2f} years")
print("\n  Sample:")
print(df[["tenure_days", "tenure_years"]].head(5))



print("\n── Feature 6: high_value_flag ──")
threshold = df["customer_value_score"].quantile(0.75)
df["high_value_flag"] = (df["customer_value_score"] >= threshold).astype(int)
print(f"  Threshold (75th percentile): {threshold:.4f}")
print(f"  High-value customers: {df['high_value_flag'].sum()} / {len(df)}")
print(f"  Percentage: {df['high_value_flag'].mean()*100:.1f}%")

print("\n  Average stats by segment:")
print(df.groupby("high_value_flag")[["income","spend","visits","satisfaction"]].mean().round(2))



print("\n── Final Dataset Summary ──")
print(f"  Original columns:   8")
print(f"  Engineered features: 6")
print(f"  Total columns:      {df.shape[1]}")
print(f"  Total rows:         {df.shape[0]}")
print(f"  Missing values:     {df.isnull().sum().sum()}")

print("\n  New columns added:")
new_cols = ["spend_to_income_ratio", "customer_value_score",
            "log_income", "age_group", "tenure_years", "high_value_flag"]
print(df[new_cols].describe().round(3))

df.to_csv("outputs/final_data.csv", index=False)
print("\n✓ Final dataset saved → outputs/final_data.csv")