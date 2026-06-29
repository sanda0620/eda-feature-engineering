"""
Run the full EDA & Feature Engineering pipeline end to end.
"""
import subprocess
import sys

scripts = [
    "src/01_load_data.py",
    "src/02_eda.py",
    "src/03_imputation.py",
    "src/04_outliers.py",
    "src/05_feature_engineering.py",
]

for script in scripts:
    print(f"\n{'='*50}")
    print(f"  Running {script}")
    print(f"{'='*50}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR in {script} — stopping pipeline.")
        break

print("\n✓ Full pipeline complete.")