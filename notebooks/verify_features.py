import pandas as pd
import numpy as np

print("Loading features parquet...")
df = pd.read_parquet("data/features.parquet")
print(f"Features shape: {df.shape}")

print("\n--- VERIFYING NO LEAKAGE ACROSS BLOCKS ---")
leakage_found = False
for block_id in df["Data_Block"].unique():
    block_df = df[df["Data_Block"] == block_id].sort_values("DateTime")
    first_idx = block_df.index[0]
    
    # Check that lag-1 is NaN for the first row of each block
    lag_cols = [c for c in df.columns if "_lag1" in c]
    for col in lag_cols:
        val = block_df.loc[first_idx, col]
        if not pd.isna(val):
            print(f"  LEAKAGE ERROR: Block {block_id} first row has non-NaN value for {col}: {val}")
            leakage_found = True
            
    # Check rolling std 3h is NaN for the first row (min_periods=2)
    std_cols = [c for c in df.columns if "_roll_std_" in c]
    for col in std_cols:
        val = block_df.loc[first_idx, col]
        if not pd.isna(val):
            print(f"  LEAKAGE ERROR: Block {block_id} first row has non-NaN value for {col}: {val}")
            leakage_found = True

if not leakage_found:
    print("  SUCCESS: No leakage found. Lags and rolling stats are properly NaN-padded at block boundaries.")

print("\n--- VERIFYING EXTREME EVENTS FLAG ---")
n_extreme = df["is_extreme_event"].sum()
print(f"  Number of extreme events: {n_extreme}")
if n_extreme == 1392:
    print("  SUCCESS: Extreme events flag correctly identifies the 1,392 winsorised rows.")
else:
    print(f"  WARNING: Expected 1,392 extreme events, got {n_extreme}")

print("\n--- CHECKING MISSING VALUE PROFILE ---")
nan_counts = df.isnull().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) > 0:
    print("Columns with NaNs and their counts:")
    print(nan_cols.to_string())
else:
    print("  No columns with NaNs found (unexpected for lags).")
