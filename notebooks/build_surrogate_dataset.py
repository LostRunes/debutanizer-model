"""
notebooks/build_surrogate_dataset.py
====================================
Phase 5.1A: Build dataset for training surrogate models (t+1 horizon).

Creates:
- bottom_temp_future_t1
- tray_temp_future_t1
- pressure_future_t1

Using shift(-1) within each Data_Block.
"""

import os
import pandas as pd
import numpy as np

FEATURES_FILE = "data/features.parquet"
OUT_FILE = "data/surrogate_data.parquet"

def main():
    print("=== Step 1: Building Surrogate Dataset ===")
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f"Missing engineered features file: {FEATURES_FILE}")
        
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded features shape: {df.shape}")
    
    # Check block distribution
    print("Block counts:")
    print(df["Data_Block"].value_counts().sort_index())
    
    # Create future targets (shift -1 within each block to avoid leakage across gaps)
    df["bottom_temp_future_t1"] = np.nan
    df["tray_temp_future_t1"] = np.nan
    df["pressure_future_t1"] = np.nan
    
    for block_val, group in df.groupby("Data_Block"):
        idx = group.index
        df.loc[idx, "bottom_temp_future_t1"] = group["Column_Bottom_Temp"].shift(-1)
        df.loc[idx, "tray_temp_future_t1"]   = group["Control_Tray_Temp"].shift(-1)
        df.loc[idx, "pressure_future_t1"]    = group["Column_Top_Pressure"].shift(-1)
        
    # Before dropping, check NaN counts
    print("\nTarget NaN counts before drop (expected 1 per block):")
    print(df[["bottom_temp_future_t1", "tray_temp_future_t1", "pressure_future_t1"]].isna().sum())
    
    # Drop rows where future targets are NaN (the last row of each block)
    initial_len = len(df)
    df = df.dropna(subset=["bottom_temp_future_t1", "tray_temp_future_t1", "pressure_future_t1"])
    dropped = initial_len - len(df)
    print(f"Dropped {dropped} boundary rows.")
    print(f"Final surrogate dataset shape: {df.shape}")
    
    # Save dataset
    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved surrogate dataset to {OUT_FILE}")
    print("==========================================")

if __name__ == "__main__":
    main()
