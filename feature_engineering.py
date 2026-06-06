"""
feature_engineering.py
======================
Phase 2: clean_data.parquet -> features.parquet

Enhancements:
  1. Preprocessing: Gap-Aware Lag Protection via block hourly resampling
  2. Feature Engineering: Pressure-Normalized Temperatures (Linear, Ratio, Gradient)
  3. Feature Engineering: Pressure Interaction Features
  4. Feature Engineering: Relative-to-Rolling-Baseline Features (dev24h)
"""

import os
import numpy as np
import pandas as pd

CLEAN_FILE = os.path.join("data", "clean_data.parquet")
RAW_FILE   = "9.DB DATA -B.xlsx"
OUT_FILE   = os.path.join("data", "features.parquet")

COLUMN_RENAME = {
    "Unnamed: 0":            "DateTime",
    "Feed Flow to DB":       "Feed_Flow",
    "Reboiler o/l Temp":     "Reboiler_Outlet_Temp",
    "Column top Temp":       "Column_Top_Temp",
    "Reboiling steam flow":  "Reboiling_Steam_Flow",
    "Reflux flow":           "Reflux_Flow",
    "Column Top pressure":   "Column_Top_Pressure",
    "Column bottom temp":    "Column_Bottom_Temp",
    "Control tay temp":      "Control_Tray_Temp",
    "C4H6 in DB bottom":     "C4H6_Bottom",
    "C4H8 in DB bottom":     "C4H8_Bottom",
}

PROCESS_NUMERIC_COLS = [
    "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Top_Pressure",
    "Column_Bottom_Temp", "Control_Tray_Temp",
]
TARGET_COLS = ["C4H6_Bottom", "C4H8_Bottom"]
ALL_NUMERIC = PROCESS_NUMERIC_COLS + TARGET_COLS

PROCESS_COLS_FOR_SHUTDOWN_CHECK = [
    "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow",
]


def add_is_extreme_event_flag(df):
    """
    Identify rows where any numeric column was modified during winsorisation.
    """
    print("Identifying extreme events (winsorised rows)...")
    raw = pd.read_excel(RAW_FILE, sheet_name="Sheet2")
    df_raw = raw.iloc[2:].copy().reset_index(drop=True)
    df_raw = df_raw.rename(columns=COLUMN_RENAME)
    df_raw["DateTime"] = pd.to_datetime(df_raw["DateTime"], errors="coerce")
    
    for col in ALL_NUMERIC:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        
    df_raw = df_raw.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    
    # Filter shutdown rows using the same threshold (0.5)
    shutdown_mask = (df_raw[PROCESS_COLS_FOR_SHUTDOWN_CHECK] < 0.5).all(axis=1)
    df_raw = df_raw[~shutdown_mask].reset_index(drop=True)
    
    # Set index for easy alignment
    df_raw = df_raw.set_index("DateTime")
    df_aligned = df.set_index("DateTime")
    
    # Check which rows were modified during winsorisation
    clipped_mask = pd.Series(False, index=df_aligned.index)
    for col in ALL_NUMERIC:
        col_diff = ~np.isclose(df_aligned[col], df_raw.loc[df_aligned.index, col], atol=1e-8, rtol=1e-8)
        clipped_mask = clipped_mask | col_diff
        
    df["is_extreme_event"] = clipped_mask.values
    print(f"  Flagged {df['is_extreme_event'].sum()} rows as extreme events.")
    return df


def compute_block_features_gap_aware(df_block):
    """
    Compute lags, rolling means, rolling stds, and 24h rolling baselines
    within a single Data_Block in a gap-aware way by resampling to an hourly grid.
    """
    orig_index = df_block["DateTime"]
    df_block_temp = df_block.set_index("DateTime")
    
    # Check for duplicate indices
    if df_block_temp.index.duplicated().any():
        df_block_temp = df_block_temp[~df_block_temp.index.duplicated(keep='first')]
        
    # Reindex to hourly range
    start_time = df_block_temp.index.min()
    end_time = df_block_temp.index.max()
    hourly_index = pd.date_range(start=start_time, end=end_time, freq="1h")
    df_grid = df_block_temp.reindex(hourly_index)
    
    # Compute lags
    lag_config = {
        "Reboiling_Steam_Flow": [1, 2, 3, 6, 12],
        "Reflux_Flow":           [1, 2, 3, 6],
        "Feed_Flow":             [1, 2, 3],
        "Column_Bottom_Temp":    [1, 2, 3],
        "Control_Tray_Temp":     [1, 2, 3, 6],
        "Reboiler_Outlet_Temp":  [1, 2, 3],
        "Column_Top_Temp":       [1, 2],
        "Column_Top_Pressure":   [1, 2],
        # Target lags
        "C4H8_Bottom":           list(range(1, 13)),
        "C4H6_Bottom":           [1, 2, 3],
    }
    for col, lags in lag_config.items():
        for lag in lags:
            df_grid[f"{col}_lag{lag}"] = df_grid[col].shift(lag)
            
    # Compute rolling means
    roll_mean_cols = ["Reboiling_Steam_Flow", "Reflux_Flow", "Feed_Flow", "Column_Bottom_Temp"]
    windows_mean   = [3, 6, 12]
    for col in roll_mean_cols:
        for w in windows_mean:
            df_grid[f"{col}_roll_mean_{w}h"] = df_grid[col].rolling(w, min_periods=1).mean()
            
    # Compute rolling stds
    roll_std_cols  = ["Reboiling_Steam_Flow", "Reflux_Flow", "Feed_Flow"]
    windows_std    = [3, 6]
    for col in roll_std_cols:
        for w in windows_std:
            df_grid[f"{col}_roll_std_{w}h"] = df_grid[col].rolling(w, min_periods=2).std()
            
    # Compute 24h rolling means for deviations (relative-to-rolling-baseline)
    dev_cols = ["Reboiling_Steam_Flow", "Reflux_Flow", "Column_Bottom_Temp", "Control_Tray_Temp", "Column_Top_Pressure"]
    for col in dev_cols:
        roll_mean_24h = df_grid[col].rolling(24, min_periods=1).mean()
        df_grid[f"{col}_dev24h"] = df_grid[col] - roll_mean_24h
        
    # Reindex back to original timestamps of this block
    df_out = df_grid.reindex(orig_index)
    df_out = df_out.reset_index()
    return df_out


def run_feature_engineering():
    print("=" * 70)
    print("RUNNING FEATURE ENGINEERING (PHASE 2 - ADVANCED DRIFT MITIGATION)")
    print("=" * 70)
    
    # 1. Load clean parquet
    print(f"Loading clean data from {CLEAN_FILE}...")
    df = pd.read_parquet(CLEAN_FILE)
    print(f"  Loaded shape: {df.shape}")
    
    # 2. Add is_extreme_event flag
    df = add_is_extreme_event_flag(df)
    
    # 3. Lags, Rolling Features & Rolling Baselines (Within-block, Gap-aware)
    print("\nComputing block-based features (gap-aware lag protection)...")
    blocks_processed = []
    for block_val, group in df.groupby("Data_Block"):
        print(f"  Processing Block {block_val} (rows: {len(group)})...")
        block_feat = compute_block_features_gap_aware(group)
        blocks_processed.append(block_feat)
    df = pd.concat(blocks_processed, ignore_index=True)
    
    # 4. Engineered Ratios
    print("\nComputing engineered ratios...")
    df["Reflux_Ratio"]     = df["Reflux_Flow"] / df["Feed_Flow"]
    df["Steam_Feed_Ratio"] = df["Reboiling_Steam_Flow"] / df["Feed_Flow"]
    df["Temp_Gradient"]    = df["Column_Bottom_Temp"] - df["Column_Top_Temp"]
    df["Reboiler_Delta"]   = df["Reboiler_Outlet_Temp"] - df["Column_Bottom_Temp"]
    
    # 5. Rate of change (1h delta, uses lag1, so within-block)
    print("\nComputing 1-hour deltas (rate of change)...")
    df["Steam_diff1"]       = df["Reboiling_Steam_Flow"] - df["Reboiling_Steam_Flow_lag1"]
    df["Reflux_diff1"]      = df["Reflux_Flow"] - df["Reflux_Flow_lag1"]
    df["Feed_diff1"]        = df["Feed_Flow"] - df["Feed_Flow_lag1"]
    df["Bottom_Temp_diff1"] = df["Column_Bottom_Temp"] - df["Column_Bottom_Temp_lag1"]
    
    # 6. Pressure-Normalized Temperatures
    print("\nComputing pressure-normalized temperatures...")
    P_ref = 4.05  # training mean top pressure
    temps_to_norm = ["Column_Top_Temp", "Control_Tray_Temp", "Column_Bottom_Temp"]
    
    # Form 1: Linear subtraction for k = 3, 5, 10
    for k in [3.0, 5.0, 10.0]:
        k_str = str(int(k))
        for t_col in temps_to_norm:
            df[f"{t_col}_Pnorm_k{k_str}"] = df[t_col] - (df["Column_Top_Pressure"] - P_ref) * k
            
    # Form 2: Ratio
    for t_col in temps_to_norm:
        df[f"{t_col}_Pratio"] = df[t_col] / df["Column_Top_Pressure"]
        
    # Form 3: Gradient Normalized
    df["Temp_Gradient_Pnorm"] = (df["Column_Bottom_Temp"] - df["Column_Top_Temp"]) / df["Column_Top_Pressure"]
    
    # 7. Pressure Interaction Features
    print("\nComputing pressure interaction features...")
    df["Pressure_x_TopTemp"] = df["Column_Top_Pressure"] * df["Column_Top_Temp"]
    df["Pressure_x_BottomTemp"] = df["Column_Top_Pressure"] * df["Column_Bottom_Temp"]
    df["Pressure_x_ControlTrayTemp"] = df["Column_Top_Pressure"] * df["Control_Tray_Temp"]
    
    # Write output
    print(f"\nWriting features to {OUT_FILE}...")
    df.to_parquet(OUT_FILE, index=False)
    print(f"  Final features shape: {df.shape}")
    print(f"  Total columns: {len(df.columns)}")
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_feature_engineering()
