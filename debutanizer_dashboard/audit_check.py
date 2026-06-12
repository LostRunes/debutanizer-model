"""Audit script to inspect features.parquet and dashboard data issues."""
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(os.path.join(BASE_DIR, "data", "features.parquet"))

print("=== BLOCK INFO ===")
for b in sorted(df["Data_Block"].dropna().unique()):
    blk = df[df["Data_Block"] == b]
    dt_min = blk["DateTime"].min()
    dt_max = blk["DateTime"].max()
    print(f"Block {int(b)}: {len(blk)} rows | {dt_min} --> {dt_max}")

print()
print("=== HISTORICAL TRENDS SCOPE (get_current_history) ===")
block4 = df[df["Data_Block"] == 4]
first_idx = block4.index[0]
last_idx = block4.index[-1]
print(f"Block 4 integer index range: {first_idx} to {last_idx}")
history_slice = df.loc[max(0, first_idx-24):first_idx]
print(f"history at first_idx: df.loc[{max(0,first_idx-24)}:{first_idx}] = {len(history_slice)} rows")
print(f"NOTE: get_current_history uses integer label loc, not positional. If index is non-contiguous this may skip rows.")

print()
print("=== TREND PAGE COLUMNS - BLOCK 4 STATS ===")
trend_cols = ["Total_C4","C4H8_Bottom","C4H6_Bottom","Reboiling_Steam_Flow",
              "Reflux_Flow","Column_Bottom_Temp","Control_Tray_Temp",
              "Column_Top_Pressure","Feed_Flow"]
for c in trend_cols:
    if c in df.columns:
        col_data = df.loc[block4.index, c]
        null_pct = col_data.isna().sum() / len(col_data) * 100
        print(f"  {c:35s}: min={col_data.min():.4f}, max={col_data.max():.4f}, null={null_pct:.1f}%")
    else:
        print(f"  {c:35s}: *** COLUMN MISSING ***")

print()
print("=== INDEX TYPE ===")
print(f"DataFrame index dtype: {df.index.dtype}")
print(f"Is monotonic increasing: {df.index.is_monotonic_increasing}")
print(f"Has duplicates: {df.index.duplicated().any()}")
print(f"Index range: {df.index.min()} to {df.index.max()}")

print()
print("=== CLEAN DATA PARQUET ===")
clean = pd.read_parquet(os.path.join(BASE_DIR, "data", "clean_data.parquet"))
print(f"clean_data shape: {clean.shape}")
print(f"clean_data columns: {list(clean.columns)}")
print(f"clean_data Data_Block: {clean['Data_Block'].unique().tolist() if 'Data_Block' in clean.columns else 'NO Data_Block column'}")
