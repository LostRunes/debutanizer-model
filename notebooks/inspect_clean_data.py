"""
inspect_clean_data.py
---------------------
Visual inspection of data/clean_data.parquet after Phase 1 preprocessing.
Run this after data_preprocessing.py.  All findings go to stdout.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/clean_data.parquet")

SEP = "=" * 70

# ============================================================
# 1. SHAPE AND SCHEMA
# ============================================================
print(SEP)
print("1. SHAPE & SCHEMA")
print(SEP)
print("Shape:", df.shape)
print("\nDtypes:")
print(df.dtypes.to_string())
print("\nFirst 3 rows:")
print(df.head(3).to_string())

# ============================================================
# 2. BLOCK STRUCTURE
# ============================================================
print("\n" + SEP)
print("2. BLOCK STRUCTURE")
print(SEP)
block_info = df.groupby("Data_Block")["DateTime"].agg(
    start="min", end="max", n_rows="count"
)
block_info["duration_days"] = (
    (block_info["end"] - block_info["start"]).dt.total_seconds() / 86400
).round(1)
print(block_info.to_string())
print("\nGaps between blocks:")
ends   = block_info["end"].values
starts = block_info["start"].values[1:]
for i, (e, s) in enumerate(zip(ends, starts)):
    gap = (pd.Timestamp(s) - pd.Timestamp(e))
    print(f"  Block {i+1} end  -> Block {i+2} start: {gap}")

# ============================================================
# 3. PROCESS VARIABLE STATS (per block)
# ============================================================
print("\n" + SEP)
print("3. PROCESS VARIABLE STATS PER BLOCK")
print(SEP)
process_cols = [
    "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Top_Pressure",
    "Column_Bottom_Temp", "Control_Tray_Temp",
]
for col in process_cols:
    print(f"\n  {col}:")
    stats = df.groupby("Data_Block")[col].agg(["mean", "std", "min", "max"])
    stats = stats.round(2)
    print(stats.to_string())

# ============================================================
# 4. TARGET VARIABLE ANALYSIS
# ============================================================
print("\n" + SEP)
print("4. TARGET VARIABLE ANALYSIS")
print(SEP)

print("\n  Total_C4 overall:")
print(df["Total_C4"].describe().round(4).to_string())

print("\n  Total_C4 per block:")
print(df.groupby("Data_Block")["Total_C4"].agg(
    mean="mean", std="std", median="median", pct_above_spec=lambda x: (x > 0.5).mean() * 100
).round(3).to_string())

print("\n  C4H6 overall (winsorised):")
print(df["C4H6_Bottom"].describe().round(4).to_string())
print("\n  C4H8 overall (winsorised):")
print(df["C4H8_Bottom"].describe().round(4).to_string())

# ============================================================
# 5. STUCK ANALYZER BREAKDOWN
# ============================================================
print("\n" + SEP)
print("5. STUCK ANALYZER BREAKDOWN")
print(SEP)

for col, stuck_col in [("C4H6_Bottom", "C4H6_Bottom_stuck"),
                        ("C4H8_Bottom", "C4H8_Bottom_stuck")]:
    print(f"\n  {col}:")
    print(f"    Total stuck rows: {df[stuck_col].sum():,}  ({df[stuck_col].mean()*100:.1f}%)")

    # How long are the stuck streaks?
    hrs_col = "hours_since_" + col + "_change"
    stuck_rows = df[df[stuck_col]]
    if len(stuck_rows):
        streak_lens = stuck_rows[hrs_col]
        print(f"    Max hours stuck: {streak_lens.max()}")
        for thresh in [12, 24, 48, 168, 336, 720]:
            n = (streak_lens >= thresh).sum()
            print(f"    Rows stuck >= {thresh:4d} h: {n:5,}  ({n/len(df)*100:.1f}%)")

    # Distribution of C4 value DURING stuck periods vs normal
    stuck_c4   = df.loc[df[stuck_col], col].describe().round(4)
    normal_c4  = df.loc[~df[stuck_col], col].describe().round(4)
    print(f"    {col} during STUCK (describe):")
    print(stuck_c4.to_string())
    print(f"    {col} during NORMAL (describe):")
    print(normal_c4.to_string())

# ============================================================
# 6. BIMODAL DISTRIBUTION INVESTIGATION
# ============================================================
print("\n" + SEP)
print("6. BIMODAL DISTRIBUTION INVESTIGATION")
print(SEP)

for col in ["Reboiler_Outlet_Temp", "Column_Top_Temp"]:
    print(f"\n  {col}  percentile breakdown:")
    for p in [5, 10, 25, 40, 50, 60, 75, 90, 95]:
        print(f"    P{p:2d}: {df[col].quantile(p/100):.2f}")
    # How many rows in the "cold" regime vs "hot" regime?
    # The bimodal split appears at ~50C
    cold = (df[col] < 50).sum()
    hot  = (df[col] >= 50).sum()
    print(f"    Below 50C: {cold:,} ({cold/len(df)*100:.1f}%) | Above 50C: {hot:,} ({hot/len(df)*100:.1f}%)")

# Does the bimodal split align with data blocks?
print("\n  Regime (temp < 50C) count per Data_Block:")
df["low_reboiler_regime"] = df["Reboiler_Outlet_Temp"] < 50
print(df.groupby("Data_Block")["low_reboiler_regime"].agg(
    n_cold=lambda x: x.sum(),
    n_hot=lambda x: (~x).sum(),
    pct_cold=lambda x: x.mean() * 100
).round(1).to_string())

print("\n  Total_C4 mean: cold regime vs hot regime:")
print(df.groupby("low_reboiler_regime")["Total_C4"].agg(
    mean="mean", std="std", pct_above_spec=lambda x: (x > 0.5).mean() * 100
).round(3).to_string())

# ============================================================
# 7. WINSORISATION CHECK
# ============================================================
print("\n" + SEP)
print("7. WINSORISATION IMPACT CHECK")
print(SEP)
print("  Checking no remaining values outside P0.5-P99.5 after clipping ...")
all_ok = True
for col in process_cols + ["C4H6_Bottom", "C4H8_Bottom"]:
    lo = df[col].quantile(0.005)
    hi = df[col].quantile(0.995)
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out > 5:
        print(f"  WARNING: {col} has {n_out} values outside P0.5-P99.5")
        all_ok = False
if all_ok:
    print("  All columns within expected range after winsorisation. OK.")

# ============================================================
# 8. NULL CHECK ON FINAL PARQUET
# ============================================================
print("\n" + SEP)
print("8. NULL CHECK ON FINAL PARQUET")
print(SEP)
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if len(nulls) == 0:
    print("  No null values in any column. OK.")
else:
    print("  NULL COUNTS FOUND:")
    print(nulls.to_string())

# ============================================================
# 9. HOURLY SAMPLING CONSISTENCY CHECK
# ============================================================
print("\n" + SEP)
print("9. SAMPLING CONSISTENCY CHECK")
print(SEP)
for block_id in df["Data_Block"].unique():
    blk = df[df["Data_Block"] == block_id].sort_values("DateTime")
    diffs = blk["DateTime"].diff().dropna()
    non_hourly = diffs[diffs != pd.Timedelta(hours=1)]
    print(f"  Block {block_id}: {len(blk)} rows, "
          f"non-1h gaps: {len(non_hourly)} "
          f"(max gap: {diffs.max()})")

print("\n" + SEP)
print("INSPECTION COMPLETE")
print(SEP)
