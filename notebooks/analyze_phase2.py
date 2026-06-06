import pandas as pd
import numpy as np

df = pd.read_parquet("data/features.parquet")

print("=== TEMP_GRADIENT and REBOILER_DELTA PER BLOCK ===")
print("Temp_Gradient per block:")
print(df.groupby("Data_Block")["Temp_Gradient"].agg(["mean","std","min","max"]).round(2))
print()
print("Reboiler_Delta per block:")
print(df.groupby("Data_Block")["Reboiler_Delta"].agg(["mean","std","min","max"]).round(2))

print()
print("=== REFLUX_RATIO vs C4 CORRELATION (top 15 features) ===")
tier1_model_features = [
    "Reflux_Ratio","Steam_Feed_Ratio","Temp_Gradient","Reboiler_Delta",
    "Feed_Flow","Reboiler_Outlet_Temp","Column_Top_Temp","Reboiling_Steam_Flow","Reflux_Flow",
    "Column_Top_Pressure","Column_Bottom_Temp","Control_Tray_Temp",
    "Steam_diff1","Reflux_diff1","Feed_diff1","Bottom_Temp_diff1"
]
corrs = df[tier1_model_features + ["Total_C4"]].corr()["Total_C4"].drop("Total_C4").abs().sort_values(ascending=False)
print("Top 15 correlations with Total_C4 (absolute value):")
print(corrs.head(15).round(4))

print()
print("=== ROLLING STD DISTRIBUTIONS (instability proxy) ===")
roll_std_cols = ["Reboiling_Steam_Flow_roll_std_6h","Reflux_Flow_roll_std_6h","Feed_Flow_roll_std_6h"]
for col in roll_std_cols:
    valid = df[col].dropna()
    pct = valid.quantile(0.95)
    print(f"  {col}: mean={valid.mean():.4f}, std={valid.std():.4f}, P95={pct:.4f}")

print()
print("=== TEMP_GRADIENT PERCENTILES ===")
percs = [1, 5, 10, 25, 40, 50, 60, 75, 90, 95, 99]
for p in percs:
    val = df["Temp_Gradient"].quantile(p / 100)
    print(f"  P{p:2d}: {val:.3f}")

print()
print("=== TRAINING ROW OVERVIEW ===")
train_mask = df["Data_Block"].isin([1, 2, 3])
test_mask  = df["Data_Block"] == 4
mA_filter  = ~df["C4H8_Bottom_stuck"]
mB_filter  = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)

# How many NaN rows will be dropped per model after filtering?
tier1_cols = [c for c in df.columns if "C4H8_Bottom_lag" not in c and "C4H6_Bottom_lag" not in c
              and c not in ["DateTime","C4H6_Bottom","C4H8_Bottom","Total_C4",
                            "C4H6_Bottom_stuck","C4H8_Bottom_stuck",
                            "hours_since_C4H6_Bottom_change","hours_since_C4H8_Bottom_change",
                            "Analyzer_Health","is_extreme_event"]]

train_A = df[train_mask & mA_filter].dropna(subset=tier1_cols)
test_A  = df[test_mask  & mA_filter].dropna(subset=tier1_cols)
train_B = df[train_mask & mB_filter].dropna(subset=tier1_cols)
test_B  = df[test_mask  & mB_filter].dropna(subset=tier1_cols)

print(f"  Model A - after dropping NaN rows in Tier1 features:")
print(f"    Train: {len(train_A)}, Test: {len(test_A)}")
print(f"  Model B - after dropping NaN rows in Tier1 features:")
print(f"    Train: {len(train_B)}, Test: {len(test_B)}")
print(f"  Extreme events in Model A train (final): {train_A['is_extreme_event'].sum()}")
print(f"  % extreme in Model A train: {train_A['is_extreme_event'].mean()*100:.1f}%")
