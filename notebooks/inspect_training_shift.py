import pandas as pd
import numpy as np

df = pd.read_parquet("data/features.parquet")

train_mask = df["Data_Block"].isin([1, 2, 3])
test_mask  = df["Data_Block"] == 4
mA_filter  = ~df["C4H8_Bottom_stuck"]
mB_filter  = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)

y_train_A = df.loc[train_mask & mA_filter, "C4H8_Bottom"]
y_test_A  = df.loc[test_mask & mA_filter, "C4H8_Bottom"]

y_train_B = df.loc[train_mask & mB_filter, "C4H6_Bottom"]
y_test_B  = df.loc[test_mask & mB_filter, "C4H6_Bottom"]

print("=== TARGET A (C4H8_Bottom) STATS ===")
print("Train A:")
print(y_train_A.describe())
print("\nTest A:")
print(y_test_A.describe())

print("\n=== TARGET B (C4H6_Bottom) STATS ===")
print("Train B:")
print(y_train_B.describe())
print("\nTest B:")
print(y_test_B.describe())

print("\n=== FEATURE DIFFERENCES TRAIN vs TEST ===")
tier1_cols = [
    "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Top_Pressure",
    "Column_Bottom_Temp", "Control_Tray_Temp", "Reflux_Ratio", "Steam_Feed_Ratio",
    "Temp_Gradient", "Reboiler_Delta"
]

for col in tier1_cols:
    train_val = df.loc[train_mask, col].mean()
    test_val = df.loc[test_mask, col].mean()
    print(f"{col:30s} | Train Mean: {train_val:10.4f} | Test Mean: {test_val:10.4f} | Diff: {test_val - train_val:+10.4f}")
