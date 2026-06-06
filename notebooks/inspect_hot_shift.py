import pandas as pd
import numpy as np

df = pd.read_parquet("data/features.parquet")

train_mask = df["Data_Block"].isin([1, 2, 3])
test_mask  = df["Data_Block"] == 4
mA_filter  = ~df["C4H8_Bottom_stuck"]

train_df = df[train_mask & mA_filter].dropna()
test_df  = df[test_mask & mA_filter].dropna()

# Hot regime only
train_hot = train_df[train_df["Reboiler_Outlet_Temp"] >= 50]
test_hot  = test_df[test_df["Reboiler_Outlet_Temp"] >= 50]

print(f"Hot Train size: {len(train_hot)} | Hot Test size: {len(test_hot)}")

# Let's inspect stats for hot train vs hot test
cols = [
    "C4H8_Bottom", "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Top_Pressure",
    "Column_Bottom_Temp", "Control_Tray_Temp", "Reflux_Ratio", "Steam_Feed_Ratio",
    "Temp_Gradient", "Reboiler_Delta"
]

print(f"{'Feature':30s} | {'Train Mean':10s} | {'Test Mean':10s} | {'Train Min':10s} | {'Train Max':10s} | {'Test Min':10s} | {'Test Max':10s}")
print("-" * 110)
for col in cols:
    tr_mean = train_hot[col].mean()
    te_mean = test_hot[col].mean()
    tr_min  = train_hot[col].min()
    tr_max  = train_hot[col].max()
    te_min  = test_hot[col].min()
    te_max  = test_hot[col].max()
    print(f"{col:30s} | {tr_mean:10.2f} | {te_mean:10.2f} | {tr_min:10.2f} | {tr_max:10.2f} | {te_min:10.2f} | {te_max:10.2f}")
