import pandas as pd
import numpy as np

df = pd.read_parquet("data/features.parquet")

train_mask = df["Data_Block"].isin([1, 2, 3])
test_mask  = df["Data_Block"] == 4
mB_filter  = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)

# Filter data
train_df = df[train_mask & mB_filter]
test_df  = df[test_mask & mB_filter]

print("================================================================================")
print("C4H6 CORRELATIONS WITH PROCESS VARIABLES")
print("================================================================================")

process_cols = [
    "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Top_Pressure",
    "Column_Bottom_Temp", "Control_Tray_Temp", "Reflux_Ratio", "Steam_Feed_Ratio",
    "Temp_Gradient", "Reboiler_Delta"
]

print(f"{'Feature':30s} | {'Train Corr':10s} | {'Test Corr':10s} | {'Drift Status':15s}")
print("-" * 75)

for col in process_cols:
    if col in train_df.columns:
        train_corr = train_df[col].corr(train_df["C4H6_Bottom"])
        test_corr = test_df[col].corr(test_df["C4H6_Bottom"])
        
        # Determine status
        if np.isnan(train_corr) or np.isnan(test_corr):
            status = "NaN Corr"
        elif train_corr * test_corr < 0 and (abs(train_corr) > 0.1 or abs(test_corr) > 0.1):
            status = "REVERSED"
        elif abs(train_corr) > 0.1 and abs(test_corr) < 0.05:
            status = "LOST SIGNAL"
        elif abs(train_corr) < 0.05 and abs(test_corr) > 0.1:
            status = "GAINED SIGNAL"
        else:
            status = "STABLE"
            
        print(f"{col:30s} | {train_corr:+10.4f} | {test_corr:+10.4f} | {status:15s}")
        
print("================================================================================")
