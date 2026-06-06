import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

RAW_FILE = "9.DB DATA -B.xlsx"
PROCESS_COLS_FOR_SHUTDOWN_CHECK = [
    "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
    "Reboiling_Steam_Flow", "Reflux_Flow",
]
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
ALL_NUMERIC = list(COLUMN_RENAME.values())[1:]

raw = pd.read_excel(RAW_FILE, sheet_name="Sheet2")
df = raw.iloc[2:].copy().reset_index(drop=True)
df = df.rename(columns=COLUMN_RENAME)
df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
for col in ALL_NUMERIC:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
df = df[~(df[PROCESS_COLS_FOR_SHUTDOWN_CHECK] < 0.5).all(axis=1)].reset_index(drop=True)

# Let's check target clips
target_clipped_mask = pd.Series(False, index=df.index)
for col in ["C4H6_Bottom", "C4H8_Bottom"]:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    col_clipped = (df[col] < lo) | (df[col] > hi)
    target_clipped_mask = target_clipped_mask | col_clipped

df["Total_C4"] = df["C4H6_Bottom"] + df["C4H8_Bottom"]

print("Target-only clipped rows count:", target_clipped_mask.sum())
target_extreme = df[target_clipped_mask]
print("Target-only clipped Total_C4 description:")
print(target_extreme["Total_C4"].describe())
print("Target-only clipped Data_Block value counts:")
from data_preprocessing import assign_blocks
df["Data_Block"] = assign_blocks(df["DateTime"], 24)
print(df[target_clipped_mask]["Data_Block"].value_counts())
