import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_parquet("data/features.parquet")

train_mask = df["Data_Block"].isin([1, 2, 3])
test_mask  = df["Data_Block"] == 4
mA_filter  = ~df["C4H8_Bottom_stuck"]

train_df = df[train_mask & mA_filter].dropna()
test_df  = df[test_mask & mA_filter].dropna()

# Seasons and metadata to exclude
TEMP_SEASON_COLS = ["month_cos", "month_sin", "hour_cos", "hour_sin", "Data_Block", "dayofweek"]
TARGET_LAG_COLS = (
    [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
    [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
)
META_COLS = [
    "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
    "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
    "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
    "Analyzer_Health", "is_extreme_event",
] + TARGET_LAG_COLS

features = [c for c in df.columns if c not in META_COLS and c not in TEMP_SEASON_COLS]

# Define hot and cold masks
train_hot_mask = train_df["Reboiler_Outlet_Temp"] >= 50
train_cold_mask = train_df["Reboiler_Outlet_Temp"] < 50

test_hot_mask = test_df["Reboiler_Outlet_Temp"] >= 50
test_cold_mask = test_df["Reboiler_Outlet_Temp"] < 50

print(f"Train set: {len(train_df)} rows | Hot: {train_hot_mask.sum()} | Cold: {train_cold_mask.sum()}")
print(f"Test set:  {len(test_df)} rows | Hot: {test_hot_mask.sum()} | Cold: {test_cold_mask.sum()}")

def evaluate_run(name, train_sub, test_sub):
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb.fit(train_sub[features], train_sub["C4H8_Bottom"])
    preds = xgb.predict(test_sub[features])
    r2 = r2_score(test_sub["C4H8_Bottom"], preds)
    mae = mean_absolute_error(test_sub["C4H8_Bottom"], preds)
    print(f"{name:50s} | R2: {r2:8.4f} | MAE: {mae:8.4f}")

# Experiment 1: Train all, test all
evaluate_run("Train: All -> Test: All", train_df, test_df)

# Experiment 2: Train all, test hot
evaluate_run("Train: All -> Test: Hot", train_df, test_df[test_hot_mask])

# Experiment 3: Train all, test cold
evaluate_run("Train: All -> Test: Cold", train_df, test_df[test_cold_mask])

# Experiment 4: Train hot, test hot
evaluate_run("Train: Hot -> Test: Hot", train_df[train_hot_mask], test_df[test_hot_mask])

# Experiment 5: Train cold, test cold
evaluate_run("Train: Cold -> Test: Cold", train_df[train_cold_mask], test_df[test_cold_mask])
