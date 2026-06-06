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

y_train = train_df["C4H8_Bottom"]
y_test  = test_df["C4H8_Bottom"]

# 1. Base TIER1 features (including months/blocks)
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

TIER1_ALL = [c for c in df.columns if c not in META_COLS]

# 2. Exclude seasonal/temporal/block features
TEMP_SEASON_COLS = ["month_cos", "month_sin", "hour_cos", "hour_sin", "Data_Block", "dayofweek"]
TIER1_PROCESS_ONLY = [c for c in TIER1_ALL if c not in TEMP_SEASON_COLS]

print(f"Total Tier 1 features: {len(TIER1_ALL)}")
print(f"Process-only features: {len(TIER1_PROCESS_ONLY)}")

# Train with all features
xgb_all = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
xgb_all.fit(train_df[TIER1_ALL], y_train)
pred_all = xgb_all.predict(test_df[TIER1_ALL])
print(f"XGBoost (with month/block): R2 = {r2_score(y_test, pred_all):.4f} | MAE = {mean_absolute_error(y_test, pred_all):.4f}")

# Train with process-only features
xgb_proc = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
xgb_proc.fit(train_df[TIER1_PROCESS_ONLY], y_train)
pred_proc = xgb_proc.predict(test_df[TIER1_PROCESS_ONLY])
print(f"XGBoost (process-only):      R2 = {r2_score(y_test, pred_proc):.4f} | MAE = {mean_absolute_error(y_test, pred_proc):.4f}")

# Train with RandomForest process-only
from sklearn.ensemble import RandomForestRegressor
rf_proc = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_proc.fit(train_df[TIER1_PROCESS_ONLY], y_train)
pred_rf = rf_proc.predict(test_df[TIER1_PROCESS_ONLY])
print(f"RandomForest (process-only): R2 = {r2_score(y_test, pred_rf):.4f} | MAE = {mean_absolute_error(y_test, pred_rf):.4f}")

# Check top features in process-only
xgb_proc_imp = pd.DataFrame({
    "Feature": TIER1_PROCESS_ONLY,
    "Importance": xgb_proc.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("\nTop 10 features (process-only):")
print(xgb_proc_imp.head(10).round(4).to_string(index=False))
