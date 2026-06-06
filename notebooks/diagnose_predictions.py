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

TIER1_PROCESS_ONLY = [c for c in df.columns if c not in META_COLS and c not in TEMP_SEASON_COLS]

xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
xgb.fit(train_df[TIER1_PROCESS_ONLY], y_train)

# Evaluate on train
pred_train = xgb.predict(train_df[TIER1_PROCESS_ONLY])
print("=== TRAIN EVALUATION ===")
print(f"R2 Train: {r2_score(y_train, pred_train):.4f}")
print(f"MAE Train: {mean_absolute_error(y_train, pred_train):.4f}")
print("True Train sample (first 10):", y_train.head(10).values)
print("Pred Train sample (first 10):", pred_train[:10])

# Evaluate on test
pred_test = xgb.predict(test_df[TIER1_PROCESS_ONLY])
print("\n=== TEST EVALUATION ===")
print(f"R2 Test: {r2_score(y_test, pred_test):.4f}")
print(f"MAE Test: {mean_absolute_error(y_test, pred_test):.4f}")
print("True Test sample (first 10):", y_test.head(10).values)
print("Pred Test sample (first 10):", pred_test[:10])

print("\nMean of True Test:", y_test.mean())
print("Mean of Pred Test:", pred_test.mean())
print("Min of Pred Test:", pred_test.min())
print("Max of Pred Test:", pred_test.max())

# Find out if features in test set are out of bounds of training set
out_of_bounds_counts = {}
for col in TIER1_PROCESS_ONLY:
    train_min = train_df[col].min()
    train_max = train_df[col].max()
    test_vals = test_df[col]
    oob = (test_vals < train_min) | (test_vals > train_max)
    oob_pct = oob.mean() * 100
    if oob_pct > 1.0:
        out_of_bounds_counts[col] = (train_min, train_max, test_vals.min(), test_vals.max(), oob_pct)

print("\n=== OUT OF BOUNDS FEATURES IN TEST SET (percentage of test set rows outside train min/max) ===")
for col, (tr_min, tr_max, te_min, te_max, pct) in sorted(out_of_bounds_counts.items(), key=lambda x: x[1][4], reverse=True):
    print(f"{col:30s} | Train: [{tr_min:8.2f}, {tr_max:8.2f}] | Test: [{te_min:8.2f}, {te_max:8.2f}] | OOB: {pct:6.2f}%")
