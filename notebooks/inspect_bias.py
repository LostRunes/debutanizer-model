import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

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

features = [c for c in df.columns if c not in META_COLS and c not in TEMP_SEASON_COLS]

xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
xgb.fit(train_df[features], y_train)

pred_test = xgb.predict(test_df[features])

corr, _ = pearsonr(y_test, pred_test)
print(f"Pearson Correlation between y_test and pred_test: {corr:.4f}")

# Check R2 if we adjust by the mean difference (bias correction)
bias = y_test.mean() - pred_test.mean()
pred_test_bias_corrected = pred_test + bias
r2_corrected = r2_score(y_test, pred_test_bias_corrected)
mae_corrected = mean_absolute_error(y_test, pred_test_bias_corrected)
print(f"Bias Corrected (shifted by {bias:+.4f}) | R2: {r2_corrected:.4f} | MAE: {mae_corrected:.4f}")

# Check if we fit a simple linear regression from pred_test to y_test
from sklearn.linear_model import LinearRegression
calibrator = LinearRegression()
calibrator.fit(pred_test.reshape(-1, 1), y_test)
pred_test_calibrated = calibrator.predict(pred_test.reshape(-1, 1))
r2_calibrated = r2_score(y_test, pred_test_calibrated)
mae_calibrated = mean_absolute_error(y_test, pred_test_calibrated)
print(f"Linear Calibrated (slope: {calibrator.coef_[0]:.4f}, intercept: {calibrator.intercept_:.4f}) | R2: {r2_calibrated:.4f} | MAE: {mae_calibrated:.4f}")
