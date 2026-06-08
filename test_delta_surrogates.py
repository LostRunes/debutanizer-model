import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor

DATA_FILE = "data/surrogate_data.parquet"

def main():
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return
        
    df = pd.read_parquet(DATA_FILE)
    
    # Exclude stuck/shutdown rows
    train_mask = df["Data_Block"].isin([1, 2, 3]) & (~df["C4H8_Bottom_stuck"])
    test_mask = (df["Data_Block"] == 4) & (~df["C4H8_Bottom_stuck"])
    
    # Compute 1h deltas as targets
    df["bottom_temp_delta"] = df["bottom_temp_future_t1"] - df["Column_Bottom_Temp"]
    df["tray_temp_delta"] = df["tray_temp_future_t1"] - df["Control_Tray_Temp"]
    df["pressure_delta"] = df["pressure_future_t1"] - df["Column_Top_Pressure"]
    
    # Feature set without absolute temperatures and pressures (to prevent campaign shift overfitting)
    feats_no_abs = [
        "Feed_Flow",
        "Reboiling_Steam_Flow",
        "Reflux_Flow",
        "Steam_Feed_Ratio",
        "Reflux_Ratio",
        "Reboiling_Steam_Flow_dev24h",
        "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h",
        "Control_Tray_Temp_dev24h",
        "Column_Top_Pressure_dev24h",
        "Reboiling_Steam_Flow_lag1",
        "Reflux_Flow_lag1",
        "Column_Bottom_Temp_lag1",
        "Control_Tray_Temp_lag1",
        "Column_Top_Pressure_lag1"
    ]
    
    train_df = df[train_mask].dropna(subset=feats_no_abs + ["bottom_temp_delta", "tray_temp_delta", "pressure_delta"])
    test_df = df[test_mask].dropna(subset=feats_no_abs + ["bottom_temp_delta", "tray_temp_delta", "pressure_delta"])
    
    targets = {
        "bottom_temp": ("bottom_temp_delta", "Column_Bottom_Temp"),
        "tray_temp": ("tray_temp_delta", "Control_Tray_Temp"),
        "pressure": ("pressure_delta", "Column_Top_Pressure")
    }
    
    for key, (t_col, curr_col) in targets.items():
        print(f"\n--- Delta prediction without absolute values for {key} ---")
        X_train = train_df[feats_no_abs]
        y_train = train_df[t_col].values
        X_test = test_df[feats_no_abs]
        y_test = test_df[t_col].values
        
        model = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05, verbose=0, random_state=42)
        model.fit(X_train, y_train)
        pred_deltas = model.predict(X_test)
        
        y_current_test = test_df[curr_col].values
        y_actual_future = test_df[key + "_future_t1"].values
        y_pred_absolute = y_current_test + pred_deltas
        
        r2_abs = r2_score(y_actual_future, y_pred_absolute)
        mae_abs = mean_absolute_error(y_actual_future, y_pred_absolute)
        
        r2_naive = r2_score(y_actual_future, y_current_test)
        mae_naive = mean_absolute_error(y_actual_future, y_current_test)
        
        print(f"  Naive Absolute: R² = {r2_naive:.5f}, MAE = {mae_naive:.5f}")
        print(f"  Delta model Absolute: R² = {r2_abs:.5f}, MAE = {mae_abs:.5f}")

if __name__ == "__main__":
    main()
