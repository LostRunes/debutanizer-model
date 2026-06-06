import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

FEATURES_FILE = "data/features.parquet"

def main():
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded dataset shape: {df.shape}")
    
    # 1. Define Campaign Anchor for C4H6 (leak-free shift(1))
    df["C4H6_last_valid"] = df["C4H6_Bottom"].copy()
    # Set stuck or zero readings to NaN for anchor computation
    df.loc[df["C4H6_Bottom_stuck"] | (df["C4H6_Bottom"] <= 0.001), "C4H6_last_valid"] = np.nan
    
    # Target anchor with 72h limit
    df["C4H6_campaign_anchor_72h"] = (
        df.groupby("Data_Block")["C4H6_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=72))
    )
    
    # Target anchor with 12h limit
    df["C4H6_campaign_anchor_12h"] = (
        df.groupby("Data_Block")["C4H6_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=12))
    )
    
    # Features lists
    feats_72h = [
        "C4H6_campaign_anchor_72h",
        "Steam_Feed_Ratio",
        "Reflux_Ratio",
        "Reboiling_Steam_Flow_dev24h",
        "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h",
        "Control_Tray_Temp_dev24h",
        "Column_Top_Pressure_dev24h"
    ]
    
    feats_12h = [
        "C4H6_campaign_anchor_12h",
        "Steam_Feed_Ratio",
        "Reflux_Ratio",
        "Reboiling_Steam_Flow_dev24h",
        "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h",
        "Control_Tray_Temp_dev24h",
        "Column_Top_Pressure_dev24h"
    ]
    
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mB_filter = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)
    
    print("\n--- EVALUATING MODEL B WITH 72H ANCHOR ---")
    train_clean_72 = df[train_mask & mB_filter].dropna(subset=feats_72h + ["C4H6_Bottom"])
    test_clean_72  = df[test_mask  & mB_filter].dropna(subset=feats_72h + ["C4H6_Bottom"])
    
    X_train_72, y_train_72 = train_clean_72[feats_72h], train_clean_72["C4H6_Bottom"].values
    X_test_72, y_test_72   = test_clean_72[feats_72h], test_clean_72["C4H6_Bottom"].values
    
    print(f"Train size: {X_train_72.shape[0]} | Test size: {X_test_72.shape[0]}")
    
    model_72 = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model_72.fit(X_train_72, y_train_72)
    preds_72 = model_72.predict(X_test_72)
    
    r2_72 = r2_score(y_test_72, preds_72)
    mae_72 = mean_absolute_error(y_test_72, preds_72)
    pearson_72, _ = pearsonr(y_test_72, preds_72)
    
    print(f"Results (72h limit):")
    print(f"  Pearson: {pearson_72:+.4f}")
    print(f"  R² Score: {r2_72:.4f}")
    print(f"  MAE:      {mae_72:.4f} wt%")
    
    print("\n--- EVALUATING MODEL B WITH 12H ANCHOR ---")
    train_clean_12 = df[train_mask & mB_filter].dropna(subset=feats_12h + ["C4H6_Bottom"])
    test_clean_12  = df[test_mask  & mB_filter].dropna(subset=feats_12h + ["C4H6_Bottom"])
    
    X_train_12, y_train_12 = train_clean_12[feats_12h], train_clean_12["C4H6_Bottom"].values
    X_test_12, y_test_12   = test_clean_12[feats_12h], test_clean_12["C4H6_Bottom"].values
    
    print(f"Train size: {X_train_12.shape[0]} | Test size: {X_test_12.shape[0]}")
    
    model_12 = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model_12.fit(X_train_12, y_train_12)
    preds_12 = model_12.predict(X_test_12)
    
    r2_12 = r2_score(y_test_12, preds_12)
    mae_12 = mean_absolute_error(y_test_12, preds_12)
    pearson_12, _ = pearsonr(y_test_12, preds_12)
    
    print(f"Results (12h limit):")
    print(f"  Pearson: {pearson_12:+.4f}")
    print(f"  R² Score: {r2_12:.4f}")
    print(f"  MAE:      {mae_12:.4f} wt%")
    
    # Check anchor coverage on Block 4
    coverage_72h = df[test_mask & mB_filter]["C4H6_campaign_anchor_72h"].notna().mean() * 100
    coverage_12h = df[test_mask & mB_filter]["C4H6_campaign_anchor_12h"].notna().mean() * 100
    print(f"\nC4H6 Anchor Coverage on Block 4 Test Set:")
    print(f"  72h limit: {coverage_72h:.2f}%")
    print(f"  12h limit: {coverage_12h:.2f}%")

if __name__ == "__main__":
    main()
