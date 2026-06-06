"""
run_anchor_analysis.py
======================
Quantifies the model's dependence on the campaign anchor by evaluating
different ffill limits (6h, 12h, 24h, 48h, 72h) and calculating the anchor's
coverage (percentage of non-null rows) in train and test blocks.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

FEATURES_FILE = "data/features.parquet"

def main():
    df = pd.read_parquet(FEATURES_FILE)
    
    # Define splits
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    # Base features for Deviations & Ratios (Subset 7)
    dev_and_ratios = [
        "Reflux_Ratio", "Steam_Feed_Ratio",
        "Reboiling_Steam_Flow_dev24h", "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h", "Control_Tray_Temp_dev24h", "Column_Top_Pressure_dev24h"
    ]
    
    limits = [6, 12, 24, 48, 72]
    
    coverage_records = []
    performance_records = []
    
    print("=" * 80)
    print("RUNNING CAMPAIGN ANCHOR SENSITIVITY & COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Target values for ffill calculation
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    
    for limit in limits:
        # 1. Compute campaign anchor for this limit (leak-free)
        anchor_col = f"C4H8_anchor_{limit}h"
        df[anchor_col] = df.groupby("Data_Block")["C4H8_last_valid"].shift(1).ffill(limit=limit)
        
        # 2. Measure coverage (on ALL rows within masks, including stuck for general coverage,
        # but let's report overall coverage of the column)
        train_coverage = df[train_mask][anchor_col].notna().mean() * 100
        test_coverage  = df[test_mask][anchor_col].notna().mean() * 100
        
        coverage_records.append({
            "Limit": f"{limit}h",
            "Train Coverage (%)": train_coverage,
            "Test Coverage (%)": test_coverage
        })
        
        # 3. Train and evaluate the model
        feats = dev_and_ratios + [anchor_col]
        
        # Prepare datasets (dropna to handle lag and anchor NaNs)
        train_df = df[train_mask & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
        test_df  = df[test_mask  & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
        
        X_train, y_train = train_df[feats], train_df["C4H8_Bottom"]
        X_test, y_test   = test_df[feats], test_df["C4H8_Bottom"]
        
        # Train default XGBoost
        model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        performance_records.append({
            "Limit": f"{limit}h",
            "R2": r2,
            "MAE": mae,
            "Train Rows": len(train_df),
            "Test Rows": len(test_df)
        })
        
        print(f"Evaluated limit = {limit:<3d} | R² = {r2:+.4f} | MAE = {mae:.4f} wt% | Train Coverage = {train_coverage:.1f}% | Test Coverage = {test_coverage:.1f}%")
        
    # Print tables
    coverage_df = pd.DataFrame(coverage_records)
    performance_df = pd.DataFrame(performance_records)
    
    print("\n" + "=" * 80)
    print("1. ANCHOR COVERAGE STATISTICS")
    print("=" * 80)
    print(coverage_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("2. MODEL PERFORMANCE VS. ANCHOR LIMIT")
    print("=" * 80)
    print(performance_df[["Limit", "R2", "MAE", "Train Rows", "Test Rows"]].to_string(index=False))
    print("=" * 80)
    
    # Save results
    performance_df.to_csv("models/anchor_limit_performance.csv", index=False)
    coverage_df.to_csv("models/anchor_limit_coverage.csv", index=False)

if __name__ == "__main__":
    main()
