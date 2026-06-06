"""
run_feature_ablation_study.py
=============================
Trains XGBoost models on different feature subsets to isolate which variables
generalize and whether a leak-free campaign anchor restores model direction.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error

FEATURES_FILE = "data/features.parquet"

def evaluate_subset(df, name, feats, train_mask, test_mask, mA_filter):
    # Prepare train/test sets
    train_df = df[train_mask & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    
    X_train, y_train = train_df[feats], train_df["C4H8_Bottom"]
    X_test, y_test   = test_df[feats], test_df["C4H8_Bottom"]
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    corr, _ = pearsonr(y_test, preds) if len(y_test) > 1 else (np.nan, None)
    
    # Feature importances
    importances = model.feature_importances_
    top_idx = np.argmax(importances)
    top_feat = feats[top_idx]
    
    print(f"Subset: {name} ({len(feats)} features)")
    print(f"  Pearson: {corr:+.4f} | R²: {r2:.4f} | MAE: {mae:.4f} wt%")
    print(f"  Top Feature: {top_feat} ({importances[top_idx]:.4f})")
    print("-" * 60)
    
    return {
        "Subset": name,
        "Features": len(feats),
        "Pearson": corr,
        "R2": r2,
        "MAE": mae,
        "Top Feature": f"{top_feat} ({importances[top_idx]:.3f})"
    }

def main():
    df = pd.read_parquet(FEATURES_FILE)
    
    # 1. Compute leak-free campaign anchor (shifted by 1 step!)
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    # IMPORTANT: shift by 1 before ffill to prevent target leakage!
    df["C4H8_campaign_anchor"] = df.groupby("Data_Block")["C4H8_last_valid"].shift(1).ffill(limit=72)
    
    # Setup masks
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    # Base TIER1 features list
    TARGET_LAG_COLS = (
        [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
        [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
    )
    META_COLS = [
        "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
        "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
        "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
        "Analyzer_Health", "is_extreme_event", "C4H8_last_valid", "C4H8_campaign_anchor"
    ] + TARGET_LAG_COLS
    
    all_process_features = [c for c in df.columns if c not in META_COLS]
    
    # Define features to exclude for different subsets
    temp_related_cols = [c for c in all_process_features if "Temp" in c or "temp" in c or "Gradient" in c or "Delta" in c]
    unstable_temps = [c for c in temp_related_cols if "Bottom" not in c and "Outlet" not in c]
    
    calendar_cols = ["month_sin", "month_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    proxies = calendar_cols + ["Data_Block"]
    
    # Define Subsets
    subsets = {}
    
    # 1. Baseline TIER1 (process features, no target lags, including month_cos, Data_Block, and raw temperatures)
    subsets["1. Baseline TIER1"] = [c for c in all_process_features if "_Pnorm" not in c and "_Pratio" not in c and "Pressure_x_" not in c and "_dev24h" not in c]
    
    # 2. Physics Only (excluding all temperature variables, and removing proxies)
    physics_only = [c for c in all_process_features if c not in temp_related_cols and c not in proxies]
    subsets["2. Physics Only (No Temps)"] = physics_only
    
    # 3. Physics + Stable Temps (excluding unstable top/tray temps, keeping bottom/outlet temp, removing proxies)
    stable_temps_feats = [c for c in all_process_features if c not in unstable_temps and c not in proxies]
    subsets["3. Physics + Stable Temps"] = stable_temps_feats
    
    # 4. Deviations & Ratios Only (excluding absolute variables that shift between campaigns)
    dev_and_ratios = [
        "Reflux_Ratio", "Steam_Feed_Ratio",
        "Reboiling_Steam_Flow_dev24h", "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h", "Control_Tray_Temp_dev24h", "Column_Top_Pressure_dev24h"
    ]
    subsets["4. Deviations & Ratios Only"] = dev_and_ratios
    
    # 5. Physics + Stable Temps + Leak-Free Campaign Anchor
    subsets["5. Physics + Stable Temps + Campaign Anchor"] = stable_temps_feats + ["C4H8_campaign_anchor"]
    
    # 6. Physics Only (No Temps) + Campaign Anchor
    subsets["6. Physics Only (No Temps) + Campaign Anchor"] = physics_only + ["C4H8_campaign_anchor"]
    
    # 7. Deviations & Ratios + Campaign Anchor
    subsets["7. Deviations & Ratios + Campaign Anchor"] = dev_and_ratios + ["C4H8_campaign_anchor"]
    
    print("=" * 80)
    print("RUNNING FEATURE ABLATION STUDY (MODEL A - C4H8)")
    print("=" * 80)
    
    results = []
    for name, feats in subsets.items():
        res = evaluate_subset(df, name, feats, train_mask, test_mask, mA_filter)
        results.append(res)
        
    summary_df = pd.DataFrame(results)[["Subset", "Features", "Pearson", "R2", "MAE", "Top Feature"]]
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    summary_df.to_csv("models/ablation_study_summary.csv", index=False)
    print("Ablation study summary saved to models/ablation_study_summary.csv")

if __name__ == "__main__":
    main()
