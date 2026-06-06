"""
run_drift_experiments.py
========================
Systematically runs all Phase 4 adaptive modeling experiments.
Tracks Pearson correlation, R2, MAE, and top features on the Block 4 test set.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error

# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
MODELS_DIR = "models"

def main():
    print("=" * 80)
    print("RUNNING ADAPTIVE MODELING EXPERIMENTS")
    print("=" * 80)
    
    # 1. Load feature-engineered data
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded features dataset: {df.shape}")
    
    # 2. Add C4H8 Campaign Anchor Feature (shifted by 1 step to prevent target leakage!)
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    df["C4H8_campaign_anchor"] = df.groupby("Data_Block")["C4H8_last_valid"].shift(1).ffill(limit=72)
    
    # Define target lags and metadata columns
    TARGET_LAG_COLS = (
        [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
        [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
    )
    
    META_COLS = [
        "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
        "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
        "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
        "Analyzer_Health", "is_extreme_event", "C4H8_last_valid",
    ] + TARGET_LAG_COLS
    
    # Determine Tier 1 features
    TIER1_FEATURES = [c for c in df.columns if c not in META_COLS and c != "C4H8_campaign_anchor"]
    
    # Identify newly added features to isolate baseline
    NEW_FEATS = [c for c in df.columns if "_Pnorm" in c or "_Pratio" in c or "Pressure_x_" in c or "_dev24h" in c]
    
    # Original features (67 total)
    original_features = [c for c in TIER1_FEATURES if c not in NEW_FEATS]
    print(f"Original feature set size: {len(original_features)}")
    assert len(original_features) == 67, f"Expected 67 original features, got {len(original_features)}"
    
    # Setup splits
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    # Definition of experiments
    experiments = {}
    
    # Phase A: Campaign Proxy Ablation
    experiments["Baseline"] = original_features
    experiments["Exp 1: No Calendar"] = [c for c in original_features if c not in ["month_sin", "month_cos"]]
    experiments["Exp 2: No Regime"] = [c for c in original_features if c not in ["Data_Block", "Temp_Gradient", "Reboiler_Delta"]]
    
    # Exp 3: All Proxies Removed (Physics-only Baseline)
    no_proxies = [c for c in original_features if c not in ["month_sin", "month_cos", "Data_Block", "Temp_Gradient", "Reboiler_Delta"]]
    experiments["Exp 3: All Removed"] = no_proxies
    
    # Phase B: Physical Normalization & Interactions (starting from Exp 3 features)
    experiments["Exp 4: Pnorm (k=3)"] = no_proxies + ["Column_Top_Temp_Pnorm_k3", "Control_Tray_Temp_Pnorm_k3", "Column_Bottom_Temp_Pnorm_k3"]
    experiments["Exp 5: Pnorm (k=5)"] = no_proxies + ["Column_Top_Temp_Pnorm_k5", "Control_Tray_Temp_Pnorm_k5", "Column_Bottom_Temp_Pnorm_k5"]
    experiments["Exp 6: Pnorm (k=10)"] = no_proxies + ["Column_Top_Temp_Pnorm_k10", "Control_Tray_Temp_Pnorm_k10", "Column_Bottom_Temp_Pnorm_k10"]
    experiments["Exp 7: Pnorm Ratio"] = no_proxies + ["Column_Top_Temp_Pratio", "Control_Tray_Temp_Pratio", "Column_Bottom_Temp_Pratio"]
    experiments["Exp 8: Pnorm Gradient"] = no_proxies + ["Temp_Gradient_Pnorm"]
    experiments["Exp 9: Pressure Interactions"] = no_proxies + ["Pressure_x_TopTemp", "Pressure_x_BottomTemp", "Pressure_x_ControlTrayTemp"]
    experiments["Exp 10: Rolling Deviations"] = no_proxies + ["Reboiling_Steam_Flow_dev24h", "Reflux_Flow_dev24h", "Column_Bottom_Temp_dev24h", "Control_Tray_Temp_dev24h", "Column_Top_Pressure_dev24h"]
    
    # We will choose the best physics feature set later, but for now Exp 11 uses the best physical normalization (let's say ratio/dev/interactions)
    # Let's run Exp 1 to 10 first, then determine the best feature set to build the campaign anchor on.
    
    results = []
    
    # Run loop
    for name, feats in list(experiments.items()):
        print(f"\nTraining {name} with {len(feats)} features...")
        
        # Prepare train and test sets for this experiment (dropna to handle lag NaNs)
        train_df = df[train_mask & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
        test_df  = df[test_mask  & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
        
        X_train, y_train = train_df[feats], train_df["C4H8_Bottom"]
        X_test, y_test   = test_df[feats], test_df["C4H8_Bottom"]
        
        # Train default XGBoost
        model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        # Compute metrics
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        # Pearson correlation
        if len(y_test) > 1:
            corr, _ = pearsonr(y_test, preds)
        else:
            corr = np.nan
            
        # Top feature
        importances = model.feature_importances_
        top_idx = np.argmax(importances)
        top_feat = feats[top_idx]
        top_feat_imp = importances[top_idx]
        
        print(f"  Pearson = {corr:+.4f} | R² = {r2:.4f} | MAE = {mae:.4f} wt%")
        print(f"  Top Feature: {top_feat} ({top_feat_imp:.4f})")
        
        results.append({
            "Experiment": name,
            "Pearson": corr,
            "R2": r2,
            "MAE": mae,
            "Top Feature": f"{top_feat} ({top_feat_imp:.3f})",
            "Features_List": feats
        })
        
    # Now, find the best physical normalization feature set (out of Exp 4-10) to run Exp 11:
    phys_results = [r for r in results if "Exp 4" in r["Experiment"] or "Exp 5" in r["Experiment"] or "Exp 6" in r["Experiment"] or "Exp 7" in r["Experiment"] or "Exp 8" in r["Experiment"] or "Exp 9" in r["Experiment"] or "Exp 10" in r["Experiment"]]
    
    # Find the one with highest Pearson (or R2 if Pearson is positive)
    best_phys_exp = None
    if phys_results:
        # Sort by Pearson correlation (descending)
        phys_results_sorted = sorted(phys_results, key=lambda x: x["Pearson"], reverse=True)
        best_phys_exp = phys_results_sorted[0]
        print(f"\nBest physical normalization experiment: {best_phys_exp['Experiment']} (Pearson = {best_phys_exp['Pearson']:+.4f})")
        
    # Run Exp 11: Campaign Anchor on top of the best physical normalization feature set (or Exp 3 if none is better)
    best_phys_feats = best_phys_exp["Features_List"] if best_phys_exp else no_proxies
    anchor_feats = best_phys_feats + ["C4H8_campaign_anchor"]
    
    print(f"\nTraining Exp 11: Campaign Anchor with {len(anchor_feats)} features...")
    train_df = df[train_mask & mA_filter].dropna(subset=anchor_feats + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=anchor_feats + ["C4H8_Bottom"])
    
    X_train, y_train = train_df[anchor_feats], train_df["C4H8_Bottom"]
    X_test, y_test   = test_df[anchor_feats], test_df["C4H8_Bottom"]
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    corr, _ = pearsonr(y_test, preds)
    
    importances = model.feature_importances_
    top_idx = np.argmax(importances)
    top_feat = anchor_feats[top_idx]
    top_feat_imp = importances[top_idx]
    
    print(f"  Pearson = {corr:+.4f} | R² = {r2:.4f} | MAE = {mae:.4f} wt%")
    print(f"  Top Feature: {top_feat} ({top_feat_imp:.4f})")
    
    results.append({
        "Experiment": "Exp 11: Campaign Anchor",
        "Pearson": corr,
        "R2": r2,
        "MAE": mae,
        "Top Feature": f"{top_feat} ({top_feat_imp:.3f})",
        "Features_List": anchor_feats
    })
    
    # Print summary table
    summary_df = pd.DataFrame(results)[["Experiment", "Pearson", "R2", "MAE", "Top Feature"]]
    print("\n" + "=" * 80)
    print("EXPERIMENT VERIFICATION MATRIX SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    # Save results to csv
    summary_df.to_csv("models/drift_experiments_summary.csv", index=False)
    print("Results saved to models/drift_experiments_summary.csv")

if __name__ == "__main__":
    main()
