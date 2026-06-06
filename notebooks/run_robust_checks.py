"""
notebooks/run_robust_checks.py
==============================
Performs the 3 final checks requested by the user:
1. Task 1: Train robust 8-feature model on Blocks 1+2, test on Block 3. Report Pearson, R2, MAE and save diagnostics.
2. Task 2: Calculate target anchor coverage % specifically for Block 4 (72h limit and 12h limit).
3. Task 3: Generate Actual-vs-Predicted and Residual-vs-Time plots for the robust 8-feature model tested on Block 4.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

FEATURES_FILE = "data/features.parquet"
DIAGNOSTICS_DIR = "experiments/diagnostics"
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

def main():
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded dataset: {df.shape}")
    
    # 1. Define Campaign Anchors (leak-free shift(1))
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    
    # Target anchor with 72h limit (from Subset 7)
    df["C4H8_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H8_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=72))
    )
    # Target anchor with 12h limit (from our optimized run)
    df["C4H8_campaign_anchor_12h"] = (
        df.groupby("Data_Block")["C4H8_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=12))
    )
    
    # Core robust 8-feature set (Subset 7)
    feats = [
        "C4H8_campaign_anchor",
        "Steam_Feed_Ratio",
        "Reflux_Ratio",
        "Reboiling_Steam_Flow_dev24h",
        "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h",
        "Control_Tray_Temp_dev24h",
        "Column_Top_Pressure_dev24h"
    ]
    
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    print("\n" + "=" * 80)
    print("TASK 1: RUNNING ROBUST MODEL ON ANOTHER SPLIT (TRAIN: BLOCKS 1+2, TEST: BLOCK 3)")
    print("=" * 80)
    
    # Masks for Task 1
    train_mask_t1 = df["Data_Block"].isin([1, 2])
    test_mask_t1  = df["Data_Block"] == 3
    
    # Prepare datasets for Task 1
    t1_train_df = df[train_mask_t1 & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    t1_test_df  = df[test_mask_t1  & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    
    X_train_t1, y_train_t1 = t1_train_df[feats], t1_train_df["C4H8_Bottom"].values
    X_test_t1, y_test_t1   = t1_test_df[feats], t1_test_df["C4H8_Bottom"].values
    
    print(f"Train set (Blocks 1+2): {X_train_t1.shape} | Test set (Block 3): {X_test_t1.shape}")
    
    # Train baseline XGBoost
    model_t1 = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model_t1.fit(X_train_t1, y_train_t1)
    preds_t1 = model_t1.predict(X_test_t1)
    
    # Metrics
    t1_r2 = r2_score(y_test_t1, preds_t1)
    t1_mae = mean_absolute_error(y_test_t1, preds_t1)
    t1_pearson, _ = pearsonr(y_test_t1, preds_t1)
    
    print(f"\nRESULTS ON BLOCK 3 TEST:")
    print(f"  Pearson: {t1_pearson:+.4f}")
    print(f"  R² Score: {t1_r2:.4f}")
    print(f"  MAE:      {t1_mae:.4f} wt%")
    
    # Generate Plots for Task 1
    # Plot 1.1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_t1, preds_t1, alpha=0.5, color="purple", edgecolors="w", s=40)
    plt.plot([y_test_t1.min(), y_test_t1.max()], [y_test_t1.min(), y_test_t1.max()], "r--", lw=2)
    plt.xlabel("Actual C4H8_Bottom (wt%)")
    plt.ylabel("Predicted C4H8_Bottom (wt%)")
    plt.title("Task 1 (Block 3 Test): Actual vs. Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_split3_plot_1_actual_vs_predicted.png"))
    plt.close()
    
    # Plot 1.2: Residual vs Time
    t1_residuals = y_test_t1 - preds_t1
    plt.figure(figsize=(12, 6))
    plt.plot(t1_test_df["DateTime"], t1_residuals, alpha=0.6, color="purple", lw=1.5)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("DateTime")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Task 1 (Block 3 Test): Residual vs. Time")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_split3_plot_4_residual_vs_time.png"))
    plt.close()
    
    # -------------------------------------------------------------------------
    # TASK 2: MEASURE ANCHOR AVAILABILITY FOR BLOCK 4
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 2: MEASURING TARGET ANCHOR AVAILABILITY ON BLOCK 4")
    print("=" * 80)
    
    block4_mask = df["Data_Block"] == 4
    
    # Check total rows in Block 4 (valid target filter applied)
    b4_total = len(df[block4_mask])
    b4_target_valid = len(df[block4_mask & mA_filter])
    
    # Anchor coverage on valid target rows
    coverage_72h = df[block4_mask & mA_filter]["C4H8_campaign_anchor"].notna().mean() * 100
    coverage_12h = df[block4_mask & mA_filter]["C4H8_campaign_anchor_12h"].notna().mean() * 100
    
    # Anchor coverage on ALL Block 4 rows (including target stuck rows for true availability)
    coverage_72h_all = df[block4_mask]["C4H8_campaign_anchor"].notna().mean() * 100
    coverage_12h_all = df[block4_mask]["C4H8_campaign_anchor_12h"].notna().mean() * 100
    
    print(f"Block 4 Dataset size: {b4_total} rows | Valid target analyzer rows: {b4_target_valid}")
    print(f"\nTarget Anchor Coverage (limit = 72h):")
    print(f"  On valid target rows: {coverage_72h:.2f}%")
    print(f"  On all Block 4 rows:  {coverage_72h_all:.2f}%")
    print(f"\nTarget Anchor Coverage (limit = 12h):")
    print(f"  On valid target rows: {coverage_12h:.2f}%")
    print(f"  On all Block 4 rows:  {coverage_12h_all:.2f}%")
    
    # -------------------------------------------------------------------------
    # TASK 3: DIAGNOSTIC PLOTS FOR ROBUST MODEL ON BLOCK 4 (TRAIN: 1,2,3 -> TEST: 4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 3: ROBUST 8-FEATURE DIAGNOSTIC PLOTS ON BLOCK 4 TEST SET")
    print("=" * 80)
    
    train_mask_t3 = df["Data_Block"].isin([1, 2, 3])
    test_mask_t3  = df["Data_Block"] == 4
    
    # Prepare datasets for Task 3
    t3_train_df = df[train_mask_t3 & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    t3_test_df  = df[test_mask_t3  & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    
    X_train_t3, y_train_t3 = t3_train_df[feats], t3_train_df["C4H8_Bottom"].values
    X_test_t3, y_test_t3   = t3_test_df[feats], t3_test_df["C4H8_Bottom"].values
    
    print(f"Train set (Blocks 1-3): {X_train_t3.shape} | Test set (Block 4): {X_test_t3.shape}")
    
    # Train default XGBoost
    model_t3 = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model_t3.fit(X_train_t3, y_train_t3)
    preds_t3 = model_t3.predict(X_test_t3)
    
    t3_r2 = r2_score(y_test_t3, preds_t3)
    t3_mae = mean_absolute_error(y_test_t3, preds_t3)
    t3_pearson, _ = pearsonr(y_test_t3, preds_t3)
    
    print(f"\nRESULTS ON BLOCK 4 TEST (72h LIMIT):")
    print(f"  Pearson: {t3_pearson:+.4f}")
    print(f"  R² Score: {t3_r2:.4f}")
    print(f"  MAE:      {t3_mae:.4f} wt%")
    
    # Generate Plots for Task 3
    # Plot 3.1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_t3, preds_t3, alpha=0.5, color="darkgreen", edgecolors="w", s=40)
    plt.plot([y_test_t3.min(), y_test_t3.max()], [y_test_t3.min(), y_test_t3.max()], "r--", lw=2)
    plt.xlabel("Actual C4H8_Bottom (wt%)")
    plt.ylabel("Predicted C4H8_Bottom (wt%)")
    plt.title("Subset 7 (Block 4 Test): Actual vs. Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_block4_plot_1_actual_vs_predicted.png"))
    plt.close()
    
    # Plot 3.2: Residual vs Time
    t3_residuals = y_test_t3 - preds_t3
    plt.figure(figsize=(12, 6))
    plt.plot(t3_test_df["DateTime"], t3_residuals, alpha=0.6, color="forestgreen", lw=1.5)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("DateTime")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Subset 7 (Block 4 Test): Residual vs. Time")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_block4_plot_4_residual_vs_time.png"))
    plt.close()
    
    print("\nAll diagnostic plots successfully generated and saved to experiments/diagnostics/")

if __name__ == "__main__":
    main()
