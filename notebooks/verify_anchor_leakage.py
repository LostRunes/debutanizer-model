"""
notebooks/verify_anchor_leakage.py
===================================
Formal verification script to:
1. Train the robust 8-feature Model A on Blocks 1 & 2.
2. Predict and evaluate on Block 3.
3. Verify that the R² matches 0.7694 and MAE matches 0.0817 wt%.
4. Programmatically prove that the campaign anchor feature is leak-free (no target look-ahead).

Purpose: Reviewer audit tool for target leakage validation.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

FEATURES_FILE = "data/features.parquet"

def main():
    print("================================================================================")
    # 1. Load data
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found. Run preprocessing and feature engineering first.")
        return

    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded dataset: {df.shape} rows.")

    # 2. Re-compute the leak-free campaign anchor to ensure we document the exact logic
    # Set stuck analyzer values to NaN
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan

    # Shift by 1 BEFORE forward-filling to prevent data leakage (predicting t using t's value)
    df["C4H8_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H8_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=72))
    )

    # 3. Define the final 8 features
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

    print("\nProduction Features:")
    for f in feats:
        print(f"  - {f}")

    # 4. Programmatically verify no target leakage
    print("\n--- RUNNING PROGRAMMATIC LEAK-FREE PROOF ---")
    
    # Proof method: If we modify a valid target value y[t], it MUST NOT affect the feature value x[t] at the same timestep.
    # It should only affect feature values at times > t.
    valid_indices = df[~df["C4H8_Bottom_stuck"] & df["C4H8_campaign_anchor"].notna()].index
    test_idx = valid_indices[100]  # Pick an arbitrary index
    
    original_target_val = df.loc[test_idx, "C4H8_Bottom"]
    original_anchor_val = df.loc[test_idx, "C4H8_campaign_anchor"]
    
    # Temporarily perturb the target at test_idx
    df_perturbed = df.copy()
    df_perturbed.loc[test_idx, "C4H8_Bottom"] = original_target_val + 5.0
    df_perturbed["C4H8_last_valid"] = df_perturbed["C4H8_Bottom"].copy()
    df_perturbed.loc[df_perturbed["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    df_perturbed["C4H8_campaign_anchor"] = (
        df_perturbed.groupby("Data_Block")["C4H8_last_valid"]
                    .transform(lambda x: x.shift(1).ffill(limit=72))
    )
    
    perturbed_anchor_val = df_perturbed.loc[test_idx, "C4H8_campaign_anchor"]
    perturbed_next_anchor_val = df_perturbed.loc[test_idx + 1, "C4H8_campaign_anchor"]
    
    print(f"Original target at t={test_idx}: {original_target_val:.4f}")
    print(f"Perturbed target at t={test_idx}: {df_perturbed.loc[test_idx, 'C4H8_Bottom']:.4f}")
    print(f"Original anchor at t={test_idx}: {original_anchor_val:.4f}")
    print(f"Perturbed anchor at t={test_idx}: {perturbed_anchor_val:.4f}")
    print(f"Perturbed anchor at t={test_idx+1} (should change): {perturbed_next_anchor_val:.4f}")
    
    # Assertions for leakage check
    assert np.isclose(original_anchor_val, perturbed_anchor_val, equal_nan=True), "LEAK DETECTED: Target at t modified feature at t!"
    if not df.loc[test_idx, "C4H8_Bottom_stuck"] and (test_idx + 1) in df.index:
        assert not np.isclose(perturbed_next_anchor_val, df.loc[test_idx + 1, "C4H8_campaign_anchor"], equal_nan=True), "Verification failed: target change did not propagate to t+1."
        
    print("[VERIFIED] Programmatic proof passed: C4H8_campaign_anchor has no current-timestep leakage.")

    # 5. Split: Train = Blocks 1+2, Test = Block 3
    train_mask = df["Data_Block"].isin([1, 2])
    test_mask = df["Data_Block"] == 3
    mA_filter = ~df["C4H8_Bottom_stuck"]

    # Filter rows: drop stuck analyzer rows and drop rows containing NaNs in our 8 features or target
    train_clean = df[train_mask & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    test_clean = df[test_mask & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])

    X_train, y_train = train_clean[feats], train_clean["C4H8_Bottom"].values
    X_test, y_test = test_clean[feats], test_clean["C4H8_Bottom"].values

    print(f"\nTraining set size (Blocks 1+2): {X_train.shape[0]} rows")
    print(f"Testing set size (Block 3):     {X_test.shape[0]} rows")

    # 6. Fit the baseline XGBoost model
    # Note: Use baseline XGBoost hyperparams to verify the R² of 0.7694
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # 7. Evaluate
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    pearson_corr, _ = pearsonr(y_test, preds)

    print("\n--- EVALUATION METRICS ON BLOCK 3 TEST ---")
    print(f"R² Score:            {r2:.4f}  (Expected: 0.7694)")
    print(f"MAE:                 {mae:.4f} wt% (Expected: 0.0817)")
    print(f"Pearson Correlation: {pearson_corr:+.4f} (Expected: +0.8848)")

    # Assert correctness
    assert np.isclose(r2, 0.7694, atol=1e-4), f"R² does not match expected value. Got {r2:.4f}"
    assert np.isclose(mae, 0.0817, atol=1e-4), f"MAE does not match expected value. Got {mae:.4f}"
    print("\n[VERIFIED] Metrics match expected values exactly!")
    print("================================================================================")

if __name__ == "__main__":
    main()
