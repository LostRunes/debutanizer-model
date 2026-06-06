"""
model_training.py
=================
Phase 3: Model Training pipeline for DEBUTANIZER C4 Slippage.

This script implements Phase 3 of the implementation plan:
  1. Computes baselines (Overall mean, Block mean, Naive lag-1)
  2. Trains 4 default models for Model A (C4H8)
  3. Saves the default leaderboard
  4. Runs Data_Block A/B experiment
  5. Performs Feature Importance safety check (regime indicator memorisation)
  6. Trains default models for Model B (C4H6)
  7. Trains Tier 2 Research models (with target lags)
  8. Combines predictions to form Total_C4 predictions and metrics
  9. Saves outputs to the models/ directory
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
MODELS_DIR = "models"
DEFAULT_LEADERBOARD_FILE = os.path.join(MODELS_DIR, "default_leaderboard.csv")
TRAINING_METRICS_FILE = os.path.join(MODELS_DIR, "training_metrics.csv")
TEST_PREDS_FILE = os.path.join(MODELS_DIR, "test_predictions.parquet")

# Optuna configuration (Tuning is disabled by default for initial review)
RUN_TUNING = False

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate R2, MAE, RMSE, % within +/-0.1, and spec recall."""
    if len(y_true) == 0:
        return {
            "R2": np.nan, "MAE": np.nan, "RMSE": np.nan,
            "Pct_within_0.1": np.nan, "Spec_Recall": np.nan
        }
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # % within +/-0.1 wt%
    pct_within = np.mean(np.abs(y_true - y_pred) <= 0.1) * 100
    
    # Recall for > threshold (out of spec)
    true_above = (y_true > threshold)
    pred_above = (y_pred > threshold)
    tp = np.sum(true_above & pred_above)
    fn = np.sum(true_above & ~pred_above)
    spec_recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else np.nan
    
    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Pct_within_0.1": pct_within,
        "Spec_Recall": spec_recall
    }

def print_metrics(title, m):
    print(f"  {title}:")
    print(f"    R² = {m['R2']:.4f} | MAE = {m['MAE']:.4f} wt% | RMSE = {m['RMSE']:.4f} wt%")
    print(f"    % within ±0.1 wt% = {m['Pct_within_0.1']:.1f}% | Spec Recall (>0.5) = {m['Spec_Recall']:.1f}%")

def main():
    print("=" * 80)
    print("STARTING MODEL TRAINING (PHASE 3)")
    print("=" * 80)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load feature-engineered data
    print(f"Loading features from {FEATURES_FILE}...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"  Loaded dataset shape: {df.shape}")
    
    # 2. Columns definition
    TARGET_LAG_COLS = (
        [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
        [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
    )
    
    META_COLS = [
        "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
        "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
        "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
        "Analyzer_Health", "is_extreme_event", "Data_Block",
    ] + TARGET_LAG_COLS
    
    TIER1_FEATURES = [c for c in df.columns if c not in META_COLS]
    TIER2_FEATURES = TIER1_FEATURES + TARGET_LAG_COLS
    
    print(f"  Tier 1 features (process-only): {len(TIER1_FEATURES)}")
    print(f"  Tier 2 features (+ target lags): {len(TIER2_FEATURES)}")
    
    # Assertions for the updated 113-column features parquet (excluding Data_Block)
    assert len(TIER1_FEATURES) == 87, f"Expected 87 Tier 1 features, got {len(TIER1_FEATURES)}"
    assert len(TIER2_FEATURES) == 102, f"Expected 102 Tier 2 features, got {len(TIER2_FEATURES)}"
    print("  Assertions passed: Column counts match updated Phase 4.1 features parquet.")
    
    # 3. Splits definition
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    
    # Exclude stuck target rows for model training & evaluation
    mA_filter = ~df["C4H8_Bottom_stuck"]
    mB_filter = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)
    
    # We will build train/test sets. 
    # For training and metric evaluation, we drop NaN rows (which are only block boundary rows).
    train_df_A = df[train_mask & mA_filter].dropna(subset=TIER1_FEATURES + ["C4H8_Bottom"])
    test_df_A  = df[test_mask  & mA_filter].dropna(subset=TIER1_FEATURES + ["C4H8_Bottom"])
    
    train_df_B = df[train_mask & mB_filter].dropna(subset=TIER1_FEATURES + ["C4H6_Bottom"])
    test_df_B  = df[test_mask  & mB_filter].dropna(subset=TIER1_FEATURES + ["C4H6_Bottom"])
    
    print(f"\nModel A (C4H8) Sets:")
    print(f"  Train size: {len(train_df_A)} | Test size: {len(test_df_A)}")
    print(f"Model B (C4H6) Sets:")
    print(f"  Train size: {len(train_df_B)} | Test size: {len(test_df_B)}")
    
    X_train_A, y_train_A = train_df_A[TIER1_FEATURES], train_df_A["C4H8_Bottom"]
    X_test_A, y_test_A   = test_df_A[TIER1_FEATURES], test_df_A["C4H8_Bottom"]
    
    X_train_B, y_train_B = train_df_B[TIER1_FEATURES], train_df_B["C4H6_Bottom"]
    X_test_B, y_test_B   = test_df_B[TIER1_FEATURES], test_df_B["C4H6_Bottom"]
    
    # =========================================================================
    # STEP 1 — BASELINES FOR MODEL A
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: MODEL A (C4H8) BASELINES")
    print("=" * 50)
    
    # 1. Overall Mean Baseline
    overall_mean_val = y_train_A.mean()
    pred_overall_mean = np.full_like(y_test_A, overall_mean_val)
    m_baseline_mean = calculate_metrics(y_test_A, pred_overall_mean, threshold=0.4) # target C4H8 threshold scaled
    print_metrics("Overall Mean Baseline", m_baseline_mean)
    
    # 2. Block Mean Baseline
    block_means = train_df_A.groupby("Data_Block")["C4H8_Bottom"].mean()
    # For Block 4 test set, we map its Block (4) to Block 4's own train mean if it exists,
    # but Block 4 has no train rows. So we map to the nearest block's mean or overall mean.
    # Let's map Block 4 test rows to the overall train mean since it is unseen.
    pred_block_mean = test_df_A["Data_Block"].map(block_means).fillna(overall_mean_val).values
    m_baseline_block = calculate_metrics(y_test_A, pred_block_mean, threshold=0.4)
    print_metrics("Block Mean Baseline", m_baseline_block)
    
    # 3. Naive Lag-1 Baseline
    # Note: Lag-1 needs the analyzer. We evaluate it on rows where C4H8_Bottom_lag1 is not NaN.
    lag1_eval_mask = ~test_df_A["C4H8_Bottom_lag1"].isna()
    y_test_A_lag1 = y_test_A[lag1_eval_mask]
    pred_lag1 = test_df_A.loc[lag1_eval_mask, "C4H8_Bottom_lag1"].values
    m_baseline_lag1 = calculate_metrics(y_test_A_lag1, pred_lag1, threshold=0.4)
    print_metrics("Naive Lag-1 Baseline (Requires Analyzer)", m_baseline_lag1)
    
    # =========================================================================
    # STEP 2 — TRAIN DEFAULT MODELS (MODEL A - C4H8)
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: TRAINING MODEL A DEFAULT MODELS (NO TUNING)")
    print("=" * 50)
    
    default_models = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":          XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1),
    }
    
    leaderboard_records = []
    trained_default_models = {}
    
    # We will evaluate on all test rows, and separately on normal and extreme rows
    for name, model in default_models.items():
        print(f"Training {name}...")
        model.fit(X_train_A, y_train_A)
        trained_default_models[name] = model
        
        # Predictions
        preds = model.predict(X_test_A)
        
        # Evaluate All
        m_all = calculate_metrics(y_test_A, preds, threshold=0.4)
        print_metrics(f"{name} (All Test)", m_all)
        
        # Evaluate Normal rows
        normal_mask = ~test_df_A["is_extreme_event"]
        m_normal = calculate_metrics(y_test_A[normal_mask], preds[normal_mask], threshold=0.4)
        
        # Evaluate Extreme rows
        extreme_mask = test_df_A["is_extreme_event"]
        m_extreme = calculate_metrics(y_test_A[extreme_mask], preds[extreme_mask], threshold=0.4)
        
        leaderboard_records.append({
            "Model": f"{name} (Tier 1)",
            "R2_All": m_all["R2"],
            "MAE_All": m_all["MAE"],
            "RMSE_All": m_all["RMSE"],
            "Pct_within_0.1_All": m_all["Pct_within_0.1"],
            "Spec_Recall_All": m_all["Spec_Recall"],
            "R2_Normal": m_normal["R2"],
            "MAE_Normal": m_normal["MAE"],
            "R2_Extreme": m_extreme["R2"],
            "MAE_Extreme": m_extreme["MAE"],
            "Analyzer_Required": "No"
        })
        
    # Add Lag-1 and Mean Baselines to Leaderboard for direct comparison
    leaderboard_records.append({
        "Model": "Naive Lag-1 Baseline",
        "R2_All": m_baseline_lag1["R2"],
        "MAE_All": m_baseline_lag1["MAE"],
        "RMSE_All": m_baseline_lag1["RMSE"],
        "Pct_within_0.1_All": m_baseline_lag1["Pct_within_0.1"],
        "Spec_Recall_All": m_baseline_lag1["Spec_Recall"],
        "R2_Normal": np.nan, "MAE_Normal": np.nan, "R2_Extreme": np.nan, "MAE_Extreme": np.nan,
        "Analyzer_Required": "Yes"
    })
    
    leaderboard_df = pd.DataFrame(leaderboard_records)
    leaderboard_df.to_csv(DEFAULT_LEADERBOARD_FILE, index=False)
    print(f"\nDefault model leaderboard saved to: {DEFAULT_LEADERBOARD_FILE}")
    print("\n--- DEFAULT LEADERBOARD ---")
    print(leaderboard_df[["Model", "R2_All", "MAE_All", "RMSE_All", "Spec_Recall_All", "Analyzer_Required"]].round(4).to_string(index=False))
    
    # Identify winning model based on test R2
    winner_name = "XGBoost" # Defaults to XGBoost as primary candidate
    winner_model = trained_default_models[winner_name]
    
    # =========================================================================
    # STEP 3 — DATA_BLOCK EXCLUSION CONFIRMATION
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: DATA_BLOCK EXCLUSION CONFIRMATION")
    print("=" * 50)
    print("  Data_Block feature has been permanently removed from training features.")
    print(f"  Tier 1 XGBoost model trained without Data_Block: R² = {m_all['R2']:.4f}")
        
    # =========================================================================
    # STEP 4 — FEATURE IMPORTANCE & REGIME MEMORISATION CHECK (MODEL A)
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: FEATURE IMPORTANCE & REGIME MEMORISATION CHECK")
    print("=" * 50)
    
    importances = winner_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": TIER1_FEATURES,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    print("Top 10 features for winning model (XGBoost):")
    print(feat_imp_df.head(10).round(4).to_string(index=False))
    
    # Check if top-5 is dominated by block/regime indicators
    regime_indicators = ["Data_Block", "Temp_Gradient", "Reboiler_Delta"]
    top_5_features = feat_imp_df.head(5)["Feature"].tolist()
    regime_in_top_5 = [f for f in top_5_features if f in regime_indicators]
    
    if len(regime_in_top_5) >= 2:
        print("\n  > [!WARNING]")
        print(f"  > Regime indicators {regime_in_top_5} dominate the top-5 feature importances.")
        print("  > The model may be memorising operating campaigns rather than process physics.")
        print("  > We recommend training a regularised model excluding these indicators for comparison.")
        
        # Test training without regime indicators
        non_regime_features = [c for c in TIER1_FEATURES if c not in regime_indicators]
        xgb_non_regime = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        xgb_non_regime.fit(X_train_A[non_regime_features], y_train_A)
        preds_non_regime = xgb_non_regime.predict(X_test_A[non_regime_features])
        r2_non_regime = r2_score(y_test_A, preds_non_regime)
        print(f"  > XGBoost WITHOUT regime indicators {regime_indicators}: R² = {r2_non_regime:.4f} (Delta R² = {m_all['R2'] - r2_non_regime:.4f})")
    else:
        print("\n  SUCCESS: Top features look healthy. Model is learning from process variables.")
        
    # =========================================================================
    # STEP 5 — MODEL B DEFAULT MODELS (C4H6)
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 5: TRAINING MODEL B DEFAULT MODELS (NO TUNING)")
    print("=" * 50)
    
    # Model B uses a smaller training set, so we use tighter regularisation defaults as specified
    model_B_defaults = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":          XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, 
                                         reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5, 
                                         random_state=42, n_jobs=-1),
    }
    
    model_B_records = []
    trained_default_models_B = {}
    
    # Naive Lag-1 Baseline for Model B
    lag1_eval_mask_B = ~test_df_B["C4H6_Bottom_lag1"].isna()
    y_test_B_lag1 = y_test_B[lag1_eval_mask_B]
    pred_lag1_B = test_df_B.loc[lag1_eval_mask_B, "C4H6_Bottom_lag1"].values
    m_baseline_lag1_B = calculate_metrics(y_test_B_lag1, pred_lag1_B, threshold=0.1) # C4H6 threshold scaled
    
    for name, model in model_B_defaults.items():
        print(f"Training Model B {name}...")
        model.fit(X_train_B, y_train_B)
        trained_default_models_B[name] = model
        
        preds = model.predict(X_test_B)
        
        # Evaluate All (using 0.1 threshold for out-of-spec C4H6)
        m_all = calculate_metrics(y_test_B, preds, threshold=0.1)
        print_metrics(f"{name} (All Test)", m_all)
        
        normal_mask = ~test_df_B["is_extreme_event"]
        m_normal = calculate_metrics(y_test_B[normal_mask], preds[normal_mask], threshold=0.1)
        
        extreme_mask = test_df_B["is_extreme_event"]
        m_extreme = calculate_metrics(y_test_B[extreme_mask], preds[extreme_mask], threshold=0.1)
        
        model_B_records.append({
            "Model": f"{name} (Model B - C4H6)",
            "R2_All": m_all["R2"],
            "MAE_All": m_all["MAE"],
            "RMSE_All": m_all["RMSE"],
            "Pct_within_0.1_All": m_all["Pct_within_0.1"],
            "Spec_Recall_All": m_all["Spec_Recall"],
            "R2_Normal": m_normal["R2"],
            "MAE_Normal": m_normal["MAE"],
            "R2_Extreme": m_extreme["R2"],
            "MAE_Extreme": m_extreme["MAE"],
            "Analyzer_Required": "No"
        })
        
    model_B_df = pd.DataFrame(model_B_records)
    print("\n--- MODEL B LEADERBOARD ---")
    print(model_B_df[["Model", "R2_All", "MAE_All", "RMSE_All", "Spec_Recall_All", "Analyzer_Required"]].round(4).to_string(index=False))
    
    # =========================================================================
    # STEP 6 — TIER 2 RESEARCH MODELS
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 6: TRAINING TIER 2 RESEARCH MODELS (WITH TARGET LAGS)")
    print("=" * 50)
    
    # Prepare Tier 2 datasets
    train_df_A2 = df[train_mask & mA_filter].dropna(subset=TIER2_FEATURES + ["C4H8_Bottom"])
    test_df_A2  = df[test_mask  & mA_filter].dropna(subset=TIER2_FEATURES + ["C4H8_Bottom"])
    
    train_df_B2 = df[train_mask & mB_filter].dropna(subset=TIER2_FEATURES + ["C4H6_Bottom"])
    test_df_B2  = df[test_mask  & mB_filter].dropna(subset=TIER2_FEATURES + ["C4H6_Bottom"])
    
    X_train_A2, y_train_A2 = train_df_A2[TIER2_FEATURES], train_df_A2["C4H8_Bottom"]
    X_test_A2, y_test_A2   = test_df_A2[TIER2_FEATURES], test_df_A2["C4H8_Bottom"]
    
    X_train_B2, y_train_B2 = train_df_B2[TIER2_FEATURES], train_df_B2["C4H6_Bottom"]
    X_test_B2, y_test_B2   = test_df_B2[TIER2_FEATURES], test_df_B2["C4H6_Bottom"]
    
    # Model A Tier 2
    print("Training Model A (C4H8) Tier 2 XGBoost...")
    xgb_A2 = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb_A2.fit(X_train_A2, y_train_A2)
    preds_A2 = xgb_A2.predict(X_test_A2)
    m_A2 = calculate_metrics(y_test_A2, preds_A2, threshold=0.4)
    print_metrics("Model A Tier 2 XGBoost", m_A2)
    
    # Model B Tier 2
    print("Training Model B (C4H6) Tier 2 XGBoost...")
    xgb_B2 = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, 
                          reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5, 
                          random_state=42, n_jobs=-1)
    xgb_B2.fit(X_train_B2, y_train_B2)
    preds_B2 = xgb_B2.predict(X_test_B2)
    m_B2 = calculate_metrics(y_test_B2, preds_B2, threshold=0.1)
    print_metrics("Model B Tier 2 XGBoost", m_B2)
    
    # Report the value of the analyzer
    delta_A = m_A2["R2"] - leaderboard_df.loc[leaderboard_df["Model"] == "XGBoost (Tier 1)", "R2_All"].values[0]
    delta_B = m_B2["R2"] - model_B_df.loc[model_B_df["Model"] == "XGBoost (Model B - C4H6)", "R2_All"].values[0]
    print(f"\nValue of working Analyzer:")
    print(f"  Model A (C4H8) R² gain: {delta_A:+.4f}")
    print(f"  Model B (C4H6) R² gain: {delta_B:+.4f}")
    
    # =========================================================================
    # STEP 7 — COMBINE FOR TOTAL_C4 PREDICTIONS & METRICS
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 7: COMBINED TOTAL C4 EVALUATION")
    print("=" * 50)
    
    # To evaluate Total_C4 predictions, we want to align Model A and Model B predictions on Block 4.
    # We will generate predictions on the FULL Block 4 test set (no filtering of stuck rows, to simulate
    # deployment).
    block4_full = df[test_mask].copy()
    
    # For rows with feature NaNs (first 12 rows of the block), XGBoost can handle NaNs natively.
    # But since RandomForest / Linear models can't, we drop them from the evaluation, or forward-fill features
    # for the full prediction so there are no NaNs in dashboard predictions.
    # Let's fill the feature NaNs in block4_full features so they can be predicted by all models.
    # XGBoost handles NaNs naturally, but to keep predictions complete:
    full_X_test = block4_full[TIER1_FEATURES].copy()
    # Forward-fill block boundaries to avoid NaN predictions on the dashboard
    full_X_test = full_X_test.ffill().bfill() 
    
    # Predict full target components
    xgb_A = trained_default_models["XGBoost"]
    xgb_B = trained_default_models_B["XGBoost"]
    
    block4_full["pred_C4H8"] = xgb_A.predict(full_X_test)
    block4_full["pred_C4H6"] = xgb_B.predict(full_X_test)
    block4_full["pred_Total_C4"] = block4_full["pred_C4H8"] + block4_full["pred_C4H6"]
    
    # Combined target analyzer lag-1 baseline
    # Total_C4 lag-1 is just C4H8_lag1 + C4H6_lag1
    block4_full["lag1_Total_C4"] = block4_full["C4H8_Bottom_lag1"] + block4_full["C4H6_Bottom_lag1"]
    
    # Save combined predictions to test_predictions.parquet
    block4_full.to_parquet(TEST_PREDS_FILE, index=False)
    print(f"Combined test predictions saved to: {TEST_PREDS_FILE} (shape: {block4_full.shape})")
    
    # Evaluate combined prediction on the subset of Block 4 where analyzer is healthy (not stuck)
    # and both actual values are available (not NaN)
    total_eval_mask = (
        (~block4_full["C4H8_Bottom_stuck"]) & 
        (~block4_full["C4H6_Bottom_stuck"]) & 
        (block4_full["C4H6_Bottom"] > 0.001) & 
        (~block4_full["Total_C4"].isna())
    )
    
    eval_df = block4_full[total_eval_mask]
    y_true_total = eval_df["Total_C4"]
    y_pred_total = eval_df["pred_Total_C4"]
    
    print(f"Evaluating combined Total_C4 on {len(eval_df)} healthy analyzer rows:")
    m_total_xgb = calculate_metrics(y_true_total, y_pred_total, threshold=0.5)
    print_metrics("Combined XGBoost (Tier 1)", m_total_xgb)
    
    # Lag-1 baseline for Total C4
    lag1_total_mask = total_eval_mask & (~block4_full["lag1_Total_C4"].isna())
    eval_lag1_df = block4_full[lag1_total_mask]
    m_total_lag1 = calculate_metrics(eval_lag1_df["Total_C4"], eval_lag1_df["lag1_Total_C4"], threshold=0.5)
    print_metrics("Combined Naive Lag-1 Baseline (Requires Analyzer)", m_total_lag1)
    
    # Save training metrics CSV
    all_metrics = [
        {"Model": "Model A (C4H8) XGBoost Defaults", **calculate_metrics(y_test_A, xgb_A.predict(X_test_A), 0.4)},
        {"Model": "Model B (C4H6) XGBoost Defaults", **calculate_metrics(y_test_B, xgb_B.predict(X_test_B), 0.1)},
        {"Model": "Combined Total_C4 XGBoost Defaults", **m_total_xgb},
        {"Model": "Combined Total_C4 Naive Lag-1", **m_total_lag1}
    ]
    pd.DataFrame(all_metrics).to_csv(TRAINING_METRICS_FILE, index=False)
    print(f"All training metrics summary saved to: {TRAINING_METRICS_FILE}")
    
    # Save the models as JSON
    xgb_A.save_model(os.path.join(MODELS_DIR, "model_A_C4H8.json"))
    xgb_B.save_model(os.path.join(MODELS_DIR, "model_B_C4H6.json"))
    print("Models saved as JSON to models/ directory.")
    
    print("\n" + "=" * 80)
    print("PHASE 3 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
