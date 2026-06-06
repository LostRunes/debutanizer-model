"""
experiment_catboost.py
======================
Trains and evaluates CatBoostRegressor on the same process-only dataset.
Compares performance to XGBoost and optimized LightGBM.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor

# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
RESULTS_FILE = os.path.join("experiments", "catboost_results.json")

def main():
    print("=" * 80)
    print("CATBOOST REGRESSOR EVALUATION")
    print("=" * 80)
    
    # 1. Load data
    df = pd.read_parquet(FEATURES_FILE)
    
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
    
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Features count: {len(feature_cols)}")
    
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    # Reconstruct train/test datasets
    train_df = df[train_mask & mA_filter].dropna(subset=feature_cols + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=feature_cols + ["C4H8_Bottom"])
    
    X_train = train_df[feature_cols]
    y_train = train_df["C4H8_Bottom"].values
    X_test  = test_df[feature_cols]
    y_test  = test_df["C4H8_Bottom"].values
    
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    
    # 2. Train CatBoost
    print("\nTraining CatBoostRegressor...")
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100
    )
    model.fit(X_train, y_train)
    
    # 3. Evaluate
    preds = model.predict(X_test)
    test_r2 = r2_score(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)
    
    print("\n" + "=" * 50)
    print("CATBOOST PERFORMANCE (Block 4 Test Set)")
    print("=" * 50)
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f} wt%")
    print("=" * 50)
    
    # 4. Compare Leaderboard
    print("\n" + "=" * 80)
    print("LEADERBOARD COMPARISON (Process-Only, No Data_Block)")
    print("=" * 80)
    
    # XGBoost metrics are from model_training.py output
    # LightGBM metrics are from experiment_optuna.py output
    comparison = [
        {"Model": "XGBoost (Default)", "R2": -1.0674, "MAE": 0.2956},
        {"Model": "LightGBM (Optimized)", "R2": -0.8876, "MAE": 0.2783},
        {"Model": "CatBoost (Default)", "R2": float(test_r2), "MAE": float(test_mae)},
    ]
    comparison_df = pd.DataFrame(comparison).sort_values(by="R2", ascending=False)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    # Save results
    results = {
        "test_r2": float(test_r2),
        "test_mae": float(test_mae)
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
        
    # Save comparison to experiments/comparison_leaderboard.csv
    comparison_df.to_csv("experiments/comparison_leaderboard.csv", index=False)
    print("Leaderboard comparison saved to experiments/comparison_leaderboard.csv")
    
    # Save CatBoost model
    model.save_model("models/model_A_CatBoost.bin")
    print("CatBoost model saved to models/model_A_CatBoost.bin")

if __name__ == "__main__":
    main()
