"""
experiment_optuna.py
====================
Performs hyperparameter optimization for LightGBM using Optuna.
Uses 5-fold TimeSeriesSplit cross-validation on Blocks 1-3.
Features exclude Data_Block (process-only, 87 features).
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb
import optuna

# Suppress Optuna logs to keep terminal output clean (except for important info)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
RESULTS_FILE = os.path.join("experiments", "optuna_results.json")
os.makedirs("experiments", exist_ok=True)

def main():
    print("=" * 80)
    print("LIGHTGBM HYPERPARAMETER OPTIMIZATION (OPTUNA)")
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
    
    # 2. Define Optuna objective
    tscv = TimeSeriesSplit(n_splits=5)
    
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 8, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs":            -1,
            "random_state":      42,
            "verbose":           -1
        }
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_val)
            r2 = r2_score(y_val, preds)
            cv_scores.append(r2)
            
        return np.mean(cv_scores)
        
    # 3. Run search
    print("\nRunning 50 Optuna trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print("\n" + "=" * 50)
    print("OPTUNA OPTIMIZATION COMPLETED")
    print("=" * 50)
    print(f"Best CV R² score: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # 4. Train final model with best params
    best_params = study.best_params.copy()
    best_params.update({"n_jobs": -1, "random_state": 42, "verbose": -1})
    
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    preds_test = best_model.predict(X_test)
    test_r2 = r2_score(y_test, preds_test)
    test_mae = mean_absolute_error(y_test, preds_test)
    
    print(f"\nFinal Optimized LightGBM Model (Block 4 Test Set):")
    print(f"  Test R²:  {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f} wt%")
    print("=" * 50)
    
    # Save results
    results = {
        "best_cv_r2": float(study.best_value),
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "best_params": study.best_params
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Optuna results saved to {RESULTS_FILE}")
    
    # Save model
    best_model.booster_.save_model("models/model_A_LGBM_opt.txt")
    print("Optimized LightGBM model saved to models/model_A_LGBM_opt.txt")

if __name__ == "__main__":
    main()
