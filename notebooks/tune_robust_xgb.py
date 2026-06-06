"""
notebooks/tune_robust_xgb.py
============================
Performs Optuna hyperparameter tuning for XGBoost on the robust 8-feature physical set:
- C4H8_campaign_anchor (12h limit or 72h limit, let's use 72h limit as Subset 7 baseline)
- Steam_Feed_Ratio
- Reflux_Ratio
- Reboiling_Steam_Flow_dev24h
- Reflux_Flow_dev24h
- Column_Bottom_Temp_dev24h
- Control_Tray_Temp_dev24h
- Column_Top_Pressure_dev24h

Tunes: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda
Uses 5-fold TimeSeriesSplit CV on train blocks (Blocks 1-3).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import optuna

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES_FILE = "data/features.parquet"
RESULTS_FILE = "experiments/robust_xgb_optuna_results.json"
DIAGNOSTICS_DIR = "experiments/diagnostics"
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

def main():
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded dataset: {df.shape}")
    
    # 1. Define Campaign Anchor (72h limit, shift(1))
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    df["C4H8_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H8_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=72))
    )
    
    # Features list
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
    
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    train_df = df[train_mask & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    
    X_train = train_df[feats]
    y_train = train_df["C4H8_Bottom"].values
    X_test  = test_df[feats]
    y_test  = test_df["C4H8_Bottom"].values
    
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    
    # 2. Define Optuna objective
    tscv = TimeSeriesSplit(n_splits=5)
    
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs":            -1,
            "random_state":      42
        }
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            cv_scores.append(r2_score(y_val, preds))
            
        return np.mean(cv_scores)
        
    print("\nRunning 50 Optuna trials for XGBoost...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print("\n" + "=" * 50)
    print("XGBOOST OPTUNA OPTIMIZATION COMPLETED")
    print("=" * 50)
    print(f"Best CV R² score: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # 3. Train final model with best params
    best_params = study.best_params.copy()
    best_params.update({"n_jobs": -1, "random_state": 42})
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    preds_test = best_model.predict(X_test)
    test_r2 = r2_score(y_test, preds_test)
    test_mae = mean_absolute_error(y_test, preds_test)
    
    print(f"\nFinal Optimized XGBoost Model (Block 4 Test Set):")
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
    best_model.save_model("models/model_A_XGBoost_robust_opt.json")
    print("Optimized XGBoost model saved to models/model_A_XGBoost_robust_opt.json")
    
    # Generate Plots for Tuned XGBoost
    residuals = y_test - preds_test
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds_test, alpha=0.5, color="darkorange", edgecolors="w", s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual C4H8_Bottom (wt%)")
    plt.ylabel("Predicted C4H8_Bottom (wt%)")
    plt.title("Optimized XGBoost: Actual vs. Predicted (Block 4)")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_opt_plot_1_actual_vs_predicted.png"))
    plt.close()
    
    # Plot 2: Residual vs Time
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["DateTime"], residuals, alpha=0.6, color="orange", lw=1.5)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("DateTime")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Optimized XGBoost: Residual vs. Time (Block 4)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_opt_plot_4_residual_vs_time.png"))
    plt.close()
    print("Saved optimized diagnostic plots to experiments/diagnostics/")

if __name__ == "__main__":
    main()
