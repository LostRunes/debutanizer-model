"""
notebooks/run_anchor_tuning.py
==============================
Trains and tunes XGBoost, LightGBM, and CatBoost on the 8-feature robust physical subset:
- Reflux_Ratio
- Steam_Feed_Ratio
- Reboiling_Steam_Flow_dev24h
- Reflux_Flow_dev24h
- Column_Bottom_Temp_dev24h
- Control_Tray_Temp_dev24h
- Column_Top_Pressure_dev24h
- C4H8_campaign_anchor (12h limit to ensure high coverage and high correlation)
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
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import shap

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
DIAGNOSTICS_DIR = os.path.join("experiments", "diagnostics")
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

# Monkeypatch SHAP base_score bug
import shap.explainers._tree
original_float = float
def custom_float_parser(val):
    if isinstance(val, str):
        val = val.strip('[] \t\n\r')
    elif isinstance(val, list):
        if len(val) > 0:
            val = val[0]
    try:
        return original_float(val)
    except ValueError:
        return original_float(str(val).strip('[] \t\n\r'))
shap.explainers._tree.float = custom_float_parser


def check_leakage(X, y, dataset_name="Train"):
    print(f"[{dataset_name}] Running leakage check...")
    leaking_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            continue
        corr = abs(pd.Series(X[col]).corr(pd.Series(y)))
        if corr > 0.98:
            print(f"  [WARNING] Possible leakage -> {col} (Corr = {corr:.4f})")
            leaking_cols.append(col)
    if not leaking_cols:
        print("  [OK] No high-correlation leakage detected.")
    return leaking_cols


def main():
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded dataset: {df.shape}")
    
    # 1. Define Campaign Anchor (12h limit, leak-free shift(1))
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    df["C4H8_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H8_last_valid"]
          .transform(lambda x: x.shift(1).rolling(72, min_periods=1).mean())
    )
    # Let's use 12h limit for high coverage and high quality in production
    df["C4H8_campaign_anchor_12h"] = (
        df.groupby("Data_Block")["C4H8_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=12))
    )
    
    # Define the 8-feature robust subset
    feats = [
        "Reflux_Ratio", "Steam_Feed_Ratio",
        "Reboiling_Steam_Flow_dev24h", "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h", "Control_Tray_Temp_dev24h", "Column_Top_Pressure_dev24h",
        "C4H8_campaign_anchor_12h"
    ]
    
    # Setup splits
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    # Drop rows with NaNs in features or target (handles lag and anchor startup rows)
    train_df = df[train_mask & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=feats + ["C4H8_Bottom"])
    
    X_train, y_train = train_df[feats], train_df["C4H8_Bottom"].values
    X_test, y_test   = test_df[feats], test_df["C4H8_Bottom"].values
    
    print(f"Robust 8-Feature Train size: {X_train.shape} | Test size: {X_test.shape}")
    
    # Run Leakage Check
    check_leakage(X_train, y_train, "Train")
    check_leakage(X_test, y_test, "Test")
    
    # -------------------------------------------------------------------------
    # 1. XGBoost Baseline
    # -------------------------------------------------------------------------
    print("\nTraining XGBoost on 8 features...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    xgb_r2 = r2_score(y_test, pred_xgb)
    xgb_mae = mean_absolute_error(y_test, pred_xgb)
    print(f"  XGBoost test R² = {xgb_r2:.4f} | MAE = {xgb_mae:.4f} wt%")
    
    # -------------------------------------------------------------------------
    # 2. CatBoost Tuning (Optuna)
    # -------------------------------------------------------------------------
    print("\nTuning CatBoost on 8 features...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    def objective_cb(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 100, 500),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth":               trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 10.0),
            "loss_function":       "RMSE",
            "eval_metric":         "RMSE",
            "random_seed":         42,
            "verbose":             0,
            "thread_count":        -1
        }
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=30,
                verbose=0
            )
            preds_val = model.predict(X_val)
            cv_scores.append(r2_score(y_val, preds_val))
            
        return np.mean(cv_scores)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_cb, n_trials=30, show_progress_bar=True)
    
    print(f"  Best CV R²: {study.best_value:.4f}")
    print("  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
        
    # Fit final CatBoost
    best_cb_params = study.best_params.copy()
    best_cb_params.update({"random_seed": 42, "verbose": 0, "thread_count": -1})
    tuned_cb = CatBoostRegressor(**best_cb_params)
    tuned_cb.fit(X_train, y_train, verbose=0)
    pred_cb = tuned_cb.predict(X_test)
    cb_r2 = r2_score(y_test, pred_cb)
    cb_mae = mean_absolute_error(y_test, pred_cb)
    print(f"  Tuned CatBoost test R² = {cb_r2:.4f} | MAE = {cb_mae:.4f} wt%")
    
    # -------------------------------------------------------------------------
    # 3. LightGBM Tuning (Optuna)
    # -------------------------------------------------------------------------
    print("\nTuning LightGBM on 8 features...")
    def objective_lgb(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 6),
            "num_leaves":        trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state":      42,
            "verbose":           -1,
            "n_jobs":            -1
        }
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds_val = model.predict(X_val)
            cv_scores.append(r2_score(y_val, preds_val))
            
        return np.mean(cv_scores)
        
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(objective_lgb, n_trials=30, show_progress_bar=True)
    
    print(f"  Best LGB CV R²: {study_lgb.best_value:.4f}")
    
    # Fit final LightGBM
    best_lgb_params = study_lgb.best_params.copy()
    best_lgb_params.update({"random_state": 42, "verbose": -1, "n_jobs": -1})
    tuned_lgb = lgb.LGBMRegressor(**best_lgb_params)
    tuned_lgb.fit(X_train, y_train)
    pred_lgb = tuned_lgb.predict(X_test)
    lgb_r2 = r2_score(y_test, pred_lgb)
    lgb_mae = mean_absolute_error(y_test, pred_lgb)
    print(f"  Tuned LightGBM test R² = {lgb_r2:.4f} | MAE = {lgb_mae:.4f} wt%")
    
    # -------------------------------------------------------------------------
    # 4. Ensemble
    # -------------------------------------------------------------------------
    print("\nEvaluating Ensemble...")
    pred_ens = 0.5 * pred_cb + 0.3 * pred_xgb + 0.2 * pred_lgb
    ens_r2 = r2_score(y_test, pred_ens)
    ens_mae = mean_absolute_error(y_test, pred_ens)
    print(f"  Ensemble test R² = {ens_r2:.4f} | MAE = {ens_mae:.4f} wt%")
    
    # -------------------------------------------------------------------------
    # 5. Diagnostics for Best Model (Tuned CatBoost)
    # -------------------------------------------------------------------------
    print("\nGenerating best model diagnostics (Tuned CatBoost)...")
    residuals = y_test - pred_cb
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred_cb, alpha=0.5, color="darkgreen", edgecolors="w", s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual C4H8_Bottom (wt%)")
    plt.ylabel("Predicted C4H8_Bottom (wt%)")
    plt.title("Robust 8-Feat: Actual vs. Predicted (Tuned CatBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_plot_1_actual_vs_predicted.png"))
    plt.close()
    
    # Plot 4: Residual vs Time
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["DateTime"], residuals, alpha=0.6, color="firebrick", lw=1.5)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("DateTime")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Robust 8-Feat: Residual vs. Time (Block 4)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_plot_4_residual_vs_time.png"))
    plt.close()
    
    # SHAP analysis
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(tuned_cb)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
    plt.title("Robust 8-Feat SHAP Summary Plot", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "robust_plot_5_shap.png"))
    plt.close()
    
    # Save the Tuned CatBoost model to models/model_A_CatBoost_robust.bin
    tuned_cb.save_model("models/model_A_CatBoost_robust.bin")
    print("Saved tuned CatBoost model to models/model_A_CatBoost_robust.bin")
    
    # Print results summary
    print("\n" + "=" * 50)
    print("ROBUST 8-FEATURE PHYSICAL MODEL COMPARISON")
    print("=" * 50)
    print(f"XGBoost R²:       {xgb_r2:.4f}  | MAE: {xgb_mae:.4f} wt%")
    print(f"CatBoost R²:      {cb_r2:.4f}  | MAE: {cb_mae:.4f} wt%")
    print(f"LightGBM R²:      {lgb_r2:.4f}  | MAE: {lgb_mae:.4f} wt%")
    print(f"Ensemble R²:      {ens_r2:.4f}  | MAE: {ens_mae:.4f} wt%")
    print("=" * 50)


if __name__ == "__main__":
    main()
