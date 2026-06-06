"""
notebooks/run_physics_and_tuning.py
====================================
1. Trains the 'Pure Physics Model' (removing all calendar and time features).
2. Generates diagnostic plots for the Pure Physics Model (Actual vs Pred, Res vs Press, Res vs Time, SHAP).
3. Investigates Pressure_x_TopTemp (Scatter of C4H8 vs Pressure_x_TopTemp, SHAP dependence plot).
4. Performs CatBoost tuning using Optuna (50 trials, TimeSeriesSplit CV).
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
import shap
import optuna

# Suppress Optuna logs
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

def main():
    df = pd.read_parquet(FEATURES_FILE)
    
    # 1. DEFINE FEATURES FOR PURE PHYSICS MODEL
    TARGET_LAG_COLS = (
        [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
        [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
    )
    TIME_PROXIES = ["month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"]
    META_COLS = [
        "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
        "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
        "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
        "Analyzer_Health", "is_extreme_event", "Data_Block",
    ] + TARGET_LAG_COLS + TIME_PROXIES
    
    physics_feats = [c for c in df.columns if c not in META_COLS]
    print("=" * 80)
    print(f"RUNNING PURE PHYSICS MODEL EXPERIMENTS ({len(physics_feats)} features)")
    print("=" * 80)
    
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    # Reconstruct train/test datasets
    train_df = df[train_mask & mA_filter].dropna(subset=physics_feats + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=physics_feats + ["C4H8_Bottom"])
    
    X_train, y_train = train_df[physics_feats], train_df["C4H8_Bottom"].values
    X_test, y_test   = test_df[physics_feats], test_df["C4H8_Bottom"].values
    
    # --- EXPERIMENT 1 & 3: PURE PHYSICS MODEL & DIAGNOSTICS ---
    print("\nTraining Pure Physics XGBoost Model...")
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    test_r2 = r2_score(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)
    print(f"Pure Physics XGBoost | R² = {test_r2:.4f} | MAE = {test_mae:.4f} wt%")
    
    residuals = y_test - preds
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.5, color="darkgreen", edgecolors="w", s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual C4H8_Bottom (wt%)")
    plt.ylabel("Predicted C4H8_Bottom (wt%)")
    plt.title("Pure Physics: Actual vs. Predicted C4H8_Bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_1_actual_vs_predicted.png"))
    plt.close()
    
    # Plot 2: Residual Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="teal", bins=50)
    plt.axvline(x=0, color="red", linestyle="--", lw=2)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Pure Physics: Residual Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_2_residual_histogram.png"))
    plt.close()
    
    # Plot 3: Residual vs Top Pressure
    plt.figure(figsize=(8, 6))
    plt.scatter(test_df["Column_Top_Pressure"], residuals, alpha=0.5, color="darkorange", edgecolors="w", s=40)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("Column_Top_Pressure (kg/cm²g)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Pure Physics: Residual vs. Column_Top_Pressure")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_3_residual_vs_pressure.png"))
    plt.close()
    
    # Plot 4: Residual vs Time
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["DateTime"], residuals, alpha=0.6, color="firebrick", lw=1.5)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("DateTime")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Pure Physics: Residual vs. Time")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_4_residual_vs_time.png"))
    plt.close()
    
    # Plot 5: SHAP summary
    print("Generating Pure Physics SHAP values...")
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.title("Pure Physics: SHAP Summary Plot (Top 20 Features)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pure_physics_plot_5_shap.png"))
    plt.close()
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 50)
    print("TOP 20 SHAP FEATURES (PURE PHYSICS MODEL)")
    print("=" * 50)
    print(shap_imp_df.head(20).to_string(index=False))
    print("=" * 50)
    shap_imp_df.to_csv(os.path.join(DIAGNOSTICS_DIR, "shap_importances_pure_physics.csv"), index=False)
    
    # --- INVESTIGATE Pressure_x_TopTemp ---
    print("\nInvestigating Pressure_x_TopTemp...")
    # Scatter of C4H8 vs Pressure_x_TopTemp by Data_Block
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df[mA_filter & (~df["C4H8_Bottom"].isna())],
        x="Pressure_x_TopTemp", y="C4H8_Bottom",
        hue="Data_Block", palette="viridis", alpha=0.6, edgecolor=None
    )
    plt.xlabel("Pressure_x_TopTemp (Pressure * Column_Top_Temp)")
    plt.ylabel("C4H8_Bottom (wt%)")
    plt.title("Pressure_x_TopTemp vs. C4H8_Bottom by Data_Block")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pressure_xtoptemp_scatter.png"))
    plt.close()
    
    # SHAP dependence plot for Pressure_x_TopTemp
    plt.figure(figsize=(8, 6))
    feat_idx = physics_feats.index("Pressure_x_TopTemp")
    shap.dependence_plot(
        "Pressure_x_TopTemp", shap_values, X_sample,
        interaction_index=None, show=False
    )
    plt.title("SHAP Dependence Plot for Pressure_x_TopTemp")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "pressure_xtoptemp_dependence.png"))
    plt.close()
    
    # --- EXPERIMENT 2: CATBOOST TUNING (OPTUNA) ---
    print("\nTuning CatBoostRegressor via Optuna CV (50 trials)...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    def objective_cb(trial):
        params = {
            "iterations":    trial.suggest_int("iterations", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth":         trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_seed":   42,
            "thread_count":  -1,
            "verbose":       0
        }
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            model_cb = CatBoostRegressor(**params)
            model_cb.fit(X_tr, y_tr)
            
            preds_val = model_cb.predict(X_val)
            r2 = r2_score(y_val, preds_val)
            cv_scores.append(r2)
            
        return np.mean(cv_scores)
        
    study_cb = optuna.create_study(direction="maximize")
    study_cb.optimize(objective_cb, n_trials=50, show_progress_bar=True)
    
    print("\n" + "=" * 50)
    print("CATBOOST OPTUNA COMPLETED")
    print("=" * 50)
    print(f"Best CV R² score: {study_cb.best_value:.4f}")
    print("Best CatBoost parameters:")
    for k, v in study_cb.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # Train final tuned CatBoost model
    best_cb_params = study_cb.best_params.copy()
    best_cb_params.update({"random_seed": 42, "verbose": 0})
    tuned_cb = CatBoostRegressor(**best_cb_params)
    tuned_cb.fit(X_train, y_train)
    
    tuned_cb_preds = tuned_cb.predict(X_test)
    tuned_cb_r2 = r2_score(y_test, tuned_cb_preds)
    tuned_cb_mae = mean_absolute_error(y_test, tuned_cb_preds)
    
    print(f"\nTuned CatBoost Model on Block 4 Test Set:")
    print(f"  Test R²:  {tuned_cb_r2:.4f}")
    print(f"  Test MAE: {tuned_cb_mae:.4f} wt%")
    print("=" * 50)
    
    # Save optimized CatBoost model
    tuned_cb.save_model("models/model_A_CatBoost_opt.bin")
    print("Optimized CatBoost model saved to models/model_A_CatBoost_opt.bin")
    
    # Save comparison data
    comparison_opt = [
        {"Model": "XGBoost (Pure Physics)", "R2": float(test_r2), "MAE": float(test_mae)},
        {"Model": "CatBoost (Default, No Month)", "R2": -0.9489, "MAE": 0.2803},
        {"Model": "CatBoost (Tuned, Pure Physics)", "R2": float(tuned_cb_r2), "MAE": float(tuned_cb_mae)}
    ]
    comparison_opt_df = pd.DataFrame(comparison_opt)
    comparison_opt_df.to_csv("experiments/comparison_opt_leaderboard.csv", index=False)
    print("Leaderboard comparison saved to experiments/comparison_opt_leaderboard.csv")
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
