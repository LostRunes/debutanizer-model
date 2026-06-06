"""
notebooks/run_advanced_experiments.py
======================================
Implements advanced modeling and analysis steps:
1. Run 1: CatBoost + larger Optuna search space + 5-fold TimeSeriesSplit CV + early stopping.
2. Run 2: Ensemble (Tuned CatBoost + XGBoost + Optimized LightGBM).
3. Run 3: No Temperature model (temperature features removed to test concept drift).
4. Run 4: Research Campaign Anchor model (using a non-leaking rolling target average).
5. Appends all results to experiments/master_leaderboard.csv.
6. Performs leakage check (correlation > 0.98 check) before every training.
7. Generates diagnostics (SHAP, Residual vs Time) for the best Tuned CatBoost model.
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

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
DIAGNOSTICS_DIR = os.path.join("experiments", "diagnostics")
MASTER_LEADERBOARD = os.path.join("experiments", "master_leaderboard.csv")
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
    """
    Step 8: Critical Leakage Check.
    Checks if any feature has a correlation > 0.98 with the target.
    """
    print(f"[{dataset_name}] Running leakage check...")
    leaking_cols = []
    for col in X.columns:
        # Ignore constant features
        if X[col].nunique() <= 1:
            continue
        corr = abs(pd.Series(X[col]).corr(pd.Series(y)))
        if corr > 0.98:
            print(f"  [WARNING] Possible leakage -> {col} (Corr = {corr:.4f})")
            leaking_cols.append(col)
    if not leaking_cols:
        print("  [OK] No high-correlation leakage detected.")
    return leaking_cols


def save_to_master_leaderboard(model_name, feature_set, r2, mae, top_feature):
    """
    Step 7: Save All Experiments to master_leaderboard.csv.
    """
    experiment_results = {
        "Model": model_name,
        "Feature_Set": feature_set,
        "R2": float(r2),
        "MAE": float(mae),
        "Top_Feature": top_feature
    }
    
    results_df = pd.DataFrame([experiment_results])
    
    results_df.to_csv(
        MASTER_LEADERBOARD,
        mode="a",
        header=not os.path.exists(MASTER_LEADERBOARD),
        index=False
    )
    print(f"Saved experiment '{model_name} ({feature_set})' to {MASTER_LEADERBOARD}")


def main():
    # Load feature-engineered dataset
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded dataset: {df.shape}")
    
    # 1. DEFINE BASE METADATA AND TARGET LAGS
    TARGET_LAG_COLS = (
        [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
        [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
    )
    TIME_PROXIES = ["month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos"]
    
    # Base meta columns (excluded from Pure Physics)
    META_COLS = [
        "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
        "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
        "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
        "Analyzer_Health", "is_extreme_event", "Data_Block",
    ] + TARGET_LAG_COLS + TIME_PROXIES
    
    physics_feats = [c for c in df.columns if c not in META_COLS]
    print(f"Found {len(physics_feats)} pure process physics features.")
    
    # Setup masks
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    # Reconstruct train/test datasets for Pure Physics
    train_df = df[train_mask & mA_filter].dropna(subset=physics_feats + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=physics_feats + ["C4H8_Bottom"])
    
    X_train, y_train = train_df[physics_feats], train_df["C4H8_Bottom"].values
    X_test, y_test   = test_df[physics_feats], test_df["C4H8_Bottom"].values
    
    print(f"Pure Physics train set: {X_train.shape} | test set: {X_test.shape}")
    
    # =========================================================================
    # RUN 1: CATBOOST STRONGER TUNING
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUN 1: STRONGER CATBOOST TUNING (5-FOLD TS-CV + EARLY STOPPING)")
    print("=" * 80)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    def objective_cb(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 200, 600),
            "learning_rate":       trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "depth":               trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 0.01, 50.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "random_strength":     trial.suggest_float("random_strength", 0.0, 20.0),
            "rsm":                 trial.suggest_float("rsm", 0.1, 0.7),
            "loss_function":       "RMSE",
            "eval_metric":         "RMSE",
            "random_seed":         42,
            "verbose":             0,
            "thread_count":        -1
        }
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            model_cb = CatBoostRegressor(**params)
            model_cb.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                use_best_model=True,
                early_stopping_rounds=50,
                verbose=0
            )
            
            preds_val = model_cb.predict(X_val)
            r2 = r2_score(y_val, preds_val)
            cv_scores.append(r2)
            
        return np.mean(cv_scores)
        
    print("Running 15 Optuna trials for CatBoost...")
    study_cb = optuna.create_study(direction="maximize")
    study_cb.optimize(objective_cb, n_trials=15, show_progress_bar=True)
    
    print("\n" + "=" * 50)
    print("CATBOOST STRONGER TUNING COMPLETED")
    print("=" * 50)
    print(f"Best 5-Fold CV R² score: {study_cb.best_value:.4f}")
    print("Best CatBoost parameters:")
    for k, v in study_cb.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # Train final tuned CatBoost model
    # Perform leakage check before final fit
    check_leakage(X_train, y_train, "Final Pure Physics Train")
    
    best_cb_params = study_cb.best_params.copy()
    best_cb_params.update({
        "loss_function": "RMSE",
        "eval_metric":   "RMSE",
        "random_seed":   42,
        "verbose":       0,
        "thread_count":  -1
    })
    
    tuned_cb = CatBoostRegressor(**best_cb_params)
    tuned_cb.fit(X_train, y_train, verbose=0)
    
    pred_cb = tuned_cb.predict(X_test)
    cb_r2 = r2_score(y_test, pred_cb)
    cb_mae = mean_absolute_error(y_test, pred_cb)
    
    print(f"Tuned CatBoost on Block 4 Test Set: R² = {cb_r2:.4f} | MAE = {cb_mae:.4f} wt%")
    
    # Extract top feature from CatBoost
    cb_importances = tuned_cb.get_feature_importance()
    top_cb_idx = np.argmax(cb_importances)
    top_cb_feature = physics_feats[top_cb_idx]
    top_cb_imp = cb_importances[top_cb_idx]
    print(f"Top CatBoost feature: {top_cb_feature} ({top_cb_imp:.4f})")
    
    save_to_master_leaderboard(
        "CatBoost (Tuned)", "Pure Physics", cb_r2, cb_mae, f"{top_cb_feature} ({top_cb_imp:.2f})"
    )
    
    # Save CatBoost model
    tuned_cb.save_model("models/model_A_CatBoost_opt2.bin")
    
    # =========================================================================
    # RUN 2: ENSEMBLE MODEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUN 2: TRAINING ENSEMBLE (CATBOOST + XGBOOST + LIGHTGBM)")
    print("=" * 80)
    
    # 2.1 XGBoost Baseline
    print("Fitting XGBoost Baseline...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    xgb_r2 = r2_score(y_test, pred_xgb)
    xgb_mae = mean_absolute_error(y_test, pred_xgb)
    
    xgb_importances = xgb.feature_importances_
    top_xgb_idx = np.argmax(xgb_importances)
    top_xgb_feature = physics_feats[top_xgb_idx]
    
    save_to_master_leaderboard(
        "XGBoost (Default)", "Pure Physics", xgb_r2, xgb_mae, f"{top_xgb_feature} ({xgb_importances[top_xgb_idx]:.3f})"
    )
    
    # 2.2 LightGBM (Tuned)
    # Load optimized LightGBM parameters if available
    lgb_params = {
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    }
    if os.path.exists("experiments/optuna_results.json"):
        try:
            with open("experiments/optuna_results.json", "r") as f:
                opt_lgb = json.load(f)
            lgb_params.update(opt_lgb["best_params"])
            print("Loaded optimized LightGBM parameters from experiments/optuna_results.json")
        except Exception as e:
            print(f"Error loading optuna_results.json ({e}), using default LightGBM...")
    else:
        print("experiments/optuna_results.json not found, using default LightGBM...")
        
    print("Fitting LightGBM (Tuned)...")
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)
    pred_lgb = lgb_model.predict(X_test)
    lgb_r2 = r2_score(y_test, pred_lgb)
    lgb_mae = mean_absolute_error(y_test, pred_lgb)
    
    lgb_importances = lgb_model.feature_importances_
    top_lgb_idx = np.argmax(lgb_importances)
    top_lgb_feature = physics_feats[top_lgb_idx]
    
    save_to_master_leaderboard(
        "LightGBM (Tuned)", "Pure Physics", lgb_r2, lgb_mae, f"{top_lgb_feature} ({lgb_importances[top_lgb_idx]:.2f})"
    )
    
    # 2.3 Ensemble Prediction
    print("Combining predictions...")
    ensemble_pred = 0.5 * pred_cb + 0.3 * pred_xgb + 0.2 * pred_lgb
    ens_r2 = r2_score(y_test, ensemble_pred)
    ens_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print("\n" + "=" * 50)
    print("ENSEMBLE RESULTS (Block 4 Test Set)")
    print("=" * 50)
    print(f"  Tuned CatBoost R²:  {cb_r2:.4f}  | MAE: {cb_mae:.4f}")
    print(f"  XGBoost R²:         {xgb_r2:.4f}  | MAE: {xgb_mae:.4f}")
    print(f"  Tuned LightGBM R²:  {lgb_r2:.4f}  | MAE: {lgb_mae:.4f}")
    print(f"  --------------------------------------------------")
    print(f"  Ensemble R²:        {ens_r2:.4f}  | MAE: {ens_mae:.4f}")
    print("=" * 50)
    
    save_to_master_leaderboard(
        "Ensemble (0.5CB+0.3XG+0.2LG)", "Pure Physics", ens_r2, ens_mae, "N/A"
    )
    
    # =========================================================================
    # RUN 3: TEST "NO TEMPERATURE" HYPOTHESIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUN 3: TEST 'NO TEMPERATURE' HYPOTHESIS")
    print("=" * 80)
    
    # Filter out temperature-related features
    TEMP_KEYS = ["Temp", "Delta", "Gradient"]
    TEMP_COLS = [c for c in physics_feats if any(tk in c for tk in TEMP_KEYS)]
    no_temp_feats = [c for c in physics_feats if c not in TEMP_COLS]
    
    print(f"Removed {len(TEMP_COLS)} temperature columns. Remaining features: {len(no_temp_feats)}")
    print("Removed columns list:")
    print(sorted(TEMP_COLS))
    
    X_train_nt = train_df[no_temp_feats]
    X_test_nt  = test_df[no_temp_feats]
    
    # Leakage check on no-temperature set
    check_leakage(X_train_nt, y_train, "No Temperature Train")
    
    # Fit CatBoost with same tuned parameters
    print("Fitting Tuned CatBoost (No Temp)...")
    cb_nt = CatBoostRegressor(**best_cb_params)
    cb_nt.fit(X_train_nt, y_train, verbose=0)
    pred_nt = cb_nt.predict(X_test_nt)
    
    nt_r2 = r2_score(y_test, pred_nt)
    nt_mae = mean_absolute_error(y_test, pred_nt)
    print(f"No Temperature Tuned CatBoost: R² = {nt_r2:.4f} | MAE = {nt_mae:.4f} wt%")
    
    nt_importances = cb_nt.get_feature_importance()
    top_nt_idx = np.argmax(nt_importances)
    top_nt_feature = no_temp_feats[top_nt_idx]
    top_nt_imp = nt_importances[top_nt_idx]
    
    save_to_master_leaderboard(
        "CatBoost (Tuned)", "No Temperature", nt_r2, nt_mae, f"{top_nt_feature} ({top_nt_imp:.2f})"
    )
    
    # =========================================================================
    # RUN 4: RESEARCH CAMPAIGN ANCHOR EXPERIMENT
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUN 4: RESEARCH CAMPAIGN ANCHOR EXPERIMENT")
    print("=" * 80)
    
    # Create the shift(1) non-leaking anchor
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    df["C4H8_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H8_last_valid"]
          .transform(lambda x: x.shift(1).rolling(72, min_periods=1).mean())
    )
    
    anchor_feats = physics_feats + ["C4H8_campaign_anchor"]
    
    # Re-extract split datasets with anchor feature
    train_df_anc = df[train_mask & mA_filter].dropna(subset=anchor_feats + ["C4H8_Bottom"])
    test_df_anc  = df[test_mask  & mA_filter].dropna(subset=anchor_feats + ["C4H8_Bottom"])
    
    X_train_anc, y_train_anc = train_df_anc[anchor_feats], train_df_anc["C4H8_Bottom"].values
    X_test_anc, y_test_anc   = test_df_anc[anchor_feats], test_df_anc["C4H8_Bottom"].values
    
    # Leakage check on Anchor dataset
    check_leakage(X_train_anc, y_train_anc, "Campaign Anchor Train")
    
    print("Fitting Tuned CatBoost with Campaign Anchor...")
    cb_anc = CatBoostRegressor(**best_cb_params)
    cb_anc.fit(X_train_anc, y_train_anc, verbose=0)
    pred_anc = cb_anc.predict(X_test_anc)
    
    anc_r2 = r2_score(y_test_anc, pred_anc)
    anc_mae = mean_absolute_error(y_test_anc, pred_anc)
    print(f"Campaign Anchor Tuned CatBoost: R² = {anc_r2:.4f} | MAE = {anc_mae:.4f} wt%")
    
    anc_importances = cb_anc.get_feature_importance()
    top_anc_idx = np.argmax(anc_importances)
    top_anc_feature = anchor_feats[top_anc_idx]
    top_anc_imp = anc_importances[top_anc_idx]
    
    save_to_master_leaderboard(
        "CatBoost (Tuned)", "Process+Anchor", anc_r2, anc_mae, f"{top_anc_feature} ({top_anc_imp:.2f})"
    )
    
    # =========================================================================
    # DIAGNOSTICS FOR BEST TUNED CATBOOST (PURE PHYSICS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING DIAGNOSTICS FOR THE TUNED CATBOOST (PURE PHYSICS)")
    print("=" * 80)
    
    # Plot: Residual vs Time for Tuned CatBoost
    cb_residuals = y_test - pred_cb
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["DateTime"], cb_residuals, alpha=0.6, color="forestgreen", lw=1.5)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("DateTime")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Tuned CatBoost (Pure Physics): Residual vs. Time (Block 4)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plot_time_path = os.path.join(DIAGNOSTICS_DIR, "pure_physics_catboost_plot_4_residual_vs_time.png")
    plt.savefig(plot_time_path)
    plt.close()
    print(f"Saved residual vs time plot to {plot_time_path}")
    
    # Plot: Actual vs Predicted for Tuned CatBoost
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, pred_cb, alpha=0.5, color="forestgreen", edgecolors="w", s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual C4H8_Bottom (wt%)")
    plt.ylabel("Predicted C4H8_Bottom (wt%)")
    plt.title("Tuned CatBoost (Pure Physics): Actual vs. Predicted")
    plt.tight_layout()
    plot_act_path = os.path.join(DIAGNOSTICS_DIR, "pure_physics_catboost_plot_1_actual_vs_predicted.png")
    plt.savefig(plot_act_path)
    plt.close()
    
    # Plot: SHAP values for Tuned CatBoost
    print("Generating SHAP values for Tuned CatBoost...")
    try:
        X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(tuned_cb)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
        plt.title("Tuned CatBoost: SHAP Summary (Top 20 Features)", fontsize=14, pad=20)
        plt.tight_layout()
        plot_shap_path = os.path.join(DIAGNOSTICS_DIR, "pure_physics_catboost_plot_5_shap.png")
        plt.savefig(plot_shap_path)
        plt.close()
        print(f"Saved SHAP plot to {plot_shap_path}")
        
        # Save SHAP importances to CSV
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_imp_df = pd.DataFrame({
            "Feature": X_sample.columns,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        shap_csv_path = os.path.join(DIAGNOSTICS_DIR, "pure_physics_catboost_shap_importances.csv")
        shap_imp_df.to_csv(shap_csv_path, index=False)
        
        print("\nTOP 20 SHAP FEATURES (TUNED CATBOOST PURE PHYSICS)")
        print("-" * 50)
        print(shap_imp_df.head(20).to_string(index=False))
        print("-" * 50)
        
    except Exception as e:
        print(f"Warning: SHAP calculation failed ({e}). Falling back to CatBoost build-in feature importances.")
        # Fallback to feature importance list
        shap_imp_df = pd.DataFrame({
            "Feature": physics_feats,
            "Importance": cb_importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        print("\nTOP 20 CATBOOST FEATURE IMPORTANCES")
        print("-" * 50)
        print(shap_imp_df.head(20).to_string(index=False))
        print("-" * 50)
        
    print("\n" + "=" * 80)
    print("ALL ADVANCED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    # Load and display master leaderboard
    m_leaderboard = pd.read_csv(MASTER_LEADERBOARD)
    print("\nMASTER LEADERBOARD:")
    print(m_leaderboard.to_markdown(index=False))


if __name__ == "__main__":
    main()
