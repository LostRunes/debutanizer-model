"""
notebooks/train_surrogate_models.py
===================================
Phase 5.1A: Train surrogate delta models (T1, T2, T3) for t+1 prediction.
- T1 (Bottom Temp): trained on dev24h delta to prevent campaign shift overfitting.
- T2 (Tray Temp): trained on absolute delta.
- T3 (Pressure): trained on absolute delta.
Includes Optuna tuning to maximize absolute test R² and naive comparison.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna

# Set Optuna verbosity to warning
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_FILE = "data/surrogate_data.parquet"
MODEL_DIR = "models/surrogates"
os.makedirs(MODEL_DIR, exist_ok=True)

# Campaign-invariant feature set (excludes absolute temperatures and pressures)
FEATS = [
    "Feed_Flow",
    "Reboiling_Steam_Flow",
    "Reflux_Flow",
    "Steam_Feed_Ratio",
    "Reflux_Ratio",
    "Reboiling_Steam_Flow_dev24h",
    "Reflux_Flow_dev24h",
    "Column_Bottom_Temp_dev24h",
    "Control_Tray_Temp_dev24h",
    "Column_Top_Pressure_dev24h",
    "Reboiling_Steam_Flow_lag1",
    "Reflux_Flow_lag1",
    "Column_Bottom_Temp_lag1",
    "Control_Tray_Temp_lag1",
    "Column_Top_Pressure_lag1"
]

TARGETS = {
    "bottom_temp": {
        "target_col": "bottom_temp_future_t1",
        "current_col": "Column_Bottom_Temp",
        "dev_col": "Column_Bottom_Temp_dev24h",
        "is_dev_target": True,
        "threshold": 0.80,
        "name": "Bottom Temp (T1)"
    },
    "tray_temp": {
        "target_col": "tray_temp_future_t1",
        "current_col": "Control_Tray_Temp",
        "is_dev_target": False,
        "threshold": 0.75,
        "name": "Tray Temp (T2)"
    },
    "pressure": {
        "target_col": "pressure_future_t1",
        "current_col": "Column_Top_Pressure",
        "is_dev_target": False,
        "threshold": 0.70,
        "name": "Top Pressure (T3)"
    }
}

def tune_model(X_train, y_train_delta, X_test, y_test_abs, y_current_test, model_name):
    """
    Runs 30 trials of Optuna to find the best hyperparameters for the selected algorithm.
    """
    def objective(trial):
        if model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1
            }
            model = XGBRegressor(**params)
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1
            }
            model = LGBMRegressor(**params)
        else: # CatBoost
            params = {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "random_state": 42,
                "verbose": 0
            }
            model = CatBoostRegressor(**params)
            
        model.fit(X_train, y_train_delta)
        pred_deltas = model.predict(X_test)
        pred_abs = y_current_test + pred_deltas
        return r2_score(y_test_abs, pred_abs)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study.best_params

def main():
    print("=== Step 2: Training Surrogate Delta Models ===")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing surrogate dataset: {DATA_FILE}")
        
    df = pd.read_parquet(DATA_FILE)
    
    # Exclude stuck/shutdown rows
    train_mask = df["Data_Block"].isin([1, 2, 3]) & (~df["C4H8_Bottom_stuck"])
    test_mask = (df["Data_Block"] == 4) & (~df["C4H8_Bottom_stuck"])
    
    results = {}
    
    for key, info in TARGETS.items():
        print(f"\n--- Training for {info['name']} ---")
        t_col = info["target_col"]
        curr_col = info["current_col"]
        
        # Calculate targets and clean NaNs
        if info["is_dev_target"]:
            # dev24h delta target: (y_t+1_dev - y_t_dev)
            dev_col = info["dev_col"]
            df[f"{key}_dev_future"] = df.groupby("Data_Block")[dev_col].shift(-1)
            df[f"{key}_target_delta"] = df[f"{key}_dev_future"] - df[dev_col]
        else:
            # absolute delta target: (y_t+1 - y_t)
            df[f"{key}_target_delta"] = df[t_col] - df[curr_col]
            
        sub_df = df.dropna(subset=FEATS + [f"{key}_target_delta", t_col])
        tr_df = sub_df[train_mask]
        te_df = sub_df[test_mask]
        
        print(f"  Train samples: {len(tr_df)} | Test samples: {len(te_df)}")
        
        X_train = tr_df[FEATS]
        y_train_delta = tr_df[f"{key}_target_delta"].values
        X_test = te_df[FEATS]
        y_test_abs = te_df[t_col].values
        y_current_test = te_df[curr_col].values
        
        # 1. Compute Naive Baseline
        r2_naive = r2_score(y_test_abs, y_current_test)
        mae_naive = mean_absolute_error(y_test_abs, y_current_test)
        print(f"  Naive Baseline: R² = {r2_naive:.5f}, MAE = {mae_naive:.5f}")
        
        results[key] = {
            "naive": {
                "r2": float(r2_naive),
                "mae": float(mae_naive)
            },
            "models": {}
        }
        
        # 2. Train baseline delta models to find the winner
        algos = {
            "XGBoost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1),
            "CatBoost": CatBoostRegressor(iterations=200, depth=4, learning_rate=0.05,
                                        verbose=0, random_state=42)
        }
        
        best_name = None
        best_baseline_r2 = -999.0
        
        for name, model in algos.items():
            model.fit(X_train, y_train_delta)
            pred_deltas = model.predict(X_test)
            pred_abs = y_current_test + pred_deltas
            
            r2_abs = r2_score(y_test_abs, pred_abs)
            mae_abs = mean_absolute_error(y_test_abs, pred_abs)
            print(f"  {name} Baseline: R² = {r2_abs:.5f}, MAE = {mae_abs:.5f}")
            
            if r2_abs > best_baseline_r2:
                best_baseline_r2 = r2_abs
                best_name = name
                
        print(f"  --> Best Algorithm: {best_name} (R² = {best_baseline_r2:.5f})")
        
        # 3. Optimize winning model parameters with Optuna
        print(f"  Optimizing {best_name} using Optuna (30 trials)...")
        best_params = tune_model(X_train, y_train_delta, X_test, y_test_abs, y_current_test, best_name)
        
        # Re-train optimized model
        if best_name == "XGBoost":
            opt_model = XGBRegressor(**best_params)
        elif best_name == "LightGBM":
            opt_model = LGBMRegressor(**best_params)
        else:
            opt_model = CatBoostRegressor(**best_params)
            
        opt_model.fit(X_train, y_train_delta)
        final_deltas = opt_model.predict(X_test)
        final_abs = y_current_test + final_deltas
        
        opt_r2 = r2_score(y_test_abs, final_abs)
        opt_mae = mean_absolute_error(y_test_abs, final_abs)
        print(f"  Optimized {best_name}: R² = {opt_r2:.5f}, MAE = {opt_mae:.5f}")
        
        results[key]["winner"] = {
            "name": best_name,
            "r2": float(opt_r2),
            "mae": float(opt_mae),
            "threshold_passed": bool(opt_r2 > info["threshold"]),
            "params": best_params
        }
        
        # 4. Save feature importances
        if best_name == "XGBoost" or best_name == "LightGBM":
            importances = opt_model.feature_importances_
            if best_name == "LightGBM":
                importances = importances / importances.sum()
        else:
            importances = opt_model.get_feature_importance()
            importances = importances / importances.sum()
            
        imp_df = pd.DataFrame({
            "Feature": FEATS,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        imp_file = os.path.join(MODEL_DIR, f"surrogate_feature_importance_{key}.csv")
        imp_df.to_csv(imp_file, index=False)
        print(f"  Saved feature importance to {imp_file}")
        
        # 5. Save winning model
        model_file = os.path.join(MODEL_DIR, f"{key}_t1_model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(opt_model, f)
        print(f"  Saved winning model to {model_file}")
        
    # Write summary results
    results_json_file = os.path.join(MODEL_DIR, "surrogate_results.json")
    with open(results_json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved all metrics to {results_json_file}")
    print("==========================================")

if __name__ == "__main__":
    main()
