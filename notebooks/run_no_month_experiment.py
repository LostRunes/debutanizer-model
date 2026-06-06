"""
notebooks/run_no_month_experiment.py
====================================
Runs an experiment comparing:
1. Current feature set (excluding Data_Block, but including month features)
2. No Month feature set (excluding both Data_Block and month_sin/month_cos)

Trains XGBoost, LightGBM, and CatBoost on both sets.
Generates SHAP analysis for the No Month model.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt

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
    print("=" * 80)
    print("RUNNING MONTH FEATURES ABLATION TEST")
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
    
    # Feature sets
    current_feats = [c for c in df.columns if c not in META_COLS]
    no_month_feats = [c for c in current_feats if c not in ["month_sin", "month_cos"]]
    
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    train_df = df[train_mask & mA_filter].dropna(subset=current_feats + ["C4H8_Bottom"])
    test_df  = df[test_mask  & mA_filter].dropna(subset=current_feats + ["C4H8_Bottom"])
    
    y_train = train_df["C4H8_Bottom"].values
    y_test  = test_df["C4H8_Bottom"].values
    
    results = []
    
    # 2. Train and evaluate loop
    for name, feats in [("Current (87 feats)", current_feats), ("No Month (85 feats)", no_month_feats)]:
        print(f"\n--- Feature Set: {name} ---")
        X_train, X_test = train_df[feats], test_df[feats]
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_preds)
        xgb_mae = mean_absolute_error(y_test, xgb_preds)
        print(f"  XGBoost  | R² = {xgb_r2:.4f} | MAE = {xgb_mae:.4f} wt%")
        results.append({"Set": name, "Model": "XGBoost", "R2": xgb_r2, "MAE": xgb_mae})
        
        # LightGBM (default parameters)
        lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict(X_test)
        lgb_r2 = r2_score(y_test, lgb_preds)
        lgb_mae = mean_absolute_error(y_test, lgb_preds)
        print(f"  LightGBM | R² = {lgb_r2:.4f} | MAE = {lgb_mae:.4f} wt%")
        results.append({"Set": name, "Model": "LightGBM", "R2": lgb_r2, "MAE": lgb_mae})
        
        # CatBoost
        cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
        cb.fit(X_train, y_train)
        cb_preds = cb.predict(X_test)
        cb_r2 = r2_score(y_test, cb_preds)
        cb_mae = mean_absolute_error(y_test, cb_preds)
        print(f"  CatBoost | R² = {cb_r2:.4f} | MAE = {cb_mae:.4f} wt%")
        results.append({"Set": name, "Model": "CatBoost", "R2": cb_r2, "MAE": cb_mae})
        
        # Generate SHAP for the No Month XGBoost model
        if "No Month" in name:
            print("  Generating SHAP summary for No Month model...")
            X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
            plt.title("SHAP Summary Plot - No Month Features (XGBoost)", fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(DIAGNOSTICS_DIR, "plot_5_shap_no_month.png"))
            plt.close()
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_imp_df = pd.DataFrame({
                "Feature": X_sample.columns,
                "Mean_Abs_SHAP": mean_abs_shap
            }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
            
            print("\n" + "=" * 50)
            print("TOP 20 SHAP FEATURES (NO MONTH FEATURES MODEL)")
            print("=" * 50)
            print(shap_imp_df.head(20).to_string(index=False))
            print("=" * 50)
            shap_imp_df.to_csv(os.path.join(DIAGNOSTICS_DIR, "shap_importances_no_month.csv"), index=False)
            
    # 3. Print Leaderboard Comparison
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS: CURRENT FEATURES VS NO MONTH FEATURES")
    print("=" * 80)
    res_df = pd.DataFrame(results)
    pivot_df = res_df.pivot(index="Model", columns="Set", values=["R2", "MAE"])
    print(pivot_df.round(4))
    print("=" * 80)

if __name__ == "__main__":
    main()
