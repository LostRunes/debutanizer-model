"""
notebooks/freeze_model_A.py
===========================
Freezes Model A:
1. Copies model_A_XGBoost_robust_opt.json to models/final/model_A_final_v1.json.
2. Loads the JSON model and pickles it as models/final/model_A_final_v1.pkl.
3. Extracts feature importance (gain) and saves to reports/model_A_feature_importance.csv.
4. Copies optuna results and key diagnostic plots to models/final/.
"""

import os
import shutil
import pickle
import json
import pandas as pd
from xgboost import XGBRegressor

# Paths
JSON_SRC = "models/model_A_XGBoost_robust_opt.json"
JSON_DEST = "models/final/model_A_final_v1.json"
PKL_DEST = "models/final/model_A_final_v1.pkl"
OPTUNA_SRC = "experiments/robust_xgb_optuna_results.json"
OPTUNA_DEST = "models/final/robust_xgb_optuna_results.json"
REPORT_DEST = "reports/model_A_feature_importance.csv"

os.makedirs("models/final", exist_ok=True)
os.makedirs("reports", exist_ok=True)

def main():
    print("=== FREEZING MODEL A ===")
    
    # 1. Copy JSON model
    if os.path.exists(JSON_SRC):
        shutil.copy(JSON_SRC, JSON_DEST)
        print(f"Copied JSON model to {JSON_DEST}")
    else:
        print(f"Error: {JSON_SRC} not found!")
        return
        
    # 2. Load and pickle the model
    model = XGBRegressor()
    model.load_model(JSON_DEST)
    
    with open(PKL_DEST, "wb") as f:
        pickle.dump(model, f)
    print(f"Pickled model saved to {PKL_DEST}")
    
    # 3. Generate feature importance (gain) report
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    
    # Ensure all 8 robust features are listed, even if they have 0 gain (not split)
    robust_features = [
        "C4H8_campaign_anchor", "Steam_Feed_Ratio", "Reflux_Ratio",
        "Reboiling_Steam_Flow_dev24h", "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h", "Control_Tray_Temp_dev24h", "Column_Top_Pressure_dev24h"
    ]
    
    importance_list = []
    for feat in robust_features:
        gain = gain_scores.get(feat, 0.0)
        importance_list.append({"Feature": feat, "Gain": gain})
        
    importance_df = pd.DataFrame(importance_list).sort_values(by="Gain", ascending=False)
    importance_df.to_csv(REPORT_DEST, index=False)
    print(f"Saved feature importance report to {REPORT_DEST}")
    print("\nFeature Importance (Gain):")
    print(importance_df.to_string(index=False))
    
    # 4. Copy Optuna results
    if os.path.exists(OPTUNA_SRC):
        shutil.copy(OPTUNA_SRC, OPTUNA_DEST)
        print(f"Copied Optuna results to {OPTUNA_DEST}")
        
    # 5. Copy best plots
    diag_plots = [
        "robust_opt_plot_1_actual_vs_predicted.png",
        "robust_opt_plot_4_residual_vs_time.png"
    ]
    for plot in diag_plots:
        src = os.path.join("experiments/diagnostics", plot)
        dest = os.path.join("models/final", plot)
        if os.path.exists(src):
            shutil.copy(src, dest)
            print(f"Copied plot {plot} to models/final/")
            
    print("\nModel A successfully frozen and documented!")

if __name__ == "__main__":
    main()
