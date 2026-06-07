"""
notebooks/surrogate_diagnostics.py
==================================
Phase 5.1A: Generate diagnostic plots for surrogate models.
Plots saved to experiments/diagnostics/surrogates/:
1. {target}_actual_vs_predicted.png
2. {target}_residual_vs_time.png
3. {target}_residual_histogram.png

Also copies feature importance files to exact names requested by user.
"""

import os
import pickle
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# Set styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

DATA_FILE = "data/surrogate_data.parquet"
MODEL_DIR = "models/surrogates"
DIAGNOSTICS_DIR = "experiments/diagnostics/surrogates"
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

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
        "name": "Bottom Temp (T1)",
        "unit": "°C"
    },
    "tray_temp": {
        "target_col": "tray_temp_future_t1",
        "current_col": "Control_Tray_Temp",
        "name": "Tray Temp (T2)",
        "unit": "°C"
    },
    "pressure": {
        "target_col": "pressure_future_t1",
        "current_col": "Column_Top_Pressure",
        "name": "Top Pressure (T3)",
        "unit": "bar"
    }
}

def main():
    print("=== Step 3 & 4: Generating Diagnostics and Plots ===")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing surrogate dataset: {DATA_FILE}")
        
    df = pd.read_parquet(DATA_FILE)
    
    # Select test set (Block 4, non-stuck)
    test_mask = (df["Data_Block"] == 4) & (~df["C4H8_Bottom_stuck"])
    test_df = df[test_mask].dropna(subset=FEATS)
    X_test = test_df[FEATS]
    
    for key, info in TARGETS.items():
        print(f"\nGenerating plots for {info['name']}...")
        t_col = info["target_col"]
        curr_col = info["current_col"]
        unit = info["unit"]
        
        # Load winning model
        model_file = os.path.join(MODEL_DIR, f"{key}_t1_model.pkl")
        with open(model_file, "rb") as f:
            model = pickle.load(f)
            
        # Predict deltas and reconstruct absolute values
        pred_deltas = model.predict(X_test)
        y_current = test_df[curr_col].values
        y_actual = test_df[t_col].values
        y_pred = y_current + pred_deltas
        
        residuals = y_actual - y_pred
        
        # Plot 1: Actual vs. Predicted
        plt.figure(figsize=(6, 5))
        plt.scatter(y_actual, y_pred, alpha=0.4, color="teal", edgecolors="w", s=25)
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], "r--", lw=2)
        plt.xlabel(f"Actual Future {key.replace('_', ' ').title()} ({unit})")
        plt.ylabel(f"Predicted Future {key.replace('_', ' ').title()} ({unit})")
        plt.title(f"{info['name']}: Actual vs. Predicted (Block 4)")
        plt.tight_layout()
        plt.savefig(os.path.join(DIAGNOSTICS_DIR, f"{key}_actual_vs_predicted.png"), dpi=150)
        plt.close()
        
        # Plot 2: Residual vs. Time
        plt.figure(figsize=(10, 4.5))
        plt.plot(test_df["DateTime"], residuals, alpha=0.5, color="purple", lw=1.2)
        plt.axhline(y=0, color="red", linestyle="--", lw=1.5)
        plt.xlabel("DateTime")
        plt.ylabel(f"Residual ({unit})")
        plt.title(f"{info['name']}: Residuals vs. Time (Block 4)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(DIAGNOSTICS_DIR, f"{key}_residual_vs_time.png"), dpi=150)
        plt.close()
        
        # Plot 3: Residual Histogram
        plt.figure(figsize=(6, 5))
        sns.histplot(residuals, kde=True, color="indigo", bins=40, stat="density")
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        plt.axvline(mean_res, color="red", linestyle="--", lw=1.5, 
                    label=f"Mean: {mean_res:.4f}\nStd: {std_res:.4f}")
        plt.xlabel(f"Residual ({unit})")
        plt.title(f"{info['name']}: Residual Distribution (Block 4)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DIAGNOSTICS_DIR, f"{key}_residual_histogram.png"), dpi=150)
        plt.close()
        
        print(f"  Saved diagnostic plots to {DIAGNOSTICS_DIR}")

    # Copy feature importance CSV files to the exact names requested by user
    renames = {
        "surrogate_feature_importance_bottom_temp.csv": "surrogate_feature_importance_bottom.csv",
        "surrogate_feature_importance_tray_temp.csv": "surrogate_feature_importance_tray.csv",
        "surrogate_feature_importance_pressure.csv": "surrogate_feature_importance_pressure.csv"
    }
    
    print("\nCopying/Renaming feature importance CSVs:")
    for src, dst in renames.items():
        if src == dst:
            continue
        src_path = os.path.join(MODEL_DIR, src)
        dst_path = os.path.join(MODEL_DIR, dst)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            print(f"  Copied {src} -> {dst}")
        else:
            print(f"  Warning: {src} not found to rename.")
            
    print("\n===========================================")

if __name__ == "__main__":
    main()
