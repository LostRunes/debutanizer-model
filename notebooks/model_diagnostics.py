"""
notebooks/model_diagnostics.py
==============================
Generates diagnostic plots and SHAP analysis for the current best model (XGBoost with block).
Plots generated:
1. Actual vs Predicted (scatter)
2. Residual histogram
3. Residual vs Column_Top_Pressure
4. Residual vs Time (DateTime)
5. SHAP Summary (top 20 features)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import shap

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Paths
FEATURES_FILE = os.path.join("data", "features.parquet")
MODEL_FILE = os.path.join("models", "model_a_with_block.json")
DIAGNOSTICS_DIR = os.path.join("experiments", "diagnostics")
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("RUNNING MODEL DIAGNOSTICS FOR CURRENT BEST MODEL")
    print("=" * 80)
    
    # 1. Load data
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded feature dataset: {df.shape}")
    
    # Define features (must match what model was trained on)
    TARGET_LAG_COLS = (
        [f"C4H8_Bottom_lag{i}" for i in range(1, 13)] +
        [f"C4H6_Bottom_lag{i}" for i in range(1, 4)]
    )
    META_COLS = [
        "DateTime", "C4H6_Bottom", "C4H8_Bottom", "Total_C4",
        "C4H6_Bottom_stuck", "C4H8_Bottom_stuck",
        "hours_since_C4H6_Bottom_change", "hours_since_C4H8_Bottom_change",
        "Analyzer_Health", "is_extreme_event",
    ] + TARGET_LAG_COLS
    
    # 2. Columns definition expected by the baseline model
    EXPECTED_FEATURES = [
        'Feed_Flow', 'Reboiler_Outlet_Temp', 'Column_Top_Temp', 'Reboiling_Steam_Flow', 
        'Reflux_Flow', 'Column_Top_Pressure', 'Column_Bottom_Temp', 'Control_Tray_Temp', 
        'Data_Block', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 
        'Reboiling_Steam_Flow_lag1', 'Reboiling_Steam_Flow_lag2', 'Reboiling_Steam_Flow_lag3', 
        'Reboiling_Steam_Flow_lag6', 'Reboiling_Steam_Flow_lag12', 'Reflux_Flow_lag1', 
        'Reflux_Flow_lag2', 'Reflux_Flow_lag3', 'Reflux_Flow_lag6', 'Feed_Flow_lag1', 
        'Feed_Flow_lag2', 'Feed_Flow_lag3', 'Column_Bottom_Temp_lag1', 'Column_Bottom_Temp_lag2', 
        'Column_Bottom_Temp_lag3', 'Control_Tray_Temp_lag1', 'Control_Tray_Temp_lag2', 
        'Control_Tray_Temp_lag3', 'Control_Tray_Temp_lag6', 'Reboiler_Outlet_Temp_lag1', 
        'Reboiler_Outlet_Temp_lag2', 'Reboiler_Outlet_Temp_lag3', 'Column_Top_Temp_lag1', 
        'Column_Top_Temp_lag2', 'Column_Top_Pressure_lag1', 'Column_Top_Pressure_lag2', 
        'Reboiling_Steam_Flow_roll_mean_3h', 'Reboiling_Steam_Flow_roll_mean_6h', 
        'Reboiling_Steam_Flow_roll_mean_12h', 'Reflux_Flow_roll_mean_3h', 'Reflux_Flow_roll_mean_6h', 
        'Reflux_Flow_roll_mean_12h', 'Feed_Flow_roll_mean_3h', 'Feed_Flow_roll_mean_6h', 
        'Feed_Flow_roll_mean_12h', 'Column_Bottom_Temp_roll_mean_3h', 'Column_Bottom_Temp_roll_mean_6h', 
        'Column_Bottom_Temp_roll_mean_12h', 'Reboiling_Steam_Flow_roll_std_3h', 
        'Reboiling_Steam_Flow_roll_std_6h', 'Reflux_Flow_roll_std_3h', 'Reflux_Flow_roll_std_6h', 
        'Feed_Flow_roll_std_3h', 'Feed_Flow_roll_std_6h', 'Reflux_Ratio', 'Steam_Feed_Ratio', 
        'Temp_Gradient', 'Reboiler_Delta', 'Reboiling_Steam_Flow_diff1', 'Reflux_Flow_diff1', 
        'Feed_Flow_diff1', 'Column_Bottom_Temp_diff1'
    ]
    
    # Filter Block 4 test data
    test_mask = df["Data_Block"] == 4
    mA_filter = ~df["C4H8_Bottom_stuck"]
    
    # Reconstruct the expected diff columns from new names
    test_df = df[test_mask & mA_filter].copy()
    test_df["Reboiling_Steam_Flow_diff1"] = test_df["Steam_diff1"]
    test_df["Reflux_Flow_diff1"] = test_df["Reflux_diff1"]
    test_df["Feed_Flow_diff1"] = test_df["Feed_diff1"]
    test_df["Column_Bottom_Temp_diff1"] = test_df["Bottom_Temp_diff1"]
    
    test_df = test_df.dropna(subset=EXPECTED_FEATURES + ["C4H8_Bottom"])
    print(f"Test data size (Block 4): {test_df.shape}")
    
    X_test = test_df[EXPECTED_FEATURES]
    y_test = test_df["C4H8_Bottom"]
    
    # 3. Fit model in memory (avoids SHAP TreeExplainer JSON-loading bug)
    print("Fitting model in memory (identical to baseline)...")
    train_mask = df["Data_Block"].isin([1, 2, 3])
    train_df = df[train_mask & mA_filter].copy()
    train_df["Reboiling_Steam_Flow_diff1"] = train_df["Steam_diff1"]
    train_df["Reflux_Flow_diff1"] = train_df["Reflux_diff1"]
    train_df["Feed_Flow_diff1"] = train_df["Feed_diff1"]
    train_df["Column_Bottom_Temp_diff1"] = train_df["Bottom_Temp_diff1"]
    train_df = train_df.dropna(subset=EXPECTED_FEATURES + ["C4H8_Bottom"])
    
    X_train = train_df[EXPECTED_FEATURES]
    y_train = train_df["C4H8_Bottom"]
    
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model fitting complete.")
    
    # 4. Predict & Calculate residuals
    preds = model.predict(X_test)
    residuals = y_test - preds
    
    # 4. Generate Plot 1: Actual vs predicted
    print("Generating Plot 1: Actual vs Predicted...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.5, color="teal", edgecolors="w", s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual C4H8_Bottom (wt%)")
    plt.ylabel("Predicted C4H8_Bottom (wt%)")
    plt.title("Plot 1: Actual vs. Predicted C4H8_Bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "plot_1_actual_vs_predicted.png"))
    plt.close()
    
    # Generate Plot 2: Residual histogram
    print("Generating Plot 2: Residual Histogram...")
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="purple", bins=50)
    plt.axvline(x=0, color="red", linestyle="--", lw=2)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Plot 2: Residual Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "plot_2_residual_histogram.png"))
    plt.close()
    
    # Generate Plot 3: Residual vs Column_Top_Pressure
    print("Generating Plot 3: Residual vs Pressure...")
    plt.figure(figsize=(8, 6))
    plt.scatter(test_df["Column_Top_Pressure"], residuals, alpha=0.5, color="coral", edgecolors="w", s=40)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("Column_Top_Pressure (kg/cm²g)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Plot 3: Residual vs. Column_Top_Pressure")
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "plot_3_residual_vs_pressure.png"))
    plt.close()
    
    # Generate Plot 4: Residual vs Time
    print("Generating Plot 4: Residual vs Time...")
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["DateTime"], residuals, alpha=0.6, color="royalblue", lw=1.5)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("DateTime")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Plot 4: Residual vs. Time")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "plot_4_residual_vs_time.png"))
    plt.close()
    
    # Generate Plot 5: SHAP Summary (using sample for speed)
    print("Generating Plot 5: SHAP Summary...")
    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    
    # Apply monkeypatch to fix SHAP TreeExplainer JSON-loading base_score bug
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
            # Fallback if there is still something strange
            return original_float(str(val).strip('[] \t\n\r'))
            
    shap.explainers._tree.float = custom_float_parser
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.title("Plot 5: SHAP Summary Plot (Top 20 Features)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(DIAGNOSTICS_DIR, "plot_5_shap_summary.png"))
    plt.close()
    
    # Print top 20 SHAP features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "Feature": X_sample.columns,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    print("\n" + "=" * 50)
    print("TOP 20 SHAP FEATURES BY MEAN ABSOLUTE VALUE")
    print("=" * 50)
    print(shap_imp_df.head(20).to_string(index=False))
    print("=" * 50)
    
    # Save SHAP importances to csv
    shap_imp_df.to_csv(os.path.join(DIAGNOSTICS_DIR, "shap_importances.csv"), index=False)
    print(f"SHAP importances saved to {os.path.join(DIAGNOSTICS_DIR, 'shap_importances.csv')}")
    print("All diagnostic plots successfully saved to experiments/diagnostics/")

if __name__ == "__main__":
    main()
