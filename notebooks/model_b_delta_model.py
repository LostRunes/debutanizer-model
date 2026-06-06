import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import shap

def main():
    features_file = "data/features.parquet"
    if not os.path.exists(features_file):
        features_file = "../data/features.parquet"
        
    df = pd.read_parquet(features_file)
    
    # 1. Define last valid C4H6 (stuck and <= 0.001 excluded)
    df["C4H6_last_valid"] = df["C4H6_Bottom"].copy()
    df.loc[df["C4H6_Bottom_stuck"] | (df["C4H6_Bottom"] <= 0.001), "C4H6_last_valid"] = np.nan
    
    # Target anchor with 12h limit (leak-free shift(1))
    df["C4H6_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H6_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=12))
    )
    
    # Target delta
    df["Delta_C4H6"] = df["C4H6_Bottom"] - df["C4H6_campaign_anchor"]
    
    # 7 Process features only
    feats = [
        "Steam_Feed_Ratio",
        "Reflux_Ratio",
        "Reboiling_Steam_Flow_dev24h",
        "Reflux_Flow_dev24h",
        "Column_Bottom_Temp_dev24h",
        "Control_Tray_Temp_dev24h",
        "Column_Top_Pressure_dev24h"
    ]
    
    # Splits (Blocks 2+3 for Train, Block 4 for Test)
    train_mask = df["Data_Block"].isin([2, 3])
    test_mask  = df["Data_Block"] == 4
    mB_filter = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)
    
    train_clean = df[train_mask & mB_filter].dropna(subset=feats + ["C4H6_Bottom", "C4H6_campaign_anchor"])
    test_clean  = df[test_mask  & mB_filter].dropna(subset=feats + ["C4H6_Bottom", "C4H6_campaign_anchor"])
    
    X_train, y_train_delta = train_clean[feats], train_clean["Delta_C4H6"].values
    X_test, y_test_delta   = test_clean[feats], test_clean["Delta_C4H6"].values
    
    y_test_actual = test_clean["C4H6_Bottom"].values
    test_anchor   = test_clean["C4H6_campaign_anchor"].values
    
    print("================================================================================")
    print("MODEL B (C4H6) DELTA CORRECTION MODEL")
    print("================================================================================")
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Train Mean Delta: {y_train_delta.mean():.6f} | Test Mean Delta: {y_test_delta.mean():.6f}")
    
    # Train delta model (XGBoost depth=3)
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_delta)
    
    # Predict delta
    pred_delta = model.predict(X_test)
    
    # Reconstruct final predictions: anchor + predicted_delta
    pred_final = test_anchor + pred_delta
    
    # Apply physical clip (C4H6 cannot be negative)
    pred_final_clipped = np.clip(pred_final, a_min=0.0, a_max=None)
    
    # Evaluate
    r2_anchor = r2_score(y_test_actual, test_anchor)
    mae_anchor = mean_absolute_error(y_test_actual, test_anchor)
    pear_anchor, _ = pearsonr(y_test_actual, test_anchor)
    
    r2_delta = r2_score(y_test_actual, pred_final_clipped)
    mae_delta = mean_absolute_error(y_test_actual, pred_final_clipped)
    pear_delta, _ = pearsonr(y_test_actual, pred_final_clipped)
    
    print("\n--- RESULTS COMPARISON (BLOCK 4 HEALTHY ROWS) ---")
    print(f"1. Baseline (12h Anchor Only):")
    print(f"   R² Score:            {r2_anchor:.6f}")
    print(f"   MAE:                 {mae_anchor:.6f} wt% ({mae_anchor*10000:.1f} ppm)")
    print(f"   Pearson Correlation: {pear_anchor:+.4f}")
    
    print(f"\n2. Delta Correction Model (12h Anchor + Predicted Delta):")
    print(f"   R² Score:            {r2_delta:.6f}")
    print(f"   MAE:                 {mae_delta:.6f} wt% ({mae_delta*10000:.1f} ppm)")
    print(f"   Pearson Correlation: {pear_delta:+.4f}")
    
    # Feature Importances (Gain)
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    importance_list = [{"Feature": f, "Gain": gain_scores.get(f, 0.0)} for f in feats]
    importance_df = pd.DataFrame(importance_list).sort_values(by="Gain", ascending=False)
    
    print("\n--- FEATURE IMPORTANCES FOR DELTA MODEL (GAIN) ---")
    print(importance_df.to_string(index=False))
    
    # SHAP top features
    print("\n--- SHAP VALUES TOP FEATURES ---")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": feats, "Mean_Abs_SHAP": mean_shap}).sort_values(by="Mean_Abs_SHAP", ascending=False)
        print(shap_df.to_string(index=False))
    except Exception as e:
        print(f"SHAP extraction failed: {e}")
        
    print("================================================================================")

if __name__ == "__main__":
    main()
