import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

FEATURES_FILE = "data/features.parquet"

def main():
    df = pd.read_parquet(FEATURES_FILE)
    
    # 1. Define Campaign Anchor for C4H6 (leak-free shift(1))
    df["C4H6_last_valid"] = df["C4H6_Bottom"].copy()
    df.loc[df["C4H6_Bottom_stuck"] | (df["C4H6_Bottom"] <= 0.001), "C4H6_last_valid"] = np.nan
    
    # Target anchor with 12h limit
    df["C4H6_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H6_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=12))
    )
    
    # Robust features set
    feats = [
        "C4H6_campaign_anchor",
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
    mB_filter = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)
    
    train_clean = df[train_mask & mB_filter].dropna(subset=feats + ["C4H6_Bottom"])
    test_clean  = df[test_mask  & mB_filter].dropna(subset=feats + ["C4H6_Bottom"])
    
    X_train, y_train = train_clean[feats], train_clean["C4H6_Bottom"].values
    X_test, y_test   = test_clean[feats], test_clean["C4H6_Bottom"].values
    
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Train mean target: {y_train.mean():.4f} wt% | Test mean target: {y_test.mean():.4f} wt%")
    print(f"Train var target:  {y_train.var():.6f} | Test var target:  {y_test.var():.6f}")
    
    models = {
        "XGBoost (depth=3)": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        "XGBoost (depth=2)": XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42),
        "LightGBM (depth=3)": LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost (depth=3)": CatBoostRegressor(iterations=100, depth=3, learning_rate=0.05, random_state=42, verbose=0),
        "Ridge Regression": Ridge(alpha=10.0),
        "Mean Predictor": None
    }
    
    print(f"\n{'Model':25s} | {'Pearson':8s} | {'R2 Score':10s} | {'MAE':10s}")
    print("-" * 65)
    
    for name, model in models.items():
        if name == "Mean Predictor":
            preds = np.full_like(y_test, y_train.mean())
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        try:
            pear, _ = pearsonr(y_test, preds)
            pear_str = f"{pear:+.4f}"
        except:
            pear_str = "N/A"
            
        print(f"{name:25s} | {pear_str:8s} | {r2:10.4f} | {mae:10.4f} wt%")
        
        # Print feature importance for XGBoost depth=3
        if name == "XGBoost (depth=3)":
            print("\nFeature Importances for XGBoost (depth=3):")
            for f, imp in zip(feats, model.feature_importances_):
                print(f"  {f:30s}: {imp:.4f}")
            print()

if __name__ == "__main__":
    main()
