import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error

def main():
    features_file = "data/features.parquet"
    if not os.path.exists(features_file):
        features_file = "../data/features.parquet"
        
    df = pd.read_parquet(features_file)
    
    # 1. Define last valid C4H6 (stuck and <= 0.001 excluded)
    df["C4H6_last_valid"] = df["C4H6_Bottom"].copy()
    df.loc[df["C4H6_Bottom_stuck"] | (df["C4H6_Bottom"] <= 0.001), "C4H6_last_valid"] = np.nan
    
    # Target anchor with 12h and 72h limits (leak-free shift(1))
    df["anchor_12h"] = df.groupby("Data_Block")["C4H6_last_valid"].transform(lambda x: x.shift(1).ffill(limit=12))
    df["anchor_72h"] = df.groupby("Data_Block")["C4H6_last_valid"].transform(lambda x: x.shift(1).ffill(limit=72))
    
    # 2. Compute 24h rolling mean of the anchor (leak-free since anchor is already shifted by 1)
    df["anchor_12h_roll24h"] = df.groupby("Data_Block")["anchor_12h"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    df["anchor_72h_roll24h"] = df.groupby("Data_Block")["anchor_72h"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    
    # 3. Filter for Block 4 and healthy target rows
    test_mask = df["Data_Block"] == 4
    mB_filter = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)
    
    test_df = df[test_mask & mB_filter]
    y_test = test_df["C4H6_Bottom"].values
    
    print("================================================================================")
    print("MODEL B (C4H6) ANCHOR-ONLY BASELINES ON BLOCK 4")
    print("================================================================================")
    print(f"Number of test rows: {len(y_test)}")
    print(f"Target distribution - Mean: {y_test.mean():.6f} | Var: {y_test.var():.8f}\n")
    
    # We define our baselines dictionary
    baselines = {
        "Baseline A (12h Anchor)":           test_df["anchor_12h"].values,
        "Baseline A (72h Anchor)":           test_df["anchor_72h"].values,
        "Baseline B (24h Roll 12h Anchor)":  test_df["anchor_12h_roll24h"].values,
        "Baseline B (24h Roll 72h Anchor)":  test_df["anchor_72h_roll24h"].values,
        "Baseline C (Block 4 Mean: 0.005663)": np.full_like(y_test, 0.005663),
        "Baseline C (Train Mean: 0.0273)":    np.full_like(y_test, 0.0273)
    }
    
    print(f"{'Baseline Model':35s} | {'R2 Score':10s} | {'MAE (wt%)':10s} | {'Pearson':8s} | {'Available %':11s}")
    print("-" * 85)
    
    for name, preds in baselines.items():
        # Drop rows where prediction is NaN to evaluate performance fairly on available data
        valid_mask = ~pd.isna(preds)
        avail_pct = valid_mask.mean() * 100
        
        if valid_mask.sum() == 0:
            print(f"{name:35s} | {'N/A':10s} | {'N/A':10s} | {'N/A':8s} | {avail_pct:10.2f}%")
            continue
            
        y_test_sub = y_test[valid_mask]
        preds_sub = preds[valid_mask]
        
        r2 = r2_score(y_test_sub, preds_sub)
        mae = mean_absolute_error(y_test_sub, preds_sub)
        
        try:
            if len(np.unique(preds_sub)) > 1:
                pear, _ = pearsonr(y_test_sub, preds_sub)
                pear_str = f"{pear:+.4f}"
            else:
                pear_str = "Constant"
        except:
            pear_str = "N/A"
            
        print(f"{name:35s} | {r2:10.4f} | {mae:10.4f}  | {pear_str:8s} | {avail_pct:10.2f}%")
    print("================================================================================")

if __name__ == "__main__":
    main()
