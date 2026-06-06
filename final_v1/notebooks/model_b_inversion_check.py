import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error

def evaluate_block_anchor(df, target_block, limit=12):
    # Filter for target block and healthy C4H6 rows
    block_df = df[df["Data_Block"] == target_block].copy()
    mB_filter = (~block_df["C4H6_Bottom_stuck"]) & (block_df["C4H6_Bottom"] > 0.001)
    
    # Calculate last valid
    block_df["C4H6_last_valid"] = block_df["C4H6_Bottom"].copy()
    block_df.loc[~mB_filter, "C4H6_last_valid"] = np.nan
    
    # Calculate leak-free anchor shifted by 1 hour
    block_df["anchor"] = block_df["C4H6_last_valid"].shift(1).ffill(limit=limit)
    
    # Filter down to evaluation set (where target is healthy and anchor is available)
    eval_df = block_df[mB_filter].dropna(subset=["anchor", "C4H6_Bottom"])
    
    y_true = eval_df["C4H6_Bottom"].values
    preds = eval_df["anchor"].values
    
    if len(y_true) < 2:
        return np.nan, np.nan, np.nan, 0.0
        
    r2 = r2_score(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    coverage = len(eval_df) / len(block_df[mB_filter]) * 100
    
    try:
        pear, _ = pearsonr(y_true, preds)
    except:
        pear = np.nan
        
    return r2, mae, pear, coverage

def main():
    features_file = "data/features.parquet"
    if not os.path.exists(features_file):
        features_file = "../data/features.parquet"
        
    df = pd.read_parquet(features_file)
    
    print("================================================================================")
    print("MODEL B ANCHOR ROBUSTNESS INVERSION CHECKS (12h LIMIT)")
    print("================================================================================")
    
    for block in [2, 3, 4]:
        r2, mae, pear, cov = evaluate_block_anchor(df, block, limit=12)
        print(f"Block {block} (Target Mean = {df.loc[(df['Data_Block'] == block) & (~df['C4H6_Bottom_stuck']) & (df['C4H6_Bottom'] > 0.001), 'C4H6_Bottom'].mean():.4f} wt%):")
        print(f"  R² Score:            {r2:.4f}")
        print(f"  MAE:                 {mae:.4f} wt% ({mae*10000:.1f} ppm)")
        print(f"  Pearson Correlation: {pear:+.4f}")
        print(f"  Healthy Coverage:    {cov:.2f}%")
        print("-" * 40)
    print("================================================================================")

if __name__ == "__main__":
    main()
