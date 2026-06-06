import os
import pandas as pd
import numpy as np

def main():
    features_file = "data/features.parquet"
    if not os.path.exists(features_file):
        features_file = "../data/features.parquet"
        
    df = pd.read_parquet(features_file)
    
    # 1. Define last valid C4H6 (stuck and <= 0.001 excluded)
    df["C4H6_last_valid"] = df["C4H6_Bottom"].copy()
    df.loc[df["C4H6_Bottom_stuck"] | (df["C4H6_Bottom"] <= 0.001), "C4H6_last_valid"] = np.nan
    
    # Let's shift by 1 hour (as in production, we only have data up to t-1)
    df["C4H6_shifted_valid"] = df.groupby("Data_Block")["C4H6_last_valid"].shift(1)
    
    # 2. Calculate Age of last valid reading in hours
    # We map valid indices and forward fill them within each block
    df["valid_idx"] = np.nan
    df.loc[df["C4H6_shifted_valid"].notna(), "valid_idx"] = df.loc[df["C4H6_shifted_valid"].notna()].index
    
    df["last_valid_idx"] = df.groupby("Data_Block")["valid_idx"].ffill()
    df["anchor_age"] = df.index - df["last_valid_idx"]
    
    # 3. Apply limits to create anchors
    df["anchor_12h"] = df["C4H6_shifted_valid"].groupby(df["Data_Block"]).ffill(limit=12)
    df["anchor_72h"] = df["C4H6_shifted_valid"].groupby(df["Data_Block"]).ffill(limit=72)
    
    # 4. Filter for Block 4
    b4_df = df[df["Data_Block"] == 4].copy()
    
    # Filter for healthy target evaluation rows
    mB_filter = (~b4_df["C4H6_Bottom_stuck"]) & (b4_df["C4H6_Bottom"] > 0.001)
    b4_healthy = b4_df[mB_filter]
    
    print("================================================================================")
    print("MODEL B (C4H6) ANCHOR AUDIT ON BLOCK 4")
    print("================================================================================")
    
    for name, df_subset, label in [("ALL BLOCK 4 ROWS", b4_df, "Overall"), ("HEALTHY ANALYZER ROWS ONLY", b4_healthy, "Healthy Target")]:
        print(f"\n--- {name} (Count: {len(df_subset)}) ---")
        
        # 12h Limit
        not_null_12 = df_subset["anchor_12h"].notna()
        coverage_12 = not_null_12.mean() * 100
        ages_12 = df_subset.loc[not_null_12, "anchor_age"]
        mean_age_12 = ages_12.mean() if len(ages_12) > 0 else np.nan
        max_age_12 = ages_12.max() if len(ages_12) > 0 else np.nan
        
        # 72h Limit
        not_null_72 = df_subset["anchor_72h"].notna()
        coverage_72 = not_null_72.mean() * 100
        ages_72 = df_subset.loc[not_null_72, "anchor_age"]
        mean_age_72 = ages_72.mean() if len(ages_72) > 0 else np.nan
        max_age_72 = ages_72.max() if len(ages_72) > 0 else np.nan
        
        print(f"C4H6 Campaign Anchor (12h Limit):")
        print(f"  Coverage:  {coverage_12:.2f}%")
        print(f"  Mean Age:  {mean_age_12:.2f} hours")
        print(f"  Max Age:   {max_age_12:.2f} hours")
        print(f"C4H6 Campaign Anchor (72h Limit):")
        print(f"  Coverage:  {coverage_72:.2f}%")
        print(f"  Mean Age:  {mean_age_72:.2f} hours")
        print(f"  Max Age:   {max_age_72:.2f} hours")
        
    print("================================================================================")

if __name__ == "__main__":
    main()
