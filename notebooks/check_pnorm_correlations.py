"""
check_pnorm_correlations.py
===========================
Inspects Pearson correlation of raw temperatures vs pressure-normalized
temperatures with C4H8_Bottom in train (Blocks 1-3) and test (Block 4).
"""

import pandas as pd
import numpy as np

FEATURES_FILE = "data/features.parquet"

def main():
    df = pd.read_parquet(FEATURES_FILE)
    
    train_mask = df["Data_Block"].isin([1, 2, 3])
    test_mask  = df["Data_Block"] == 4
    
    # Exclude stuck rows for correlation analysis
    df_valid = df[~df["C4H8_Bottom_stuck"]].copy()
    
    train_df = df_valid[train_mask]
    test_df  = df_valid[test_mask]
    
    cols_to_check = [
        "Column_Top_Temp", "Control_Tray_Temp", "Column_Bottom_Temp",
        "Column_Top_Temp_Pnorm_k3", "Column_Top_Temp_Pnorm_k5", "Column_Top_Temp_Pnorm_k10",
        "Control_Tray_Temp_Pnorm_k3", "Control_Tray_Temp_Pnorm_k5", "Control_Tray_Temp_Pnorm_k10",
        "Column_Bottom_Temp_Pnorm_k3", "Column_Bottom_Temp_Pnorm_k5", "Column_Bottom_Temp_Pnorm_k10",
        "Column_Top_Temp_Pratio", "Control_Tray_Temp_Pratio", "Column_Bottom_Temp_Pratio",
        "Temp_Gradient", "Temp_Gradient_Pnorm"
    ]
    
    print("=" * 80)
    print("CORRELATION ANALYSIS: TEMPERATURES VS C4H8_BOTTOM")
    print("=" * 80)
    print(f"{'Feature':<30} | {'Train Corr':<10} | {'Test Corr':<10} | {'Sign Match?':<12}")
    print("-" * 80)
    
    for col in cols_to_check:
        if col not in df.columns:
            continue
        corr_train = train_df[col].corr(train_df["C4H8_Bottom"])
        corr_test  = test_df[col].corr(test_df["C4H8_Bottom"])
        
        sign_match = "YES" if np.sign(corr_train) == np.sign(corr_test) else "NO"
        print(f"{col:<30} | {corr_train:<10.4f} | {corr_test:<10.4f} | {sign_match:<12}")
        
    print("=" * 80)

if __name__ == "__main__":
    main()
