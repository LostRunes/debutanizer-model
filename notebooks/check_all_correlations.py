"""
check_all_correlations.py
=========================
Inspects Pearson correlation of all core process variables and ratios
with C4H8_Bottom in train (Blocks 1-3) and test (Block 4).
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
    
    train_df = df_valid[train_mask & (df_valid["Reboiler_Outlet_Temp"] >= 50)]
    test_df  = df_valid[test_mask & (df_valid["Reboiler_Outlet_Temp"] >= 50)]
    
    cols_to_check = [
        "Feed_Flow", "Reboiler_Outlet_Temp", "Column_Top_Temp",
        "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Top_Pressure",
        "Column_Bottom_Temp", "Control_Tray_Temp",
        "Reflux_Ratio", "Steam_Feed_Ratio", "Temp_Gradient", "Reboiler_Delta"
    ]
    
    print("=" * 80)
    print("ALL CORE VARIABLES VS C4H8_BOTTOM CORRELATION")
    print("=" * 80)
    print(f"{'Feature':<25} | {'Train Corr':<10} | {'Test Corr':<10} | {'Sign Match?':<12}")
    print("-" * 80)
    
    for col in cols_to_check:
        corr_train = train_df[col].corr(train_df["C4H8_Bottom"])
        corr_test  = test_df[col].corr(test_df["C4H8_Bottom"])
        
        sign_match = "YES" if np.sign(corr_train) == np.sign(corr_test) else "NO"
        print(f"{col:<25} | {corr_train:<10.4f} | {corr_test:<10.4f} | {sign_match:<12}")
        
    print("=" * 80)

if __name__ == "__main__":
    main()
