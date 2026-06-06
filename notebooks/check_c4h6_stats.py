import pandas as pd
import numpy as np

df = pd.read_parquet("data/features.parquet")

print("================================================================================")
print("C4H6 TARGET STATISTICS BY BLOCK")
print("================================================================================")

for block in [1, 2, 3, 4]:
    block_df = df[df["Data_Block"] == block]
    print(f"\n--- Data Block {block} (Total rows: {len(block_df)}) ---")
    
    # Analyzer stuck status
    stuck_pct = block_df["C4H6_Bottom_stuck"].mean() * 100
    print(f"Stuck readings: {stuck_pct:.2f}% of rows")
    
    # Stuck values
    stuck_vals = block_df.loc[block_df["C4H6_Bottom_stuck"], "C4H6_Bottom"]
    print(f"Stuck values statistics: mean={stuck_vals.mean():.4f}, median={stuck_vals.median():.4f}, min={stuck_vals.min():.4f}, max={stuck_vals.max():.4f}")
    
    # Non-stuck values
    non_stuck_vals = block_df.loc[~block_df["C4H6_Bottom_stuck"], "C4H6_Bottom"]
    print(f"Non-stuck values statistics (Total: {len(non_stuck_vals)}):")
    print(f"  Mean:   {non_stuck_vals.mean():.4f} wt%")
    print(f"  Median: {non_stuck_vals.median():.4f} wt%")
    print(f"  Std:    {non_stuck_vals.std():.4f} wt%")
    print(f"  Min:    {non_stuck_vals.min():.4f} wt%")
    print(f"  Max:    {non_stuck_vals.max():.4f} wt%")
    
    # Exact zeros or near-zeros (<= 0.001) in non-stuck
    zero_cnt = (non_stuck_vals <= 0.001).sum()
    zero_pct = (non_stuck_vals <= 0.001).mean() * 100
    print(f"  Non-stuck readings <= 0.001: {zero_cnt} ({zero_pct:.2f}%)")
    
    # Filtered valid rows for Model B (> 0.001 and non-stuck)
    valid_vals = non_stuck_vals[non_stuck_vals > 0.001]
    print(f"  Filtered training/eval rows (> 0.001) (Total: {len(valid_vals)}):")
    print(f"    Mean:   {valid_vals.mean():.4f} wt%")
    print(f"    Median: {valid_vals.median():.4f} wt%")
    print(f"    Std:    {valid_vals.std():.4f} wt%")
    print(f"    Min:    {valid_vals.min():.4f} wt%")
    print(f"    Max:    {valid_vals.max():.4f} wt%")
    print(f"    % > 0.1 spec: {(valid_vals > 0.1).mean()*100:.2f}%")
print("================================================================================")
