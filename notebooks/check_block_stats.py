import pandas as pd
import numpy as np

df = pd.read_parquet("data/features.parquet")
mA_filter = ~df["C4H8_Bottom_stuck"]

print("=== C4H8 Target Stats by Data_Block ===")
for block in [1, 2, 3, 4]:
    block_df = df[(df["Data_Block"] == block) & mA_filter]
    target = block_df["C4H8_Bottom"].dropna()
    print(f"Block {block}:")
    print(f"  Count:  {len(target)}")
    print(f"  Mean:   {target.mean():.4f} wt%")
    print(f"  Median: {target.median():.4f} wt%")
    print(f"  Std:    {target.std():.4f} wt%")
    print(f"  Min:    {target.min():.4f} wt%")
    print(f"  Max:    {target.max():.4f} wt%")
    print(f"  % > 0.5 spec: {(target > 0.5).mean()*100:.1f}%")
    print("-" * 30)

print("\n=== Target Mean of Train (Blocks 1-3) vs Test (Block 4) ===")
train_target = df[df["Data_Block"].isin([1, 2, 3]) & mA_filter]["C4H8_Bottom"].dropna()
test_target = df[(df["Data_Block"] == 4) & mA_filter]["C4H8_Bottom"].dropna()
print(f"Train (Blocks 1-3) Mean: {train_target.mean():.4f} wt% | Std: {train_target.std():.4f}")
print(f"Test  (Block 4)    Mean: {test_target.mean():.4f} wt% | Std: {test_target.std():.4f}")
