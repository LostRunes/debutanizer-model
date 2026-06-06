import pandas as pd
import numpy as np

def main():
    features_file = "data/features.parquet"
    if not os.path.exists(features_file):
        # Fallback path if run from a different directory level
        features_file = "../data/features.parquet"
        
    df = pd.read_parquet(features_file)
    
    print("================================================================================")
    print("MODEL B (C4H6_Bottom) TARGET AUDIT BY BLOCK (HEALTHY ANALYZER ROWS > 0.001)")
    print("================================================================================")
    
    # Filter for valid Model B training rows
    mB_filter = (~df["C4H6_Bottom_stuck"]) & (df["C4H6_Bottom"] > 0.001)
    
    audit_data = []
    
    for block in [1, 2, 3, 4]:
        block_df = df[(df["Data_Block"] == block) & mB_filter]
        target = block_df["C4H6_Bottom"]
        
        if len(target) > 0:
            stats = {
                "Block": block,
                "Mean": target.mean(),
                "Median": target.median(),
                "Std": target.std(),
                "P95": target.quantile(0.95),
                "P99": target.quantile(0.99),
                "Max": target.max(),
                "Count": len(target)
            }
        else:
            stats = {
                "Block": block,
                "Mean": np.nan,
                "Median": np.nan,
                "Std": np.nan,
                "P95": np.nan,
                "P99": np.nan,
                "Max": np.nan,
                "Count": 0
            }
        audit_data.append(stats)
        
    audit_df = pd.DataFrame(audit_data)
    
    # Print nice table
    print(audit_df.to_string(index=False, formatters={
        "Mean": "{:.6f}".format,
        "Median": "{:.6f}".format,
        "Std": "{:.6f}".format,
        "P95": "{:.6f}".format,
        "P99": "{:.6f}".format,
        "Max": "{:.6f}".format,
        "Count": "{:d}".format
    }))
    print("================================================================================")

if __name__ == "__main__":
    import os
    main()
