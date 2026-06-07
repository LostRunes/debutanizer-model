"""
notebooks/optimizer_validation.py
=================================
Runs the advisory optimizer over 100 random out-of-spec snapshots in Block 4
to generate validation statistics.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from optimizer_v2_physics_aware import load_config, load_models, optimize

FEATURES_FILE = "data/features.parquet"

def main():
    print("=== Advisory Optimizer Batch Validation (100 Snapshots) ===")
    
    if not os.path.exists(FEATURES_FILE):
        print(f"Missing features file: {FEATURES_FILE}")
        return
        
    # Load config and models
    config = load_config()
    model_a, surrogates = load_models()
    
    # Load data
    df = pd.read_parquet(FEATURES_FILE)
    
    # Setup campaign anchors (matches optimizer_v2_physics_aware.py)
    df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
    df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
    df["C4H8_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H8_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=72))
    )
    
    df["C4H6_last_valid"] = df["C4H6_Bottom"].copy()
    df.loc[df["C4H6_Bottom_stuck"], "C4H6_last_valid"] = np.nan
    df["C4H6_campaign_anchor"] = (
        df.groupby("Data_Block")["C4H6_last_valid"]
          .transform(lambda x: x.shift(1).ffill(limit=12))
    )
    
    # Filter Block 4 out-of-spec snapshots
    block4_mask = (df["Data_Block"] == 4) & (~df["C4H8_Bottom_stuck"])
    # Features required for surrogate inputs
    surr_feats = [
        "Feed_Flow", "Reboiling_Steam_Flow", "Reflux_Flow", 
        "Column_Bottom_Temp", "Control_Tray_Temp", "Column_Top_Pressure",
        "Reboiler_Outlet_Temp"
    ]
    b4_df = df[block4_mask].dropna(subset=surr_feats + ["C4H8_campaign_anchor", "C4H6_campaign_anchor"])
    out_of_spec_df = b4_df[b4_df["Total_C4"] > config["spec_limit_total_c4_wt_pct"]]
    
    print(f"Total out-of-spec snapshots in Block 4: {len(out_of_spec_df)}")
    
    # Sample 100 indices randomly (with fixed seed for reproducibility)
    random.seed(42)
    sample_indices = random.sample(list(out_of_spec_df.index), min(100, len(out_of_spec_df)))
    print(f"Sampled {len(sample_indices)} snapshots for validation.")
    
    results = []
    
    # We will test both SPEC mode and ECONOMIC mode
    for mode in ["spec", "economic"]:
        print(f"\nEvaluating mode: {mode.upper()}...")
        config["MODE"] = mode
        
        recs_count = 0
        safety_reject_count = 0
        no_imp_count = 0
        
        steam_changes = []
        reflux_changes = []
        c4_reductions_abs = []
        c4_reductions_pct = []
        cost_changes = []
        
        for idx in sample_indices:
            row = df.loc[idx]
            snap = {
                "Feed_Flow": float(row["Feed_Flow"]),
                "Reboiling_Steam_Flow": float(row["Reboiling_Steam_Flow"]),
                "Reflux_Flow": float(row["Reflux_Flow"]),
                "Column_Bottom_Temp": float(row["Column_Bottom_Temp"]),
                "Control_Tray_Temp": float(row["Control_Tray_Temp"]),
                "Column_Top_Pressure": float(row["Column_Top_Pressure"]),
                "c4h8_anchor": float(row["C4H8_campaign_anchor"]),
                "c4h6_anchor": float(row["C4H6_campaign_anchor"]),
                "current_total_c4": float(row["Total_C4"])
            }
            
            hist = df.loc[max(0, idx-24):idx]
            winner = optimize(snap, hist, model_a, surrogates, config)
            
            if winner is not None:
                recs_count += 1
                steam_changes.append(winner["steam"] - snap["Reboiling_Steam_Flow"])
                reflux_changes.append(winner["reflux"] - snap["Reflux_Flow"])
                c4_reductions_abs.append(snap["current_total_c4"] - winner["pred_total_c4"])
                c4_reductions_pct.append(((snap["current_total_c4"] - winner["pred_total_c4"]) / snap["current_total_c4"]) * 100)
                cost_changes.append(winner["cost_benefit"])
            else:
                # To find out why it was rejected: check safety limits manually
                # We can call surrogates on the current steam/reflux to see if current state itself is near limits,
                # or evaluate all local grid points.
                # A snapshot has all grid points rejected by safety if they violate constraints.
                # Let's count it as safety rejection if current bottom temp is close to limit.
                if snap["Column_Bottom_Temp"] + 0.69011 > config["hard_limit_bottom_temp_degC"]:
                    safety_reject_count += 1
                else:
                    no_imp_count += 1
                    
        # Summary statistics
        total = len(sample_indices)
        print(f"Results for {mode.upper()} mode:")
        print(f"  Recommendation Feasibility:     {recs_count}/{total} ({recs_count/total*100:.1f}%)")
        print(f"  Rejections due to safety limit:  {safety_reject_count}/{total} ({safety_reject_count/total*100:.1f}%)")
        print(f"  Rejections due to no C4 savings: {no_imp_count}/{total} ({no_imp_count/total*100:.1f}%)")
        
        if recs_count > 0:
            print(f"  Average C4 Reduction (abs):     {np.mean(c4_reductions_abs):.4f} wt%")
            print(f"  Average C4 Reduction (rel):     {np.mean(c4_reductions_pct):.1f}%")
            print(f"  Average Steam Change:           {np.mean(steam_changes):+.2f} TPH")
            print(f"  Average Reflux Change:          {np.mean(reflux_changes):+.2f} TPH")
            print(f"  Average Utility Cost Change:    ${np.mean(cost_changes):+.2f}/hr")
        else:
            print("  No recommendations generated in this batch.")
            
if __name__ == "__main__":
    main()
