"""
notebooks/optimizer_v2_physics_aware.py
=======================================
Phase 5.2: Physics-Aware Advisory Optimizer.
Varies Steam Flow and Reflux Flow within safe local move constraints,
predicts process response using surrogate delta models, checks constraints
using safety buffers (MAE), predicts Total C4, and applies a spec-first objective.
"""

import os
import json
import pickle
import argparse
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Paths
ECONOMICS_CONFIG = "configs/economics.json"
SURROGATE_DIR = "models/surrogates"
MODEL_A_PKL = "models/final/model_A_final_v1.pkl"
MODEL_A_JSON = "models/final/model_A_final_v1.json"
FEATURES_FILE = "data/features.parquet"

# Surrogate MAE Safety Buffers (from surrogate_results.json)
MAE_BOTTOM_TEMP = 0.69011
MAE_TRAY_TEMP = 1.96629
MAE_PRESSURE = 0.01361

# Model A Feature Order
MODEL_A_FEATURES = [
    "C4H8_campaign_anchor",
    "Steam_Feed_Ratio",
    "Reflux_Ratio",
    "Reboiling_Steam_Flow_dev24h",
    "Reflux_Flow_dev24h",
    "Column_Bottom_Temp_dev24h",
    "Control_Tray_Temp_dev24h",
    "Column_Top_Pressure_dev24h"
]

# Surrogate Model Feature Order
SURROGATE_FEATURES = [
    "Feed_Flow",
    "Reboiling_Steam_Flow",
    "Reflux_Flow",
    "Steam_Feed_Ratio",
    "Reflux_Ratio",
    "Reboiling_Steam_Flow_dev24h",
    "Reflux_Flow_dev24h",
    "Column_Bottom_Temp_dev24h",
    "Control_Tray_Temp_dev24h",
    "Column_Top_Pressure_dev24h",
    "Reboiling_Steam_Flow_lag1",
    "Reflux_Flow_lag1",
    "Column_Bottom_Temp_lag1",
    "Control_Tray_Temp_lag1",
    "Column_Top_Pressure_lag1"
]

def load_config():
    if not os.path.exists(ECONOMICS_CONFIG):
        # Fallback to defaults if not found
        return {
            "MODE": "spec",
            "c4_value_per_wt_pct": 100.0,
            "steam_cost_per_tph": 5.0,
            "reflux_cost_per_tph": 1.0,
            "spec_limit_total_c4_wt_pct": 0.50,
            "hard_limit_bottom_temp_degC": 115.0,
            "hard_limit_top_pressure_bar": 5.0,
            "steam_min_tph": 18.0,
            "steam_max_tph": 24.4,
            "reflux_min_tph": 80.0,
            "reflux_max_tph": 103.9,
            "max_steam_change_tph": 2.0,
            "max_reflux_change_tph": 10.0
        }
    with open(ECONOMICS_CONFIG, "r") as f:
        return json.load(f)

def load_models():
    # Load Model A
    model_a = None
    if os.path.exists(MODEL_A_PKL):
        with open(MODEL_A_PKL, "rb") as f:
            model_a = pickle.load(f)
    elif os.path.exists(MODEL_A_JSON):
        model_a = XGBRegressor()
        model_a.load_model(MODEL_A_JSON)
    else:
        raise FileNotFoundError(f"Missing Model A final weights: {MODEL_A_PKL} or {MODEL_A_JSON}")
        
    # Load Surrogate Models (T1, T2, T3)
    surrogates = {}
    for name in ["bottom_temp", "tray_temp", "pressure"]:
        path = os.path.join(SURROGATE_DIR, f"{name}_t1_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing surrogate model: {path}")
        with open(path, "rb") as f:
            surrogates[name] = pickle.load(f)
            
    return model_a, surrogates

def get_safety_confidence(pred_bot_temp, pred_pressure, config):
    """
    Computes Safety Confidence level based on distance from hard safety limits.
    """
    bot_temp_limit = config["hard_limit_bottom_temp_degC"]
    pressure_limit = config["hard_limit_top_pressure_bar"]
    
    # Distance including MAE safety buffers
    bot_temp_dist = bot_temp_limit - (pred_bot_temp + MAE_BOTTOM_TEMP)
    pressure_dist = pressure_limit - (pred_pressure + MAE_PRESSURE)
    
    if bot_temp_dist >= 3.0 and pressure_dist >= 0.10:
        return "HIGH"
    elif bot_temp_dist < 1.0 or pressure_dist < 0.03:
        return "LOW"
    else:
        return "MEDIUM"

def optimize(snapshot, history_24h, model_a, surrogates, config):
    """
    Evaluates candidates in local grid search and returns best recommendation.
    """
    current_steam = snapshot["Reboiling_Steam_Flow"]
    current_reflux = snapshot["Reflux_Flow"]
    current_feed = snapshot["Feed_Flow"]
    
    # Define local search bounds
    steam_min = max(config["steam_min_tph"], current_steam - config["max_steam_change_tph"])
    steam_max = min(config["steam_max_tph"], current_steam + config["max_steam_change_tph"])
    reflux_min = max(config["reflux_min_tph"], current_reflux - config["max_reflux_change_tph"])
    reflux_max = min(config["reflux_max_tph"], current_reflux + config["max_reflux_change_tph"])
    
    # Search grid
    steam_grid = np.arange(steam_min, steam_max + 0.01, 0.2)
    reflux_grid = np.arange(reflux_min, reflux_max + 0.01, 1.0)
    
    # 24h rolling means from history
    mean_steam_24h = history_24h["Reboiling_Steam_Flow"].mean()
    mean_reflux_24h = history_24h["Reflux_Flow"].mean()
    mean_bot_temp_24h = history_24h["Column_Bottom_Temp"].mean()
    mean_tray_temp_24h = history_24h["Control_Tray_Temp"].mean()
    mean_pressure_24h = history_24h["Column_Top_Pressure"].mean()
    
    # Lags from history (the last row represents index -1, which is lag 0 relative to t+1 prediction)
    last_row = history_24h.iloc[-1]
    
    candidates = []
    
    for steam_cand in steam_grid:
        for reflux_cand in reflux_grid:
            # Step A: Build surrogate features to predict delta
            steam_feed_ratio = steam_cand / current_feed
            reflux_ratio = reflux_cand / current_feed
            
            surr_feat = pd.DataFrame([{
                "Feed_Flow": current_feed,
                "Reboiling_Steam_Flow": steam_cand,
                "Reflux_Flow": reflux_cand,
                "Steam_Feed_Ratio": steam_feed_ratio,
                "Reflux_Ratio": reflux_ratio,
                "Reboiling_Steam_Flow_dev24h": steam_cand - mean_steam_24h,
                "Reflux_Flow_dev24h": reflux_cand - mean_reflux_24h,
                "Column_Bottom_Temp_dev24h": snapshot["Column_Bottom_Temp"] - mean_bot_temp_24h,
                "Control_Tray_Temp_dev24h": snapshot["Control_Tray_Temp"] - mean_tray_temp_24h,
                "Column_Top_Pressure_dev24h": snapshot["Column_Top_Pressure"] - mean_pressure_24h,
                "Reboiling_Steam_Flow_lag1": last_row["Reboiling_Steam_Flow"],
                "Reflux_Flow_lag1": last_row["Reflux_Flow"],
                "Column_Bottom_Temp_lag1": last_row["Column_Bottom_Temp"],
                "Control_Tray_Temp_lag1": last_row["Control_Tray_Temp"],
                "Column_Top_Pressure_lag1": last_row["Column_Top_Pressure"]
            }], columns=SURROGATE_FEATURES)
            
            # Predict deltas
            delta_bot = surrogates["bottom_temp"].predict(surr_feat)[0]
            delta_tray = surrogates["tray_temp"].predict(surr_feat)[0]
            delta_pres = surrogates["pressure"].predict(surr_feat)[0]
            
            # Reconstruct absolute values at t+1
            pred_bot = snapshot["Column_Bottom_Temp"] + delta_bot
            pred_tray = snapshot["Control_Tray_Temp"] + delta_tray
            pred_pres = snapshot["Column_Top_Pressure"] + delta_pres
            
            # Step B: Safety Checks with MAE buffers
            if pred_bot + MAE_BOTTOM_TEMP > config["hard_limit_bottom_temp_degC"]:
                continue
            if pred_pres + MAE_PRESSURE > config["hard_limit_top_pressure_bar"]:
                continue
                
            # Step C: Predict C4H8 using Model A
            ma_feat = pd.DataFrame([{
                "C4H8_campaign_anchor": snapshot["c4h8_anchor"],
                "Steam_Feed_Ratio": steam_feed_ratio,
                "Reflux_Ratio": reflux_ratio,
                "Reboiling_Steam_Flow_dev24h": steam_cand - mean_steam_24h,
                "Reflux_Flow_dev24h": reflux_cand - mean_reflux_24h,
                "Column_Bottom_Temp_dev24h": pred_bot - mean_bot_temp_24h,
                "Control_Tray_Temp_dev24h": pred_tray - mean_tray_temp_24h,
                "Column_Top_Pressure_dev24h": pred_pres - mean_pressure_24h
            }], columns=MODEL_A_FEATURES)
            
            pred_c4h8 = float(model_a.predict(ma_feat)[0])
            pred_c4h6 = snapshot["c4h6_anchor"] # Model B remains at latest anchor
            pred_total = pred_c4h8 + pred_c4h6
            
            # Step D: Only accept improvement
            if pred_total >= snapshot["current_total_c4"]:
                continue
                
            # Step E: Cost Penalty (if Mode is economic)
            delta_s = steam_cand - current_steam
            delta_r = reflux_cand - current_reflux
            cost_benefit = (config["steam_cost_per_tph"] * delta_s) + (config["reflux_cost_per_tph"] * delta_r)
            
            candidates.append({
                "steam": steam_cand,
                "reflux": reflux_cand,
                "pred_bot_temp": pred_bot,
                "pred_tray_temp": pred_tray,
                "pred_pressure": pred_pres,
                "pred_c4h8": pred_c4h8,
                "pred_c4h6": pred_c4h6,
                "pred_total_c4": pred_total,
                "cost_benefit": cost_benefit
            })
            
    if not candidates:
        return None # No feasible recommendations found
        
    df_cands = pd.DataFrame(candidates)
    
    # Two-Stage Selection
    spec_limit = config["spec_limit_total_c4_wt_pct"]
    spec_compliant = df_cands[df_cands["pred_total_c4"] < spec_limit]
    
    if not spec_compliant.empty:
        # We can meet the spec!
        if config["MODE"] == "economic":
            # Minimize utility costs among compliant candidates
            winner = spec_compliant.sort_values(by="cost_benefit").iloc[0].to_dict()
        else:
            # Spec mode: minimize raw Total C4 slippage
            winner = spec_compliant.sort_values(by="pred_total_c4").iloc[0].to_dict()
    else:
        # Cannot meet spec in one step; minimize C4 regardless of mode
        winner = df_cands.sort_values(by="pred_total_c4").iloc[0].to_dict()
        
    return winner

def print_recommendation(snapshot, winner, config):
    print("\n=======================================================")
    print("        DEBUTANIZER ADVISORY OPTIMIZER v2.0")
    print("=======================================================")
    print("CURRENT CONDITIONS:")
    print(f"  Steam Flow:       {snapshot['Reboiling_Steam_Flow']:.1f} TPH")
    print(f"  Reflux Flow:      {snapshot['Reflux_Flow']:.1f} TPH")
    print(f"  Feed Flow:        {snapshot['Feed_Flow']:.1f} TPH (fixed)")
    print(f"  Bottom Temp:      {snapshot['Column_Bottom_Temp']:.2f} C")
    print(f"  Tray Temp:        {snapshot['Control_Tray_Temp']:.2f} C")
    print(f"  Pressure:         {snapshot['Column_Top_Pressure']:.3f} bar")
    print(f"  C4H8 Anchor:      {snapshot['c4h8_anchor']:.4f} wt%")
    print(f"  C4H6 Anchor:      {snapshot['c4h6_anchor']:.4f} wt%")
    print(f"  Total C4 Slippage:{snapshot['current_total_c4']:.4f} wt%")
    
    if snapshot['current_total_c4'] > config["spec_limit_total_c4_wt_pct"]:
         print("  --> STATUS: [WARNING] OUT OF SPECIFICATION (> 0.50 wt%)")
    else:
         print("  --> STATUS: [OK] IN SPECIFICATION")
         
    if winner is None:
        print("\nRECOMMENDATION:")
        print("  [ERROR] No feasible moves found that reduce C4 without violating safety bounds.")
        print("=======================================================")
        return
        
    # Reconstruct details
    confidence = get_safety_confidence(winner["pred_bot_temp"], winner["pred_pressure"], config)
    
    print("\nRECOMMENDED SET-POINTS:")
    steam_change = winner["steam"] - snapshot["Reboiling_Steam_Flow"]
    reflux_change = winner["reflux"] - snapshot["Reflux_Flow"]
    print(f"  Reboiling Steam Flow: {snapshot['Reboiling_Steam_Flow']:.1f} -> {winner['steam']:.1f} TPH  (Delta {steam_change:+.1f} TPH)")
    print(f"  Reflux Flow:          {snapshot['Reflux_Flow']:.1f} -> {winner['reflux']:.1f} TPH  (Delta {reflux_change:+.1f} TPH)")
    
    print("\nPREDICTED PROCESS RESPONSE (T+1):")
    print(f"  Bottom Temp:      {winner['pred_bot_temp']:.2f} +/- {MAE_BOTTOM_TEMP:.2f} C  (Limit: {config['hard_limit_bottom_temp_degC']} C)")
    print(f"  Top Pressure:     {winner['pred_pressure']:.3f} +/- {MAE_PRESSURE:.3f} bar (Limit: {config['hard_limit_top_pressure_bar']} bar)")
    print(f"  Tray Temp:        {winner['pred_tray_temp']:.2f} +/- {MAE_TRAY_TEMP:.2f} C")
    
    print("\nPREDICTED COMPOSITION:")
    print(f"  C4H8 (Model A):   {winner['pred_c4h8']:.4f} wt%")
    print(f"  C4H6 (Model B):   {winner['pred_c4h6']:.4f} wt% *")
    print(f"  Total C4:         {winner['pred_total_c4']:.4f} wt%  (Expected reduction: {((snapshot['current_total_c4'] - winner['pred_total_c4']) / snapshot['current_total_c4']) * 100:.1f}%)")
    
    print(f"\nSAFETY CONFIDENCE: {confidence}")
    print("\nUTILITY COST ANALYSIS:")
    if winner["cost_benefit"] > 0:
        print(f"  Utility cost change: +${winner['cost_benefit']:.2f}/hr (cost increase)")
    else:
        print(f"  Utility cost change: -${abs(winner['cost_benefit']):.2f}/hr (cost savings)")
        
    print("\nDISCLAIMER:")
    print("  * Optimizer assumes C4H6 remains at its latest analyzer-estimated value because")
    print("    the validated Model B architecture contains no manipulable-variable response model.")
    print("=======================================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=str, help="JSON string representing current conditions snapshot")
    parser.add_argument("--mode", type=str, choices=["spec", "economic"], help="Override optimization mode")
    args = parser.parse_args()
    
    config = load_config()
    if args.mode:
        config["MODE"] = args.mode
        
    model_a, surrogates = load_models()
    
    if args.snapshot:
        snap = json.loads(args.snapshot)
        # Create dummy history for CLI demo testing if history isn't supplied
        dummy_hist = pd.DataFrame([snap] * 24)
        winner = optimize(snap, dummy_hist, model_a, surrogates, config)
        print_recommendation(snap, winner, config)
    else:
        # Run validation tests on historical Block 4 snapshots
        print("No CLI snapshot provided. Running validation check on Block 4 out-of-spec snapshots...")
        if not os.path.exists(FEATURES_FILE):
            print(f"Missing features file for validation check: {FEATURES_FILE}")
            return
            
        df = pd.read_parquet(FEATURES_FILE)
        
        # Define campaign anchor limit (72 hours like Model A training config)
        df["C4H8_last_valid"] = df["C4H8_Bottom"].copy()
        df.loc[df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
        df["C4H8_campaign_anchor"] = (
            df.groupby("Data_Block")["C4H8_last_valid"]
              .transform(lambda x: x.shift(1).ffill(limit=72))
        )
        
        # Similar for C4H6 anchor (12 hour limit)
        df["C4H6_last_valid"] = df["C4H6_Bottom"].copy()
        df.loc[df["C4H6_Bottom_stuck"], "C4H6_last_valid"] = np.nan
        df["C4H6_campaign_anchor"] = (
            df.groupby("Data_Block")["C4H6_last_valid"]
              .transform(lambda x: x.shift(1).ffill(limit=12))
        )
        
        # Filter for Block 4 out-of-spec points with valid anchors
        block4_mask = (df["Data_Block"] == 4) & (~df["C4H8_Bottom_stuck"])
        b4_df = df[block4_mask].dropna(subset=SURROGATE_FEATURES + ["C4H8_campaign_anchor", "C4H6_campaign_anchor"])
        
        # Find points where Total C4 is out-of-spec (> 0.50)
        out_of_spec_df = b4_df[b4_df["Total_C4"] > 0.50]
        print(f"Found {len(out_of_spec_df)} out-of-spec snapshots in Block 4.")
        
        if out_of_spec_df.empty:
            print("No out-of-spec snapshots found in Block 4 for validation.")
            return
            
        # Select 3 representative snapshots
        indices = [out_of_spec_df.index[0], out_of_spec_df.index[len(out_of_spec_df)//2], out_of_spec_df.index[-1]]
        
        for idx in indices:
            # Construct snapshot dict
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
            
            # Extract 24h history preceding this snapshot
            hist_start_idx = max(0, idx - 24)
            hist = df.loc[hist_start_idx:idx]
            
            print(f"\n--- Running Optimization on Snapshot at Index {idx} (DateTime: {row['DateTime']}) ---")
            # Test Spec Mode
            config["MODE"] = "spec"
            winner_spec = optimize(snap, hist, model_a, surrogates, config)
            print("SPEC MODE OPTIMIZATION:")
            print_recommendation(snap, winner_spec, config)
            
            # Test Economic Mode
            config["MODE"] = "economic"
            winner_econ = optimize(snap, hist, model_a, surrogates, config)
            print("ECONOMIC MODE OPTIMIZATION:")
            print_recommendation(snap, winner_econ, config)

if __name__ == "__main__":
    main()
