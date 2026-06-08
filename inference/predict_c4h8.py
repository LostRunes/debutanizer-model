"""
inference/predict_c4h8.py
==========================
Production inference script for Model A (C4H8).

Inputs:
- process_history: pd.DataFrame with at least 24 rows containing:
  ['Feed_Flow', 'Reboiling_Steam_Flow', 'Reflux_Flow', 'Column_Bottom_Temp', 'Control_Tray_Temp', 'Column_Top_Pressure']
- latest_valid_analyzer: dict containing:
  - 'value': float or None (the most recent C4H8_Bottom analyzer reading)
  - 'hours_ago': int (the number of hours since that reading was taken)
- previous_predictions: list of floats or None (history of recent model predictions for fallback)

Outputs:
- dict containing:
  - 'predicted_c4h8': float (the final predicted composition in wt%)
  - 'model_used': str ('Model A' or 'Fallback (Last Prediction)' or 'Fallback (Rolling Mean)' or 'Fallback (Default)')
  - 'anchor_available': bool (whether the campaign anchor was available)
  - 'anchor_value': float or None (the value used as the anchor)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Default paths
MODEL_PKL_PATH = "models/final/model_A_final_v1.pkl"
MODEL_JSON_PATH = "models/final/model_A_final_v1.json"
CONFIG_PATH = "configs/model_A_features.json"

# Production features list in correct order
FEATURE_ORDER = [
    "C4H8_campaign_anchor",
    "Steam_Feed_Ratio",
    "Reflux_Ratio",
    "Reboiling_Steam_Flow_dev24h",
    "Reflux_Flow_dev24h",
    "Column_Bottom_Temp_dev24h",
    "Control_Tray_Temp_dev24h",
    "Column_Top_Pressure_dev24h"
]

def load_model(pkl_path=MODEL_PKL_PATH, json_path=MODEL_JSON_PATH):
    """
    Loads the frozen Model A XGBoost model.
    Tries pickle first, then falls back to XGBoost JSON format.
    """
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        return model
    elif os.path.exists(json_path):
        model = XGBRegressor()
        model.load_model(json_path)
        return model
    else:
        raise FileNotFoundError(f"No model found at {pkl_path} or {json_path}. Please freeze the model first.")

def predict_c4h8(process_history: pd.DataFrame, 
                 latest_valid_analyzer: dict, 
                 previous_predictions: list = None,
                 pkl_path=MODEL_PKL_PATH,
                 json_path=MODEL_JSON_PATH) -> dict:
    """
    Computes the feature vector at the current timestep t and predicts C4H8 wt%.
    If the analyzer anchor is unavailable (stuck > 72 hours or None), uses fallback logic.
    """
    # 1. Validate inputs
    required_cols = [
        'Feed_Flow', 'Reboiling_Steam_Flow', 'Reflux_Flow', 
        'Column_Bottom_Temp', 'Control_Tray_Temp', 'Column_Top_Pressure'
    ]
    for col in required_cols:
        if col not in process_history.columns:
            raise ValueError(f"Missing required process column: {col}")
            
    if len(process_history) < 24:
        raise ValueError(f"process_history must contain at least 24 rows, got {len(process_history)}")

    # Sort history to ensure the last row is the current timestep t
    process_history = process_history.tail(24).copy()
    
    # 2. Check anchor availability (Limit = 72 hours as per Subset 7)
    anchor_available = False
    anchor_value = None
    
    if latest_valid_analyzer is not None:
        val = latest_valid_analyzer.get("value")
        hours_ago = latest_valid_analyzer.get("hours_ago")
        if val is not None and not (isinstance(val, float) and np.isnan(val)) and hours_ago is not None and hours_ago <= 72 and hours_ago >= 1:
            anchor_available = True
            anchor_value = float(val)

    # 3. Process prediction logic
    if anchor_available:
        try:
            # Load Model
            model = load_model(pkl_path, json_path)
            
            # Extract current values (at t, the last row in history)
            current_vals = process_history.iloc[-1]
            
            # Compute Ratios
            steam_feed_ratio = current_vals["Reboiling_Steam_Flow"] / current_vals["Feed_Flow"]
            reflux_ratio = current_vals["Reflux_Flow"] / current_vals["Feed_Flow"]
            
            # Compute 24h rolling deviations (current val minus 24h mean)
            dev_steam = current_vals["Reboiling_Steam_Flow"] - process_history["Reboiling_Steam_Flow"].mean()
            dev_reflux = current_vals["Reflux_Flow"] - process_history["Reflux_Flow"].mean()
            dev_bot_temp = current_vals["Column_Bottom_Temp"] - process_history["Column_Bottom_Temp"].mean()
            dev_tray_temp = current_vals["Control_Tray_Temp"] - process_history["Control_Tray_Temp"].mean()
            dev_pressure = current_vals["Column_Top_Pressure"] - process_history["Column_Top_Pressure"].mean()
            
            # Build feature vector
            feature_vector = pd.DataFrame([{
                "C4H8_campaign_anchor": anchor_value,
                "Steam_Feed_Ratio": steam_feed_ratio,
                "Reflux_Ratio": reflux_ratio,
                "Reboiling_Steam_Flow_dev24h": dev_steam,
                "Reflux_Flow_dev24h": dev_reflux,
                "Column_Bottom_Temp_dev24h": dev_bot_temp,
                "Control_Tray_Temp_dev24h": dev_tray_temp,
                "Column_Top_Pressure_dev24h": dev_pressure
            }], columns=FEATURE_ORDER)
            
            # Predict
            pred = float(model.predict(feature_vector)[0])
            
            return {
                "predicted_c4h8": pred,
                "model_used": "Model A",
                "anchor_available": True,
                "anchor_value": anchor_value,
                "prediction_health": "GREEN",
                "fallback_reason": "None"
            }
            
        except Exception as e:
            # If Model A fails to load or execute, fail gracefully to fallback
            print(f"Warning: Model A execution failed: {e}. Falling back...")
            
    # 4. Fallback Logic (if anchor is unavailable or model execution failed)
    # Check for hard timeout of 168 hours (7 days) on the analyzer
    hard_timeout = False
    analyzer_hours = None
    if latest_valid_analyzer is not None:
        analyzer_hours = latest_valid_analyzer.get("hours_ago")
        if analyzer_hours is not None and analyzer_hours > 168:
            hard_timeout = True

    # Check for stale analyzer warning (> 72 hours)
    analyzer_stale = False
    if analyzer_hours is not None and analyzer_hours > 72:
        analyzer_stale = True

    if previous_predictions and len(previous_predictions) > 0 and not hard_timeout:
        valid_preds = [p for p in previous_predictions[-24:] if p is not None]
        if len(valid_preds) >= 6:
            rolling_mean_pred = float(np.mean(valid_preds))
            reason = "Analyzer stale >72h" if analyzer_stale else "Model execution fallback"
            return {
                "predicted_c4h8": rolling_mean_pred,
                "model_used": "Fallback (Rolling Mean)",
                "anchor_available": False,
                "anchor_value": None,
                "prediction_health": "YELLOW",
                "fallback_reason": reason
            }
        else:
            reason = "Too few predictions available (<6h)"
            return {
                "predicted_c4h8": 0.480,
                "model_used": "Fallback (Default)",
                "anchor_available": False,
                "anchor_value": None,
                "prediction_health": "RED",
                "fallback_reason": reason
            }
    else:
        # Determine specific reason for default fallback
        if hard_timeout:
            reason = "Analyzer offline >168h"
        elif not previous_predictions or len(previous_predictions) == 0:
            reason = "No previous predictions available"
        else:
            reason = "Model execution failure"
            
        return {
            "predicted_c4h8": 0.480,
            "model_used": "Fallback (Default)",
            "anchor_available": False,
            "anchor_value": None,
            "prediction_health": "RED",
            "fallback_reason": reason
        }

if __name__ == "__main__":
    print("=== TESTING predict_c4h8.py ===")
    
    # Generate dummy process history (24 hours)
    dummy_history = pd.DataFrame({
        "Feed_Flow":             np.random.normal(80, 5, 24),
        "Reboiling_Steam_Flow":  np.random.normal(21, 1, 24),
        "Reflux_Flow":           np.random.normal(90, 3, 24),
        "Column_Bottom_Temp":    np.random.normal(107, 2, 24),
        "Control_Tray_Temp":     np.random.normal(72, 5, 24),
        "Column_Top_Pressure":   np.random.normal(4.05, 0.1, 24)
    })
    
    # Test Scenario 1: Anchor Available
    analyzer_ok = {"value": 0.45, "hours_ago": 4}
    res_ok = predict_c4h8(dummy_history, analyzer_ok)
    print(f"Scenario 1 (Anchor Available):")
    print(f"  Result:      {res_ok['predicted_c4h8']:.4f} wt%")
    print(f"  Model Used:  {res_ok['model_used']}")
    print(f"  Anchor OK:   {res_ok['anchor_available']}")
    print(f"  Anchor Val:  {res_ok['anchor_value']}")
    
    # Test Scenario 2: Anchor Stale (> 72 hours) - Fallback with previous predictions
    analyzer_stale = {"value": 0.45, "hours_ago": 73}
    prev_preds = [0.44, 0.46, 0.45, 0.47, 0.48, 0.45]
    res_stale = predict_c4h8(dummy_history, analyzer_stale, previous_predictions=prev_preds)
    print(f"\nScenario 2 (Anchor Stale, Fallback with predictions):")
    print(f"  Result:      {res_stale['predicted_c4h8']:.4f} wt%")
    print(f"  Model Used:  {res_stale['model_used']}")
    print(f"  Anchor OK:   {res_stale['anchor_available']}")
    
    # Test Scenario 3: No Analyzer and No Predictions
    res_none = predict_c4h8(dummy_history, None, previous_predictions=None)
    print(f"\nScenario 3 (No Data, Default Fallback):")
    print(f"  Result:      {res_none['predicted_c4h8']:.4f} wt%")
    print(f"  Model Used:  {res_none['model_used']}")
    print(f"  Anchor OK:   {res_none['anchor_available']}")
