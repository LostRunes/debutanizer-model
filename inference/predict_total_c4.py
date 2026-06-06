"""
inference/predict_total_c4.py
==============================
Unified prediction script for the Debutanizer Column.
Combines:
- Model A: ML-based prediction for C4H8 (Butene) composition.
- Model B: Deterministic anchor-based prediction for C4H6 (Butadiene) composition.
Outputs the combined Total C4 (C4H8 + C4H6) slippage.

Inputs:
- process_history: pd.DataFrame with at least 24 rows containing the raw process variables.
- latest_valid_c4h8: dict with 'value' and 'hours_ago' for Model A analyzer status.
- latest_valid_c4h6: dict with 'value' and 'hours_ago' for Model B analyzer status.
- previous_c4h8_preds: list of floats for Model A prediction fallback (optional).
- previous_c4h6_preds: list of floats for Model B prediction fallback (optional).

Outputs:
- dict containing final predictions, individual components, and fallback status metrics.
"""

import os
import pandas as pd
import numpy as np

# Import predictions from local modules
from predict_c4h8 import predict_c4h8
from predict_c4h6 import predict_c4h6

def predict_total_c4(process_history: pd.DataFrame,
                     latest_valid_c4h8: dict,
                     latest_valid_c4h6: dict,
                     previous_c4h8_preds: list = None,
                     previous_c4h6_preds: list = None,
                     model_a_pkl_path: str = "models/final/model_A_final_v1.pkl",
                     model_a_json_path: str = "models/final/model_A_final_v1.json") -> dict:
    """
    Computes predictions for C4H8 and C4H6, and sums them to obtain Total_C4.
    Returns a unified diagnostic dictionary.
    """
    # 1. Predict C4H8 (Model A)
    res_a = predict_c4h8(
        process_history=process_history,
        latest_valid_analyzer=latest_valid_c4h8,
        previous_predictions=previous_c4h8_preds,
        pkl_path=model_a_pkl_path,
        json_path=model_a_json_path
    )
    
    # 2. Predict C4H6 (Model B)
    res_b = predict_c4h6(
        latest_valid_analyzer=latest_valid_c4h6,
        previous_predictions=previous_c4h6_preds
    )
    
    # 3. Sum to calculate Total C4
    pred_c4h8 = res_a["predicted_c4h8"]
    pred_c4h6 = res_b["predicted_c4h6"]
    pred_total = pred_c4h8 + pred_c4h6
    
    # Out of spec check (Spec = 0.5 wt% for Total C4)
    is_out_of_spec = pred_total > 0.5
    
    # Combined health check logic (RED overrides YELLOW, YELLOW overrides GREEN)
    health_a = res_a.get("prediction_health", "GREEN")
    health_b = res_b.get("prediction_health", "GREEN")
    
    if health_a == "RED" or health_b == "RED":
        combined_health = "RED"
    elif health_a == "YELLOW" or health_b == "YELLOW":
        combined_health = "YELLOW"
    else:
        combined_health = "GREEN"
        
    return {
        "predicted_total_c4": pred_total,
        "predicted_c4h8":     pred_c4h8,
        "predicted_c4h6":     pred_c4h6,
        "is_out_of_spec":     bool(is_out_of_spec),
        "prediction_health":  combined_health,
        "model_a_used":       res_a["model_used"],
        "model_b_used":       res_b["model_used"],
        "model_a_health":     health_a,
        "model_b_health":     health_b,
        "model_a_reason":     res_a.get("fallback_reason", "None"),
        "model_b_reason":     res_b.get("fallback_reason", "None"),
        "model_a_anchor_ok":  res_a["anchor_available"],
        "model_b_anchor_ok":  res_b["anchor_available"]
    }

if __name__ == "__main__":
    print("=== TESTING predict_total_c4.py ===")
    
    # Create dummy process history (24 hours)
    dummy_history = pd.DataFrame({
        "Feed_Flow":             np.random.normal(80, 5, 24),
        "Reboiling_Steam_Flow":  np.random.normal(21, 1, 24),
        "Reflux_Flow":           np.random.normal(90, 3, 24),
        "Column_Bottom_Temp":    np.random.normal(107, 2, 24),
        "Control_Tray_Temp":     np.random.normal(72, 5, 24),
        "Column_Top_Pressure":   np.random.normal(4.05, 0.1, 24)
    })
    
    # Scenario: Both analyzers healthy
    res = predict_total_c4(
        process_history=dummy_history,
        latest_valid_c4h8={"value": 0.42, "hours_ago": 2},
        latest_valid_c4h6={"value": 0.0045, "hours_ago": 2}
    )
    
    print("\nCombined Output (Normal Operations):")
    print(f"  Predicted C4H8:     {res['predicted_c4h8']:.4f} wt%")
    print(f"  Predicted C4H6:     {res['predicted_c4h6']:.6f} wt% ({res['predicted_c4h6']*10000:.1f} ppm)")
    print(f"  Predicted Total C4: {res['predicted_total_c4']:.4f} wt%")
    print(f"  Out of Spec (>0.5): {res['is_out_of_spec']}")
    print(f"  Overall Health:     {res['prediction_health']}")
    print(f"  Model A:            {res['model_a_used']} (Health: {res['model_a_health']}, Reason: {res['model_a_reason']})")
    print(f"  Model B:            {res['model_b_used']} (Health: {res['model_b_health']}, Reason: {res['model_b_reason']})")
