"""
inference/predict_c4h6.py
==========================
Production inference script for Model B (C4H6).
Performs a deterministic prediction using the campaign analyzer anchor:
- Level 1: 12-hour Campaign Anchor (shifted by 1h, leak-free).
- Level 2: 24-hour Rolling Mean of recent predictions (fallback).
- Level 3: Block 4 Historical Mean default of 0.005663 wt%.

Inputs:
- latest_valid_analyzer: dict containing:
  - 'value': float or None (the most recent C4H6_Bottom analyzer reading)
  - 'hours_ago': int (the number of hours since that reading was taken)
- previous_predictions: list of floats or None (history of recent model predictions for fallback)

Outputs:
- dict containing:
  - 'predicted_c4h6': float (the final predicted composition in wt%)
  - 'model_used': str ('Model B (12h Anchor)' or 'Fallback (Rolling Mean)' or 'Fallback (Default)')
  - 'anchor_available': bool (whether the campaign anchor was available)
  - 'anchor_value': float or None (the value used as the anchor)
"""

import numpy as np

# Constant parameters
DEFAULT_C4H6_MEAN = 0.005663  # Block 4 target mean wt%

def predict_c4h6(latest_valid_analyzer: dict, 
                 previous_predictions: list = None) -> dict:
    """
    Predicts C4H6 wt% using the deterministic anchor fallback hierarchy.
    """
    anchor_available = False
    anchor_value = None
    
    # Check Level 1: 12-hour Anchor (stuck checking and zero filtering handled at source)
    if latest_valid_analyzer is not None:
        val = latest_valid_analyzer.get("value")
        hours_ago = latest_valid_analyzer.get("hours_ago")
        if val is not None and hours_ago is not None and hours_ago <= 12 and hours_ago >= 1:
            anchor_available = True
            anchor_value = float(val)

    if anchor_available:
        return {
            "predicted_c4h6": anchor_value,
            "model_used": "Model B (12h Anchor)",
            "anchor_available": True,
            "anchor_value": anchor_value,
            "prediction_health": "GREEN",
            "fallback_reason": "None"
        }
        
    # Check for hard timeout of 168 hours (7 days) on the analyzer
    hard_timeout = False
    analyzer_hours = None
    if latest_valid_analyzer is not None:
        analyzer_hours = latest_valid_analyzer.get("hours_ago")
        if analyzer_hours is not None and analyzer_hours > 168:
            hard_timeout = True

    # Check for stale analyzer warning (> 12 hours)
    analyzer_stale = False
    if analyzer_hours is not None and analyzer_hours > 12:
        analyzer_stale = True

    # Level 2 Fallback: Rolling average of recent predictions
    if previous_predictions and len(previous_predictions) > 0 and not hard_timeout:
        valid_preds = [p for p in previous_predictions[-24:] if p is not None]
        if len(valid_preds) >= 6:
            rolling_mean_pred = float(np.mean(valid_preds))
            reason = "Analyzer stale >12h" if analyzer_stale else "Generic fallback"
            return {
                "predicted_c4h6": rolling_mean_pred,
                "model_used": "Fallback (Rolling Mean)",
                "anchor_available": False,
                "anchor_value": None,
                "prediction_health": "YELLOW",
                "fallback_reason": reason
            }
        else:
            reason = "Too few predictions available (<6h)"
            return {
                "predicted_c4h6": DEFAULT_C4H6_MEAN,
                "model_used": "Fallback (Default)",
                "anchor_available": False,
                "anchor_value": None,
                "prediction_health": "RED",
                "fallback_reason": reason
            }
            
    # Level 3 Fallback: Campaign default mean
    if hard_timeout:
        reason = "Analyzer offline >168h"
    elif not previous_predictions or len(previous_predictions) == 0:
        reason = "No previous predictions available"
    else:
        reason = "Model execution failure"
        
    return {
        "predicted_c4h6": DEFAULT_C4H6_MEAN,
        "model_used": "Fallback (Default)",
        "anchor_available": False,
        "anchor_value": None,
        "prediction_health": "RED",
        "fallback_reason": reason
    }

if __name__ == "__main__":
    print("=== TESTING predict_c4h6.py ===")
    
    # Test Scenario 1: Anchor Available (within 12 hours)
    analyzer_ok = {"value": 0.0042, "hours_ago": 3}
    res_ok = predict_c4h6(analyzer_ok)
    print(f"Scenario 1 (Anchor Available):")
    print(f"  Result:      {res_ok['predicted_c4h6']:.6f} wt% ({res_ok['predicted_c4h6']*10000:.1f} ppm)")
    print(f"  Model Used:  {res_ok['model_used']}")
    print(f"  Anchor OK:   {res_ok['anchor_available']}")
    
    # Test Scenario 2: Anchor Stale (> 12 hours)
    analyzer_stale = {"value": 0.0042, "hours_ago": 13}
    prev_preds = [0.0050, 0.0048, 0.0052, 0.0051, 0.0049, 0.0047, 0.0048]
    res_stale = predict_c4h6(analyzer_stale, previous_predictions=prev_preds)
    print(f"\nScenario 2 (Anchor Stale, Fallback with predictions):")
    print(f"  Result:      {res_stale['predicted_c4h6']:.6f} wt% ({res_stale['predicted_c4h6']*10000:.1f} ppm)")
    print(f"  Model Used:  {res_stale['model_used']}")
    
    # Test Scenario 3: No Data
    res_none = predict_c4h6(None, None)
    print(f"\nScenario 3 (Default Fallback):")
    print(f"  Result:      {res_none['predicted_c4h6']:.6f} wt% ({res_none['predicted_c4h6']*10000:.1f} ppm)")
    print(f"  Model Used:  {res_none['model_used']}")
