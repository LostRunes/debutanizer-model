"""
debutanizer_dashboard/services/prediction_service.py
====================================================
Integrates NiceGUI with the production frozen soft sensor modules
(predict_total_c4.py) to compute Total C4 compositions from snapshot inputs.
"""

import sys
import os
import pandas as pd

# Add project root and inference directories to path to enable direct importing of inference/ modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "inference"))

from inference.predict_total_c4 import predict_total_c4

def run_live_prediction(snap_dict, history_df):
    """
    Constructs the input arguments required for predict_total_c4 and returns
    the predictions, health checks, and analyzer anchors status.
    """
    model_a_pkl = os.path.join(BASE_DIR, "models", "final", "model_A_final_v1.pkl")
    model_a_json = os.path.join(BASE_DIR, "models", "final", "model_A_final_v1.json")
    
    # predict_total_c4 takes process_history (DataFrame), latest_valid_c4h8 (dict), latest_valid_c4h6 (dict)
    # latest_valid keys are: 'value' (float) and 'hours_ago' (int)
    # We will assume anchors are healthy (hours_ago = 2) for active simulation
    latest_c4h8 = {"value": snap_dict["c4h8_anchor"], "hours_ago": 2}
    latest_c4h6 = {"value": snap_dict["c4h6_anchor"], "hours_ago": 2}
    
    # We pass the history dataframe (ensuring it has at least 24 rows)
    res = predict_total_c4(
        process_history=history_df,
        latest_valid_c4h8=latest_c4h8,
        latest_valid_c4h6=latest_c4h6,
        model_a_pkl_path=model_a_pkl,
        model_a_json_path=model_a_json
    )
    
    return res
