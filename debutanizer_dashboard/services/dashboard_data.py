"""
debutanizer_dashboard/services/dashboard_data.py
================================================
Centralized service to aggregate historian snapshots, run soft-sensor
predictions, calculate analyzer staleness, and evaluate column health.
"""

import os
import sys
import pandas as pd
import numpy as np

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "notebooks"))

from services.state_service import state
from services.prediction_service import run_live_prediction
from services.optimizer_service import optimizer_service

def safe_num(val, fmt="{:.3f}", default="--"):
    """
    Safely formats a numerical value. Returns default if nan or None.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return fmt.format(val)

def get_dashboard_data():
    """
    Aggregates all data required for rendering the dashboard UI.
    """
    snap = state.get_current_snapshot()
    history = state.get_current_history()
    
    if snap is None or history is None:
        return None
        
    current_idx = state.current_idx
    df_all = state.df
    
    # Calculate C4H8 analyzer status and hours ago
    c4h8_valid_rows = df_all.loc[:current_idx]
    c4h8_valid_rows = c4h8_valid_rows[~c4h8_valid_rows["C4H8_Bottom_stuck"] & c4h8_valid_rows["C4H8_Bottom"].notna()]
    if not c4h8_valid_rows.empty:
        last_c4h8_idx = c4h8_valid_rows.index[-1]
        c4h8_hours_ago = int(current_idx - last_c4h8_idx)
        c4h8_last_val = float(c4h8_valid_rows.loc[last_c4h8_idx, "C4H8_Bottom"])
    else:
        c4h8_hours_ago = 999
        c4h8_last_val = np.nan
        
    # Calculate C4H6 analyzer status and hours ago
    c4h6_valid_rows = df_all.loc[:current_idx]
    c4h6_valid_rows = c4h6_valid_rows[~c4h6_valid_rows["C4H6_Bottom_stuck"] & c4h6_valid_rows["C4H6_Bottom"].notna()]
    if not c4h6_valid_rows.empty:
        last_c4h6_idx = c4h6_valid_rows.index[-1]
        c4h6_hours_ago = int(current_idx - last_c4h6_idx)
        c4h6_last_val = float(c4h6_valid_rows.loc[last_c4h6_idx, "C4H6_Bottom"])
    else:
        c4h6_hours_ago = 999
        c4h6_last_val = np.nan

    # Run predictions
    pred_res = run_live_prediction(snap, history)
    pred_total = pred_res["predicted_total_c4"]
    pred_c4h8 = pred_res["predicted_c4h8"]
    pred_c4h6 = pred_res["predicted_c4h6"]
    
    # Calculate loss: Loss = Feed_Flow * Bottom_Total_C4% * Pricing (142 Rs/kg/hr)
    if not np.isnan(pred_total):
        c4_frac = pred_total / 100.0
        feed_kg_hr = snap["Feed_Flow"] * 1000.0
        c4_loss_kg_hr = feed_kg_hr * c4_frac
        loss_val_rs = c4_loss_kg_hr * 142.0
    else:
        loss_val_rs = np.nan
        
    # Spec Status
    if np.isnan(pred_total):
        spec_status = "INVALID"
    elif pred_total > state.config["spec_limit_total_c4_wt_pct"]:
        spec_status = "NON-COMPLIANT"
    else:
        spec_status = "COMPLIANT"
        
    # Column Temperature & Pressure Health status
    temp_ok = snap["Column_Bottom_Temp"] < state.config["hard_limit_bottom_temp_degC"]
    pres_ok = snap["Column_Top_Pressure"] < state.config["hard_limit_top_pressure_bar"]
    
    # Analyzer health
    c4h8_healthy = c4h8_hours_ago <= 72
    c4h6_healthy = c4h6_hours_ago <= 12
    analyzer_overall = "Healthy" if (c4h8_healthy and c4h6_healthy) else "Degraded" if (c4h8_healthy or c4h6_healthy) else "Offline"
    
    overall_health = "Healthy" if (temp_ok and pres_ok and c4h8_healthy and c4h6_healthy) else "Maintenance Needed"
    
    column_health = {
        "temp": "Normal" if temp_ok else "HIGH TEMP ALERT",
        "pressure": "Normal" if pres_ok else "HIGH PRESSURE ALERT",
        "analyzer": analyzer_overall,
        "overall": overall_health
    }
    
    # Run optimizer for preview card
    winner = optimizer_service.run_optimizer(snap, history, state.config)
    
    return {
        "snap": snap,
        "c4h8_hours_ago": c4h8_hours_ago,
        "c4h8_last_val": c4h8_last_val,
        "c4h8_status": "ONLINE" if c4h8_healthy else "OFFLINE",
        "c4h6_hours_ago": c4h6_hours_ago,
        "c4h6_last_val": c4h6_last_val,
        "c4h6_status": "ONLINE" if c4h6_healthy else "OFFLINE",
        "pred_total_c4": pred_total,
        "pred_c4h8": pred_c4h8,
        "pred_c4h6": pred_c4h6,
        "loss_rs": loss_val_rs,
        "spec_status": spec_status,
        "column_health": column_health,
        "winner": winner
    }
