"""
debutanizer_dashboard/services/state_service.py
===============================================
Manages global UI state: selected snapshot index, loaded economics configs,
and utility handlers to retrieve rows from data/features.parquet.
"""

import os
import json
import pandas as pd
import numpy as np

# Resolve path relative to debutanizer_dashboard/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_FILE = os.path.join(BASE_DIR, "data", "features.parquet")
ECONOMICS_CONFIG = os.path.join(BASE_DIR, "configs", "economics.json")

class GlobalState:
    def __init__(self):
        self.df = None
        self.block4_indices = []
        self.current_idx = None
        self.config = {}
        
        # Load parquet data
        if os.path.exists(FEATURES_FILE):
            self.df = pd.read_parquet(FEATURES_FILE)
            # Add campaign anchors if missing
            if "C4H8_campaign_anchor" not in self.df.columns:
                self.df["C4H8_last_valid"] = self.df["C4H8_Bottom"].copy()
                self.df.loc[self.df["C4H8_Bottom_stuck"], "C4H8_last_valid"] = np.nan
                self.df["C4H8_campaign_anchor"] = (
                    self.df.groupby("Data_Block")["C4H8_last_valid"]
                      .transform(lambda x: x.shift(1).ffill(limit=72))
                )
            if "C4H6_campaign_anchor" not in self.df.columns:
                self.df["C4H6_last_valid"] = self.df["C4H6_Bottom"].copy()
                self.df.loc[self.df["C4H6_Bottom_stuck"], "C4H6_last_valid"] = np.nan
                self.df["C4H6_campaign_anchor"] = (
                    self.df.groupby("Data_Block")["C4H6_last_valid"]
                      .transform(lambda x: x.shift(1).ffill(limit=12))
                )
            
            # Find Block 4 indices (refinery validation set)
            self.block4_indices = list(self.df[self.df["Data_Block"] == 4].index)
            if self.block4_indices:
                # Default to the first index in Block 4
                self.current_idx = self.block4_indices[0]
        else:
            print(f"Warning: Features file not found at {FEATURES_FILE}")
            
        # Load config
        self.load_economics_config()
        
    def load_economics_config(self):
        if os.path.exists(ECONOMICS_CONFIG):
            with open(ECONOMICS_CONFIG, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {
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
            
    def save_economics_config(self):
        with open(ECONOMICS_CONFIG, "w") as f:
            json.dump(self.config, f, indent=4)
            
    def get_current_snapshot(self):
        if self.df is None or self.current_idx is None:
            return None
        row = self.df.loc[self.current_idx]
        return {
            "DateTime": str(row["DateTime"]),
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
        
    def get_current_history(self):
        if self.df is None or self.current_idx is None:
            return None
        start_idx = max(0, self.current_idx - 24)
        return self.df.loc[start_idx:self.current_idx]

# Single shared instance
state = GlobalState()
