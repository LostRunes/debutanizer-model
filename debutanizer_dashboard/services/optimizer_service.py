"""
debutanizer_dashboard/services/optimizer_service.py
===================================================
Wrapper to run optimizer calculations from the NiceGUI dashboard using
the production physics-aware advisory optimizer code.
Vectorized for extreme performance to support dynamic timeseries plots.
"""

import sys
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "notebooks"))

from notebooks.optimizer_v2_physics_aware import load_models, MAE_BOTTOM_TEMP, MAE_PRESSURE

class OptimizerService:
    def __init__(self):
        self.model_a = None
        self.surrogates = None
        
    def init_models(self):
        if self.model_a is None:
            self.model_a, self.surrogates = load_models()
            
    def run_optimizer(self, snap_dict, history_df, config):
        """
        Runs physics-aware optimization for the current snapshot and history
        using loaded models and configurations. Vectorized for maximum speed.
        """
        self.init_models()
        return self._optimize_vectorized(snap_dict, history_df, config)
        
    def _optimize_vectorized(self, snapshot, history_24h, config):
        current_steam = snapshot["Reboiling_Steam_Flow"]
        current_reflux = snapshot["Reflux_Flow"]
        current_feed = snapshot["Feed_Flow"]
        
        # Check for NaNs in critical inputs
        if np.isnan(current_steam) or np.isnan(current_reflux) or np.isnan(current_feed) or np.isnan(snapshot["Column_Bottom_Temp"]):
            return None
            
        steam_min = max(config["steam_min_tph"], current_steam - config["max_steam_change_tph"])
        steam_max = min(config["steam_max_tph"], current_steam + config["max_steam_change_tph"])
        reflux_min = max(config["reflux_min_tph"], current_reflux - config["max_reflux_change_tph"])
        reflux_max = min(config["reflux_max_tph"], current_reflux + config["max_reflux_change_tph"])
        
        steam_grid = np.arange(steam_min, steam_max + 0.01, 0.2)
        reflux_grid = np.arange(reflux_min, reflux_max + 0.01, 1.0)
        
        steam_cands, reflux_cands = np.meshgrid(steam_grid, reflux_grid)
        steam_cands = steam_cands.flatten()
        reflux_cands = reflux_cands.flatten()
        n_cands = len(steam_cands)
        
        mean_steam_24h = history_24h["Reboiling_Steam_Flow"].mean()
        mean_reflux_24h = history_24h["Reflux_Flow"].mean()
        mean_bot_temp_24h = history_24h["Column_Bottom_Temp"].mean()
        mean_tray_temp_24h = history_24h["Control_Tray_Temp"].mean()
        mean_pressure_24h = history_24h["Column_Top_Pressure"].mean()
        
        last_row = history_24h.iloc[-1]
        
        SURROGATE_FEATURES = [
            "Feed_Flow", "Reboiling_Steam_Flow", "Reflux_Flow", "Steam_Feed_Ratio", "Reflux_Ratio",
            "Reboiling_Steam_Flow_dev24h", "Reflux_Flow_dev24h", "Column_Bottom_Temp_dev24h",
            "Control_Tray_Temp_dev24h", "Column_Top_Pressure_dev24h", "Reboiling_Steam_Flow_lag1",
            "Reflux_Flow_lag1", "Column_Bottom_Temp_lag1", "Control_Tray_Temp_lag1", "Column_Top_Pressure_lag1"
        ]
        
        # Build features dataframe
        surr_feat = pd.DataFrame({
            "Feed_Flow": np.full(n_cands, current_feed),
            "Reboiling_Steam_Flow": steam_cands,
            "Reflux_Flow": reflux_cands,
            "Steam_Feed_Ratio": steam_cands / current_feed,
            "Reflux_Ratio": reflux_cands / current_feed,
            "Reboiling_Steam_Flow_dev24h": steam_cands - mean_steam_24h,
            "Reflux_Flow_dev24h": reflux_cands - mean_reflux_24h,
            "Column_Bottom_Temp_dev24h": np.full(n_cands, snapshot["Column_Bottom_Temp"] - mean_bot_temp_24h),
            "Control_Tray_Temp_dev24h": np.full(n_cands, snapshot["Control_Tray_Temp"] - mean_tray_temp_24h),
            "Column_Top_Pressure_dev24h": np.full(n_cands, snapshot["Column_Top_Pressure"] - mean_pressure_24h),
            "Reboiling_Steam_Flow_lag1": np.full(n_cands, last_row["Reboiling_Steam_Flow"]),
            "Reflux_Flow_lag1": np.full(n_cands, last_row["Reflux_Flow"]),
            "Column_Bottom_Temp_lag1": np.full(n_cands, last_row["Column_Bottom_Temp"]),
            "Control_Tray_Temp_lag1": np.full(n_cands, last_row["Control_Tray_Temp"]),
            "Column_Top_Pressure_lag1": np.full(n_cands, last_row["Column_Top_Pressure"])
        })[SURROGATE_FEATURES]
        
        delta_bot = self.surrogates["bottom_temp"].predict(surr_feat)
        delta_tray = self.surrogates["tray_temp"].predict(surr_feat)
        delta_pres = self.surrogates["pressure"].predict(surr_feat)
        
        pred_bot = snapshot["Column_Bottom_Temp"] + delta_bot
        pred_tray = snapshot["Control_Tray_Temp"] + delta_tray
        pred_pres = snapshot["Column_Top_Pressure"] + delta_pres
        
        # Safety limits checks
        valid_mask = (
            (pred_bot + MAE_BOTTOM_TEMP <= config["hard_limit_bottom_temp_degC"]) &
            (pred_pres + MAE_PRESSURE <= config["hard_limit_top_pressure_bar"])
        )
        
        if not np.any(valid_mask):
            return None
            
        valid_indices = np.where(valid_mask)[0]
        
        steam_cands_v = steam_cands[valid_indices]
        reflux_cands_v = reflux_cands[valid_indices]
        pred_bot_v = pred_bot[valid_indices]
        pred_tray_v = pred_tray[valid_indices]
        pred_pres_v = pred_pres[valid_indices]
        
        MODEL_A_FEATURES = [
            "C4H8_campaign_anchor", "Steam_Feed_Ratio", "Reflux_Ratio",
            "Reboiling_Steam_Flow_dev24h", "Reflux_Flow_dev24h",
            "Column_Bottom_Temp_dev24h", "Control_Tray_Temp_dev24h", "Column_Top_Pressure_dev24h"
        ]
        
        # Check if C4H8 anchor is NaN
        c4h8_anchor = snapshot["c4h8_anchor"]
        c4h6_anchor = snapshot["c4h6_anchor"]
        current_total = snapshot["current_total_c4"]
        
        # If any are NaN, fall back safely
        if np.isnan(c4h8_anchor) or np.isnan(c4h6_anchor) or np.isnan(current_total):
            return None
            
        ma_feat = pd.DataFrame({
            "C4H8_campaign_anchor": np.full(len(valid_indices), c4h8_anchor),
            "Steam_Feed_Ratio": steam_cands_v / current_feed,
            "Reflux_Ratio": reflux_cands_v / current_feed,
            "Reboiling_Steam_Flow_dev24h": steam_cands_v - mean_steam_24h,
            "Reflux_Flow_dev24h": reflux_cands_v - mean_reflux_24h,
            "Column_Bottom_Temp_dev24h": pred_bot_v - mean_bot_temp_24h,
            "Control_Tray_Temp_dev24h": pred_tray_v - mean_tray_temp_24h,
            "Column_Top_Pressure_dev24h": pred_pres_v - mean_pressure_24h
        })[MODEL_A_FEATURES]
        
        pred_c4h8 = self.model_a.predict(ma_feat)
        pred_total = pred_c4h8 + c4h6_anchor
        
        better_mask = pred_total < current_total
        if not np.any(better_mask):
            return None
            
        better_indices = np.where(better_mask)[0]
        
        candidates = []
        for bi in better_indices:
            steam_cand = steam_cands_v[bi]
            reflux_cand = reflux_cands_v[bi]
            delta_s = steam_cand - current_steam
            delta_r = reflux_cand - current_reflux
            cost_benefit = (config["steam_cost_per_tph"] * delta_s) + (config["reflux_cost_per_tph"] * delta_r)
            
            candidates.append({
                "steam": steam_cand,
                "reflux": reflux_cand,
                "pred_bot_temp": pred_bot_v[bi],
                "pred_tray_temp": pred_tray_v[bi],
                "pred_pressure": pred_pres_v[bi],
                "pred_c4h8": pred_c4h8[bi],
                "pred_c4h6": c4h6_anchor,
                "pred_total_c4": pred_total[bi],
                "cost_benefit": cost_benefit
            })
            
        df_cands = pd.DataFrame(candidates)
        
        spec_limit = config["spec_limit_total_c4_wt_pct"]
        spec_compliant = df_cands[df_cands["pred_total_c4"] < spec_limit]
        
        if not spec_compliant.empty:
            if config["MODE"] == "economic":
                winner = spec_compliant.sort_values(by="cost_benefit").iloc[0].to_dict()
            else:
                winner = spec_compliant.sort_values(by="pred_total_c4").iloc[0].to_dict()
        else:
            winner = df_cands.sort_values(by="pred_total_c4").iloc[0].to_dict()
            
        return winner

    def run_optimizer_history(self, state_instance, history_indices, config):
        """
        Runs optimization sequentially for the list of indices (history window)
        and returns a list of optimized Total C4 values.
        """
        self.init_models()
        optimized_c4_list = []
        
        df_all = state_instance.df
        for idx in history_indices:
            # Build snapshot dictionary for this idx
            row = df_all.loc[idx]
            
            # Check if actual C4 is NaN
            actual_total = row["Total_C4"]
            if np.isnan(actual_total):
                optimized_c4_list.append(np.nan)
                continue
                
            snap = {
                "DateTime": str(row["DateTime"]),
                "Feed_Flow": float(row["Feed_Flow"]),
                "Reboiling_Steam_Flow": float(row["Reboiling_Steam_Flow"]),
                "Reflux_Flow": float(row["Reflux_Flow"]),
                "Column_Bottom_Temp": float(row["Column_Bottom_Temp"]),
                "Control_Tray_Temp": float(row["Control_Tray_Temp"]),
                "Column_Top_Pressure": float(row["Column_Top_Pressure"]),
                "c4h8_anchor": float(row["C4H8_campaign_anchor"]) if "C4H8_campaign_anchor" in df_all.columns else np.nan,
                "c4h6_anchor": float(row["C4H6_campaign_anchor"]) if "C4H6_campaign_anchor" in df_all.columns else np.nan,
                "current_total_c4": float(actual_total)
            }
            
            # Extract 24h history preceding this idx
            start_idx = max(0, idx - 24)
            hist = df_all.loc[start_idx:idx]
            
            # Run optimizer
            winner = self._optimize_vectorized(snap, hist, config)
            if winner is not None:
                optimized_c4_list.append(winner["pred_total_c4"])
            else:
                optimized_c4_list.append(actual_total)
                
        return optimized_c4_list

optimizer_service = OptimizerService()
