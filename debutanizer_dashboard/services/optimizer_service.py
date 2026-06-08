"""
debutanizer_dashboard/services/optimizer_service.py
===================================================
Wrapper to run optimizer calculations from the NiceGUI dashboard using
the production physics-aware advisory optimizer code.
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "notebooks"))

from notebooks.optimizer_v2_physics_aware import load_models, optimize

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
        using loaded models and configurations.
        """
        self.init_models()
        return optimize(snap_dict, history_df, self.model_a, self.surrogates, config)

optimizer_service = OptimizerService()
