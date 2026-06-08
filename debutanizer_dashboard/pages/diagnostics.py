"""
debutanizer_dashboard/pages/diagnostics.py
=========================================
Renders the Model Diagnostics tab. Displays feature importance bar charts
for T1 (Bottom Temp), T2 (Tray Temp), and T3 (Pressure) delta models.
"""

import os
import pandas as pd
from nicegui import ui
from components.charts import create_feature_importance_chart

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURROGATE_DIR = os.path.join(BASE_DIR, "models", "surrogates")

def build_diagnostics():
    
    ui.label('Model Diagnostics').classes('text-h4 text-white font-extrabold mb-2')
    ui.label('Inspect the feature contributions for each of the process surrogate models.').classes('text-xs text-grey-5 mb-4')
    
    # Check if files exist
    bot_file = os.path.join(SURROGATE_DIR, "surrogate_feature_importance_bottom.csv")
    tray_file = os.path.join(SURROGATE_DIR, "surrogate_feature_importance_tray.csv")
    pres_file = os.path.join(SURROGATE_DIR, "surrogate_feature_importance_pressure.csv")
    
    with ui.row().classes('w-full gap-4 items-stretch'):
        # Bottom Temp T1
        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
            if os.path.exists(bot_file):
                df_bot = pd.read_csv(bot_file)
                create_feature_importance_chart(df_bot, "Bottom Temp (T1) Feature Importance")
            else:
                ui.label("Bottom Temp T1 feature importance CSV missing.").classes('text-grey-5')
                
        # Tray Temp T2
        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
            if os.path.exists(tray_file):
                df_tray = pd.read_csv(tray_file)
                create_feature_importance_chart(df_tray, "Tray Temp (T2) Feature Importance")
            else:
                ui.label("Tray Temp T2 feature importance CSV missing.").classes('text-grey-5')
                
    with ui.row().classes('w-full gap-4 items-stretch mt-4'):
        # Top Pressure T3
        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
            if os.path.exists(pres_file):
                df_pres = pd.read_csv(pres_file)
                create_feature_importance_chart(df_pres, "Top Pressure (T3) Feature Importance")
            else:
                ui.label("Top Pressure T3 feature importance CSV missing.").classes('text-grey-5')
