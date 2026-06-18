"""
debutanizer_dashboard/pages/soft_sensor.py
=========================================
Renders the Soft Sensor Live Prediction simulation interface.
Allows manual adjustments to process inputs to compute real-time outputs.
"""

from nicegui import ui
import pandas as pd
from services.state_service import state
from services.prediction_service import run_live_prediction


def build_soft_sensor():
    
    snap = state.get_current_snapshot()
    if snap is None:
        ui.label("Error: No snapshot loaded").classes('text-red-500')
        return

    # Initialize simulation inputs fresh from the current snapshot each time the page loads
    sim_inputs = {}
    for col in ["Feed_Flow", "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Bottom_Temp", "Control_Tray_Temp", "Column_Top_Pressure"]:
        sim_inputs[col] = snap[col]

    # Keep campaign anchors fixed (represent current campaign baseline)
    sim_inputs["c4h8_anchor"] = snap["c4h8_anchor"]
    sim_inputs["c4h6_anchor"] = snap["c4h6_anchor"]

    ui.label('Live Soft Sensor Simulator').classes('text-h4 text-white font-extrabold mb-2')
    ui.label('Tweak the debutanizer column process variables to simulate live C4 slippage predictions.').classes('text-xs text-grey-5 mb-4')

    with ui.row().classes('w-full gap-4 items-stretch'):
        # Input controls
        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
            ui.label("Process Manipulated & Disturbance Variables").classes('text-white font-bold text-sm mb-4 uppercase tracking-wider')
            
            with ui.row().classes('w-full gap-4'):
                with ui.column().classes('grow gap-4'):
                    # Feed Flow
                    ui.label("Feed Flow to Column (TPH)").classes('text-xs font-bold text-grey-5 uppercase')
                    feed_input = ui.number(value=sim_inputs["Feed_Flow"], format="%.1f", step=0.5).classes('w-full')
                    
                    # Steam Flow
                    ui.label("Reboiling Steam Flow (TPH)").classes('text-xs font-bold text-grey-5 uppercase')
                    steam_input = ui.number(value=sim_inputs["Reboiling_Steam_Flow"], format="%.1f", step=0.1).classes('w-full')
                    
                    # Reflux Flow
                    ui.label("Reflux Flow (TPH)").classes('text-xs font-bold text-grey-5 uppercase')
                    reflux_input = ui.number(value=sim_inputs["Reflux_Flow"], format="%.1f", step=0.5).classes('w-full')
                    
                with ui.column().classes('grow gap-4'):
                    # Bottom Temp
                    ui.label("Column Bottom Temp (C)").classes('text-xs font-bold text-grey-5 uppercase')
                    bot_input = ui.number(value=sim_inputs["Column_Bottom_Temp"], format="%.2f", step=0.1).classes('w-full')
                    
                    # Tray Temp
                    ui.label("Control Tray Temp (C)").classes('text-xs font-bold text-grey-5 uppercase')
                    tray_input = ui.number(value=sim_inputs["Control_Tray_Temp"], format="%.2f", step=0.1).classes('w-full')
                    
                    # Pressure
                    ui.label("Column Top Pressure (bar)").classes('text-xs font-bold text-grey-5 uppercase')
                    pres_input = ui.number(value=sim_inputs["Column_Top_Pressure"], format="%.3f", step=0.01).classes('w-full')

        # Output / Results Panel
        with ui.card().classes('w-96 bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md justify-between'):
            ui.label("Soft Sensor Outputs (T+1 prediction)").classes('text-white font-bold text-sm mb-4 uppercase tracking-wider')
            
            # Prediction values display container
            out_col = ui.column().classes('w-full gap-4 grow justify-center')
            
            with out_col:
                total_c4_label = ui.label("Total C4: Running prediction...").classes('text-2xl font-extrabold text-blue-400')
                c4h8_label = ui.label("C4H8 Butene: -- wt%")
                c4h6_label = ui.label("C4H6 Butadiene: -- wt%")
                health_badge = ui.label("Health: --").classes('px-2 py-0.5 rounded text-xs uppercase tracking-wider')
            
            def run_simulation_prediction():
                # Read current inputs
                sim_inputs["Feed_Flow"] = feed_input.value
                sim_inputs["Reboiling_Steam_Flow"] = steam_input.value
                sim_inputs["Reflux_Flow"] = reflux_input.value
                sim_inputs["Column_Bottom_Temp"] = bot_input.value
                sim_inputs["Control_Tray_Temp"] = tray_input.value
                sim_inputs["Column_Top_Pressure"] = pres_input.value
                
                # Fetch history for deviations calculations. We will create a local copy and inject simulator values
                hist = state.get_prediction_window().copy()
                # Overwrite last row with our simulated inputs
                last_row_idx = hist.index[-1]
                for col in ["Feed_Flow", "Reboiling_Steam_Flow", "Reflux_Flow", "Column_Bottom_Temp", "Control_Tray_Temp", "Column_Top_Pressure"]:
                    hist.loc[last_row_idx, col] = sim_inputs[col]
                    
                res = run_live_prediction(sim_inputs, hist)
                
                # Update labels
                total_c4_label.set_text(f"Total C4: {res['predicted_total_c4']:.4f} wt%")
                c4h8_label.set_text(f"Butene (C4H8): {res['predicted_c4h8']:.4f} wt%")
                c4h6_label.set_text(f"Butadiene (C4H6): {res['predicted_c4h6']:.5f} wt%")
                
                # Health indicator badge styling
                health = res.get("prediction_health", "GREEN")
                health_badge.set_text(f"Prediction Health: {health}")
                if health == "GREEN":
                    health_badge.classes(replace='px-2 py-0.5 rounded text-xs uppercase tracking-wider bg-green-950/40 text-green-400 border border-green-500/30')
                elif health == "YELLOW":
                    health_badge.classes(replace='px-2 py-0.5 rounded text-xs uppercase tracking-wider bg-yellow-950/40 text-yellow-400 border border-yellow-500/30')
                else:
                    health_badge.classes(replace='px-2 py-0.5 rounded text-xs uppercase tracking-wider bg-red-950/40 text-red-400 border border-red-500/30')
            
            # Predict Button
            predict_btn = ui.button('Execute Prediction Run', on_click=run_simulation_prediction).classes('w-full mt-4 bg-blue-700 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300')
            
            # Run initial prediction when loaded
            ui.timer(0.1, run_simulation_prediction, once=True)
