"""
debutanizer_dashboard/pages/optimizer.py
=======================================
Renders the Advisory Optimizer recommendations interface.
Fetches recommendations dynamically from optimizer_service.py and shows
current vs recommended set-points, safety gauges, cost benefits, and Safety Confidence.
"""

from nicegui import ui
import numpy as np
from services.state_service import state
from services.dashboard_data import get_dashboard_data, safe_num
from components.cards import safety_confidence_card
from components.charts import create_safety_gauge, create_before_after_chart
from services.optimizer_service import optimizer_service

# MAE references
MAE_BOTTOM_TEMP = 0.67992
MAE_TRAY_TEMP = 1.98644
MAE_PRESSURE = 0.01477

def build_optimizer(on_state_change_callback):
    """
    Builds the optimizer advisory recommendations page.
    on_state_change_callback: triggers layout reload when MODE is changed
    """
    data = get_dashboard_data()
    if data is None:
        ui.label("Error: No data loaded from historian.").classes('text-red-500 text-h6')
        return

    snap = data["snap"]
    winner = data["winner"]
    
    ui.label('Physics-Aware Advisory Optimizer').classes('text-h4 text-white font-extrabold mb-2')
    ui.label('Advisory recommendations to minimize C4 slippage within safety operating limits.').classes('text-xs text-grey-5 mb-4')

    # Optimization mode selection header
    with ui.row().classes('w-full items-center justify-between bg-zinc-900/40 p-4 rounded-xl border border-zinc-800 backdrop-blur-md mb-4'):
        with ui.column().classes('gap-1'):
            ui.label("ACTIVE OPTIMIZATION MODE").classes('text-[10px] text-grey-5 font-bold tracking-wider uppercase')
            ui.label(f"Spec-First Optimization ({state.config['MODE'].upper()} mode)").classes('text-md font-bold text-white')
            
        with ui.row().classes('items-center gap-2'):
            ui.label("Switch Mode:").classes('text-xs font-bold text-white')

            def on_mode_change(e):
                state.config["MODE"] = e.value
                state.save_economics_config()
                on_state_change_callback()

            ui.select(
                options={"spec": "Spec Mode (Minimize C4)", "economic": "Economic Mode (Minimize Cost)"},
                value=state.config["MODE"],
                on_change=on_mode_change
            ).classes('w-56 text-xs text-white').props('dark')

    # UI Layout split
    with ui.row().classes('w-full gap-4 items-stretch'):
        # 1. Recommendations Column
        with ui.column().classes('grow gap-4'):
            # Current vs Recommended cards side by side
            with ui.row().classes('w-full gap-4'):
                # Current State card
                with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
                    ui.label("CURRENT OPERATING SETPOINTS").classes('text-xs text-grey-5 font-bold uppercase tracking-wider')
                    with ui.column().classes('gap-2 mt-2'):
                        ui.label(f"Steam Flow: {safe_num(snap['Reboiling_Steam_Flow'], '{:.1f}')} TPH").classes('text-md text-white font-bold')
                        ui.label(f"Reflux Flow: {safe_num(snap['Reflux_Flow'], '{:.1f}')} TPH").classes('text-md text-white font-bold')
                        ui.label(f"Total C4: {safe_num(snap['current_total_c4'], '{:.4f}')} wt%").classes('text-md text-grey-4')
                        
                # Recommended State card
                with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
                    ui.label("RECOMMENDED ADVISORY MOVE").classes('text-xs text-grey-5 font-bold uppercase tracking-wider')
                    if winner is None:
                        ui.label("✓ STABLE OPERATION — NO MOVE NEEDED").classes('text-sm text-green-400 font-extrabold mt-2')
                        ui.label("Current setpoints already satisfy the spec limit or no candidate improves C4 within safety buffers.").classes('text-xs text-grey-5 leading-normal mt-1')
                        with ui.column().classes('gap-1 mt-3'):
                            ui.label(f"Current Steam: {safe_num(snap['Reboiling_Steam_Flow'], '{:.1f}')} TPH").classes('text-sm text-white')
                            ui.label(f"Current Reflux: {safe_num(snap['Reflux_Flow'], '{:.1f}')} TPH").classes('text-sm text-white')
                            ui.label(f"Current Total C4: {safe_num(snap['current_total_c4'], '{:.4f}')} wt%").classes('text-sm text-grey-4')
                    else:
                        steam_change = winner["steam"] - snap["Reboiling_Steam_Flow"]
                        reflux_change = winner["reflux"] - snap["Reflux_Flow"]
                        with ui.column().classes('gap-2 mt-2'):
                            ui.label(f"Steam Flow: {safe_num(winner['steam'], '{:.1f}')} TPH (Δ {steam_change:+.1f} TPH)").classes('text-md text-green-400 font-bold')
                            ui.label(f"Reflux Flow: {safe_num(winner['reflux'], '{:.1f}')} TPH (Δ {reflux_change:+.1f} TPH)").classes('text-md text-green-400 font-bold')
                            ui.label(f"Expected Total C4: {safe_num(winner['pred_total_c4'], '{:.4f}')} wt%").classes('text-md text-white font-bold')
                            c4_red_pct = ((snap['current_total_c4'] - winner['pred_total_c4']) / snap['current_total_c4']) * 100 if snap['current_total_c4'] > 0 else 0
                            ui.label(f"Expected C4 Reduction: {safe_num(c4_red_pct, '{:.1f}%')}").classes('text-sm text-blue-300 font-semibold')
                            
            # Safety Limits gauge block
            with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
                ui.label("Safety Margin Checks (MAE Uncertainty Buffers)").classes('text-xs text-grey-5 font-bold uppercase tracking-wider mb-2')
                
                if winner is not None:
                    create_safety_gauge(winner["pred_bot_temp"], state.config["hard_limit_bottom_temp_degC"], MAE_BOTTOM_TEMP, "Predicted Bottom Temp", "C")
                    ui.element('div').classes('h-2')
                    create_safety_gauge(winner["pred_pressure"], state.config["hard_limit_top_pressure_bar"], MAE_PRESSURE, "Predicted Top Pressure", "bar")
                else:
                    create_safety_gauge(snap["Column_Bottom_Temp"], state.config["hard_limit_bottom_temp_degC"], MAE_BOTTOM_TEMP, "Current Bottom Temp", "C")
                    ui.element('div').classes('h-2')
                    create_safety_gauge(snap["Column_Top_Pressure"], state.config["hard_limit_top_pressure_bar"], MAE_PRESSURE, "Current Top Pressure", "bar")

        # 2. Side Panel - Cost Analysis and Safety Confidence
        with ui.column().classes('w-96 gap-4'):
            if winner is not None:
                # Calculate Safety Confidence
                bot_temp_limit = state.config["hard_limit_bottom_temp_degC"]
                pressure_limit = state.config["hard_limit_top_pressure_bar"]
                
                # Distance including safety buffers
                bot_temp_dist = bot_temp_limit - (winner["pred_bot_temp"] + MAE_BOTTOM_TEMP)
                pressure_dist = pressure_limit - (winner["pred_pressure"] + MAE_PRESSURE)
                
                if bot_temp_dist >= 3.0 and pressure_dist >= 0.10:
                    confidence = "HIGH"
                    conf_details = "Column operates comfortably below thermal and pressure constraints."
                elif bot_temp_dist < 1.0 or pressure_dist < 0.03:
                    confidence = "LOW"
                    conf_details = "[WARNING] Operating close to column limits. Verify instrumentation status."
                else:
                    confidence = "MEDIUM"
                    conf_details = "Adequate safety margins remain under steady-state conditions."
                    
                # Renders safety confidence badge
                safety_confidence_card(confidence, conf_details)
                
                # Utility cost delta analysis
                cost_benefit = winner["cost_benefit"]
                cost_color = "red-400" if cost_benefit > 0 else "green-400"
                cost_prefix = "+" if cost_benefit > 0 else ""
                
                with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
                    ui.label("Utility Cost Analysis").classes('text-xs text-grey-5 font-bold uppercase tracking-wider mb-2')
                    ui.label(f"Hourly Cost Change: {cost_prefix}₹{safe_num(cost_benefit, '{:.2f}')}/hr").classes(f'text-lg font-bold text-{cost_color}')
                    
                    # expected reduction percentage
                    c4_red_pct = ((snap["current_total_c4"] - winner["pred_total_c4"]) / snap["current_total_c4"]) * 100 if snap["current_total_c4"] > 0 else 0
                    ui.label(f"Expected C4 Reduction: {safe_num(c4_red_pct, '{:.1f}%')}").classes('text-sm text-white font-semibold mt-1')
                    
            # Disclaimer section
            with ui.card().classes('w-full bg-zinc-900/10 border border-zinc-900 rounded-xl p-4 shadow-sm'):
                ui.label("PHYSICAL DISCLAIMER").classes('text-[10px] text-orange-400 font-bold tracking-wider uppercase')
                ui.label("Optimizer assumes C4H6 remains at its latest analyzer-estimated value because the validated Model B architecture contains no manipulable-variable response model.").classes('text-[11px] text-grey-5 leading-normal mt-1')

    # 3. Optimization Benefit Timeline Comparison Graph
    history = state.get_current_history()
    if history is not None and not history.empty:
        history_indices = list(history.index)
        optimized_c4 = optimizer_service.run_optimizer_history(state, history_indices, state.config)
        with ui.row().classes('w-full mt-4'):
            with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
                create_before_after_chart(history, optimized_c4, state.config["spec_limit_total_c4_wt_pct"])
