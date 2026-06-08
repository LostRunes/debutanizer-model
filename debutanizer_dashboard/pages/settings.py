"""
debutanizer_dashboard/pages/settings.py
======================================
Renders the Settings configurations page. Allows real-time modifications to
DCS safety limits, optimization modes, and economic parameters.
"""

from nicegui import ui
from services.state_service import state

def build_settings(on_state_change_callback):
    """
    Builds the Settings panel page.
    """
    
    ui.label('System Settings & Configuration').classes('text-h4 text-white font-extrabold mb-2')
    ui.label('Calibrate optimization limits, spec thresholds, and economic pricing metrics.').classes('text-xs text-grey-5 mb-4')
    
    with ui.row().classes('w-full gap-4 items-stretch'):
        # Safety & Constraint limits card
        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
            ui.label("DCS SAFETY & OPERATING BOUNDS").classes('text-white font-bold text-sm mb-4 uppercase tracking-wider')
            
            with ui.column().classes('w-full gap-4'):
                # Bottom Temp Limit
                with ui.column().classes('w-full gap-1'):
                    ui.label("Hard Bottom Temp Limit (C):").classes('text-xs font-semibold text-grey-4')
                    bot_limit_input = ui.number(value=state.config["hard_limit_bottom_temp_degC"], format="%.1f", step=0.5).classes('w-full')
                    
                # Top Pressure Limit
                with ui.column().classes('w-full gap-1'):
                    ui.label("Hard Top Pressure Limit (bar):").classes('text-xs font-semibold text-grey-4')
                    pres_limit_input = ui.number(value=state.config["hard_limit_top_pressure_bar"], format="%.2f", step=0.1).classes('w-full')
                    
                # Spec Limit
                with ui.column().classes('w-full gap-1'):
                    ui.label("Spec Limit C4 Slippage (wt%):").classes('text-xs font-semibold text-grey-4')
                    spec_limit_input = ui.number(value=state.config["spec_limit_total_c4_wt_pct"], format="%.2f", step=0.05).classes('w-full')
                    
                # Max Steam move
                with ui.column().classes('w-full gap-1'):
                    ui.label("Max Steam Flow single-step change (TPH):").classes('text-xs font-semibold text-grey-4')
                    steam_move_input = ui.number(value=state.config["max_steam_change_tph"], format="%.1f", step=0.1).classes('w-full')
                    
                # Max Reflux move
                with ui.column().classes('w-full gap-1'):
                    ui.label("Max Reflux Flow single-step change (TPH):").classes('text-xs font-semibold text-grey-4')
                    reflux_move_input = ui.number(value=state.config["max_reflux_change_tph"], format="%.1f", step=0.5).classes('w-full')

        # Utility Costs & C4 value card
        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
            ui.label("PROCESS ECONOMICS CALIBRATION").classes('text-white font-bold text-sm mb-4 uppercase tracking-wider')
            
            with ui.column().classes('w-full gap-4'):
                # Steam TPH cost
                with ui.column().classes('w-full gap-1'):
                    ui.label("Utility Steam cost (₹/TPH):").classes('text-xs font-semibold text-grey-4')
                    steam_cost_input = ui.number(value=state.config["steam_cost_per_tph"], format="%.2f", step=0.5).classes('w-full')
                    
                # Reflux TPH cost
                with ui.column().classes('w-full gap-1'):
                    ui.label("Reflux pump power cost (₹/TPH):").classes('text-xs font-semibold text-grey-4')
                    reflux_cost_input = ui.number(value=state.config["reflux_cost_per_tph"], format="%.2f", step=0.1).classes('w-full')
                    
                # C4 value
                with ui.column().classes('w-full gap-1'):
                    ui.label("C4 Slippage Penalty (₹/wt%):").classes('text-xs font-semibold text-grey-4')
                    c4_val_input = ui.number(value=state.config["c4_value_per_wt_pct"], format="%.2f", step=10.0).classes('w-full')

    # Save button row
    with ui.row().classes('w-full justify-end mt-4'):
        def save_config_changes():
            # Update state configuration
            state.config["hard_limit_bottom_temp_degC"] = bot_limit_input.value
            state.config["hard_limit_top_pressure_bar"] = pres_limit_input.value
            state.config["spec_limit_total_c4_wt_pct"] = spec_limit_input.value
            state.config["max_steam_change_tph"] = steam_move_input.value
            state.config["max_reflux_change_tph"] = reflux_move_input.value
            state.config["steam_cost_per_tph"] = steam_cost_input.value
            state.config["reflux_cost_per_tph"] = reflux_cost_input.value
            state.config["c4_value_per_wt_pct"] = c4_val_input.value
            
            # Save to configs/economics.json
            state.save_economics_config()
            ui.notify("Configuration saved successfully!", type="positive")
            
            # Refresh layout
            on_state_change_callback()
            
        ui.button('Save Configurations', on_click=save_config_changes).classes('bg-blue-700 hover:bg-blue-600 text-white font-bold py-2 px-6 rounded-lg shadow-md transition-all duration-300')
