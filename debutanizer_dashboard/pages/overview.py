"""
debutanizer_dashboard/pages/overview.py
=======================================
Renders the Overview page including KPIs, a column profile display,
a snapshot selector, and process details.
"""

from nicegui import ui
import numpy as np
from services.state_service import state
from services.dashboard_data import get_dashboard_data, safe_num
from components.cards import kpi_card

def build_overview(on_state_change_callback):
    """
    Builds the overview page content.
    """
    data = get_dashboard_data()
    if data is None:
        ui.label("Error: No data loaded from historian.").classes('text-red-500 text-h6')
        return

    snap = data["snap"]
    
    # 1. Header with compliance status
    with ui.row().classes('w-full justify-between items-center bg-zinc-900/40 p-4 rounded-xl border border-zinc-800 backdrop-blur-md'):
        with ui.column().classes('gap-1'):
            ui.label("Debutanizer AI Platform").classes('text-xs text-grey-5 uppercase font-bold tracking-wider')
            ui.label(f"Historian Feed — DateTime: {snap['DateTime']}").classes('text-lg font-bold text-white')
        
        # Spec indicator badge
        status = data["spec_status"]
        if status == "NON-COMPLIANT":
            badge_color = "red-950/40 border-red-500/50 text-red-400"
            badge_text = "OUT OF SPECIFICATION (> 0.50 wt%)"
            icon = "error"
        elif status == "INVALID":
            badge_color = "amber-950/40 border-amber-500/50 text-amber-400"
            badge_text = "DATA INVALID / OFFLINE"
            icon = "warning"
        else:
            badge_color = "green-950/40 border-green-500/50 text-green-400"
            badge_text = "IN SPECIFICATION (< 0.50 wt%)"
            icon = "check_circle"
            
        with ui.row().classes(f'items-center gap-2 px-3 py-1.5 rounded-lg border {badge_color}'):
            ui.icon(icon)
            ui.label(badge_text).classes('font-bold text-xs uppercase tracking-wider')

    # 2. Snapshot Selector with Timeline Markers
    with ui.column().classes('w-full mt-4 bg-zinc-900/20 p-4 rounded-xl border border-zinc-900 gap-2'):
        with ui.row().classes('w-full items-center gap-4'):
            ui.label("Scrub Historian Timeline:").classes('text-white font-bold text-sm')
            
            slider = ui.slider(
                min=state.block4_indices[0], 
                max=state.block4_indices[-1], 
                value=state.current_idx
            ).classes('grow')
            
            def on_slider_change(e):
                closest_idx = min(state.block4_indices, key=lambda x: abs(x - slider.value))
                state.current_idx = closest_idx
                on_state_change_callback()
                
            slider.on('change', on_slider_change)
            ui.label(f"Index: {state.current_idx}").classes('text-grey-5 font-bold text-xs')
            
        # Timeline block markers
        with ui.row().classes('w-full justify-between text-[10px] text-grey-6 px-1'):
            ui.label("Block 4 Start")
            ui.label("Campaign Midway")
            ui.label("Validation End")

    # 3. KPI Row (Color Coded)
    # Total C4 status color
    t_c4 = data["pred_total_c4"]
    c4_status = "red" if (np.isnan(t_c4) or t_c4 > 0.50) else "yellow" if t_c4 >= 0.35 else "green"
    
    # C4H8 status color
    c4h8 = data["pred_c4h8"]
    c4h8_status = "red" if (np.isnan(c4h8) or c4h8 > 0.45) else "yellow" if c4h8 >= 0.30 else "green"
    
    # C4H6 status color
    c4h6 = data["pred_c4h6"]
    c4h6_status = "red" if (np.isnan(c4h6) or c4h6 > 0.010) else "yellow" if c4h6 >= 0.005 else "green"
    
    # Loss status color
    loss = data["loss_rs"]
    loss_status = "red" if (np.isnan(loss) or loss > 5000) else "yellow" if loss >= 2000 else "green"

    with ui.row().classes('w-full mt-4 justify-between gap-4'):
        kpi_card("Total C4 Slippage", safe_num(t_c4, "{:.3f} wt%"), icon="science", color="primary", status_color=c4_status)
        kpi_card("Butene (C4H8)", safe_num(c4h8, "{:.4f} wt%"), icon="trending_up", color="accent", status_color=c4h8_status)
        kpi_card("Butadiene (C4H6)", safe_num(c4h6, "{:.5f} wt%"), icon="commit", color="secondary", status_color=c4h6_status)
        kpi_card("Hourly Loss Estimation", "₹ " + safe_num(loss, "{:,.0f}/hr") if not np.isnan(loss) else "--", icon="payments", color="amber-500", status_color=loss_status)

    # 4. Main Body: Column silhouette, health, analyzers, and recommendations
    with ui.row().classes('w-full mt-4 gap-4 items-stretch'):
        # Left Panel: Silhouette Profile
        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
            ui.label("Column Temperature & Pressure Profile").classes('text-white font-bold text-sm mb-4 uppercase tracking-wider')
            
            with ui.row().classes('w-full items-center justify-around no-wrap mt-2'):
                with ui.column().classes('items-center bg-zinc-900/60 p-4 rounded-xl border border-zinc-800 w-32'):
                    ui.label("COLUMN TOP").classes('text-[10px] text-grey-5 font-bold')
                    ui.icon("compress").classes('text-xl text-blue-400 mt-2')
                    ui.label(safe_num(snap['Column_Top_Pressure'], "{:.3f} bar")).classes('text-sm font-bold text-white mt-1')
                    
                with ui.column().classes('items-center bg-zinc-900/60 p-4 rounded-xl border border-zinc-800 w-32'):
                    ui.label("CONTROL TRAY").classes('text-[10px] text-grey-5 font-bold')
                    ui.icon("thermostat").classes('text-xl text-orange-400 mt-2')
                    ui.label(safe_num(snap['Control_Tray_Temp'], "{:.1f} C")).classes('text-sm font-bold text-white mt-1')
                    
                with ui.column().classes('items-center bg-zinc-900/60 p-4 rounded-xl border border-zinc-800 w-32'):
                    ui.label("COLUMN BOTTOM").classes('text-[10px] text-grey-5 font-bold')
                    ui.icon("local_fire_department").classes('text-xl text-red-500 mt-2')
                    ui.label(safe_num(snap['Column_Bottom_Temp'], "{:.1f} C")).classes('text-sm font-bold text-white mt-1')
                    
            # Formulas display
            with ui.row().classes('w-full gap-4 mt-6'):
                with ui.column().classes('grow gap-1 bg-zinc-900/30 p-3 rounded-lg border border-zinc-900'):
                    ui.label("Reflux Ratio (L/F):").classes('font-bold text-xs text-white')
                    ui.html("<span class='text-blue-400 font-mono text-[11px]'>Reflux Ratio = Reflux Flow / Feed Flow</span>")
                    r_ratio = snap['Reflux_Flow'] / snap['Feed_Flow'] if snap['Feed_Flow'] > 0 else np.nan
                    ui.label(f"Current: {safe_num(snap['Reflux_Flow'], '{:.1f}')} / {safe_num(snap['Feed_Flow'], '{:.1f}')} = {safe_num(r_ratio, '{:.3f}')}").classes('text-[10px] text-grey-5')
                    
                with ui.column().classes('grow gap-1 bg-zinc-900/30 p-3 rounded-lg border border-zinc-900'):
                    ui.label("Steam/Feed Ratio (V/F):").classes('font-bold text-xs text-white')
                    ui.html("<span class='text-blue-400 font-mono text-[11px]'>Steam Ratio = Steam Flow / Feed Flow</span>")
                    s_ratio = snap['Reboiling_Steam_Flow'] / snap['Feed_Flow'] if snap['Feed_Flow'] > 0 else np.nan
                    ui.label(f"Current: {safe_num(snap['Reboiling_Steam_Flow'], '{:.1f}')} / {safe_num(snap['Feed_Flow'], '{:.1f}')} = {safe_num(s_ratio, '{:.3f}')}").classes('text-[10px] text-grey-5')

        # Middle Panel: Column Health and Analyzer Status
        with ui.column().classes('w-80 gap-4'):
            # Column Health Card
            with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
                ui.label("Column Health Status").classes('text-white font-bold text-xs mb-3 uppercase tracking-wider')
                
                # Overall status indicator
                h_overall = data["column_health"]["overall"]
                dot_color = "bg-green-400" if h_overall == "Healthy" else "bg-red-400"
                with ui.row().classes('items-center gap-2 mb-3 bg-zinc-900/40 p-2 rounded-lg border border-zinc-800'):
                    ui.element('span').classes(f'w-2 h-2 rounded-full {dot_color} animate-pulse')
                    ui.label(h_overall).classes('font-extrabold text-sm text-white')
                
                # Component health list
                with ui.column().classes('w-full gap-2 text-xs'):
                    # Temp
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label("Temperature Profile:")
                        t_lbl = data["column_health"]["temp"]
                        t_col = "text-green-400" if t_lbl == "Normal" else "text-red-400"
                        ui.label(t_lbl).classes(f"font-bold {t_col}")
                    # Pressure
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label("Pressure Profile:")
                        p_lbl = data["column_health"]["pressure"]
                        p_col = "text-green-400" if p_lbl == "Normal" else "text-red-400"
                        ui.label(p_lbl).classes(f"font-bold {p_col}")
                    # Analyzer
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label("Analyzers overall:")
                        a_lbl = data["column_health"]["analyzer"]
                        a_col = "text-green-400" if a_lbl == "Healthy" else "text-yellow-400" if a_lbl == "Degraded" else "text-red-400"
                        ui.label(a_lbl).classes(f"font-bold {a_col}")

            # Analyzer Status Card
            with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
                ui.label("Analyzer Status & Staleness").classes('text-white font-bold text-xs mb-3 uppercase tracking-wider')
                
                # C4H8 Analyzer
                h8_ago = data["c4h8_hours_ago"]
                h8_status = data["c4h8_status"]
                h8_color = "text-green-400" if h8_status == "ONLINE" else "text-red-400"
                with ui.column().classes('w-full gap-1 mb-2 bg-zinc-900/20 p-2 rounded border border-zinc-900'):
                    with ui.row().classes('w-full justify-between items-center text-xs'):
                        ui.label("C4H8 Butene Analyzer").classes('font-bold text-white')
                        ui.label(h8_status).classes(f"font-extrabold {h8_color}")
                    ui.label(f"Last Valid Reading: {h8_ago if h8_ago < 999 else '--'} hours ago").classes('text-[10px] text-grey-5')
                
                # C4H6 Analyzer
                h6_ago = data["c4h6_hours_ago"]
                h6_status = data["c4h6_status"]
                h6_color = "text-green-400" if h6_status == "ONLINE" else "text-red-400"
                with ui.column().classes('w-full gap-1 bg-zinc-900/20 p-2 rounded border border-zinc-900'):
                    with ui.row().classes('w-full justify-between items-center text-xs'):
                        ui.label("C4H6 Butadiene Analyzer").classes('font-bold text-white')
                        ui.label(h6_status).classes(f"font-extrabold {h6_color}")
                    ui.label(f"Last Valid Reading: {h6_ago if h6_ago < 999 else '--'} hours ago").classes('text-[10px] text-grey-5')

        # Right Panel: Recommendation Preview Card
        with ui.card().classes('w-80 bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md justify-between'):
            with ui.column().classes('w-full gap-2'):
                ui.label("RECOMMENDED ACTION PREVIEW").classes('text-orange-400 font-extrabold text-xs tracking-wider uppercase mb-2')
                
                winner = data["winner"]
                if winner is None:
                    ui.label("STABLE OPERATION").classes('text-md font-bold text-green-400 mt-2')
                    ui.label("No optimization moves recommended. Operating variables are within specification and limits.").classes('text-xs text-grey-5 leading-normal')
                else:
                    steam_delta = winner["steam"] - snap["Reboiling_Steam_Flow"]
                    reflux_delta = winner["reflux"] - snap["Reflux_Flow"]
                    
                    with ui.column().classes('gap-2 mt-1'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon("arrow_circle_up" if steam_delta >= 0 else "arrow_circle_down", color="green" if steam_delta >= 0 else "amber")
                            ui.label(f"Steam: {steam_delta:+.1f} TPH").classes('text-xs font-bold text-white')
                            
                        with ui.row().classes('items-center gap-2'):
                            ui.icon("arrow_circle_up" if reflux_delta >= 0 else "arrow_circle_down", color="green" if reflux_delta >= 0 else "amber")
                            ui.label(f"Reflux: {reflux_delta:+.1f} TPH").classes('text-xs font-bold text-white')
                        
                        # Expected C4 reduction preview
                        with ui.column().classes('bg-zinc-900/40 p-2 rounded border border-zinc-800 text-xs mt-2 w-full'):
                            ui.label("Expected Total C4:").classes('text-grey-5 font-semibold text-[10px]')
                            ui.label(f"{snap['current_total_c4']:.3f} → {winner['pred_total_c4']:.3f} wt%").classes('text-xs font-extrabold text-green-400')
            
            ui.button('Go to Optimizer', on_click=lambda: ui.notify("Select the Advisory Optimizer tab in the left sidebar to apply setpoints.")).classes('w-full bg-blue-800 hover:bg-blue-700 text-white font-bold text-xs py-1.5 rounded-lg')
