"""
debutanizer_dashboard/pages/trends.py
=====================================
Renders the Historical Trends tab. Allows the user to select process variables
and displays an interactive dual-axis line plot.
"""

from nicegui import ui
from services.state_service import state
from components.charts import create_trend_chart

# Local view state variables
y_axis_1 = "Total_C4"
y_axis_2 = "Reboiling_Steam_Flow"

def build_trends(on_state_change_callback):
    """
    Builds the trends tab content.
    """
    
    history = state.get_current_history()
    if history is None or history.empty:
        ui.label("Error: No historian history loaded.").classes('text-red-500')
        return
        
    global y_axis_1, y_axis_2
    
    ui.label('Historical Trends').classes('text-h4 text-white font-extrabold mb-2')
    ui.label('Observe the dynamic relationship between process inputs and C4 slippage compositions.').classes('text-xs text-grey-5 mb-4')
    
    # Selectors row
    with ui.row().classes('w-full items-center gap-4 bg-zinc-900/40 p-4 rounded-xl border border-zinc-800 backdrop-blur-md mb-4'):
        # Y Axis 1 selector
        with ui.row().classes('items-center gap-2'):
            ui.label("Y-Axis 1 (Green):").classes('text-xs font-bold text-white')
            y1_select = ui.select(
                options={
                    "Total_C4": "Total C4 Slippage (wt%)",
                    "C4H8_Bottom": "Butene C4H8 (wt%)",
                    "C4H6_Bottom": "Butadiene C4H6 (wt%)",
                    "Reboiling_Steam_Flow": "Steam Flow (TPH)",
                    "Reflux_Flow": "Reflux Flow (TPH)",
                    "Column_Bottom_Temp": "Column Bottom Temp (C)",
                    "Control_Tray_Temp": "Control Tray Temp (C)",
                    "Column_Top_Pressure": "Top Pressure (bar)",
                    "Feed_Flow": "Feed Flow (TPH)"
                },
                value=y_axis_1
            ).classes('w-56 text-xs')
            
            def on_y1_change(e):
                global y_axis_1
                y_axis_1 = y1_select.value
                on_state_change_callback()
                
            y1_select.on('change', on_y1_change)
            
        # Y Axis 2 selector
        with ui.row().classes('items-center gap-2'):
            ui.label("Y-Axis 2 (Blue Dash):").classes('text-xs font-bold text-white')
            y2_select = ui.select(
                options={
                    "Total_C4": "Total C4 Slippage (wt%)",
                    "C4H8_Bottom": "Butene C4H8 (wt%)",
                    "C4H6_Bottom": "Butadiene C4H6 (wt%)",
                    "Reboiling_Steam_Flow": "Steam Flow (TPH)",
                    "Reflux_Flow": "Reflux Flow (TPH)",
                    "Column_Bottom_Temp": "Column Bottom Temp (C)",
                    "Control_Tray_Temp": "Control Tray Temp (C)",
                    "Column_Top_Pressure": "Top Pressure (bar)",
                    "Feed_Flow": "Feed Flow (TPH)"
                },
                value=y_axis_2
            ).classes('w-56 text-xs')
            
            def on_y2_change(e):
                global y_axis_2
                y_axis_2 = y2_select.value
                on_state_change_callback()
                
            y2_select.on('change', on_y2_change)

    # Render Plotly Chart
    with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
        create_trend_chart(history, x_col="DateTime", y_col1=y_axis_1, y_col2=y_axis_2)
