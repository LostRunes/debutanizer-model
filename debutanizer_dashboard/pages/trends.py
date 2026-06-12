"""
debutanizer_dashboard/pages/trends.py
=====================================
Renders the Historical Trends tab. Allows the user to select process variables
and displays an interactive dual-axis line plot.

FIXES APPLIED:
- Replaced broken Python closure pattern (chart=None captured before assignment)
  with a mutable list container `chart_ref` so the inner update_plot() always
  sees the live ui.plotly element after it is created.
- Removed module-level globals; axis state now lives in a per-call dict so
  multiple browser tabs don't corrupt each other.
- Changed select event binding from .on('change', cb) to idiomatic NiceGUI
  on_change= constructor parameter, which is the reliable way to get value
  updates from ui.select.
- Increased chart height to h-96 for better readability.
"""

from nicegui import ui
from services.state_service import state
from components.charts import build_trend_fig


COLUMN_OPTIONS = {
    "Total_C4": "Total C4 Slippage (wt%)",
    "C4H8_Bottom": "Butene C4H8 (wt%)",
    "C4H6_Bottom": "Butadiene C4H6 (wt%)",
    "Reboiling_Steam_Flow": "Steam Flow (TPH)",
    "Reflux_Flow": "Reflux Flow (TPH)",
    "Column_Bottom_Temp": "Column Bottom Temp (°C)",
    "Control_Tray_Temp": "Control Tray Temp (°C)",
    "Column_Top_Pressure": "Top Pressure (bar)",
    "Feed_Flow": "Feed Flow (TPH)",
}


def build_trends(on_state_change_callback):
    """
    Builds the trends tab content.
    """
    # y1, y2 selection and hours window state container (mutable dictionary for closure safety)
    window_state = {
        "hours": 720,  # default 30 days
        "y1": "Total_C4",
        "y2": "Reboiling_Steam_Flow"
    }

    # Fetch initial history data
    history = state.get_trend_history(window_hours=window_state["hours"])
    if history is None or history.empty:
        ui.label("Error: No historian history loaded.").classes('text-red-500')
        return

    # Mutable container for the chart element reference
    chart_ref = [None]
    
    # UI Elements references that need updating dynamically
    date_range_label = [None]
    warning_label = [None]

    ui.label('Historical Trends').classes('text-h4 text-white font-extrabold mb-2')
    ui.label(
        'Observe the dynamic relationship between process inputs and C4 slippage compositions.'
    ).classes('text-xs text-grey-5 mb-4')

    def get_actual_history():
        # Clamps window to block start by leveraging state_service logic
        return state.get_trend_history(window_hours=window_state["hours"])

    def update_warnings_and_dates(df):
        if df is not None and not df.empty:
            start_time = df["DateTime"].iloc[0]
            end_time = df["DateTime"].iloc[-1]
            date_range_label[0].set_text(f"Range: {start_time} to {end_time} ({len(df)} rows)")
            
            # Check NaN percentage for y1 and y2
            warnings = []
            for y_key, y_col in [("Y1", window_state["y1"]), ("Y2", window_state["y2"])]:
                if y_col in df.columns:
                    n_null = df[y_col].isna().sum()
                    pct_null = (n_null / len(df)) * 100
                    if pct_null > 30:
                        warnings.append(f"⚠️ {y_key} ({COLUMN_OPTIONS[y_col]}) has {pct_null:.1f}% missing/null values in this window.")
            
            if warnings:
                warning_label[0].set_text("\n".join(warnings))
                warning_label[0].classes('text-yellow-500 text-xs mt-2', remove='hidden')
            else:
                warning_label[0].set_text("")
                warning_label[0].classes('hidden')

    def update_plot():
        df = get_actual_history()
        if chart_ref[0] is not None and df is not None and not df.empty:
            new_fig = build_trend_fig(df, "DateTime", window_state["y1"], window_state["y2"])
            chart_ref[0].figure = new_fig
            chart_ref[0].update()
            update_warnings_and_dates(df)

    # --- Selectors row ---
    with ui.row().classes(
        'w-full items-center flex-wrap gap-4 '
        'bg-zinc-900/40 p-4 rounded-xl border border-zinc-800 backdrop-blur-md mb-2'
    ):
        # Time Window Selector
        with ui.row().classes('items-center gap-2 mr-4'):
            ui.label("Time Window:").classes('text-xs font-bold text-white')
            
            def set_window(hours):
                window_state["hours"] = hours
                update_plot()

            with ui.row().classes('gap-1'):
                ui.button("24h", on_click=lambda: set_window(24)).props('size=sm outline').classes('text-xs')
                ui.button("7d", on_click=lambda: set_window(168)).props('size=sm outline').classes('text-xs')
                ui.button("30d", on_click=lambda: set_window(720)).props('size=sm outline').classes('text-xs')
                ui.button("All Block", on_click=lambda: set_window(999999)).props('size=sm outline').classes('text-xs')

        # Y Axis 1 selector
        with ui.row().classes('items-center gap-2'):
            ui.label("Y-Axis 1 (Green):").classes('text-xs font-bold text-white')

            def _on_y1_change(e):
                window_state["y1"] = e.value
                update_plot()

            ui.select(
                options=COLUMN_OPTIONS,
                value=window_state["y1"],
                on_change=_on_y1_change
            ).classes('w-56 text-xs text-white').props('dark')

        # Y Axis 2 selector
        with ui.row().classes('items-center gap-2'):
            ui.label("Y-Axis 2 (Blue Dash):").classes('text-xs font-bold text-white')

            def _on_y2_change(e):
                window_state["y2"] = e.value
                update_plot()

            ui.select(
                options=COLUMN_OPTIONS,
                value=window_state["y2"],
                on_change=_on_y2_change
            ).classes('w-56 text-xs text-white').props('dark')

        # Update button
        ui.button("Update Plot", on_click=update_plot).props('icon=refresh color=primary').classes('text-xs px-4 py-2')

    # Date Range Display & Warnings Row
    with ui.row().classes('w-full items-center justify-between mb-4 px-2'):
        start_t = history["DateTime"].iloc[0]
        end_t = history["DateTime"].iloc[-1]
        date_range_label[0] = ui.label(f"Range: {start_t} to {end_t} ({len(history)} rows)").classes('text-xs font-semibold text-zinc-400')
        warning_label[0] = ui.label("").classes('hidden')

    # --- Chart area ---
    with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
        fig = build_trend_fig(history, "DateTime", window_state["y1"], window_state["y2"])
        # Set chart height to h-[500px] as per specification
        chart_ref[0] = ui.plotly(fig).classes('w-full h-[500px]')

    # Run initial warning update
    update_warnings_and_dates(history)
