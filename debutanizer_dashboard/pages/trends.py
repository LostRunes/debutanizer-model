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

    history = state.get_current_history()
    if history is None or history.empty:
        ui.label("Error: No historian history loaded.").classes('text-red-500')
        return

    # --- Mutable state containers (avoid global vars & broken closure captures) ---
    # chart_ref[0] will hold the ui.plotly element once created below
    chart_ref = [None]
    # axis_state dict is shared by all closures in this call frame
    axis_state = {"y1": "Total_C4", "y2": "Reboiling_Steam_Flow"}

    ui.label('Historical Trends').classes('text-h4 text-white font-extrabold mb-2')
    ui.label(
        'Observe the dynamic relationship between process inputs and C4 slippage compositions.'
    ).classes('text-xs text-grey-5 mb-4')

    def update_plot():
        """
        Re-builds the Plotly figure and pushes it into the existing chart element
        without rebuilding the DOM.  Works because chart_ref[0] is mutated (not
        rebound) after the plot widget is instantiated below.
        """
        if chart_ref[0] is not None:
            new_fig = build_trend_fig(history, "DateTime", axis_state["y1"], axis_state["y2"])
            chart_ref[0].figure = new_fig
            chart_ref[0].update()

    # --- Selectors row ---
    with ui.row().classes(
        'w-full items-center flex-wrap gap-4 '
        'bg-zinc-900/40 p-4 rounded-xl border border-zinc-800 backdrop-blur-md mb-4'
    ):
        # Y Axis 1 selector — uses on_change= (idiomatic NiceGUI)
        with ui.row().classes('items-center gap-2'):
            ui.label("Y-Axis 1 (Green):").classes('text-xs font-bold text-white')

            def _on_y1_change(e):
                axis_state["y1"] = e.value
                update_plot()

            ui.select(
                options=COLUMN_OPTIONS,
                value=axis_state["y1"],
                on_change=_on_y1_change
            ).classes('w-56 text-xs text-white').props('dark')

        # Y Axis 2 selector
        with ui.row().classes('items-center gap-2'):
            ui.label("Y-Axis 2 (Blue Dash):").classes('text-xs font-bold text-white')

            def _on_y2_change(e):
                axis_state["y2"] = e.value
                update_plot()

            ui.select(
                options=COLUMN_OPTIONS,
                value=axis_state["y2"],
                on_change=_on_y2_change
            ).classes('w-56 text-xs text-white').props('dark')

        # Manual fallback button — always triggers update regardless of event state
        ui.button("Update Plot", on_click=update_plot).props('icon=refresh color=primary').classes('text-xs px-4 py-2')

    # --- Chart area ---
    with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md'):
        fig = build_trend_fig(history, "DateTime", axis_state["y1"], axis_state["y2"])
        # IMPORTANT: assign into chart_ref[0] so closures above can call .figure= / .update()
        chart_ref[0] = ui.plotly(fig).classes('w-full h-96')
