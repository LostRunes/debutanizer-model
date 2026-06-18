"""
debutanizer_dashboard/pages/model_analysis.py
============================================
New dashboard tab to display before vs. after model performance,
model residual analysis, and analyzer timeline tracking using pre-generated predictions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from nicegui import ui

from services.state_service import state
from components.charts import build_before_after_fig, build_residual_fig, build_analyzer_timeline_fig

# Resolve prediction data path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICTIONS_FILE = os.path.join(BASE_DIR, "models", "test_predictions.parquet")

def build_model_analysis():
    # Read prediction parquet file
    if not os.path.exists(PREDICTIONS_FILE):
        ui.label(f"Error: Prediction file not found at {PREDICTIONS_FILE}").classes('text-red-500')
        return
        
    df = pd.read_parquet(PREDICTIONS_FILE)
    if df.empty:
        ui.label("Error: Prediction file is empty.").classes('text-red-500')
        return

    # Ensure DateTime column is formatted correctly
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])

    # Mutable state for time window (default to 30 days = 720 rows/hours)
    window_state = {
        "hours": 720
    }

    # Reference variables for the charts and containers
    kpi_container = [None]
    before_after_chart = [None]
    residual_chart = [None]
    timeline_chart = [None]
    date_range_label = [None]

    ui.label('Model Analysis & Performance').classes('text-h4 text-white font-extrabold mb-2')
    ui.label('Evaluate model validation predictions, residual deviations, and GC analyzer uptime.').classes('text-xs text-grey-5 mb-4')

    def get_windowed_data():
        hours = window_state["hours"]
        # Preds file is Block 4 only, so just slice from the end
        if hours >= len(df):
            return df
        return df.iloc[-hours:]

    def update_kpi_cards(subset_df):
        kpi_container[0].clear()
        
        # Calculate Metrics
        actual = subset_df["Total_C4"].values
        pred = subset_df["pred_Total_C4"].values
        naive = subset_df["lag1_Total_C4"].values
        
        # Filter NaNs for calculations just in case
        mask = ~np.isnan(actual) & ~np.isnan(pred) & ~np.isnan(naive)
        if not np.any(mask):
            r2_val, mae_val, naive_mae_val, improvement = 0.0, 0.0, 0.0, 0.0
        else:
            actual_clean = actual[mask]
            pred_clean = pred[mask]
            naive_clean = naive[mask]
            
            r2_val = r2_score(actual_clean, pred_clean)
            mae_val = mean_absolute_error(actual_clean, pred_clean)
            naive_mae_val = mean_absolute_error(actual_clean, naive_clean)
            
            if naive_mae_val > 0:
                improvement = ((naive_mae_val - mae_val) / naive_mae_val) * 100
            else:
                improvement = 0.0

        with kpi_container[0]:
            # R2 Card
            with ui.card().classes('grow bg-zinc-900/40 border border-zinc-800 p-4 rounded-xl shadow-md min-w-[200px]'):
                ui.label("Model R² Score").classes('text-xs font-bold text-zinc-400 uppercase tracking-wider')
                ui.label(f"{r2_val:.3f}").classes('text-3xl font-extrabold text-green-400 mt-1')
                ui.label("Variance explained").classes('text-[10px] text-zinc-500')

            # Model MAE Card
            with ui.card().classes('grow bg-zinc-900/40 border border-zinc-800 p-4 rounded-xl shadow-md min-w-[200px]'):
                ui.label("Model MAE").classes('text-xs font-bold text-zinc-400 uppercase tracking-wider')
                ui.label(f"{mae_val:.4f} wt%").classes('text-3xl font-extrabold text-blue-400 mt-1')
                ui.label("Mean Absolute Error").classes('text-[10px] text-zinc-500')

            # Naive MAE Card
            with ui.card().classes('grow bg-zinc-900/40 border border-zinc-800 p-4 rounded-xl shadow-md min-w-[200px]'):
                ui.label("Baseline MAE").classes('text-xs font-bold text-zinc-400 uppercase tracking-wider')
                ui.label(f"{naive_mae_val:.4f} wt%").classes('text-3xl font-extrabold text-orange-400 mt-1')
                ui.label("Lag-1 persistence baseline").classes('text-[10px] text-zinc-500')

            # Improvement Card
            with ui.card().classes('grow bg-zinc-900/40 border border-zinc-800 p-4 rounded-xl shadow-md min-w-[200px]'):
                ui.label("Error Reduction").classes('text-xs font-bold text-zinc-400 uppercase tracking-wider')
                ui.label(f"{improvement:+.1f}%").classes('text-3xl font-extrabold text-green-400 mt-1')
                ui.label("Compared to baseline").classes('text-[10px] text-zinc-500')

    def update_view():
        subset_df = get_windowed_data()
        
        # Update date range label
        start_t = subset_df["DateTime"].iloc[0].strftime("%Y-%m-%d %H:%M")
        end_t = subset_df["DateTime"].iloc[-1].strftime("%Y-%m-%d %H:%M")
        date_range_label[0].set_text(f"Timeline Span: {start_t} to {end_t} ({len(subset_df)} hours)")
        
        # Update KPIs
        update_kpi_cards(subset_df)
        
        # Update Before/After Plot
        if before_after_chart[0] is not None:
            before_after_chart[0].figure = build_before_after_fig(
                subset_df, "DateTime", "Total_C4", "pred_Total_C4", "lag1_Total_C4", spec_limit=state.config.get("spec_limit_total_c4_wt_pct", 0.50)
            )
            before_after_chart[0].update()

        # Update Residual Plot
        if residual_chart[0] is not None:
            residual_chart[0].figure = build_residual_fig(
                subset_df, "DateTime", "Total_C4", "pred_Total_C4"
            )
            residual_chart[0].update()

        # Update Timeline Plot
        if timeline_chart[0] is not None:
            timeline_chart[0].figure = build_analyzer_timeline_fig(
                subset_df, "DateTime", "C4H8_Bottom_stuck"
            )
            timeline_chart[0].update()

    # Time Window selector row
    with ui.row().classes('w-full items-center justify-between mb-4 bg-zinc-900/40 p-4 rounded-xl border border-zinc-800'):
        with ui.row().classes('items-center gap-2'):
            ui.label("Time Window:").classes('text-xs font-bold text-white')
            
            def set_window(hours):
                window_state["hours"] = hours
                update_view()

            with ui.row().classes('gap-1'):
                ui.button("7d", on_click=lambda: set_window(168)).props('size=sm outline').classes('text-xs')
                ui.button("30d", on_click=lambda: set_window(720)).props('size=sm outline').classes('text-xs')
                ui.button("All Block 4", on_click=lambda: set_window(999999)).props('size=sm outline').classes('text-xs')
                
        date_range_label[0] = ui.label("Timeline Span: --").classes('text-xs text-zinc-400 font-semibold')

    # Metric KPI Row
    kpi_container[0] = ui.row().classes('w-full gap-4 items-stretch mb-6')

    # Row 1: Before/After & Residual Side by Side
    with ui.row().classes('w-full gap-4 items-stretch mb-6 flex-wrap lg:flex-nowrap'):
        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md w-full lg:w-1/2'):
            ui.label("Actual vs. Model Prediction vs. Naive Baseline").classes('text-white font-bold text-sm mb-4 uppercase tracking-wider')
            init_df = get_windowed_data()
            fig_ba = build_before_after_fig(
                init_df, "DateTime", "Total_C4", "pred_Total_C4", "lag1_Total_C4", spec_limit=state.config.get("spec_limit_total_c4_wt_pct", 0.50)
            )
            before_after_chart[0] = ui.plotly(fig_ba).classes('w-full h-96')

        with ui.card().classes('grow bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md w-full lg:w-1/2'):
            ui.label("Model Prediction Residuals (wt%)").classes('text-white font-bold text-sm mb-4 uppercase tracking-wider')
            init_df = get_windowed_data()
            fig_res = build_residual_fig(init_df, "DateTime", "Total_C4", "pred_Total_C4")
            residual_chart[0] = ui.plotly(fig_res).classes('w-full h-96')

    # Row 2: GC Analyzer stuck status timeline
    with ui.card().classes('w-full bg-zinc-950/40 border border-zinc-800 rounded-xl p-4 shadow-md mb-6'):
        ui.label("Analyzer Stuck Status Timeline (1 = Online, 0 = Offline/Stuck)").classes('text-white font-bold text-sm mb-4 uppercase tracking-wider')
        init_df = get_windowed_data()
        fig_time = build_analyzer_timeline_fig(init_df, "DateTime", "C4H8_Bottom_stuck")
        timeline_chart[0] = ui.plotly(fig_time).classes('w-full h-48')

    # Perform initial fill
    update_view()
