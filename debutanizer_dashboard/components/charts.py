"""
debutanizer_dashboard/components/charts.py
==========================================
Plotly chart components for trends, safety limits gauges, and feature importances.
"""

import plotly.graph_objects as go
from nicegui import ui

def build_trend_fig(history_df, x_col="DateTime", y_col1="Total_C4", y_col2="Reboiling_Steam_Flow"):
    """
    Builds the Plotly Figure object for the trend chart.
    """
    if history_df is None or history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No history data loaded", showarrow=False, font=dict(color="white"))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
        
    fig = go.Figure()
    
    # Trace 1: Primary Y-axis
    fig.add_trace(go.Scatter(
        x=history_df[x_col],
        y=history_df[y_col1],
        name=y_col1.replace("_", " "),
        line=dict(color="#00E676", width=2),
        mode='lines'
    ))
    
    # Trace 2: Secondary Y-axis if different column selected
    if y_col1 != y_col2:
        fig.add_trace(go.Scatter(
            x=history_df[x_col],
            y=history_df[y_col2],
            name=y_col2.replace("_", " "),
            line=dict(color="#42A5F5", width=2, dash='dash'),
            yaxis="y2"
        ))
        layout_y2 = dict(
            title=dict(text=y_col2.replace("_", " "), font=dict(color="#42A5F5")),
            tickfont=dict(color="#42A5F5"),
            overlaying="y",
            side="right",
            gridcolor="rgba(80, 80, 80, 0.15)"
        )
    else:
        layout_y2 = None
        
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(gridcolor="rgba(80, 80, 80, 0.15)"),
        yaxis=dict(
            title=dict(text=y_col1.replace("_", " "), font=dict(color="#00E676")),
            tickfont=dict(color="#00E676"),
            gridcolor="rgba(80, 80, 80, 0.15)"
        ),
        yaxis2=layout_y2,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x"
    )
    return fig

def create_trend_chart(history_df, x_col="DateTime", y_col1="Total_C4", y_col2="Reboiling_Steam_Flow"):
    """
    Renders a Plotly dual-axis line trend chart.
    """
    fig = build_trend_fig(history_df, x_col, y_col1, y_col2)
    return ui.plotly(fig).classes('w-full h-80')

def create_feature_importance_chart(imp_df, title="Feature Importances"):
    """
    Renders a horizontal bar chart of feature importances.
    """
    fig = go.Figure()
    
    # Take top 10 for clean look
    top_df = imp_df.head(10).iloc[::-1]
    
    fig.add_trace(go.Bar(
        x=top_df["Importance"],
        y=[f.replace("_", " ") for f in top_df["Feature"]],
        orientation='h',
        marker=dict(
            color='rgba(66, 165, 245, 0.8)',
            line=dict(color='rgba(66, 165, 245, 1.0)', width=1)
        )
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=40, t=40, b=40),
        xaxis=dict(gridcolor="rgba(80, 80, 80, 0.15)"),
        yaxis=dict(gridcolor="rgba(80, 80, 80, 0.15)"),
    )
    
    return ui.plotly(fig).classes('w-full h-80')

def create_safety_gauge(value, limit, mae, name="Bottom Temp", unit="C"):
    """
    Renders a horizontal indicator bar showing current value, MAE buffer, and limit.
    """
    # Percentage distance
    total_range = limit * 1.1 if limit > 0 else 100
    val_pct = min(100, (value / total_range) * 100)
    mae_pct = min(100, ((value + mae) / total_range) * 100)
    limit_pct = (limit / total_range) * 100
    
    color = "green"
    if value + mae >= limit:
        color = "red"
    elif value + mae >= limit * 0.95:
        color = "yellow"
        
    # CSS-based simple progress gauge using NiceGUI raw HTML elements
    with ui.column().classes('w-full gap-1 p-2 bg-zinc-900/30 rounded-lg border border-zinc-800'):
        with ui.row().classes('w-full justify-between text-xs text-grey-5'):
            ui.label(f"{name} ({unit})")
            ui.label(f"Limit: {limit:.2f} {unit}")
        with ui.row().classes('w-full items-center gap-2 no-wrap'):
            # Current value and mae range
            with ui.element('div').classes('grow bg-zinc-800 h-2.5 rounded-full relative overflow-hidden'):
                # MAE range (light color)
                ui.element('div').classes(f'absolute top-0 left-0 h-full bg-{color}-800/40').style(f'width: {mae_pct}%')
                # Actual value (dark color)
                ui.element('div').classes(f'absolute top-0 left-0 h-full bg-{color}-500').style(f'width: {val_pct}%')
                # Safety Limit line indicator
                ui.element('div').classes('absolute top-0 h-full w-0.5 bg-red-600').style(f'left: {limit_pct}%')
            ui.label(f"{value:.2f} +/- {mae:.2f}").classes('text-xs font-bold text-white shrink-0')

def create_before_after_chart(history_df, optimized_c4_list, spec_limit=0.50):
    """
    Renders a Plotly comparison chart showing Actual C4 vs Optimized C4 over the 24h history.
    """
    fig = go.Figure()
    
    # 1. Actual C4 trace (historian / analyzer before)
    fig.add_trace(go.Scatter(
        x=history_df["DateTime"],
        y=history_df["Total_C4"],
        name="Actual C4 Slippage (Before)",
        line=dict(color="#FF9100", width=2.5), # Rich amber/orange
        mode='lines+markers',
        marker=dict(size=5)
    ))
    
    # 2. Optimized C4 trace (after applying optimizer model)
    fig.add_trace(go.Scatter(
        x=history_df["DateTime"],
        y=optimized_c4_list,
        name="Optimized C4 Slippage (After)",
        line=dict(color="#00E676", width=2.5), # Rich green
        mode='lines+markers',
        marker=dict(size=5)
    ))
    
    # 3. Spec limit reference line
    fig.add_trace(go.Scatter(
        x=history_df["DateTime"],
        y=[spec_limit] * len(history_df),
        name="Spec Limit (0.50 wt%)",
        line=dict(color="#FF1744", width=1.5, dash='dash'), # Bright red dashed
        mode='lines'
    ))
    
    fig.update_layout(
        title=dict(
            text="Optimization Impact: Actual vs. Optimized C4 Slippage (24-Hour Timeline)",
            font=dict(size=14, color="white", family="Outfit")
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=40, t=50, b=40),
        xaxis=dict(
            title=dict(text="DateTime", font=dict(color="grey")),
            gridcolor="rgba(80, 80, 80, 0.15)"
        ),
        yaxis=dict(
            title=dict(text="Total C4 Slippage (wt%)", font=dict(color="white")),
            gridcolor="rgba(80, 80, 80, 0.15)",
            zeroline=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return ui.plotly(fig).classes('w-full h-80')

def build_before_after_fig(df, time_col, actual_col, pred_col, naive_col, spec_limit=0.50):
    """
    Builds the Plotly Figure comparison chart showing Actual C4, Model Predicted C4, and Naive Baseline.
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    
    # 1. Actual (True value from analyzer)
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[actual_col],
        name="Actual (True)",
        line=dict(color="#FFFFFF", width=2),
        mode='lines'
    ))
    
    # 2. Predicted (Model Prediction)
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[pred_col],
        name="Model Prediction",
        line=dict(color="#00E676", width=2),
        mode='lines'
    ))
    
    # 3. Naive Baseline (lag-1 target)
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[naive_col],
        name="Naive Baseline",
        line=dict(color="#FF1744", width=1.5, dash='dash'),
        mode='lines'
    ))
    
    # 4. Spec Limit Line
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=[spec_limit] * len(df),
        name="Spec Limit",
        line=dict(color="#FF9100", width=1.2, dash='dot'),
        mode='lines'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=40, t=20, b=40),
        xaxis=dict(gridcolor="rgba(80, 80, 80, 0.15)"),
        yaxis=dict(
            title=dict(text="Total C4 wt%", font=dict(color="white")),
            gridcolor="rgba(80, 80, 80, 0.15)"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig

def build_residual_fig(df, time_col, actual_col, pred_col):
    """
    Builds a bar chart showing the model residual (predicted - actual) over time.
    """
    import plotly.graph_objects as go
    import numpy as np
    
    residuals = df[pred_col] - df[actual_col]
    
    # Color code bars based on residual size
    colors = []
    for r in residuals:
        abs_r = abs(r)
        if abs_r <= 0.05:
            colors.append("#00E676")  # Green for highly accurate
        elif abs_r <= 0.15:
            colors.append("#FFD600")  # Yellow for medium deviation
        else:
            colors.append("#FF1744")  # Red for high deviation
            
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[time_col],
        y=residuals,
        name="Residual (Pred - Actual)",
        marker_color=colors
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=40, t=20, b=40),
        xaxis=dict(gridcolor="rgba(80, 80, 80, 0.15)"),
        yaxis=dict(
            title=dict(text="Residual (wt%)", font=dict(color="white")),
            gridcolor="rgba(80, 80, 80, 0.15)"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x"
    )
    return fig

def build_analyzer_timeline_fig(df, time_col, stuck_col):
    """
    Builds a timeline diagram showing periods of analyzer online/stuck status.
    """
    import plotly.graph_objects as go
    
    # Map stuck status to numbers for visualization: 1 = ONLINE (not stuck), 0 = STUCK (offline)
    status_numeric = (~df[stuck_col]).astype(int)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=status_numeric,
        mode="lines",
        line=dict(shape="hv", width=2.5, color="#29B6F6"),
        fill="tozeroy",
        fillcolor="rgba(41, 182, 246, 0.15)",
        name="Analyzer Status"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=40, t=20, b=40),
        xaxis=dict(gridcolor="rgba(80, 80, 80, 0.15)"),
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=["OFFLINE (Stuck)", "ONLINE"],
            gridcolor="rgba(80, 80, 80, 0.15)",
            range=[-0.2, 1.2]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
