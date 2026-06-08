"""
debutanizer_dashboard/components/charts.py
==========================================
Plotly chart components for trends, safety limits gauges, and feature importances.
"""

import plotly.graph_objects as go
from nicegui import ui

def create_trend_chart(history_df, x_col="DateTime", y_col1="Total_C4", y_col2="Reboiling_Steam_Flow"):
    """
    Renders a Plotly dual-axis line trend chart.
    """
    if history_df is None or history_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No history data loaded", showarrow=False, font=dict(color="white"))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return ui.plotly(fig).classes('w-full h-80')
        
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
