"""
debutanizer_dashboard/components/cards.py
=========================================
Reusable Quasar/Tailwind styled cards for KPI indicators and Safety Confidence ratings.
"""

from nicegui import ui

def kpi_card(title: str, value: str, icon: str = "analytics", color: str = "primary", border_color: str = "zinc-800", status_color: str = "default"):
    """
    Renders a glassmorphic KPI card with a modern dark theme style.
    """
    # Map status colors
    val_class = "text-white"
    border_class = f"border-{border_color}"
    if status_color == "green":
        val_class = "text-green-400"
        border_class = "border-green-800/40 bg-green-950/10"
    elif status_color == "yellow":
        val_class = "text-yellow-400"
        border_class = "border-yellow-800/40 bg-yellow-950/10"
    elif status_color == "red":
        val_class = "text-red-400"
        border_class = "border-red-800/40 bg-red-950/10"
        
    with ui.card().classes(f'w-64 bg-opacity-40 backdrop-blur-md bg-zinc-900 border {border_class} rounded-xl p-4 shadow-lg hover:scale-105 transition-all duration-300'):
        with ui.row().classes('w-full justify-between items-center no-wrap'):
            ui.label(title).classes('text-grey-5 font-semibold text-xs tracking-wider uppercase')
            ui.icon(icon).classes(f'text-lg text-{color}')
        ui.label(value).classes(f'text-3xl font-extrabold {val_class} mt-1')

def safety_confidence_card(level: str, details: str = ""):
    """
    Renders the safety confidence rating using a colored badge based on operating distance from limits.
    """
    if level == "HIGH":
        bg_class = "bg-green-950/40 border-green-500/50 text-green-400"
        dot_color = "bg-green-400"
    elif level == "MEDIUM":
        bg_class = "bg-yellow-950/40 border-yellow-500/50 text-yellow-400"
        dot_color = "bg-yellow-400"
    else:
        bg_class = "bg-red-950/40 border-red-500/50 text-red-400"
        dot_color = "bg-red-400"
        
    with ui.card().classes(f'w-full {bg_class} border rounded-xl p-4 shadow-md'):
        with ui.row().classes('items-center gap-2'):
            ui.element('span').classes(f'w-2.5 h-2.5 rounded-full {dot_color} animate-pulse')
            ui.label('SAFETY CONFIDENCE:').classes('font-bold text-xs uppercase tracking-wider')
            ui.label(level).classes('font-extrabold text-sm uppercase')
        if details:
            ui.label(details).classes('text-xs text-grey-5 mt-1')
