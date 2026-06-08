"""
debutanizer_dashboard/app.py
============================
Main entry point for the Debutanizer Column AI Platform NiceGUI web dashboard.
Manages global page layouts, reactive updates via ui.refreshable, and styles.
"""

import sys
import os

# Set up paths to make imports robust to launching directory
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dashboard_dir)
sys.path.append(os.path.dirname(dashboard_dir))

from nicegui import ui

# Import page builder functions
from pages.overview import build_overview
from pages.soft_sensor import build_soft_sensor
from pages.optimizer import build_optimizer
from pages.trends import build_trends
from pages.diagnostics import build_diagnostics
from pages.settings import build_settings

# NiceGUI Styling Configuration
ui.colors(
    primary='#0D47A1',
    secondary='#1565C0',
    accent='#42A5F5',
    dark='#121212'
)

# Active Page state variable
active_page = "overview"

# Custom CSS styling definition
CSS_STYLING = """
body {
    font-family: 'Inter', 'Outfit', -apple-system, sans-serif;
    background-color: #0c0c0e;
}
.q-drawer {
    background-color: #121214 !important;
    border-right: 1px solid #1f1f23 !important;
}
.q-btn {
    border-radius: 8px !important;
    text-transform: none !important;
    font-weight: 600 !important;
}
.nav-active {
    background: rgba(13, 71, 161, 0.2) !important;
    color: #42A5F5 !important;
    border-left: 3px solid #0D47A1 !important;
}
.glass-panel {
    background: rgba(20, 20, 25, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
}
"""

@ui.refreshable
def render_content():
    """
    Renders the active subpage content. Called reactively when navigation tab changes.
    """
    if active_page == "overview":
        build_overview(render_content.refresh)
    elif active_page == "soft_sensor":
        build_soft_sensor()
    elif active_page == "optimizer":
        build_optimizer(render_content.refresh)
    elif active_page == "trends":
        build_trends(render_content.refresh)
    elif active_page == "diagnostics":
        build_diagnostics()
    elif active_page == "settings":
        build_settings(render_content.refresh)

# Main Application Layout
ui.add_head_html(f"<style>{CSS_STYLING}</style>")
ui.add_head_html("<link href='https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;700&display=swap' rel='stylesheet'>")

# Left sidebar navigation drawer
with ui.left_drawer(value=True, fixed=True).classes('w-64 p-4 gap-6 column no-wrap justify-between shadow-2xl'):
    # Brand logo block
    with ui.column().classes('w-full items-center gap-1'):
        ui.icon("science", size="2.5rem").classes('text-blue-500 mt-2')
        ui.label("IOCL DEBUTANIZER").classes('text-md font-extrabold text-white tracking-widest mt-2 uppercase')
        ui.label("AI Control Platform").classes('text-[10px] text-grey-5 font-bold tracking-wider uppercase')
        
    # Navigation buttons
    with ui.column().classes('w-full grow gap-2 justify-center'):
        def set_active_page(page_name):
            global active_page
            active_page = page_name
            render_content.refresh()
            # Force active button highlight updates via full navigation drawer reload
            render_navigation.refresh()
            
        @ui.refreshable
        def render_navigation():
            ui.button('Overview', icon="dashboard", 
                      on_click=lambda: set_active_page("overview")).classes(f'w-full justify-start text-sm py-2.5 {"nav-active" if active_page == "overview" else "text-grey-4"}')
            ui.button('Soft Sensor', icon="analytics", 
                      on_click=lambda: set_active_page("soft_sensor")).classes(f'w-full justify-start text-sm py-2.5 {"nav-active" if active_page == "soft_sensor" else "text-grey-4"}')
            ui.button('Advisory Optimizer', icon="insights", 
                      on_click=lambda: set_active_page("optimizer")).classes(f'w-full justify-start text-sm py-2.5 {"nav-active" if active_page == "optimizer" else "text-grey-4"}')
            ui.button('Historical Trends', icon="show_chart", 
                      on_click=lambda: set_active_page("trends")).classes(f'w-full justify-start text-sm py-2.5 {"nav-active" if active_page == "trends" else "text-grey-4"}')
            ui.button('Diagnostics', icon="troubleshoot", 
                      on_click=lambda: set_active_page("diagnostics")).classes(f'w-full justify-start text-sm py-2.5 {"nav-active" if active_page == "diagnostics" else "text-grey-4"}')
            ui.button('Settings', icon="settings", 
                      on_click=lambda: set_active_page("settings")).classes(f'w-full justify-start text-sm py-2.5 {"nav-active" if active_page == "settings" else "text-grey-4"}')

        render_navigation()
        
    # Footer block
    with ui.column().classes('w-full items-center text-[10px] text-grey-6 border-t border-zinc-800/80 pt-4 mb-2'):
        ui.label("Platform version: v2.1-Advisory")
        ui.label("Status: Connected to DCS")

# Main Content Area
with ui.column().classes('grow p-6 w-full max-w-7xl mx-auto gap-4 bg-transparent'):
    render_content()

# Run server (disabled reload for direct integration inside production environments)
ui.run(
    title="IOCL Debutanizer AI Dashboard",
    reload=False,
    port=8080
)
