# pages/__init__.py
from .generation import render_generation_page
from .visualization import render_visualization_page
from .analysis import render_analysis_page
from .comparison import render_comparison_page

__all__ = [
    'render_generation_page',
    'render_visualization_page',
    'render_analysis_page',
    'render_comparison_page'
]
