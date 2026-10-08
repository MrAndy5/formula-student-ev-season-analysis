"""
Formula Student EV - Reporting and Visualization Package.

Modules:
- figures: Generates high-resolution engineering figures.
- report_builder: Compiles Markdown and standalone HTML engineering reports.
"""

from .figures import FigureGenerator
from .report_builder import ReportBuilder

__all__ = [
    "FigureGenerator",
    "ReportBuilder",
]
