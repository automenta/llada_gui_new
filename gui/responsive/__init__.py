"""
Responsive UI components for the LLaDA GUI.

This package provides components and utilities for improving UI responsiveness
and ensuring that long-running operations are properly managed.
"""

from .background_task import BackgroundTask, BackgroundTaskManager
from .progress_indicator import ProgressIndicator, TaskProgressDialog
from .responsive_ui import make_responsive, prevent_ui_freeze
