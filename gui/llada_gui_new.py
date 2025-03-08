# -*- coding: utf-8 -*-

"""
LLaDA GUI - New OpenGL Visualization-Centric GUI for LLaDA.
"""

import sys

import torch
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QShortcut, QKeySequence, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox,
    QCheckBox, QProgressBar, QSplitter, QMessageBox, QGridLayout,
    QScrollArea, QDoubleSpinBox, QTabWidget, QRadioButton, QButtonGroup,
    QSizePolicy, QStatusBar, QOpenGLWidget, QVBoxLayout
)
from PyQt6.QtOpenGL import QOpenGLVersionProfile, QSurfaceFormat

from OpenGL.GL import *  # pylint: disable=W0614,W0611

# Import default parameters
from core.config import DEFAULT_GENERATION_PARAMS as DEFAULT_PARAMS


class GLVisualizationWidget(QOpenGLWidget):
    """OpenGL widget for visualization."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.object = 0

    def initializeGL(self):
        """Initialize OpenGL context and settings."""
        version_profile = QOpenGLVersionProfile()
        version_profile.setVersion(4, 1)  # Request OpenGL 4.1 - adjust if needed
        format_ = QSurfaceFormat()
        format_.setVersion(4, 1)
        format_.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(format_)

        glClearColor(0.1, 0.1, 0.2, 1.0) # Dark background

    def paintGL(self):
        """Paint the OpenGL scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Simple colored square for testing
        glBegin(GL_QUADS)
        glColor3f(1.0, 1.0, 0.0) # Yellow color
        glVertex2f(-0.5, -0.5)
        glVertex2f(0.5, -0.5)
        glVertex2f(0.5, 0.5)
        glVertex2f(-0.5, 0.5)
        glEnd()

    def resizeGL(self, width, height):
        """Handle viewport resizing."""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1) # Orthographic projection
        glMatrixMode(GL_MODELVIEW)


class LLaDAGUINew(QMainWindow):
    """New OpenGL Visualization-Centric GUI for LLaDA application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLaDA GUI - OpenGL Viz - Prototype")
        self.resize(1200, 900)  # Slightly larger initial size

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # 1. Prompt Input Area (Top)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        main_layout.addWidget(self.prompt_input)

        # 2. Visualization and Sidebar Area (Center - Horizontal Layout)
        viz_sidebar_layout = QHBoxLayout()
        main_layout.addLayout(viz_sidebar_layout)

        # 2.1. OpenGL Visualization Widget (Left)
        self.opengl_viz_widget = GLVisualizationWidget()  # Use the new OpenGL widget
        viz_sidebar_layout.addWidget(self.opengl_viz_widget)

        # 2.2. Sidebar (Right - Scrollable)
        self.sidebar_scroll_area = QScrollArea()
        self.sidebar_scroll_area.setWidgetResizable(True)  # Important for scroll area to work correctly
        self.sidebar_widget = QWidget()  # Widget to hold sidebar content
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)  # Layout for sidebar content
        self.sidebar_scroll_area.setWidget(self.sidebar_widget)  # Set widget to scroll area
        viz_sidebar_layout.addWidget(self.sidebar_scroll_area)

        # Sidebar Sections (Placeholders for now)
        self.add_sidebar_sections()

        # 3. Status Bar (Bottom)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")  # Initial status message

        # Set the central widget
        self.setCentralWidget(main_widget)

    def add_sidebar_sections(self):
        """Adds placeholder sections to the sidebar."""

        # Generation Settings ⚙️
        generation_group = QGroupBox("⚙️ Generation Settings")
        generation_layout = QGridLayout() # Use GridLayout for better organization

        # Generation Length
        self.gen_length_spin = QSpinBox()
        self.gen_length_spin.setRange(16, 512)
        self.gen_length_spin.setValue(DEFAULT_PARAMS['gen_length'])
        self.gen_length_spin.setSingleStep(16)
        self.gen_length_spin.valueChanged.connect(lambda value: print(f"Generation Length changed: {value}")) # Placeholder
        generation_layout.addWidget(QLabel("Length:"), 0, 0)
        generation_layout.addWidget(self.gen_length_spin, 0, 1)

        # Sampling Steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(16, 512)
        self.steps_spin.setValue(DEFAULT_PARAMS['steps'])
        self.steps_spin.setSingleStep(16)
        self.steps_spin.valueChanged.connect(lambda value: print(f"Sampling Steps changed: {value}")) # Placeholder
        generation_layout.addWidget(QLabel("Steps:"), 1, 0)
        generation_layout.addWidget(self.steps_spin, 1, 1)

        # Block Length
        self.block_length_spin = QSpinBox()
        self.block_length_spin.setRange(16, 256)
        self.block_length_spin.setValue(DEFAULT_PARAMS['block_length'])
        self.block_length_spin.setSingleStep(16)
        self.block_length_spin.valueChanged.connect(lambda value: print(f"Block Length changed: {value}")) # Placeholder
        generation_layout.addWidget(QLabel("Block Size:"), 2, 0)
        generation_layout.addWidget(self.block_length_spin, 2, 1)

        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0, 2)
        self.temperature_spin.setValue(DEFAULT_PARAMS['temperature'])
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.valueChanged.connect(lambda value: print(f"Temperature changed: {value}")) # Placeholder
        generation_layout.addWidget(QLabel("Temperature:"), 3, 0)
        generation_layout.addWidget(self.temperature_spin, 3, 1)

        # CFG Scale
        self.cfg_scale_spin = QDoubleSpinBox()
        self.cfg_scale_spin.setRange(0, 5)
        self.cfg_scale_spin.setValue(DEFAULT_PARAMS['cfg_scale'])
        self.cfg_scale_spin.setSingleStep(0.1)
        self.cfg_scale_spin.valueChanged.connect(lambda value: print(f"CFG Scale changed: {value}")) # Placeholder
        generation_layout.addWidget(QLabel("CFG Scale:"), 4, 0)
        generation_layout.addWidget(self.cfg_scale_spin, 4, 1)

        # Remasking Strategy
        self.remasking_combo = QComboBox()
        self.remasking_combo.addItems(["low_confidence", "random"])
        self.remasking_combo.setCurrentText(DEFAULT_PARAMS['remasking'])
        self.remasking_combo.currentTextChanged.connect(lambda text: print(f"Remasking Strategy changed: {text}")) # Placeholder
        generation_layout.addWidget(QLabel("Remasking:"), 5, 0)
        generation_layout.addWidget(self.remasking_combo, 5, 1)

        generation_group.setLayout(generation_layout)
        self.sidebar_layout.addWidget(generation_group)

        # Model & Hardware 🧠
        model_group = QGroupBox("🧠 Model & Hardware")
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Model/Hardware Options Here"))  # Placeholder
        model_group.setLayout(model_layout)
        self.sidebar_layout.addWidget(model_group)

        # Memory Integration 💾
        memory_group = QGroupBox("💾 Memory Integration")
        memory_layout = QVBoxLayout()
        memory_layout.addWidget(QLabel("Memory Options Here"))  # Placeholder
        memory_group.setLayout(memory_layout)
        self.sidebar_layout.addWidget(memory_group)

        # Realtime Statistics 📊
        stats_group = QGroupBox("📊 Realtime Statistics")
        stats_layout = QVBoxLayout()
        stats_layout.addWidget(QLabel("Statistics Display Here"))  # Placeholder
        stats_group.setLayout(stats_layout)
        self.sidebar_layout.addWidget(stats_group)

        # Visualization Settings 👁️
        viz_settings_group = QGroupBox("👁️ Visualization Settings")
        viz_settings_layout = QVBoxLayout()
        viz_settings_layout.addWidget(QLabel("Visualization Options Here")) # Placeholder
        viz_settings_group.setLayout(viz_settings_layout)
        self.sidebar_layout.addWidget(viz_settings_group)

        # Add stretch to bottom to push groups to the top
        self.sidebar_layout.addStretch(1)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = LLaDAGUINew()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
