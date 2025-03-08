# -*- coding: utf-8 -*-

"""
LLaDA GUI - New OpenGL Visualization-Centric GUI for LLaDA.
"""

import sys
import random

import torch
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
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
from OpenGL import GLU  # Import GLU for gluDisk
import numpy as np


# Import our modules with updated paths
from core.config import DEFAULT_GENERATION_PARAMS as DEFAULT_PARAMS
from core.llada_worker import LLaDAWorker
from core.utils import format_error


class GLVisualizationWidget(QOpenGLWidget):
    """OpenGL widget for visualization."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.object = 0
        self.visualization_type = "Token Stream" # Default visualization type
        self.token_stream_data = [] # Placeholder for token stream data
        self.color_scheme = "Cool" # Default color scheme
        self.animation_timer = QTimer(self) # Timer for animation
        self.animation_timer.timeout.connect(self.update) # Trigger repaint on timer
        self.animation_time = 0.0 # Time counter for animation
        self.animation_speed = 0.01 # Animation speed factor - default
        self.animation_timer.start(20) # 20ms interval for ~50fps animation
        self.token_shape = "Circle" # Default token shape
        self.token_size = 0.03 # Default token size
        self.token_spacing = 0.07 # Default token spacing
        self.zoom_level = 1.0 # Default zoom level
        self.pan_x = 0.0 # Default pan X
        self.pan_y = 0.0 # Default pan Y


    def set_visualization_type(self, viz_type):
        """Set the current visualization type."""
        self.visualization_type = viz_type
        self.update() # Trigger repaint

    def set_color_scheme(self, scheme):
        """Set the color scheme for visualizations."""
        self.color_scheme = scheme
        self.update()

    def set_token_stream_data(self, data):
        """Set data for token stream visualization."""
        self.token_stream_data = data
        self.update()

    def set_token_shape(self, shape):
        """Set the shape of tokens in visualizations."""
        self.token_shape = shape
        self.update()

    def set_animation_speed(self, speed):
        """Set the animation speed factor."""
        self.animation_speed = speed
        self.update()

    def set_token_size(self, size):
        """Set the size of tokens in visualizations."""
        self.token_size = size
        self.update()

    def set_token_spacing(self, spacing):
        """Set the spacing between tokens in visualizations."""
        self.token_spacing = spacing
        self.update()

    def set_zoom_level(self, zoom):
        """Set the zoom level for visualization."""
        self.zoom_level = zoom
        self.update()

    def set_pan(self, pan_x, pan_y):
        """Set the pan values for visualization."""
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.update()


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
        glEnable(GL_BLEND) # Enable blending for smooth shapes
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) # Standard alpha blending

        # Setup zoom and pan transformations
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1 * self.zoom_level + self.pan_x, 1 * self.zoom_level + self.pan_x,
                -1 * self.zoom_level + self.pan_y, 1 * self.zoom_level + self.pan_y, -1, 1)
        glMatrixMode(GL_MODELVIEW)


        self.animation_time += self.animation_speed # Increment animation time

        if self.visualization_type == "Token Stream":
            self.draw_token_stream()
        elif self.visualization_type == "Test Square":
            self.draw_test_square() # Default or fallback visualization
        elif self.visualization_type == "Memory Influence Map":
            self.draw_memory_influence_map()
        elif self.visualization_type == "Abstract Token Cloud":
            self.draw_abstract_token_cloud()

        glDisable(GL_BLEND) # Disable blending when done


    def draw_test_square(self):
        """Draw a simple colored square for testing."""
        glBegin(GL_QUADS)
        glColor3f(1.0, 1.0, 0.0) # Yellow color
        glVertex2f(-0.5, -0.5)
        glVertex2f(0.5, -0.5)
        glVertex2f(0.5, 0.5)
        glVertex2f(-0.5, 0.5)
        glEnd()

    def draw_token_stream(self):
        """Draw the Token Stream visualization."""
        num_tokens = 25 # Increased number of tokens for better stream
        spacing = self.token_spacing # Use parameter for spacing
        start_x = - (num_tokens - 1) * spacing / 2 # Center tokens

        for i in range(num_tokens):
            x = start_x + i * spacing
            y_offset = np.sin(self.animation_time * 2.0 + i * 0.5) * 0.02 # Wavy motion
            y = y_offset # Vertical wave motion
            size = self.token_size + (i % 5) * 0.003 # Use parameter for base size, keep variation

            # Get color based on scheme
            color = self.get_token_color(i, num_tokens)
            glColor4f(color.redF(), color.greenF(), color.blueF(), 0.7) # Slightly more transparent

            glPushMatrix() # Prepare transformation matrix for each token
            glTranslatef(x, y, 0.0) # Translate to token position
            glRotatef(self.animation_time * 30 + i * 10, 0, 0, 1) # Example rotation - adjust as needed

            # Draw shape based on selected token_shape
            if self.token_shape == "Circle":
                gluDisk(
                    quad=gluNewQuadric(), # Create quadric object
                    innerRadius=0,
                    outerRadius=size,
                    slices=32, # Smooth circle
                    loops=32
                )
            elif self.token_shape == "Square":
                glBegin(GL_QUADS)
                glVertex2f(-size, -size)
                glVertex2f(size, -size)
                glVertex2f(size, size)
                glVertex2f(-size, size)
                glEnd()
            elif self.token_shape == "Triangle":
                glBegin(GL_TRIANGLES)
                glVertex2f(0, size)
                glVertex2f(-size, -size)
                glVertex2f(size, -size)
                glVertex2f(size, -size)
                glEnd()
            elif self.token_shape == "Line": # Example line shape
                glLineWidth(2.0) # Set line width
                glBegin(GL_LINES)
                glVertex2f(0, -size)
                glVertex2f(0, size)
                glEnd()

            glPopMatrix() # Restore transformation

    def draw_memory_influence_map(self):
        """Placeholder for Memory Influence Map visualization."""
        glColor3f(0.5, 0.5, 0.8) # Light purple for placeholder
        glBegin(GL_QUADS)
        glVertex2f(-0.8, -0.8)
        glVertex2f(0.8, -0.8)
        glVertex2f(0.8, 0.8)
        glVertex2f(-0.8, 0.8)
        glEnd()
        # Add text or shapes to indicate it's a placeholder
        glColor3f(1.0, 1.0, 1.0) # White text
        self.render_text(0, 0, "Memory Influence Map (Placeholder)")


    def draw_abstract_token_cloud(self):
        """Placeholder for Abstract Token Cloud visualization."""
        glColor3f(0.3, 0.7, 0.5) # Greenish for placeholder
        glBegin(GL_TRIANGLES)
        glVertex2f(-0.6, -0.6)
        glVertex2f(0.6, -0.6)
        glVertex2f(0.0, 0.6)
        glEnd()
        # Add text or shapes to indicate it's a placeholder
        glColor3f(1.0, 1.0, 1.0) # White text
        self.render_text(0, 0, "Abstract Token Cloud (Placeholder)")


    def render_text(self, x, y, text, font_size=16):
        """Helper to render text in OpenGL - basic implementation."""
        # Note: This is a very basic text rendering, consider using proper text rendering for better quality
        from PyQt6.QtGui import QFont, QColor
        from PyQt6.QtWidgets import QApplication

        font = QFont("Arial", font_size)
        color = QColor(255, 255, 255) # White

        # Get current context and painter from QApplication
        context = QApplication.instance()

        # Use renderText with world coordinates
        self.renderText(x, y, 0, text, font)


    def get_token_color(self, index, total_tokens):
        """Get color for token based on color scheme."""
        hue = (index * 360 / total_tokens) % 360 / 360.0 # Hue progression

        if self.color_scheme == "Cool":
            # Cool scheme - blues and greens
            return QColor.fromHslF(hue * 0.5 + 0.5, 0.8, 0.7) # Adjusted hue, saturation, lightness
        elif self.color_scheme == "Warm":
            # Warm scheme - reds and oranges
            return QColor.fromHslF(hue * 0.1, 0.7, 0.7) # Adjusted hue, saturation, lightness
        elif self.color_scheme == "GrayScale":
            # Gray scale - simple gray based on index
            gray_val = 0.2 + (1.0 - (index / total_tokens) * 0.7) # Darker to lighter gray
            return QColor.fromRgbF(gray_val, gray_val, gray_val)
        elif self.color_scheme == "Rainbow":
            # Full spectrum rainbow
            return QColor.fromHslF(hue, 0.9, 0.6)
        else:
            # Default - Cool (as fallback)
            return QColor.fromHslF(hue * 0.5 + 0.5, 0.8, 0.7)


    def resizeGL(self, width, height):
        """Handle viewport resizing."""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1) # Orthographic projection
        glMatrixMode(GL_MODELVIEW)

    def wheelEvent(self, event):
        """Handle mouse wheel event for zoom."""
        zoom_factor = 0.05 # Zoom sensitivity
        delta = event.angleDelta().y()
        if delta > 0: # Zoom in
            self.zoom_level *= (1.0 - zoom_factor)
        elif delta < 0: # Zoom out
            self.zoom_level *= (1.0 + zoom_factor)
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0)) # Clamp zoom level
        self.update() # Trigger repaint


class LLaDAGUINew(QMainWindow):
    """New OpenGL Visualization-Centric GUI for LLaDA application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLaDA GUI - OpenGL Viz - Prototype")
        self.resize(1200, 900)  # Slightly larger initial size

        # Worker thread reference
        self.worker = None

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

        # Add Generate and Stop buttons to status bar
        self.generate_button_status_bar = QPushButton("Generate")
        self.generate_button_status_bar.clicked.connect(self.on_generate_clicked)
        self.stop_button_status_bar = QPushButton("Stop")
        self.stop_button_status_bar.clicked.connect(self.on_stop_clicked)
        self.stop_button_status_bar.setEnabled(False) # Initially disabled

        self.status_bar.addPermanentWidget(self.generate_button_status_bar)
        self.status_bar.addPermanentWidget(self.stop_button_status_bar)


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
        generation_layout.addWidget(QLabel("Length:"), 0, 0)
        generation_layout.addWidget(self.gen_length_spin, 0, 1)

        # Sampling Steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(16, 512)
        self.steps_spin.setValue(DEFAULT_PARAMS['steps'])
        self.steps_spin.setSingleStep(16)
        generation_layout.addWidget(QLabel("Steps:"), 1, 0)
        generation_layout.addWidget(self.steps_spin, 1, 1)

        # Block Length
        self.block_length_spin = QSpinBox()
        self.block_length_spin.setRange(16, 256)
        self.block_length_spin.setValue(DEFAULT_PARAMS['block_length'])
        self.block_length_spin.setSingleStep(16)
        generation_layout.addWidget(QLabel("Block Size:"), 2, 0)
        generation_layout.addWidget(self.block_length_spin, 2, 1)

        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0, 2)
        self.temperature_spin.setValue(DEFAULT_PARAMS['temperature'])
        self.temperature_spin.setSingleStep(0.1)
        generation_layout.addWidget(QLabel("Temperature:"), 3, 0)
        generation_layout.addWidget(self.temperature_spin, 3, 1)

        # CFG Scale
        self.cfg_scale_spin = QDoubleSpinBox()
        self.cfg_scale_spin.setRange(0, 5)
        self.cfg_scale_spin.setValue(DEFAULT_PARAMS['cfg_scale'])
        self.cfg_scale_spin.setSingleStep(0.1)
        generation_layout.addWidget(QLabel("CFG Scale:"), 4, 0)
        generation_layout.addWidget(self.cfg_scale_spin, 4, 1)

        # Remasking Strategy
        self.remasking_combo = QComboBox()
        self.remasking_combo.addItems(["low_confidence", "random"])
        self.remasking_combo.setCurrentText(DEFAULT_PARAMS['remasking'])
        generation_layout.addWidget(QLabel("Remasking:"), 5, 0)
        generation_layout.addWidget(self.remasking_combo, 5, 1)

        generation_group.setLayout(generation_layout)
        self.sidebar_layout.addWidget(generation_group)

        # Model & Hardware 🧠
        model_group = QGroupBox("🧠 Model & Hardware")
        model_layout = QGridLayout()

        # Device Selection
        device_layout = QHBoxLayout()
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        self.device_group = QButtonGroup()
        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.gpu_radio)
        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.gpu_radio)
        model_layout.addWidget(QLabel("Device:"), 0, 0)
        model_layout.addLayout(device_layout, 0, 1)

        # Precision Options
        precision_layout = QHBoxLayout()
        self.normal_precision_radio = QRadioButton("Normal")
        self.quant_8bit_radio = QRadioButton("8-bit")
        self.quant_4bit_radio = QRadioButton("4-bit")
        self.precision_group = QButtonGroup()
        self.precision_group.addButton(self.normal_precision_radio)
        self.precision_group.addButton(self.quant_8bit_radio)
        self.precision_group.addButton(self.quant_4bit_radio)
        precision_layout.addWidget(self.normal_precision_radio)
        precision_layout.addWidget(self.quant_8bit_radio)
        precision_layout.addWidget(self.quant_4bit_radio)
        model_layout.addWidget(QLabel("Precision:"), 1, 0)
        model_layout.addLayout(precision_layout, 1, 1)

        # Extreme Mode Checkbox
        self.extreme_mode_checkbox = QCheckBox("Extreme Mode")
        model_layout.addWidget(self.extreme_mode_checkbox, 2, 1)

        # Fast Mode Checkbox
        self.fast_mode_checkbox = QCheckBox("Fast Mode")
        model_layout.addWidget(self.fast_mode_checkbox, 3, 1)


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
        stats_layout = QGridLayout()

        # Token Rate Display
        self.token_rate_label = QLabel("Token Rate: - tokens/s")
        stats_layout.addWidget(self.token_rate_label, 0, 0)

        # Step Time Display
        self.step_time_label = QLabel("Step Time: - ms/step")
        stats_layout.addWidget(self.step_time_label, 1, 0)

        # Detailed Memory Usage Display (Placeholder - expandable later)
        self.detailed_memory_label = QLabel("Memory Usage: - ")
        stats_layout.addWidget(self.detailed_memory_label, 2, 0)


        stats_group.setLayout(stats_layout)
        self.sidebar_layout.addWidget(stats_group)

        # Visualization Settings 👁️
        viz_settings_group = QGroupBox("👁️ Visualization Settings")
        viz_settings_layout = QGridLayout()

        # Visualization Type Selection
        self.visualization_type_combo = QComboBox()
        self.visualization_type_combo.addItems(["Token Stream", "Test Square", "Memory Influence Map", "Abstract Token Cloud"]) # Example types
        self.visualization_type_combo.currentTextChanged.connect(self.opengl_viz_widget.set_visualization_type)
        viz_settings_layout.addWidget(QLabel("Type:"), 0, 0)
        viz_settings_layout.addWidget(self.visualization_type_combo, 0, 1)

        # Color Scheme Selection
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Cool", "Warm", "GrayScale", "Rainbow"])
        self.color_scheme_combo.setCurrentText("Cool")
        self.color_scheme_combo.currentTextChanged.connect(self.opengl_viz_widget.set_color_scheme)
        viz_settings_layout.addWidget(QLabel("Color Scheme:"), 1, 0)
        viz_settings_layout.addWidget(self.color_scheme_combo, 1, 1)

        # Token Shape Selection
        self.token_shape_combo = QComboBox()
        self.token_shape_combo.addItems(["Circle", "Square", "Triangle", "Line"])
        self.token_shape_combo.setCurrentText("Circle")
        self.token_shape_combo.currentTextChanged.connect(self.opengl_viz_widget.set_token_shape)
        viz_settings_layout.addWidget(QLabel("Token Shape:"), 2, 0)
        viz_settings_layout.addWidget(self.token_shape_combo, 2, 1)

        # Animation Speed Control
        self.animation_speed_spin = QDoubleSpinBox()
        self.animation_speed_spin.setRange(0.001, 0.1)
        self.animation_speed_spin.setValue(0.01)
        self.animation_speed_spin.setSingleStep(0.005)
        self.animation_speed_spin.valueChanged.connect(self.opengl_viz_widget.set_animation_speed)
        viz_settings_layout.addWidget(QLabel("Animation Speed:"), 3, 0)
        viz_settings_layout.addWidget(self.animation_speed_spin, 3, 1)

        # Token Size Control
        self.token_size_spin = QDoubleSpinBox()
        self.token_size_spin.setRange(0.01, 0.1)
        self.token_size_spin.setValue(0.03)
        self.token_size_spin.setSingleStep(0.005)
        self.token_size_spin.valueChanged.connect(self.opengl_viz_widget.set_token_size)
        viz_settings_layout.addWidget(QLabel("Token Size:"), 4, 0)
        viz_settings_layout.addWidget(self.token_size_spin, 4, 1)

        # Token Spacing Control
        self.token_spacing_spin = QDoubleSpinBox()
        self.token_spacing_spin.setRange(0.01, 0.2)
        self.token_spacing_spin.setValue(0.07)
        self.token_spacing_spin.setSingleStep(0.01)
        self.token_spacing_spin.valueChanged.connect(self.opengl_viz_widget.set_token_spacing)
        viz_settings_layout.addWidget(QLabel("Token Spacing:"), 5, 0)
        viz_settings_layout.addWidget(self.token_spacing_spin, 5, 1)

        # Zoom Level Control - Example for future zoom control
        self.zoom_level_spin = QDoubleSpinBox()
        self.zoom_level_spin.setRange(0.1, 5.0)
        self.zoom_level_spin.setValue(1.0)
        self.zoom_level_spin.setSingleStep(0.1)
        self.zoom_level_spin.valueChanged.connect(self.opengl_viz_widget.set_zoom_level)
        viz_settings_layout.addWidget(QLabel("Zoom Level:"), 6, 0)
        viz_settings_layout.addWidget(self.zoom_level_spin, 6, 1)


        viz_settings_group.setLayout(viz_settings_layout)
        self.sidebar_layout.addWidget(viz_settings_group)

        # Add stretch to bottom to push groups to the top
        self.sidebar_layout.addStretch(1)

    def get_generation_config(self):
        """Get the current generation configuration from UI elements."""
        device = 'cuda' if self.gpu_radio.isChecked() and torch.cuda.is_available() else 'cpu'
        return {
            'gen_length': self.gen_length_spin.value(),
            'steps': self.steps_spin.value(),
            'block_length': self.block_length_spin.value(),
            'temperature': self.temperature_spin.value(),
            'cfg_scale': self.cfg_scale_spin.value(),
            'remasking': self.remasking_combo.currentText(),
            'device': device,
            'use_8bit': self.use_8bit.isChecked() and device == 'cuda',
            'use_4bit': self.use_4bit.isChecked() and device == 'cuda',
            'extreme_mode': self.extreme_mode_checkbox.isChecked(),
            'fast_mode': self.fast_mode_checkbox.isChecked(),
            'use_memory': self.enable_memory_checkbox.isChecked() # This checkbox does not exist in this GUI - might be from old GUI
        }

    @pyqtSlot()
    def on_generate_clicked(self):
        """Handle Generate button click: start generation."""
        prompt_text = self.prompt_input.toPlainText().strip()
        if not prompt_text:
            QMessageBox.warning(self, "Input Error", "Please enter a prompt.")
            return

        config = self.get_generation_config()

        # Disable UI elements and enable stop button
        self.set_ui_generating(True)

        # Create worker thread and connect signals
        self.worker = LLaDAWorker(prompt_text, config) # Pass config
        self.worker.progress.connect(self.update_progress) # Connect progress signal
        self.worker.step_update.connect(self.update_visualization) # Connect step_update
        self.worker.finished.connect(self.generation_finished) # Connect finished signal
        self.worker.error.connect(self.generation_error) # Connect error signal
        self.worker.realtime_stats.connect(self.update_realtime_stats_display) # Connect realtime stats signal
        self.worker.start() # Start the worker thread

    @pyqtSlot()
    def on_stop_clicked(self):
        """Handle Stop button click: stop generation."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.set_ui_generating(False) # Re-enable UI, keep stop disabled

    @pyqtSlot(int, str, dict)
    def update_progress(self, progress_percent, message, data):
        """Update progress bar and status message during generation."""
        self.status_bar.showMessage(f"Generating - {message}")
        # Placeholder for progress bar update if we add one to status bar

    @pyqtSlot(int, list, list, list)
    def update_visualization(self, step, tokens, masks, confidences):
        """Update visualization during each step."""
        # For now, just print step info to console - replace with OpenGL viz update later
        print(f"Step: {step}, Tokens: {tokens[:10]}..., Masks: {masks[:10]}..., Confidences: {confidences[:10]}...")
        self.opengl_viz_widget.set_token_stream_data(tokens) # Example of sending data to OpenGL widget

    @pyqtSlot(str)
    def generation_finished(self, output_text):
        """Handle generation finished signal."""
        self.status_bar.showMessage("Generation Finished")
        # For now, print output to console - replace with proper output display later
        print(f"Generated Output: {output_text}")
        self.prompt_input.setPlainText(output_text) # Just for testing, replace with proper output display later
        self.set_ui_generating(False) # Re-enable UI

    @pyqtSlot(str)
    def generation_error(self, error_message):
        """Handle generation error signal."""
        self.status_bar.showMessage(f"Generation Error: {error_message}")
        QMessageBox.critical(self, "Generation Error", f"Error: {error_message}")
        self.set_ui_generating(False) # Re-enable UI

    def set_ui_generating(self, is_generating):
        """Enable/disable UI elements based on generation status."""
        self.generate_button_status_bar.setEnabled(not is_generating)
        self.stop_button_status_bar.setEnabled(is_generating)
        self.prompt_input.setEnabled(not is_generating)
        # ... (Add other UI elements to disable as needed)

    @pyqtSlot(dict)
    def update_realtime_stats_display(self, stats):
        """Update realtime statistics in the sidebar."""
        self.token_rate_label.setText(f"Token Rate: {stats.get('token_rate', '-')}")
        self.step_time_label.setText(f"Step Time: {stats.get('step_time', '-')} ms/step")
        self.detailed_memory_label.setText(f"Memory Usage: {stats.get('memory_usage', '-')}")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = LLaDAGUINew()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
