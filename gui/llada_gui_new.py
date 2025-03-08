# -*- coding: utf-8 -*-

"""
LLaDA GUI - New OpenGL Visualization-Centric GUI for LLaDA.
"""

import random
import sys
import numpy as np
import torch
from OpenGL.GL import *
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSettings
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QPushButton, QLabel, QSpinBox, QComboBox, QGroupBox,
    QCheckBox, QProgressBar, QSplitter, QMessageBox, QGridLayout,
    QScrollArea, QDoubleSpinBox, QRadioButton, QButtonGroup, QStatusBar
)
import matplotlib.pyplot as plt

from core.config import DEFAULT_GENERATION_PARAMS as DEFAULT_PARAMS
from core.llada_worker import LLaDAWorker
from gui.memory_monitor import MemoryMonitor


class GLVisualizationWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.visualization_type = "Token Stream"
        self.token_stream_data_strings = [] # List of token display strings or token IDs
        self.token_stream_mask_indices = [] # List of mask indices (bool)
        self.token_stream_confidence_scores = [] # List of confidence scores
        self.memory_influence_data = None
        self.color_scheme = "Cool"
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update)
        self.animation_time = 0.0
        self.animation_speed = 0.01
        self.animation_timer.start(20)
        self.token_shape = "Circle"
        self.token_size = 0.03
        self.token_spacing = 0.07
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.setMouseTracking(True)
        self.last_mouse_pos = None
        self.text_textures = {}  # Cache for glyph textures
        self.text_coords = {}   # Cache for glyph coordinates
        self.token_stream_data_mode = "Decoded Tokens" # Default mode

    def set_visualization_type(self, viz_type): self.visualization_type = viz_type; self.update()
    def set_color_scheme(self, scheme): self.color_scheme = scheme; self.update()
    def set_token_stream_data(self, data, masks, confidences, data_mode="Decoded Tokens"): # Update to accept lists and data_mode
        self.token_stream_data_strings = data
        self.token_stream_mask_indices = masks
        self.token_stream_confidence_scores = confidences
        self.token_stream_data_mode = data_mode # Store data mode
        self.update()
    def set_memory_influence_data(self, data): self.memory_influence_data = data; self.update()
    def set_token_shape(self, shape): self.token_shape = shape; self.update()
    def set_animation_speed(self, speed): self.animation_speed = speed; self.update()
    def set_token_size(self, size): self.token_size = size; self.update()
    def set_token_spacing(self, spacing): self.token_spacing = spacing; self.update()
    def set_zoom_level(self, zoom): self.zoom_level = zoom; self.update()
    def set_pan(self, pan_x, pan_y): self.pan_x = pan_x; self.pan_y = pan_y; self.update()
    def set_token_stream_data_mode(self, mode): self.token_stream_data_mode = mode; self.update()


    def initializeGL(self):
        if not self.context().isValid():
            raise RuntimeError("Failed to create a valid OpenGL context")
        if self.context().format().majorVersion() < 3:
            print("Warning: OpenGL 3.3+ not supported.")
        glClearColor(0.1, 0.1, 0.2, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1 * self.zoom_level + self.pan_x, 1 * self.zoom_level + self.pan_x,
                -1 * self.zoom_level + self.pan_y, 1 * self.zoom_level + self.pan_y, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        self.animation_time += self.animation_speed

        if self.visualization_type == "Token Stream":
            self.draw_token_stream()
        elif self.visualization_type == "Test Square":
            self.draw_test_square()
        elif self.visualization_type == "Memory Influence Map":
            self.draw_memory_influence_map()
        elif self.visualization_type == "Abstract Token Cloud":
            self.draw_abstract_token_cloud()
            self.render_text(-0.5, 0, "Token Cloud Active", QColor(255, 255, 255))

    def draw_shape(self, shape_type, x, y, size, color, num_vertices=None):
        vertices = []
        colors = []
        color_vals = [color.redF(), color.greenF(), color.blueF(), 0.7]
        if shape_type == "Circle":
            num_vertices = num_vertices or 32
            for theta in np.linspace(0, 2 * np.pi, num_vertices):
                x0 = x + size * np.cos(theta)
                y0 = y + size * np.sin(theta)
                x1 = x + size * np.cos(theta + 2 * np.pi / num_vertices)
                y1 = y + size * np.sin(theta + 2 * np.pi / num_vertices)
                vertices.extend([x, y, 0.0, x0, y0, 0.0, x1, y1, 0.0])
                colors.extend(color_vals * 3)
            mode = GL_TRIANGLES
        elif shape_type == "Square":
            vertices.extend([x - size, y - size, 0.0, x + size, y - size, 0.0, x + size, y + size, 0.0, x - size, y + size, 0.0])
            colors.extend(color_vals * 4)
            mode = GL_QUADS
        elif shape_type == "Triangle":
            vertices.extend([x, y + size, 0.0, x - size, y - size, 0.0, x + size, y - size, 0.0])
            colors.extend(color_vals * 3)
            mode = GL_TRIANGLES
        else:  # Line
            vertices.extend([x, y - size, 0.0, x, y + size, 0.0])
            colors.extend(color_vals * 2)
            mode = GL_LINES

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glColorPointer(4, GL_FLOAT, 0, colors)
        if mode == GL_LINES: glLineWidth(2.0)
        glDrawArrays(mode, 0, len(vertices) // 3)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def draw_token_stream(self):
        num_tokens = len(self.token_stream_data_strings) # Use actual data length
        if num_tokens == 0: return # Handle empty data
        spacing = self.token_spacing
        start_x = - (num_tokens - 1) * spacing / 2

        for i in range(num_tokens):
            x = start_x + i * spacing
            y = np.sin(self.animation_time * 2.0 + i * 0.5) * 0.02
            size = self.token_size # Base size
            color = self.get_token_color(i, num_tokens)
            shape = self.token_shape

            if self.token_stream_mask_indices[i]: # Masked token
                shape = "Square" # Indicate masked tokens with squares
                size *= 0.8 # Slightly smaller size for masked tokens
                color = color.darker(150) # Darker color for masked tokens
            else:
                confidence = self.token_stream_confidence_scores[i]
                size += confidence * 0.01 # Size varies with confidence
                color = self.adjust_color_alpha(color, 0.5 + confidence * 0.5) # Alpha varies with confidence

            text_to_render = ""
            if self.token_stream_data_mode == "Decoded Tokens":
                text_to_render = self.token_stream_data_strings[i]
            elif self.token_stream_data_mode == "Token IDs":
                text_to_render = str(self.token_stream_data_strings[i]) # Assuming token_stream_data_strings is now token IDs in this mode

            self.draw_shape(shape, x, y, size, color)
            self.render_text(x - self.token_spacing * 0.3, y - self.token_size * 2, text_to_render, QColor(200, 200, 220), font_size=10) # Render token text below shape


    def adjust_color_alpha(self, color, alpha_factor):
        """Adjust the alpha (transparency) of a QColor."""
        current_alpha = color.alphaF()
        new_alpha = max(0.0, min(1.0, current_alpha * alpha_factor)) # Clamp alpha to [0, 1]
        return QColor.fromRgbaF(color.redF(), color.greenF(), color.blueF(), new_alpha)


    def draw_test_square(self):
        self.draw_shape("Square", 0, 0, 0.5, QColor.fromRgbF(1.0, 1.0, 0.0))

    def draw_memory_influence_map(self):
        grid_resolution = 32
        influence_data = self.memory_influence_data if self.memory_influence_data is not None else np.random.rand(grid_resolution, grid_resolution)
        cmap = self.get_colormap(self.color_scheme)
        normalized_data = (influence_data - influence_data.min()) / (influence_data.max() - influence_data.min() + 1e-8)
        colors = cmap(normalized_data.flatten())
        cell_size = 2.0 / grid_resolution
        vertices = []
        vertex_colors = []
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                x = -1.0 + j * cell_size
                y = 1.0 - i * cell_size
                color = colors[i * grid_resolution + j]
                vertices.extend([x, y, 0.0, x + cell_size, y, 0.0, x + cell_size, y - cell_size, 0.0, x, y - cell_size, 0.0])
                vertex_colors.extend([color[0], color[1], color[2], 1.0] * 4)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glColorPointer(4, GL_FLOAT, 0, vertex_colors)
        glDrawArrays(GL_QUADS, 0, len(vertices) // 3)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def draw_abstract_token_cloud(self):
        num_clouds = 50
        for _ in range(num_clouds):
            x = random.uniform(-0.9, 0.9)
            y = random.uniform(-0.9, 0.9)
            size = random.uniform(0.01, 0.05)
            color = QColor.fromRgbF(*[random.uniform(0.3, 0.7) for _ in range(3)])
            self.draw_shape("Triangle", x, y, size, color)

    def get_colormap(self, scheme_name="Cool"):
        schemes = {"Cool": plt.cm.viridis, "Warm": plt.cm.magma, "GrayScale": plt.cm.gray,
                   "Rainbow": plt.cm.rainbow, "CoolWarm": plt.cm.coolwarm, "Plasma": plt.cm.plasma}
        return schemes.get(scheme_name, plt.cm.viridis)

    def get_token_color(self, index, total_tokens):
        hue = (index * 360 / total_tokens) % 360 / 360.0
        schemes = {
            "Cool": lambda h: QColor.fromHslF(h * 0.5 + 0.5, 0.8, 0.7),
            "Warm": lambda h: QColor.fromHslF(h * 0.1, 0.7, 0.7),
            "GrayScale": lambda h: QColor.fromRgbF(0.2 + (1.0 - (index / total_tokens) * 0.7), 0.2 + (1.0 - (index / total_tokens) * 0.7), 0.2 + (1.0 - (index / total_tokens) * 0.7)),
            "Rainbow": lambda h: QColor.fromHslF(h, 0.9, 0.6),
            "CoolWarm": lambda h: QColor.fromHslF(h * 0.5, 0.7, 0.7),
            "Plasma": lambda h: QColor.fromHslF(h * 0.2, 0.8, 0.6)
        }
        return schemes.get(self.color_scheme, lambda h: QColor.fromHslF(h * 0.5 + 0.5, 0.8, 0.7))(hue)

    def render_text(self, x, y, text, color, font_size=16):
        """Self-contained text glyph rendering using bitmap textures."""
        font = QFont("Arial", font_size)
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()

        # Generate bitmap for the text
        image = QImage(text_width, text_height, QImage.Format.Format_ARGB32)
        #image.fill(Qt.transparent)
        painter = QPainter(image)
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(0, metrics.ascent(), text)
        painter.end()

        # Cache texture if not already generated
        if text not in self.text_textures:
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            image_data = image.bits().asstring(image.sizeInBytes())
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, image_data)
            self.text_textures[text] = texture_id
            self.text_coords[text] = (text_width / (self.width() * self.zoom_level), text_height / (self.height() * self.zoom_level))

        # Render texture
        glBindTexture(GL_TEXTURE_2D, self.text_textures[text])
        w, h = self.text_coords[text]
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x, y + h)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)

    def wheelEvent(self, event):
        zoom_factor = 0.15
        delta = event.angleDelta().y()
        self.zoom_level *= (1.0 - zoom_factor) if delta > 0 else (1.0 + zoom_factor)
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0))
        if self.parent_widget: self.parent_widget.zoom_level_spin.setValue(self.zoom_level)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is not None:
            current_mouse_pos = event.pos()
            delta_x = current_mouse_pos.x() - self.last_mouse_pos.x()
            delta_y = current_mouse_pos.y() - self.last_mouse_pos.y()
            pan_speed = 0.005 * self.zoom_level
            self.pan_x -= delta_x * pan_speed
            self.pan_y += delta_y * pan_speed
            self.last_mouse_pos = current_mouse_pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.last_mouse_pos = None


class LLaDAGUINew(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLaDA GUI - OpenGL Viz - Prototype")
        self.resize(1200, 900)
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.update.connect(self.update_memory_status_bar)
        self.memory_monitor.start()
        self.worker = None
        self.token_stream_data_mode = "Decoded Tokens" # Store token stream data mode
        self.setup_ui()
        self.load_settings()

    def add_widget(self, layout, label_text, widget, row, col, tooltip=None):
        """Macro-like function to add labeled widgets to a layout."""
        label = QLabel(label_text)
        if tooltip: widget.setToolTip(tooltip)
        layout.addWidget(label, row, col)
        layout.addWidget(widget, row, col + 1)

    def setup_ui(self):
        main_widget = QSplitter(Qt.Orientation.Vertical)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        main_widget.addWidget(self.prompt_input)

        center_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_widget.addWidget(center_splitter)
        self.opengl_viz_widget = GLVisualizationWidget(self)
        center_splitter.addWidget(self.opengl_viz_widget)

        self.sidebar_scroll_area = QScrollArea()
        self.sidebar_scroll_area.setWidgetResizable(True)
        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_scroll_area.setWidget(self.sidebar_widget)
        center_splitter.addWidget(self.sidebar_scroll_area)
        center_splitter.setSizes([600, 600])
        main_widget.setSizes([175, 725])

        # Visualization Settings
        viz_settings_group = QGroupBox("👁️ Visualization Settings")
        viz_settings_layout = QGridLayout()
        self.visualization_type_combo = QComboBox()
        self.visualization_type_combo.addItems(["Token Stream", "Test Square", "Memory Influence Map", "Abstract Token Cloud"])
        self.visualization_type_combo.currentTextChanged.connect(self.on_visualization_type_changed)
        self.add_widget(viz_settings_layout, "Type:", self.visualization_type_combo, 0, 0)

        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["Cool", "Warm", "GrayScale", "Rainbow", "CoolWarm", "Plasma"])
        self.color_scheme_combo.setCurrentText("Cool")
        self.color_scheme_combo.currentTextChanged.connect(self.opengl_viz_widget.set_color_scheme)
        self.add_widget(viz_settings_layout, "Color Scheme:", self.color_scheme_combo, 1, 0)

        self.token_shape_combo = QComboBox()
        self.token_shape_combo.addItems(["Circle", "Square", "Triangle", "Line"])
        self.token_shape_combo.setCurrentText("Circle")
        self.token_shape_combo.currentTextChanged.connect(self.opengl_viz_widget.set_token_shape)
        self.add_widget(viz_settings_layout, "Token Shape:", self.token_shape_combo, 2, 0)

        self.animation_speed_spin = QDoubleSpinBox()
        self.animation_speed_spin.setRange(0.001, 0.1); self.animation_speed_spin.setValue(0.01); self.animation_speed_spin.setSingleStep(0.005)
        self.animation_speed_spin.valueChanged.connect(self.opengl_viz_widget.set_animation_speed)
        self.add_widget(viz_settings_layout, "Animation Speed:", self.animation_speed_spin, 3, 0)

        self.token_size_spin = QDoubleSpinBox()
        self.token_size_spin.setRange(0.01, 0.1); self.token_size_spin.setValue(0.03); self.token_size_spin.setSingleStep(0.005)
        self.token_size_spin.valueChanged.connect(self.opengl_viz_widget.set_token_size)
        self.add_widget(viz_settings_layout, "Token Size:", self.token_size_spin, 4, 0)

        self.token_spacing_spin = QDoubleSpinBox()
        self.token_spacing_spin.setRange(0.01, 0.2); self.token_spacing_spin.setValue(0.07); self.token_spacing_spin.setSingleStep(0.01)
        self.token_spacing_spin.valueChanged.connect(self.opengl_viz_widget.set_token_spacing)
        self.add_widget(viz_settings_layout, "Token Spacing:", self.token_spacing_spin, 5, 0)

        self.zoom_level_spin = QDoubleSpinBox()
        self.zoom_level_spin.setRange(0.1, 5.0); self.zoom_level_spin.setValue(1.0); self.zoom_level_spin.setSingleStep(0.1)
        self.zoom_level_spin.valueChanged.connect(self.opengl_viz_widget.set_zoom_level)
        self.add_widget(viz_settings_layout, "Zoom Level:", self.zoom_level_spin, 6, 0)

        self.token_data_mode_combo = QComboBox() # New ComboBox for token data mode
        self.token_data_mode_combo.addItems(["Decoded Tokens", "Token IDs"])
        self.token_data_mode_combo.setCurrentText("Decoded Tokens")
        self.token_data_mode_combo.currentTextChanged.connect(self.on_token_stream_data_mode_changed) # Connect to new slot
        self.add_widget(viz_settings_layout, "Token Data:", self.token_data_mode_combo, 7, 0) # Add to layout


        viz_settings_group.setLayout(viz_settings_layout)
        self.sidebar_layout.addWidget(viz_settings_group)

        # Generation Parameters
        gen_params_group = QGroupBox("⚙️ Generation Parameters")
        gen_params_layout = QGridLayout()
        self.gen_length_spin = QSpinBox()
        self.gen_length_spin.setRange(16, 512); self.gen_length_spin.setValue(DEFAULT_PARAMS['gen_length']); self.gen_length_spin.setSingleStep(16)
        self.add_widget(gen_params_layout, "Length:", self.gen_length_spin, 0, 0)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(16, 512); self.steps_spin.setValue(DEFAULT_PARAMS['steps']); self.steps_spin.setSingleStep(16)
        self.add_widget(gen_params_layout, "Steps:", self.steps_spin, 0, 2)

        self.block_length_spin = QSpinBox()
        self.block_length_spin.setRange(16, 256); self.block_length_spin.setValue(DEFAULT_PARAMS['block_length']); self.block_length_spin.setSingleStep(16)
        self.add_widget(gen_params_layout, "Block Length:", self.block_length_spin, 1, 0)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0, 2); self.temperature_spin.setValue(DEFAULT_PARAMS['temperature']); self.temperature_spin.setSingleStep(0.1)
        self.add_widget(gen_params_layout, "Temperature:", self.temperature_spin, 1, 2)

        self.cfg_scale_spin = QDoubleSpinBox()
        self.cfg_scale_spin.setRange(0, 5); self.cfg_scale_spin.setValue(DEFAULT_PARAMS['cfg_scale']); self.cfg_scale_spin.setSingleStep(0.1)
        self.add_widget(gen_params_layout, "CFG Scale:", self.cfg_scale_spin, 2, 0, "Classifier-Free Guidance scale")

        self.remasking_combo = QComboBox()
        self.remasking_combo.addItems(["low_confidence", "random"]); self.remasking_combo.setCurrentText(DEFAULT_PARAMS['remasking'])
        self.add_widget(gen_params_layout, "Remasking:", self.remasking_combo, 2, 2, "Method for remasking tokens during generation")

        gen_params_group.setLayout(gen_params_layout)
        self.sidebar_layout.addWidget(gen_params_group)

        # Hardware & Memory Options
        hw_memory_group = QGroupBox("⚙️ Hardware & Memory Options")
        hw_memory_layout = QGridLayout()
        hw_memory_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_group = QButtonGroup()
        self.cpu_radio = QRadioButton("CPU")
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        if torch.cuda.is_available(): self.gpu_radio.setChecked(True)
        else: self.cpu_radio.setChecked(True); self.gpu_radio.setEnabled(False)
        self.device_group.addButton(self.cpu_radio, 0); self.device_group.addButton(self.gpu_radio, 1)
        hw_memory_layout.addWidget(self.cpu_radio, 0, 1); hw_memory_layout.addWidget(self.gpu_radio, 0, 2)

        hw_memory_layout.addWidget(QLabel("Precision:"), 1, 0)
        self.precision_group = QButtonGroup()
        self.use_normal = QRadioButton("Normal"); self.use_8bit = QRadioButton("8-bit"); self.use_4bit = QRadioButton("4-bit")
        self.use_8bit.setChecked(True)
        self.precision_group.addButton(self.use_normal, 0); self.precision_group.addButton(self.use_8bit, 1); self.precision_group.addButton(self.use_4bit, 2)
        hw_memory_layout.addWidget(self.use_normal, 1, 1); hw_memory_layout.addWidget(self.use_8bit, 1, 2); hw_memory_layout.addWidget(self.use_4bit, 1, 3)

        self.extreme_mode_checkbox = QCheckBox("Extreme Mode"); self.extreme_mode_checkbox.setToolTip("Enable extreme memory optimizations")
        hw_memory_layout.addWidget(self.extreme_mode_checkbox, 2, 0, 1, 3)
        self.fast_mode_checkbox = QCheckBox("Fast Mode"); self.fast_mode_checkbox.setToolTip("Enable faster generation (lower quality)")
        hw_memory_layout.addWidget(self.fast_mode_checkbox, 3, 0, 1, 3)
        self.enable_memory_checkbox = QCheckBox("Enable Memory Integration"); self.enable_memory_checkbox.setToolTip("Enable memory integration for context-aware generation")
        hw_memory_layout.addWidget(self.enable_memory_checkbox, 4, 0, 1, 3)

        hw_memory_group.setLayout(hw_memory_layout)
        self.sidebar_layout.addWidget(hw_memory_group)

        # Realtime Statistics Display
        stats_group = QGroupBox("📊 Realtime Statistics")
        stats_layout = QGridLayout()
        self.token_rate_label = QLabel("Token Rate: -")
        self.step_time_label = QLabel("Step Time: - ms/step")
        self.detailed_memory_label = QLabel("Memory Usage: -")
        stats_layout.addWidget(self.token_rate_label, 0, 0)
        stats_layout.addWidget(self.step_time_label, 1, 0)
        stats_layout.addWidget(self.detailed_memory_label, 2, 0)
        stats_group.setLayout(stats_layout)
        self.sidebar_layout.addWidget(stats_group)

        # Memory Status Display
        memory_status_group = QGroupBox("💾 Memory Status")
        memory_status_layout = QGridLayout()
        self.system_ram_progress_sidebar = QProgressBar(); self.system_ram_progress_sidebar.setRange(0, 100); self.system_ram_progress_sidebar.setTextVisible(False)
        self.system_ram_label_sidebar = QLabel("- / - GB (-%)")
        self.add_widget(memory_status_layout, "System RAM:", self.system_ram_progress_sidebar, 0, 0)
        memory_status_layout.addWidget(self.system_ram_label_sidebar, 0, 2)

        self.gpu_vram_progress_sidebar = QProgressBar(); self.gpu_vram_progress_sidebar.setRange(0, 100); self.gpu_vram_progress_sidebar.setTextVisible(False)
        self.gpu_vram_label_sidebar = QLabel("- / - GB (-%)")
        self.add_widget(memory_status_layout, "GPU VRAM:", self.gpu_vram_progress_sidebar, 1, 0)
        memory_status_layout.addWidget(self.gpu_vram_label_sidebar, 1, 2)

        memory_status_group.setLayout(memory_status_layout)
        self.sidebar_layout.addWidget(memory_status_group)
        self.sidebar_layout.addStretch(1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        self.generate_button_status_bar = QPushButton("Generate"); self.generate_button_status_bar.clicked.connect(self.on_generate_clicked)
        self.stop_button_status_bar = QPushButton("Stop"); self.stop_button_status_bar.clicked.connect(self.on_stop_clicked); self.stop_button_status_bar.setEnabled(False)
        self.clear_button_status_bar = QPushButton("Clear"); self.clear_button_status_bar.clicked.connect(self.clear_output)
        status_button_layout = QHBoxLayout()
        status_button_layout.addStretch(1)
        status_button_layout.addWidget(self.generate_button_status_bar)
        status_button_layout.addWidget(self.stop_button_status_bar)
        status_button_layout.addWidget(self.clear_button_status_bar)
        status_button_widget = QWidget(); status_button_widget.setLayout(status_button_layout)
        self.status_bar.addPermanentWidget(status_button_widget)

        self.ram_indicator = QProgressBar(); self.ram_indicator.setMaximumWidth(70); self.ram_indicator.setTextVisible(False)
        self.gpu_indicator = QProgressBar(); self.gpu_indicator.setMaximumWidth(70); self.gpu_indicator.setTextVisible(False)
        self.status_bar.addPermanentWidget(QLabel("RAM:")); self.status_bar.addPermanentWidget(self.ram_indicator)
        self.status_bar.addPermanentWidget(QLabel("VRAM:")); self.status_bar.addPermanentWidget(self.gpu_indicator)

        self.setCentralWidget(main_widget)

    def on_visualization_type_changed(self, viz_type):
        self.opengl_viz_widget.set_visualization_type(viz_type)
        is_token_stream = viz_type == "Token Stream"
        self.token_shape_combo.setEnabled(is_token_stream)
        self.animation_speed_spin.setEnabled(is_token_stream or viz_type == "Abstract Token Cloud")
        self.token_size_spin.setEnabled(is_token_stream)
        self.token_spacing_spin.setEnabled(is_token_stream)
        self.token_data_mode_combo.setEnabled(is_token_stream) # Enable/disable token data mode combo

    def on_token_stream_data_mode_changed(self, mode): # New slot to handle token data mode change
        self.token_stream_data_mode = mode
        self.opengl_viz_widget.set_token_stream_data_mode(mode) # Pass mode to GL widget


    def get_generation_config(self):
        device = 'cuda' if self.gpu_radio.isChecked() and torch.cuda.is_available() else 'cpu'
        return {
            'gen_length': self.gen_length_spin.value(), 'steps': self.steps_spin.value(), 'block_length': self.block_length_spin.value(),
            'temperature': self.temperature_spin.value(), 'cfg_scale': self.cfg_scale_spin.value(), 'remasking': self.remasking_combo.currentText(),
            'device': device, 'use_8bit': self.use_8bit.isChecked() and device == 'cuda', 'use_4bit': self.use_4bit.isChecked() and device == 'cuda',
            'extreme_mode': self.extreme_mode_checkbox.isChecked(), 'fast_mode': self.fast_mode_checkbox.isChecked(),
            'use_memory': self.enable_memory_checkbox.isChecked()
        }

    @pyqtSlot()
    def on_generate_clicked(self):
        prompt_text = self.prompt_input.toPlainText().strip()
        if not prompt_text: return QMessageBox.warning(self, "Input Error", "Please enter a prompt.")
        config = self.get_generation_config()
        self.set_ui_generating(True)
        self.worker = LLaDAWorker(prompt_text, config)
        self.worker.progress.connect(self.update_progress)
        self.worker.step_update.connect(self.update_visualization)
        self.worker.finished.connect(self.generation_finished)
        self.worker.error.connect(self.generation_error)
        self.worker.realtime_stats.connect(self.update_realtime_stats_display)
        self.worker.memory_influence_update.connect(self.opengl_viz_widget.set_memory_influence_data)
        self.worker.start()

    @pyqtSlot()
    def on_stop_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.set_ui_generating(False)

    @pyqtSlot(int, str, dict)
    def update_progress(self, progress_percent, message, data):
        self.status_bar.showMessage(f"Generating - {message}")

    @pyqtSlot(int, list, list, list, list) # Updated signal params
    def update_visualization(self, step, tokens, masks, confidences, token_ids): # Updated slot params
        print(f"Step: {step}, Tokens: {tokens[:10]}..., Masks: {masks[:10]}..., Confidences: {confidences[:10]}...")
        if self.opengl_viz_widget.visualization_type == "Token Stream":
            data_to_visualize = tokens if self.token_stream_data_mode == "Decoded Tokens" else token_ids # Choose data based on mode
            self.opengl_viz_widget.set_token_stream_data(data_to_visualize, masks, confidences, self.token_stream_data_mode) # Pass data and mode
        elif self.opengl_viz_widget.visualization_type == "Memory Influence Map":
            self.opengl_viz_widget.set_memory_influence_data(np.random.rand(32, 32)) # Keep dummy for now

    @pyqtSlot(str)
    def generation_finished(self, output_text):
        self.status_bar.showMessage("Generation Finished")
        print(f"Generated Output: {output_text}")
        self.prompt_input.setPlainText(output_text)
        self.set_ui_generating(False)

    @pyqtSlot(str)
    def generation_error(self, error_message):
        self.status_bar.showMessage(f"Generation Error: {error_message}")
        QMessageBox.critical(self, "Generation Error", f"Error: {error_message}")
        self.set_ui_generating(False)

    def set_ui_generating(self, is_generating):
        self.generate_button_status_bar.setEnabled(not is_generating)
        self.stop_button_status_bar.setEnabled(is_generating)
        self.prompt_input.setEnabled(not is_generating)
        self.visualization_type_combo.setEnabled(not is_generating)
        self.color_scheme_combo.setEnabled(not is_generating)
        self.token_shape_combo.setEnabled(not is_generating and self.visualization_type_combo.currentText() == "Token Stream")
        self.animation_speed_spin.setEnabled(not is_generating)
        self.token_size_spin.setEnabled(not is_generating and self.visualization_type_combo.currentText() == "Token Stream")
        self.token_spacing_spin.setEnabled(not is_generating and self.visualization_type_combo.currentText() == "Token Stream")
        self.zoom_level_spin.setEnabled(not is_generating)
        self.token_data_mode_combo.setEnabled(not is_generating and self.visualization_type_combo.currentText() == "Token Stream") # Enable/disable token data mode combo
        self.gen_length_spin.setEnabled(not is_generating)
        self.steps_spin.setEnabled(not is_generating)
        self.block_length_spin.setEnabled(not is_generating)
        self.temperature_spin.setEnabled(not is_generating)
        self.cfg_scale_spin.setEnabled(not is_generating)
        self.remasking_combo.setEnabled(not is_generating)
        self.cpu_radio.setEnabled(not is_generating)
        self.gpu_radio.setEnabled(not is_generating)
        self.use_normal.setEnabled(not is_generating)
        self.use_8bit.setEnabled(not is_generating)
        self.use_4bit.setEnabled(not is_generating)
        self.extreme_mode_checkbox.setEnabled(not is_generating)
        self.fast_mode_checkbox.setEnabled(not is_generating)
        self.enable_memory_checkbox.setEnabled(not is_generating)

    @pyqtSlot(dict)
    def update_realtime_stats_display(self, stats):
        self.token_rate_label.setText(f"Token Rate: {stats.get('token_rate', '-')}")
        self.step_time_label.setText(f"Step Time: {stats.get('step_time', '-')} ms/step")
        self.detailed_memory_label.setText(f"Memory Usage: {stats.get('memory_usage', '-')}")

    @pyqtSlot()
    def clear_output(self):
        self.prompt_input.clear()
        self.opengl_viz_widget.set_visualization_type("Token Stream")
        self.opengl_viz_widget.set_color_scheme("Cool")
        self.opengl_viz_widget.set_token_shape("Circle")
        self.opengl_viz_widget.set_animation_speed(0.01)
        self.opengl_viz_widget.set_token_size(0.03)
        self.opengl_viz_widget.set_token_spacing(0.07)
        self.zoom_level_spin.setValue(1.0)
        self.opengl_viz_widget.set_zoom_level(1.0)
        self.token_data_mode_combo.setCurrentText("Decoded Tokens") # Reset token data mode

    @pyqtSlot(dict)
    def update_memory_status_bar(self, memory_stats):
        system_percent = memory_stats.get('system_percent', 0)
        gpu_percent = memory_stats.get('gpu_percent', 0)
        system_used = memory_stats.get('system_used', 0)
        system_total = memory_stats.get('system_total', 0)
        gpu_used = memory_stats.get('gpu_used', 0)
        gpu_total = memory_stats.get('gpu_total', 0)
        gpu_available = memory_stats.get('gpu_available', False)
        self.ram_indicator.setValue(int(system_percent))
        self.gpu_indicator.setValue(int(gpu_percent))
        self.system_ram_progress_sidebar.setValue(int(system_percent))
        self.gpu_vram_progress_sidebar.setValue(int(gpu_percent))
        self.system_ram_label_sidebar.setText(f"{system_used:.2f} / {system_total:.2f} GB ({system_percent:.0f}%)")
        self.gpu_vram_label_sidebar.setText(f"{gpu_used:.2f} / {gpu_total:.2f} GB ({gpu_percent:.0f}%)" if gpu_available else "N/A")

    def save_settings(self):
        settings = QSettings("MyCompany", "LLaDAGUI")
        settings.setValue("visualization_type", self.visualization_type_combo.currentText())
        settings.setValue("color_scheme", self.color_scheme_combo.currentText())
        settings.setValue("token_shape", self.token_shape_combo.currentText())
        settings.setValue("animation_speed", self.animation_speed_spin.value())
        settings.setValue("token_size", self.token_size_spin.value())
        settings.setValue("token_spacing", self.token_spacing_spin.value())
        settings.setValue("zoom_level", self.zoom_level_spin.value())
        settings.setValue("gen_length", self.gen_length_spin.value())
        settings.setValue("steps", self.steps_spin.value())
        settings.setValue("block_length", self.block_length_spin.value())
        settings.setValue("temperature", self.temperature_spin.value())
        settings.setValue("cfg_scale", self.cfg_scale_spin.value())
        settings.setValue("remasking", self.remasking_combo.currentText())
        settings.setValue("device", "cuda" if self.gpu_radio.isChecked() else "cpu")
        settings.setValue("use_8bit", self.use_8bit.isChecked())
        settings.setValue("use_4bit", self.use_4bit.isChecked())
        settings.setValue("extreme_mode", self.extreme_mode_checkbox.isChecked())
        settings.setValue("fast_mode", self.fast_mode_checkbox.isChecked())
        settings.setValue("use_memory", self.enable_memory_checkbox.isChecked())
        settings.setValue("token_stream_data_mode", self.token_data_mode_combo.currentText()) # Save token data mode

    def load_settings(self):
        settings = QSettings("MyCompany", "LLaDAGUI")
        self.visualization_type_combo.setCurrentText(settings.value("visualization_type", "Token Stream"))
        self.color_scheme_combo.setCurrentText(settings.value("color_scheme", "Cool"))
        self.token_shape_combo.setCurrentText(settings.value("token_shape", "Circle"))
        self.animation_speed_spin.setValue(float(settings.value("animation_speed", 0.01)))
        self.token_size_spin.setValue(float(settings.value("token_size", 0.03)))
        self.token_spacing_spin.setValue(float(settings.value("token_spacing", 0.07)))
        self.zoom_level_spin.setValue(float(settings.value("zoom_level", 1.0)))
        self.gen_length_spin.setValue(int(settings.value("gen_length", DEFAULT_PARAMS['gen_length'])))
        self.steps_spin.setValue(int(settings.value("steps", DEFAULT_PARAMS['steps'])))
        self.block_length_spin.setValue(int(settings.value("block_length", DEFAULT_PARAMS['block_length'])))
        self.temperature_spin.setValue(float(settings.value("temperature", DEFAULT_PARAMS['temperature'])))
        self.cfg_scale_spin.setValue(float(settings.value("cfg_scale", DEFAULT_PARAMS['cfg_scale'])))
        self.remasking_combo.setCurrentText(settings.value("remasking", DEFAULT_PARAMS['remasking']))
        device = settings.value("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and torch.cuda.is_available(): self.gpu_radio.setChecked(True)
        else: self.cpu_radio.setChecked(True)
        self.use_8bit.setChecked(bool(settings.value("use_8bit", True)))
        self.use_4bit.setChecked(bool(settings.value("use_4bit", False)))
        self.extreme_mode_checkbox.setChecked(bool(settings.value("extreme_mode", False)))
        self.fast_mode_checkbox.setChecked(bool(settings.value("fast_mode", False)))
        self.enable_memory_checkbox.setChecked(bool(settings.value("use_memory", False)))
        self.token_data_mode_combo.setCurrentText(settings.value("token_stream_data_mode", "Decoded Tokens")) # Load token data mode

    def closeEvent(self, event):
        self.save_settings()
        self.memory_monitor.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = LLaDAGUINew()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
