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
        self.memory_influence_data = None # Placeholder for memory influence data
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
        self.setMouseTracking(True) # Enable mouse tracking for events even when no button is pressed
        self.last_mouse_pos = None # For mouse tracking


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

    def set_memory_influence_data(self, data):
        """Set data for memory influence map visualization."""
        self.memory_influence_data = data
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
        """Draw Memory Influence Map visualization."""
        grid_size = 10 # Example grid size
        cell_size = 2.0 / grid_size # Normalized cell size

        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate position for each cell
                x = -1.0 + col * cell_size + cell_size / 2.0
                y = 1.0 - row * cell_size - cell_size / 2.0

                # Generate a random influence value for demonstration
                influence = random.random() # 0.0 to 1.0

                # Choose color based on influence (example: grayscale)
                gray_scale = 0.3 + influence * 0.7 # Map 0-1 to gray scale range
                glColor3f(gray_scale, gray_scale, gray_scale)

                # Draw a square for each cell
                glBegin(GL_QUADS)
                glVertex2f(x - cell_size / 2.0, y - cell_size / 2.0) # bottom-left
                glVertex2f(x + cell_size / 2.0, y - cell_size / 2.0) # bottom-right
                glVertex2f(x + cell_size / 2.0, y + cell_size / 2.0) # top-right
                glVertex2f(x - cell_size / 2.0, y + cell_size / 2.0) # top-left
                glEnd()


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
        zoom_factor = 0.15 # Zoom sensitivity - increased for more effect
        delta = event.angleDelta().y()
        if delta > 0: # Zoom in
            self.zoom_level *= (1.0 - zoom_factor)
        elif delta < 0: # Zoom out
            self.zoom_level *= (1.0 + zoom_factor)
        self.zoom_level = max(0.1, min(self.zoom_level, 5.0)) # Clamp zoom level
        self.zoom_level_spin.setValue(self.zoom_level) # Update spinbox to reflect zoom
        self.update() # Trigger repaint

    def mousePressEvent(self, event):
        """Handle mouse press events for panning."""
        if event.button() == Qt.MouseButton.LeftButton: # Left button for pan
            self.last_mouse_pos = event.pos() # Store position


    def mouseMoveEvent(self, event):
        """Handle mouse move events for panning."""
        if self.last_mouse_pos is not None: # If left button is pressed and dragging
            current_mouse_pos = event.pos()
            delta_x = current_mouse_pos.x() - self.last_mouse_pos.x()
            delta_y = current_mouse_pos.y() - self.last_mouse_pos.y()

            # Calculate pan speed based on zoom level - zoomed in -> slower pan
            pan_speed = 0.005 * self.zoom_level # Adjust pan speed factor as needed

            self.pan_x += delta_x * pan_speed
            self.pan_y -= delta_y * pan_speed # Invert Y axis for natural drag

            self.last_mouse_pos = current_mouse_pos # Update last mouse position
            self.update() # Trigger repaint


    def mouseReleaseEvent(self, event):
        """Handle mouse release events for panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None # Clear last mouse position on release


class LLaDAGUINew(QMainWindow):
    """New OpenGL Visualization-Centric GUI for LLaDA application."""

    # ... (rest of LLaDAGUINew class is unchanged)
