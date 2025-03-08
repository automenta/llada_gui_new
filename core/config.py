#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration and constants for the LLaDA GUI application.
"""

import os
import sys

# Base paths for the application
APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLADA_REPO_PATH = "/home/ty/Repositories/ai_workspace/llada_gui"

# Model information
DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
LOCAL_MODEL_PATH = os.path.join(LLADA_REPO_PATH, "GSAI-ML_LLaDA-8B-Instruct")

# Add paths to Python path
sys.path.append(APP_PATH)
sys.path.append(os.path.join(APP_PATH, "core"))
sys.path.append(os.path.join(APP_PATH, "gui"))

# Default generation parameters
DEFAULT_PARAMS = {
    # Optimized parameters for better performance on RTX 3060 and similar GPUs
    'gen_length': 128,  # Increased from 64 for better quality on 12GB GPUs
    'steps': 128,  # Increased to match gen_length for optimal diffusion balance
    'block_length': 32,
    'temperature': 1.0,  # Standard temperature for creative text
    'cfg_scale': 3.0,  # Standard CFG scale value for guided generation
    'remasking': 'low_confidence',
}

# Memory optimization constants
OPTIMIZED_GPU_MEMORY = True
CACHE_PRECISION = "bfloat16"  # Use bfloat16 for better performance with minimal precision loss
ENABLE_ATTENTION_SLICING = True  # Slice attention for lower memory usage
ENABLE_FLASH_ATTENTION = True  # Use flash attention if available

# Memory monitoring constants
MEMORY_CHECK_INTERVAL = 1.0  # seconds
MEMORY_WARNING_THRESHOLD = 90  # percentage
MEMORY_CAUTION_THRESHOLD = 75  # percentage
CRITICAL_GPU_MEMORY_THRESHOLD = 0.3  # GB

# Memory integration constants
MEMORY_ENABLED = True
MEMORY_SERVER_PORT = 3000
MEMORY_DATA_DIR = os.path.join(APP_PATH, "data", "memory")

# UI-related constants
WINDOW_TITLE = "LLaDA GUI - Large Language Diffusion Model"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 800  # Reduced height to fit better on typical screens
SPLITTER_RATIO = [250, 550]  # Adjusted ratio for better proportions

# Create necessary directories
os.makedirs(MEMORY_DATA_DIR, exist_ok=True)
