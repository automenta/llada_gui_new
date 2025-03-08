#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the LLaDA GUI application.
"""

import gc
import logging
import os
import traceback
import random
import numpy as np

import torch
import psutil

from core.config import LOCAL_MODEL_PATH, DEFAULT_MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device_status():
    """
    Retrieves and formats device status information (CPU and CUDA).

    Returns:
        dict: A dictionary containing device status information.
              Keys include 'cuda_available', 'cpu_available', 'device_count',
              and lists of 'cpu_info' and 'gpu_info' dictionaries.
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cpu_available': True,
        'device_count': 0,
        'gpu_info': [],
        'cpu_info': {}
    }

    # CPU Information
    memory = psutil.virtual_memory()
    device_info['cpu_info'] = {
        'total_memory': memory.total / (1024 ** 3),  # GB
        'used_memory': memory.used / (1024 ** 3),  # GB
        'memory_percent': memory.percent
    }

    # GPU Information (if CUDA is available)
    if device_info['cuda_available']:
        device_info['device_count'] = torch.cuda.device_count()
        for i in range(device_info['device_count']):
            gpu_properties = torch.cuda.get_device_properties(i)
            gpu_memory_usage = torch.cuda.memory_allocated(i) + torch.cuda.memory_reserved(i)
            gpu_device_info = {
                'name': torch.cuda.get_device_name(i),
                'index': i,
                'total_memory': gpu_properties.total_memory / (1024 ** 3),  # GB
                'used_memory': gpu_memory_usage / (1024 ** 3),  # GB
                'free_memory': (gpu_properties.total_memory - gpu_memory_usage) / (1024 ** 3),
                'used_percent': (gpu_memory_usage / gpu_properties.total_memory) * 100
            }
            device_info['gpu_info'].append(gpu_device_info)

    return device_info


def cleanup_gpu_memory():
    """
    Attempts to clean up GPU memory by clearing CUDA cache and triggering garbage collection.
    """
    logger.info("Initiating GPU memory cleanup...")

    for _ in range(2): # Reduced from 3 to 2 iterations; testing for efficacy
        gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            gc.collect() # Immediate garbage collection after cache clear

            used_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU memory usage post cleanup: {used_memory:.2f}GB / {total_memory:.2f}GB")

        except Exception as e:
            logger.warning(f"GPU memory cleanup encountered an issue: {e}")


def optimize_model_memory(model):
    """
    Applies memory optimization techniques to the given model.

    Currently focuses on disabling cache and enabling attention slicing if available.

    Args:
        model: The model to be optimized.

    Returns:
        The potentially optimized model.
    """
    if not torch.cuda.is_available():
        return model

    logger.info("Applying model memory optimizations...")

    try:
        if hasattr(model, "config"):
            model.config.use_cache = False  # Disable caching

        if hasattr(model, "enable_attention_slicing"):
            logger.info("Attention slicing enabled.")
            model.enable_attention_slicing() # No size argument; using default

        # Future LLaDA-specific optimizations can be added here if needed.
        # model = _apply_llada_optimizations(model) # Removed call to empty function

        cleanup_gpu_memory() # Clean up after optimizations

        return model

    except Exception as e:
        logger.warning(f"Model memory optimization error: {e}")
        return model


def get_model_path():
    """
    Determines the correct model path, prioritizing the local model path if it exists.

    Returns:
        str: The path to the model.
    """
    return LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else DEFAULT_MODEL_PATH


def format_error(exception):
    """
    Formats an exception into a readable error message with traceback.

    Args:
        exception: The exception object.

    Returns:
        str: Formatted error message string.
    """
    return f"Error: {str(exception)}\n\n{traceback.format_exc()}"


def format_memory_info(device_stats):
    """
    Formats memory statistics for system and GPU into human-readable strings.

    Args:
        device_stats (dict): Dictionary containing device statistics
                             as returned by get_device_status().

    Returns:
        tuple: (system_memory_text, gpu_memory_text) - formatted memory info strings.
               gpu_memory_text will be "N/A" if GPU is not available.
    """
    cpu_stats = device_stats.get('cpu_info', {})
    gpu_stats_list = device_stats.get('gpu_info', [])

    system_text = (f"{cpu_stats.get('used_memory', 0):.2f} / {cpu_stats.get('total_memory', 0):.2f} GB "
                   f"({cpu_stats.get('memory_percent', 0):.1f}%)")
    if gpu_stats_list:
        gpu_total = gpu_stats_list[0].get('total_memory', 0) # Assumes single GPU or aggregates.
        gpu_used = gpu_stats_list[0].get('used_memory', 0)
        gpu_percent = gpu_stats_list[0].get('used_percent', 0)
        gpu_text = f"{gpu_used:.2f} / {gpu_total:.2f} GB ({gpu_percent:.1f}%)"
    else:
        gpu_text = "N/A" # Or "No GPU Available" for clarity

    return system_text, gpu_text


def set_seeds(seed=42):
    """
    Sets the random seeds for Python's `random`, `numpy`, and `torch` (CPU & CUDA).

    Ensures reproducibility across different runs when using the same seed.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_directories():
    """
    Creates essential data directories if they do not already exist.

    Directories created: 'data', 'data/memory', 'data/models', 'data/models/onnx_models'.
    All directories are created relative to the repository root.

    Returns:
        bool: True if directories were created or already exist, False if an error occurred.
    """
    logger.info("Verifying and creating data directories...")
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directories = [
        os.path.join(repo_dir, "data"),
        os.path.join(repo_dir, "data", "memory"),
        os.path.join(repo_dir, "data", "models"),
        os.path.join(repo_dir, "data", "models", "onnx_models"),
    ]

    try:
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True) # No error if directory exists
            logger.debug(f"Directory verified/created: {dir_path}") # Debug level logging

        logger.info("Data directories verification complete.")
        return True

    except OSError as e:
        logger.error(f"Error creating data directories: {e}")
        return False
