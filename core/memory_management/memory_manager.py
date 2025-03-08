#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory Manager for LLaDA GUI - Handles GPU and system memory optimizations.

This module provides advanced memory management features including:
- Dynamic parameter adjustment based on available resources
- Memory usage tracking and warnings
- Automatic optimization selection
- Scheduled memory cleanup
"""

import gc
import logging
import threading
import time

import psutil
import torch

# Configure logging
logger = logging.getLogger(__name__)

# Singleton instance
_memory_manager = None


def get_memory_manager():
    """
    Get or create the global memory manager instance.
    
    Returns:
        MemoryManager: The singleton memory manager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


class MemoryManager:
    """
    Advanced memory management for LLaDA operations.
    
    This class provides comprehensive memory management capabilities to prevent
    out-of-memory errors and optimize performance based on available resources.
    """

    def __init__(self):
        """Initialize the memory manager."""
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.memory_listeners = []
        self.last_memory_stats = {}
        self.history = []
        self.history_max_len = 100  # Keep last 100 snapshots
        self.monitoring_interval = 1.0  # 1 second by default

        # Thresholds for warnings and adjustments
        self.warning_threshold = 90  # Percentage
        self.critical_threshold = 95  # Percentage
        self.adjustment_threshold = 85  # Percentage

        # Default adjustment parameters
        self.default_parameters = {
            'gen_length': {
                'min': 16,
                'default': 64,
                'max': 512,
                'step': 16,
                'memory_multiplier': 0.5  # How much memory this impacts (0-1)
            },
            'steps': {
                'min': 16,
                'default': 64,
                'max': 512,
                'step': 16,
                'memory_multiplier': 0.3
            },
            'block_length': {
                'min': 16,
                'default': 32,
                'max': 256,
                'step': 16,
                'memory_multiplier': 0.2
            }
        }

        # Initialize monitoring
        self.start_monitoring()

    def __del__(self):
        """Clean up resources."""
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(2)  # Wait up to 2 seconds

    def start_monitoring(self):
        """Start the memory monitoring thread."""
        with self.lock:
            if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
                self.stop_monitoring.clear()
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self.monitoring_thread.start()
                logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop the memory monitoring thread."""
        with self.lock:
            self.stop_monitoring.set()
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(2)  # Wait up to 2 seconds
                logger.info("Memory monitoring stopped")

    def _monitoring_loop(self):
        """Background thread for memory monitoring."""
        while not self.stop_monitoring.is_set():
            try:
                # Get current memory stats
                stats = self.get_memory_stats()

                # Store in history
                self.history.append((time.time(), stats))
                if len(self.history) > self.history_max_len:
                    self.history.pop(0)  # Remove oldest entry

                # Check for critical thresholds
                self._check_thresholds(stats)

                # Notify listeners
                self._notify_listeners(stats)

                # Store as last stats
                self.last_memory_stats = stats

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

            # Sleep until next check
            time.sleep(self.monitoring_interval)

    def _check_thresholds(self, stats):
        """
        Check memory usage against thresholds.
        
        Args:
            stats: Current memory statistics
        """
        # Check GPU memory if available
        if stats.get('gpu_available', False):
            gpu_percent = stats['gpu_percent']

            if gpu_percent > self.critical_threshold:
                logger.warning(f"CRITICAL: GPU memory usage at {gpu_percent:.1f}%")
                # Force garbage collection and memory cleanup
                self.force_cleanup()
            elif gpu_percent > self.warning_threshold:
                logger.warning(f"WARNING: High GPU memory usage at {gpu_percent:.1f}%")

        # Check system memory
        system_percent = stats.get('system_percent', 0)
        if system_percent > self.critical_threshold:
            logger.warning(f"CRITICAL: System memory usage at {system_percent:.1f}%")
            # Force garbage collection
            gc.collect()

    def _notify_listeners(self, stats):
        """
        Notify all registered memory listeners.
        
        Args:
            stats: Current memory statistics
        """
        for listener in self.memory_listeners:
            try:
                listener(stats)
            except Exception as e:
                logger.error(f"Error notifying memory listener: {e}")

    def add_listener(self, listener_func):
        """
        Add a memory statistics listener.
        
        Args:
            listener_func: Function to call with memory stats
        """
        if listener_func not in self.memory_listeners:
            self.memory_listeners.append(listener_func)

            # Immediately send current stats if available
            if self.last_memory_stats:
                try:
                    listener_func(self.last_memory_stats)
                except Exception as e:
                    logger.error(f"Error notifying new memory listener: {e}")

    def remove_listener(self, listener_func):
        """
        Remove a memory statistics listener.
        
        Args:
            listener_func: Listener function to remove
        """
        if listener_func in self.memory_listeners:
            self.memory_listeners.remove(listener_func)

    def get_memory_stats(self):
        """
        Get current memory statistics.
        
        Returns:
            dict: Memory statistics for system and GPU
        """
        stats = {
            'timestamp': time.time(),
            'system_available': True,
            'gpu_available': torch.cuda.is_available()
        }

        # Get system memory
        memory = psutil.virtual_memory()
        stats['system_total'] = memory.total / (1024 ** 3)  # GB
        stats['system_used'] = memory.used / (1024 ** 3)  # GB
        stats['system_free'] = memory.available / (1024 ** 3)  # GB
        stats['system_percent'] = memory.percent

        # Get GPU memory if available
        if stats['gpu_available']:
            try:
                device_id = torch.cuda.current_device()

                # Get allocated and reserved memory
                allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # GB
                total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)  # GB

                # Calculate used and free memory
                used = allocated + (reserved - allocated)  # Account for reserved but not allocated
                free = total - used

                stats['gpu_device'] = device_id
                stats['gpu_name'] = torch.cuda.get_device_name(device_id)
                stats['gpu_total'] = total
                stats['gpu_used'] = used
                stats['gpu_free'] = free
                stats['gpu_allocated'] = allocated
                stats['gpu_reserved'] = reserved
                stats['gpu_percent'] = (used / total) * 100 if total > 0 else 0

            except Exception as e:
                logger.error(f"Error getting GPU memory stats: {e}")
                stats['gpu_available'] = False

        return stats

    def force_cleanup(self):
        """Force memory cleanup operations."""
        logger.info("Forcing memory cleanup")

        # Run garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clean GPU memory if available
        if torch.cuda.is_available():
            try:
                # Empty cache
                torch.cuda.empty_cache()

                # Show memory after cleanup
                device_id = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
                used = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
                logger.info(f"GPU memory after cleanup: {used:.2f}GB / {total:.2f}GB")

            except Exception as e:
                logger.error(f"Error cleaning GPU memory: {e}")

    def optimize_parameters(self, current_params, available_memory_gb=None):
        """
        Dynamically optimize generation parameters based on available memory.
        
        Args:
            current_params: Current generation parameters
            available_memory_gb: Available GPU memory in GB (if None, will be detected)
            
        Returns:
            dict: Optimized parameters
        """
        if not torch.cuda.is_available():
            # No GPU, no need to optimize
            return current_params

        # Get memory stats if not provided
        if available_memory_gb is None:
            stats = self.get_memory_stats()
            if stats.get('gpu_available', False):
                available_memory_gb = stats.get('gpu_free', 0)
            else:
                # No GPU stats available
                return current_params

        logger.info(f"Optimizing parameters for {available_memory_gb:.2f}GB available memory")

        # Clone the parameters to avoid modifying the original
        optimized = dict(current_params)

        # Adjust parameters based on available memory
        if available_memory_gb < 2:  # Extreme constraint for very limited memory
            # Very limited memory
            logger.warning("Critically limited GPU memory available, applying aggressive optimizations")

            # Constrain most impactful parameters
            self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.75)
            self._adjust_parameter(optimized, 'steps', decrease_factor=0.75)
            self._adjust_parameter(optimized, 'block_length', decrease_factor=0.5)

            # Enable extreme memory optimizations
            optimized['use_4bit'] = True
            optimized['extreme_mode'] = True

        elif available_memory_gb < 4:
            # Very limited memory
            logger.warning("Very limited GPU memory available, applying strong optimizations")

            # Constrain most impactful parameters
            self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.5)
            self._adjust_parameter(optimized, 'steps', decrease_factor=0.5)

            # Enable extreme memory optimizations
            optimized['use_4bit'] = True
            optimized['extreme_mode'] = True

        elif available_memory_gb < 6:
            # Limited memory
            logger.info("Limited GPU memory available, applying moderate optimizations")

            # Apply moderate constraints
            self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.25)
            self._adjust_parameter(optimized, 'steps', decrease_factor=0.25)

            # Enable memory optimizations
            optimized['use_8bit'] = True

        elif available_memory_gb < 8:
            # Somewhat limited memory
            logger.info("Somewhat limited GPU memory available, applying light optimizations")

            # Apply very light constraints for larger models
            if optimized.get('gen_length', 64) > 256 or optimized.get('steps', 64) > 256:
                self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.1)
                self._adjust_parameter(optimized, 'steps', decrease_factor=0.1)
                optimized['use_8bit'] = True

        # For 12GB+ GPUs like RTX 3060, don't apply any automatic reductions unless generation
        # settings are extremely high
        elif available_memory_gb < 11:  # Approaching 12GB limit
            logger.info("Good GPU memory available (approaching limits for extreme settings)")

            # Only constrain extremely high values
            if optimized.get('gen_length', 64) > 384 and optimized.get('steps', 64) > 384:
                self._adjust_parameter(optimized, 'gen_length', decrease_factor=0.05)
                self._adjust_parameter(optimized, 'steps', decrease_factor=0.05)

        return optimized

    def _adjust_parameter(self, params, param_name, decrease_factor=None, target_value=None):
        """
        Adjust a specific parameter within its constraints.
        
        Args:
            params: Parameter dictionary to modify
            param_name: Name of parameter to adjust
            decrease_factor: Factor to decrease by (0.0-1.0)
            target_value: Specific target value (overrides decrease_factor)
            
        Returns:
            bool: True if parameter was adjusted
        """
        if param_name not in params or param_name not in self.default_parameters:
            return False

        # Get parameter constraints
        constraints = self.default_parameters[param_name]
        current_value = params[param_name]

        # Determine new value
        if target_value is not None:
            new_value = target_value
        elif decrease_factor is not None:
            # Calculate reduction but ensure it's at least one step
            reduction = max(int(current_value * decrease_factor), constraints['step'])
            new_value = current_value - reduction
        else:
            return False

        # Ensure new value is within constraints
        new_value = max(constraints['min'], min(constraints['max'], new_value))

        # Round to nearest step if needed
        if constraints['step'] > 1:
            new_value = round(new_value / constraints['step']) * constraints['step']

        # Apply the change if different
        if new_value != current_value:
            params[param_name] = new_value
            logger.info(f"Adjusted {param_name} from {current_value} to {new_value}")
            return True

        return False

    def estimate_memory_requirement(self, params):
        """
        Estimate the memory requirement for a generation with the given parameters.
        
        Args:
            params: Generation parameters
            
        Returns:
            float: Estimated memory requirement in GB
        """
        # Base memory requirement for the model
        base_memory_gb = 3.5  # Base memory for LLaDA-8B model - adjusted based on real observations

        # Adjustments based on quantization
        if params.get('use_4bit', False):
            base_memory_gb *= 0.5  # 4-bit requires ~50% of normal memory
        elif params.get('use_8bit', False):
            base_memory_gb *= 0.75  # 8-bit requires ~75% of normal memory

        # Dynamic adjustments based on generation parameters
        memory_gb = base_memory_gb

        # Generation length impacts memory - adjusted based on real observations for RTX 3060
        gen_length = params.get('gen_length', 64)
        memory_gb += (gen_length / 64) * 0.4  # Each 64 tokens adds ~0.4GB

        # Sampling steps impacts memory - adjusted based on real observations
        steps = params.get('steps', 64)
        memory_gb += (steps / 64) * 0.25  # Each 64 steps adds ~0.25GB

        # Block length impacts memory
        block_length = params.get('block_length', 32)
        memory_gb += (block_length / 32) * 0.15  # Each 32 tokens adds ~0.15GB

        # Extreme mode adjustments
        if params.get('extreme_mode', False):
            memory_gb *= 0.7  # Extreme mode can save ~30% memory

        logger.info(f"Estimated memory requirement: {memory_gb:.2f}GB for generation parameters")
        return memory_gb

    def can_fit_in_memory(self, params):
        """
        Check if the given parameters can fit in available memory.
        
        Args:
            params: Generation parameters
            
        Returns:
            bool: True if generation can fit in available memory
        """
        # Get current memory stats
        stats = self.get_memory_stats()

        # If no GPU available, assume we can fit (will use CPU)
        if not stats.get('gpu_available', False):
            return True

        # Get available GPU memory
        available_memory_gb = stats.get('gpu_free', 0)

        # Get estimated memory requirement
        estimated_memory_gb = self.estimate_memory_requirement(params)

        # Check if we have enough memory (with 0.5GB buffer)
        can_fit = available_memory_gb >= (estimated_memory_gb + 0.5)

        if not can_fit:
            logger.warning(
                f"Insufficient memory: Need {estimated_memory_gb:.2f}GB, "
                f"but only {available_memory_gb:.2f}GB available"
            )

        return can_fit

    def get_memory_usage_history(self, duration_seconds=None):
        """
        Get memory usage history for analysis.
        
        Args:
            duration_seconds: How many seconds of history to return (None for all)
            
        Returns:
            dict: Memory usage history data
        """
        with self.lock:
            if not self.history:
                return {'timestamps': [], 'system_percent': [], 'gpu_percent': []}

            # Filter by duration if specified
            if duration_seconds is not None:
                cutoff_time = time.time() - duration_seconds
                history = [(t, s) for t, s in self.history if t >= cutoff_time]
            else:
                history = self.history

            # Extract data
            timestamps = [t for t, _ in history]
            system_percent = [s.get('system_percent', 0) for _, s in history]
            gpu_percent = [s.get('gpu_percent', 0) for _, s in history
                           if s.get('gpu_available', False)]

            # Ensure gpu_percent is the same length as timestamps
            if len(gpu_percent) < len(timestamps):
                # Pad with zeros if no GPU data
                gpu_percent.extend([0] * (len(timestamps) - len(gpu_percent)))

            return {
                'timestamps': timestamps,
                'system_percent': system_percent,
                'gpu_percent': gpu_percent
            }

    def suggest_optimal_parameters(self, task_requirement=None):
        """
        Suggest optimal parameters based on current memory availability and task.
        
        Args:
            task_requirement: Task requirement level (1-5, where 5 is highest quality)
            
        Returns:
            dict: Suggested optimal parameters
        """
        # Get memory stats
        stats = self.get_memory_stats()

        # Start with default parameters
        params = {param: config['default'] for param, config in self.default_parameters.items()}

        # Add default device, quantization, etc.
        params['device'] = 'cuda' if stats.get('gpu_available', False) else 'cpu'
        params['use_8bit'] = False
        params['use_4bit'] = False
        params['extreme_mode'] = False

        # If task requirement specified, adjust base parameters
        if task_requirement is not None:
            if task_requirement >= 5:  # Highest quality
                params['gen_length'] = 256
                params['steps'] = 128
            elif task_requirement >= 4:
                params['gen_length'] = 192
                params['steps'] = 96
            elif task_requirement >= 3:
                params['gen_length'] = 128
                params['steps'] = 80
            elif task_requirement >= 2:
                params['gen_length'] = 96
                params['steps'] = 64
            # Level 1 uses defaults

        # If using GPU, optimize based on available memory
        if params['device'] == 'cuda':
            available_memory_gb = stats.get('gpu_free', 0)
            params = self.optimize_parameters(params, available_memory_gb)

        return params

    def get_memory_warning(self, params):
        """
        Get a warning message if memory usage would be problematic.
        
        Args:
            params: Generation parameters
            
        Returns:
            str or None: Warning message if memory usage would be problematic, None otherwise
        """
        if not torch.cuda.is_available() or params.get('device', 'cuda') != 'cuda':
            # No warning needed for CPU mode
            return None

        # Get current memory stats
        stats = self.get_memory_stats()
        if not stats.get('gpu_available', False):
            return None

        # Check available memory
        available_memory_gb = stats.get('gpu_free', 0)
        estimated_memory_gb = self.estimate_memory_requirement(params)

        # Check if we need a warning
        if available_memory_gb < estimated_memory_gb:
            # Critical warning - not enough memory
            return (
                f"Warning: Your current parameters require approximately {estimated_memory_gb:.1f}GB "
                f"of GPU memory, but only {available_memory_gb:.1f}GB is available. "
                f"This may cause out-of-memory errors during generation. "
                f"Consider reducing generation length, steps, or enabling memory optimizations."
            )
        elif available_memory_gb < (estimated_memory_gb * 1.2):
            # Warning - close to memory limit
            return (
                f"Caution: Your current parameters will use approximately {estimated_memory_gb:.1f}GB "
                f"of GPU memory. You have {available_memory_gb:.1f}GB available, which "
                f"is close to the limit. Consider enabling memory optimizations."
            )

        # No warning needed
        return None
