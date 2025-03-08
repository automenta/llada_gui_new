#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaDA Model Generation: worker thread
"""

import logging
from typing import Optional, Callable, Dict, Any
import time

import torch
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from transformers import AutoTokenizer, AutoModel

from core.utils import cleanup_gpu_memory, get_model_path, format_error
from core.generate import generate  # Import from our optimized generate.py
from core.generation_mode import GenerationMode # Import GenerationMode Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
MEMORY_WARNING_THRESHOLD_GB = 1.0

class LLaDAWorker(QThread):
    """Worker thread for handling LLaDA generation."""
    progress = pyqtSignal(int, str, dict)
    step_update = pyqtSignal(int, list, list, list)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    memory_warning = pyqtSignal(str)
    realtime_stats = pyqtSignal(dict) # Signal for realtime stats

    def __init__(self, prompt: str, config: Dict[str, Any], parent: Optional[Any] = None): # Type hints
        super().__init__(parent)
        self.prompt = prompt
        self.config = config
        self.stopped = False
        self.current_step = 0
        self.total_steps = config.get('steps', 64)
        self.mask_id = 126336  # Default mask token ID - Consider making configurable via config
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self._delayed_cleanup_gpu_memory)
        self.memory_cleanup_delay = config.get('memory_cleanup_delay', 300000) # Default 5 minutes in milliseconds
        self.generation_mode = GenerationMode.STANDARD # Default to STANDARD mode
        self.last_step_time = None
        self.step_times = []


    def stop(self):
        """Stop the generation process."""
        self.stopped = True
        if self.cleanup_timer.isActive():
            self.cleanup_timer.stop() # Stop timer if generation is stopped

    def update_progress(self, progress_percentage: float, tokens: Optional[torch.Tensor] = None):
        """
        Update progress callback from the generator.

        Args:
            progress_percentage: Float between 0 and 1 indicating progress
            tokens: Current token tensor
        """
        if self.stopped:
            return

        step = int(progress_percentage * 100)
        if step == self.current_step: # Prevent redundant updates
            return
        self.current_step = step

        # Emit progress update
        self.progress.emit(
            step,
            f"Generating: {step}% complete",
            {'partial_progress': step}
        )

        if tokens is None: # Early exit if no tokens for visualization
            return

        # Visualization data extraction - Moved visualization logic to a separate method
        self._emit_step_update_signal(tokens)
        self._emit_realtime_stats() # Emit realtime stats at each step


    def _emit_step_update_signal(self, tokens: torch.Tensor):
        """Extract visualization data and emit step_update signal."""
        try:
            token_ids = tokens[0].cpu().tolist()
            mask_indices = [1 if t == self.mask_id else 0 for t in token_ids]
            token_display = ["[MASK]" if t == self.mask_id else str(t) for t in token_ids] # List comprehension
            confidence_scores = [0.0 if m else 1.0 for m in mask_indices]
            mask_indices_bool = [bool(m) for m in mask_indices] # List comprehension

            self.step_update.emit(
                self.current_step,
                token_display,
                mask_indices_bool,
                confidence_scores
            )
        except Exception as e:
            logger.error(f"Error in step update: {e}")

    def _emit_realtime_stats(self):
        """Calculate and emit realtime statistics."""
        current_time = time.time()
        step_time = 0
        if self.last_step_time is not None:
            step_time = (current_time - self.last_step_time) * 1000 # in milliseconds
            self.step_times.append(step_time)
            if len(self.step_times) > 10: # Keep last 10 step times for moving average
                self.step_times.pop(0)
        self.last_step_time = current_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0


        tokens_per_sec = 0
        if avg_step_time > 0:
            tokens_per_sec = 1000 / avg_step_time # tokens per second, assuming 1 token per step

        memory_usage_str = "-"
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2) # MB
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2) # MB
            memory_usage_str = f"{memory_allocated:.2f}MB allocated, {memory_reserved:.2f}MB reserved"


        stats = {
            "token_rate": f"{tokens_per_sec:.2f} tokens/s",
            "step_time": f"{avg_step_time:.2f}",
            "memory_usage": memory_usage_str
        }
        self.realtime_stats.emit(stats)


    def run(self):
        model = None # Initialize model and tokenizer outside try block for broader scope
        tokenizer = None
        try:
            # Stop any existing timer in case of rapid re-generation requests
            if self.cleanup_timer.isActive():
                self.cleanup_timer.stop()

            # Determine device
            device = 'cuda' if torch.cuda.is_available() and self.config['device'] == 'cuda' else 'cpu'
            self.progress.emit(5, f"Starting with device: {device}", {})

            if device == 'cuda':
                # cleanup_gpu_memory() # REMOVED immediate cleanup - this was the cause of premature cleanup
                free_memory = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) - ((torch.cuda.memory_allocated(0) + torch.cuda.memory_reserved(0)) / (1024 ** 3)) # Simplified calculation
                if free_memory < MEMORY_WARNING_THRESHOLD_GB:
                    self.memory_warning.emit(f"Low GPU memory warning: Only {free_memory:.2f}GB available. CPU offloading will be enabled.")

            model_path = get_model_path()

            # Load tokenizer and model - Encapsulated loading into a separate method
            tokenizer, model = self._load_model_and_tokenizer(model_path, device)
            if model is None or tokenizer is None: # Handle loading failure
                return # _load_model_and_tokenizer already emitted error

            # Prepare input - Input preparation into a separate method
            input_ids = self._prepare_input(tokenizer, device)
            if input_ids is None: # Handle input preparation failure
                return # _prepare_input already emitted error

            # Generation parameters - Extract parameters directly from config
            gen_length = self.config.get('gen_length', 64)
            steps = self.config.get('steps', 64)
            block_length = self.config.get('block_length', 32)
            temperature = self.config.get('temperature', 0.0)
            cfg_scale = self.config.get('cfg_scale', 0.0)
            remasking = self.config.get('remasking', 'low_confidence')
            fast_mode = self.config.get('fast_mode', False) # Still reading fast_mode for backward compatibility
            if fast_mode:
                self.generation_mode = GenerationMode.FAST
            else:
                self.generation_mode = GenerationMode.STANDARD

            cpu_offload = device == 'cuda' and self.generation_mode == GenerationMode.STANDARD # CPU offload only in STANDARD mode
            adaptive_steps = True
            chunk_size = 256 if self.generation_mode == GenerationMode.FAST else 512
            confidence_threshold = 0.8 if self.generation_mode == GenerationMode.FAST else 0.9


            self.progress.emit(40, f"Starting generation in {self.generation_mode.value} mode (steps: {steps}, length: {gen_length})...", {
                'prompt_length': input_ids.shape[1],
                'params': { # Params dict is cleaner
                    'gen_length': gen_length, 'steps': steps, 'block_length': block_length,
                    'temperature': temperature, 'cfg_scale': cfg_scale, 'remasking': remasking,
                    'device': device, 'cpu_offload': cpu_offload, 'generation_mode': self.generation_mode.value,
                    'adaptive_steps': adaptive_steps, 'chunk_size': chunk_size
                }
            })

            # Initialize step timing
            self.last_step_time = time.time()
            self.step_times = []


            # Generate text
            out = generate(
                model=model, prompt=input_ids, steps=steps, gen_length=gen_length,
                block_length=block_length, temperature=temperature, cfg_scale=cfg_scale,
                remasking=remasking, progress_callback=self.update_progress, cpu_offload=cpu_offload,
                mask_id=self.mask_id, adaptive_steps=adaptive_steps, chunk_size=chunk_size,
                confidence_threshold=confidence_threshold
            )

            if self.stopped:
                self.error.emit("Generation cancelled.")
                return

            self.progress.emit(95, "Decoding output...", {})
            answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            self.progress.emit(100, "Generation complete", {'output': answer})
            self.finished.emit(answer)

            # Start the delayed cleanup timer after successful generation
            if device == 'cuda':
                self.cleanup_timer.start(self.memory_cleanup_delay)


        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            self.error.emit(format_error(e))
            if self.cleanup_timer.isActive():
                self.cleanup_timer.stop() # Stop timer in case of error

        # finally block removed - cleanup is now handled by timer


    def _load_model_and_tokenizer(self, model_path: str, device: str) -> tuple[AutoTokenizer, AutoModel]:
        """Loads the tokenizer and model, emitting progress signals and handling errors."""
        tokenizer = None
        model = None
        try:
            self.progress.emit(10, "Loading tokenizer...", {})
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True, cache_dir="data")
            if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                self.mask_id = tokenizer.mask_token_id # Update mask_id from tokenizer if available
            # tokenizer.mask_token_id = tokenizer.mask_token_id or MASK_TOKEN_ID_DEFAULT # Alternative default setting
            self.progress.emit(15, f"Loading model (device: {device})...", {})
            dtype = torch.float16 if device == 'cuda' else torch.float32 # Consistent dtype setting
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=dtype,
                device_map="auto" if device == 'cuda' else None, cache_dir="data", resume_download=True
            )
            model.tie_weights()  # Explicitly tie weights
            if device == 'cpu':
                model = model.to('cpu')
            model.eval() # Ensure eval mode

            self.progress.emit(25, "Model loaded successfully", {})
            return tokenizer, model

        except Exception as e:
            error_msg = f"Error loading model: {format_error(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            return None, None # Indicate loading failure


    def _prepare_input(self, tokenizer: AutoTokenizer, device: str) -> Optional[torch.Tensor]:
        """Tokenizes the input prompt and handles potential errors."""
        try:
            self.progress.emit(30, "Tokenizing input...", {})

            messages = [{"role": "user", "content": self.prompt}]

            user_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(user_input)['input_ids'] # Consider pre-compiling tokenizer for speed
            input_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # Create tensor once

            return input_ids_tensor

        except Exception as e:
            error_msg = f"Error preparing input: {format_error(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            return None # Indicate input preparation failure

    def _delayed_cleanup_gpu_memory(self):
        """Perform delayed GPU memory cleanup."""
        logger.info("Performing delayed GPU memory cleanup...")
        cleanup_gpu_memory()
        self.cleanup_timer.stop() # Ensure timer is stopped after cleanup
