#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized generation algorithm for LLaDA models.
This version implements several optimizations:
1. Memory-efficient processing with CPU offloading
2. Adaptive step scheduling
3. Memory guidance integration
"""

import logging
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
MASK_TOKEN_ID_DEFAULT = 126336  # Default mask token ID
CONFIDENCE_THRESHOLD_DEFAULT = 0.9  # Default confidence threshold
EPSILON = 1e-10  # Small value to avoid log(0)


class TokenBuffer:
    """Memory-efficient token buffer that can offload to CPU when needed."""

    def __init__(self, data, device=DEVICE, cpu_offload=True):
        """Initialize buffer with data on the specified device."""
        self.cpu_offload = cpu_offload
        self.device = device
        self._data = data.to(device if not cpu_offload else CPU_DEVICE)
        self._is_on_gpu = not cpu_offload

    @property
    def data(self) -> torch.Tensor:
        """Get data, moving to GPU if needed."""
        if not self._is_on_gpu and self.cpu_offload:
            self._data = self._data.to(self.device)
            self._is_on_gpu = True
        return self._data

    def update(self, data):
        """Update buffer with new data."""
        if self._is_on_gpu or not self.cpu_offload:
            self._data = data
        else:
            self._data = data.to(CPU_DEVICE)

    def to_cpu(self):
        """Move data to CPU if not already there."""
        if self._is_on_gpu and self.cpu_offload:
            self._data = self._data.to(CPU_DEVICE)
            self._is_on_gpu = False

    def to_gpu(self):
        """Move data to GPU if not already there."""
        if not self._is_on_gpu and self.cpu_offload:
            self._data = self._data.to(self.device)
            self._is_on_gpu = True

    def clone(self) -> 'TokenBuffer':
        """Create a clone of the buffer."""
        return TokenBuffer(self._data.clone(), self.device, self.cpu_offload)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Add Gumbel noise for sampling categorical distributions.

    Args:
        logits: Raw logits from the model
        temperature: Sampling temperature

    Returns:
        Logits with Gumbel noise applied
    """
    if temperature <= 0:
        return logits

    dtype = torch.float32
    if logits.dtype == torch.bfloat16:
        logits = logits.to(dtype)
    elif logits.dtype != dtype:
        logits = logits.to(dtype)

    noise = torch.rand_like(logits, dtype=dtype)
    gumbel_noise = (-torch.log(noise + EPSILON)) ** temperature
    return logits / gumbel_noise


def get_adaptive_transfer_schedule(mask_index: torch.Tensor, steps: int, min_steps: int = 4,
                                   confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT) -> torch.Tensor:
    """
    Create an adaptive schedule for token transfers, front-loading more tokens in earlier steps.

    Args:
        mask_index: Boolean tensor indicating masked tokens
        steps: Number of steps to schedule
        min_steps: Minimum number of steps (default: 4)
        confidence_threshold: Confidence threshold for token transfer (default: 0.9)

    Returns:
        Tensor of shape (batch_size, steps) with number of tokens to transfer per step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)

    weights = torch.linspace(1.5, 0.5, steps)
    weights = weights / weights.sum() * steps

    for i in range(mask_num.size(0)):
        weighted_tokens = (weights * (mask_num[i].item() / steps)).round().to(torch.int64)
        diff = mask_num[i].item() - weighted_tokens.sum().item()
        if diff > 0:
            for j in range(diff):
                weighted_tokens[j % steps] += 1
        elif diff < 0:
            for j in range(-diff):
                index = -(j % steps) - 1
                if weighted_tokens[index] > 0:
                    weighted_tokens[index] -= 1
        num_transfer_tokens[i] = weighted_tokens

    return num_transfer_tokens


def chunk_processing(model, tokens: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """
    Process tokens in chunks to reduce memory usage for long sequences.

    Args:
        model: Language model
        tokens: Input token tensor
        chunk_size: Size of each chunk (default: 512)

    Returns:
        Concatenated logits from model output
    """
    seq_len = tokens.shape[1]
    if seq_len <= chunk_size:
        return model(tokens).logits

    all_logits = []
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        chunk = tokens[:, i:end_idx]
        with torch.no_grad():
            chunk_output = model(chunk).logits
        all_logits.append(chunk_output)
    return torch.cat(all_logits, dim=1)


def process_model(model, x, cfg_scale, chunk_size, mask_id, prompt_index):
    """
    Process the sequence through the model with optional classifier-free guidance.

    Args:
        model: Language model
        x: Current sequence tokens
        cfg_scale: Classifier-free guidance scale
        chunk_size: Size of chunks for processing
        mask_id: Mask token ID
        prompt_index: Boolean tensor indicating prompt positions

    Returns:
        Logits from the model
    """
    if cfg_scale > 0:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        x_ = torch.cat([x, un_x], dim=0)
        logits = chunk_processing(model, x_, chunk_size=chunk_size)
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

    logits = chunk_processing(model, x, chunk_size=chunk_size)
    return logits


def apply_memory_guidance(logits, x, memory_integration, mask_index):
    """
    Apply memory guidance to adjust logits based on external memory.

    Args:
        logits: Model logits
        x: Current sequence tokens
        memory_integration: Memory integration module (optional)
        mask_index: Boolean tensor indicating masked tokens

    Returns:
        Adjusted logits
    """
    if memory_integration and memory_integration.is_initialized():
        token_probs = F.softmax(logits, dim=-1)
        token_sequence = x[0].cpu().numpy().tolist()
        for pos in range(x.size(1)):
            if mask_index[0, pos]:
                pos_probs = token_probs[0, pos].cpu().numpy()
                guided_probs = memory_integration.apply_memory_guidance(
                    pos_probs, token_sequence, token_probs.size(-1)
                )
                token_probs[0, pos] = torch.tensor(guided_probs, device=token_probs.device, dtype=token_probs.dtype)
        logits = torch.log(token_probs + EPSILON)
    return logits


def select_tokens_to_transfer(confidence, mask_index, num_transfer_tokens, step):
    """
    Select tokens to transfer based on confidence scores and schedule.

    Args:
        confidence: Confidence scores for each token
        mask_index: Boolean tensor indicating masked tokens
        num_transfer_tokens: Tensor with number of tokens to transfer per step
        step: Current step index

    Returns:
        Boolean tensor indicating tokens to transfer
    """
    batch_size = confidence.shape[0]
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool, device=confidence.device)
    for j in range(batch_size):
        masked_positions = mask_index[j].nonzero().squeeze(1)
        num_masked = masked_positions.size(0)
        tokens_to_transfer = num_transfer_tokens[j, step].item()
        if tokens_to_transfer > 0 and num_masked > 0:
            k = min(tokens_to_transfer, num_masked)
            _, select_index = torch.topk(confidence[j, masked_positions], k=k)
            transfer_index[j, masked_positions[select_index]] = True
    return transfer_index


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: int = MASK_TOKEN_ID_DEFAULT,
    cpu_offload: bool = True,
    adaptive_steps: bool = True,
    progress_callback: Optional[Callable[[float, torch.Tensor], None]] = None,
    memory_integration=None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    chunk_size: int = 512,
    device: torch.device = DEVICE
) -> torch.Tensor:
    """
    Generate text using the LLaDA model with optimized memory and step scheduling.

    Args:
        model: Language model
        prompt: Input prompt tensor (shape: [1, seq_len])
        steps: Maximum number of sampling steps (default: 128)
        gen_length: Length of text to generate (default: 128)
        block_length: Size of generation blocks (default: 128)
        temperature: Sampling temperature (default: 0.0)
        cfg_scale: Classifier-free guidance scale (default: 0.0)
        remasking: Remasking strategy ('low_confidence' or 'random', default: 'low_confidence')
        mask_id: Mask token ID (default: MASK_TOKEN_ID_DEFAULT)
        cpu_offload: Enable CPU offloading to save GPU memory (default: True)
        adaptive_steps: Use adaptive step scheduling (default: True)
        progress_callback: Optional callback for progress updates
        memory_integration: Optional memory integration module
        confidence_threshold: Confidence threshold (default: CONFIDENCE_THRESHOLD_DEFAULT)
        chunk_size: Chunk size for processing long sequences (default: 512)
        device: Device for computation (default: DEVICE)

    Returns:
        Generated token tensor (shape: [1, prompt_len + gen_length])
    """
    model.eval()

    # Adjust gen_length to be divisible by block_length
    if gen_length % block_length != 0:
        gen_length = ((gen_length + block_length // 2) // block_length) * block_length
        logger.warning(f"Adjusted gen_length to {gen_length} to be divisible by block_length {block_length}")

    # Initialize sequence and token buffer
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)
    token_buffer = TokenBuffer(x, device=device, cpu_offload=cpu_offload)
    mask_index = (x == mask_id)

    num_blocks = gen_length // block_length

    # Determine steps per block
    if not adaptive_steps:
        steps_per_block = steps // num_blocks
        if steps % num_blocks != 0:
            steps = steps_per_block * num_blocks
            logger.warning(f"Adjusted steps to {steps} to be divisible by num_blocks {num_blocks}")
    else:
        steps_per_block_list = [max(4, int(steps / num_blocks * (0.8 ** b))) for b in range(num_blocks)]
        total_steps = sum(steps_per_block_list)
        if total_steps != steps:
            steps_per_block_list = [max(4, int(s * steps / total_steps)) for s in steps_per_block_list]

    total_steps_expected = steps if not adaptive_steps else sum(steps_per_block_list)
    total_steps_completed = 0

    for num_block in range(num_blocks):
        steps_per_block = steps_per_block_list[num_block] if adaptive_steps else steps_per_block
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length
        block_mask_index = mask_index[:, block_start:block_end]

        if not block_mask_index.any():
            continue

        # Compute transfer schedule
        num_transfer_tokens = (
            get_adaptive_transfer_schedule(block_mask_index, steps_per_block, confidence_threshold=confidence_threshold)
            if adaptive_steps else
            torch.div(block_mask_index.sum(dim=1, keepdim=True), steps_per_block, rounding_mode='floor').repeat(1, steps_per_block)
        )

        for i in range(steps_per_block):
            x = token_buffer.data  # Automatically moves to GPU if needed

            if progress_callback:
                total_steps_completed += 1
                progress_callback(total_steps_completed / total_steps_expected, x.clone())

            if not mask_index.any():
                break

            # Model processing and guidance
            logits = process_model(model, x, cfg_scale, chunk_size, mask_id, prompt_index)
            logits = apply_memory_guidance(logits, x, memory_integration, mask_index)

            # Sample tokens
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Compute confidence for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1) if temperature > 0 else F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, dtype=torch.float, device=x0.device)
            else:
                raise NotImplementedError(f"Unsupported remasking strategy: {remasking}")

            # Restrict confidence to current block
            confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=x.device))
            confidence[:, :block_start] = -float('inf')
            confidence[:, block_end:] = -float('inf')

            # Select and transfer tokens
            transfer_index = select_tokens_to_transfer(confidence, mask_index, num_transfer_tokens, i)
            x[transfer_index] = x0[transfer_index]
            mask_index[transfer_index] = False
            token_buffer.update(x)

            # Early exit if block is fully generated
            if not mask_index[:, block_start:block_end].any():
                break

            # Offload to CPU between steps
            if cpu_offload and i < steps_per_block - 1:
                gpu_offload_to_cpu(token_buffer)

        if cpu_offload:
            gpu_offload_to_cpu(token_buffer)

    token_buffer.to_gpu()
    return token_buffer.data


def gpu_offload_to_cpu(token_buffer):
    token_buffer.to_cpu()
    torch.cuda.empty_cache()
