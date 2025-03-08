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
from typing import List, Optional, Callable, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Configure a default device based on availability - Define constants at the top
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
MASK_TOKEN_ID_DEFAULT = 126336 # Define default mask token ID as constant
CONFIDENCE_THRESHOLD_DEFAULT = 0.9 # Define default confidence threshold

class TokenBuffer:
    """Memory-efficient token buffer that can offload to CPU when needed."""

    def __init__(self,  torch.Tensor, device: torch.device = DEVICE, cpu_offload: bool = True):
        """Initialize buffer with data, optionally on a specific device."""
        self.cpu_offload = cpu_offload
        self.device = device
        self._data = data.to(self.device if not cpu_offload else CPU_DEVICE)
        self._is_on_gpu = not cpu_offload

    @property
    def data(self) -> torch.Tensor:
        """Get data, moving to GPU if needed."""
        if not self._is_on_gpu and self.cpu_offload:
            self._data = self._data.to(self.device)
            self._is_on_gpu = True
        return self._data

    def update(self,  torch.Tensor):
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


class AttentionCache:
    """Cache for past key/values to avoid recomputation."""

    def __init__(self, model, cpu_offload: bool = True):
        """Initialize empty cache for a specific model."""
        self.model = model
        self.cache: Dict[str, Union[torch.Tensor, None]] = {} # Explicitly type cache
        self.cpu_offload = cpu_offload

    def get(self, key: str, default=None) -> Union[torch.Tensor, None]:
        """Get a value from the cache, loading to the correct device if needed."""
        if key not in self.cache:
            return default

        # Load from CPU if needed
        if self.cpu_offload and isinstance(self.cache[key], torch.Tensor) and self.cache[key].device.type == "cpu":
            self.cache[key] = self.cache[key].to(DEVICE)

        return self.cache[key]

    def set(self, key: str, value: torch.Tensor):
        """Set a value in the cache, offloading to CPU if configured."""
        self.cache[key] = value.to(CPU_DEVICE) if self.cpu_offload and isinstance(value, torch.Tensor) and value.device.type != "cpu" else value

    def clear(self):
        """Clear the cache."""
        self.cache = {}


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Add Gumbel noise for sampling categorical distributions.

    Args:
        logits: Raw logits from the model
        temperature: Sampling temperature

    Returns:
        Logits with Gumbel noise applied
    """
    if temperature <= 0: # Simplified condition
        if logits.dtype == torch.bfloat16:
            return logits.to(torch.float16) # Consistent return type
        return logits

    # For non-zero temperatures, use float32 for better compatibility
    dtype = torch.float32
    # Convert bfloat16 to float32 if needed
    if logits.dtype == torch.bfloat16:
        logits = logits.to(torch.float32)
    elif logits.dtype != torch.float32:
        logits = logits.to(dtype)

    noise = torch.rand_like(logits, dtype=dtype)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_adaptive_transfer_schedule(mask_index: torch.Tensor, steps: int, min_steps: int = 4, confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT) -> torch.Tensor:
    """
    Creates an adaptive schedule for token transfers, using fewer steps for easier tokens.

    Args:
        mask_index: Boolean tensor indicating masked tokens
        steps: Maximum number of steps to use
        min_steps: Minimum number of steps to use
        confidence_threshold: Threshold for considering a token "confident"

    Returns:
        Tensor containing number of tokens to transfer at each step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    # Create the transfer schedule, front-loading more tokens in earlier steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)

    # Weighted distribution - transfer more tokens in early steps
    weights = torch.linspace(1.5, 0.5, steps)
    weights = weights / weights.sum() * steps

    for i in range(mask_num.size(0)):
        # Distribute tokens according to weights, ensuring we use all tokens
        weighted_tokens = (weights * (mask_num[i].item() / steps)).round().to(torch.int64)

        # Ensure we assign all tokens - Simplified logic
        diff = mask_num[i].item() - weighted_tokens.sum().item()
        if diff > 0:
            for j in range(diff):
                weighted_tokens[j % steps] += 1
        elif diff < 0:
            for j in range(-diff):
                if weighted_tokens[-(j % steps) - 1] > 0:
                    weighted_tokens[-(j % steps) - 1] -= 1

        num_transfer_tokens[i] = weighted_tokens

    return num_transfer_tokens


def chunk_processing(model, tokens: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """
    Process tokens in chunks to reduce memory usage.

    Args:
        model: The language model
        tokens: Input tokens
        chunk_size: Size of chunks to process

    Returns:
        Model output logits
    """
    seq_len = tokens.shape[1]

    if seq_len <= chunk_size: # Simplified condition
        return model(tokens).logits

    all_logits = []
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        chunk = tokens[:, i:end_idx]

        with torch.no_grad():
            chunk_output = model(chunk).logits

        all_logits.append(chunk_output)

    return torch.cat(all_logits, dim=1)


@torch.no_grad()
def generate(
        model,
        prompt: torch.Tensor,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        mask_id: int = MASK_TOKEN_ID_DEFAULT, # Use constant
        cpu_offload: bool = True,
        adaptive_steps: bool = True,
        progress_callback: Optional[Callable[[float, torch.Tensor], None]] = None,
        memory_integration = None, # Type hint removed for forward compatibility
        confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT, # Use constant
        chunk_size: int = 512,
        memory_weight: float = 0.3, # Unused parameter, consider removing if truly unused
        tokenizer = None, # Type hint removed for forward compatibility
        device: torch.device = DEVICE
) -> torch.Tensor:
    """
    Optimized generation function for LLaDA models.

    Args:
        model: The language model
        prompt: Input prompt tokens
        steps: Maximum number of sampling steps
        gen_length: Length of the generated text
        block_length: Block size for generation
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        remasking: Strategy for remasking tokens ('low_confidence' or 'random')
        mask_id: Token ID for the mask token
        cpu_offload: Whether to offload tensors to CPU when not in use
        adaptive_steps: Whether to use adaptive step scheduling
        progress_callback: Callback function for progress updates
        memory_integration: Optional memory integration module
        confidence_threshold: Confidence threshold for early stopping
        chunk_size: Size of chunks for processing long sequences
        device: Device to use for computation

    Returns:
        Generated tokens
    """
    model.eval()  # Ensure model is in evaluation mode

    # Validate parameters - Moved validation to the beginning
    if gen_length % block_length != 0:
        gen_length = ((gen_length + block_length // 2) // block_length) * block_length
        logger.warning(f"Adjusted gen_length to {gen_length} to be divisible by block_length {block_length}")

    model_device = next(model.parameters()).device
    cpu_offload = cpu_offload and model_device.type == "cuda"

    # Initialize output tensor and token buffer
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id) # Keep track of prompt indices
    token_buffer = TokenBuffer(x, device=device, cpu_offload=cpu_offload)

    num_blocks = gen_length // block_length

    if not adaptive_steps:
        if steps % num_blocks != 0:
            steps_per_block = steps // num_blocks
            steps = steps_per_block * num_blocks
            logger.warning(f"Adjusted steps to {steps} to be divisible by num_blocks {num_blocks}")
        else:
            steps_per_block = steps // num_blocks
    else:
        steps_per_block_list = [max(4, int(steps / num_blocks * (0.8 ** b))) for b in range(num_blocks)] # List comprehension
        total_steps = sum(steps_per_block_list)
        if total_steps != steps:
            scaling_factor = steps / total_steps
            steps_per_block_list = [max(4, int(s * scaling_factor)) for s in steps_per_block_list]

    cache = AttentionCache(model, cpu_offload=cpu_offload)
    total_steps_completed = 0
    total_steps_expected = steps if not adaptive_steps else sum(steps_per_block_list)

    for num_block in range(num_blocks):
        steps_per_block = steps_per_block_list[num_block] if adaptive_steps else steps_per_block # Conditional assignment

        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        token_buffer.to_gpu()
        x = token_buffer.data
        block_mask_index = (x[:, block_start:block_end] == mask_id)

        if not block_mask_index.any():
            continue

        num_transfer_tokens = get_adaptive_transfer_schedule(
            block_mask_index,
            steps_per_block,
            min_steps=4,
            confidence_threshold=confidence_threshold
        ) if adaptive_steps else torch.div(
            block_mask_index.sum(dim=1, keepdim=True),
            steps_per_block,
            rounding_mode='floor'
        ).repeat(1, steps_per_block) # Conditional num_transfer_tokens calculation

        if not adaptive_steps and block_mask_index.sum(dim=1, keepdim=True) % steps_per_block > 0: # Remainder handling only if not adaptive
            remainder = block_mask_index.sum(dim=1, keepdim=True) % steps_per_block
            for i in range(remainder.shape[0]):
                num_transfer_tokens[i, :remainder[i]] += 1

        cache.clear() # Clear cache at the start of each block

        for i in range(steps_per_block):
            x = token_buffer.data if token_buffer._is_on_gpu else token_buffer.to_gpu().data # Simplified GPU access

            if progress_callback:
                total_steps_completed += 1
                progress_percentage = total_steps_completed / total_steps_expected
                progress_callback(progress_percentage, x.clone())

            mask_index = (x == mask_id)
            if not mask_index.any():
                break

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = chunk_processing(model, x_, chunk_size=chunk_size)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = chunk_processing(model, x, chunk_size=chunk_size)

            if memory_integration and memory_integration.is_initialized():
                token_probs = F.softmax(logits, dim=-1)
                token_sequence = x[0, :].cpu().numpy().tolist()

                for pos in range(x.size(1)):
                    if mask_index[0, pos]:
                        vocab_size = token_probs.size(-1)
                        pos_probs = token_probs[0, pos].cpu().numpy()
                        guided_probs = memory_integration.apply_memory_guidance(
                            pos_probs, token_sequence, vocab_size
                        )
                        token_probs[0, pos] = torch.tensor(
                            guided_probs,
                            device=token_probs.device,
                            dtype=token_probs.dtype
                        )
                epsilon = 1e-10
                logits = torch.log(token_probs + epsilon)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1) if temperature > 0 else F.softmax(logits.to(torch.float64), dim=-1) # Conditional softmax dtype
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf # Confidence masking outside block

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                tokens_to_transfer = num_transfer_tokens[j, i].item()
                if tokens_to_transfer > 0:
                    _, select_index = torch.topk(confidence[j],
                                                 k=min(tokens_to_transfer, torch.sum(mask_index[j]).item()))
                    transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]
            token_buffer.update(x)

            if cpu_offload and i < steps_per_block - 1:
                gpu_to_cpu_offload(token_buffer)

        if cpu_offload:
            gpu_to_cpu_offload(token_buffer)

    token_buffer.to_gpu()
    return token_buffer.data


def gpu_to_cpu_offload(token_buffer: TokenBuffer): # Type hint for TokenBuffer
    token_buffer.to_cpu()
    torch.cuda.empty_cache()
