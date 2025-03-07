"""
Memory optimization modules for the LLaDA model.

This package contains optimizations for running LLaDA efficiently on different hardware:
- Standard optimizations for GPUs with 16GB+ VRAM
- Extreme optimizations for GPUs with limited VRAM (8-12GB)

The optimizations are applied through the optimizer modules and are automatically
selected based on your hardware and settings in the GUI.
"""
