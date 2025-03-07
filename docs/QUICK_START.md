# LLaDA GUI - Quick Start Guide

This guide will help you get up and running with LLaDA GUI quickly.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/llada_gui.git
   cd llada_gui
   ```

2. **Run the installation script**:
   ```bash
   ./scripts/install.sh
   ```
   This will:
   - Create a Python virtual environment
   - Install dependencies
   - Set up desktop shortcuts
   - Create necessary directories

## Running LLaDA GUI

### Using the GUI

After installation, you can launch LLaDA GUI in several ways:

1. **Standard Mode**:
   ```bash
   ./run.py
   ```

2. **Optimized Mode** (recommended for most users):
   ```bash
   ./run.py --optimize
   ```

3. **Extreme Memory Mode** (for GPUs with 8-12GB VRAM):
   ```bash
   ./run.py --extreme
   ```

4. **Memory Integration Mode** (context-aware generation):
   ```bash
   ./run.py --memory
   ```

### Using Desktop Shortcuts

After installation, you can also launch LLaDA GUI from your applications menu:

- **LLaDA GUI** - Standard mode
- **LLaDA GUI (Optimized)** - With standard optimizations
- **LLaDA GUI (Extreme Optimization)** - For GPUs with limited VRAM
- **LLaDA GUI with Memory** - With memory integration

## Using the Interface

1. **Enter your prompt** in the text area at the top
2. **Adjust generation parameters** as needed:
   - Generation Length: Longer text requires more memory
   - Sampling Steps: More steps = better quality but slower
   - Block Length: Size of text chunks for processing
   - Temperature: Higher = more creative, lower = more deterministic
   - CFG Scale: Guidance scale for generation
   - Remasking Strategy: Approach for selecting tokens to remask

3. **Select hardware options**:
   - Device: CPU or GPU (CUDA)
   - Memory Optimization: Normal, 8-bit, or 4-bit quantization
   - Extreme Memory Mode: For GPUs with limited VRAM (8-12GB)

4. **Click "Generate"** to start the diffusion process

5. **View the output** in the tabs below:
   - Text Output: The generated text
   - Diffusion Visualization: Visual representation of the generation process

## Troubleshooting

If you encounter any issues:

- **Out of Memory Errors**: Try reducing generation length, using 4-bit quantization, or enabling Extreme Memory Mode
- **Installation Problems**: Make sure you have Python 3.8+ and pip installed
- **Model Loading Errors**: Check that the model path is correctly set in config.py

## Next Steps

For more information, see:
- [Structure Overview](STRUCTURE.md): Project organization details
- [Optimization Details](OPTIMIZATION_DETAILS.md): Memory optimization strategies
