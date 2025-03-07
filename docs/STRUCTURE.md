# LLaDA GUI Project Structure

This document outlines the organization of the LLaDA GUI project, which has been restructured for better maintainability and clarity.

## Directory Structure

```
llada_gui_new/
├── core/               # Core functionality
│   ├── config.py       # Configuration constants
│   ├── generate.py     # Generation functions
│   ├── llada_worker.py # Worker thread for generation
│   ├── memory/         # Memory management components
│   └── utils.py        # Utility functions
│
├── gui/                # GUI components
│   ├── llada_gui.py    # Main GUI application
│   ├── memory_monitor.py # Memory monitoring component
│   └── visualizations/ # Visualization components
│       └── diffusion_visualization.py # Diffusion process visualizer
│
├── optimizations/      # Memory optimizations
│   ├── standard/       # Standard optimizations (16GB+ VRAM)
│   └── extreme/        # Extreme optimizations (8-12GB VRAM)
│
├── scripts/            # Utility scripts
│   ├── install.sh      # Installation script
│   ├── start_gui.sh    # Script to start the GUI
│   └── start_memory.sh # Script to start with memory integration
│
├── resources/          # Resources (images, icons, desktop files)
├── docs/               # Documentation
├── requirements.txt    # Python dependencies
└── run.py              # Main entry point
```

## Key Components

1. **Core Components**
   - `config.py`: Central configuration for the application
   - `llada_worker.py`: Thread that handles model loading and generation
   - `generate.py`: Functions for text generation
   - `memory/`: Memory integration for context-aware generation

2. **GUI Components**
   - `llada_gui.py`: Main GUI window and application
   - `memory_monitor.py`: Real-time memory usage monitoring
   - `visualizations/`: Components for visualizing the diffusion process

3. **Memory Optimizations**
   - Standard optimizations for GPUs with 16GB+ VRAM
   - Extreme optimizations for GPUs with limited VRAM (8-12GB)

## Optimization System

The optimizations are organized into two categories:

1. **Standard Optimizations**
   - Designed for GPUs with 16GB+ VRAM
   - Provides memory efficiency without significant compromises
   - Applied via `--optimize` flag or standard optimizer in GUI

2. **Extreme Optimizations**
   - Designed for GPUs with limited VRAM (8-12GB)
   - Uses aggressive techniques to reduce memory usage
   - Applied via `--extreme` flag or extreme optimizer in GUI
   - May display warnings for parameters that could cause memory issues

Optimizations are applied holistically rather than as individual components, ensuring that all optimizations work together seamlessly and avoiding edge cases that could arise from applying only some optimizations.

## Usage

The main entry point is `run.py`, which handles:
- Environment setup
- Path configuration
- Dependency verification
- Optimization application
- Error handling
- Launching the main GUI

### Command-line Options

- `--optimize`: Apply standard memory optimizations
- `--extreme`: Apply extreme memory optimizations
- `--memory`: Enable memory integration
- `--debug`: Enable debug logging

## Error Handling

The application uses a warning-based approach rather than hard limits. When using extreme optimizations, the GUI will warn users about parameter combinations that might cause out-of-memory errors, but will allow users to proceed if they choose to do so.

If an out-of-memory error does occur, the application provides helpful suggestions for reducing memory usage rather than enforcing strict limits.
