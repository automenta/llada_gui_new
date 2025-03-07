# LLaDA GUI Memory System

This directory contains the memory system implementation for the LLaDA GUI project. It provides cognitive memory capabilities to enhance text generation with context awareness and coherence.

## Components

The memory system consists of several key components:

1. **memory_integration.py** - Integrates the memory system with the LLaDA GUI
2. **memory_embeddings.py** - Manages text embeddings for memory operations
3. **titan_memory.py** - Core implementation of the Titan Memory model
4. **memory_server/** - Flask server for memory operations with auto-start capability

## Architecture

The memory system uses a client-server architecture:

- The Python **memory server** exposes a REST API for memory operations
- The **memory_integration.py** module connects to this server from the GUI
- The GUI can visualize and control the memory state via the Memory tab

## Usage

The memory system is automatically integrated into the LLaDA GUI. When the application starts:

1. A server manager is initialized (but not started)
2. When the user enables "Use Memory Guidance" in the GUI:
   - The system tries to connect to the memory server
   - If the server isn't running, it automatically starts it
   - Memory is then used to guide the text generation process

## Manual Testing

You can test the memory system manually using:

```bash
# Test the memory server
cd memory_server
python test_server.py

# Start the server manually
python server_manager.py start
```

## Key Features

- **Auto-start**: The server is automatically started when needed
- **Memory visualization**: Real-time display of memory state in the GUI
- **Configurable influence**: Users can adjust how much memory affects generation
- **Persistence**: Memory models can be saved/loaded between sessions

## Integration with Main Application

The system is integrated with the main LLaDA GUI application through:

1. **MCPTitanMemoryInterface** class that handles communication with the memory server
2. **MemoryVisualizationWidget** for displaying and controlling memory state
3. **MemoryGuidanceDiffusionWorker** that incorporates memory in the diffusion process
4. **EnhancedGUI** decorator that adds memory capabilities to the base GUI

## Requirements

The memory system requires:
- Python 3.8+
- PyTorch
- Flask 
- NumPy
- Requests
