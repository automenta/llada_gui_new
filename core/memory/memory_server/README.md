# MCP Titan Memory Server

This directory contains the Memory Server for the LLaDA GUI application. The memory server provides long-term memory capabilities to the LLaDA model, allowing it to maintain context and coherence across generations.

## Overview

The memory system consists of:

1. A standalone server (either Python or Node.js based)
2. An API for interacting with the memory model
3. Integration with the LLaDA GUI

## Available Implementations

Two server implementations are provided:

1. **Node.js Server** (`server.js`): Uses TensorFlow.js for the memory model
2. **Python Server** (`server.py`): Uses PyTorch for the memory model

Both provide the same API and functionality, but the Python server may be easier to integrate if you're already using Python.

## Setup

### Setup with Node.js (Recommended)

1. Ensure Node.js is installed on your system
2. Run the setup script:
   ```
   ./setup.sh
   ```
3. Start the server:
   ```
   ./start_memory_server.sh
   ```

### Setup with Python

1. Install the Python dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Start the Python server:
   ```
   python server.py
   ```

## API Endpoints

The server provides the following API endpoints:

- `GET /status`: Get server status
- `POST /api/init_model`: Initialize the memory model
- `POST /api/forward_pass`: Run a forward pass through the model
- `POST /api/train_step`: Perform a training step
- `POST /api/save_model`: Save the model to a file
- `POST /api/load_model`: Load the model from a file
- `POST /api/reset_memory`: Reset the memory state
- `GET /api/memory_state`: Get the current memory state

## Integration with LLaDA GUI

The memory system is integrated with the LLaDA GUI through the `memory_integration.py` module. This provides a high-level interface for using the memory system with the LLaDA model.

## Customization

You can customize the memory model by modifying the following files:

- `config.py`: Configuration parameters for the server
- `server.js` or `server.py`: Server implementation
- `memory_integration.py`: Integration with the LLaDA GUI

## Models Directory

The `models/` directory is used to store saved memory models. You can save and load memory models through the API.
