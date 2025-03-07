#!/bin/bash

# Start the memory server first, then launch the GUI with memory integration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Ensure memory server directory exists
MEMORY_SERVER_DIR="$REPO_DIR/core/memory/memory_server"
if [ ! -d "$MEMORY_SERVER_DIR" ]; then
    echo "Error: Memory server directory not found at $MEMORY_SERVER_DIR"
    exit 1
fi

# Activate virtual environment
if [ -d "$REPO_DIR/venv" ]; then
    source "$REPO_DIR/venv/bin/activate"
fi

# Check if memory server is already running
if pgrep -f "node.*server.js" > /dev/null; then
    echo "Memory server is already running."
else
    # Get the MCP server configuration
    MCP_CONFIG="$REPO_DIR/mcp-config.json"
    if [ ! -f "$MCP_CONFIG" ]; then
        MCP_CONFIG="/home/ty/Repositories/ai_workspace/llada_gui/mcp-config.json"
    fi
    
    # Start memory server if configuration exists
    if [ -f "$MCP_CONFIG" ]; then
        echo "Starting memory server..."
        cd "$MEMORY_SERVER_DIR" || exit 1
        node server.js &
        SERVER_PID=$!
        echo "Memory server started with PID: $SERVER_PID"
        
        # Wait for server to initialize
        sleep 2
    else
        echo "Warning: MCP configuration file not found. Memory server not started."
    fi
fi

# Start the GUI application with memory integration
echo "Starting LLaDA GUI with memory integration..."
cd "$REPO_DIR" || exit 1
python run.py --memory

# If we started a server, try to shut it down when the GUI closes
if [ -n "$SERVER_PID" ]; then
    echo "Stopping memory server (PID: $SERVER_PID)..."
    kill $SERVER_PID
fi

# Deactivate virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
