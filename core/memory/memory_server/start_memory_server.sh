#!/bin/bash

# Start the MCP Titan Memory Server
echo "Starting MCP Titan Memory Server..."

# Change to script directory
cd "$(dirname "$0")"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js and try again."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the server
echo "Starting server on port 3000..."
node server.js

# This script should be made executable with:
# chmod +x start_memory_server.sh
