#!/bin/bash

# Setup script for the MCP Titan Memory Server
echo "Setting up MCP Titan Memory Server..."

# Change to script directory
cd "$(dirname "$0")"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js and try again."
    exit 1
fi

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

# Setup Python environment
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create models directory if it doesn't exist
mkdir -p models

# Make start script executable
chmod +x start_memory_server.sh

echo "Setup complete. You can now start the memory server using ./start_memory_server.sh"
