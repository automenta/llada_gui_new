#!/bin/bash

# LLaDA GUI Installation Script - Simplified for easy use
echo "Installing LLaDA GUI..."

# Get the project root directory
PROJECT_ROOT=$(pwd)

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install all requirements
    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements_memory.txt
else
    # Activate existing virtual environment
    source venv/bin/activate
    echo "Using existing virtual environment"
fi

# Install additional dependencies for memory integration
pip install flask flask-cors numpy torch
# Create necessary directories
mkdir -p data/memory/vector_db
mkdir -p core/memory/memory_server/models

# Make scripts executable
chmod +x run.sh
chmod +x run_with_memory.sh
chmod +x fix_run_memory.sh
chmod +x core/memory/memory_server/server.py
chmod +x core/memory/memory_server/server_manager.py

# Define Python path
PYTHON="venv/bin/python"

# Fix memory integration
echo "Applying memory fixes..."
if [ -f "fix_memory_db.py" ]; then
    "$PYTHON" fix_memory_db.py
else
    echo "Warning: fix_memory_db.py not found, skipping"
fi

if [ -f "fix_titan_memory.py" ]; then
    "$PYTHON" fix_titan_memory.py
else
    echo "Warning: fix_titan_memory.py not found, skipping"
fi

if [ -f "direct_memory_fix.py" ]; then
    "$PYTHON" direct_memory_fix.py --prepare-only
else
    echo "Warning: direct_memory_fix.py not found, skipping"
fi
# Create desktop icons (preferred method)
echo "Creating desktop icons..."
if [ -f "create_desktop_icons.sh" ]; then
    chmod +x create_desktop_icons.sh
    ./create_desktop_icons.sh
    echo "Desktop icons created successfully"
else
    echo "Warning: create_desktop_icons.sh not found, skipping"
fi

echo "Installation complete! You can now run LLaDA GUI from your desktop icons."
