#!/bin/bash

# Installation script for LLaDA GUI
# This will set up the virtual environment and install dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR" || exit 1

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install the venv module."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing core dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Installing memory-related dependencies..."
pip install -r requirements_memory.txt

# Make scripts executable
echo "Setting up scripts..."
chmod +x "$REPO_DIR/scripts/"*.sh
chmod +x "$REPO_DIR/run.py"

# Install desktop shortcuts
echo "Installing desktop shortcuts..."
"$REPO_DIR/scripts/install_desktop.sh"

# Create necessary directories
echo "Creating data directories..."
mkdir -p "$REPO_DIR/data/memory"
mkdir -p "$REPO_DIR/data/models"

echo "Installation complete!"
echo "You can now run LLaDA GUI using:"
echo "  $REPO_DIR/run.py"
echo "Or through your applications menu."
