#!/bin/bash

# Unified launcher for LLaDA GUI with optional memory integration
# Usage: ./llada_gui.sh [--memory]

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Python virtual environment
if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python"
fi

# Launch the unified GUI launcher
"$PYTHON" llada_launcher.py "$@"
