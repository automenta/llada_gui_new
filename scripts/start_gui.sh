#!/bin/bash

# Start LLaDA GUI with standard optimizations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment if it exists
if [ -d "$REPO_DIR/venv" ]; then
    source "$REPO_DIR/venv/bin/activate"
fi

# Run with standard optimizations
python "$REPO_DIR/run.py" --optimize

# Deactivate virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
