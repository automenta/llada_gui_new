#!/bin/bash

# Launcher for LLaDA GUI with memory integration
echo "Starting LLaDA GUI..."

# Parse command line arguments
MEMORY_MODE="standard" # Options: standard, improved
while [[ $# -gt 0 ]]; do
    case $1 in
        --improved-memory)
            MEMORY_MODE="improved"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --improved-memory    Use improved memory server management"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate Python virtual environment if it exists
if [ -d "${SCRIPT_DIR}/venv" ]; then
    echo "Activating virtual environment..."
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

# Fix memory server dependencies before launching
echo "Checking memory server dependencies..."
python "${SCRIPT_DIR}/fix_memory_dependencies.py" || {
    echo "Warning: Failed to fix memory server dependencies"
    # Continue anyway - it might still work with existing dependencies
}

# Run the application
if [ "$MEMORY_MODE" = "improved" ]; then
    echo "Launching application with improved memory integration..."
    python "${SCRIPT_DIR}/run_with_memory_improved.py"
else
    echo "Launching application with standard memory integration..."
    python "${SCRIPT_DIR}/run.py" --memory
fi

# Deactivate virtual environment if activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
