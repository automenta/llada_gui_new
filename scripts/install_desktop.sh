#!/bin/bash

# Script to install desktop shortcuts for LLaDA GUI
# This will create desktop entries in the user's applications menu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DESKTOP_DIR="$REPO_DIR/resources/desktop"
TARGET_DIR="$HOME/.local/share/applications"

# Make sure the target directory exists
mkdir -p "$TARGET_DIR"

# Copy all desktop files
echo "Installing desktop shortcuts..."
for file in "$DESKTOP_DIR"/*.desktop; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        cp "$file" "$TARGET_DIR/$filename"
        echo "  Installed: $filename"
    fi
done

# Update desktop database
echo "Updating desktop database..."
update-desktop-database "$TARGET_DIR" 2>/dev/null || true

echo "Desktop shortcuts installed successfully!"
echo "You can now find LLaDA GUI in your applications menu."
