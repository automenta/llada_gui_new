#!/bin/bash

# Script to create a single desktop icon for LLaDA GUI with Memory

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create memory-enabled desktop icon
echo "Creating LLaDA GUI with Memory desktop icon..."
cat > ~/Desktop/LLaDA_GUI.desktop << EOF
[Desktop Entry]
Type=Application
Name=LLaDA GUI
Comment=LLaDA Text Generation with Memory Integration
Exec=bash -c "cd ${PROJECT_DIR} && ./run_memory.sh"
Icon=applications-science
Terminal=false
Categories=AI;MachineLearning;
Keywords=AI;Text;Generation;LLM;Memory;
StartupNotify=true
EOF

# Make it executable
chmod +x ~/Desktop/LLaDA_GUI.desktop

echo "Desktop icon created successfully!"
