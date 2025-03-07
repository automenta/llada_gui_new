# Memory System Quick Start Guide

This guide helps you get started with the LLaDA GUI memory system.

## What is the Memory System?

The memory system enhances text generation by providing coherence and context awareness across interactions. It uses a neural memory model to maintain a representation of the conversation and guide the diffusion process.

## Setup

1. Run the setup script to install required dependencies:

```bash
cd core/memory/memory_server
./setup.sh
```

## Using Memory in the LLaDA GUI

1. **Enable Memory Integration**:
   - In the LLaDA GUI, check the "Enable Memory Integration" option in the Hardware/Memory Options section
   - When enabled, the system will automatically start the memory server if needed

2. **Memory Visualization Tab**:
   - Click on the "Memory Visualization" tab to see the current memory state
   - The status indicator shows whether the memory system is connected (green) or disconnected (red)

3. **Memory Influence**:
   - Use the "Memory Influence" slider to control how strongly the memory affects generation
   - Higher values (e.g., 50-70%) result in more consistent outputs related to previous generations
   - Lower values (e.g., 10-30%) allow more creative freedom while maintaining some coherence

4. **Memory Controls**:
   - **Connect/Disconnect**: Toggle the memory system connection
   - **Reset Memory**: Clear the memory state to start fresh

## Tips for Effective Use

1. **Start with a clear context**:
   - Reset memory when starting a new conversation or topic
   - This prevents previous interactions from influencing the new generation

2. **Use memory for related prompts**:
   - Memory is most effective when generating related content
   - Example: A series of messages in a conversation, or chapters in a story

3. **Adjust influence based on needs**:
   - Higher influence (50-70%): For maintaining character traits, conversation style, or factual consistency
   - Medium influence (30-50%): For thematic consistency while allowing new ideas
   - Lower influence (10-30%): For subtle guidance with maximum creative freedom

4. **Experiment with memory reset**:
   - If you want a major change in direction, reset the memory
   - If you want to evolve within the same context, keep the memory active

## Examples

**Example 1: Consistent Storytelling**
1. Enable memory integration
2. Set memory influence to 40-50%
3. Generate the first part of a story
4. For subsequent parts, the memory will help maintain character names, setting details, and plot elements

**Example 2: Q&A Session**
1. Enable memory integration
2. Set memory influence to 30-40%
3. Ask related questions on a topic
4. The memory will help maintain consistent terminology and keep track of what's been covered

## Troubleshooting

If you encounter issues with the memory system, check the TROUBLESHOOTING.md file for common problems and solutions.
