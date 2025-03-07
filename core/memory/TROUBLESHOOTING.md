# Memory System Troubleshooting Guide

This guide helps you troubleshoot common issues with the LLaDA GUI memory system.

## Common Errors

### "Error initializing memory integration"

This is typically caused by one of the following issues:

1. **Memory server is not running**:
   - Run `python memory_server/server_manager.py start` to start the server manually
   - Check if the server is running with `python memory_server/server_manager.py status`

2. **Missing dependencies**:
   - Run the setup script: `./memory_server/setup.sh`
   - This will install required packages like Flask, torch, numpy, and requests

3. **Port conflict**:
   - The default port is 3000. If something else is using this port:
   - Edit `memory_server/config.py` to change DEFAULT_PORT
   - Or use the --port option when starting the server

4. **Python import issues**:
   - Make sure you're running from the project root directory
   - Try running with the full path: `python /path/to/llada_gui_new/core/memory/memory.py`

### "Memory server returned status X" or "Could not connect to memory server"

This indicates an issue with the memory server:

1. **Check server logs**:
   - Start the server in debug mode: `python memory_server/server.py --debug`
   - Look for error messages in the output

2. **Check server process**:
   - Use `ps aux | grep "server.py"` to see if the process is running
   - Kill any stalled processes with `kill <pid>`

3. **Network issues**:
   - Ensure localhost (127.0.0.1) is accessible
   - Try changing the host to 0.0.0.0 for broader access

### "Memory influence not working"

If memory guidance doesn't seem to affect generation:

1. **Check memory state**:
   - In the Memory Visualization tab, ensure the status shows "Connected"
   - Check that the memory influence slider is set to a reasonable value (e.g., 30%)

2. **Reset memory**:
   - Click "Reset Memory" to clear the state and try again
   - This can help if the memory state has become corrupted

## Debugging Steps

### 1. Run the test script

```bash
python memory_server/test_server.py
```

This performs a comprehensive test of the memory server functionality.

### 2. Check memory server status

```bash
python memory_server/server_manager.py status
```

### 3. Restart the memory server

```bash
python memory_server/server_manager.py stop
python memory_server/server_manager.py start
```

### 4. Check logs

If the server isn't starting properly, run it in the foreground with debug mode:

```bash
python memory_server/server.py --debug
```

### 5. Verify API endpoints

Use curl to test the API endpoints directly:

```bash
curl http://localhost:3000/status
curl -X POST -H "Content-Type: application/json" -d '{"inputDim": 32, "outputDim": 32}' http://localhost:3000/init
```

## Advanced Troubleshooting

### Server Configuration

You can modify the server configuration in `memory_server/config.py`:

- `DEFAULT_HOST` - Host to bind to (default: 127.0.0.1)
- `DEFAULT_PORT` - Port to use (default: 3000)
- `DEFAULT_MODEL_CONFIG` - Model parameters
- `AUTO_START_SERVER` - Whether to auto-start the server
- `AUTO_INIT_MODEL` - Whether to auto-initialize the model

### Database/Memory Path

By default, memory models are saved to `memory_server/models/`. If you're having permissions issues:

1. Check the directory exists: `mkdir -p memory_server/models`
2. Check permissions: `chmod -R 755 memory_server/models`
