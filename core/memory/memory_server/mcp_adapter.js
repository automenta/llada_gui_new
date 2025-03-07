/**
 * MCP Adapter for Titan Memory Server
 * 
 * This module provides an adapter layer between the MCP protocol and
 * the Titan Memory Server's API.
 */

const express = require('express');
const bodyParser = require('body-parser');
const { TitanMemoryModel } = require('./server');

// Create Express router
const mcpRouter = express.Router();
mcpRouter.use(bodyParser.json());

// Global model instance (shared with main server)
let model = null;

/**
 * Initialize the model for MCP use
 */
function initializeMCPModel() {
    if (!model) {
        console.log('Initializing MCP Titan Memory model');
        model = new TitanMemoryModel();
    }
    return model;
}

/**
 * MCP run endpoint
 * Handles MCP protocol requests and maps them to memory operations
 */
mcpRouter.post('/api/mcp/run', async (req, res) => {
    try {
        const { operation, params } = req.body;
        
        // Initialize model if not already initialized
        if (!model) {
            model = initializeMCPModel();
        }
        
        let result;
        
        switch (operation) {
            case 'initialize':
                result = {
                    success: true,
                    config: model.config
                };
                break;
                
            case 'forward':
                if (!params || !params.input) {
                    return res.status(400).json({ error: 'Missing input parameter' });
                }
                
                result = await model.forward(params.input, params.memory);
                break;
                
            case 'train':
                if (!params || !params.current || !params.next) {
                    return res.status(400).json({ error: 'Missing required parameters' });
                }
                
                const trainResult = await model.trainStep(
                    params.current, 
                    params.next, 
                    params.memory
                );
                
                result = {
                    ...trainResult,
                    memoryState: model.getMemoryState()
                };
                break;
                
            case 'reset':
                model.resetMemory();
                result = {
                    success: true,
                    memoryState: model.getMemoryState()
                };
                break;
                
            case 'save':
                if (!params || !params.path) {
                    return res.status(400).json({ error: 'Missing path parameter' });
                }
                
                await model.saveModel(params.path);
                result = { success: true };
                break;
                
            case 'load':
                if (!params || !params.path) {
                    return res.status(400).json({ error: 'Missing path parameter' });
                }
                
                await model.loadModel(params.path);
                result = { success: true };
                break;
                
            case 'getState':
                result = {
                    memoryState: model.getMemoryState()
                };
                break;
                
            default:
                return res.status(400).json({ error: `Unknown operation: ${operation}` });
        }
        
        res.json({
            status: 'success',
            result
        });
    } catch (error) {
        console.error('MCP operation error:', error);
        res.status(500).json({
            status: 'error',
            error: error.message
        });
    }
});

/**
 * MCP status endpoint
 */
mcpRouter.get('/api/mcp/status', (req, res) => {
    res.json({
        status: 'active',
        model: model ? 'initialized' : 'not initialized',
        capabilities: [
            'memory_initialize',
            'memory_forward',
            'memory_train',
            'memory_reset',
            'memory_save',
            'memory_load',
            'memory_getState'
        ]
    });
});

/**
 * Add MCP routes to an existing Express app
 * @param {Express.Application} app - Express app
 */
function addMCPRoutes(app) {
    app.use(mcpRouter);
    console.log('MCP routes added to server');
}

module.exports = { addMCPRoutes, initializeMCPModel };
