#!/usr/bin/env node

/**
 * MCP Titan Memory Server for LLaDA GUI
 * 
 * This server provides an HTTP API for the Titan memory system,
 * allowing the LLaDA GUI to incorporate memory-guided generation.
 * 
 * The server uses a lightweight neural memory model to maintain state
 * across interactions and provide continuity in generation.
 */

const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

// Configuration defaults
const DEFAULT_CONFIG = {
    inputDim: 64,
    outputDim: 64,
    hiddenDim: 32,
    learningRate: 0.001,
    forgetGateInit: 0.01
};

/**
 * Titan Memory Model implementation using TensorFlow.js
 */
class TitanMemoryModel {
    /**
     * Create a new memory model
     * @param {Object} config - Configuration parameters
     */
    constructor(config = {}) {
        // Apply default values for any missing config
        this.config = {
            ...DEFAULT_CONFIG,
            ...config
        };

        // Extract dimensions from config
        const { inputDim, outputDim, hiddenDim, forgetGateInit } = this.config;
        
        // Create model layers
        this.fc1 = tf.layers.dense({
            units: hiddenDim,
            inputShape: [inputDim + outputDim],
            activation: 'relu',
            name: 'fc1'
        });
        
        this.fc2 = tf.layers.dense({
            units: inputDim + outputDim,
            name: 'fc2'
        });
        
        // Create forget gate parameter
        this.forgetGate = tf.variable(tf.scalar(forgetGateInit));
        
        // Initialize optimizer
        this.optimizer = tf.train.adam(this.config.learningRate);
        
        // Initialize memory state
        this.resetMemory();
    }
    
    /**
     * Reset the memory state to zeros
     */
    resetMemory() {
        this.memoryState = tf.zeros([this.config.outputDim]);
    }
    
    /**
     * Get the current memory state
     * @returns {Array} Memory state as array
     */
    getMemoryState() {
        return this.memoryState.arraySync();
    }
    
    /**
     * Set the memory state manually
     * @param {Array|Tensor} state - New memory state
     */
    setMemoryState(state) {
        if (Array.isArray(state)) {
            this.memoryState = tf.tensor(state);
        } else if (state instanceof tf.Tensor) {
            this.memoryState = state.clone();
        } else {
            throw new Error(`Unsupported memory state type: ${typeof state}`);
        }
    }
    
    /**
     * Forward pass through the network
     * @param {Array|Tensor} x - Input tensor
     * @param {Array|Tensor} memory - Optional memory state (defaults to current)
     * @returns {Object} Object with predicted, newMemory, and surprise values
     */
    async forward(x, memory = null) {
        return tf.tidy(() => {
            // Convert input to tensor if needed
            const xTensor = Array.isArray(x) ? tf.tensor(x) : x;
            
            // Use provided memory or current state
            const memTensor = memory ? 
                (Array.isArray(memory) ? tf.tensor(memory) : memory) : 
                this.memoryState;
            
            // Apply forget gate to memory state
            const forgetValue = tf.sigmoid(this.forgetGate);
            const gatedMemory = tf.mul(memTensor, tf.sub(tf.scalar(1), forgetValue));
            
            // Combine input and gated memory
            const combined = tf.concat([xTensor, gatedMemory]);
            
            // MLP forward pass
            const hidden = tf.relu(this.fc1.apply(combined.expandDims(0)).squeeze());
            const output = this.fc2.apply(hidden.expandDims(0)).squeeze();
            
            // Split output into new memory and predicted next input
            const newMemory = output.slice([0], [this.config.outputDim]);
            const predicted = output.slice([this.config.outputDim], [this.config.inputDim]);
            
            // Calculate surprise (MSE between predicted and actual input)
            const diff = tf.sub(predicted, xTensor);
            const surprise = tf.mean(tf.mul(diff, diff));
            
            return {
                predicted: predicted.arraySync(),
                newMemory: newMemory.arraySync(),
                surprise: surprise.arraySync()
            };
        });
    }
    
    /**
     * Perform a training step
     * @param {Array|Tensor} xT - Current input tensor
     * @param {Array|Tensor} xNext - Next input tensor (target)
     * @param {Array|Tensor} memory - Current memory state
     * @returns {Object} Object with cost value
     */
    async trainStep(xT, xNext, memory) {
        // Convert inputs to tensors if needed
        const xTTensor = Array.isArray(xT) ? tf.tensor(xT) : xT;
        const xNextTensor = Array.isArray(xNext) ? tf.tensor(xNext) : xNext;
        const memTensor = memory ? 
            (Array.isArray(memory) ? tf.tensor(memory) : memory) : 
            this.memoryState;
        
        // Define training function
        const trainingFunction = () => {
            // Forward pass
            const combined = tf.concat([xTTensor, memTensor]);
            const hidden = tf.relu(this.fc1.apply(combined.expandDims(0)).squeeze());
            const output = this.fc2.apply(hidden.expandDims(0)).squeeze();
            
            // Split output
            const newMemory = output.slice([0], [this.config.outputDim]);
            const predicted = output.slice([this.config.outputDim], [this.config.inputDim]);
            
            // Calculate surprise
            const predDiff = tf.sub(predicted, xTTensor);
            const surprise = tf.mean(tf.mul(predDiff, predDiff));
            
            // Calculate loss (MSE between predicted and next + small surprise penalty)
            const diff = tf.sub(predicted, xNextTensor);
            const mseLoss = tf.mean(tf.mul(diff, diff));
            const totalLoss = tf.add(mseLoss, tf.mul(surprise, tf.scalar(0.01)));
            
            return { loss: totalLoss, newMemory };
        };
        
        // Run optimization
        const { value, grads } = this.optimizer.computeGradients(trainingFunction);
        
        // Apply gradients
        this.optimizer.applyGradients(grads);
        
        // Update memory state
        const result = await trainingFunction();
        this.memoryState = result.newMemory;
        
        // Return cost
        return {
            cost: await value.array()
        };
    }
    
    /**
     * Save model to file
     * @param {string} filePath - Path to save model
     */
    async saveModel(filePath) {
        try {
            // Create directory if it doesn't exist
            const dir = path.dirname(filePath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            
            // Get model state
            const saveData = {
                config: this.config,
                weights: {
                    fc1: {
                        kernel: this.fc1.getWeights()[0].arraySync(),
                        bias: this.fc1.getWeights()[1].arraySync()
                    },
                    fc2: {
                        kernel: this.fc2.getWeights()[0].arraySync(),
                        bias: this.fc2.getWeights()[1].arraySync()
                    },
                    forgetGate: this.forgetGate.arraySync()
                },
                memoryState: this.memoryState.arraySync()
            };
            
            // Save to file
            fs.writeFileSync(filePath, JSON.stringify(saveData, null, 2));
            return { success: true };
        } catch (error) {
            console.error('Error saving model:', error);
            throw error;
        }
    }
    
    /**
     * Load model from file
     * @param {string} filePath - Path to load model from
     */
    async loadModel(filePath) {
        try {
            // Read file
            const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
            
            // Update config
            this.config = { ...this.config, ...data.config };
            
            // Load weights
            if (data.weights) {
                // Load FC1 weights
                const fc1Kernel = tf.tensor(data.weights.fc1.kernel);
                const fc1Bias = tf.tensor(data.weights.fc1.bias);
                this.fc1.setWeights([fc1Kernel, fc1Bias]);
                
                // Load FC2 weights
                const fc2Kernel = tf.tensor(data.weights.fc2.kernel);
                const fc2Bias = tf.tensor(data.weights.fc2.bias);
                this.fc2.setWeights([fc2Kernel, fc2Bias]);
                
                // Load forget gate
                if (data.weights.forgetGate) {
                    this.forgetGate.assign(tf.scalar(data.weights.forgetGate));
                }
            }
            
            // Load memory state
            if (data.memoryState) {
                this.memoryState = tf.tensor(data.memoryState);
            } else {
                this.resetMemory();
            }
            
            return { success: true };
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }
}

// Create Express app
const app = express();
const port = process.env.PORT || 3000;

// Use JSON parser middleware
app.use(bodyParser.json());

// Import MCP adapter (if available)
let mcpAdapter = null;
try {
    mcpAdapter = require('./mcp_adapter');
    mcpAdapter.addMCPRoutes(app);
    console.log('MCP adapter loaded successfully');
} catch (error) {
    console.log('MCP adapter not available, running in standalone mode');
}

// Global model instance
let model = null;

// Define routes
app.get('/', (req, res) => {
    res.send('MCP Titan Memory Server');
});

// Status endpoint
app.get('/api/status', (req, res) => {
    if (!model) {
        return res.json({ status: 'No model initialized' });
    }
    
    res.json({ 
        status: 'Model initialized',
        config: model.config
    });
});

// Initialize model endpoint
app.post('/api/init_model', async (req, res) => {
    try {
        const { inputDim = 64, outputDim = 64 } = req.body || {};
        
        const config = {
            inputDim,
            outputDim,
            hiddenDim: req.body.hiddenDim || 32,
            learningRate: req.body.learningRate || 0.001
        };
        
        // Initialize model
        model = new TitanMemoryModel(config);
        
        res.json({
            message: 'Model initialized',
            config: model.config
        });
    } catch (error) {
        console.error('Model initialization error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Forward pass endpoint
app.post('/api/forward_pass', async (req, res) => {
    if (!model) {
        return res.status(400).json({ error: 'Model not initialized' });
    }
    
    try {
        const { x, memoryState } = req.body;
        
        if (!x) {
            return res.status(400).json({ error: 'Missing input vector' });
        }
        
        // Set memory state if provided
        if (memoryState) {
            model.setMemoryState(memoryState);
        }
        
        // Run forward pass
        const result = await model.forward(x);
        
        // Update memory state
        model.setMemoryState(result.newMemory);
        
        res.json({
            predicted: result.predicted,
            newMemory: result.newMemory,
            surprise: result.surprise
        });
    } catch (error) {
        console.error('Forward pass error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Train step endpoint
app.post('/api/train_step', async (req, res) => {
    if (!model) {
        return res.status(400).json({ error: 'Model not initialized' });
    }
    
    try {
        const { xT, xNext, memoryState } = req.body;
        
        if (!xT || !xNext) {
            return res.status(400).json({ error: 'Missing required parameters' });
        }
        
        // Set memory state if provided
        if (memoryState) {
            model.setMemoryState(memoryState);
        }
        
        // Run training step
        const result = await model.trainStep(xT, xNext, model.memoryState);
        
        // Get forward result for predictions
        const forwardResult = await model.forward(xT);
        
        res.json({
            cost: result.cost,
            predicted: forwardResult.predicted,
            surprise: forwardResult.surprise
        });
    } catch (error) {
        console.error('Train step error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Save model endpoint
app.post('/api/save_model', async (req, res) => {
    if (!model) {
        return res.status(400).json({ error: 'Model not initialized' });
    }
    
    try {
        const { path } = req.body;
        
        if (!path) {
            return res.status(400).json({ error: 'Missing path parameter' });
        }
        
        await model.saveModel(path);
        res.json({ message: 'Model saved successfully' });
    } catch (error) {
        console.error('Save model error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Load model endpoint
app.post('/api/load_model', async (req, res) => {
    try {
        const { path } = req.body;
        
        if (!path) {
            return res.status(400).json({ error: 'Missing path parameter' });
        }
        
        // Initialize model if not initialized
        if (!model) {
            model = new TitanMemoryModel();
        }
        
        await model.loadModel(path);
        res.json({ message: 'Model loaded successfully' });
    } catch (error) {
        console.error('Load model error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Reset memory endpoint
app.post('/api/reset_memory', (req, res) => {
    if (!model) {
        return res.status(400).json({ error: 'Model not initialized' });
    }
    
    try {
        model.resetMemory();
        res.json({ 
            message: 'Memory reset successfully',
            memoryState: model.getMemoryState()
        });
    } catch (error) {
        console.error('Reset memory error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Get memory state endpoint
app.get('/api/memory_state', (req, res) => {
    if (!model) {
        return res.status(400).json({ error: 'Model not initialized' });
    }
    
    try {
        const memoryState = model.getMemoryState();
        res.json({ memoryState });
    } catch (error) {
        console.error('Get memory state error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Start server
app.listen(port, () => {
    console.log(`MCP Titan Memory Server running on port ${port}`);
});

// Export model and app for testing
module.exports = { TitanMemoryModel, app };
