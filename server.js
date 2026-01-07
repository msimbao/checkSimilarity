const express = require('express');
const cors = require('cors');
require('@tensorflow/tfjs-node');  // This MUST come first!
const use = require('@tensorflow-models/universal-sentence-encoder');


const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Global variable to store the loaded model
let model = null;
let modelLoadingPromise = null;

// Function to load the model
async function loadModel() {
  if (model) {
    return model;
  }
  
  if (modelLoadingPromise) {
    return modelLoadingPromise;
  }
  
  console.log('Loading Universal Sentence Encoder model...');
  modelLoadingPromise = use.load();
  
  try {
    model = await modelLoadingPromise;
    console.log('Model loaded successfully!');
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    modelLoadingPromise = null;
    throw error;
  }
}

// Calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

// Health check endpoint
app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Universal Sentence Encoder API is running',
    modelLoaded: model !== null
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    modelLoaded: model !== null
  });
});

// Similarity endpoint
app.post('/similarity', async (req, res) => {
  try {
    const { text1, text2 } = req.body;
    
    // Validate input
    if (!text1 || !text2) {
      return res.status(400).json({
        error: 'Both text1 and text2 are required'
      });
    }
    
    if (typeof text1 !== 'string' || typeof text2 !== 'string') {
      return res.status(400).json({
        error: 'text1 and text2 must be strings'
      });
    }
    
    // Ensure model is loaded
    const loadedModel = await loadModel();
    
    // Generate embeddings
    const embeddings = await loadedModel.embed([text1, text2]);
    const embeddingsArray = await embeddings.array();
    
    // Calculate similarity
    const similarity = cosineSimilarity(embeddingsArray[0], embeddingsArray[1]);
    
    // Clean up tensors to prevent memory leaks
    embeddings.dispose();
    
    res.json({
      text1,
      text2,
      similarity: similarity,
      similarityPercentage: `${(similarity * 100).toFixed(2)}%`
    });
    
  } catch (error) {
    console.error('Error calculating similarity:', error);
    res.status(500).json({
      error: 'Failed to calculate similarity',
      message: error.message
    });
  }
});

// Batch similarity endpoint - compare one text against multiple texts
app.post('/similarity/batch', async (req, res) => {
  try {
    const { text, texts } = req.body;
    
    // Validate input
    if (!text || !texts) {
      return res.status(400).json({
        error: 'Both text and texts array are required'
      });
    }
    
    if (!Array.isArray(texts)) {
      return res.status(400).json({
        error: 'texts must be an array'
      });
    }
    
    // Ensure model is loaded
    const loadedModel = await loadModel();
    
    // Generate embeddings for all texts
    const allTexts = [text, ...texts];
    const embeddings = await loadedModel.embed(allTexts);
    const embeddingsArray = await embeddings.array();
    
    // Calculate similarities
    const baseEmbedding = embeddingsArray[0];
    const similarities = texts.map((comparisonText, index) => {
      const similarity = cosineSimilarity(baseEmbedding, embeddingsArray[index + 1]);
      return {
        text: comparisonText,
        similarity: similarity,
        similarityPercentage: `${(similarity * 100).toFixed(2)}%`
      };
    });
    
    // Clean up tensors
    embeddings.dispose();
    
    res.json({
      baseText: text,
      results: similarities
    });
    
  } catch (error) {
    console.error('Error calculating batch similarity:', error);
    res.status(500).json({
      error: 'Failed to calculate batch similarity',
      message: error.message
    });
  }
});

// Start the server and load the model
async function startServer() {
  try {
    // Pre-load the model before accepting requests
    await loadModel();
    
    app.listen(PORT, () => {
      console.log(`Server is running on port ${PORT}`);
      console.log(`Health check: http://localhost:${PORT}/health`);
      console.log(`Similarity endpoint: POST http://localhost:${PORT}/similarity`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();