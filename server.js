// server.js
import express from 'express';
import cors from 'cors';
import { pipeline, cos_sim } from '@xenova/transformers';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Cache the model
let extractor = null;

async function getExtractor() {
  if (!extractor) {
    console.log('Loading model...');
    extractor = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2',
      { quantized: true }
    );
    console.log('Model loaded!');
  }
  return extractor;
}

// Jaro-Winkler similarity
function jaroWinkler(s1, s2) {
  const m = s1.length;
  const n = s2.length;
  
  if (m === 0 && n === 0) return 1.0;
  if (m === 0 || n === 0) return 0.0;
  
  const matchWindow = Math.floor(Math.max(m, n) / 2) - 1;
  const s1Matches = new Array(m).fill(false);
  const s2Matches = new Array(n).fill(false);
  
  let matches = 0;
  let transpositions = 0;
  
  for (let i = 0; i < m; i++) {
    const start = Math.max(0, i - matchWindow);
    const end = Math.min(i + matchWindow + 1, n);
    
    for (let j = start; j < end; j++) {
      if (s2Matches[j] || s1[i] !== s2[j]) continue;
      s1Matches[i] = true;
      s2Matches[j] = true;
      matches++;
      break;
    }
  }
  
  if (matches === 0) return 0.0;
  
  let k = 0;
  for (let i = 0; i < m; i++) {
    if (!s1Matches[i]) continue;
    while (!s2Matches[k]) k++;
    if (s1[i] !== s2[k]) transpositions++;
    k++;
  }
  
  const jaro = (matches / m + matches / n + (matches - transpositions / 2) / matches) / 3;
  
  let prefix = 0;
  for (let i = 0; i < Math.min(m, n, 4); i++) {
    if (s1[i] === s2[i]) prefix++;
    else break;
  }
  
  return jaro + prefix * 0.1 * (1 - jaro);
}

// Normalize text
function normalize(text) {
  return text.trim().replace(/\s+/g, ' ');
}

// Health check endpoint
app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    message: 'Answer Comparison API',
    modelLoaded: extractor !== null
  });
});

// Compare endpoint
app.post('/api/compare', async (req, res) => {
  try {
    const { userAnswer, correctAnswer, threshold = 0.75 } = req.body;
    
    if (!userAnswer || !correctAnswer) {
      return res.status(400).json({ 
        error: 'Both userAnswer and correctAnswer are required' 
      });
    }
    
    const normalizedUser = normalize(userAnswer);
    const normalizedCorrect = normalize(correctAnswer);
    
    // Exact match check
    if (normalizedUser.toLowerCase() === normalizedCorrect.toLowerCase()) {
      return res.json({
        isCorrect: true,
        confidence: 1.0,
        scores: {
          exact: 1.0,
          semantic: 1.0,
          jaroWinkler: 1.0,
          combined: 1.0
        }
      });
    }
    
    // Get the model
    const model = await getExtractor();
    
    // Get embeddings
    const output = await model([normalizedUser, normalizedCorrect], {
      pooling: 'mean',
      normalize: true
    });
    
    // Calculate cosine similarity
    const similarity = cos_sim(output[0].data, output[1].data);
    const semanticScore = Number(similarity);
    
    // Calculate Jaro-Winkler similarity
    const jaroScore = jaroWinkler(
      normalizedUser.toLowerCase(), 
      normalizedCorrect.toLowerCase()
    );
    
    // Weighted combination: 80% semantic, 20% Jaro-Winkler
    const combinedScore = semanticScore * 0.8 + jaroScore * 0.2;
    
    const isCorrect = combinedScore >= threshold;
    
    res.json({
      isCorrect,
      confidence: combinedScore,
      scores: {
        semantic: semanticScore,
        jaroWinkler: jaroScore,
        combined: combinedScore
      },
      threshold
    });
    
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message
    });
  }
});

// Start server
app.listen(PORT, async () => {
  console.log(`Server running on port ${PORT}`);
  // Pre-load model on startup
  await getExtractor();
  console.log('Ready to accept requests!');
});
