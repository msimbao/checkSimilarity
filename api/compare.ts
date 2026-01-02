// api/compare.ts
import type { VercelRequest, VercelResponse } from '@vercel/node';
import { pipeline, cos_sim } from '@xenova/transformers';

// Cache the model between invocations
let extractor: any = null;

async function getExtractor() {
  if (!extractor) {
    extractor = await pipeline(
      'feature-extraction',
      'Xenova/all-MiniLM-L6-v2',
      { quantized: true }
    );
  }
  return extractor;
}

// Jaro-Winkler similarity
function jaroWinkler(s1: string, s2: string): number {
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
function normalize(text: string): string {
  return text.trim().replace(/\s+/g, ' ');
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
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
      return res.status(200).json({
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
    
    return res.status(200).json({
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
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

export const config = {
  maxDuration: 60,
};
