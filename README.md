# Universal Sentence Encoder Similarity API

A backend server that uses TensorFlow.js Universal Sentence Encoder to calculate semantic similarity between text strings.

## Features

- Calculate similarity between two text strings
- Batch similarity comparison (one text vs multiple texts)
- Pre-loads model at startup for fast responses
- CORS enabled for frontend integration
- Health check endpoints

## Setup

### Local Development

1. Install dependencies:
```bash
npm install
```

2. Start the server:
```bash
npm start
```

The server will start on port 3000 (or the PORT environment variable if set).

### Deploy to Render

1. Push this code to a GitHub repository
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Render will automatically detect the Node.js project
5. Set the following:
   - **Build Command**: `npm install`
   - **Start Command**: `npm start`
   - **Environment**: Node
6. Deploy!

**Important**: The first deployment will take a bit longer as it downloads and caches the model. Subsequent requests will be fast.

## API Endpoints

### Health Check
```
GET /
GET /health
```

Returns server status and whether the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "modelLoaded": true
}
```

### Calculate Similarity (Two Texts)
```
POST /similarity
```

Calculate semantic similarity between two text strings.

**Request Body:**
```json
{
  "text1": "I love programming",
  "text2": "I enjoy coding"
}
```

**Response:**
```json
{
  "text1": "I love programming",
  "text2": "I enjoy coding",
  "similarity": 0.8234,
  "similarityPercentage": "82.34%"
}
```

### Batch Similarity
```
POST /similarity/batch
```

Compare one text against multiple texts.

**Request Body:**
```json
{
  "text": "I love dogs",
  "texts": [
    "I adore puppies",
    "Cats are great",
    "The weather is nice"
  ]
}
```

**Response:**
```json
{
  "baseText": "I love dogs",
  "results": [
    {
      "text": "I adore puppies",
      "similarity": 0.7821,
      "similarityPercentage": "78.21%"
    },
    {
      "text": "Cats are great",
      "similarity": 0.5432,
      "similarityPercentage": "54.32%"
    },
    {
      "text": "The weather is nice",
      "similarity": 0.2341,
      "similarityPercentage": "23.41%"
    }
  ]
}
```

## Example Usage

### Using fetch (JavaScript)
```javascript
const response = await fetch('https://your-render-url.onrender.com/similarity', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text1: 'Hello world',
    text2: 'Hi there'
  })
});

const data = await response.json();
console.log(data.similarity); // e.g., 0.7234
```

### Using curl
```bash
curl -X POST https://your-render-url.onrender.com/similarity \
  -H "Content-Type: application/json" \
  -d '{"text1": "Hello world", "text2": "Hi there"}'
```

## Understanding Similarity Scores

- **1.0**: Identical or extremely similar meaning
- **0.7-0.9**: Very similar meaning
- **0.5-0.7**: Moderately similar
- **0.3-0.5**: Some similarity
- **0.0-0.3**: Little to no similarity

## Performance Notes

- First request after deployment: ~10-30 seconds (model loading)
- Subsequent requests: Fast (~100-500ms depending on text length)
- Model stays loaded in memory while server is running
- On Render free tier, server may spin down after 15 minutes of inactivity

## Memory Requirements

Recommended minimum: 2GB RAM (Render Starter plan or higher)

## License

MIT