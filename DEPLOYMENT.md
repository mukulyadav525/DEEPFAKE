# Online Deployment — Drishyam Forensic Engine

Since your project is on Netlify, you need to host this **Drishyam Backend** on a platform that supports persistent Python environments and large file processing (like Video).

## 🚀 Recommended Hosting (Low Cost / Free Tiers)

1.  **Railway.app** (Easiest / Auto-detects Python)
2.  **Render.com** (Supports Web Services + FFmpeg)
3.  **Hugging Face Spaces** (Perfect for AI/ML specialized backends)
4.  **AWS App Runner / Google Cloud Run** (Scalable with Docker)

---

## 📦 Deployment via Docker (Universal)

The easiest way to ensure FFmpeg and all libraries work is using a `Dockerfile`.

### 1. Create a `Dockerfile`
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use a production server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 2. Push to GitHub
Deploy your `forensics-api` to a GitHub repository.

### 3. Connect to Railway/Render
- Link the repository.
- Add your Environment Variable: `DEEPFAKE_API_KEY`.
- The platform will provide a public URL (e.g., `https://drishyam-api.up.railway.app`).

---

## 🔗 Connecting Netlify to Drishyam

Once you have your **Public API URL**, you can call it from your Netlify project.

### Netlify Function (Node.js) Example
Create this in `netlify/functions/analyze-media.js`:

```javascript
const axios = require('axios');

exports.handler = async (event) => {
    const API_URL = process.env.DRISHYAM_API_URL;
    const API_KEY = process.env.DRISHYAM_API_KEY;

    // Proxy the request to the Drishyam Engine
    // (This keeps your API Key hidden from the browser)
    const response = await axios.post(`${API_URL}/analyze`, event.body, {
        headers: {
            'X-API-KEY': API_KEY,
            'Content-Type': event.headers['content-type']
        }
    });

    return {
        statusCode: 200,
        body: JSON.stringify(response.data)
    };
};
```

### Client-Side (React/Vue/Vanilla)
```javascript
const analyzeFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/.netlify/functions/analyze-media', {
    method: 'POST',
    body: formData
  });
  
  const data = await res.json();
  console.log("Analysis Job ID:", data.job_id);
};
```
