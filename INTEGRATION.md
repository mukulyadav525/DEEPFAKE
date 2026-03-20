# Integration Guide — DRISHYAM-ULTIMATE-V3

This guide explains how to integrate the Drishyam forensic engine into your own Python projects.

## Option 1: Over the API (Recommended)

Running Drishyam as a standalone service (via Docker or Uvicorn) allows you to offload heavy forensic processing.

### Python Client Example

```python
import requests
import time

API_URL = "http://localhost:8000"
API_KEY = "your_api_key_here"

def analyze_media(file_path):
    headers = {"X-API-KEY": API_KEY}
    
    with open(file_path, "rb") as f:
        # 1. Start Analysis
        r = requests.post(f"{API_URL}/analyze", headers=headers, files={"file": f})
        job = r.json()
        job_id = job["job_id"]
        
    # 2. Poll for Results
    while True:
        status_r = requests.get(f"{API_URL}/status/{job_id}", headers=headers)
        res = status_r.json()
        
        if res["status"] == "done":
            return res["result"]
        elif res["status"] == "failed":
            raise Exception("Analysis failed")
            
        time.sleep(2)

# Usage
result = analyze_media("video.mp4")
print(f"Verdict: {result['result']} (Score: {result['score']})")
```

---

## Option 2: Modular Import (Internal Project)

If you want to use the forensic logic directly in your code without running a web server.

### 1. Add Drishyam to your Search Path
Ensure the `forensics-api` directory is in your `PYTHONPATH`.

### 2. Direct Analysis Example

```python
import asyncio
from pathlib import Path
from analyzers.video import analyze_video
from services.scoring import fuse_signals
from services.explanation import build_report

async def run_internal_forensics(file_path):
    path = Path(file_path)
    
    # 1. Run low-level forensic analyzer
    # (Automated file type detection can be done via services.classifier)
    raw_signals = await analyze_video(path)
    
    # 2. Fuse signals using V3 Ultimate Engine
    score, band, metrics = fuse_signals(raw_signals, file_type="video")
    
    # 3. Build human-readable report
    report = build_report("video", score, band, {**raw_signals, "metrics": metrics})
    
    return report

# Execution
report = asyncio.run(run_internal_forensics("evidence.mp4"))
print(report["technical_breakdown"])
```

## Option 3: Using with Netlify (Bridge)

If your project is on Netlify, you cannot host the Drishyam engine directly on Netlify (due to video/ML size limits). Instead, host Drishyam on **Railway/Render** and bridge them.

### Netlify Proxy Function (Node.js)
Create `netlify/functions/analyze.js` to hide your API Key:

```javascript
/* netlify/functions/analyze.js */
const axios = require('axios');

exports.handler = async (event) => {
    try {
        const DRISHYAM_URL = process.env.DRISHYAM_API_URL;
        const DRISHYAM_KEY = process.env.DRISHYAM_API_KEY;

        const response = await axios.post(`${DRISHYAM_URL}/analyze`, event.body, {
            headers: {
                'X-API-KEY': DRISHYAM_KEY,
                'Content-Type': event.headers['content-type']
            }
        });

        return { statusCode: 200, body: JSON.stringify(response.data) };
    } catch (error) {
        return { statusCode: 500, body: error.message };
    }
};
```

### In Your Netlify Frontend (React/Vanilla)
```javascript
const res = await fetch('/.netlify/functions/analyze', {
    method: 'POST',
    body: formData // from your file input
});
const { job_id } = await res.json();
```

See **[DEPLOYMENT.md](file:///Users/mukul/Desktop/DEEPFAKE/DEPLOYMENT.md)** for how to host the Drishyam Backend.

## Requirements for Integration
- **FFmpeg**: Must be installed on the host (Railway/Render).
- **Python 3.10+**: For the backend service.
- **Environment Variables**: `DRISHYAM_API_URL` and `DRISHYAM_API_KEY` set in the Netlify Dashboard.
