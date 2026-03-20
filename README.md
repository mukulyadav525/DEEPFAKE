# DRISHYAM-ULTIMATE-V3 — Advanced Media Forensics API

A state-of-the-art multi-signal forensics pipeline for detecting AI-generated or manipulated images, audio, video, and PDFs.

## 🚀 Key Features (v3.0.0 Ultimate)

- **Cross-Modal Forensics**: Lip-sync (SyncNet) and Acoustic Environment matching.
- **Consensus-Override Scoring**: High-confidence 'smoking gun' detection.
- **Forensic Audit Trail**: Detailed lineage showing signal influence for every verdict.
- **Enterprise Security**: Mandatory `X-API-KEY` protection.
- **Model Health Dashboard**: Dedicated `/stats` endpoint for performance monitoring.

## 📦 Project Structure

```
forensics-api/
├── main.py                  ← FastAPI entry point (/analyze, /status, /stats)
├── requirements.txt
├── analyzers/
│   ├── image.py             ← GAN/Diffusion differentiation + ELA
│   ├── audio.py             ← Spectrogram + Splicing + Env Tone matching
│   ├── video.py             ← Frame consistency + Lip-Sync (SyncNet)
│   └── pdf.py               ← Metadata + Font + NLP (GPT Fingerprinting)
└── services/
    ├── scoring.py           ← V3.0.0 Ultimate Fusion Engine
    └── explanation.py       ← V3.0.0 Report Builder (w/ Audit Trail)
```

## 🛠 Setup

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API Key (copy from .env.example)
export DEEPFAKE_API_KEY="drishyam_admin_2026"

# Run the API
uvicorn main:app --reload --port 8000
```

## 🔌 API Usage

### 1. File Analysis
```bash
curl -X POST http://localhost:8000/analyze \
  -H "X-API-KEY: drishyam_admin_2026" \
  -F "file=@source_media.mp4"
```

### 2. Poll Status
```bash
curl -H "X-API-KEY: drishyam_admin_2026" \
  http://localhost:8000/status/{job_id}
```

### 3. Check Base Model Stats
```bash
curl -H "X-API-KEY: drishyam_admin_2026" \
  http://localhost:8000/stats
```

## 🧬 Scoring Engine (v3.0.0)

All signals are fused with **Consensus Override** logic:
- **Major Signals**: `cross_modal` (40%), `model` (35%), `tamper` (15%).
- **Override**: Any signal > 0.95 certainty overrides the average for absolute reinforcement.

## 🤝 Project Integration

To integrate this model directly into your code without the API, see [INTEGRATION.md](file:///Users/mukul/Desktop/DEEPFAKE/INTEGRATION.md).
