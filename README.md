# DRISHYAM Forensics API

Multi-modal forensic API for detecting AI-generated or digitally altered images, audio, video, PDF, DOCX, and TXT files.

## What It Does

- Image: metadata checks, ELA, clone/copy-move cues, resampling/compositing artifacts, pixel-level deepfake signals.
- Audio: synthetic-voice heuristics, splice detection, waveform and spectral consistency checks.
- Video: frame forensics, temporal consistency, lip-sync/audio-visual mismatch checks.
- Documents: metadata, hidden-content, structure, font drift, and AI-writing pattern checks.
- Hybrid runtime: heuristic pipeline always runs, with optional learned detectors via local Transformers or Hugging Face Inference.

## API Endpoints

- `POST /analyze-sync`
  Use this from your main project when you want the final verdict in one request.
- `POST /analyze`
  Queues analysis and returns a `job_id`.
- `GET /status/{job_id}`
  Poll queued jobs.
- `GET /health`
  Railway healthcheck endpoint.
- `GET /stats`
  Runtime capabilities and model/deployment status.

## Local Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export DEEPFAKE_API_KEY="change-me"
uvicorn main:app --reload --port 8000
```

## Railway Deployment

This repo includes:

- [Dockerfile](/Users/mukul/Desktop/DEEPFAKE/Dockerfile)
- [Procfile](/Users/mukul/Desktop/DEEPFAKE/Procfile)
- [railway.toml](/Users/mukul/Desktop/DEEPFAKE/railway.toml)

Railway will detect the root `Dockerfile` automatically, which matches Railway’s Dockerfile deployment flow in their docs: [Dockerfiles](https://docs.railway.com/deploy/dockerfiles). Railway also supports config-as-code via `railway.toml`: [Config as Code](https://docs.railway.com/config-as-code).

Set these variables in Railway:

```bash
DEEPFAKE_API_KEY=replace-with-a-secret
MAX_SIZE_MB=100
SYNC_MAX_SIZE_MB=30
JOB_RETENTION_MINUTES=60
CORS_ALLOW_ORIGINS=https://your-frontend.com
```

Optional learned-model variables:

```bash
HF_TOKEN=hf_xxx
HF_IMAGE_MODEL_ID=prithivMLmods/deepfake-detector-model-v1
HF_AUDIO_MODEL_ID=
HF_TEXT_MODEL_ID=

LOCAL_IMAGE_MODEL_ID=
LOCAL_AUDIO_MODEL_ID=
LOCAL_TEXT_MODEL_ID=
```

If no learned-model variables are configured, the service falls back to the built-in forensic ensemble.

## Call It From Your Project

Recommended: call `POST /analyze-sync` for images, PDFs, audio, and smaller uploads.

```bash
curl -X POST https://your-railway-domain.up.railway.app/analyze-sync \
  -H "X-API-KEY: replace-with-a-secret" \
  -F "file=@sample.png"
```

For larger or slower files like video, use the async workflow:

```bash
curl -X POST https://your-railway-domain.up.railway.app/analyze \
  -H "X-API-KEY: replace-with-a-secret" \
  -F "file=@sample.mp4"
```

Then poll:

```bash
curl -H "X-API-KEY: replace-with-a-secret" \
  https://your-railway-domain.up.railway.app/status/<job_id>
```

## Notes

- Job state is currently in memory, which is fine for a single Railway service instance.
- For multi-instance scaling or long-lived history, move job state to Redis/Postgres.
- Local uploaded files are cleaned up after analysis to keep the Railway instance lean.
