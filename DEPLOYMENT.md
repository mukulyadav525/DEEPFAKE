# Railway Deployment

This API is ready to deploy on Railway as a Dockerfile service.

Railway uses a root-level `Dockerfile` automatically for Dockerfile-based deploys: [Dockerfiles](https://docs.railway.com/deploy/dockerfiles). Railway also supports keeping deploy settings in `railway.toml`: [Config as Code](https://docs.railway.com/config-as-code).

## 1. Create the Service

1. Push this repo to GitHub.
2. In Railway, create a new project and link the repo.
3. Deploy the service from the repo root.

## 2. Set Railway Variables

Required:

```bash
DEEPFAKE_API_KEY=replace-with-a-secret
```

Recommended:

```bash
MAX_SIZE_MB=100
SYNC_MAX_SIZE_MB=30
JOB_RETENTION_MINUTES=60
CORS_ALLOW_ORIGINS=https://your-app.com
```

Optional learned-model runtime:

```bash
HF_TOKEN=hf_xxx
HF_IMAGE_MODEL_ID=prithivMLmods/deepfake-detector-model-v1
HF_AUDIO_MODEL_ID=
HF_TEXT_MODEL_ID=
```

Optional local transformer runtime:

```bash
LOCAL_IMAGE_MODEL_ID=
LOCAL_AUDIO_MODEL_ID=
LOCAL_TEXT_MODEL_ID=
```

## 3. Healthcheck

The included `railway.toml` points Railway at:

```text
/health
```

That endpoint returns a lightweight health payload and is safe to use for deployment checks.

## 4. Recommended API Pattern

Use `POST /analyze-sync` from your main project for:

- images
- audio
- PDFs
- DOCX/TXT
- smaller files

Use `POST /analyze` plus `GET /status/{job_id}` for:

- larger videos
- slower forensic runs
- any request where you do not want the client request to stay open

## 5. Example Client Call

```javascript
const formData = new FormData();
formData.append("file", file);

const res = await fetch("https://your-service.up.railway.app/analyze-sync", {
  method: "POST",
  headers: {
    "X-API-KEY": process.env.DRISHYAM_API_KEY,
  },
  body: formData,
});

const result = await res.json();
```

## 6. Scaling Note

Job state is currently stored in memory. That works well on a single Railway instance. If you later scale horizontally or want durable job history, move queued job state to Redis or Postgres.
