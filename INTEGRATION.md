# Integration Guide

## Best Option: Call the Railway API

For most projects, call the deployed API instead of importing the analyzers directly.

### Synchronous Request

Use this for images, audio, PDFs, and most regular uploads:

```python
import requests

API_URL = "https://your-service.up.railway.app"
API_KEY = "replace-with-a-secret"

def analyze_file_sync(file_path: str) -> dict:
    with open(file_path, "rb") as fh:
        response = requests.post(
            f"{API_URL}/analyze-sync",
            headers={"X-API-KEY": API_KEY},
            files={"file": fh},
            timeout=180,
        )
    response.raise_for_status()
    return response.json()
```

### Async Request

Use this for larger video files:

```python
import requests
import time

API_URL = "https://your-service.up.railway.app"
API_KEY = "replace-with-a-secret"

def analyze_file_async(file_path: str) -> dict:
    with open(file_path, "rb") as fh:
        job = requests.post(
            f"{API_URL}/analyze",
            headers={"X-API-KEY": API_KEY},
            files={"file": fh},
            timeout=120,
        ).json()

    job_id = job["job_id"]

    while True:
        status = requests.get(
            f"{API_URL}/status/{job_id}",
            headers={"X-API-KEY": API_KEY},
            timeout=60,
        ).json()

        if status["status"] == "done":
            return status
        if status["status"] == "failed":
            raise RuntimeError(status["result"])
        time.sleep(2)
```

## Internal Python Import

If you really want to run the pipeline inside another Python app:

```python
import asyncio
from pathlib import Path

from analyzers.image import analyze_image
from services.scoring import fuse_signals
from services.explanation import build_report

async def run_internal(path: str):
    file_path = Path(path)
    signals = await analyze_image(file_path)
    score, band, metrics = fuse_signals(signals, "image")
    return build_report("image", score, band, {**signals, "metrics": metrics})

report = asyncio.run(run_internal("sample.png"))
```

The deployed API is still the cleaner choice for your main project because it keeps forensic dependencies and scaling concerns in one place.
