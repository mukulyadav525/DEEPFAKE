"""
services/model_runtime.py

Optional learned-model adapters for image, audio, video-frame, and text
classification. These adapters are designed to be best-effort:

- If local transformers/torch checkpoints are configured, use them.
- Else if Hugging Face Inference configuration is present, call that endpoint.
- Else return None and let heuristic analyzers drive the result.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx


HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
DEFAULT_TIMEOUT_S = 30.0


def _normalize_env(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _truthy(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _config() -> dict[str, Any]:
    return {
        "hf_token": _normalize_env(os.getenv("HF_TOKEN")),
        "hf_image_model_id": _normalize_env(
            os.getenv("HF_IMAGE_MODEL_ID", "prithivMLmods/deepfake-detector-model-v1")
        ),
        "hf_audio_model_id": _normalize_env(os.getenv("HF_AUDIO_MODEL_ID")),
        "hf_text_model_id": _normalize_env(os.getenv("HF_TEXT_MODEL_ID")),
        "local_image_model_id": _normalize_env(os.getenv("LOCAL_IMAGE_MODEL_ID")),
        "local_audio_model_id": _normalize_env(os.getenv("LOCAL_AUDIO_MODEL_ID")),
        "local_text_model_id": _normalize_env(os.getenv("LOCAL_TEXT_MODEL_ID")),
        "disable_remote_models": _truthy("DISABLE_REMOTE_MODELS"),
    }


def _score_from_label(label: str, score: float) -> float:
    label_lower = label.lower()
    positive_terms = (
        "fake",
        "deepfake",
        "spoof",
        "synthetic",
        "ai-generated",
        "ai generated",
        "generated",
        "manipulated",
        "altered",
    )
    negative_terms = (
        "real",
        "authentic",
        "genuine",
        "human",
        "bonafide",
        "bona fide",
        "original",
    )

    if any(term in label_lower for term in positive_terms):
        return _clamp(score)
    if any(term in label_lower for term in negative_terms):
        return _clamp(1.0 - score)
    return _clamp(score)


def _top_prediction(predictions: list[dict]) -> dict | None:
    if not predictions:
        return None
    return max(predictions, key=lambda item: float(item.get("score", 0.0)))


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return None


def _router_url(model_id: str) -> str:
    return f"{HF_ROUTER_BASE}/{model_id}"


@lru_cache(maxsize=4)
def _load_local_text_stack(model_id: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model


@lru_cache(maxsize=4)
def _load_local_image_stack(model_id: str):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    return processor, model


@lru_cache(maxsize=4)
def _load_local_audio_stack(model_id: str):
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    return extractor, model


def _local_text_prediction(text: str) -> dict | None:
    config = _config()
    model_id = config["local_text_model_id"]
    if not model_id:
        return None

    try:
        import torch

        tokenizer, model = _load_local_text_stack(model_id)
        inputs = tokenizer(text[:4000], return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        id2label = getattr(model.config, "id2label", {}) or {}
        predictions = []
        for idx, prob in enumerate(probs.tolist()):
            label = str(id2label.get(idx, f"class_{idx}"))
            predictions.append({"label": label, "score": float(prob)})

        top = _top_prediction(predictions)
        if not top:
            return None
        return {
            "score": round(_score_from_label(top["label"], top["score"]), 3),
            "source": f"local_text_model:{model_id}",
            "label": top["label"],
            "raw": predictions,
        }
    except Exception:
        return None


def _local_image_prediction(image_bytes: bytes) -> dict | None:
    config = _config()
    model_id = config["local_image_model_id"]
    if not model_id:
        return None

    try:
        import torch
        from PIL import Image

        processor, model = _load_local_image_stack(model_id)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        id2label = getattr(model.config, "id2label", {}) or {}
        predictions = []
        for idx, prob in enumerate(probs.tolist()):
            label = str(id2label.get(idx, f"class_{idx}"))
            predictions.append({"label": label, "score": float(prob)})

        top = _top_prediction(predictions)
        if not top:
            return None
        return {
            "score": round(_score_from_label(top["label"], top["score"]), 3),
            "source": f"local_image_model:{model_id}",
            "label": top["label"],
            "raw": predictions,
        }
    except Exception:
        return None


def _local_audio_prediction(path: Path) -> dict | None:
    config = _config()
    model_id = config["local_audio_model_id"]
    if not model_id:
        return None

    try:
        import librosa
        import torch

        extractor, model = _load_local_audio_stack(model_id)
        target_sr = getattr(extractor, "sampling_rate", 16000)
        waveform, sr = librosa.load(str(path), sr=target_sr, mono=True)
        inputs = extractor(waveform, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        id2label = getattr(model.config, "id2label", {}) or {}
        predictions = []
        for idx, prob in enumerate(probs.tolist()):
            label = str(id2label.get(idx, f"class_{idx}"))
            predictions.append({"label": label, "score": float(prob)})

        top = _top_prediction(predictions)
        if not top:
            return None
        return {
            "score": round(_score_from_label(top["label"], top["score"]), 3),
            "source": f"local_audio_model:{model_id}",
            "label": top["label"],
            "raw": predictions,
        }
    except Exception:
        return None


def _remote_model_request(
    model_id: str,
    *,
    content: bytes | None = None,
    json_payload: dict | None = None,
    content_type: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[dict] | None:
    config = _config()
    token = config["hf_token"]
    if not token or config["disable_remote_models"] or not model_id:
        return None

    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type

    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                _router_url(model_id),
                headers=headers,
                content=content,
                json=json_payload,
            )
        if response.status_code >= 400:
            return None
        payload = _safe_json(response)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if "error" in payload:
                return None
            if "label" in payload and "score" in payload:
                return [payload]
        return None
    except Exception:
        return None


def infer_text_model(text: str) -> dict | None:
    if not text.strip():
        return None

    local = _local_text_prediction(text)
    if local:
        return local

    config = _config()
    model_id = config["hf_text_model_id"]
    if not model_id:
        return None

    predictions = _remote_model_request(
        model_id,
        json_payload={"inputs": text[:4000], "parameters": {"top_k": 5}},
        content_type="application/json",
    )
    top = _top_prediction(predictions or [])
    if not top:
        return None
    return {
        "score": round(_score_from_label(str(top["label"]), float(top["score"])), 3),
        "source": f"hf_text_model:{model_id}",
        "label": str(top["label"]),
        "raw": predictions,
    }


def infer_image_model(image_bytes: bytes) -> dict | None:
    local = _local_image_prediction(image_bytes)
    if local:
        return local

    config = _config()
    model_id = config["hf_image_model_id"]
    if not model_id:
        return None

    predictions = _remote_model_request(
        model_id,
        content=image_bytes,
        content_type="application/octet-stream",
    )
    top = _top_prediction(predictions or [])
    if not top:
        return None
    return {
        "score": round(_score_from_label(str(top["label"]), float(top["score"])), 3),
        "source": f"hf_image_model:{model_id}",
        "label": str(top["label"]),
        "raw": predictions,
    }


def infer_image_model_from_path(path: Path) -> dict | None:
    try:
        return infer_image_model(path.read_bytes())
    except Exception:
        return None


def infer_audio_model(path: Path) -> dict | None:
    local = _local_audio_prediction(path)
    if local:
        return local

    config = _config()
    model_id = config["hf_audio_model_id"]
    if not model_id:
        return None

    try:
        predictions = _remote_model_request(
            model_id,
            content=path.read_bytes(),
            content_type="application/octet-stream",
            timeout_s=60.0,
        )
    except Exception:
        return None

    top = _top_prediction(predictions or [])
    if not top:
        return None
    return {
        "score": round(_score_from_label(str(top["label"]), float(top["score"])), 3),
        "source": f"hf_audio_model:{model_id}",
        "label": str(top["label"]),
        "raw": predictions,
    }


def get_external_model_status() -> dict[str, Any]:
    config = _config()
    return {
        "remote_enabled": bool(config["hf_token"]) and not config["disable_remote_models"],
        "local_image_model_id": config["local_image_model_id"],
        "local_audio_model_id": config["local_audio_model_id"],
        "local_text_model_id": config["local_text_model_id"],
        "hf_image_model_id": config["hf_image_model_id"],
        "hf_audio_model_id": config["hf_audio_model_id"],
        "hf_text_model_id": config["hf_text_model_id"],
    }
