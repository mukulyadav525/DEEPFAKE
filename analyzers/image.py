"""
analyzers/image.py

Image forensics pipeline.

Checks:
  1. EXIF metadata extraction
  2. Error Level Analysis (ELA) — compression inconsistency
  3. AI-generation classifier (CNN/ViT binary head)
  4. Copy-move / splice detection (placeholder for ORB/SIFT approach)
  5. GAN/diffusion artifact heuristics

To swap in a real model, replace the stubs marked [MODEL] with your
PyTorch inference calls. The rest of the pipeline stays the same.
"""

import asyncio
from pathlib import Path
from typing import Any


# ── Helpers ─────────────────────────────────────────────────────────────────

def _extract_exif(path: Path) -> dict:
    """Extract EXIF metadata using Pillow."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(path)
        raw = img._getexif()
        if not raw:
            return {}
        return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return {}


def _run_ela(path: Path, quality: int = 90) -> float:
    """
    Error Level Analysis.
    Returns a 0–1 score where high = suspicious compression inconsistency.
    Real implementation resaves at known quality and compares pixel deltas.
    """
    try:
        import io
        import numpy as np
        from PIL import Image

        original = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        original.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")

        ela_arr = np.abs(np.array(original, dtype=float) - np.array(resaved, dtype=float))
        mean_ela = ela_arr.mean()
        # Normalize roughly: real photos ~10–25, spliced ~40+, diffusion gen ~60+
        score = min(mean_ela / 80.0, 1.0)
        return round(score, 3)
    except Exception:
        return 0.5  # neutral if fails


def _run_ai_classifier(path: Path) -> dict:
    """
    [MODEL] Multi-head classifier: detects specific synthetic architectures.
    Returns probabilities for different generation methods.
    
    In a real scenario, this would call a model like CNNDetect or UnivNet.
    """
    # STUB - Simulating a powerful ensemble result
    return {
        "gan_score": 0.42,
        "diffusion_score": 0.15,
        "vae_score": 0.08,
        "overall_synthetic": 0.55
    }


def _check_metadata_anomalies(exif: dict) -> tuple[float, list[str]]:
    """
    Returns (anomaly_score 0–1, list_of_findings).
    Higher score = more anomalous metadata.
    """
    findings = []
    suspicious = 0

    if not exif:
        findings.append("No EXIF metadata found — camera-generated images always have this.")
        suspicious += 2

    for editing_tool in ["Adobe", "Photoshop", "GIMP", "Canva", "Lightroom"]:
        software = str(exif.get("Software", ""))
        if editing_tool.lower() in software.lower():
            findings.append(f"Editing software detected in metadata: {software}")
            suspicious += 1

    if "GPSInfo" not in exif and "Make" not in exif:
        findings.append("No camera make/model or GPS data.")
        suspicious += 1

    score = min(suspicious / 4.0, 1.0)
    return round(score, 3), findings


# ── Main analyzer ────────────────────────────────────────────────────────────

async def analyze_image(path: Path) -> dict:
    """
    Full image forensics pipeline.
    Returns signals dict ready for the scoring engine.
    """
    loop = asyncio.get_event_loop()

    # Run blocking operations in thread pool
    exif = await loop.run_in_executor(None, _extract_exif, path)
    ela_score = await loop.run_in_executor(None, _run_ela, path)
    model_score = await loop.run_in_executor(None, _run_ai_classifier, path)
    meta_score, meta_findings = await loop.run_in_executor(None, _check_metadata_anomalies, exif)

    # Aggregate evidence strings
    evidence = list(meta_findings)

    if ela_score > 0.55:
        evidence.append(
            f"High ELA score ({ela_score:.2f}) — possible compression inconsistency "
            "suggesting splicing or editing."
        )
    elif ela_score < 0.2:
        evidence.append("Low ELA score — compression appears uniform (good sign).")

    # Unpack model metrics
    overall_prob = model_score.get("overall_synthetic", 0.5)
    gan_prob = model_score.get("gan_score", 0.0)
    diff_prob = model_score.get("diffusion_score", 0.0)

    if overall_prob > 0.65:
        evidence.append(
            f"AI-generation classifier confidence: {overall_prob:.0%} — "
            f"detected potential {('GAN' if gan_prob > diff_prob else 'Diffusion')} artifacts."
        )
    elif overall_prob < 0.35:
        evidence.append("AI-generation classifier found no strong synthetic artifacts.")

    flags = []
    if meta_score > 0.7 and overall_prob > 0.7:
        flags.append("Both metadata and AI-classifier strongly indicate non-authentic image.")
    if ela_score > 0.7:
        flags.append("Severe ELA anomaly — likely edited or spliced region present.")

    return {
        "metadata":   meta_score,
        "forensic":   ela_score,
        "model":      overall_prob,
        "tamper":     ela_score,         # reuse ELA as tamper proxy
        "cross_modal": None,             # N/A for still images
        "evidence":   evidence,
        "flags":      flags,
        # Future: attach heatmap base64 here
        "suspicious_regions": [],
    }
