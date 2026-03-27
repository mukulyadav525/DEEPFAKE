"""
analyzers/image.py

Image forensics pipeline with metadata, pixel-level deepfake cues, and
digital-alteration detectors.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from pathlib import Path

from services.model_runtime import infer_image_model_from_path


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _parse_exif_datetime(value: object) -> datetime | None:
    if not value:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    text = str(value).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _extract_image_metadata(path: Path) -> dict:
    """Extract EXIF plus container-level image metadata."""
    try:
        import numpy as np
        from PIL import Image
        from PIL.ExifTags import TAGS

        image = Image.open(path)
        exif_raw = image.getexif() or {}
        exif = {TAGS.get(key, key): value for key, value in exif_raw.items()}

        info = dict(image.info or {})
        bands = image.getbands()
        has_alpha = "A" in bands
        partial_alpha_ratio = 0.0
        transparent_ratio = 0.0

        if has_alpha:
            alpha = np.asarray(image.getchannel("A"), dtype="uint8")
            partial_alpha_ratio = float(((alpha > 0) & (alpha < 255)).mean())
            transparent_ratio = float((alpha == 0).mean())

        software = str(
            exif.get("Software")
            or info.get("software")
            or info.get("Software")
            or info.get("Comment")
            or ""
        )

        metadata_richness = sum(
            1
            for value in [
                exif.get("Make"),
                exif.get("Model"),
                exif.get("DateTimeOriginal"),
                exif.get("DateTime"),
                exif.get("GPSInfo"),
            ]
            if value
        )

        return {
            "format": image.format or path.suffix.lstrip(".").upper(),
            "mode": image.mode,
            "width": image.width,
            "height": image.height,
            "bands": list(bands),
            "has_alpha": has_alpha,
            "partial_alpha_ratio": round(partial_alpha_ratio, 4),
            "transparent_ratio": round(transparent_ratio, 4),
            "icc_profile": bool(info.get("icc_profile")),
            "dpi": info.get("dpi"),
            "software": software,
            "info_keys": sorted(info.keys()),
            "exif_count": len(exif),
            "metadata_richness": metadata_richness,
            "exif": exif,
        }
    except Exception:
        return {
            "format": path.suffix.lstrip(".").upper(),
            "mode": "unknown",
            "width": 0,
            "height": 0,
            "bands": [],
            "has_alpha": False,
            "partial_alpha_ratio": 0.0,
            "transparent_ratio": 0.0,
            "icc_profile": False,
            "dpi": None,
            "software": "",
            "info_keys": [],
            "exif_count": 0,
            "metadata_richness": 0,
            "exif": {},
        }


def _peak_ratio_1d(signal) -> float:
    import numpy as np

    arr = np.asarray(signal, dtype="float32")
    if arr.size < 64:
        return 1.0

    arr = arr - float(arr.mean())
    if float(np.std(arr)) < 1e-6:
        return 1.0

    window = np.hanning(arr.size)
    spectrum = np.abs(np.fft.rfft(arr * window))
    upper = max(8, spectrum.size // 2)
    band = spectrum[4:upper]
    if band.size == 0:
        return 1.0

    median_energy = float(np.median(band)) + 1e-6
    peak_energy = float(np.max(band))
    return peak_energy / median_energy


def _block_map_regions(score_map, block_size: int, source: str, strategy: str = "high", max_regions: int = 4) -> list[dict]:
    import numpy as np

    arr = np.asarray(score_map, dtype="float32")
    if arr.size == 0:
        return []

    if strategy == "deviation":
        ranking = np.abs(arr - float(arr.mean())) / (float(arr.std()) + 1e-6)
        threshold = max(1.2, float(np.percentile(ranking, 80)))
    else:
        ranking = arr
        threshold = max(float(arr.mean() + arr.std()), float(np.percentile(arr, 85)))

    regions = []
    for y, x in sorted(
        ((row, col) for row in range(arr.shape[0]) for col in range(arr.shape[1])),
        key=lambda idx: float(ranking[idx]),
        reverse=True,
    ):
        value = float(ranking[y, x])
        if value < threshold:
            break

        x0 = int(x * block_size)
        y0 = int(y * block_size)
        region = {"x": x0, "y": y0, "w": block_size, "h": block_size, "source": source}

        overlaps = False
        for existing in regions:
            if abs(existing["x"] - region["x"]) < block_size and abs(existing["y"] - region["y"]) < block_size:
                overlaps = True
                break
        if overlaps:
            continue

        regions.append(region)
        if len(regions) >= max_regions:
            break

    return regions


def _estimate_jpeg_blockiness(gray) -> float:
    import numpy as np

    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.2

    boundary_samples = []
    interior_samples = []

    for idx in range(8, gray.shape[1], 8):
        boundary_samples.append(np.abs(gray[:, idx] - gray[:, idx - 1]))
        if idx + 1 < gray.shape[1]:
            interior_samples.append(np.abs(gray[:, idx + 1] - gray[:, idx]))
        if idx - 2 >= 0:
            interior_samples.append(np.abs(gray[:, idx - 1] - gray[:, idx - 2]))

    for idx in range(8, gray.shape[0], 8):
        boundary_samples.append(np.abs(gray[idx, :] - gray[idx - 1, :]))
        if idx + 1 < gray.shape[0]:
            interior_samples.append(np.abs(gray[idx + 1, :] - gray[idx, :]))
        if idx - 2 >= 0:
            interior_samples.append(np.abs(gray[idx - 1, :] - gray[idx - 2, :]))

    if not boundary_samples or not interior_samples:
        return 0.2

    boundary_mean = float(np.concatenate(boundary_samples).mean())
    interior_mean = float(np.concatenate(interior_samples).mean()) + 1e-6
    ratio = boundary_mean / interior_mean
    return round(_clamp((ratio - 1.10) / 0.55), 3)


def _estimate_noise_inconsistency(gray) -> tuple[float, dict, list[dict]]:
    import cv2
    import numpy as np

    if gray.shape[0] < 64 or gray.shape[1] < 64:
        return 0.3, {"residual_mean": 0.0, "block_cv": 0.0}, []

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, blurred)

    block_size = 32
    blocks = []
    for y in range(0, gray.shape[0] - block_size + 1, block_size):
        row = []
        for x in range(0, gray.shape[1] - block_size + 1, block_size):
            block = residual[y : y + block_size, x : x + block_size]
            row.append(float(block.std()))
        if row:
            blocks.append(row)

    if not blocks:
        return 0.3, {"residual_mean": 0.0, "block_cv": 0.0}, []

    block_map = np.asarray(blocks, dtype="float32")
    block_mean = float(block_map.mean()) + 1e-6
    block_cv = float(block_map.std() / block_mean)
    score = round(_clamp((block_cv - 0.28) / 0.45), 3)
    diagnostics = {
        "residual_mean": round(float(residual.mean()), 3),
        "block_cv": round(block_cv, 3),
    }
    return score, diagnostics, _block_map_regions(block_map, block_size, "noise_inconsistency", "deviation")


def _estimate_entropy_features(gray) -> tuple[dict, list[dict]]:
    import numpy as np

    if gray.shape[0] < 64 or gray.shape[1] < 64:
        return {"mean_entropy": 0.0, "entropy_cv": 0.0, "inconsistency_score": 0.2, "synthetic_flatness_score": 0.2}, []

    gray_u8 = gray.astype("uint8")
    block_size = 32
    blocks = []
    for y in range(0, gray_u8.shape[0] - block_size + 1, block_size):
        row = []
        for x in range(0, gray_u8.shape[1] - block_size + 1, block_size):
            block = gray_u8[y : y + block_size, x : x + block_size]
            quantized = (block // 8).ravel()
            hist = np.bincount(quantized, minlength=32).astype("float32")
            probs = hist / max(float(hist.sum()), 1.0)
            probs = probs[probs > 0]
            entropy = float(-(probs * np.log2(probs)).sum())
            row.append(entropy)
        if row:
            blocks.append(row)

    entropy_map = np.asarray(blocks, dtype="float32")
    mean_entropy = float(entropy_map.mean()) if entropy_map.size else 0.0
    entropy_cv = float(entropy_map.std() / (mean_entropy + 1e-6)) if entropy_map.size else 0.0

    inconsistency_score = _clamp((entropy_cv - 0.18) / 0.32)
    low_mean_score = _clamp((4.8 - mean_entropy) / 2.0)
    low_var_score = _clamp((0.10 - entropy_cv) / 0.10)
    synthetic_flatness_score = _clamp((low_mean_score * 0.60) + (low_var_score * 0.40))

    diagnostics = {
        "mean_entropy": round(mean_entropy, 3),
        "entropy_cv": round(entropy_cv, 3),
        "inconsistency_score": round(inconsistency_score, 3),
        "synthetic_flatness_score": round(synthetic_flatness_score, 3),
    }
    regions = _block_map_regions(entropy_map, block_size, "entropy_outlier", "deviation")
    return diagnostics, regions


def _estimate_resampling_score(gray) -> tuple[float, dict]:
    import numpy as np

    if gray.shape[0] < 64 or gray.shape[1] < 64:
        return 0.15, {"peak_ratio_x": 1.0, "peak_ratio_y": 1.0}

    second_x = np.abs(np.diff(gray, n=2, axis=1)).mean(axis=0)
    second_y = np.abs(np.diff(gray, n=2, axis=0)).mean(axis=1)

    peak_ratio_x = _peak_ratio_1d(second_x)
    peak_ratio_y = _peak_ratio_1d(second_y)
    peak_ratio = max(peak_ratio_x, peak_ratio_y)
    score = round(_clamp((peak_ratio - 3.5) / 6.5), 3)
    return score, {"peak_ratio_x": round(peak_ratio_x, 3), "peak_ratio_y": round(peak_ratio_y, 3)}


def _estimate_edge_halo_score(gray) -> tuple[float, dict]:
    import cv2
    import numpy as np

    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    highpass = np.abs(gray - blur)

    edges = cv2.Canny(gray.astype("uint8"), 80, 160)
    edge_mask = edges > 0
    if not edge_mask.any():
        return 0.0, {"edge_highpass_ratio": 1.0}

    dilated = cv2.dilate(edges, np.ones((3, 3), dtype="uint8"), iterations=1) > 0
    non_edge_mask = ~dilated
    edge_highpass = float(highpass[edge_mask].mean()) if edge_mask.any() else 0.0
    non_edge_highpass = float(highpass[non_edge_mask].mean()) if non_edge_mask.any() else 0.0
    ratio = edge_highpass / (non_edge_highpass + 1e-6)
    score = round(_clamp((ratio - 2.2) / 3.0), 3)
    return score, {"edge_highpass_ratio": round(ratio, 3)}


def _detect_duplicate_regions(gray) -> tuple[float, list[dict]]:
    import cv2

    orb = cv2.ORB_create(nfeatures=700)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 20:
        return 0.0, []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(descriptors, descriptors, k=4)

    suspicious_pairs = []
    for group in matches:
        candidates = [match for match in group if match.queryIdx != match.trainIdx]
        if not candidates:
            continue
        match = candidates[0]
        point_a = keypoints[match.queryIdx].pt
        point_b = keypoints[match.trainIdx].pt
        spatial_distance = ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5
        if match.distance < 32 and spatial_distance > 30:
            suspicious_pairs.append((point_a, point_b))

    score = round(_clamp((len(suspicious_pairs) - 6) / 18.0), 3)
    regions = []
    for point_a, point_b in suspicious_pairs[:4]:
        for point in (point_a, point_b):
            regions.append(
                {
                    "x": max(int(point[0]) - 18, 0),
                    "y": max(int(point[1]) - 18, 0),
                    "w": 36,
                    "h": 36,
                    "source": "duplicate_pattern",
                }
            )
    return score, regions


def _estimate_alpha_composite_score(metadata: dict) -> float:
    if not metadata.get("has_alpha"):
        return 0.0

    partial = float(metadata.get("partial_alpha_ratio", 0.0))
    transparent = float(metadata.get("transparent_ratio", 0.0))
    score = (partial * 9.0) + (transparent * 1.5)
    return round(_clamp(score), 3)


def _estimate_synthetic_patterns(
    gray,
    rgb,
    blockiness: float,
    noise_inconsistency: float,
    entropy_features: dict,
    resampling_score: float,
) -> tuple[float, dict]:
    import cv2
    import numpy as np

    gray_f = gray.astype("float32")
    smooth = cv2.GaussianBlur(gray_f, (0, 0), 1.6)
    residual_energy = float(np.mean(np.abs(gray_f - smooth)))
    laplacian_variance = float(cv2.Laplacian(gray_f, cv2.CV_32F).var())
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation_std = float(hsv[:, :, 1].std())

    too_clean_score = _clamp((6.5 - residual_energy) / 6.5)
    oversharp_score = _clamp((laplacian_variance - 1500.0) / 2500.0)
    flat_palette_score = _clamp((20.0 - saturation_std) / 20.0)

    score = (
        (too_clean_score * 0.27)
        + (oversharp_score * 0.23)
        + (flat_palette_score * 0.15)
        + (entropy_features["synthetic_flatness_score"] * 0.20)
        + (blockiness * 0.07)
        + (noise_inconsistency * 0.05)
        + (resampling_score * 0.03)
    )

    diagnostics = {
        "residual_energy": round(residual_energy, 3),
        "laplacian_variance": round(laplacian_variance, 3),
        "saturation_std": round(saturation_std, 3),
        "synthetic_flatness_score": entropy_features["synthetic_flatness_score"],
    }
    return round(_clamp(score), 3), diagnostics


def _extract_image_features(path: Path) -> dict:
    import numpy as np
    from PIL import Image
    import cv2

    source = Image.open(path)
    rgb = np.asarray(source.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype("float32")

    buffer = io.BytesIO()
    source.convert("RGB").save(buffer, "JPEG", quality=90)
    buffer.seek(0)
    resaved_rgb = np.asarray(Image.open(buffer).convert("RGB"))

    diff = np.mean(np.abs(rgb.astype("float32") - resaved_rgb.astype("float32")), axis=2)
    ela_score = round(_clamp((float(diff.mean()) - 8.0) / 35.0), 3)

    threshold = float(np.percentile(diff, 98))
    ela_mask = (diff >= threshold).astype("uint8") * 255
    ela_mask = cv2.morphologyEx(ela_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype="uint8"))
    contours, _ = cv2.findContours(ela_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(32, int(rgb.shape[0] * rgb.shape[1] * 0.002))
    suspicious_regions = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        suspicious_regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "source": "ela"})
        if len(suspicious_regions) >= 6:
            break

    blockiness = _estimate_jpeg_blockiness(gray)
    noise_inconsistency, noise_diag, noise_regions = _estimate_noise_inconsistency(gray)
    entropy_diag, entropy_regions = _estimate_entropy_features(gray)
    resampling_score, resampling_diag = _estimate_resampling_score(gray)
    halo_score, halo_diag = _estimate_edge_halo_score(gray)
    duplicate_score, duplicate_regions = _detect_duplicate_regions(gray.astype("uint8"))
    synthetic_score, synthetic_diag = _estimate_synthetic_patterns(
        gray,
        rgb,
        blockiness,
        noise_inconsistency,
        entropy_diag,
        resampling_score,
    )

    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "ela_score": ela_score,
        "blockiness": blockiness,
        "noise_inconsistency": noise_inconsistency,
        "duplicate_score": duplicate_score,
        "resampling_score": resampling_score,
        "halo_score": halo_score,
        "entropy_inconsistency": entropy_diag["inconsistency_score"],
        "synthetic_score": synthetic_score,
        "pixel_diagnostics": {
            **noise_diag,
            **entropy_diag,
            **resampling_diag,
            **halo_diag,
            **synthetic_diag,
        },
        "suspicious_regions": (suspicious_regions + noise_regions + entropy_regions + duplicate_regions)[:12],
    }


def _check_metadata_anomalies(metadata: dict, path: Path) -> tuple[float, list[str], dict]:
    findings = []
    suspicious = 0.0

    exif = metadata.get("exif", {})
    suffix = path.suffix.lower()
    software = str(metadata.get("software", "") or "")
    alpha_ratio = float(metadata.get("partial_alpha_ratio", 0.0))

    if not exif and suffix in {".jpg", ".jpeg", ".tiff"}:
        findings.append(
            "No EXIF metadata found in a camera-native format; this is suspicious but not conclusive."
        )
        suspicious += 1.0

    editing_tools = ["adobe", "photoshop", "gimp", "canva", "lightroom", "pixelmator", "snapseed"]
    ai_tools = ["midjourney", "stable diffusion", "dall-e", "comfyui", "automatic1111", "fooocus", "firefly"]

    software_lower = software.lower()
    if software and any(tool in software_lower for tool in editing_tools):
        findings.append(f"Editing software recorded in metadata: {software}")
        suspicious += 1.3

    ai_metadata = software and any(tool in software_lower for tool in ai_tools)
    if ai_metadata:
        findings.append(f"AI-generation tooling string found in metadata: {software}")
        suspicious += 1.6

    camera_make = exif.get("Make")
    camera_model = exif.get("Model")
    if suffix in {".jpg", ".jpeg", ".tiff"} and not camera_make and not camera_model:
        findings.append("No camera make/model present in EXIF.")
        suspicious += 0.6

    if software and not camera_make and not camera_model:
        findings.append("Software tag exists without camera identifiers.")
        suspicious += 0.5

    dt_original = _parse_exif_datetime(exif.get("DateTimeOriginal"))
    dt_digitized = _parse_exif_datetime(exif.get("DateTimeDigitized"))
    dt_modified = _parse_exif_datetime(exif.get("DateTime"))
    if dt_original and dt_digitized and dt_digitized < dt_original:
        findings.append("DateTimeDigitized precedes DateTimeOriginal.")
        suspicious += 0.8
    if dt_modified and dt_original and dt_modified < dt_original:
        findings.append("Last modified time precedes original capture time.")
        suspicious += 1.0

    if metadata.get("has_alpha") and alpha_ratio > 0.01:
        findings.append(
            f"Image contains semi-transparent pixels ({alpha_ratio:.1%}), which can be consistent with compositing."
        )
        suspicious += 0.5

    if not findings:
        findings.append("No strong metadata anomalies were detected.")

    diagnostics = {
        "format": metadata.get("format"),
        "mode": metadata.get("mode"),
        "bands": metadata.get("bands"),
        "has_alpha": metadata.get("has_alpha"),
        "partial_alpha_ratio": metadata.get("partial_alpha_ratio"),
        "icc_profile": metadata.get("icc_profile"),
        "dpi": metadata.get("dpi"),
        "exif_count": metadata.get("exif_count"),
        "metadata_richness": metadata.get("metadata_richness"),
        "software": software or None,
        "ai_metadata_hint": ai_metadata,
    }
    return round(_clamp(suspicious / 4.5), 3), findings, diagnostics


async def analyze_image(path: Path) -> dict:
    loop = asyncio.get_running_loop()

    metadata_future = loop.run_in_executor(None, _extract_image_metadata, path)
    features_future = loop.run_in_executor(None, _extract_image_features, path)
    metadata, features = await asyncio.gather(metadata_future, features_future)

    meta_score, meta_findings, metadata_diagnostics = await loop.run_in_executor(
        None,
        _check_metadata_anomalies,
        metadata,
        path,
    )
    learned_model_result = await loop.run_in_executor(None, infer_image_model_from_path, path)

    alpha_composite_score = _estimate_alpha_composite_score(metadata)

    forensic_score = round(
        (
            (features["ela_score"] * 0.22)
            + (features["blockiness"] * 0.15)
            + (features["noise_inconsistency"] * 0.18)
            + (features["entropy_inconsistency"] * 0.15)
            + (features["resampling_score"] * 0.15)
            + (features["halo_score"] * 0.15)
        ),
        3,
    )
    tamper_score = round(
        (
            (features["ela_score"] * 0.24)
            + (features["noise_inconsistency"] * 0.16)
            + (features["duplicate_score"] * 0.20)
            + (features["resampling_score"] * 0.15)
            + (features["halo_score"] * 0.12)
            + (features["entropy_inconsistency"] * 0.08)
            + (alpha_composite_score * 0.05)
        ),
        3,
    )
    heuristic_model_score = features["synthetic_score"]
    model_score = heuristic_model_score
    if learned_model_result:
        model_score = round(
            (heuristic_model_score * 0.45) + (learned_model_result["score"] * 0.55),
            3,
        )

    evidence = [
        f"Image size: {features['width']}x{features['height']} | Format: {metadata_diagnostics['format']} | Mode: {metadata_diagnostics['mode']}",
        *meta_findings,
    ]

    if features["ela_score"] >= 0.50:
        evidence.append(
            f"ELA hotspots are elevated ({features['ela_score']:.0%}), suggesting localized edits or recompression differences."
        )
    if features["noise_inconsistency"] >= 0.50:
        evidence.append(
            f"Pixel-noise consistency is weak ({features['noise_inconsistency']:.0%}); local regions do not share a stable sensor-noise profile."
        )
    if features["resampling_score"] >= 0.50:
        evidence.append(
            f"Resampling periodicity is elevated ({features['resampling_score']:.0%}), which can happen after resize/warp edits."
        )
    if features["halo_score"] >= 0.50:
        evidence.append(
            f"Edge halo score is elevated ({features['halo_score']:.0%}), consistent with compositing or heavy sharpening around boundaries."
        )
    if features["duplicate_score"] >= 0.45:
        evidence.append(
            f"Repeated local keypoint patterns suggest possible copy-move editing ({features['duplicate_score']:.0%})."
        )
    if alpha_composite_score >= 0.45:
        evidence.append(
            f"Transparency/composite cue score is elevated ({alpha_composite_score:.0%})."
        )
    if learned_model_result:
        evidence.append(
            "Learned deepfake detector returned "
            f"{learned_model_result['score']:.0%} suspicious probability "
            f"via {learned_model_result['source']} ({learned_model_result['label']})."
        )
    if model_score >= 0.55:
        pixel_diag = features["pixel_diagnostics"]
        evidence.append(
            "Pixel-level synthetic/deepfake pattern score is elevated "
            f"({model_score:.0%}); texture/noise profile looks unusually clean "
            f"(residual energy {pixel_diag['residual_energy']}, "
            f"laplacian variance {pixel_diag['laplacian_variance']}, "
            f"entropy flatness {pixel_diag['synthetic_flatness_score']})."
        )

    flags = []
    manipulation_hints = []

    if model_score >= 0.68:
        flags.append("Pixel-level deepfake/synthetic image cues are strong.")
        manipulation_hints.append("synthetic generation")
    if tamper_score >= 0.62:
        flags.append("Pixel-level tamper cues indicate splicing, cloning, resizing, or object insertion/removal.")
        manipulation_hints.append("localized digital editing")
    if features["duplicate_score"] >= 0.50:
        manipulation_hints.append("copy-move or clone editing")
    if features["resampling_score"] >= 0.50 or features["halo_score"] >= 0.50 or alpha_composite_score >= 0.45:
        manipulation_hints.append("resized or composited content")
    if meta_score >= 0.55:
        manipulation_hints.append("metadata inconsistency")
    if metadata_diagnostics.get("ai_metadata_hint"):
        flags.append("Metadata and pixel signals both point toward synthetic image generation.")
        manipulation_hints.append("synthetic generation")

    metadata_richness = float(metadata.get("metadata_richness", 0.0))
    metadata_quality = _clamp(0.35 + (metadata_richness * 0.13) + (0.12 if metadata.get("exif_count", 0) else 0.0))

    signal_quality = {
        "metadata": round(metadata_quality, 3),
        "forensic": 0.86,
        "model": 0.90 if learned_model_result else 0.78,
        "tamper": 0.88 if features["duplicate_score"] >= 0.45 or features["resampling_score"] >= 0.50 else 0.80,
    }

    return {
        "metadata": meta_score,
        "forensic": forensic_score,
        "model": model_score,
        "tamper": tamper_score,
        "cross_modal": None,
        "evidence": evidence,
        "flags": flags,
        "signal_quality": signal_quality,
        "manipulation_hints": list(dict.fromkeys(manipulation_hints)),
        "suspicious_regions": features["suspicious_regions"],
        "metadata_diagnostics": metadata_diagnostics,
        "image_diagnostics": {
            **features["pixel_diagnostics"],
            "alpha_composite_score": alpha_composite_score,
            "heuristic_model_score": heuristic_model_score,
            "learned_model_score": learned_model_result["score"] if learned_model_result else None,
        },
        "external_model_evidence": learned_model_result,
    }
