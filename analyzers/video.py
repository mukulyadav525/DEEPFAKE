"""
analyzers/video.py

Video forensics pipeline with frame, temporal, metadata, and audio-video checks.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from services.model_runtime import infer_image_model


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _extract_frames(path: Path, max_frames: int = 24) -> tuple[list[tuple[float, object]], float, float]:
    """Extract evenly spaced frames with timestamps."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        duration = (total / fps) if fps and total > 0 else 0.0

        if total <= 0:
            frames = []
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = (len(frames) / fps) if fps else 0.0
                frames.append((timestamp, frame))
            cap.release()
            return frames, duration, fps

        frame_indices = sorted(
            {
                min(total - 1, int((index / max(max_frames - 1, 1)) * (total - 1)))
                for index in range(max_frames)
            }
        )

        frames = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                timestamp = (frame_index / fps) if fps else 0.0
                frames.append((timestamp, frame))

        cap.release()
        return frames, duration, fps
    except Exception:
        return [], 0.0, 0.0


def _analyze_faces_and_motion(frames: list[tuple[float, object]]) -> tuple[float, list[str], list[dict], dict]:
    import cv2
    import numpy as np

    if not frames:
        return 0.3, ["No frames extracted from video."], [], {"mouth_track": [], "face_coverage": 0.0}

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        return 0.3, ["OpenCV face detector unavailable; frame forensics limited."], [], {
            "mouth_track": [],
            "face_coverage": 0.0,
        }

    suspicious_frames = []
    findings = []
    face_hits = 0
    suspicious_hits = 0
    mouth_track = []
    previous_mouth = None
    bbox_areas = []

    for timestamp, frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            previous_mouth = None
            continue

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_hits += 1
        bbox_areas.append(w * h)

        face_roi = gray[y : y + h, x : x + w]
        if face_roi.size == 0:
            continue

        border = np.concatenate(
            [
                face_roi[:6, :].ravel(),
                face_roi[-6:, :].ravel(),
                face_roi[:, :6].ravel(),
                face_roi[:, -6:].ravel(),
            ]
        )
        inner = face_roi[h // 4 : max(h // 4 + 1, (3 * h) // 4), w // 4 : max(w // 4 + 1, (3 * w) // 4)]
        edge_density = float(cv2.Canny(face_roi, 50, 150).mean() / 255.0)
        border_ratio = float(border.std() / (inner.std() + 1e-6)) if inner.size else 1.0

        if border_ratio > 1.35 and edge_density > 0.10:
            suspicious_hits += 1
            suspicious_frames.append(
                {
                    "time_s": round(timestamp, 2),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "source": "face_boundary_artifact",
                }
            )

        mouth = face_roi[int(h * 0.60) :, int(w * 0.15) : int(w * 0.85)]
        if mouth.size > 0:
            mouth_resized = cv2.resize(mouth, (96, 48))
            if previous_mouth is not None:
                motion = float(np.mean(np.abs(mouth_resized.astype("float32") - previous_mouth.astype("float32"))) / 255.0)
                mouth_track.append({"time_s": round(timestamp, 3), "motion": round(motion, 5)})
            previous_mouth = mouth_resized

    face_coverage = face_hits / max(len(frames), 1)
    suspicious_ratio = suspicious_hits / max(face_hits, 1)
    area_cv = 0.0
    if len(bbox_areas) > 1:
        area_cv = float(np.std(bbox_areas) / (np.mean(bbox_areas) + 1e-6))

    score = _clamp((suspicious_ratio * 0.75) + (_clamp((area_cv - 0.40) / 0.35) * 0.25))

    if face_hits:
        findings.append(
            f"Face detected in {face_hits}/{len(frames)} sampled frames."
        )
    else:
        findings.append("No stable face track found; face-swap analysis is limited.")

    if suspicious_hits:
        findings.append(
            f"Boundary artifacts detected in {suspicious_hits}/{max(face_hits, 1)} face frames."
        )

    return round(score, 3), findings, suspicious_frames[:10], {
        "mouth_track": mouth_track,
        "face_coverage": round(face_coverage, 3),
    }


def _check_temporal_consistency(frames: list[tuple[float, object]]) -> tuple[float, list[str]]:
    import cv2
    import numpy as np

    if len(frames) < 3:
        return 0.2, ["Not enough frames for temporal analysis."]

    transition_scores = []
    for idx in range(1, len(frames)):
        prev = cv2.resize(frames[idx - 1][1], (96, 96)).astype("float32")
        curr = cv2.resize(frames[idx][1], (96, 96)).astype("float32")

        prev_gray = cv2.cvtColor(prev.astype("uint8"), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr.astype("uint8"), cv2.COLOR_BGR2GRAY)

        pixel_diff = float(np.mean(np.abs(prev - curr)) / 255.0)
        prev_hist = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
        curr_hist = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])
        prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
        curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
        hist_diff = float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))

        transition_scores.append((pixel_diff * 0.60) + (hist_diff * 0.40))

    mean_diff = float(np.mean(transition_scores))
    std_diff = float(np.std(transition_scores))
    spike_threshold = mean_diff + (2.5 * std_diff)
    spikes = [score for score in transition_scores if score > spike_threshold and score < 0.70]

    findings = []
    if spikes:
        findings.append(
            f"Temporal jitter spikes detected at {len(spikes)} transition(s)."
        )
    else:
        findings.append("Temporal consistency looks normal across sampled transitions.")

    score = _clamp((len(spikes) - 1) / 4.0)
    return round(score, 3), findings


def _check_video_metadata(path: Path) -> tuple[float, list[str]]:
    """Check video container metadata for editing tool traces."""
    try:
        import json

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if not result.stdout.strip():
            return 0.3, ["ffprobe returned no metadata."]

        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        findings = []
        suspicious = 0.0

        encoder = tags.get("encoder", "") or tags.get("ENCODER", "")
        if encoder:
            findings.append(f"Video encoder: {encoder}")
            editing_tools = ["davinci", "premiere", "final cut", "capcut", "handbrake"]
            if any(tool in encoder.lower() for tool in editing_tools):
                findings.append(f"Encoding chain indicates export/edit software: {encoder}")
                suspicious += 1.2

        if not tags:
            findings.append("No metadata tags found in the video container.")
            suspicious += 0.8

        return round(_clamp(suspicious / 2.0), 3), findings
    except Exception:
        return 0.3, ["ffprobe not available; metadata check skipped."]


def _extract_audio_envelope(path: Path) -> tuple[list[float], list[float]]:
    temp_path = None
    try:
        import librosa
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                temp_path,
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if result.returncode != 0:
            return [], []

        y, sr = librosa.load(temp_path, sr=16000, mono=True)
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)
        return times.tolist(), np.asarray(rms, dtype="float32").tolist()
    except Exception:
        return [], []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _frame_model_score(frames: list[tuple[float, object]]) -> dict | None:
    import cv2

    if not frames:
        return None

    sampled = []
    if len(frames) <= 3:
        sampled = frames
    else:
        sampled = [frames[0], frames[len(frames) // 2], frames[-1]]

    scores = []
    labels = []
    sources = []
    for _, frame in sampled:
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            continue
        prediction = infer_image_model(encoded.tobytes())
        if not prediction:
            continue
        scores.append(prediction["score"])
        labels.append(prediction["label"])
        sources.append(prediction["source"])

    if not scores:
        return None

    return {
        "score": round(sum(scores) / len(scores), 3),
        "source": sources[0],
        "label": labels[0] if len(set(labels)) == 1 else "frame_ensemble",
        "raw": {"scores": scores, "labels": labels},
    }


def _check_lip_sync(path: Path, face_motion: dict) -> tuple[float | None, list[str]]:
    import numpy as np

    mouth_track = face_motion.get("mouth_track", [])
    motion_times = [entry["time_s"] for entry in mouth_track]
    motion_values = [entry["motion"] for entry in mouth_track]
    audio_times, audio_values = _extract_audio_envelope(path)

    if len(motion_values) < 4 or len(audio_values) < 8:
        return None, ["Lip-sync check skipped due to insufficient face or audio samples."]

    interp_audio = np.interp(motion_times, audio_times, audio_values)
    motion_array = np.asarray(motion_values, dtype="float32")
    audio_array = np.asarray(interp_audio, dtype="float32")

    if float(np.std(motion_array)) < 1e-6 or float(np.std(audio_array)) < 1e-6:
        return None, ["Lip-sync check skipped because motion/audio variance is too low."]

    correlation = float(np.corrcoef(motion_array, audio_array)[0, 1])
    if correlation != correlation:
        return None, ["Lip-sync correlation was not stable enough to score."]

    mismatch_score = _clamp((0.35 - correlation) / 0.70)
    finding = f"Audio-to-mouth correlation: {correlation:.2f}"
    if mismatch_score >= 0.60:
        finding += " (weak alignment)"
    return round(mismatch_score, 3), [finding]


async def analyze_video(path: Path) -> dict:
    loop = asyncio.get_running_loop()

    frames, duration, fps = await loop.run_in_executor(None, _extract_frames, path)

    face_future = loop.run_in_executor(None, _analyze_faces_and_motion, frames)
    temporal_future = loop.run_in_executor(None, _check_temporal_consistency, frames)
    metadata_future = loop.run_in_executor(None, _check_video_metadata, path)

    (frame_score, frame_findings, suspicious_regions, face_motion), (
        temporal_score,
        temporal_findings,
    ), (meta_score, meta_findings) = await asyncio.gather(
        face_future,
        temporal_future,
        metadata_future,
    )

    sync_score, sync_findings = await loop.run_in_executor(None, _check_lip_sync, path, face_motion)
    learned_frame_model = await loop.run_in_executor(None, _frame_model_score, frames)

    model_components = [(frame_score, 0.45), (temporal_score, 0.25)]
    if sync_score is not None:
        model_components.append((sync_score, 0.30))
    if learned_frame_model:
        model_components.append((learned_frame_model["score"], 0.40))
    weight_total = sum(weight for _, weight in model_components) or 1.0
    model_score = round(sum(value * weight for value, weight in model_components) / weight_total, 3)

    evidence = [
        f"Duration: {duration:.1f}s | FPS: {fps:.1f}",
        *frame_findings,
        *temporal_findings,
        *meta_findings,
    ]

    if learned_frame_model:
        evidence.append(
            "Learned frame deepfake detector returned "
            f"{learned_frame_model['score']:.0%} suspicious probability "
            f"via {learned_frame_model['source']} ({learned_frame_model['label']})."
        )
    if model_score >= 0.60:
        evidence.append(f"Combined deepfake score from frame/temporal evidence is {model_score:.0%}.")
    if sync_findings:
        evidence.extend(sync_findings)

    flags = []
    manipulation_hints = []

    if frame_score >= 0.60:
        flags.append("Face-boundary artifacts are repeatedly visible in sampled frames.")
        manipulation_hints.append("face swap or compositing")
    if temporal_score >= 0.55:
        flags.append("Temporal instability suggests frame-level manipulation or identity drift.")
        manipulation_hints.append("frame tampering")
    if sync_score is not None and sync_score >= 0.60:
        flags.append("Audio and mouth motion align poorly.")
        manipulation_hints.append("audio-video mismatch")
    if meta_score >= 0.60:
        manipulation_hints.append("metadata inconsistency")

    signal_quality = {
        "metadata": 0.62 if meta_findings else 0.40,
        "forensic": 0.82 if face_motion.get("face_coverage", 0.0) >= 0.30 else 0.66,
        "model": 0.90 if learned_frame_model else (0.76 if sync_score is not None else 0.68),
        "tamper": 0.80,
    }
    if sync_score is not None:
        signal_quality["cross_modal"] = 0.88

    return {
        "metadata": meta_score,
        "forensic": frame_score,
        "model": model_score,
        "tamper": temporal_score,
        "cross_modal": sync_score,
        "evidence": evidence,
        "flags": flags,
        "signal_quality": signal_quality,
        "manipulation_hints": list(dict.fromkeys(manipulation_hints)),
        "suspicious_regions": suspicious_regions,
        "external_model_evidence": learned_frame_model,
    }
