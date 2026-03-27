"""
analyzers/audio.py

Audio forensics pipeline for synthetic-voice and edit detection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from services.model_runtime import infer_audio_model


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _load_audio(path: Path) -> tuple:
    """Load audio as numpy array via librosa. Returns (y, sr)."""
    try:
        import librosa

        y, sr = librosa.load(str(path), sr=16000, mono=True)
        return y, sr
    except Exception as exc:
        raise RuntimeError(f"Could not load audio: {exc}") from exc


def _check_basic_properties(y, sr) -> tuple[float, list[str], dict]:
    import numpy as np

    findings = []
    suspicious = 0.0

    duration = len(y) / sr if sr else 0.0
    silence_ratio = float(np.mean(np.abs(y) < 0.01))
    clip_ratio = float(np.mean(np.abs(y) > 0.99))
    dynamic_range = float(np.percentile(np.abs(y), 95) - np.percentile(np.abs(y), 5))

    findings.append(f"Duration: {duration:.1f}s | Sample rate: {sr}Hz")

    if silence_ratio > 0.45:
        findings.append(f"High silence ratio ({silence_ratio:.0%}) suggests unusually gated audio.")
        suspicious += 1.0

    if clip_ratio > 0.01:
        findings.append("Clipping artifacts detected, consistent with re-recording or poor export.")
        suspicious += 1.0

    if dynamic_range < 0.10:
        findings.append(
            f"Low dynamic range ({dynamic_range:.3f}) can indicate aggressive normalization or synthesis."
        )
        suspicious += 1.0

    diagnostics = {
        "duration": round(duration, 2),
        "silence_ratio": round(silence_ratio, 3),
        "clip_ratio": round(clip_ratio, 4),
        "dynamic_range": round(dynamic_range, 4),
    }
    return round(_clamp(suspicious / 3.0), 3), findings, diagnostics


def _compute_spectrogram_artifacts(y, sr) -> tuple[float, dict]:
    import numpy as np
    import librosa

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    spectral_flux = float(np.mean(np.abs(np.diff(mel_db, axis=1))))

    rms = librosa.feature.rms(y=y)[0]
    energy_cv = float(np.std(rms) / (np.mean(rms) + 1e-6))

    low_flux_score = _clamp((5.0 - spectral_flux) / 5.0)
    low_energy_cv_score = _clamp((0.35 - energy_cv) / 0.35)
    score = (low_flux_score * 0.60) + (low_energy_cv_score * 0.40)

    diagnostics = {
        "spectral_flux": round(spectral_flux, 3),
        "energy_cv": round(energy_cv, 3),
    }
    return round(_clamp(score), 3), diagnostics


def _estimate_synthetic_voice_score(y, sr) -> tuple[float, dict]:
    import numpy as np
    import librosa

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_cv = float(np.std(centroid) / (np.mean(centroid) + 1e-6))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_cv = float(np.std(zcr) / (np.mean(zcr) + 1e-6))

    low_mfcc_variance = _clamp((250.0 - mfcc_variance) / 250.0)
    low_centroid_cv = _clamp((0.30 - centroid_cv) / 0.30)
    low_zcr_cv = _clamp((0.40 - zcr_cv) / 0.40)

    score = (
        (low_mfcc_variance * 0.40)
        + (low_centroid_cv * 0.35)
        + (low_zcr_cv * 0.25)
    )

    diagnostics = {
        "mfcc_variance": round(mfcc_variance, 3),
        "centroid_cv": round(centroid_cv, 3),
        "zcr_cv": round(zcr_cv, 3),
    }
    return round(_clamp(score), 3), diagnostics


def _merge_close_indices(indices: list[int], gap: int = 2) -> list[int]:
    if not indices:
        return []

    merged = [indices[0]]
    for index in indices[1:]:
        if index - merged[-1] > gap:
            merged.append(index)
    return merged


def _detect_splices(y, sr) -> tuple[float, list[dict]]:
    import numpy as np
    import librosa

    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    rms_diff = np.abs(np.diff(rms))
    centroid_diff = np.abs(np.diff(centroid))

    rms_threshold = float(np.mean(rms_diff) + (3 * np.std(rms_diff)))
    centroid_threshold = float(np.mean(centroid_diff) + (3 * np.std(centroid_diff)))

    candidate_frames = set(np.where(rms_diff > rms_threshold)[0].tolist())
    candidate_frames.update(np.where(centroid_diff > centroid_threshold)[0].tolist())
    merged_frames = _merge_close_indices(sorted(candidate_frames))

    segments = []
    for frame_idx in merged_frames[:12]:
        severity = 0.0
        if frame_idx < len(rms_diff) and rms_threshold > 0:
            severity += min(float(rms_diff[frame_idx] / rms_threshold), 2.0) * 0.5
        if frame_idx < len(centroid_diff) and centroid_threshold > 0:
            severity += min(float(centroid_diff[frame_idx] / centroid_threshold), 2.0) * 0.5
        segments.append(
            {
                "time_s": round(frame_idx * hop / sr, 2),
                "severity": round(min(severity, 1.0), 3),
                "source": "energy_or_spectral_jump",
            }
        )

    score = round(_clamp((len(merged_frames) - 1) / 7.0), 3)
    return score, segments


async def analyze_audio(path: Path) -> dict:
    loop = asyncio.get_running_loop()

    y, sr = await loop.run_in_executor(None, _load_audio, path)

    basic_future = loop.run_in_executor(None, _check_basic_properties, y, sr)
    spectrogram_future = loop.run_in_executor(None, _compute_spectrogram_artifacts, y, sr)
    synthetic_future = loop.run_in_executor(None, _estimate_synthetic_voice_score, y, sr)
    splice_future = loop.run_in_executor(None, _detect_splices, y, sr)

    (basic_score, basic_findings, basic_diag), (spec_score, spec_diag), (
        synthetic_score,
        synthetic_diag,
    ), (splice_score, splice_segments) = await asyncio.gather(
        basic_future,
        spectrogram_future,
        synthetic_future,
        splice_future,
    )
    learned_model_result = await loop.run_in_executor(None, infer_audio_model, path)

    forensic_score = round((spec_score * 0.65) + (synthetic_score * 0.35), 3)
    heuristic_model_score = round((synthetic_score * 0.70) + (spec_score * 0.30), 3)
    model_score = heuristic_model_score
    if learned_model_result:
        model_score = round(
            (heuristic_model_score * 0.40) + (learned_model_result["score"] * 0.60),
            3,
        )

    evidence = list(basic_findings)

    if spec_score >= 0.55:
        evidence.append(
            "Spectrogram dynamics are unusually smooth "
            f"(flux {spec_diag['spectral_flux']}, energy CV {spec_diag['energy_cv']})."
        )
    if learned_model_result:
        evidence.append(
            "Learned voice-spoof detector returned "
            f"{learned_model_result['score']:.0%} suspicious probability "
            f"via {learned_model_result['source']} ({learned_model_result['label']})."
        )
    if model_score >= 0.60:
        evidence.append(
            "Voice-pattern score is elevated "
            f"(MFCC variance {synthetic_diag['mfcc_variance']}, "
            f"centroid CV {synthetic_diag['centroid_cv']}, "
            f"ZCR CV {synthetic_diag['zcr_cv']})."
        )
    if splice_score >= 0.35:
        evidence.append(
            f"Detected {len(splice_segments)} likely edit boundary point(s) in the waveform."
        )

    flags = []
    manipulation_hints = []

    if model_score >= 0.68:
        flags.append("Synthetic-voice pattern score is strong.")
        manipulation_hints.append("voice cloning or synthetic speech")
    if splice_score >= 0.60:
        flags.append("Multiple abrupt spectral/energy jumps suggest splicing or assembly edits.")
        manipulation_hints.append("splicing or assembly edits")
    if basic_score >= 0.60:
        manipulation_hints.append("processing or export artifacts")

    signal_quality = {
        "metadata": 0.58,
        "forensic": 0.82,
        "model": 0.88 if learned_model_result else 0.72,
        "tamper": 0.86 if splice_segments else 0.74,
    }

    return {
        "metadata": basic_score,
        "forensic": forensic_score,
        "model": model_score,
        "tamper": splice_score,
        "cross_modal": None,
        "evidence": evidence,
        "flags": flags,
        "signal_quality": signal_quality,
        "manipulation_hints": list(dict.fromkeys(manipulation_hints)),
        "suspicious_segments": splice_segments,
        "audio_diagnostics": {
            **basic_diag,
            **spec_diag,
            **synthetic_diag,
            "heuristic_model_score": heuristic_model_score,
            "learned_model_score": learned_model_result["score"] if learned_model_result else None,
        },
        "external_model_evidence": learned_model_result,
    }
