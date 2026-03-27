"""
services/scoring.py

Modality-aware forensic fusion for image, audio, video, and document analysis.
"""

from __future__ import annotations

from statistics import mean, pstdev

ENGINE_NAME = "DRISHYAM-ULTIMATE-V4"
ENGINE_VERSION = "4.0.0"
ENGINE_UPDATED_AT = "2026-03-25"
DEFAULT_SIGNAL_QUALITY = 0.65

SIGNAL_LABELS = {
    "metadata": "Metadata",
    "forensic": "Forensic artifacts",
    "model": "Synthetic-pattern detector",
    "tamper": "Tamper detector",
    "cross_modal": "Cross-modal mismatch",
}

MODALITY_PROFILES = {
    "image": {
        "name": "Image multi-signal forensics",
        "supports_cross_modal": False,
        "weights": {
            "metadata": 0.15,
            "forensic": 0.31,
            "model": 0.30,
            "tamper": 0.24,
        },
        "focus": [
            "metadata integrity",
            "AI-generated imagery",
            "splicing and cloning",
            "resampling/compositing artifacts",
        ],
    },
    "audio": {
        "name": "Audio synthesis and splice analysis",
        "supports_cross_modal": False,
        "weights": {
            "metadata": 0.16,
            "forensic": 0.27,
            "model": 0.35,
            "tamper": 0.22,
        },
        "focus": ["voice cloning", "TTS", "audio splicing"],
    },
    "video": {
        "name": "Video deepfake and temporal integrity analysis",
        "supports_cross_modal": True,
        "weights": {
            "metadata": 0.10,
            "forensic": 0.24,
            "model": 0.28,
            "tamper": 0.18,
            "cross_modal": 0.20,
        },
        "focus": ["face swap", "frame tampering", "audio-video mismatch"],
    },
    "pdf": {
        "name": "Document authenticity and AI-writing analysis",
        "supports_cross_modal": False,
        "weights": {
            "metadata": 0.18,
            "forensic": 0.25,
            "model": 0.32,
            "tamper": 0.25,
        },
        "focus": ["metadata tampering", "document assembly", "AI-written content"],
    },
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalize_signal(value: object) -> float | None:
    try:
        if value is None:
            return None
        return _clamp(float(value))
    except (TypeError, ValueError):
        return None


def _derive_band(coverage: float, quality: float, consensus: float) -> str:
    combined = (coverage * 0.45) + (quality * 0.35) + (consensus * 0.20)
    if combined >= 0.90 and coverage >= 0.85:
        return "Ultimate"
    if combined >= 0.75:
        return "High"
    if combined >= 0.55:
        return "Medium"
    return "Low"


def infer_manipulation_type(signals: dict) -> str:
    synthetic_score = _normalize_signal(signals.get("model")) or 0.5
    tamper_components = [
        _normalize_signal(signals.get("forensic")),
        _normalize_signal(signals.get("tamper")),
        _normalize_signal(signals.get("cross_modal")),
    ]
    tamper_scores = [score for score in tamper_components if score is not None]
    tamper_score = mean(tamper_scores) if tamper_scores else 0.5
    metadata_score = _normalize_signal(signals.get("metadata")) or 0.5

    if synthetic_score >= 0.68 and synthetic_score - tamper_score >= 0.08:
        return "synthetic_generation"
    if tamper_score >= 0.68 and tamper_score - synthetic_score >= 0.08:
        return "digital_alteration"
    if metadata_score >= 0.72 and tamper_score >= 0.55:
        return "document_or_container_tampering"
    if synthetic_score >= 0.6 and tamper_score >= 0.6:
        return "mixed_manipulation"
    return "suspicious_artifacts"


def infer_possible_manipulations(signals: dict, file_type: str) -> list[str]:
    hints = list(dict.fromkeys(signals.get("manipulation_hints", [])))
    if hints:
        return hints

    possible = []
    if (_normalize_signal(signals.get("model")) or 0.0) >= 0.65:
        if file_type == "audio":
            possible.append("voice cloning or synthetic speech")
        elif file_type == "pdf":
            possible.append("AI-generated document text")
        else:
            possible.append("synthetic generation")

    if (_normalize_signal(signals.get("tamper")) or 0.0) >= 0.60:
        if file_type == "audio":
            possible.append("splicing or assembly edits")
        elif file_type == "pdf":
            possible.append("document assembly or hidden-content tampering")
        else:
            possible.append("localized digital editing")

    if (_normalize_signal(signals.get("cross_modal")) or 0.0) >= 0.60:
        possible.append("cross-modal mismatch")

    if (_normalize_signal(signals.get("metadata")) or 0.0) >= 0.60:
        possible.append("metadata inconsistency")

    if not possible:
        possible.append("no dominant manipulation type isolated")

    return list(dict.fromkeys(possible))


def fuse_signals(signals: dict, file_type: str = "image") -> tuple[float, str, dict]:
    profile = MODALITY_PROFILES.get(file_type, MODALITY_PROFILES["image"])
    weights = profile["weights"]
    signal_quality = signals.get("signal_quality", {})

    possible_weight = sum(weights.values()) or 1.0
    present_base_weight = 0.0
    effective_weight = 0.0
    weighted_sum = 0.0
    components = []
    audit_trail = []

    for signal_name, base_weight in weights.items():
        value = _normalize_signal(signals.get(signal_name))
        if value is None:
            continue

        quality = _clamp(signal_quality.get(signal_name, DEFAULT_SIGNAL_QUALITY), 0.2, 1.0)
        reliability_multiplier = 0.45 + (0.55 * quality)
        signal_weight = base_weight * reliability_multiplier
        contribution = value * signal_weight
        present_base_weight += base_weight
        effective_weight += signal_weight
        weighted_sum += contribution

        component = {
            "signal": signal_name,
            "label": SIGNAL_LABELS.get(signal_name, signal_name),
            "value": round(value, 3),
            "quality": round(quality, 3),
            "base_weight": round(base_weight, 3),
            "effective_weight": round(signal_weight, 3),
            "deviation": round(abs(value - 0.5) * 2, 3),
            "direction": "suspicious" if value >= 0.5 else "authentic",
        }
        components.append(component)
        audit_trail.append(
            {
                "signal": signal_name,
                "value": round(value, 3),
                "quality": round(quality, 3),
                "effective_weight": round(signal_weight, 3),
            }
        )

    if not components:
        metrics = {
            "probability": 0.5,
            "coverage": 0.0,
            "evidence_strength": 0.0,
            "consensus": 0.0,
            "signal_breakdown": [],
            "dominant_signals": [],
            "possible_manipulations": ["insufficient evidence"],
            "manipulation_type": "insufficient_evidence",
            "model_engine": {
                "name": ENGINE_NAME,
                "version": ENGINE_VERSION,
                "last_updated": ENGINE_UPDATED_AT,
                "profile": profile["name"],
                "audit_trail": [],
            },
            "mathematical_terms": {
                "uncertainty": 1.0,
                "confidence_interval": [38.0, 62.0],
            },
        }
        return 50.0, "Low", metrics

    base_probability = weighted_sum / effective_weight
    values = [component["value"] for component in components]
    qualities = [component["quality"] for component in components]

    coverage = present_base_weight / possible_weight
    avg_quality = mean(qualities)
    consensus = _clamp(1.0 - ((pstdev(values) / 0.35) if len(values) > 1 else 0.0))

    signed_center = mean(values) - 0.5
    consensus_shift = signed_center * 0.12 * consensus
    final_probability = base_probability + consensus_shift

    high_signals = [component for component in components if component["value"] >= 0.70]
    low_signals = [component for component in components if component["value"] <= 0.30]
    contradiction = bool(high_signals and low_signals)
    if contradiction:
        final_probability = final_probability + ((0.5 - final_probability) * 0.18)
        audit_trail.append(
            {
                "signal": "contradiction_penalty",
                "value": round(final_probability, 3),
                "quality": round(avg_quality, 3),
                "effective_weight": 0.0,
            }
        )

    tamper_signal = _normalize_signal(signals.get("tamper"))
    forensic_signal = _normalize_signal(signals.get("forensic"))
    manipulation_hints = set(signals.get("manipulation_hints", []))
    digital_tamper_hints = {
        "localized digital editing",
        "copy-move or clone editing",
        "resized or composited content",
        "splicing or assembly edits",
        "document assembly or hidden-content tampering",
        "frame tampering",
        "face swap or compositing",
    }
    if tamper_signal is not None and tamper_signal >= 0.58 and manipulation_hints & digital_tamper_hints:
        anchor_forensic = forensic_signal if forensic_signal is not None else tamper_signal
        tamper_anchor = _clamp((tamper_signal * 0.78) + (anchor_forensic * 0.22))
        if tamper_anchor > final_probability:
            final_probability = tamper_anchor
            audit_trail.append(
                {
                    "signal": "tamper_anchor",
                    "value": round(tamper_anchor, 3),
                    "quality": round(avg_quality, 3),
                    "effective_weight": 0.0,
                }
            )

    override_candidates = [
        component
        for component in components
        if component["quality"] >= 0.75 and (component["value"] >= 0.94 or component["value"] <= 0.06)
    ]
    override_event = None
    if override_candidates:
        strongest = max(
            override_candidates,
            key=lambda item: (item["quality"], abs(item["value"] - 0.5)),
        )
        override_strength = 0.45 + (0.25 * strongest["quality"])
        final_probability = (
            (1.0 - override_strength) * final_probability
            + (override_strength * strongest["value"])
        )
        override_event = (
            f"High-confidence override from {strongest['label'].lower()} "
            f"({strongest['value']:.2f})"
        )
        audit_trail.append(
            {
                "signal": "override",
                "value": round(strongest["value"], 3),
                "quality": strongest["quality"],
                "effective_weight": round(override_strength, 3),
            }
        )

    final_probability = _clamp(final_probability)
    final_score = round(final_probability * 100.0, 1)

    certainty = _clamp(
        (coverage * 0.50)
        + (avg_quality * 0.35)
        + (consensus * 0.15)
        - (0.12 if contradiction else 0.0)
    )
    band = _derive_band(coverage, avg_quality, consensus)
    ci_half_width = round(4.0 + ((1.0 - certainty) * 18.0), 1)

    dominant_signals = sorted(
        components,
        key=lambda item: abs(item["value"] - 0.5) * item["effective_weight"],
        reverse=True,
    )[:3]

    metrics = {
        "probability": round(final_probability, 4),
        "coverage": round(coverage, 3),
        "evidence_strength": round(avg_quality, 3),
        "consensus": round(consensus, 3),
        "signal_breakdown": components,
        "dominant_signals": dominant_signals,
        "possible_manipulations": infer_possible_manipulations(signals, file_type),
        "manipulation_type": infer_manipulation_type(signals),
        "model_engine": {
            "name": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "last_updated": ENGINE_UPDATED_AT,
            "profile": profile["name"],
            "audit_trail": audit_trail,
            "override_event": override_event,
        },
        "mathematical_terms": {
            "uncertainty": round(1.0 - certainty, 3),
            "confidence_interval": [
                round(max(0.0, final_score - ci_half_width), 1),
                round(min(100.0, final_score + ci_half_width), 1),
            ],
        },
    }

    return final_score, band, metrics


def get_model_metrics(file_type: str) -> dict:
    profile = MODALITY_PROFILES.get(file_type, MODALITY_PROFILES["image"])
    return {
        "engine": profile["name"],
        "supports_cross_modal": profile["supports_cross_modal"],
        "supported_signals": list(profile["weights"].keys()),
        "weight_profile": profile["weights"],
        "focus": profile["focus"],
        "calibration_status": "heuristic_ensemble",
    }


def score_to_label(score: float, signals: dict | None = None) -> str:
    if score < 25:
        return "Likely Authentic"
    if score < 45:
        return "Mostly Authentic, Minor Artifacts"
    if score < 65:
        return "Suspicious / Needs Review"

    manipulation_type = infer_manipulation_type(signals or {})
    if manipulation_type == "synthetic_generation":
        return "Likely AI-Generated / Synthetic"
    if manipulation_type == "digital_alteration":
        return "Likely Digitally Altered"
    if manipulation_type == "document_or_container_tampering":
        return "Likely Metadata / Document Tampering"
    return "Likely Manipulated"
