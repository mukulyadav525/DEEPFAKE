"""
services/explanation.py
Builds a user-facing report from raw signals and fused scoring output.
"""

from __future__ import annotations

from services.scoring import (
    ENGINE_NAME,
    ENGINE_VERSION,
    infer_manipulation_type,
    infer_possible_manipulations,
    score_to_label,
)


def _risk_level(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 45:
        return "Guarded"
    if score < 65:
        return "Elevated"
    if score < 85:
        return "High"
    return "Critical"


def _summary(file_type: str, label: str, score: float, manipulation_type: str, top_reason: str) -> str:
    readable_type = manipulation_type.replace("_", " ")
    return (
        f"{file_type.capitalize()} analysis returned '{label}' with a score of {score:.1f}/100. "
        f"Primary pattern: {readable_type}. Strongest reason: {top_reason}"
    )


def build_report(file_type: str, score: float, band: str, signals: dict) -> dict:
    metrics = signals.get("metrics", {})
    label = score_to_label(score, signals)

    evidence = signals.get("evidence", [])
    flags = signals.get("flags", [])

    if flags:
        top_reason = flags[0]
    else:
        suspicious_terms = (
            "elevated",
            "mismatch",
            "anomal",
            "synthetic",
            "hidden",
            "editing",
            "tamper",
            "splice",
            "copy-move",
            "resampling",
            "composit",
            "weak",
        )
        top_reason = next(
            (item for item in evidence if any(term in item.lower() for term in suspicious_terms)),
            evidence[0] if evidence else "Insufficient evidence for a strong conclusion.",
        )

    manipulation_type = metrics.get("manipulation_type") or infer_manipulation_type(signals)
    possible_manipulations = metrics.get("possible_manipulations") or infer_possible_manipulations(
        signals,
        file_type,
    )

    report = {
        "file_type": file_type,
        "result": label,
        "score": score,
        "confidence_band": band,
        "risk_level": _risk_level(score),
        "manipulation_type": manipulation_type,
        "possible_manipulations": possible_manipulations,
        "top_reason": top_reason,
        "summary": _summary(file_type, label, score, manipulation_type, top_reason),
        "evidence": evidence[:12],
        "flags": flags,
        "technical_breakdown": {
            "model_engine": f"{ENGINE_NAME} {ENGINE_VERSION}",
            "analysis_depth": "Modality-aware multi-signal fusion",
            "signal_coverage": metrics.get("coverage"),
            "evidence_strength": metrics.get("evidence_strength"),
            "consensus": metrics.get("consensus"),
            "dominant_signals": metrics.get("dominant_signals", []),
            "signal_breakdown": metrics.get("signal_breakdown", []),
            "forensic_audit": metrics.get("model_engine", {}).get("audit_trail", []),
            "confidence_interval": metrics.get("mathematical_terms", {}).get("confidence_interval"),
            "metadata_diagnostics": signals.get("metadata_diagnostics"),
            "modality_diagnostics": (
                signals.get("image_diagnostics")
                or signals.get("audio_diagnostics")
            ),
            "external_model_evidence": signals.get("external_model_evidence"),
        },
        "warning": (
            "This result is a forensic screening output, not a legal certification. "
            "For high-stakes use, validate against calibrated models and manual review."
        ),
    }

    for key in [
        "suspicious_regions",
        "suspicious_segments",
        "suspicious_pages",
        "document_kind",
        "metadata_diagnostics",
        "image_diagnostics",
        "audio_diagnostics",
        "external_model_evidence",
    ]:
        if key in signals:
            report[key] = signals[key]

    return report
