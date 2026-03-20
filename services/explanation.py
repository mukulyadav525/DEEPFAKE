"""
services/explanation.py
Builds the final user-facing report from raw signals + score.
"""

from services.scoring import score_to_label


def build_report(file_type: str, score: float, band: str, signals: dict) -> dict:
    """
    Ultimate Forensic Report (v3.0.0).
    Surfaces the granular 'Audit Trail' from the scoring engine.
    """
    label = score_to_label(score)

    evidence = signals.get("evidence", [])
    flags = signals.get("flags", [])

    # Compose top reason
    if flags:
        top_reason = flags[0]
    elif evidence:
        top_reason = evidence[0]
    else:
        top_reason = "Insufficient data for strong conclusion."

    report = {
        "file_type": file_type,
        "result": label,
        "score": score,
        "confidence_band": band,
        "top_reason": top_reason,
        "evidence": evidence,
        "flags": flags,
        "technical_breakdown": {
            "model_engine": "DRISHYAM-ULTIMATE-V3",
            "analysis_depth": "Ultimate Cross-Modal Lineage",
            "modality_coverage": band,
            "forensic_audit": signals.get("metrics", {}).get("model_engine", {}).get("audit_trail", [])
        },
        "warning": (
            "This is an Ultimate-tier forensic assessment. "
            "It utilizes consensus-override logic and cross-modal verification."
        ),
    }

    # Attach modality-specific detail
    for key in ["suspicious_regions", "suspicious_segments", "suspicious_pages"]:
        if key in signals:
            report[key] = signals[key]

    return report
