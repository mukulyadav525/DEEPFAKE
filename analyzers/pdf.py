"""
analyzers/pdf.py

Document forensics for PDF, DOCX, and TXT inputs.
"""

from __future__ import annotations

import asyncio
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from services.model_runtime import infer_text_model


DOCX_NS = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _score_text_patterns(text: str) -> tuple[float, list[str], dict]:
    clean_text = text.strip()
    if len(clean_text) < 120:
        return 0.25, ["Not enough text to run strong linguistic analysis."], {}

    words = re.findall(r"[A-Za-z0-9']+", clean_text.lower())
    if len(words) < 50:
        return 0.25, ["Text payload is too small for a stable AI-writing estimate."], {}

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean_text)
        if len(sentence.strip().split()) >= 4
    ]
    paragraphs = [paragraph.strip() for paragraph in clean_text.splitlines() if len(paragraph.strip()) > 20]

    sentence_lengths = [len(sentence.split()) for sentence in sentences] or [0]
    paragraph_lengths = [len(paragraph.split()) for paragraph in paragraphs] or [0]

    sentence_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    paragraph_cv = (
        statistics.pstdev(paragraph_lengths) / (statistics.mean(paragraph_lengths) + 1e-6)
        if len(paragraph_lengths) > 1
        else 0.0
    )

    lexical_diversity = len(set(words)) / max(len(words), 1)
    bigrams = list(zip(words, words[1:]))
    max_bigram_share = 0.0
    if bigrams:
        from collections import Counter

        bigram_counts = Counter(bigrams)
        max_bigram_share = max(bigram_counts.values()) / len(bigrams)

    opener_counts = {}
    for sentence in sentences:
        opener = " ".join(sentence.lower().split()[:3])
        if opener:
            opener_counts[opener] = opener_counts.get(opener, 0) + 1
    max_opener_repeat = max(opener_counts.values()) if opener_counts else 0

    hedges = [
        "it is worth noting",
        "it is important to note",
        "in conclusion",
        "in summary",
        "as an ai",
        "delve into",
        "dive deep",
        "leverage",
        "utilize",
        "furthermore",
        "moreover",
        "in today's world",
    ]
    hit_phrases = [phrase for phrase in hedges if phrase in clean_text.lower()]

    low_sentence_variance = _clamp((25.0 - sentence_variance) / 25.0)
    low_lexical_diversity = _clamp((0.38 - lexical_diversity) / 0.18)
    repetition_score = _clamp((max_bigram_share - 0.015) / 0.05)
    low_paragraph_cv = _clamp((0.35 - paragraph_cv) / 0.35)
    hedge_score = _clamp((len(hit_phrases) - 2) / 4.0)
    opener_score = _clamp((max_opener_repeat - 2) / 4.0)

    score = (
        (low_sentence_variance * 0.20)
        + (low_lexical_diversity * 0.20)
        + (repetition_score * 0.20)
        + (low_paragraph_cv * 0.15)
        + (hedge_score * 0.15)
        + (opener_score * 0.10)
    )

    findings = []
    if low_sentence_variance >= 0.55:
        findings.append(
            f"Sentence-length variance is low ({sentence_variance:.1f}), which can indicate templated text."
        )
    if low_lexical_diversity >= 0.55:
        findings.append(
            f"Lexical diversity is low ({lexical_diversity:.2f}), suggesting repetitive wording."
        )
    if repetition_score >= 0.45:
        findings.append(
            f"Repeated phrase density is elevated (top bigram share {max_bigram_share:.3f})."
        )
    if hedge_score >= 0.45:
        findings.append(f"Several AI-associated discourse markers appear: {', '.join(hit_phrases[:4])}.")
    if not findings:
        findings.append("No dominant AI-writing pattern stood out in the text layer.")

    diagnostics = {
        "sentence_variance": round(sentence_variance, 3),
        "lexical_diversity": round(lexical_diversity, 3),
        "max_bigram_share": round(max_bigram_share, 4),
        "paragraph_cv": round(paragraph_cv, 3),
        "hedge_hits": len(hit_phrases),
        "max_opener_repeat": max_opener_repeat,
    }
    return round(_clamp(score), 3), findings, diagnostics


def _extract_pdf_payload(path: Path) -> dict:
    import fitz

    doc = fitz.open(str(path))
    page_count = doc.page_count
    metadata = doc.metadata or {}
    creator = metadata.get("creator", "") or ""
    producer = metadata.get("producer", "") or ""
    mod_date = metadata.get("modDate", "") or ""
    creation_date = metadata.get("creationDate", "") or ""

    meta_findings = [
        f"Pages: {page_count} | Creator: {creator or 'None'} | Producer: {producer or 'None'}"
    ]
    meta_suspicious = 0.0

    if not creator and not producer:
        meta_findings.append("Creator/producer metadata is missing.")
        meta_suspicious += 0.8

    if mod_date and creation_date and mod_date < creation_date:
        meta_findings.append("Modification date precedes creation date.")
        meta_suspicious += 1.2

    known_ai_tools = ["chatgpt", "gpt", "jasper", "copy.ai", "writesonic"]
    combined_meta = f"{creator} {producer}".lower()
    for tool in known_ai_tools:
        if tool in combined_meta:
            meta_findings.append(f"AI-associated tool string found in metadata: {tool}")
            meta_suspicious += 1.5
            break

    all_text = []
    page_font_sets = []
    hidden_pages = set()
    image_heavy_pages = set()

    for page_index, page in enumerate(doc):
        all_text.append(page.get_text())

        fonts = {font[3] for font in page.get_fonts(full=True) if len(font) > 3 and font[3]}
        page_font_sets.append(fonts)

        if len(page.get_images(full=True)) >= 4:
            image_heavy_pages.add(page_index + 1)

        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("size", 12) < 2:
                        hidden_pages.add(page_index + 1)

    doc.close()

    all_fonts = set().union(*page_font_sets) if page_font_sets else set()
    drifted_pages = 0
    if len(page_font_sets) > 1:
        base_fonts = page_font_sets[0]
        drifted_pages = sum(1 for font_set in page_font_sets[1:] if font_set != base_fonts)

    forensic_findings = []
    forensic_suspicious = 0.0
    tamper_suspicious = 0.0

    if len(all_fonts) > 8:
        forensic_findings.append(
            f"High font diversity detected ({len(all_fonts)} fonts), which can indicate assembled pages."
        )
        forensic_suspicious += 1.0

    if drifted_pages > max(1, int(page_count * 0.3)):
        forensic_findings.append(f"Font usage changes across {drifted_pages} page(s).")
        forensic_suspicious += 0.8
        tamper_suspicious += 0.7

    if hidden_pages:
        forensic_findings.append(f"Very small or hidden text appears on pages: {sorted(hidden_pages)}.")
        forensic_suspicious += 0.7
        tamper_suspicious += 1.2

    if image_heavy_pages:
        forensic_findings.append(f"Image-heavy pages detected: {sorted(image_heavy_pages)}.")
        forensic_suspicious += 0.5

    if not forensic_findings:
        forensic_findings.append("PDF structure looks consistent at the page/font layer.")

    suspicious_pages = sorted(hidden_pages | image_heavy_pages)
    return {
        "document_kind": "pdf",
        "text": "\n".join(all_text).strip(),
        "meta_score": round(_clamp(meta_suspicious / 3.0), 3),
        "meta_findings": meta_findings,
        "forensic_score": round(_clamp(forensic_suspicious / 3.0), 3),
        "tamper_score": round(_clamp(tamper_suspicious / 2.0), 3),
        "forensic_findings": forensic_findings,
        "suspicious_pages": suspicious_pages,
    }


def _extract_docx_payload(path: Path) -> dict:
    meta_findings = []
    forensic_findings = []
    suspicious_pages = []

    with zipfile.ZipFile(path) as archive:
        core_root = None
        document_root = None
        document_xml = ""
        font_names = set()

        if "docProps/core.xml" in archive.namelist():
            core_root = ET.fromstring(archive.read("docProps/core.xml"))

        if "word/document.xml" in archive.namelist():
            raw_document = archive.read("word/document.xml")
            document_xml = raw_document.decode("utf-8", errors="ignore")
            document_root = ET.fromstring(raw_document)

        if "word/fontTable.xml" in archive.namelist():
            font_root = ET.fromstring(archive.read("word/fontTable.xml"))
            for node in font_root.findall(".//w:font", DOCX_NS):
                name = node.attrib.get(f"{{{DOCX_NS['w']}}}name")
                if name:
                    font_names.add(name)

        creator = ""
        last_modified_by = ""
        created = ""
        modified = ""

        if core_root is not None:
            creator = core_root.findtext(".//dc:creator", default="", namespaces=DOCX_NS)
            last_modified_by = core_root.findtext(
                ".//cp:lastModifiedBy",
                default="",
                namespaces=DOCX_NS,
            )
            created = core_root.findtext(".//dcterms:created", default="", namespaces=DOCX_NS)
            modified = core_root.findtext(".//dcterms:modified", default="", namespaces=DOCX_NS)

        meta_findings.append(
            "Creator: "
            f"{creator or 'None'} | Last modified by: {last_modified_by or 'None'}"
        )

        meta_suspicious = 0.0
        if not creator and not last_modified_by:
            meta_findings.append("DOCX core properties are sparse.")
            meta_suspicious += 0.7

        if modified and created and modified < created:
            meta_findings.append("Modified timestamp precedes created timestamp.")
            meta_suspicious += 1.0

        actor_fields = f"{creator} {last_modified_by}".lower()
        if any(tool in actor_fields for tool in ["chatgpt", "gpt", "jasper", "copy.ai"]):
            meta_findings.append("AI-associated string found in DOCX core properties.")
            meta_suspicious += 1.4

        text_content = ""
        revisions = 0
        comments = 0
        hidden_runs = 0
        if document_root is not None:
            text_nodes = document_root.findall(".//w:t", DOCX_NS)
            text_content = " ".join((node.text or "") for node in text_nodes)
            revisions = len(document_root.findall(".//w:ins", DOCX_NS)) + len(
                document_root.findall(".//w:del", DOCX_NS)
            )
            comments = len(document_root.findall(".//w:commentRangeStart", DOCX_NS))
            hidden_runs = document_xml.count("w:vanish")

        forensic_suspicious = 0.0
        tamper_suspicious = 0.0

        if len(font_names) > 8:
            forensic_findings.append(f"Document references many fonts ({len(font_names)}).")
            forensic_suspicious += 0.8

        if revisions:
            forensic_findings.append(f"Tracked insert/delete revisions detected ({revisions}).")
            forensic_suspicious += 0.6
            tamper_suspicious += 0.8

        if comments:
            forensic_findings.append(f"Comment anchors detected ({comments}).")
            tamper_suspicious += 0.4

        if hidden_runs:
            forensic_findings.append(f"Hidden or vanished text markers detected ({hidden_runs}).")
            forensic_suspicious += 0.7
            tamper_suspicious += 1.2

        if not forensic_findings:
            forensic_findings.append("DOCX structure looks ordinary at the style/revision layer.")

    return {
        "document_kind": "docx",
        "text": text_content.strip(),
        "meta_score": round(_clamp(meta_suspicious / 2.5), 3),
        "meta_findings": meta_findings,
        "forensic_score": round(_clamp(forensic_suspicious / 2.5), 3),
        "tamper_score": round(_clamp(tamper_suspicious / 2.0), 3),
        "forensic_findings": forensic_findings,
        "suspicious_pages": suspicious_pages,
    }


def _extract_txt_payload(path: Path) -> dict:
    content = path.read_text(encoding="utf-8", errors="ignore")
    zero_width_count = len(re.findall(r"[\u200B-\u200F\uFEFF]", content))

    meta_findings = [
        "Plain text file detected; metadata-based validation is limited.",
        f"Character count: {len(content)}",
    ]
    forensic_findings = []
    tamper_score = 0.0

    if zero_width_count:
        forensic_findings.append(f"Zero-width/invisible characters detected ({zero_width_count}).")
        tamper_score = _clamp((zero_width_count - 1) / 10.0)
    else:
        forensic_findings.append("No hidden zero-width characters detected.")

    return {
        "document_kind": "txt",
        "text": content.strip(),
        "meta_score": 0.10,
        "meta_findings": meta_findings,
        "forensic_score": 0.15 if zero_width_count == 0 else round(tamper_score, 3),
        "tamper_score": round(tamper_score, 3),
        "forensic_findings": forensic_findings,
        "suspicious_pages": [],
    }


def _extract_document_payload(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_payload(path)
    if suffix == ".docx":
        return _extract_docx_payload(path)
    if suffix == ".txt":
        return _extract_txt_payload(path)
    raise ValueError(f"Unsupported document type: {suffix}")


async def analyze_pdf(path: Path) -> dict:
    loop = asyncio.get_running_loop()

    payload = await loop.run_in_executor(None, _extract_document_payload, path)
    text_score, text_findings, text_diag = await loop.run_in_executor(
        None,
        _score_text_patterns,
        payload["text"],
    )
    learned_model_result = None
    if len(payload["text"]) >= 120:
        learned_model_result = await loop.run_in_executor(None, infer_text_model, payload["text"])
        if learned_model_result:
            text_score = round((text_score * 0.45) + (learned_model_result["score"] * 0.55), 3)

    evidence = payload["meta_findings"] + payload["forensic_findings"] + text_findings

    if learned_model_result:
        evidence.append(
            "Learned text-authenticity detector returned "
            f"{learned_model_result['score']:.0%} suspicious probability "
            f"via {learned_model_result['source']} ({learned_model_result['label']})."
        )
    if text_score >= 0.60:
        evidence.append(
            "Text-layer generation score is elevated "
            f"(lexical diversity {text_diag.get('lexical_diversity', 'n/a')}, "
            f"sentence variance {text_diag.get('sentence_variance', 'n/a')})."
        )

    flags = []
    manipulation_hints = []

    if payload["meta_score"] >= 0.60:
        flags.append("Document metadata contains notable inconsistencies.")
        manipulation_hints.append("metadata inconsistency")
    if payload["tamper_score"] >= 0.55:
        flags.append("Document structure indicates possible hidden content or assembly edits.")
        manipulation_hints.append("document assembly or hidden-content tampering")
    if text_score >= 0.70:
        flags.append("Writing pattern score is strongly consistent with AI-generated text.")
        manipulation_hints.append("AI-generated document text")

    signal_quality = {
        "metadata": 0.62,
        "forensic": 0.78,
        "model": 0.86 if learned_model_result else (0.68 if len(payload["text"]) >= 200 else 0.52),
        "tamper": 0.82 if payload["suspicious_pages"] else 0.70,
    }

    return {
        "metadata": payload["meta_score"],
        "forensic": payload["forensic_score"],
        "model": text_score,
        "tamper": payload["tamper_score"],
        "cross_modal": None,
        "evidence": evidence,
        "flags": flags,
        "signal_quality": signal_quality,
        "manipulation_hints": list(dict.fromkeys(manipulation_hints)),
        "suspicious_pages": payload["suspicious_pages"],
        "document_kind": payload["document_kind"],
        "external_model_evidence": learned_model_result,
    }
