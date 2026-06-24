"""
keyword_scanner.py — Keyword-based pre-screening tool.

Scans note text for celiac-related phrases defined in
``celiac_keywords_latest.yaml`` and returns structured signals matching
the clinician's diagnostic categories.

Refactored from the ``keyword_prescreen_chunk`` logic in ``rag_diagnose.py``.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from pheno_agent.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KeywordSignals:
    """Keyword detection results for a single note."""
    outside_biopsy: bool = False
    marsh_positive: bool = False
    marsh_indeterminate: bool = False
    iel_positive: bool = False
    iel_negative: bool = False
    villous_abnormal: bool = False
    villous_normal: bool = False
    decision: str = "Negative"  # derived from signal combination


@dataclass
class PatientKeywordReport:
    """Aggregated keyword scan results across all notes."""
    grid: str = ""
    per_note: List[KeywordSignals] = field(default_factory=list)
    aggregated_decision: str = "Negative"
    summary_text: str = ""


# ---------------------------------------------------------------------------
# Keyword loader (cached)
# ---------------------------------------------------------------------------

_keywords: Optional[Dict[str, Any]] = None


def _load_keywords(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load celiac keyword lists from YAML (cached after first call)."""
    global _keywords
    if _keywords is not None:
        return _keywords

    path = path or cfg.keywords_path
    with open(path, "r") as f:
        _keywords = yaml.safe_load(f)
    logger.info("Loaded keyword dictionaries from %s", path)
    return _keywords


def _contains_any(text: str, phrases: List[str]) -> bool:
    """Return True if any phrase is found in text (case-insensitive)."""
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in phrases)


# ---------------------------------------------------------------------------
# Per-note scanning
# ---------------------------------------------------------------------------

def scan_note_for_keywords(
    note_text: str, keywords_path: Optional[Path] = None,
) -> KeywordSignals:
    """
    Run keyword-based celiac signal detection on a single note.

    Mirrors the detection logic from ``rag_diagnose.py::keyword_prescreen_chunk``.
    Negative/normal phrases are removed from the text before checking for
    positive/abnormal phrases to avoid false positives from negated contexts.

    Parameters
    ----------
    note_text : str
        The text of a single medical note.
    keywords_path : Path, optional
        Override the default keywords YAML path.

    Returns
    -------
    KeywordSignals
        Detected signals and the derived decision.
    """
    kw = _load_keywords(keywords_path)
    report_lower = note_text.lower()

    # Build a "cleaned" version with negatives removed for positive matching
    cleaned = report_lower
    for phrase in kw.get("iel_negative_phrases", []):
        cleaned = cleaned.replace(phrase.lower(), "")
    for phrase in kw.get("normal_villous", []):
        cleaned = cleaned.replace(phrase.lower(), "")

    signals = KeywordSignals(
        outside_biopsy=_contains_any(report_lower, kw.get("outside_biopsy_confirmed", [])),
        marsh_positive=_contains_any(report_lower, kw.get("marsh_positive", [])),
        marsh_indeterminate=_contains_any(report_lower, kw.get("marsh_indeterminate", [])),
        iel_positive=_contains_any(cleaned, kw.get("iel_positive_phrases", [])),
        iel_negative=_contains_any(report_lower, kw.get("iel_negative_phrases", [])),
        villous_abnormal=_contains_any(cleaned, kw.get("abnormal_villous", [])),
        villous_normal=_contains_any(report_lower, kw.get("normal_villous", [])),
    )

    # Derive decision (same precedence as rag_diagnose.py)
    if signals.outside_biopsy:
        signals.decision = "Positive"
    elif signals.marsh_positive:
        signals.decision = "Positive"
    elif signals.marsh_indeterminate:
        signals.decision = "Indeterminate"
    elif signals.iel_positive and signals.villous_abnormal:
        signals.decision = "Positive"
    elif signals.iel_negative and signals.villous_normal:
        signals.decision = "Negative"
    elif not any([
        signals.outside_biopsy, signals.marsh_positive, signals.marsh_indeterminate,
        signals.iel_positive, signals.iel_negative,
        signals.villous_abnormal, signals.villous_normal,
    ]):
        signals.decision = "Negative"
    else:
        signals.decision = "Indeterminate"

    return signals


# ---------------------------------------------------------------------------
# Patient-level scanning
# ---------------------------------------------------------------------------

def _aggregate_decisions(decisions: List[str]) -> str:
    """Aggregate note-level decisions: Positive > Indeterminate > Negative."""
    if "Positive" in decisions:
        return "Positive"
    if "Indeterminate" in decisions:
        return "Indeterminate"
    return "Negative"


def scan_patient_notes(
    notes_texts: List[str],
    grid: str = "",
    keywords_path: Optional[Path] = None,
) -> PatientKeywordReport:
    """
    Run keyword pre-screen on all notes for a patient.

    Parameters
    ----------
    notes_texts : list[str]
        List of note text strings.
    grid : str
        Patient identifier (for reporting).
    keywords_path : Path, optional
        Override the default keywords YAML path.

    Returns
    -------
    PatientKeywordReport
        Per-note signals and the aggregated decision.
    """
    report = PatientKeywordReport(grid=grid)

    for text in notes_texts:
        sig = scan_note_for_keywords(text, keywords_path)
        report.per_note.append(sig)

    decisions = [s.decision for s in report.per_note]
    report.aggregated_decision = _aggregate_decisions(decisions)

    # Build summary text
    lines = [f"Keyword pre-screen for {grid}: {report.aggregated_decision}"]
    for i, sig in enumerate(report.per_note):
        active = [k for k in [
            "outside_biopsy", "marsh_positive", "marsh_indeterminate",
            "iel_positive", "iel_negative", "villous_abnormal", "villous_normal",
        ] if getattr(sig, k)]
        if active:
            lines.append(f"  Note {i}: {', '.join(active)} → {sig.decision}")
    report.summary_text = "\n".join(lines)

    return report
