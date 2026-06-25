"""
critic.py — Critic / Verification Agent.

Verifies extracted signals against the raw note text, inspired by
DeepRare's Check_Agent.  Performs two stages:

1. **Quote verification** — checks that supporting quotes actually
   appear in the note (fuzzy match).
2. **Signal consistency** — uses the reasoning LLM to verify cases
   where keyword scanner and LLM extractor disagree, negation might
   be missed, or "past celiac diagnosis" might be a false positive.

The clinician's negation/uncertainty handling rules (Section 4 of
``diagnosis_logic_from_clinician.md``) are included in the LLM prompt.
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from pheno_agent.agents.signal_extractor import CriticFeedback, NoteSignals
from pheno_agent.config import cfg
from pheno_agent.llm import OllamaHandler, parse_json_response
from pheno_agent.tools.keyword_scanner import KeywordSignals

class VerificationSchema(BaseModel):
    note_label: str
    field: Literal["iel_status", "villous_architecture", "external_confirmation", "past_celiac_diagnosis"]
    correct_value: str
    reasoning: str

class VerificationResponseSchema(BaseModel):
    verifications: List[VerificationSchema]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Output of the Critic agent."""
    verified_signals: List[NoteSignals] = field(default_factory=list)
    issues: List[CriticFeedback] = field(default_factory=list)
    needs_re_extraction: bool = False


# ---------------------------------------------------------------------------
# Diagnosis logic loader
# ---------------------------------------------------------------------------

_diagnosis_logic: Optional[str] = None


def _load_diagnosis_logic(path: Optional[Path] = None) -> str:
    global _diagnosis_logic
    if _diagnosis_logic is not None:
        return _diagnosis_logic
    if path is None:
        path = cfg.diagnosis_logic_path
    with open(path, "r") as f:
        _diagnosis_logic = f.read()
    return _diagnosis_logic


# ---------------------------------------------------------------------------
# Quote verification (deterministic)
# ---------------------------------------------------------------------------

def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.75) -> bool:
    """
    Check if ``needle`` appears in ``haystack`` with fuzzy matching.

    Uses SequenceMatcher ratio on a sliding window for efficiency.
    """
    if not needle or not haystack:
        return False

    needle_lower = needle.lower().strip()
    haystack_lower = haystack.lower()

    # Exact substring check first
    if needle_lower in haystack_lower:
        return True

    # Sliding window fuzzy match for short phrases
    if len(needle_lower) > 200:
        # For very long quotes, just check a substring
        return needle_lower[:100] in haystack_lower

    window_size = len(needle_lower)
    for i in range(0, max(1, len(haystack_lower) - window_size + 1), window_size // 4 or 1):
        window = haystack_lower[i:i + window_size + 20]
        ratio = SequenceMatcher(None, needle_lower, window).ratio()
        if ratio >= threshold:
            return True

    return False


def _verify_quotes(
    signals: List[NoteSignals],
    note_texts: List[str],
) -> List[CriticFeedback]:
    """
    Verify that supporting quotes actually appear in the note text.

    Returns a list of CriticFeedback for phantom (fabricated) quotes.
    """
    issues = []
    for sig, text in zip(signals, note_texts):
        for quote in sig.supporting_quotes:
            if not _fuzzy_contains(text, quote):
                issues.append(CriticFeedback(
                    note_label=sig.note_label,
                    issue_type="phantom_quote",
                    description=(
                        f'Quote not found in note text: "{quote[:100]}…"'
                        if len(quote) > 100 else
                        f'Quote not found in note text: "{quote}"'
                    ),
                ))
    return issues


# ---------------------------------------------------------------------------
# Signal consistency check (LLM-based)
# ---------------------------------------------------------------------------

def _find_signal_mismatches(
    signals: List[NoteSignals],
    keyword_signals: List[KeywordSignals],
) -> List[dict]:
    """
    Identify cases where keyword scanner and LLM extractor disagree.

    Returns a list of mismatch descriptions for the LLM to review.
    """
    mismatches = []
    for sig, kw in zip(signals, keyword_signals):
        # IEL mismatch
        if kw.iel_positive and sig.iel_status == "not_found":
            mismatches.append({
                "note_label": sig.note_label,
                "field": "iel_status",
                "keyword_says": "positive (keyword hit)",
                "llm_says": sig.iel_status,
            })
        if kw.iel_negative and sig.iel_status == "positive":
            mismatches.append({
                "note_label": sig.note_label,
                "field": "iel_status",
                "keyword_says": "negative (negation keyword hit)",
                "llm_says": "positive",
            })

        # Villous mismatch
        if kw.villous_abnormal and sig.villous_architecture == "not_found":
            mismatches.append({
                "note_label": sig.note_label,
                "field": "villous_architecture",
                "keyword_says": "abnormal (keyword hit)",
                "llm_says": sig.villous_architecture,
            })
        if kw.villous_normal and sig.villous_architecture == "abnormal":
            mismatches.append({
                "note_label": sig.note_label,
                "field": "villous_architecture",
                "keyword_says": "normal (keyword hit)",
                "llm_says": "abnormal",
            })

        # External confirmation mismatch
        if kw.outside_biopsy and not sig.external_confirmation:
            mismatches.append({
                "note_label": sig.note_label,
                "field": "external_confirmation",
                "keyword_says": "true (biopsy confirmation keyword)",
                "llm_says": "false",
            })

    return mismatches


def _build_consistency_prompt(
    mismatches: list,
    past_dx_signals: list,
    note_texts_by_label: dict,
) -> str:
    """Build the LLM prompt for signal consistency verification."""
    diagnosis_logic = _load_diagnosis_logic()

    sections = []

    if mismatches:
        mm_lines = []
        for mm in mismatches:
            note_text = note_texts_by_label.get(mm["note_label"], "(not available)")
            # Truncate very long notes
            if len(note_text) > 3000:
                note_text = note_text[:3000] + "\n… [truncated]"
            mm_lines.append(
                f"### {mm['note_label']}: {mm['field']}\n"
                f"- Keyword scanner says: {mm['keyword_says']}\n"
                f"- LLM extractor says: {mm['llm_says']}\n"
                f"- Note text:\n{note_text}\n"
            )
        sections.append(
            "## Signal Mismatches to Verify\n" + "\n---\n".join(mm_lines)
        )

    if past_dx_signals:
        dx_lines = []
        for item in past_dx_signals:
            note_text = note_texts_by_label.get(item["note_label"], "(not available)")
            if len(note_text) > 3000:
                note_text = note_text[:3000] + "\n… [truncated]"
            dx_lines.append(
                f"### {item['note_label']}\n"
                f"- Claimed past celiac diagnosis: True\n"
                f"- Supporting quote: \"{item['quote']}\"\n"
                f"- Note text:\n{note_text}\n"
            )
        sections.append(
            "## Past Celiac Diagnosis Claims to Verify\n" + "\n---\n".join(dx_lines)
        )

    if not sections:
        return ""

    prompt = f"""You are a clinical verification specialist for celiac disease diagnosis.

## Clinician's Extraction & Tie-Breaking Rules

{diagnosis_logic}

## Your Task

Review each case below and determine if the LLM extractor's signal is correct.

{chr(10).join(sections)}

Respond ONLY with valid JSON:
{{
  "verifications": [
    {{
      "note_label": "Note_X",
      "field": "iel_status | villous_architecture | external_confirmation | past_celiac_diagnosis",
      "correct_value": "<the correct value after your review>",
      "reasoning": "<brief explanation>"
    }}
  ]
}}

Rules:
- A clearly negated phrase (e.g. "no increased intraepithelial lymphocytes") means the signal is NEGATIVE.
- "Rule out celiac" or "family history of celiac" does NOT constitute a past celiac diagnosis.
- "Celiac disease" in a problem list or PMH section DOES constitute a past celiac diagnosis.
- Prefer the final diagnostic impression over preliminary descriptions.
- If Marsh grade is present, apply it directly even if component phrases conflict.
- Do not include text outside the JSON."""

    return prompt


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------

class Critic:
    """
    Verification agent that checks extracted signals for accuracy.

    Performs quote verification (deterministic) and signal consistency
    checking (LLM-based) using the clinician's rules.
    """

    def __init__(self, llm: OllamaHandler):
        self.llm = llm
        self.model = cfg.models.reasoning_model

    def verify(
        self,
        signals: List[NoteSignals],
        note_texts: List[str],
        keyword_signals: Optional[List[KeywordSignals]] = None,
    ) -> VerificationResult:
        """
        Verify extracted signals against note text and keyword results.

        Parameters
        ----------
        signals : list[NoteSignals]
            Signals from the Signal Extractor.
        note_texts : list[str]
            Corresponding note texts (same order as signals).
        keyword_signals : list[KeywordSignals], optional
            Keyword scan results (same order).

        Returns
        -------
        VerificationResult
            Verified signals, issues found, and re-extraction flag.
        """
        result = VerificationResult()
        all_issues: List[CriticFeedback] = []

        # Stage 1: Quote verification (deterministic)
        logger.info("[Critic] Stage 1: Quote verification …")
        quote_issues = _verify_quotes(signals, note_texts)
        all_issues.extend(quote_issues)
        if quote_issues:
            logger.info("[Critic] Found %d phantom quotes.", len(quote_issues))

        # Stage 2: Signal consistency (LLM-based, only if needed)
        needs_llm_check = False
        mismatches = []
        past_dx_claims = []

        if keyword_signals:
            mismatches = _find_signal_mismatches(signals, keyword_signals)
            if mismatches:
                needs_llm_check = True
                logger.info("[Critic] Found %d keyword/LLM mismatches.", len(mismatches))

        # Check past_celiac_diagnosis claims
        for sig in signals:
            if sig.past_celiac_diagnosis:
                past_dx_claims.append({
                    "note_label": sig.note_label,
                    "quote": sig.supporting_quotes[0] if sig.supporting_quotes else "(no quote)",
                })
                needs_llm_check = True

        if needs_llm_check:
            logger.info("[Critic] Stage 2: LLM consistency check …")
            note_texts_by_label = {
                sig.note_label: text for sig, text in zip(signals, note_texts)
            }
            prompt = _build_consistency_prompt(
                mismatches, past_dx_claims, note_texts_by_label,
            )

            if prompt:
                system_prompt = (
                    "You are a clinical verification specialist. "
                    "Verify celiac disease signal extractions. "
                    "Respond only in valid JSON."
                )
                parsed_response = self.llm.get_structured(
                    system_prompt, prompt, VerificationResponseSchema, model=self.model,
                )
                self._apply_corrections(signals, parsed_response.verifications, all_issues)

        # Determine if re-extraction is needed
        # Re-extract if there are significant issues (not just phantom quotes)
        significant_issues = [
            iss for iss in all_issues
            if iss.issue_type in ("signal_mismatch", "negation_error", "false_diagnosis")
        ]
        result.needs_re_extraction = len(significant_issues) > 0
        result.verified_signals = signals
        result.issues = all_issues

        logger.info(
            "[Critic] Verification complete: %d issues, re-extraction=%s",
            len(all_issues), result.needs_re_extraction,
        )
        return result

    def _apply_corrections(
        self,
        signals: List[NoteSignals],
        corrections: List[VerificationSchema],
        issues: List[CriticFeedback],
    ):
        """Apply LLM corrections to signals and log issues."""
        sig_by_label = {s.note_label: s for s in signals}

        for corr in corrections:
            label = corr.note_label
            field_name = corr.field
            correct_value = corr.correct_value
            reasoning = corr.reasoning

            sig = sig_by_label.get(label)
            if not sig:
                continue

            current_value = getattr(sig, field_name, None)
            if current_value is None:
                continue

            # Check if correction differs from current value
            if str(current_value).lower() != str(correct_value).lower():
                logger.info(
                    "[Critic] Correcting %s.%s: %s → %s (%s)",
                    label, field_name, current_value, correct_value, reasoning,
                )
                # Apply correction
                if field_name in ("external_confirmation", "past_celiac_diagnosis"):
                    setattr(sig, field_name, str(correct_value).lower() in ("true", "1", "yes"))
                else:
                    setattr(sig, field_name, str(correct_value).lower())

                issue_type = "negation_error" if "negat" in reasoning.lower() else "signal_mismatch"
                if field_name == "past_celiac_diagnosis":
                    issue_type = "false_diagnosis"

                issues.append(CriticFeedback(
                    note_label=label,
                    issue_type=issue_type,
                    description=f"{field_name}: {current_value} → {correct_value}. {reasoning}",
                ))
