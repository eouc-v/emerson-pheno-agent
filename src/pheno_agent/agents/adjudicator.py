"""
adjudicator.py — Clinical Adjudicator Agent.

Applies the clinician's decision table deterministically, then uses an
LLM to generate a human-readable reasoning chain citing the specific
evidence, note IDs, and decision rules.

The decision table is a pure Python implementation of Section 3 from
``diagnosis_logic_from_clinician.md``.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pheno_agent.agents.signal_extractor import NoteSignals
from pheno_agent.config import cfg
from pheno_agent.llm import OllamaHandler
from pheno_agent.tools.lab_lookup import LabSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NoteDecision:
    """Decision for a single note."""
    note_label: str
    note_date: str
    decision: str  # Positive / Negative / Indeterminate
    rule: str      # Which rule triggered this decision


@dataclass
class FinalDiagnosis:
    """Complete diagnosis output for a patient."""
    grid: str
    diagnosis: str = "Negative"  # Positive / Negative / Indeterminate
    confidence: float = 0.0
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)
    decision_path: str = ""
    note_decisions: List[NoteDecision] = field(default_factory=list)
    lab_decision: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a flat dict for CSV output."""
        return {
            "grid": self.grid,
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": "; ".join(self.evidence[:10]),
            "decision_path": self.decision_path,
            "lab_decision": self.lab_decision,
        }


# ---------------------------------------------------------------------------
# Decision table (Section 3 of diagnosis_logic_from_clinician.md)
# ---------------------------------------------------------------------------

def apply_decision_table(signals: NoteSignals) -> NoteDecision:
    """
    Apply the clinician's decision table to a single note's signals.

    Decision priority:
      1. Past celiac diagnosis → Positive (celiac cannot be cured)
      2. External biopsy confirmation → Positive
      3. Marsh 3/4 → Positive
      4. Marsh 1/2 → Indeterminate
      5. IEL positive + Villous abnormal → Positive
      6. IEL negative + Villous normal → Negative
      7. IEL positive + Villous normal → Indeterminate
      8. IEL negative + Villous abnormal → Indeterminate
      9. No signal found → Negative

    Parameters
    ----------
    signals : NoteSignals
        Extracted and verified signals for one note.

    Returns
    -------
    NoteDecision
        The decision and which rule triggered it.
    """
    iel = signals.iel_status
    villous = signals.villous_architecture
    marsh = signals.marsh_grade
    external = signals.external_confirmation
    past_dx = signals.past_celiac_diagnosis

    if past_dx:
        return NoteDecision(
            note_label=signals.note_label,
            note_date=signals.note_date,
            decision="Positive",
            rule="Past celiac diagnosis (celiac cannot be cured)",
        )
    if external:
        return NoteDecision(
            note_label=signals.note_label,
            note_date=signals.note_date,
            decision="Positive",
            rule="External biopsy confirmed",
        )
    if marsh == "positive":
        return NoteDecision(
            note_label=signals.note_label,
            note_date=signals.note_date,
            decision="Positive",
            rule="Marsh grade 3/4 (positive)",
        )
    if marsh == "indeterminate":
        return NoteDecision(
            note_label=signals.note_label,
            note_date=signals.note_date,
            decision="Indeterminate",
            rule="Marsh grade 1/2 (indeterminate)",
        )
    if iel == "positive" and villous == "abnormal":
        return NoteDecision(
            note_label=signals.note_label,
            note_date=signals.note_date,
            decision="Positive",
            rule="IEL positive + Villous abnormal",
        )
    if iel == "negative" and villous == "normal":
        return NoteDecision(
            note_label=signals.note_label,
            note_date=signals.note_date,
            decision="Negative",
            rule="IEL negative + Villous normal",
        )
    if iel == "positive" and villous == "normal":
        return NoteDecision(
            note_label=signals.note_label,
            note_date=signals.note_date,
            decision="Indeterminate",
            rule="IEL positive + Villous normal",
        )
    if iel == "negative" and villous == "abnormal":
        return NoteDecision(
            note_label=signals.note_label,
            note_date=signals.note_date,
            decision="Indeterminate",
            rule="IEL negative + Villous abnormal",
        )
    # No signal found
    return NoteDecision(
        note_label=signals.note_label,
        note_date=signals.note_date,
        decision="Negative",
        rule="No pathological signal found",
    )


def aggregate_decisions(decisions: List[str]) -> str:
    """Aggregate note-level decisions: Positive > Indeterminate > Negative."""
    if "Positive" in decisions:
        return "Positive"
    if "Indeterminate" in decisions:
        return "Indeterminate"
    return "Negative"


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
# Adjudicator Agent
# ---------------------------------------------------------------------------

class Adjudicator:
    """
    Clinical Adjudicator agent.

    Applies the clinician's decision table deterministically, then
    generates a human-readable reasoning chain using the LLM.
    """

    def __init__(self, llm: OllamaHandler):
        self.llm = llm
        self.model = cfg.models.reasoning_model

    def adjudicate(
        self,
        grid: str,
        verified_signals: List[NoteSignals],
        lab_summary: Optional[LabSummary] = None,
        keyword_decision: str = "Negative",
    ) -> FinalDiagnosis:
        """
        Produce the final diagnosis for a patient.

        Parameters
        ----------
        grid : str
            Patient identifier.
        verified_signals : list[NoteSignals]
            Critic-verified signals.
        lab_summary : LabSummary, optional
            TTG-IgA lab data.
        keyword_decision : str
            Aggregated keyword pre-screen decision.

        Returns
        -------
        FinalDiagnosis
            Complete diagnosis with reasoning chain.
        """
        result = FinalDiagnosis(grid=grid)
        lab_decision = lab_summary.lab_decision if lab_summary else "excluded"
        result.lab_decision = lab_decision

        # Step 1: TTG override
        if lab_decision == "case":
            result.diagnosis = "Positive"
            result.confidence = 1.0
            result.decision_path = "TTG-IgA lab override (TTG > cutoff)"
            result.reasoning = (
                f"TTG-IgA lab decision is 'case' (TTG > cutoff). "
                f"Per clinician's rules: 'If TTG > 100 or > 10x upper limit of normal, "
                f"then positive regardless of other labs or notes.' "
                f"Auto-assigned Positive."
            )
            if lab_summary:
                result.evidence = [lab_summary.summary_text]
            logger.info("[Adjudicator] %s → Positive (TTG override)", grid)
            return result

        # Step 2: Apply decision table per note
        note_decisions = []
        all_quotes = []
        for sig in verified_signals:
            nd = apply_decision_table(sig)
            note_decisions.append(nd)
            all_quotes.extend(sig.supporting_quotes)

        result.note_decisions = note_decisions

        # Step 3: Aggregate across notes
        note_dx = [nd.decision for nd in note_decisions]
        llm_agg = aggregate_decisions(note_dx)

        # Use LLM decision as final. Do NOT let keyword scanning override the LLM's deep reading.
        final = llm_agg
        result.diagnosis = final
        result.evidence = all_quotes[:10]

        # Confidence based on agreement
        if llm_agg == keyword_decision:
            result.confidence = 0.95
        elif final == "Positive":
            result.confidence = 0.85
        else:
            result.confidence = 0.65

        # Build decision path
        note_path_parts = [
            f"{nd.note_label}({nd.note_date}): {nd.decision} [{nd.rule}]"
            for nd in note_decisions
        ]
        result.decision_path = (
            f"Per-note: {'; '.join(note_path_parts)}. "
            f"LLM aggregated: {llm_agg}. Keyword: {keyword_decision}. "
            f"Final: {final}."
        )

        # Step 4: Generate reasoning chain (LLM)
        logger.info("[Adjudicator] Generating reasoning for %s …", grid)
        result.reasoning = self._generate_reasoning(result, lab_summary)

        logger.info("[Adjudicator] %s → %s (confidence=%.2f)", grid, final, result.confidence)
        return result

    def _generate_reasoning(
        self,
        result: FinalDiagnosis,
        lab_summary: Optional[LabSummary],
    ) -> str:
        """Use LLM to generate a human-readable reasoning chain."""
        diagnosis_logic = _load_diagnosis_logic()

        lab_text = lab_summary.summary_text if lab_summary else "No lab data."
        evidence_text = "\n".join(f"- {q}" for q in result.evidence) or "No supporting quotes."
        decisions_text = "\n".join(
            f"- {nd.note_label} ({nd.note_date}): {nd.decision} — Rule: {nd.rule}"
            for nd in result.note_decisions
        ) or "No per-note decisions."

        prompt = f"""Based on the following clinical analysis, write a clear, concise reasoning
chain explaining why this patient received a **{result.diagnosis}** celiac disease diagnosis.

## Clinician's Decision Rules
{diagnosis_logic}

## Lab Data
{lab_text}

## Per-Note Decisions (from decision table)
{decisions_text}

## Supporting Evidence (quotes from notes)
{evidence_text}

## Decision Path
{result.decision_path}

## Instructions
- Write 3–5 sentences explaining the diagnosis step by step.
- Reference specific evidence (note dates, quotes, lab values).
- Reference which decision rule was triggered.
- If the diagnosis is Negative, explain what was absent.
- Be factual and concise. Do not speculate beyond the evidence."""

        system_prompt = (
            "You are a clinical reasoning assistant. Write a clear, evidence-based "
            "reasoning chain for a celiac disease diagnosis decision."
        )

        raw = self.llm.get_completion(system_prompt, prompt, model=self.model)
        return raw.strip() if raw else result.decision_path
