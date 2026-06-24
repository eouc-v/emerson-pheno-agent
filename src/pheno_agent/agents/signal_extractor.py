"""
signal_extractor.py — Signal Extractor Agent.

Uses an LLM to extract structured pathological signals from each medical
note.  The clinician's diagnosis logic (phrase dictionaries, Marsh grading,
external confirmation rules) is injected into every prompt so the LLM
knows exactly what to look for.

Supports a **reflection mode**: when the Critic returns feedback, the
Extractor re-processes only the flagged notes with the feedback included.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pheno_agent.config import cfg
from pheno_agent.llm import OllamaHandler, parse_json_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NoteSignals:
    """Extracted celiac signals for a single note."""
    note_label: str = ""
    note_date: str = ""
    note_type: str = ""
    iel_status: str = "not_found"        # positive / negative / not_found
    villous_architecture: str = "not_found"  # abnormal / normal / not_found
    marsh_grade: str = "not_found"       # positive / indeterminate / not_found
    external_confirmation: bool = False
    past_celiac_diagnosis: bool = False
    supporting_quotes: List[str] = field(default_factory=list)


@dataclass
class CriticFeedback:
    """Feedback from the Critic agent for re-extraction."""
    note_label: str
    issue_type: str  # phantom_quote / signal_mismatch / negation_error / false_diagnosis
    description: str


# ---------------------------------------------------------------------------
# Diagnosis logic loader
# ---------------------------------------------------------------------------

_diagnosis_logic: Optional[str] = None


def _load_diagnosis_logic(path: Optional[Path] = None) -> str:
    """Load the clinician's diagnosis logic markdown (cached)."""
    global _diagnosis_logic
    if _diagnosis_logic is not None:
        return _diagnosis_logic

    if path is None:
        path = cfg.diagnosis_logic_path
    with open(path, "r") as f:
        _diagnosis_logic = f.read()
    return _diagnosis_logic


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_extraction_prompt(
    notes_batch: list,
    keyword_hints: str = "",
    critic_feedback: Optional[List[CriticFeedback]] = None,
) -> str:
    """
    Build the signal extraction prompt.

    Includes the full clinician's diagnosis logic as a reference guide.

    Parameters
    ----------
    notes_batch : list
        List of dicts with keys: label, date, source, text.
    keyword_hints : str
        Optional keyword pre-screen summary to guide attention.
    critic_feedback : list[CriticFeedback], optional
        If provided, adds re-extraction instructions for flagged notes.
    """
    diagnosis_logic = _load_diagnosis_logic()

    # Format notes
    notes_parts = []
    for note in notes_batch:
        notes_parts.append(
            f"### {note['label']} (Date: {note['date']}, Type: {note['source']})\n"
            f"{note['text']}"
        )
    notes_text = "\n\n---\n\n".join(notes_parts) if notes_parts else "(No notes provided)"

    # Keyword hints section
    kw_section = ""
    if keyword_hints:
        kw_section = f"""
## Keyword Pre-Screen Hints
The following keyword-based signals were detected automatically. Use these as
hints to guide your attention, but verify each one against the actual note text:

{keyword_hints}
"""

    # Critic feedback section (for reflection loop)
    feedback_section = ""
    if critic_feedback:
        fb_lines = []
        for fb in critic_feedback:
            fb_lines.append(
                f"- **{fb.note_label}**: [{fb.issue_type}] {fb.description}"
            )
        feedback_section = f"""
## ⚠ Critic Feedback — Please Re-examine
The following issues were found in your previous extraction. Please carefully
re-examine the flagged notes and correct any errors:

{chr(10).join(fb_lines)}
"""

    prompt = f"""You are a clinical data extraction assistant specialising in celiac disease pathology.

## Reference: Clinician's Diagnosis Logic

{diagnosis_logic}

{kw_section}
{feedback_section}
## Patient Medical Notes

{notes_text}

## Task

For EACH note above, extract the pathological signals. Do NOT determine a diagnosis — only extract signals.

Respond ONLY with valid JSON:
{{
  "notes": [
    {{
      "note_label": "Note_1",
      "iel_status": "positive" | "negative" | "not_found",
      "villous_architecture": "abnormal" | "normal" | "not_found",
      "marsh_grade": "positive" | "indeterminate" | "not_found",
      "external_confirmation": true | false,
      "past_celiac_diagnosis": true | false,
      "supporting_quotes": ["<exact phrase from note>"]
    }}
  ]
}}

Rules:
- Report signals for EACH note separately.
- Use "not_found" if a signal category is absent from the note.
- Prefer the final diagnostic impression over preliminary descriptions.
- If both positive and negative IEL phrases appear, prefer the statement from the final diagnosis section.
- A clearly negated phrase (e.g. "no increased intraepithelial lymphocytes") means NEGATIVE.
- Set past_celiac_diagnosis to true if the note mentions a prior or established celiac diagnosis (problem list, PMH, assessment, etc.).
- Do NOT set past_celiac_diagnosis to true for "rule out celiac" or "family history of celiac" — those are NOT confirmed diagnoses.
- For supporting_quotes, copy EXACT phrases from the note text. Do not paraphrase.
- Do not include text outside the JSON."""

    return prompt


# ---------------------------------------------------------------------------
# Signal Extractor Agent
# ---------------------------------------------------------------------------

class SignalExtractor:
    """
    LLM-based agent that extracts structured celiac signals from notes.

    Uses the clinician's phrase dictionaries and diagnosis logic as
    in-context reference to guide extraction.
    """

    def __init__(self, llm: OllamaHandler):
        self.llm = llm
        self.model = cfg.models.extraction_model

    def extract(
        self,
        notes: list,
        keyword_hints: str = "",
        critic_feedback: Optional[List[CriticFeedback]] = None,
    ) -> List[NoteSignals]:
        """
        Extract signals from a list of notes.

        Parameters
        ----------
        notes : list
            List of NoteEntry objects (from DataGatherer).
        keyword_hints : str
            Keyword pre-screen summary text.
        critic_feedback : list[CriticFeedback], optional
            Feedback from Critic for re-extraction.

        Returns
        -------
        list[NoteSignals]
            Extracted signals for each note.
        """
        if not notes:
            return []

        # Prepare note dicts for the prompt
        note_dicts = []
        for i, note in enumerate(notes):
            note_dicts.append({
                "label": f"Note_{i + 1}",
                "date": getattr(note, "date", "unknown"),
                "source": getattr(note, "source", "unknown"),
                "text": getattr(note, "text", str(note)),
            })

        # Process in batches
        batch_size = cfg.agent.extraction_batch_size
        all_signals: List[NoteSignals] = []

        for batch_start in range(0, len(note_dicts), batch_size):
            batch = note_dicts[batch_start:batch_start + batch_size]

            # Filter critic feedback to only this batch's notes
            batch_labels = {n["label"] for n in batch}
            batch_feedback = None
            if critic_feedback:
                batch_feedback = [
                    fb for fb in critic_feedback if fb.note_label in batch_labels
                ]
                if not batch_feedback:
                    batch_feedback = None

            prompt = _build_extraction_prompt(
                batch, keyword_hints=keyword_hints, critic_feedback=batch_feedback,
            )

            system_prompt = (
                "You are a clinical data extraction assistant. "
                "Extract structured celiac disease signals from medical notes. "
                "Respond only in valid JSON format."
            )

            logger.info(
                "[SignalExtractor] Processing notes %d–%d of %d …",
                batch_start + 1, batch_start + len(batch), len(note_dicts),
            )

            raw = self.llm.get_completion(
                system_prompt, prompt, model=self.model,
            )

            signals = self._parse_signals(raw, batch)
            all_signals.extend(signals)

        return all_signals

    def _parse_signals(
        self, raw: str, note_dicts: list,
    ) -> List[NoteSignals]:
        """Parse LLM response into NoteSignals objects."""
        parsed = parse_json_response(raw)

        if parsed is None:
            logger.warning("Failed to parse extraction response. Returning empty signals.")
            return [
                NoteSignals(
                    note_label=n["label"],
                    note_date=n["date"],
                    note_type=n["source"],
                )
                for n in note_dicts
            ]

        notes_data = parsed.get("notes", [])
        if not isinstance(notes_data, list):
            notes_data = [parsed] if "iel_status" in parsed else []

        signals = []
        for i, nd in enumerate(notes_data):
            label = nd.get("note_label", f"Note_{i + 1}")
            # Match back to note_dicts for date/type
            matching_dict = next(
                (d for d in note_dicts if d["label"] == label), {}
            )

            sig = NoteSignals(
                note_label=label,
                note_date=matching_dict.get("date", nd.get("note_date", "")),
                note_type=matching_dict.get("source", nd.get("note_type", "")),
                iel_status=str(nd.get("iel_status", "not_found")).lower(),
                villous_architecture=str(nd.get("villous_architecture", "not_found")).lower(),
                marsh_grade=str(nd.get("marsh_grade", "not_found")).lower(),
                external_confirmation=bool(nd.get("external_confirmation", False)),
                past_celiac_diagnosis=bool(nd.get("past_celiac_diagnosis", False)),
                supporting_quotes=nd.get("supporting_quotes", []) or [],
            )
            signals.append(sig)

        return signals
