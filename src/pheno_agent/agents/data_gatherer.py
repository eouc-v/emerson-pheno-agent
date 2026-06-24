"""
data_gatherer.py — Data Gatherer Agent.

Compiles a complete patient dossier from all available data sources:
  - EHR markdown files (notes + labs)
  - Structured TTG-IgA lab data
  - Keyword pre-screen results
  - (Optional) ChromaDB relevant chunks

This agent is mostly deterministic — no LLM calls.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from pheno_agent.tools.ehr_reader import (
    NoteEntry,
    ParsedEHR,
    parse_ehr_sections,
    read_patient_ehr,
)
from pheno_agent.tools.keyword_scanner import (
    PatientKeywordReport,
    scan_patient_notes,
)
from pheno_agent.tools.lab_lookup import LabSummary, lookup_ttg_labs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class: the complete patient dossier
# ---------------------------------------------------------------------------

@dataclass
class PatientDossier:
    """Everything an agent needs to know about a patient."""

    grid: str

    # Parsed EHR
    parsed_ehr: Optional[ParsedEHR] = None

    # Lab analysis
    lab_summary: Optional[LabSummary] = None

    # Keyword pre-screen
    keyword_report: Optional[PatientKeywordReport] = None

    # Notes selected for LLM processing (filtered/ranked subset)
    relevant_notes: List[NoteEntry] = field(default_factory=list)

    # Full raw markdown (for critic to cross-reference quotes)
    full_ehr_markdown: str = ""

    def summary(self) -> str:
        """Short human-readable summary of the dossier."""
        parts = [f"PatientDossier for {self.grid}"]
        if self.parsed_ehr:
            parts.append(f"  Labs: {len(self.parsed_ehr.labs)} entries")
            parts.append(f"  Notes: {len(self.parsed_ehr.notes)} notes")
        if self.lab_summary:
            parts.append(f"  TTG decision: {self.lab_summary.lab_decision}")
        if self.keyword_report:
            parts.append(f"  Keyword decision: {self.keyword_report.aggregated_decision}")
        parts.append(f"  Relevant notes for LLM: {len(self.relevant_notes)}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data Gatherer Agent
# ---------------------------------------------------------------------------

class DataGatherer:
    """
    Tool-using agent that compiles a complete patient dossier.

    Does not use any LLM calls — purely deterministic data aggregation.
    """

    def __init__(self, use_chroma: bool = True):
        self.use_chroma = use_chroma

    def gather(self, grid: str) -> PatientDossier:
        """
        Gather all available data for a patient.

        Parameters
        ----------
        grid : str
            Patient identifier.

        Returns
        -------
        PatientDossier
            Complete dossier ready for downstream agents.
        """
        dossier = PatientDossier(grid=grid)

        # Step 1: Read EHR markdown
        logger.info("[DataGatherer] Reading EHR for %s …", grid)
        raw_md = read_patient_ehr(grid)
        if raw_md is None:
            logger.warning("[DataGatherer] No EHR file found for %s.", grid)
            return dossier

        dossier.full_ehr_markdown = raw_md
        dossier.parsed_ehr = parse_ehr_sections(raw_md)

        # Step 2: Look up TTG-IgA labs
        logger.info("[DataGatherer] Looking up TTG labs for %s …", grid)
        dossier.lab_summary = lookup_ttg_labs(grid)

        # Step 3: Keyword scan on each note
        logger.info("[DataGatherer] Running keyword scan for %s …", grid)
        note_texts = [n.text for n in dossier.parsed_ehr.notes]
        dossier.keyword_report = scan_patient_notes(note_texts, grid=grid)

        # Step 4: Select relevant notes for LLM processing
        # Strategy: include ALL notes that have any keyword signal,
        # plus any notes with celiac-related content.
        # If no notes have signals, include all notes (let LLM decide).
        dossier.relevant_notes = self._select_relevant_notes(dossier)

        logger.info("[DataGatherer] Dossier complete:\n%s", dossier.summary())
        return dossier

    def _select_relevant_notes(self, dossier: PatientDossier) -> List[NoteEntry]:
        """
        Select which notes should be sent to the Signal Extractor.

        Priority:
        1. Notes with keyword hits (Positive or Indeterminate signals)
        2. Notes containing any celiac-related popular keywords
        3. If nothing found, include all notes
        """
        if not dossier.parsed_ehr or not dossier.keyword_report:
            return []

        notes = dossier.parsed_ehr.notes
        kw_signals = dossier.keyword_report.per_note

        # Notes with actual keyword signals
        signalled = []
        non_signalled = []
        for i, (note, sig) in enumerate(zip(notes, kw_signals)):
            has_signal = any([
                sig.outside_biopsy, sig.marsh_positive, sig.marsh_indeterminate,
                sig.iel_positive, sig.iel_negative,
                sig.villous_abnormal, sig.villous_normal,
            ])
            if has_signal:
                signalled.append(note)
            else:
                non_signalled.append(note)

        if signalled:
            # Also include notes that mention celiac/sprue/gluten in any form
            celiac_keywords = ["celiac", "coeliac", "sprue", "gluten", "ttg", "marsh"]
            extra = [
                n for n in non_signalled
                if any(kw in n.text.lower() for kw in celiac_keywords)
            ]
            selected = signalled + extra
            logger.debug(
                "Selected %d signalled + %d celiac-mentioned notes (of %d total).",
                len(signalled), len(extra), len(notes),
            )
            return selected

        # No keyword signals at all — send all notes
        logger.debug("No keyword signals found. Including all %d notes.", len(notes))
        return list(notes)
