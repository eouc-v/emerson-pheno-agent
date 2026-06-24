"""
orchestrator.py — Central host that coordinates the agent workflow.

For each patient:
  1. DataGatherer → compile dossier
  2. Check TTG override → short-circuit if lab_decision == "case"
  3. SignalExtractor → extract per-note signals
  4. Critic → verify signals (reflection loop up to max_reflection_loops)
  5. Adjudicator → apply decision table + generate reasoning

Inspired by DeepRare's ``diagnosis.py::make_diagnosis`` workflow.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pheno_agent.agents.adjudicator import Adjudicator, FinalDiagnosis
from pheno_agent.agents.critic import Critic
from pheno_agent.agents.data_gatherer import DataGatherer, PatientDossier
from pheno_agent.agents.signal_extractor import NoteSignals, SignalExtractor
from pheno_agent.config import cfg
from pheno_agent.llm import OllamaHandler

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central host that coordinates the multi-agent diagnostic workflow.

    Manages the DataGatherer → SignalExtractor → Critic → Adjudicator
    pipeline with a reflection loop between Extractor and Critic.
    """

    def __init__(
        self,
        reasoning_model: Optional[str] = None,
        extraction_model: Optional[str] = None,
        fast_model: Optional[str] = None,
        max_reflections: Optional[int] = None,
        use_chroma: bool = True,
    ):
        # Override models if specified
        if reasoning_model:
            cfg.models.reasoning_model = reasoning_model
        if extraction_model:
            cfg.models.extraction_model = extraction_model
        if fast_model:
            cfg.models.fast_model = fast_model

        self.max_reflections = max_reflections or cfg.agent.max_reflection_loops
        self.use_chroma = use_chroma

        # Initialise LLM handlers
        self.reasoning_llm = OllamaHandler(default_model=cfg.models.reasoning_model)
        self.extraction_llm = OllamaHandler(default_model=cfg.models.extraction_model)

        # Initialise agents
        self.data_gatherer = DataGatherer(use_chroma=use_chroma)
        self.signal_extractor = SignalExtractor(llm=self.extraction_llm)
        self.critic = Critic(llm=self.reasoning_llm)
        self.adjudicator = Adjudicator(llm=self.reasoning_llm)

        logger.info("Orchestrator initialised.")
        logger.info("  Reasoning model: %s", cfg.models.reasoning_model)
        logger.info("  Extraction model: %s", cfg.models.extraction_model)
        logger.info("  Fast model: %s", cfg.models.fast_model)
        logger.info("  Max reflections: %d", self.max_reflections)

    def diagnose_patient(self, grid: str) -> Dict[str, Any]:
        """
        Run the full diagnostic workflow for a single patient.

        Parameters
        ----------
        grid : str
            Patient identifier.

        Returns
        -------
        dict
            Complete result including diagnosis, reasoning, evidence,
            and full agent trace.
        """
        t0 = time.time()
        trace: Dict[str, Any] = {"grid": grid, "steps": []}

        # ---- Step 1: Data Gatherer ----------------------------------------
        logger.info("=" * 60)
        logger.info("PATIENT %s — Step 1: Data Gathering", grid)
        logger.info("=" * 60)

        dossier = self.data_gatherer.gather(grid)
        trace["steps"].append({
            "agent": "DataGatherer",
            "summary": dossier.summary(),
        })

        if not dossier.parsed_ehr or not dossier.parsed_ehr.notes:
            logger.warning("No EHR data for %s. Assigning Negative.", grid)
            return self._build_result(
                FinalDiagnosis(
                    grid=grid,
                    diagnosis="Negative",
                    confidence=0.0,
                    reasoning="No EHR data available.",
                ),
                trace, t0,
            )

        # ---- Step 2: TTG Override Check -----------------------------------
        lab_decision = dossier.lab_summary.lab_decision if dossier.lab_summary else "excluded"

        if lab_decision == "case":
            logger.info("PATIENT %s — TTG override: lab_decision='case'. → Positive", grid)
            diagnosis = FinalDiagnosis(
                grid=grid,
                diagnosis="Positive",
                confidence=1.0,
                decision_path="TTG-IgA lab override (TTG > cutoff)",
                reasoning=(
                    f"TTG-IgA lab decision is 'case' (TTG > cutoff). "
                    f"Per clinician's rules: 'If TTG > 100 or > 10x upper limit of normal, "
                    f"then positive regardless of other labs or notes.'"
                ),
                lab_decision=lab_decision,
            )
            if dossier.lab_summary:
                diagnosis.evidence = [dossier.lab_summary.summary_text]
            trace["steps"].append({
                "agent": "TTG_Override",
                "lab_decision": lab_decision,
                "result": "Positive",
            })
            return self._build_result(diagnosis, trace, t0)

        # ---- Step 3: Signal Extraction ------------------------------------
        logger.info("PATIENT %s — Step 3: Signal Extraction", grid)
        keyword_hints = dossier.keyword_report.summary_text if dossier.keyword_report else ""

        signals = self.signal_extractor.extract(
            notes=dossier.relevant_notes,
            keyword_hints=keyword_hints,
        )

        trace["steps"].append({
            "agent": "SignalExtractor",
            "num_notes": len(dossier.relevant_notes),
            "num_signals": len(signals),
            "signals": [self._signal_to_dict(s) for s in signals],
        })

        # ---- Step 4: Critic Verification (with reflection loop) -----------
        note_texts = [n.text for n in dossier.relevant_notes]
        kw_signals = None
        if dossier.keyword_report and dossier.keyword_report.per_note:
            # Map keyword signals to match relevant_notes order
            kw_signals = self._align_keyword_signals(dossier)

        for reflection in range(self.max_reflections + 1):
            logger.info(
                "PATIENT %s — Step 4: Critic Verification (round %d/%d)",
                grid, reflection + 1, self.max_reflections + 1,
            )

            verification = self.critic.verify(
                signals=signals,
                note_texts=note_texts,
                keyword_signals=kw_signals,
            )

            trace["steps"].append({
                "agent": "Critic",
                "round": reflection + 1,
                "num_issues": len(verification.issues),
                "needs_re_extraction": verification.needs_re_extraction,
                "issues": [
                    {"note": i.note_label, "type": i.issue_type, "desc": i.description}
                    for i in verification.issues
                ],
            })

            if not verification.needs_re_extraction or reflection >= self.max_reflections:
                signals = verification.verified_signals
                break

            # Re-extract with critic feedback
            logger.info(
                "PATIENT %s — Re-extraction requested (%d issues)",
                grid, len(verification.issues),
            )
            signals = self.signal_extractor.extract(
                notes=dossier.relevant_notes,
                keyword_hints=keyword_hints,
                critic_feedback=verification.issues,
            )
            trace["steps"].append({
                "agent": "SignalExtractor",
                "mode": "re-extraction",
                "round": reflection + 1,
                "signals": [self._signal_to_dict(s) for s in signals],
            })

        # ---- Step 5: Adjudicator ------------------------------------------
        logger.info("PATIENT %s — Step 5: Adjudication", grid)
        keyword_decision = dossier.keyword_report.aggregated_decision if dossier.keyword_report else "Negative"

        diagnosis = self.adjudicator.adjudicate(
            grid=grid,
            verified_signals=signals,
            lab_summary=dossier.lab_summary,
            keyword_decision=keyword_decision,
        )

        trace["steps"].append({
            "agent": "Adjudicator",
            "diagnosis": diagnosis.diagnosis,
            "confidence": diagnosis.confidence,
            "decision_path": diagnosis.decision_path,
            "note_decisions": [
                {"note": nd.note_label, "date": nd.note_date, "decision": nd.decision, "rule": nd.rule}
                for nd in diagnosis.note_decisions
            ],
        })

        return self._build_result(diagnosis, trace, t0)

    def _align_keyword_signals(self, dossier: PatientDossier):
        """
        Align keyword signals to match the order of relevant_notes.

        The keyword_report has signals for ALL notes; we need signals
        for only the relevant_notes subset.
        """
        if not dossier.parsed_ehr or not dossier.keyword_report:
            return None

        all_notes = dossier.parsed_ehr.notes
        all_kw = dossier.keyword_report.per_note
        relevant_set = set(id(n) for n in dossier.relevant_notes)

        aligned = []
        for note, kw in zip(all_notes, all_kw):
            if id(note) in relevant_set:
                aligned.append(kw)

        # Fallback: if alignment fails, return None
        if len(aligned) != len(dossier.relevant_notes):
            logger.warning(
                "Keyword signal alignment mismatch (%d vs %d). Skipping keyword comparison.",
                len(aligned), len(dossier.relevant_notes),
            )
            return None

        return aligned

    def _signal_to_dict(self, sig: NoteSignals) -> dict:
        """Serialise a NoteSignals to a JSON-safe dict."""
        return {
            "note_label": sig.note_label,
            "note_date": sig.note_date,
            "note_type": sig.note_type,
            "iel_status": sig.iel_status,
            "villous_architecture": sig.villous_architecture,
            "marsh_grade": sig.marsh_grade,
            "external_confirmation": sig.external_confirmation,
            "past_celiac_diagnosis": sig.past_celiac_diagnosis,
            "supporting_quotes": sig.supporting_quotes,
        }

    def _build_result(
        self,
        diagnosis: FinalDiagnosis,
        trace: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        """Build the final result dict."""
        elapsed = time.time() - start_time
        return {
            "grid": diagnosis.grid,
            "diagnosis": diagnosis.diagnosis,
            "confidence": diagnosis.confidence,
            "reasoning": diagnosis.reasoning,
            "evidence": diagnosis.evidence,
            "decision_path": diagnosis.decision_path,
            "lab_decision": diagnosis.lab_decision,
            "time_taken_s": round(elapsed, 1),
            "trace": trace,
            "llm_stats": {
                "reasoning": self.reasoning_llm.stats(),
                "extraction": self.extraction_llm.stats(),
            },
        }

    def save_result(self, result: Dict[str, Any], output_dir: Optional[Path] = None):
        """Save per-patient result as JSON."""
        output_dir = output_dir or cfg.agent_results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{result['grid']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Result saved to %s", path)
