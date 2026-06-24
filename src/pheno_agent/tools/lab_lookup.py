"""
lab_lookup.py — Structured TTG-IgA lab value analysis tool.

Reuses the date-aware cutoff logic from ``celiac_identification_by_labs.py``
to compute a deterministic lab decision for each patient.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pheno_agent.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assay change date cutoffs (from celiac_identification_by_labs.py)
# ---------------------------------------------------------------------------
ASSAY_CHANGE_DATE1 = pd.to_datetime("2023-06-12").date()
ASSAY_CHANGE_DATE2 = pd.to_datetime("2024-03-01").date()

# TTG concept name filter
TTG_CONCEPT = "Tissue transglutaminase IgA Ab [Units/volume] in Serum by Immunoassay"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TTGLabValue:
    """A single TTG-IgA lab measurement."""
    date: str
    value_raw: str
    value_numeric: Optional[float]
    unit: str
    decision: str  # case / secondary_check / excluded / unknown


@dataclass
class LabSummary:
    """Aggregated TTG-IgA lab summary for one patient."""
    grid: str
    ttg_values: List[TTGLabValue] = field(default_factory=list)
    max_ttg: Optional[float] = None
    lab_decision: str = "excluded"  # aggregated: case > secondary_check > excluded
    summary_text: str = ""


# ---------------------------------------------------------------------------
# Decision logic (ported from celiac_identification_by_labs.py)
# ---------------------------------------------------------------------------

def _phenotype_single_lab(date, value_source: str) -> str:
    """
    Determine lab phenotype for a single TTG-IgA measurement.

    Uses date-aware assay cutoffs.
    """
    source = value_source.strip().lower()

    # Determine cutoffs based on assay date
    if ASSAY_CHANGE_DATE1 <= date < ASSAY_CHANGE_DATE2:
        case_cutoff, positive_cutoff = 40, 4
    elif date >= ASSAY_CHANGE_DATE2:
        case_cutoff, positive_cutoff = 100, 15
    else:
        case_cutoff, positive_cutoff = 100, 20

    # Try numeric interpretation first
    try:
        num = float(source)
        if num > case_cutoff:
            return "case"
        elif num > positive_cutoff:
            return "secondary_check"
        else:
            return "excluded"
    except ValueError:
        pass

    # String-based fallbacks
    if source.startswith(">") and "100" in source:
        return "case"
    if "positive" in source or "weak" in source:
        return "secondary_check"
    if source.startswith("<") or source == ">1.23" or "negative" in source:
        return "excluded"
    return "unknown"


def _combine_decisions(decisions: List[str]) -> str:
    """Aggregate multiple per-measurement decisions: case > secondary_check > excluded."""
    if "case" in decisions:
        return "case"
    if "secondary_check" in decisions:
        return "secondary_check"
    return "excluded"


# ---------------------------------------------------------------------------
# Lab loader (lazy singleton)
# ---------------------------------------------------------------------------

_labs_df: Optional[pd.DataFrame] = None


def _load_labs(lab_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and preprocess the TTG labs CSV (cached after first call)."""
    global _labs_df
    if _labs_df is not None:
        return _labs_df

    lab_path = lab_path or cfg.lab_csv_path
    logger.info("Loading TTG-IgA labs from %s …", lab_path)
    df = pd.read_csv(lab_path, low_memory=False)

    # Parse dates
    df["measurement_datetime"] = pd.to_datetime(
        df["measurement_datetime"], format="mixed", errors="coerce", utc=True
    )
    df["date"] = df["measurement_datetime"].dt.date

    # Filter to TTG-IgA only
    df = df[df["concept_name"] == TTG_CONCEPT].copy()

    # Normalise patient ID column
    if "person_source_value" in df.columns:
        df = df.rename(columns={"person_source_value": "grid"})

    logger.info("Loaded %d TTG-IgA lab records for %d patients.", len(df), df["grid"].nunique())
    _labs_df = df
    return _labs_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_ttg_labs(grid: str, lab_path: Optional[Path] = None) -> LabSummary:
    """
    Look up all TTG-IgA labs for a patient and compute the lab decision.

    Parameters
    ----------
    grid : str
        Patient identifier.
    lab_path : Path, optional
        Override the default lab CSV path.

    Returns
    -------
    LabSummary
        Aggregated lab data and decision.
    """
    df = _load_labs(lab_path)
    patient_labs = df[df["grid"] == grid]

    summary = LabSummary(grid=grid)

    if patient_labs.empty:
        summary.summary_text = "No TTG-IgA lab data available."
        return summary

    decisions = []
    numeric_values = []
    text_parts = []

    for _, row in patient_labs.iterrows():
        raw_val = str(row.get("value_source_value", "")).strip()
        num_val = row.get("value_as_number", None)
        try:
            num_val = float(num_val) if pd.notna(num_val) else None
        except (ValueError, TypeError):
            num_val = None

        unit = str(row.get("unit_source_value", "")) if pd.notna(row.get("unit_source_value")) else ""
        date = row["date"]
        decision = _phenotype_single_lab(date, raw_val)
        decisions.append(decision)

        if num_val is not None:
            numeric_values.append(num_val)

        lab_val = TTGLabValue(
            date=str(date),
            value_raw=raw_val,
            value_numeric=num_val,
            unit=unit,
            decision=decision,
        )
        summary.ttg_values.append(lab_val)
        text_parts.append(f"  [{date}] TTG-IgA: {raw_val} {unit} → {decision}")

    summary.max_ttg = max(numeric_values) if numeric_values else None
    summary.lab_decision = _combine_decisions(decisions)

    header = f"TTG-IgA labs for {grid}: {len(summary.ttg_values)} measurements"
    header += f", max={summary.max_ttg}" if summary.max_ttg is not None else ""
    header += f", overall_decision={summary.lab_decision}"
    summary.summary_text = header + "\n" + "\n".join(text_parts)

    return summary
