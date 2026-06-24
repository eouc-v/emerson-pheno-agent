"""
ehr_reader.py — Read and parse patient EHR markdown files.

Each file in ``data/ehr_markdown_dataset/`` follows the format produced by
``curate_dataset.py``:

    # Grid: R201643869

    ## Labs
    - [2011-04-20 15:38:00] Tissue transglutaminase IgA …: 9 Units

    ## Medical Notes
    ### [2007-02-28 18:10:00] Clinic Visit
    **Source:** CLINIC NOTE

    <note text>
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pheno_agent.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LabEntry:
    """A single parsed lab entry from the markdown."""
    date: str
    concept: str
    value: str
    unit: str = ""
    ref_range: str = ""


@dataclass
class NoteEntry:
    """A single parsed medical note from the markdown."""
    date: str
    title: str
    source: str
    text: str


@dataclass
class ParsedEHR:
    """Complete parsed EHR for one patient."""
    grid: str
    labs: List[LabEntry] = field(default_factory=list)
    notes: List[NoteEntry] = field(default_factory=list)
    raw_markdown: str = ""


# ---------------------------------------------------------------------------
# Reader functions
# ---------------------------------------------------------------------------

def read_patient_ehr(grid: str, ehr_dir: Optional[Path] = None) -> Optional[str]:
    """
    Read the full markdown content for a patient.

    Returns None if the file does not exist.
    """
    ehr_dir = ehr_dir or cfg.ehr_markdown_dir
    path = ehr_dir / f"{grid}.md"
    if not path.exists():
        logger.warning("EHR file not found: %s", path)
        return None
    return path.read_text(encoding="utf-8")


def parse_ehr_sections(markdown: str) -> ParsedEHR:
    """
    Parse an EHR markdown string into structured sections.

    Parameters
    ----------
    markdown : str
        Full content of a patient's EHR markdown file.

    Returns
    -------
    ParsedEHR
        Structured representation with labs and notes.
    """
    # Extract grid
    grid_match = re.search(r"^# Grid:\s*(\S+)", markdown, re.MULTILINE)
    grid = grid_match.group(1) if grid_match else "unknown"

    parsed = ParsedEHR(grid=grid, raw_markdown=markdown)

    # --- Parse labs section ---------------------------------------------------
    labs_match = re.search(
        r"^## Labs\n(.*?)(?=^## |\Z)", markdown, re.MULTILINE | re.DOTALL
    )
    if labs_match:
        labs_text = labs_match.group(1)
        for line in labs_text.strip().splitlines():
            line = line.strip()
            if not line.startswith("- ["):
                continue
            # Format: - [DATE] CONCEPT: VALUE UNIT (ref: LOW-HIGH)
            m = re.match(
                r"- \[([^\]]+)\]\s+(.+?):\s+(.+?)(?:\s+\(ref:\s+(.+?)\))?$",
                line,
            )
            if m:
                concept_and_value = m.group(2).strip()
                value_part = m.group(3).strip()
                # Split value and unit — value is everything before the last word
                # if the last word looks like a unit
                val_tokens = value_part.split()
                if len(val_tokens) >= 2 and not val_tokens[-1].replace(".", "").replace("-", "").isdigit():
                    value = " ".join(val_tokens[:-1])
                    unit = val_tokens[-1]
                else:
                    value = value_part
                    unit = ""

                parsed.labs.append(LabEntry(
                    date=m.group(1).strip(),
                    concept=concept_and_value,
                    value=value,
                    unit=unit,
                    ref_range=m.group(4).strip() if m.group(4) else "",
                ))

    # --- Parse notes section --------------------------------------------------
    notes_pattern = re.compile(
        r"^### \[([^\]]+)\]\s+(.+?)\n\*\*Source:\*\*\s+(.+?)\n\n(.*?)(?=^### |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for m in notes_pattern.finditer(markdown):
        parsed.notes.append(NoteEntry(
            date=m.group(1).strip(),
            title=m.group(2).strip(),
            source=m.group(3).strip(),
            text=m.group(4).strip(),
        ))

    logger.debug(
        "Parsed EHR for %s: %d labs, %d notes",
        grid, len(parsed.labs), len(parsed.notes),
    )
    return parsed


def get_available_grids(ehr_dir: Optional[Path] = None) -> List[str]:
    """Return a sorted list of all patient grid IDs with EHR files."""
    ehr_dir = ehr_dir or cfg.ehr_markdown_dir
    return sorted(p.stem for p in ehr_dir.glob("*.md"))
