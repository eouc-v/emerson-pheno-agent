"""
pipeline.py — CLI entry point for the agentic celiac diagnosis system.

Usage examples:
  # Diagnose a single patient
  uv run python src/pheno_agent/pipeline.py --grids R201643869

  # Diagnose patients from the manual review file
  uv run python src/pheno_agent/pipeline.py \\
    --grids-from-file "data/Celiac Diagnosis by Manual Review.xlsx" \\
    --sample 10

  # Diagnose all patients in the EHR dataset
  uv run python src/pheno_agent/pipeline.py --all

  # Evaluate against ground truth after diagnosis
  uv run python src/pheno_agent/pipeline.py \\
    --grids-from-file "data/Celiac Diagnosis by Manual Review.xlsx" \\
    --evaluate

  # Override models
  uv run python src/pheno_agent/pipeline.py \\
    --grids R201643869 \\
    --reasoning-model qwen3.5:122b \\
    --extraction-model gemma4:31b-it-q4_K_M
"""

import argparse
import logging
import sys
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pheno_agent.config import cfg
from pheno_agent.orchestrator import Orchestrator
from pheno_agent.tools.ehr_reader import get_available_grids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "positive": "Positive", "yes": "Positive", "1": "Positive",
    "negative": "Negative", "no": "Negative", "0": "Negative",
    "indeterminate": "Indeterminate", "uncertain": "Indeterminate",
    "unknown": "Indeterminate", "pmh": "Indeterminate",
}


def run_evaluation(results_csv: Path, ground_truth_path: Path):
    """Compare predictions to ground truth and print metrics."""
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        classification_report,
        confusion_matrix,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load predictions
    preds = pd.read_csv(results_csv)
    preds["grid"] = preds["grid"].astype(str).str.strip()
    preds["pred_label"] = preds["diagnosis"].str.strip().str.lower().map(LABEL_MAP).fillna("Indeterminate")

    # Load ground truth
    gt = pd.read_excel(ground_truth_path)
    gt.columns = [c.strip() for c in gt.columns]
    id_col = next((c for c in gt.columns if "patient" in c.lower() or "grid" in c.lower() or "id" in c.lower()), gt.columns[0])
    diag_col = next((c for c in gt.columns if "diagnosis" in c.lower()), gt.columns[1])
    gt = gt[[id_col, diag_col]].copy()
    gt.columns = ["grid", "true_label"]
    gt["grid"] = gt["grid"].astype(str).str.strip()
    gt["true_label"] = gt["true_label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    gt = gt.dropna(subset=["true_label"])

    # Merge
    merged = gt.merge(preds[["grid", "pred_label"]], on="grid", how="inner")
    logger.info("Matched %d patients between predictions and ground truth.", len(merged))

    if merged.empty:
        logger.error("No matching patients found.")
        return

    labels = ["Positive", "Negative", "Indeterminate"]
    report = classification_report(merged["true_label"], merged["pred_label"], labels=labels, zero_division=0)
    logger.info("\nClassification Report:\n%s", report)

    # Confusion matrix
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(merged["true_label"], merged["pred_label"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Agentic Celiac Diagnosis: Manual vs. Predicted", fontsize=16)
    plt.xlabel("Predicted", fontsize=13)
    plt.ylabel("Manual Review", fontsize=13)
    plt.tight_layout()
    cm_path = fig_dir / "agent_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", cm_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic Celiac Disease Diagnosis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Patient selection
    parser.add_argument("--grids", nargs="+", help="Specific patient GRIDs")
    parser.add_argument("--grids-from-file", type=Path, default=None,
                        help="Excel/CSV file with patient GRIDs")
    parser.add_argument("--all", action="store_true",
                        help="Process all patients in EHR dataset")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to first N patients")

    # Model overrides
    parser.add_argument("--reasoning-model", type=str, default=None)
    parser.add_argument("--extraction-model", type=str, default=None)
    parser.add_argument("--fast-model", type=str, default=None)

    # Agent parameters
    parser.add_argument("--max-reflections", type=int, default=None)
    parser.add_argument("--no-chroma", action="store_true",
                        help="Skip ChromaDB retrieval")
    parser.add_argument("--use-local-ollama", action="store_true",
                        help="Use local Ollama instead of remote")
    parser.add_argument("--ollama-url", type=str, default=None,
                        help="Specific Ollama URL to use (e.g. http://localhost:11436)")
    
    # Sharding
    parser.add_argument("--shard", type=int, default=1, help="Shard index (1-indexed)")
    parser.add_argument("--shard-total", type=int, default=1, help="Total number of shards")

    # Output
    parser.add_argument("--output-dir", type=Path, default=cfg.agent_results_dir / "json_results")
    parser.add_argument("--output-csv", type=Path, default=cfg.agent_results_csv)

    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation after diagnosis")
    parser.add_argument("--ground-truth", type=Path, default=cfg.ground_truth_path)

    # Resume
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't skip already-processed patients")

    return parser.parse_args()


def resolve_grids(args) -> list:
    """Determine which patient grids to process."""
    if args.grids:
        grids = args.grids
    elif args.grids_from_file:
        gf = args.grids_from_file
        if gf.suffix in (".xlsx", ".xls"):
            gf_df = pd.read_excel(gf)
        else:
            gf_df = pd.read_csv(gf)
        id_col = next(
            (c for c in gf_df.columns if "patient" in c.lower() or "grid" in c.lower() or "id" in c.lower()),
            gf_df.columns[0],
        )
        grids = list(dict.fromkeys(gf_df[id_col].astype(str).str.strip().tolist()))
        logger.info("Loaded %d grids from %s (column: '%s').", len(grids), gf.name, id_col)
    elif args.all:
        grids = get_available_grids()
        logger.info("Found %d grids in EHR dataset.", len(grids))
    else:
        logger.error("No patients specified. Use --grids, --grids-from-file, or --all.")
        sys.exit(1)

    if args.sample:
        grids = grids[:args.sample]

    return grids


def main():
    args = parse_args()

    if args.ollama_url:
        cfg.agent.ollama_host_remote = args.ollama_url
        cfg.agent.use_remote_ollama = True
    elif args.use_local_ollama:
        cfg.agent.use_remote_ollama = False

    grids = resolve_grids(args)
    
    if args.shard_total > 1:
        chunk_size = math.ceil(len(grids) / args.shard_total)
        start_idx = (args.shard - 1) * chunk_size
        end_idx = start_idx + chunk_size
        grids = grids[start_idx:end_idx]
        logger.info("Sharding enabled: Processing shard %d/%d", args.shard, args.shard_total)

    logger.info("Processing %d patients.", len(grids))

    # Check for already-processed patients
    already_done = set()
    if not args.no_resume and args.output_dir.exists():
        already_done = {p.stem for p in args.output_dir.glob("*.json")}
        if already_done:
            logger.info("Resuming: %d patients already processed.", len(already_done))

    pending = [g for g in grids if g not in already_done]
    logger.info("Pending: %d patients to process.", len(pending))

    if not pending:
        logger.info("All patients already processed. Use --no-resume to reprocess.")
    else:
        # Initialise orchestrator
        orchestrator = Orchestrator(
            reasoning_model=args.reasoning_model,
            extraction_model=args.extraction_model,
            fast_model=args.fast_model,
            max_reflections=args.max_reflections,
            use_chroma=not args.no_chroma,
        )

        # Process patients sequentially
        all_results = []
        for grid in tqdm(pending, desc="Diagnosing patients", unit="patient"):
            try:
                result = orchestrator.diagnose_patient(grid)
                orchestrator.save_result(result, args.output_dir)
                all_results.append(result)
                logger.info(
                    "Patient %s → %s (confidence=%.2f, time=%.1fs)",
                    grid, result["diagnosis"], result["confidence"], result["time_taken_s"],
                )
            except Exception as e:
                logger.error("Error processing %s: %s", grid, e, exc_info=True)
                all_results.append({
                    "grid": grid,
                    "diagnosis": "ERROR",
                    "confidence": 0.0,
                    "reasoning": f"Processing error: {e}",
                    "evidence": [],
                    "decision_path": "",
                    "lab_decision": "",
                    "time_taken_s": 0.0,
                })

        # Save aggregated CSV
        if all_results:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            flat = []
            for r in all_results:
                flat.append({
                    "grid": r["grid"],
                    "diagnosis": r["diagnosis"],
                    "confidence": r["confidence"],
                    "reasoning": r.get("reasoning", ""),
                    "evidence": "; ".join(r.get("evidence", [])[:5]),
                    "decision_path": r.get("decision_path", ""),
                    "lab_decision": r.get("lab_decision", ""),
                    "time_taken_s": r.get("time_taken_s", 0),
                })
            pd.DataFrame(flat).to_csv(args.output_csv, index=False)
            logger.info("Aggregated results saved to %s", args.output_csv)

    # Evaluation
    if args.evaluate:
        logger.info("=" * 60)
        logger.info("Running Evaluation")
        logger.info("=" * 60)
        # Build CSV from all JSON results if not already built
        if not args.output_csv.exists():
            json_results = []
            for p in args.output_dir.glob("*.json"):
                import json
                with open(p) as f:
                    data = json.load(f)
                json_results.append({
                    "grid": data["grid"],
                    "diagnosis": data["diagnosis"],
                    "confidence": data.get("confidence", 0),
                })
            if json_results:
                pd.DataFrame(json_results).to_csv(args.output_csv, index=False)

        run_evaluation(args.output_csv, args.ground_truth)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
