#!/bin/bash
# run_rest_temp.sh — Resume pending Shard 1 patients using the other two GPU nodes.
# This script:
# 1. Establishes the SSH tunnel to Remote GPU 2 (10.151.30.80) on port 11436.
# 2. Checks local Ollama status on port 11434.
# 3. Dynamically identifies the pending patients from Shard 1.
# 4. Splits the pending patients into two equal groups.
# 5. Runs the first group on Remote GPU 2 (11436) and the second group on Local GPU (11434) in parallel.
# 6. Consolidates all JSON results into the final CSV.
# 7. Generates the final evaluation report.

set -e

# Configuration
EXP_NAME="celiac_agent_v3"
OUT_DIR="results/${EXP_NAME}"
JSON_DIR="${OUT_DIR}/json_results"
SUMMARY_DIR="${OUT_DIR}/summary"
CSV_FILE="${OUT_DIR}/${EXP_NAME}_results.csv"

# ANSI Colors for premium logs
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BOLD="\033[1m"
NC="\033[0m" # No Color

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}   Celiac Disease Agentic Pipeline: Resuming Shard 1 Patients   ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"
echo ""

# Create output dirs if not exist
mkdir -p "$JSON_DIR"
mkdir -p "$SUMMARY_DIR"

# 1. Establish SSH tunnel to Remote GPU 2 (10.151.30.80:11434 -> localhost:11436)
echo -e "${YELLOW}Checking SSH tunnel to Remote GPU 2 (port 11436)...${NC}"
if ! ss -tulpn 2>/dev/null | grep -q "11436"; then
    echo -e "${BLUE}SSH tunnel on port 11436 is not open. Establishing connection to 10.151.30.80...${NC}"
    echo -e "${YELLOW}Please enter the password for biand@10.151.30.80 when prompted below:${NC}"
    ssh -N -f -L 11436:localhost:11434 biand@10.151.30.80
    
    # Wait a few seconds to let it establish
    sleep 3
    if ! ss -tulpn 2>/dev/null | grep -q "11436"; then
        echo -e "${RED}${BOLD}ERROR: Failed to establish SSH tunnel on port 11436. Please check credentials or host availability.${NC}"
        exit 1
    fi
    echo -e "${GREEN}SSH tunnel to Remote GPU 2 successfully established on port 11436!${NC}"
else
    echo -e "${GREEN}SSH tunnel to Remote GPU 2 is already active on port 11436.${NC}"
fi

# 2. Check Local GPU status (localhost:11434)
echo -e "${YELLOW}Checking Local GPU status (port 11434)...${NC}"
if ! ss -tulpn 2>/dev/null | grep -q "11434"; then
    echo -e "${RED}${BOLD}ERROR: Local Ollama is not active on port 11434. Please start it before proceeding.${NC}"
    exit 1
fi
echo -e "${GREEN}Local Ollama is active on port 11434.${NC}"
echo ""

# 3. Dynamically find the pending patients of Shard 1
echo -e "${YELLOW}Identifying pending patients of Shard 1...${NC}"
PENDING_GRIDS=$(uv run python -c "
import pandas as pd, glob, os
df = pd.read_excel('data/Celiac Diagnosis by Manual Review.xlsx')
grids = df.iloc[:, 0].astype(str).str.strip().tolist()
# Shard 1 corresponds to indices 0 to 177 (first chunk of 3-way sharding of 530 total patients)
shard1_grids = grids[0:177]
done_grids = {os.path.basename(f).replace('.json', '') for f in glob.glob('${JSON_DIR}/*.json')}
pending = [g for g in shard1_grids if g not in done_grids]
print(' '.join(pending))
")

read -r -a GRIDS_ARR <<< "$PENDING_GRIDS"
TOTAL_PENDING=${#GRIDS_ARR[@]}

echo -e "${GREEN}Found ${BOLD}${TOTAL_PENDING}${NC}${GREEN} pending patients from Shard 1.${NC}"

if [ "$TOTAL_PENDING" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All Shard 1 patients have already been processed! Proceeding directly to evaluation.${NC}"
    echo ""
else
    # 4. Split the pending patients into two halves
    HALF=$(( (TOTAL_PENDING + 1) / 2 ))
    
    # Slice the arrays
    PART_A=("${GRIDS_ARR[@]:0:HALF}")
    PART_B=("${GRIDS_ARR[@]:HALF}")
    
    echo -e "${BLUE}Splitting workload:${NC}"
    echo -e "  - ${BOLD}Part A:${NC} ${#PART_A[@]} patients -> Remote GPU 2 (10.151.30.80 on port 11436)"
    echo -e "  - ${BOLD}Part B:${NC} ${#PART_B[@]} patients -> Local GPU (localhost on port 11434)"
    echo ""
    
    # Convert parts back to space-separated lists for pipeline.py
    GRIDS_A="${PART_A[*]}"
    GRIDS_B="${PART_B[*]}"
    
    # 5. Start parallel runs
    echo -e "${YELLOW}Starting Part A on Remote GPU 2 (output in $OUT_DIR/shard1_part1.log)...${NC}"
    uv run python src/celiac_agent/pipeline.py \
        --grids $GRIDS_A \
        --ollama-url "http://localhost:11436" \
        --output-dir "$JSON_DIR" \
        --output-csv "$CSV_FILE" \
        > "$OUT_DIR/shard1_part1.log" 2>&1 &
    PID_A=$!
    
    echo -e "${YELLOW}Starting Part B on Local GPU (output in $OUT_DIR/shard1_part2.log)...${NC}"
    uv run python src/celiac_agent/pipeline.py \
        --grids $GRIDS_B \
        --use-local-ollama \
        --output-dir "$JSON_DIR" \
        --output-csv "$CSV_FILE" \
        > "$OUT_DIR/shard1_part2.log" 2>&1 &
    PID_B=$!
    
    echo -e "${GREEN}Both parts running in the background!${NC}"
    echo -e "  - PID Part A: ${BOLD}$PID_A${NC}"
    echo -e "  - PID Part B: ${BOLD}$PID_B${NC}"
    echo ""
    echo -e "You can monitor the progress with:"
    echo -e "${BLUE}tail -f $OUT_DIR/shard1_part1.log $OUT_DIR/shard1_part2.log${NC}"
    echo ""
    
    echo -e "${YELLOW}Waiting for both parts to complete...${NC}"
    wait $PID_A
    echo -e "${GREEN}Part A finished.${NC}"
    wait $PID_B
    echo -e "${GREEN}Part B finished.${NC}"
    echo ""
fi

# 6. Consolidate results into final CSV from all JSONs
echo -e "${YELLOW}Consolidating all JSON results into the final CSV...${NC}"
uv run python -c "
import pandas as pd
import glob
import json
from pathlib import Path

json_dir = Path('${JSON_DIR}')
csv_file = Path('${CSV_FILE}')

all_results = []
for p in sorted(json_dir.glob('*.json')):
    with open(p) as f:
        try:
            data = json.load(f)
        except Exception:
            continue
    all_results.append({
        'grid': data.get('grid', p.stem),
        'diagnosis': data.get('diagnosis', 'Unknown'),
        'confidence': data.get('confidence', 0.0),
        'reasoning': data.get('reasoning', ''),
        'evidence': '; '.join(data.get('evidence', [])[:5]) if isinstance(data.get('evidence'), list) else '',
        'decision_path': data.get('decision_path', ''),
        'lab_decision': data.get('lab_decision', ''),
        'time_taken_s': data.get('time_taken_s', 0.0),
    })

pd.DataFrame(all_results).to_csv(csv_file, index=False)
print(f'Successfully consolidated {len(all_results)} results into {csv_file}')
"

# 7. Generate Evaluation Report and Confusion Matrix
echo -e "${YELLOW}Generating Evaluation Report and Confusion Matrix...${NC}"
uv run python src/generate_eval_report.py \
    --res_dir "$JSON_DIR" \
    --out_file "${SUMMARY_DIR}/evaluation_report.md"

echo ""
echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}                      Pipeline Resumed & Completed!             ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "Results directory:  ${BLUE}${OUT_DIR}${NC}"
echo -e "Evaluation report:  ${BLUE}${SUMMARY_DIR}/evaluation_report.md${NC}"
echo -e "Confusion matrix:   ${BLUE}${SUMMARY_DIR}/confusion_matrix.png${NC}"
