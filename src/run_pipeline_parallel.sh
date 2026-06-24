#!/bin/bash

# Configuration
EXP_NAME="celiac_agent_v3"
OUT_DIR="results/${EXP_NAME}"
JSON_DIR="${OUT_DIR}/json_results"
SUMMARY_DIR="${OUT_DIR}/summary"
CSV_FILE="${OUT_DIR}/${EXP_NAME}_results.csv"

echo "Creating output directories..."
mkdir -p "$JSON_DIR"
mkdir -p "$SUMMARY_DIR"

echo "Setting up background port forwarding..."
# You must configure passwordless SSH or replace 'node1' and 'node2' with actual aliases
ssh -N -f -L 11435:localhost:11434 biand@10.151.30.252 || echo "Port 11435 already forwarded or SSH failed."
ssh -N -f -L 11436:localhost:11434 biand@10.151.30.80 || echo "Port 11436 already forwarded or SSH failed."

echo "Starting Shard 1/3 on Remote GPU 1 (port 11435)..."
uv run python src/celiac_agent/pipeline.py \
    --grids-from-file "data/Celiac Diagnosis by Manual Review.xlsx" \
    --shard 1 \
    --shard-total 3 \
    --ollama-url "http://localhost:11435" \
    --output-dir "$JSON_DIR" \
    --output-csv "$CSV_FILE" \
    > "$OUT_DIR/shard1.log" 2>&1 &
PID1=$!

echo "Starting Shard 2/3 on Remote GPU 2 (port 11436)..."
uv run python src/celiac_agent/pipeline.py \
    --grids-from-file "data/Celiac Diagnosis by Manual Review.xlsx" \
    --shard 2 \
    --shard-total 3 \
    --ollama-url "http://localhost:11436" \
    --output-dir "$JSON_DIR" \
    --output-csv "$CSV_FILE" \
    > "$OUT_DIR/shard2.log" 2>&1 &
PID2=$!

echo "Starting Shard 3/3 on Local GPU (port 11434)..."
uv run python src/celiac_agent/pipeline.py \
    --grids-from-file "data/Celiac Diagnosis by Manual Review.xlsx" \
    --shard 3 \
    --shard-total 3 \
    --use-local-ollama \
    --output-dir "$JSON_DIR" \
    --output-csv "$CSV_FILE" \
    > "$OUT_DIR/shard3.log" 2>&1 &
PID3=$!

echo "All 3 shards started in the background!"
echo "PID for Remote GPU 1 process: $PID1"
echo "PID for Remote GPU 2 process: $PID2"
echo "PID for Local GPU process:  $PID3"
echo ""
echo "You can monitor their progress by running:"
echo "tail -f $OUT_DIR/shard1.log $OUT_DIR/shard2.log $OUT_DIR/shard3.log"
echo ""

echo "Waiting for all shards to finish before generating the evaluation report..."
wait $PID1
wait $PID2
wait $PID3

echo "All shards finished. Generating Evaluation Report and Confusion Matrix..."
uv run python src/generate_eval_report.py \
    --res_dir "$JSON_DIR" \
    --out_file "${SUMMARY_DIR}/evaluation_report.md"

echo "Pipeline complete! Results saved in $OUT_DIR"
