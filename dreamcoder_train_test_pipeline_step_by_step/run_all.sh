#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/dreamcoder/bin/python}"

TRAIN_DATASET_FILE="${TRAIN_DATASET_FILE:-dataset/T=2_train.json}"
TEST_DATASET_FILE="${TEST_DATASET_FILE:-dataset/T=2_test.json}"
DREAMCODER_REPO_ROOT="${DREAMCODER_REPO_ROOT:-/mnt/c/BA/ec}"
DREAMCODER_TIMEOUT="${DREAMCODER_TIMEOUT:-60}"
DREAMCODER_TESTING_TIMEOUT="${DREAMCODER_TESTING_TIMEOUT:-60}"
DREAMCODER_ITERATIONS="${DREAMCODER_ITERATIONS:-3}"
DREAMCODER_FRONTIER_SIZE="${DREAMCODER_FRONTIER_SIZE:-50}"
DREAMCODER_USE_RECOGNITION="${DREAMCODER_USE_RECOGNITION:-true}"
DREAMCODER_NO_CONSOLIDATION="${DREAMCODER_NO_CONSOLIDATION:-false}"
MAX_TRAIN_TASKS="${MAX_TRAIN_TASKS:-0}"
DREAMCODER_CPUS="${DREAMCODER_CPUS:-1}"

TRAIN_PATH="$SCRIPT_DIR/$TRAIN_DATASET_FILE"
TEST_PATH="$SCRIPT_DIR/$TEST_DATASET_FILE"
TRAIN_KEY="$(basename "$TRAIN_DATASET_FILE" .json | tr '=.-' '___')"
TEST_KEY="$(basename "$TEST_DATASET_FILE" .json | tr '=.-' '___')"
RUN_KEY="train_${TRAIN_KEY}__test_${TEST_KEY}__ET_${DREAMCODER_TIMEOUT}_TT_${DREAMCODER_TESTING_TIMEOUT}_it_${DREAMCODER_ITERATIONS}_MF_${DREAMCODER_FRONTIER_SIZE}_rec_${DREAMCODER_USE_RECOGNITION}_nocons_${DREAMCODER_NO_CONSOLIDATION}"
OUTPUT_ROOT="$SCRIPT_DIR/outputs/$RUN_KEY"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

exec > >(tee "$LOG_DIR/pipeline.log") 2>&1

echo "Pipeline started: $(date)"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "TRAIN_DATASET_FILE=$TRAIN_DATASET_FILE"
echo "TEST_DATASET_FILE=$TEST_DATASET_FILE"
echo "TRAIN_PATH=$TRAIN_PATH"
echo "TEST_PATH=$TEST_PATH"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "DREAMCODER_REPO_ROOT=$DREAMCODER_REPO_ROOT"
echo "DREAMCODER_TIMEOUT=$DREAMCODER_TIMEOUT"
echo "DREAMCODER_TESTING_TIMEOUT=$DREAMCODER_TESTING_TIMEOUT"
echo "DREAMCODER_ITERATIONS=$DREAMCODER_ITERATIONS"
echo "DREAMCODER_FRONTIER_SIZE=$DREAMCODER_FRONTIER_SIZE"
echo "DREAMCODER_USE_RECOGNITION=$DREAMCODER_USE_RECOGNITION"
echo "DREAMCODER_NO_CONSOLIDATION=$DREAMCODER_NO_CONSOLIDATION"
echo "MAX_TRAIN_TASKS=$MAX_TRAIN_TASKS"
echo "DREAMCODER_CPUS=$DREAMCODER_CPUS"
echo "PYTHON=$PYTHON"
echo

if [[ ! -f "$TRAIN_PATH" ]]; then
  echo "Missing training dataset: $TRAIN_PATH"
  echo "Copy the file into dataset/ or set TRAIN_DATASET_FILE explicitly."
  exit 1
fi
if [[ ! -f "$TEST_PATH" ]]; then
  echo "Missing test dataset: $TEST_PATH"
  echo "Copy the file into dataset/ or set TEST_DATASET_FILE explicitly."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/step01_validate_datasets" "$OUTPUT_ROOT/step02_convert_train_test" "$OUTPUT_ROOT/step03_create_task_pickles" "$OUTPUT_ROOT/step04_run_dreamcoder" "$OUTPUT_ROOT/step05_detect_operations" "$OUTPUT_ROOT/step06_normalize_programs" "$OUTPUT_ROOT/step07_calculate_metrics" "$OUTPUT_ROOT/step08_summarize_results"

echo "step01_validate_datasets"
"$PYTHON" "$SCRIPT_DIR/scripts/step01_validate_datasets.py" "$TRAIN_PATH" "$TEST_PATH" "$OUTPUT_ROOT/step01_validate_datasets" | tee "$LOG_DIR/step01_validate_datasets.log"

echo "step02_convert_train_test_tasks"
"$PYTHON" "$SCRIPT_DIR/scripts/step02_convert_train_test_tasks.py" "$TRAIN_PATH" "$TEST_PATH" "$OUTPUT_ROOT/step02_convert_train_test" "$MAX_TRAIN_TASKS" | tee "$LOG_DIR/step02_convert_train_test_tasks.log"

echo "step03_create_dreamcoder_task_pickles"
"$PYTHON" "$SCRIPT_DIR/scripts/step03_create_dreamcoder_task_pickles.py" "$OUTPUT_ROOT/step02_convert_train_test/step02_train_tasks.json" "$OUTPUT_ROOT/step02_convert_train_test/step02_test_tasks.json" "$OUTPUT_ROOT/step03_create_task_pickles" "$DREAMCODER_REPO_ROOT" | tee "$LOG_DIR/step03_create_dreamcoder_task_pickles.log"

echo "step04_run_dreamcoder_train_test"
"$PYTHON" "$SCRIPT_DIR/scripts/step04_run_dreamcoder_train_test.py" \
  "$OUTPUT_ROOT/step03_create_task_pickles/step03_train_tasks.pkl" \
  "$OUTPUT_ROOT/step03_create_task_pickles/step03_test_tasks.pkl" \
  "$OUTPUT_ROOT/step02_convert_train_test/step02_train_tasks.json" \
  "$OUTPUT_ROOT/step02_convert_train_test/step02_test_tasks.json" \
  "$OUTPUT_ROOT/step04_run_dreamcoder" \
  "$DREAMCODER_REPO_ROOT" \
  "$DREAMCODER_TIMEOUT" \
  "$DREAMCODER_TESTING_TIMEOUT" \
  "$DREAMCODER_ITERATIONS" \
  "$DREAMCODER_FRONTIER_SIZE" \
  "$DREAMCODER_USE_RECOGNITION" \
  "$DREAMCODER_NO_CONSOLIDATION" \
  "$DREAMCODER_CPUS" | tee "$LOG_DIR/step04_run_dreamcoder_train_test.log"

echo "step05_detect_operations"
"$PYTHON" "$SCRIPT_DIR/scripts/step05_detect_operations.py" "$OUTPUT_ROOT/step04_run_dreamcoder/step04_test_results.csv" "$OUTPUT_ROOT/step05_detect_operations" | tee "$LOG_DIR/step05_detect_operations.log"

echo "step06_normalize_programs"
"$PYTHON" "$SCRIPT_DIR/scripts/step06_normalize_programs.py" "$OUTPUT_ROOT/step04_run_dreamcoder/step04_test_results.csv" "$OUTPUT_ROOT/step06_normalize_programs" | tee "$LOG_DIR/step06_normalize_programs.log"

echo "step07_calculate_metrics"
"$PYTHON" "$SCRIPT_DIR/scripts/step07_calculate_metrics.py" "$OUTPUT_ROOT/step06_normalize_programs/step06_normalized_test_programs.csv" "$OUTPUT_ROOT/step07_calculate_metrics" | tee "$LOG_DIR/step07_calculate_metrics.log"

echo "step08_summarize_results"
"$PYTHON" "$SCRIPT_DIR/scripts/step08_summarize_results.py" "$OUTPUT_ROOT/step07_calculate_metrics/step07_test_results_with_metrics.csv" "$OUTPUT_ROOT/step07_calculate_metrics/step07_metrics_summary.json" "$OUTPUT_ROOT/step04_run_dreamcoder/step04_train_test_summary.json" "$OUTPUT_ROOT/step08_summarize_results" | tee "$LOG_DIR/step08_summarize_results.log"

echo "Pipeline finished: $(date)"
echo "Summary: $OUTPUT_ROOT/step08_summarize_results/step08_summary.txt"
