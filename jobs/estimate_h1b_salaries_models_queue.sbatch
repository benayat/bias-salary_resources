#!/bin/bash
#SBATCH --job-name=h1b_salary_queue
#SBATCH --partition=power-gpu
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --qos=owner_730
#SBATCH --ntasks=1
#SBATCH --account=cycle2_biu_kraus_prj
#SBATCH --signal=TERM@900
#SBATCH --container-image=/home/users/benaya7/vllm_env_v2.sqsh
#SBATCH --container-mounts=/home/users/benaya7:/home/users/benaya7
#SBATCH --output=/home/users/benaya7/projects/bias-salary_resources/logs/h1b_salary_queue_%j.out
#SBATCH --error=/home/users/benaya7/projects/bias-salary_resources/logs/h1b_salary_queue_%j.err

set -euo pipefail
export PYTHONPATH=.
BASE_DIR="${BASE_DIR:-/home/users/benaya7/projects/bias-salary_resources}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE_DIR}"
LOG_DIR="${LOG_DIR:-$BASE_DIR/logs}"

INPUT_CSV="${INPUT_CSV:-$PROJECT_ROOT/data/h1b_fy2025_q4_qwen36_main_sampled_2000.csv}"
MODELS_FILE="${MODELS_FILE:-$PROJECT_ROOT/configs/salary_prediction_models.txt}"

RUN_NAME="${RUN_NAME:-h1b_fy2025_q4_qwen36_main_sampled_2000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/h1b-lca-disclosure-data-2020-2024}"

PERSONA="${PERSONA:-salary_estimator}"
LLM_CONFIG="${LLM_CONFIG:-hpc2h200}"
CHUNK_SIZE="${CHUNK_SIZE:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Missing input CSV: $INPUT_CSV" >&2
  exit 3
fi

if [[ ! -f "$MODELS_FILE" ]]; then
  echo "Missing model list: $MODELS_FILE" >&2
  exit 4
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PROJECT_ROOT/.cache/vllm}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

pip install ml-dtypes

# SLURM usually sets CUDA_VISIBLE_DEVICES to assigned GPU IDs.
# If it does not, assume local 4-GPU allocation.
ALLOCATED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "$ALLOCATED_CUDA_DEVICES"

WORKER_COUNT="${WORKER_COUNT:-${#GPU_IDS[@]}}"
if (( WORKER_COUNT > ${#GPU_IDS[@]} )); then
  WORKER_COUNT="${#GPU_IDS[@]}"
fi
if (( WORKER_COUNT < 1 )); then
  echo "No visible GPUs. CUDA_VISIBLE_DEVICES='$ALLOCATED_CUDA_DEVICES'" >&2
  exit 5
fi

QUEUE_DIR="$OUTPUT_ROOT/.salary_queue_${SLURM_JOB_ID:-manual}"
NEXT_FILE="$QUEUE_DIR/next_index.txt"
LOCK_FILE="$QUEUE_DIR/queue.lock"
FAIL_FILE="$QUEUE_DIR/failed_models.tsv"
FILTERED_MODELS="$QUEUE_DIR/models.txt"

mkdir -p "$QUEUE_DIR"
printf '1\n' > "$NEXT_FILE"
: > "$LOCK_FILE"
: > "$FAIL_FILE"

grep -Ev '^[[:space:]]*($|#)' "$MODELS_FILE" > "$FILTERED_MODELS"
NUM_MODELS="$(wc -l < "$FILTERED_MODELS" | tr -d ' ')"

if [[ "$NUM_MODELS" == "0" ]]; then
  echo "No models found in $MODELS_FILE" >&2
  exit 6
fi

echo "H1B salary-estimation queue"
echo "Project root: $PROJECT_ROOT"
echo "Input CSV: $INPUT_CSV"
echo "Output root: $OUTPUT_ROOT"
echo "Run name: $RUN_NAME"
echo "Models file: $MODELS_FILE"
echo "Models: $NUM_MODELS"
echo "Persona: $PERSONA"
echo "LLM config: $LLM_CONFIG"
echo "Chunk size: $CHUNK_SIZE"
echo "Save interval: $SAVE_INTERVAL"
echo "Allocated CUDA_VISIBLE_DEVICES: $ALLOCATED_CUDA_DEVICES"
echo "Workers: $WORKER_COUNT"

slug_model() {
  local model="$1"
  uv run python - "$model" <<'PY'
import re
import sys

model = sys.argv[1]
slug = model.strip().split("/")[-1]
slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug)
slug = slug.strip("_")
print(slug or "model")
PY
}

claim_next_model() {
  local claimed_index
  (
    flock -x 200
    local current
    current="$(cat "$NEXT_FILE")"
    if (( current > NUM_MODELS )); then
      exit 100
    fi
    claimed_index="$current"
    echo $((current + 1)) > "$NEXT_FILE"
    printf '%s\n' "$claimed_index"
  ) 200>"$LOCK_FILE"
}

run_model_on_gpu() {
  local worker_id="$1"
  local gpu_id="$2"
  local model_index="$3"
  local model="$4"

  local model_slug
  model_slug="$(slug_model "$model")"

  local out_dir="$OUTPUT_ROOT/sampled-${RUN_NAME}/${model_slug}"
  local run_log="$out_dir/h1b_salary_estimate__${model_slug}__job_${SLURM_JOB_ID:-manual}__worker_${worker_id}.log"
  local expected_csv="$out_dir/llm_estimated_salaries-${PERSONA}.csv"

  mkdir -p "$out_dir"

  {
    echo "[$(date --iso-8601=seconds)] START worker=${worker_id} gpu=${gpu_id} model_index=${model_index}/${NUM_MODELS} model=${model}"
    echo "Expected output CSV: $expected_csv"

    CUDA_VISIBLE_DEVICES="$gpu_id" uv run python scripts/h1b_dataset/estimate_h1b_salaries.py \
      --input-csv-file "$INPUT_CSV" \
      --client-type vllm \
      --llm-config "$LLM_CONFIG" \
      --model "$model" \
      --tensor-parallel-size 1 \
      --chunk-size "$CHUNK_SIZE" \
      --personas-to-use "$PERSONA" \
      --output-root "$OUTPUT_ROOT" \
      --run-name "$RUN_NAME" \
      --save-interval "$SAVE_INTERVAL"

     uv run python - "$expected_csv" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Missing expected output: {path}")

df = pd.read_csv(path, low_memory=False)

required = [
    "estimated_salary_in_usd",
    "salary_parse_status",
    "salary_raw_output",
    "IS_AI",
    "PREVAILING_WAGE",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing output columns: {missing}")

status = df["salary_parse_status"].fillna("pending").astype(str)
n = len(df)
completed = status.isin({
    "ok",
    "empty",
    "no_number",
    "multiple_numbers_or_range",
    "non_integer_number",
    "out_of_plausible_range",
    "malformed",
}).sum()
ok = status.eq("ok").sum()
pending = status.eq("pending").sum()

print(f"Validation rows={n} completed={completed} ok={ok} pending={pending}")
print(status.value_counts(dropna=False).to_string())

if n == 0:
    raise SystemExit("Output has zero rows")
if pending > 0:
    raise SystemExit(f"Output still has pending rows: {pending}")
if ok == 0:
    raise SystemExit("No parse-ok salary estimates")
PY

    echo "[$(date --iso-8601=seconds)] DONE worker=${worker_id} gpu=${gpu_id} model=${model}"
  } > "$run_log" 2>&1
}

worker_loop() {
  local worker_id="$1"
  local gpu_id="$2"
  echo "Worker ${worker_id} started on GPU ${gpu_id}"

  while true; do
    local model_index
    if ! model_index="$(claim_next_model)"; then
      echo "Worker ${worker_id}: queue empty"
      break
    fi

    local model
    model="$(sed -n "${model_index}p" "$FILTERED_MODELS")"
    echo "Worker ${worker_id}: claimed model ${model_index}/${NUM_MODELS}: ${model}"

    if ! run_model_on_gpu "$worker_id" "$gpu_id" "$model_index" "$model"; then
      echo -e "${model_index}\t${gpu_id}\t${model}" >> "$FAIL_FILE"
      echo "Worker ${worker_id}: FAILED model ${model_index}/${NUM_MODELS}: ${model}" >&2
      # Continue to next model so one bad model does not waste remaining GPUs.
    fi
  done

  echo "Worker ${worker_id} finished"
}

PIDS=()
for ((i=0; i<WORKER_COUNT; i++)); do
  worker_loop "$i" "${GPU_IDS[$i]}" &
  PIDS+=("$!")
done

STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    STATUS=1
  fi
done

if [[ -s "$FAIL_FILE" ]]; then
  echo "One or more models failed:" >&2
  cat "$FAIL_FILE" >&2
  STATUS=1
fi

if (( STATUS == 0 )); then
  echo "All salary-estimation models completed successfully."
else
  echo "Salary-estimation queue completed with failures." >&2
fi

exit "$STATUS"
