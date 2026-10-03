#!/bin/bash
# Run a task list on several local GPUs, one worker per GPU.
# Each line of TASKS is the argument list of scripts/run_task.py.
# Lines are popped atomically, so workers never duplicate a task; a failed task
# is appended to TASKS.failed and the worker moves on. Safe to restart: finished
# cells are resumed (skipped) by run_task.py.
#
#   scripts/gpu_queue.sh tasks/dream.txt 0,1,2,3,4,5,6,7
set -uo pipefail
TASKS="${1:?task file}"; GPUS="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/results/logs"; mkdir -p "$LOGS"
touch "$TASKS"

pop() {
  exec 9>"$TASKS.lock"; flock 9
  local line; line=$(grep -v '^\s*\(#\|$\)' "$TASKS" | head -n1)
  if [ -n "$line" ]; then grep -vxF -- "$line" "$TASKS" > "$TASKS.tmp"; mv "$TASKS.tmp" "$TASKS"; fi
  flock -u 9; echo "$line"
}

worker() {
  local gpu="$1"
  while :; do
    local task; task=$(pop)
    [ -z "$task" ] && { echo "[gpu $gpu] queue empty, exiting"; return; }
    local name; name=$(echo "$task" | tr -s ' ' '\n' | grep -A1 -- '--config' | tail -1 | xargs basename | sed 's/.yaml//')
    local bench; bench=$(echo "$task" | tr -s ' ' '\n' | grep -A1 -- '--benchmark' | tail -1)
    local log="$LOGS/${name}_${bench}_gpu${gpu}_$(date +%H%M%S).log"
    echo "[gpu $gpu] $(date +%H:%M:%S) start: $task"
    local runner
    case " $task " in
      *" --model dream "*) runner="${DREAM_PYTHON:-python3}" ;;
      *" --model fastdllm "*) runner="${FASTDLLM_PYTHON:-python3}" ;;
      *" --model sdar "*) runner="${SDAR_PYTHON:-python3}" ;;
      *) echo "[gpu $gpu] unknown model in task: $task" >> "$log"; echo "$task" >> "$TASKS.failed"; continue ;;
    esac
    if CUDA_VISIBLE_DEVICES="$gpu" "$runner" "$ROOT/scripts/run_task.py" $task > "$log" 2>&1; then
      echo "[gpu $gpu] $(date +%H:%M:%S) done: $task"
    else
      echo "$task" >> "$TASKS.failed"
      echo "[gpu $gpu] $(date +%H:%M:%S) FAILED ($log): $task"
    fi
  done
}

IFS=',' read -ra ids <<< "$GPUS"
for g in "${ids[@]}"; do worker "$g" & done
wait
