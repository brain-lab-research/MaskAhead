#!/bin/bash
# Slurm job that holds one PPU card and runs tasks from a task list back to back
# (the card is not released between tasks). Exits after IDLE_MIN minutes of an
# empty list. Submit up to as many as you may hold:
#   sbatch -J kvdllm_eval --gres=gpu:1 --time=4:00:00 launch/ppu/slurm_worker.sh tasks/fastdllm.txt
#SBATCH -o results/logs/slurm-%j.log
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
TASKS="${1:?task file}"; IDLE_MIN="${IDLE_MIN:-15}"
mkdir -p results/logs
pop() {
  exec 9>"$TASKS.lock"; flock 9
  local line; line=$(grep -v '^\s*\(#\|$\)' "$TASKS" | head -n1)
  if [ -n "$line" ]; then grep -vxF -- "$line" "$TASKS" > "$TASKS.tmp"; mv "$TASKS.tmp" "$TASKS"; fi
  flock -u 9; echo "$line"
}
idle=0
while :; do
  task=$(pop)
  if [ -z "$task" ]; then
    [ "$idle" -ge $((IDLE_MIN * 2)) ] && exit 0
    idle=$((idle + 1)); sleep 30; continue
  fi
  idle=0
  echo "== $(date +%H:%M:%S) start: $task"
  if launch/ppu/container.sh "$FASTDLLM_PYTHON" scripts/run_task.py $task > "results/logs/$(date +%s)_$SLURM_JOB_ID.log" 2>&1; then
    echo "== $(date +%H:%M:%S) done: $task"
  else
    echo "$task" >> "$TASKS.failed"; echo "== FAILED: $task"
  fi
done
