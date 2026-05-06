#!/usr/bin/env bash

# 顺序运行 4 个 checkpoint benchmark，并汇总总时长。
# 在 books/ai-systems-performance-engineering/chapters/ch03 目录下运行：
#
#   ./dcp-async-checkpoint/run_all.sh

set -uo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAPTER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCRIPTS=(
  "torch_save.py"
  "dcp_save.py"
  "dcp_async_save.py"
  "dcp_async_default_stager.py"
)

NAMES=()
STATUSES=()
DURATIONS=()

format_duration() {
  local seconds="$1"
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local remaining=$((seconds % 60))

  if ((hours > 0)); then
    printf "%dh %dm %ds" "$hours" "$minutes" "$remaining"
  elif ((minutes > 0)); then
    printf "%dm %ds" "$minutes" "$remaining"
  else
    printf "%ds" "$remaining"
  fi
}

run_script() {
  local script_name="$1"
  local start_time
  local end_time
  local duration
  local status

  printf '=%.0s' {1..88}
  printf '\n'
  printf 'Running %s\n' "$script_name"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'Command: uv run python dcp-async-checkpoint/%s\n' "$script_name"
  printf '=%.0s' {1..88}
  printf '\n'

  start_time="$(date +%s)"
  (
    cd "$CHAPTER_DIR" &&
      uv run python "dcp-async-checkpoint/${script_name}"
  )
  status="$?"
  end_time="$(date +%s)"
  duration=$((end_time - start_time))

  printf -- '-%.0s' {1..88}
  printf '\n'
  printf 'Finished %s: status=%s duration=%s\n' \
    "$script_name" "$status" "$(format_duration "$duration")"
  printf -- '-%.0s' {1..88}
  printf '\n'

  NAMES+=("$script_name")
  STATUSES+=("$status")
  DURATIONS+=("$duration")
}

print_summary() {
  local failed=0

  printf '\nBenchmark summary\n'
  printf '%-27s  %12s\n' "script" "duration"
  printf '%-27s  %12s\n' "---------------------------" "------------"

  for i in "${!NAMES[@]}"; do
    if [[ "${STATUSES[$i]}" != "0" ]]; then
      failed=1
    fi

    printf '%-27s  %12s\n' \
      "${NAMES[$i]}" \
      "$(format_duration "${DURATIONS[$i]}")"
  done

  return "$failed"
}

main() {
  for script_name in "${SCRIPTS[@]}"; do
    run_script "$script_name"
  done

  print_summary
}

main "$@"
