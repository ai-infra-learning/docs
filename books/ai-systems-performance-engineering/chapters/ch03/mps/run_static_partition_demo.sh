#!/usr/bin/env bash
set -euo pipefail

# 对比 Basic MPS 和 static SM partitioning 的吞吐。
# 两轮都运行相同数量的 CUDA client，每个 client 都执行相同时间的矩阵乘法。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH03_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_INDEX="${GPU_INDEX:-0}"
RUN_SECONDS="${RUN_SECONDS:-30}"
N="${N:-1024}"
PARTITION_CHUNKS="${PARTITION_CHUNKS:-7 7 7 6}"

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

MPS_STARTED=0
PIDS=()
BASIC_OUT_FILES=()
STATIC_OUT_FILES=()
PARTITIONS=()
WAIT_FAILED=0

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done

  if [[ "$MPS_STARTED" -eq 1 ]]; then
    # 关闭本次启动的 MPS control daemon。
    echo quit | "${SUDO[@]}" nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi

  # 恢复 GPU 默认 compute mode，避免影响后续任务。
  "${SUDO[@]}" nvidia-smi -i "$GPU_INDEX" -c DEFAULT >/dev/null 2>&1 || true

  for file in "${BASIC_OUT_FILES[@]}" "${STATIC_OUT_FILES[@]}"; do
    if [[ -n "$file" ]]; then
      rm -f "$file"
    fi
  done
}
trap cleanup EXIT

sum_iters() {
  sed -n 's/.*iters=\([0-9][0-9]*\).*/\1/p' "$1" | awk '{sum += $1} END {print sum + 0}'
}

sum_files() {
  local total=0
  local file

  for file in "$@"; do
    total=$((total + $(sum_iters "$file")))
  done

  echo "$total"
}

partition_count() {
  local count=0
  local chunk

  for chunk in $PARTITION_CHUNKS; do
    count=$((count + 1))
  done

  echo "$count"
}

print_config() {
  echo "GPU_INDEX=$GPU_INDEX"
  echo "RUN_SECONDS=$RUN_SECONDS"
  echo "N=$N"
  echo "PARTITION_CHUNKS=$PARTITION_CHUNKS"
  echo "CLIENTS=$(partition_count)"
}

start_mps() {
  local mode="$1"

  # EXCLUSIVE_PROCESS 让 nvidia-cuda-mps-server 成为这张 GPU 的统一入口。
  "${SUDO[@]}" nvidia-smi -i "$GPU_INDEX" -c EXCLUSIVE_PROCESS >/dev/null

  if [[ "$mode" == "static" ]]; then
    # -S 开启 static SM partitioning。
    "${SUDO[@]}" env CUDA_VISIBLE_DEVICES="$GPU_INDEX" nvidia-cuda-mps-control -d -S
  else
    "${SUDO[@]}" env CUDA_VISIBLE_DEVICES="$GPU_INDEX" nvidia-cuda-mps-control -d
  fi

  MPS_STARTED=1
}

stop_mps() {
  if [[ "$MPS_STARTED" -eq 1 ]]; then
    echo quit | "${SUDO[@]}" nvidia-cuda-mps-control >/dev/null 2>&1 || true
    MPS_STARTED=0
    sleep 1
  fi
}

run_demo_client() {
  local out_file="$1"
  local partition="${2:-}"

  (
    cd "$CH03_DIR"
    if [[ -n "$partition" ]]; then
      # CUDA_MPS_SM_PARTITION 必须在 CUDA 初始化前设置。
      CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
        CUDA_MPS_SM_PARTITION="$partition" \
        uv run python mps/mps_demo.py \
          --workers 1 \
          --seconds "$RUN_SECONDS" \
          --n "$N"
    else
      CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
        uv run python mps/mps_demo.py \
          --workers 1 \
          --seconds "$RUN_SECONDS" \
          --n "$N"
    fi
  ) >"$out_file" 2>&1 &

  PIDS+=("$!")
}

wait_clients() {
  local pid

  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      WAIT_FAILED=1
    fi
  done

  PIDS=()
}

print_client_output() {
  local title="$1"
  shift

  echo
  echo "================================================================================"
  echo "$title"
  echo "================================================================================"

  local file
  for file in "$@"; do
    cat "$file"
  done
}

run_basic_mps() {
  local clients
  local i
  local out_file

  echo
  echo "================================================================================"
  echo "Run Basic MPS"
  echo "================================================================================"

  start_mps "basic"
  clients="$(partition_count)"

  for ((i = 0; i < clients; i++)); do
    out_file="$(mktemp)"
    BASIC_OUT_FILES+=("$out_file")
    run_demo_client "$out_file"
  done

  sleep 5

  echo
  echo "# echo ps | sudo nvidia-cuda-mps-control"
  echo "--------------------------------------------------------------------------------"
  echo ps | "${SUDO[@]}" nvidia-cuda-mps-control || true

  wait_clients
  print_client_output "Basic MPS client output" "${BASIC_OUT_FILES[@]}"
  stop_mps
}

create_partition() {
  local chunk="$1"
  local output
  local partition

  # 创建指定 chunk 数的 SM partition。
  # 新版 MPS control 的输出形如 "Partition GPU-.../... created"。
  # CUDA_MPS_SM_PARTITION 只接受中间的 "GPU-.../..." 这一段。
  output="$(echo "sm_partition add $GPU_UUID $chunk" | "${SUDO[@]}" nvidia-cuda-mps-control)"
  partition="$(sed -nE 's/^.*(GPU-[^[:space:]]+\/[^[:space:]]+).*$/\1/p' <<<"$output" | tail -n 1)"
  if [[ "$partition" != */* ]]; then
    echo "Failed to create SM partition for chunk=$chunk. Output: $output" >&2
    exit 1
  fi

  echo "$partition"
}

run_static_partitions() {
  local chunk
  local out_file
  local partition

  echo
  echo "================================================================================"
  echo "Run Static SM Partitioning"
  echo "================================================================================"

  start_mps "static"

  GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU_INDEX" | tr -d '[:space:]')"
  echo "GPU_UUID=$GPU_UUID"

  echo
  echo "# echo \"lspart\" | sudo nvidia-cuda-mps-control"
  echo "--------------------------------------------------------------------------------"
  echo "lspart" | "${SUDO[@]}" nvidia-cuda-mps-control

  echo
  echo "Create partitions"
  echo "--------------------------------------------------------------------------------"
  for chunk in $PARTITION_CHUNKS; do
    partition="$(create_partition "$chunk")"
    PARTITIONS+=("$partition")
    echo "chunk=$chunk partition=$partition"
  done

  for partition in "${PARTITIONS[@]}"; do
    out_file="$(mktemp)"
    STATIC_OUT_FILES+=("$out_file")
    run_demo_client "$out_file" "$partition"
  done

  sleep 5

  echo
  echo "# echo \"lspart\" | sudo nvidia-cuda-mps-control"
  echo "--------------------------------------------------------------------------------"
  echo "lspart" | "${SUDO[@]}" nvidia-cuda-mps-control

  echo
  echo "# echo ps | sudo nvidia-cuda-mps-control"
  echo "--------------------------------------------------------------------------------"
  echo ps | "${SUDO[@]}" nvidia-cuda-mps-control || true

  wait_clients
  print_client_output "Static SM partitioning client output" "${STATIC_OUT_FILES[@]}"
  stop_mps
}

print_summary() {
  local basic_total
  local static_total
  local ratio
  local i=0
  local chunk
  local basic_iters
  local static_iters

  basic_total="$(sum_files "${BASIC_OUT_FILES[@]}")"
  static_total="$(sum_files "${STATIC_OUT_FILES[@]}")"
  ratio="$(awk -v static="$static_total" -v basic="$basic_total" 'BEGIN { if (basic > 0) printf "%.2fx", static / basic; else printf "n/a" }')"

  echo
  echo "================================================================================"
  echo "Benchmark summary"
  echo "================================================================================"
  printf "%-14s %-10s %14s %14s\n" "client" "chunk" "basic_iters" "static_iters"

  for chunk in $PARTITION_CHUNKS; do
    basic_iters="$(sum_iters "${BASIC_OUT_FILES[$i]}")"
    static_iters="$(sum_iters "${STATIC_OUT_FILES[$i]}")"
    printf "%-14s %-10s %14s %14s\n" "client_$i" "$chunk" "$basic_iters" "$static_iters"
    i=$((i + 1))
  done

  printf "%-25s %14s\n" "basic_total_iters" "$basic_total"
  printf "%-25s %14s\n" "static_total_iters" "$static_total"
  echo "static / basic = $ratio"
}

print_config
run_basic_mps
run_static_partitions
print_summary

if [[ "$WAIT_FAILED" -ne 0 ]]; then
  echo "One or more clients failed. Check the client output above." >&2
  exit 1
fi
