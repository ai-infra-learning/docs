#!/usr/bin/env bash
set -euo pipefail

# 对比不开 MPS 与开启 Basic MPS 时，多个 CUDA 进程在同一张 GPU 上的吞吐。
# 当前 demo 固定运行时间，因此比较的是所有 worker 的 iters 总和。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH03_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GPU_INDEX="${GPU_INDEX:-0}"
WORKERS="${WORKERS:-4}"
SECONDS="${SECONDS:-60}"
N="${N:-1024}"

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

BASELINE_OUT="$(mktemp)"
MPS_OUT="$(mktemp)"
MPS_STARTED=0

cleanup() {
  if [[ "$MPS_STARTED" -eq 1 ]]; then
    # 关闭本次启动的 MPS control daemon。
    echo quit | "${SUDO[@]}" nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi

  # 恢复 GPU 默认 compute mode，避免影响后续任务。
  "${SUDO[@]}" nvidia-smi -i "$GPU_INDEX" -c DEFAULT >/dev/null 2>&1 || true

  rm -f "$BASELINE_OUT" "$MPS_OUT"
}
trap cleanup EXIT

sum_iters() {
  sed -n 's/.*iters=\([0-9][0-9]*\).*/\1/p' "$1" | awk '{sum += $1} END {print sum + 0}'
}

print_config() {
  echo "GPU_INDEX=$GPU_INDEX"
  echo "WORKERS=$WORKERS"
  echo "SECONDS=$SECONDS"
  echo "N=$N"
}

run_without_mps() {
  echo "================================================================================"
  echo "Running without MPS"
  echo "================================================================================"

  # baseline 使用 DEFAULT compute mode，允许多个进程直接创建各自的 CUDA context。
  "${SUDO[@]}" nvidia-smi -i "$GPU_INDEX" -c DEFAULT >/dev/null

  cd "$CH03_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    uv run python mps/mps_demo.py \
      --workers "$WORKERS" \
      --seconds "$SECONDS" \
      --n "$N" | tee "$BASELINE_OUT"
}

run_with_mps() {
  echo "================================================================================"
  echo "Running with MPS"
  echo "================================================================================"

  # EXCLUSIVE_PROCESS 让这张 GPU 同一时间只允许一个 CUDA process 直接创建 context。
  # MPS 场景中，这个 process 通常是 nvidia-cuda-mps-server。
  "${SUDO[@]}" nvidia-smi -i "$GPU_INDEX" -c EXCLUSIVE_PROCESS >/dev/null

  # 启动 MPS control daemon，并限制它只管理指定 GPU。
  "${SUDO[@]}" env CUDA_VISIBLE_DEVICES="$GPU_INDEX" nvidia-cuda-mps-control -d
  MPS_STARTED=1

  cd "$CH03_DIR"

  # client 侧清理 CUDA_VISIBLE_DEVICES，避免 device index 重映射混淆。
  env -u CUDA_VISIBLE_DEVICES \
    uv run python mps/mps_demo.py \
      --workers "$WORKERS" \
      --seconds "$SECONDS" \
      --n "$N" >"$MPS_OUT" 2>&1 &

  local demo_pid=$!

  # 给 worker 留出 CUDA 初始化时间，然后查看 MPS client 和 server 状态。
  sleep 5

  echo "--------------------------------------------------------------------------------"
  echo "MPS clients"
  echo "--------------------------------------------------------------------------------"
  echo ps | "${SUDO[@]}" nvidia-cuda-mps-control || true

  echo "--------------------------------------------------------------------------------"
  echo "nvidia-smi"
  echo "--------------------------------------------------------------------------------"
  nvidia-smi -i "$GPU_INDEX" || true

  wait "$demo_pid"
  cat "$MPS_OUT"
}

print_summary() {
  local baseline_iters
  local mps_iters
  local ratio

  baseline_iters="$(sum_iters "$BASELINE_OUT")"
  mps_iters="$(sum_iters "$MPS_OUT")"
  ratio="$(awk -v mps="$mps_iters" -v base="$baseline_iters" 'BEGIN { if (base > 0) printf "%.2fx", mps / base; else printf "n/a" }')"

  echo "================================================================================"
  echo "Benchmark summary"
  echo "================================================================================"
  printf "%-16s %12s\n" "mode" "total_iters"
  printf "%-16s %12s\n" "without_mps" "$baseline_iters"
  printf "%-16s %12s\n" "with_mps" "$mps_iters"
  echo "mps / without_mps = $ratio"
  echo
  echo "Note: higher total_iters is better. MPS is not guaranteed to improve every workload."
}

print_config
run_without_mps
run_with_mps
print_summary
