#!/usr/bin/env bash
# A100 静态 MIG：开启 MIG 模式并按 mig-parted-config.yaml 预切。
# 切好后由 DRA driver 把已存在的 MIG 设备发布成 ResourceSlice，再用 mig.nvidia.com 分配。
#
# 需要 root，且节点上没有其它组件在管 MIG（device plugin / GPU Operator MIG Manager）。
# 用法：sudo -E ./run-static-mig.sh [GPU_INDEX] [CONFIG_NAME]
set -euo pipefail

CONFIG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
GPU_INDEX="${1:-0}"
CONFIG_NAME="${2:-balanced}"

echo "[1/3] Enabling MIG mode on GPU ${GPU_INDEX}"
nvidia-smi -i "${GPU_INDEX}" -mig 1

echo "[2/3] Applying mig-parted layout: ${CONFIG_NAME}"
nvidia-mig-parted apply -f "${CONFIG_DIR}/mig-parted-config.yaml" -c "${CONFIG_NAME}"

echo "[3/3] Current MIG devices:"
nvidia-smi -L
nvidia-smi mig -lgi

echo "Done. Next: kubectl apply -f mig-static.yaml"
