#!/usr/bin/env bash
# 一键部署 DRA 实验环境（幂等）：
#   docker → NVIDIA Container Toolkit → kubectl → helm → kind → kind 集群 → DRA driver。
# 每一步先检查，已存在则 skip，可重复执行而结果一致。
#
# 主要面向 Ubuntu/Debian GPU 主机：
#   - docker / kubectl / helm / kind / NVIDIA Container Toolkit 缺失时自动安装（apt 或官方二进制）。
#   - Container Toolkit 会自动配置：设为 docker 默认 runtime + 开启 volume-mount 注入 GPU，仅在有改动时重启 docker。
#   - GPU 驱动（nvidia-smi）不自动安装：涉及内核模块与重启，需先手动装好。
# 其它发行版 / macOS：相应步骤跳过并给出手动安装提示。
#
# 可用环境变量覆盖默认值：
#   CLUSTER_NAME, KIND_NODE_IMAGE, KIND_VERSION, KIND_CONFIG,
#   DRA_NAMESPACE, DRA_CHART, DRA_VERSION, ENABLE_DYNAMIC_MIG
#
# 用法：
#   ./setup-dra-env.sh                       # 基础 DRA 环境
#   ENABLE_DYNAMIC_MIG=true ./setup-dra-env.sh   # H100+ 额外开启动态 MIG
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

CLUSTER_NAME="${CLUSTER_NAME:-dra-demo}"
KIND_NODE_IMAGE="${KIND_NODE_IMAGE:-kindest/node:v1.34.3}"
KIND_VERSION="${KIND_VERSION:-v0.30.0}"
KIND_CONFIG="${KIND_CONFIG:-${SCRIPT_DIR}/kind-cluster-config.yaml}"
DRA_NAMESPACE="${DRA_NAMESPACE:-dra-driver-nvidia-gpu}"
DRA_CHART="${DRA_CHART:-oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu}"
DRA_VERSION="${DRA_VERSION:-0.4.0}"
ENABLE_DYNAMIC_MIG="${ENABLE_DYNAMIC_MIG:-false}"

log()  { printf '\033[0;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[warn]\033[0m %s\n'  "$*" >&2; }
err()  { printf '\033[0;31m[error]\033[0m %s\n' "$*" >&2; }

OS="$(uname -s)"
OS_LOWER="$(echo "${OS}" | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64|amd64)  ARCH=amd64 ;;
  aarch64|arm64) ARCH=arm64 ;;
esac

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

ensure_docker() {
  if command -v docker >/dev/null 2>&1; then
    log "docker already installed: $(docker --version)"; return
  fi
  if [ "${OS}" != "Linux" ]; then
    err "docker not found. On ${OS}, install Docker Desktop manually and re-run."; exit 1
  fi
  log "Installing docker ..."
  curl -fsSL https://get.docker.com | ${SUDO} sh
  ${SUDO} systemctl enable --now docker || true
}

ensure_kubectl() {
  if command -v kubectl >/dev/null 2>&1; then
    log "kubectl already installed: $(kubectl version --client 2>/dev/null | head -n1)"; return
  fi
  log "Installing kubectl ..."
  local ver; ver="$(curl -fsSL https://dl.k8s.io/release/stable.txt)"
  curl -fsSLo /tmp/kubectl "https://dl.k8s.io/release/${ver}/bin/${OS_LOWER}/${ARCH}/kubectl"
  ${SUDO} install -m 0755 /tmp/kubectl /usr/local/bin/kubectl
}

ensure_helm() {
  if command -v helm >/dev/null 2>&1; then
    log "helm already installed: $(helm version --short)"; return
  fi
  log "Installing helm ..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
}

ensure_kind() {
  if command -v kind >/dev/null 2>&1; then
    log "kind already installed: $(kind version)"; return
  fi
  log "Installing kind ${KIND_VERSION} ..."
  curl -fsSLo /tmp/kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-${OS_LOWER}-${ARCH}"
  ${SUDO} install -m 0755 /tmp/kind /usr/local/bin/kind
}

ensure_container_toolkit() {
  # nvidia-smi 来自 GPU 驱动，不由本脚本安装（涉及内核模块与重启，需先手动装好）。
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPUs:"; nvidia-smi -L || true
  else
    warn "nvidia-smi not found. Install the NVIDIA GPU driver first (not handled by this script)."
  fi

  # NVIDIA Container Toolkit 只在 Ubuntu/Debian(apt) 上自动安装与配置。
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "Non-apt OS: install/configure the NVIDIA Container Toolkit manually."
    warn "See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    return
  fi

  local need_restart=false

  # 1) 安装 nvidia-container-toolkit（缺失才装，配置官方 apt 源）
  if command -v nvidia-ctk >/dev/null 2>&1; then
    log "NVIDIA Container Toolkit already installed: $(nvidia-ctk --version 2>/dev/null | head -n1)"
  else
    log "Installing NVIDIA Container Toolkit (apt) ..."
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y --no-install-recommends ca-certificates curl gnupg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | ${SUDO} gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y nvidia-container-toolkit
    need_restart=true
  fi

  # 2) 把 NVIDIA runtime 配为 docker 默认 runtime（幂等）
  if grep -q '"default-runtime"[[:space:]]*:[[:space:]]*"nvidia"' /etc/docker/daemon.json 2>/dev/null; then
    log "docker default-runtime already nvidia, skipping."
  else
    log "Configuring NVIDIA runtime as docker default ..."
    ${SUDO} nvidia-ctk runtime configure --runtime=docker --set-as-default
    need_restart=true
  fi

  # 3) 开启 volume-mount 注入 GPU（kind 透传 GPU 需要；幂等）
  local cfg=/etc/nvidia-container-runtime/config.toml
  if [ -f "${cfg}" ] && grep -q 'accept-nvidia-visible-devices-as-volume-mounts[[:space:]]*=[[:space:]]*true' "${cfg}"; then
    log "accept-nvidia-visible-devices-as-volume-mounts already true, skipping."
  else
    log "Setting accept-nvidia-visible-devices-as-volume-mounts=true ..."
    ${SUDO} nvidia-ctk config --in-place --set accept-nvidia-visible-devices-as-volume-mounts=true
    need_restart=true
  fi

  # 4) 仅在有改动时重启 docker（避免打断已存在的 kind 集群）
  if [ "${need_restart}" = "true" ]; then
    log "Restarting docker to apply NVIDIA runtime changes ..."
    ${SUDO} systemctl restart docker || true
  fi
}

ensure_cluster() {
  if kind get clusters 2>/dev/null | grep -qx "${CLUSTER_NAME}"; then
    log "kind cluster ${CLUSTER_NAME} already exists, skipping."; return
  fi
  log "Creating kind cluster ${CLUSTER_NAME} (image=${KIND_NODE_IMAGE}, config=${KIND_CONFIG}) ..."
  kind create cluster --name "${CLUSTER_NAME}" --image "${KIND_NODE_IMAGE}" --config "${KIND_CONFIG}"
}

ensure_dra() {
  # 给 worker 打 NFD 风格的 GPU label（kind 配置已带，兜底再打一次）。
  kubectl label node -l node-role.x-k8s.io/worker --overwrite nvidia.com/gpu.present=true 2>/dev/null || true

  # TimeSlicingSettings 不与任何特性互斥，始终开启（共享策略示例 mps-timeslicing/ 需要它）。
  # MPSSupport 与 DynamicMIG 互斥，只在非动态 MIG 时开启。
  local args=(--set gpuResourcesEnabledOverride=true --set featureGates.TimeSlicingSettings=true)
  if [ "${ENABLE_DYNAMIC_MIG}" = "true" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | grep -qi "A100"; then
      err "ENABLE_DYNAMIC_MIG=true but an A100 was detected; dynamic MIG is not supported on A100. Use the static route instead."; exit 1
    fi
    args+=(--set featureGates.DynamicMIG=true)
    log "Enabling featureGates: TimeSlicingSettings, DynamicMIG"
    warn "DynamicMIG is mutually exclusive with MPSSupport; MPSSupport will NOT be enabled."
  else
    args+=(--set featureGates.MPSSupport=true)
    log "Enabling featureGates: TimeSlicingSettings, MPSSupport"
  fi

  log "Installing/upgrading DRA driver (helm upgrade -i, idempotent) ..."
  helm upgrade -i dra-driver-nvidia-gpu "${DRA_CHART}" \
    --version "${DRA_VERSION}" \
    --create-namespace --namespace "${DRA_NAMESPACE}" \
    "${args[@]}" --wait
}

verify() {
  log "Verifying DRA environment:"
  kubectl get pod -n "${DRA_NAMESPACE}" || true
  kubectl get deviceclass 2>/dev/null || true
  kubectl get resourceslice -o wide 2>/dev/null || true
}

main() {
  log "OS=${OS} ARCH=${ARCH} CLUSTER=${CLUSTER_NAME} DYNAMIC_MIG=${ENABLE_DYNAMIC_MIG}"
  ensure_docker
  ensure_container_toolkit
  ensure_kubectl
  ensure_helm
  ensure_kind
  ensure_cluster
  ensure_dra
  verify
  log "Done. Next, try the examples under quickstart/, static-mig-a100/, or dynamic-mig-h100/."
}

main "$@"
