# scripts

一键部署 DRA 实验环境的脚本，以及它使用的 kind 集群配置。脚本的运行输出为英文，注释为中文。

- `setup-dra-env.sh` — 一键部署脚本（见下）。
- `kind-cluster-config.yaml` — 共享的 kind 集群配置：注入主机 GPU、开启 DRA 与 `DRAPartitionableDevices`，所有路线通用。

## `setup-dra-env.sh`

幂等地把整套实验环境装好，按顺序执行并逐步检查（已存在即 skip）：

1. `docker` — 缺失时在 Ubuntu/Debian 用 `get.docker.com` 安装；macOS 提示手动装 Docker Desktop。
2. NVIDIA Container Toolkit — Ubuntu/Debian 上缺失时用 apt 安装（配置官方源），并自动配置：设为 docker 默认 runtime、开启 `accept-nvidia-visible-devices-as-volume-mounts`，仅在有改动时重启 docker。GPU 驱动（`nvidia-smi`）不自动安装。
3. `kubectl` — 按 `stable.txt` 下载对应平台二进制。
4. `helm` — 用官方 `get-helm-3`。
5. `kind` — 下载 `KIND_VERSION` 对应二进制。
6. kind 集群 — 不存在才创建（用同目录的 `kind-cluster-config.yaml`）。
7. DRA driver — `helm upgrade -i`（天然幂等）。

```bash
./setup-dra-env.sh                          # 基础 DRA 环境
ENABLE_DYNAMIC_MIG=true ./setup-dra-env.sh  # H100+ 额外开启动态 MIG（检测到 A100 会报错退出）
```

可调环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `CLUSTER_NAME` | `dra-demo` | kind 集群名 |
| `KIND_NODE_IMAGE` | `kindest/node:v1.34.3` | kind 节点镜像（K8s 版本需 `1.34.2+`） |
| `KIND_VERSION` | `v0.30.0` | kind 二进制版本 |
| `KIND_CONFIG` | `kind-cluster-config.yaml`（同目录） | kind 集群配置 |
| `DRA_NAMESPACE` | `dra-driver-nvidia-gpu` | DRA driver 命名空间 |
| `DRA_CHART` | `oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu` | helm chart |
| `DRA_VERSION` | `0.4.0` | chart 版本（registry.k8s.io 上的最新稳定版） |
| `ENABLE_DYNAMIC_MIG` | `false` | 是否开启 `featureGates.DynamicMIG`（仅 H100+） |

前置条件：主机需已装 NVIDIA GPU 驱动（`v565+`，提供 `nvidia-smi`）。这一步涉及内核模块与重启，脚本不代装，只检查并提示。NVIDIA Container Toolkit 的安装与配置（含设为 docker 默认 runtime、`accept-nvidia-visible-devices-as-volume-mounts`）在 Ubuntu/Debian 上由脚本自动完成，参考[官方安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

重复执行安全：已装的工具、已存在的集群、已安装的 release 都会跳过或原地升级，不会重复创建。
