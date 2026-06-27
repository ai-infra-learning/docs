# 用 DRA 分配与共享 NVIDIA GPU 的示例

这组文件配合第 3 章，演示如何用 Kubernetes [Dynamic Resource Allocation（DRA）](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/) 分配与共享 NVIDIA GPU —— 既有整卡分配、软件级共享（TimeSlicing / MPS），也有 MIG（静态预切与动态切分）。对应驱动是 [DRA Driver for NVIDIA GPUs](https://dra-driver-nvidia-gpu.sigs.k8s.io/docs/)。

每个子目录都有自己的 `README.md`，包含完整的运行步骤与命令。

## 示例索引

### GPU 分配与共享（非 MIG）


| 目录                                              | 内容                      | 适用        |
| ----------------------------------------------- | ----------------------- | --------- |
| `[quickstart/](quickstart/README.md)`           | 整卡分配、容器间共享、跨 Pod 共享     | 任意支持的 GPU |
| `[mps-timeslicing/](mps-timeslicing/README.md)` | 软件级共享：TimeSlicing 与 MPS | 任意支持的 GPU |




### MIG（硬件级分区）


| 目录                                                          | 内容                              | 适用          |
| ----------------------------------------------------------- | ------------------------------- | ----------- |
| `[static-mig-a100/](static-mig-a100/README.md)`             | `nvidia-mig-parted` 预切 + DRA 分配 | A100、H100 等 |
| `[dynamic-mig-h100/](dynamic-mig-h100/README.md)`           | 按 `ResourceClaim` 动态切 MIG       | H100 及更新    |
| `[partitionable-devices/](partitionable-devices/README.md)` | 动态 MIG 背后的 counter 机制（教学示例）     | 概念理解        |




### 环境与工具


| 目录                              | 内容                                                         |
| ------------------------------- | ---------------------------------------------------------- |
| `[scripts/](scripts/README.md)` | 一键部署 DRA 实验环境（docker / kubectl / helm / kind + DRA driver） |




## 通用要求

- Kubernetes `v1.34.2+`；动态 MIG 在 K8s `1.33–1.35` 还需开启 `DRAPartitionableDevices` feature gate。
- NVIDIA Driver `v565+`，容器运行时开启 CDI。
- 节点上的 GPU 分配只能二选一：传统 device plugin 或 DRA driver，不能同时管同一张卡。
- 动态 MIG 是 `[v25.12.0](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu/releases/tag/v25.12.0)` 引入的 alpha 能力，**只支持 H100+**；A100 无法在运行时自由开关 MIG 模式，走静态路线。

部署环境与各示例的具体命令，见对应子目录的 `README.md`。