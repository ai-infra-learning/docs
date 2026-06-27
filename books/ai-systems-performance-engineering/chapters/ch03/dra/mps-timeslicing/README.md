# mps-timeslicing

在同一张物理 GPU 上对多个容器做软件级共享的两种策略：**TimeSlicing**（时间片轮流）和 **MPS**（Multi-Process Service，多进程并发）。通过 DRA 的 `GpuConfig` opaque config 表达，和 MIG 的硬件级隔离互补。

`timeslicing-and-mps-sharing.yaml` 一个文件含 Namespace + ResourceClaimTemplate + Pod：

```text
┌─ Pod ───────────────────────────────────────────┐
│ ts-ctr0  ts-ctr1   →  ts-gpu  (TimeSlicing)       │
│ mps-ctr0 mps-ctr1  →  mps-gpu (MPS)               │
└───────────────────────────────────────────────────┘
```

`ts-gpu` 与 `mps-gpu` 是模板里的两个 request（两张不同的 GPU），容器用 `claims[].request` 选择走哪种共享。

## 前置条件

这两种策略是 DRA driver 的 alpha 能力，**必须开启对应 feature gate**，否则 kubelet 在 `NodePrepareResources` 阶段会拒绝执行 `GpuConfig`：


| 策略                                                     | 需要的 feature gate      |
| ------------------------------------------------------ | --------------------- |
| `sharing.strategy: TimeSlicing`（含 `timeSlicingConfig`） | `TimeSlicingSettings` |
| `sharing.strategy: MPS`                                | `MPSSupport`          |


用一键脚本部署时默认就会开这两个（非动态 MIG 模式下）：

```bash
../scripts/setup-dra-env.sh
```

或手动升级 driver 时显式打开：

```bash
helm upgrade -i dra-driver-nvidia-gpu \
  oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu --version 0.4.1 \
  --create-namespace --namespace dra-driver-nvidia-gpu \
  --set gpuResourcesEnabledOverride=true \
  --set featureGates.TimeSlicingSettings=true \
  --set featureGates.MPSSupport=true
```

> `MPSSupport` 与 `DynamicMIG` **互斥**，不能同时开。要跑这个 MPS 示例，节点就不能同时启用动态 MIG。



## 运行

```bash
kubectl apply -f timeslicing-and-mps-sharing.yaml
kubectl get pod -n gpu-test5 -w

# 四个容器都在跑 nbody benchmark
kubectl logs -n gpu-test5 pod0 -c ts-ctr0
kubectl logs -n gpu-test5 pod0 -c mps-ctr0
```

镜像是 nbody CUDA sample，会持续跑负载，便于观察两种共享下的行为差异。

## TimeSlicing 与 MPS 的区别


|      | TimeSlicing                        | MPS                                                                        |
| ---- | ---------------------------------- | -------------------------------------------------------------------------- |
| 执行方式 | 多个进程**轮流**占用 GPU（时间片切换）            | 多个进程**并发**提交到同一 GPU                                                        |
| 隔离   | 无内存/算力隔离，靠时间片                      | 可限制每进程线程占比、显存上限                                                            |
| 本例配置 | `timeSlicingConfig.interval: Long` | `defaultActiveThreadPercentage: 50`、`defaultPinnedDeviceMemoryLimit: 10Gi` |
| 适用   | 低优先级/突发负载共享                        | 多个中小负载并发、需一定 QoS                                                           |


与 MIG 的区别：MIG 是**硬件级**分区（独立 SM、显存、L2），TimeSlicing/MPS 是**软件级**共享，无硬件隔离。

## 清理

```bash
kubectl delete -f timeslicing-and-mps-sharing.yaml
```

