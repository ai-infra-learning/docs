# dynamic-mig-h100

H100 及更新架构上的**动态 MIG**：DRA driver 根据 `ResourceClaim` 在卡上现切 MIG，无需提前 `nvidia-mig-parted`。这是 [`v25.12.0`](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu/releases/tag/v25.12.0) 引入的 alpha 能力。

> **A100 不支持**（无法在运行时自由开关 MIG 模式）。A100 请走 [`../static-mig-a100/README.md`](../static-mig-a100/README.md)。

## 文件

| 文件 | 作用 |
|---|---|
| `mig-multi-profile.yaml` | Namespace + ResourceClaimTemplate（`1g+2g+4g`，同卡）+ Pod |

集群配置与一键部署见 [`../scripts/`](../scripts/README.md)（`setup-dra-env.sh` + `kind-cluster-config.yaml`）。

## 前置条件

- H100 / H200 / B200 等支持运行时切换 MIG 模式的卡。
- Kubernetes `v1.34.2+`，且 `DRAPartitionableDevices` 已在 apiserver / scheduler / kubelet 开启。kind 用户用 [`../scripts/kind-cluster-config.yaml`](../scripts/kind-cluster-config.yaml) 即可（顶层 `featureGates` 会下发到各组件）。
- NVIDIA Driver `v565+`。

一键拉起（含集群 + driver + 动态 MIG）：

```bash
ENABLE_DYNAMIC_MIG=true ../scripts/setup-dra-env.sh
```

集群已就绪、只想装/升级 driver 时（检测到 A100 会失败，动态 MIG 不支持 A100）：

```bash
helm upgrade -i dra-driver-nvidia-gpu \
  oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu --version 0.4.0 \
  --create-namespace --namespace dra-driver-nvidia-gpu \
  --set gpuResourcesEnabledOverride=true \
  --set featureGates.DynamicMIG=true
```

## 步骤

```bash
# 1) 确认 MIG profile 已发布
kubectl get resourceslice -o yaml | grep -i profile | sort -u
#   期望看到 1g.10gb / 2g.20gb / 3g.40gb / 4g.40gb / 7g.80gb

# 2) 现切 1g+2g+4g 并跑 Pod
kubectl apply -f mig-multi-profile.yaml
kubectl get pod -n dynamic-mig-test -w
kubectl logs mig-pod -n dynamic-mig-test    # nvidia-smi -L 应列出 3 个 MIG 设备

# 3) 在节点上对比，确认是动态创建的
nvidia-smi mig -lgi
```

`mig-multi-profile.yaml` 用 `matchAttribute: gpu.nvidia.com/parentUUID` 约束三个 MIG 落在同一张物理卡。只想要单个 profile 时，删掉多余 request、容器 claim 改成 `{ name: mig, request: mig-1g }`。

## 关键约束

- 启用 `DynamicMIG` 后，kubelet plugin **接管整节点 MIG**，启动会拆掉它不认识的 MIG 设备 —— 该节点不能再跑 `mig-parted` 或 GPU Operator MIG Manager。
- `DynamicMIG` 与 `MPSSupport`、`NVMLDeviceHealthCheck`、`PassthroughSupport` **互斥**。
- 一个 claim 会把模板里**所有 request 全部切出来**，Pod 只是按 `claims[].request` 选择性注入；别堆出超过单卡容量的组合，否则 Pod `Pending`。

## 清理

```bash
kubectl delete -f mig-multi-profile.yaml
```
