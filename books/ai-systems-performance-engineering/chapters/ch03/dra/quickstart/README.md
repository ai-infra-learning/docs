# quickstart

DRA 分配整卡 GPU 与共享 GPU 的入门示例。每个文件自带 Namespace + ResourceClaim(Template) + Pod，顶部有 ASCII 拓扑图。

## 前置条件

- 已部署 DRA 环境（见 [`../scripts/setup-dra-env.sh`](../scripts/README.md)），且 `gpu.nvidia.com` DeviceClass 已注册：

```bash
kubectl get deviceclass | grep gpu.nvidia.com
kubectl get resourceslice -o wide        # 应能看到 gpu.nvidia.com 设备
```

## 三个示例

| 文件                                  | 拓扑                             | 关键点                                                         |
| ----------------------------------- | ------------------------------ | ----------------------------------------------------------- |
| `two-pods-one-gpu-each.yaml`        | 2 个 Pod，各独占 1 张 GPU            | 用 `ResourceClaimTemplate`，每个 Pod 各生成独立 claim → 不同 GPU       |
| `two-containers-share-one-gpu.yaml` | 1 个 Pod、2 容器共享 1 张 GPU         | 两容器引用同一个 claim → 同一张 GPU                                    |
| `two-pods-share-one-gpu.yaml`       | 2 个 Pod 共享 1 张 GPU             | 用共享的 `ResourceClaim`（非 Template），两 Pod 引用同名 claim → 同一张 GPU |

TimeSlicing / MPS 的软件级共享示例已移到 [`../mps-timeslicing/`](../mps-timeslicing/README.md)。




## 运行

```bash
kubectl apply -f two-pods-one-gpu-each.yaml          # 换成任意一个示例

# 看每个容器实际拿到的 GPU UUID
kubectl logs -n gpu-test1 pod1 --all-containers --prefix
kubectl logs -n gpu-test1 pod2 --all-containers --prefix
```

`two-pods-one-gpu-each.yaml` 用 `gpu-test1` 命名空间，其余分别是 `gpu-test2`、`gpu-test3`。共享类示例里多个容器/Pod 打印的 GPU UUID 应当相同，独占类则不同。

## 清理

```bash
kubectl delete -f two-pods-one-gpu-each.yaml
# 或按命名空间：kubectl delete namespace gpu-test1
```



