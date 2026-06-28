# partitionable-devices

动态 MIG 背后的通用机制示例：DRA 用 **counter（计数器）** 表达"一个设备可以切成多个互斥的子设备"。对应 Kubernetes [KEP-4815](https://github.com/kubernetes/enhancements/tree/master/keps/sig-scheduling/4815-dra-partitionable-devices)（`DRAPartitionableDevices` feature gate）。

这是**教学示例，和厂商无关**，用的是假的 `gizmo.example.com` driver；真实 NVIDIA GPU 由 DRA driver 自动发布对应的 `ResourceSlice`，无需手写。

## 文件

| 文件 | 作用 |
|---|---|
| `counterset-slice.yaml` | 用 `sharedCounters` 声明资源池总量（`counterset-1` = 40Gi/4cpu） |
| `devices-slice.yaml` | 每个 device 用 `consumesCounters` 从池里扣额度 |

两个 `ResourceSlice` 属于同一个 pool（`gizmo-pool`，`resourceSliceCount: 2`）。`sharedCounters` 与 `devices` 必须分处不同 slice（Kubernetes 1.35+ 强制）。

## sharedCounters ↔ consumesCounters

`sharedCounters` 是一块共享资源池（总量），每个 device 用 `consumesCounters` 声明自己从池里扣多少。多个 device 重叠在同一个池上，调度器靠记账保证“合计不超过池”。

```text
              sharedCounters  (资源池 / 总量)
              ┌────────────────────────┐
              │ counterset-1           │
              │   memory: 40Gi         │
              │   cpus:   4            │
              └───────────┬────────────┘
                          ▲
        consumesCounters: │ 每个 device 都从 counterset-1 扣额度
          ┌───────────────┼────────────────┐
          │               │                │
   ┌──────┴──────┐ ┌──────┴──────┐ ┌───────┴─────┐
   │large-device │ │small-device1│ │small-device2│
   │ 40Gi / 4cpu │ │ 20Gi / 2cpu │ │ 20Gi / 2cpu │
   └─────────────┘ └─────────────┘ └─────────────┘
```

记账（可用 = 池 − Σ 已分配 device 的 consumesCounters）：

```text
方案 A:  large-device              → 40Gi/4cpu，占满池 → 两个 small 都不可再分
方案 B:  small-device1 + small-device2 = 40Gi/4cpu → 占满池 → large 不可分
非法  :  large + 任一 small = 60Gi/6cpu > 40Gi/4cpu → 超分，调度器拒绝
```

所以这三个 device 是**互斥重叠**的：要么 1 个 large，要么 2 个 small。换成真实 GPU，`counterset-1` 就是显存/算力池，`large/small` 就是不同大小的 MIG（见下方映射表）。

## 机制要点

- `large-device`（吃满 `counterset-1`）与两个 `small-device`（各占一半）都消耗同一个 CounterSet，因此**互斥**：要么 1 个 large，要么 2 个 small。
- 调度器对每个 CounterSet 记账，剩余额度不足就不分配 —— 由此保证不超分。
- 这正是动态 MIG 的底层：一张 GPU 的算力/显存建模成共享 counter 池，所有可能的 MIG 摆放作为重叠 device 发布，调度器挑出合法组合，kubelet plugin 再真正切。

映射关系：

| 本示例 | 真实 MIG |
|---|---|
| `counterset-1`（资源池） | 一张 GPU 的 compute slice + 显存池 |
| `large-device` | 大 MIG，如 `4g.40gb` |
| `small-device-1/2` | 小 MIG，如 `2g.20gb` |
| 1 大 ⊻ 2 小（互斥） | MIG 几何约束（slice 总数 ≤ 7） |

## 运行（仅用于观察调度记账）

需要集群已开 `DRAPartitionableDevices`。这两个文件不对应真实硬件，apply 后只是让调度器看到这些"可切设备"，不会真正分到 GPU。

```bash
kubectl apply -f counterset-slice.yaml -f devices-slice.yaml
kubectl get resourceslice
kubectl describe resourceslice devices-slice    # 查看 consumesCounters 记账
kubectl delete -f counterset-slice.yaml -f devices-slice.yaml
```

读懂这套机制后，再看 [`../dynamic-mig-h100/README.md`](../dynamic-mig-h100/README.md) 就清楚动态 MIG 是怎么落地的。
