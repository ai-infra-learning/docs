# static-mig-a100

A100（或任意支持 MIG 的卡）上的 MIG 用法：先用 `nvidia-mig-parted` 在节点上**预切**好 MIG，再由 DRA driver 把已存在的 MIG 设备发布出来供 `ResourceClaim` 分配。DRA 在这条路线下**不现切**，只做分配。

```text
nvidia-mig-parted 预切  →  DRA driver 发布 ResourceSlice  →  ResourceClaim 选已存在的 MIG
```

## 文件


| 文件                       | 作用                                                                 |
| ------------------------ | ------------------------------------------------------------------ |
| `mig-parted-config.yaml` | MIG 布局：`balanced` = `1g.10gb×2 + 2g.20gb + 3g.40gb`（= 7g，切满 GPU 0） |
| `run-static-mig.sh`      | 开 MIG 模式 + 应用上面的布局                                                 |
| `mig-static.yaml`        | Namespace + ResourceClaimTemplate（选 `1g.10gb`）+ Pod                |


## 前置条件

- 已部署 DRA 环境（见 `[../scripts/setup-dra-env.sh](../scripts/README.md)`），`mig.nvidia.com` DeviceClass 已注册。
- 节点上没有其它组件在管 MIG（传统 device plugin、GPU Operator MIG Manager）。
- 在节点本机执行 `run-static-mig.sh`（需要 root，直接操作 GPU）。



## 步骤

```bash
# 1) 在节点上预切 MIG（默认 GPU 0、balanced 布局）
sudo -E ./run-static-mig.sh
# 也可指定：sudo -E ./run-static-mig.sh <GPU_INDEX> <CONFIG_NAME>

# 2) 确认 MIG profile 被 DRA 发布出来
kubectl get resourceslice -o yaml | grep -i profile | sort -u
#   期望看到 1g.10gb / 2g.20gb / 3g.40gb

# 3) 分配并验证
kubectl apply -f mig-static.yaml
kubectl get pod -n static-mig-test -w
kubectl logs mig-pod -n static-mig-test   # nvidia-smi -L 应显示一个 MIG 设备
```

`mig-static.yaml` 里的 selector 是 `profile == '1g.10gb'`，要和 `mig-parted` 实际切出来的 profile 名一致；改尺寸就改这个表达式。

## 关闭 MIG

```bash
sudo nvidia-smi -i 0 -mig 0
```

