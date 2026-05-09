#!/usr/bin/env python3
"""NVIDIA MPS 验证脚本。

这个脚本会启动多个 Python 子进程。每个子进程都会在 GPU 上反复执行
矩阵乘法，用来观察这些 CUDA 进程是否通过 NVIDIA MPS 共享同一张 GPU。
"""

import argparse
import multiprocessing as mp
import os
import time


def worker(worker_id: int, seconds: int, n: int) -> None:
    """单个 worker 进程执行的逻辑。"""

    # torch 放在 worker 内部 import，避免主进程过早初始化 CUDA。
    # 多进程 CUDA 程序建议使用 spawn 启动方式。
    import torch

    # 脚本通过 CUDA_VISIBLE_DEVICES 选择物理 GPU。
    # Python 进程内部固定使用当前可见设备中的 cuda:0。
    device = 0
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    device_uuid = torch.cuda.get_device_properties(device).uuid

    # 创建两个 n x n 的 GPU 矩阵。
    # n 越大，计算量和显存占用越高。
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    # 预热几次，减少首次 CUDA 初始化和 kernel 调度开销对观察结果的影响。
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()

    # 在指定时间内反复执行矩阵乘法，制造持续 GPU workload。
    start = time.perf_counter()
    iters = 0

    while time.perf_counter() - start < seconds or iters == 0:
        _ = a @ b
        iters += 1

        # 每隔一段时间同步一次，避免所有 CUDA work 只是在队列里堆积。
        if iters % 20 == 0:
            torch.cuda.synchronize()

    # 等待当前进程提交的 GPU work 全部完成。
    torch.cuda.synchronize()

    # PID 可以和 `echo ps | nvidia-cuda-mps-control` 的输出对应起来。
    print(
        f"worker={worker_id} pid={os.getpid()} "
        f"iters={iters} cuda:{device}={device_name} uuid={device_uuid}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a simple CUDA workload for NVIDIA MPS testing.",
    )

    # 启动多少个 worker 子进程。
    # 每个 worker 都会创建自己的 CUDA context，用来验证多个进程是否通过 MPS 共享 GPU。
    parser.add_argument("--workers", type=int, default=4)

    # 每个 worker 持续执行 GPU 矩阵乘法的秒数。
    # 建议设置长一点，例如 60 秒，方便观察 nvidia-smi 和 mps-control 输出。
    parser.add_argument("--seconds", type=int, default=60)

    # 矩阵大小。n=1024 表示计算 1024 x 1024 矩阵乘法。
    # 调大可以增加 GPU 负载，但也会增加显存占用。
    parser.add_argument("--n", type=int, default=1024)

    args = parser.parse_args()

    # 使用 spawn 启动方式，避免 fork 已经初始化 CUDA 的进程。
    ctx = mp.get_context("spawn")

    procs = [
        ctx.Process(target=worker, args=(i, args.seconds, args.n))
        for i in range(args.workers)
    ]

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    failed = [proc.exitcode for proc in procs if proc.exitcode != 0]
    if failed:
        raise SystemExit(f"{len(failed)} worker process(es) failed: exit codes {failed}")


if __name__ == "__main__":
    main()
