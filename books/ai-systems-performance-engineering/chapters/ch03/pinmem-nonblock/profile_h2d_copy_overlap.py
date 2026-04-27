import contextlib

import torch
from torch.cuda import Stream


assert torch.cuda.is_available(), "A CUDA device is required to run this script"

# 这条 stream 用来观察“把 H2D copy 放到独立 stream”之后，能否和默认
# stream 上的计算重叠。
s = Stream()

torch.manual_seed(42)

# 两份 CPU tensor 大小相同，只差在 host memory 类型：
# - t1_cpu_pinned 位于 pinned/page-locked memory，更适合异步 DMA
# - t2_cpu_paged 位于普通 pageable memory，H2D copy 前可能需要 staging
t1_cpu_pinned = torch.randn(1024**2 * 5, pin_memory=True)
t2_cpu_paged = torch.randn(1024**2 * 5, pin_memory=False)

# t3_cuda 一开始就在 GPU 上。后面的乘法计算使用它，而不是使用刚刚
# copy 到 GPU 的 t1/t2。这样才能观察“拷贝一份数据，同时计算另一份
# 已经在显存里的数据”。
t3_cuda = torch.randn(1024**2 * 5, device="cuda:0")

device = torch.device("cuda", torch.cuda.current_device())


# 被 profiler 观察的核心函数：
# - pinned 控制 H2D copy 的源 tensor 是否在 pinned memory 中
# - streamed 控制 H2D copy 是否提交到独立 CUDA stream
def inner(pinned: bool, streamed: bool):
    with torch.cuda.stream(s) if streamed else contextlib.nullcontext():
        if pinned:
            t1_cuda = t1_cpu_pinned.to(device, non_blocking=True)
        else:
            t2_cuda = t2_cpu_paged.to(device, non_blocking=True)

        # 记录 H2D copy 所在 stream 的事件。后面 synchronize 用它保证
        # 本轮 copy 完成，避免 profiler step 之间互相串扰。
        t_star_cuda_h2d_event = s.record_event()

    # 独立的 GPU 计算任务。它不依赖刚 copy 上来的 tensor；如果条件满足，
    # 这段乘法 kernel 可以和 H2D copy 在 GPU 侧重叠。
    t3_cuda_mul = t3_cuda * t3_cuda * t3_cuda

    # 记录默认 stream 上乘法计算之后的事件。
    t3_cuda_h2d_event = torch.cuda.current_stream().record_event()

    # 等待 copy 和 compute 都完成，确保 profiler 能完整记录本轮 step。
    t_star_cuda_h2d_event.synchronize()
    t3_cuda_h2d_event.synchronize()


# 运行一次 profiler，并把 Chrome trace 导出为 JSON。
def benchmark_with_profiler(
    pinned,
    streamed,
) -> None:
    # 让 profiler 记录 CUDA 同步事件，便于在 trace 里看到 Event Sync。
    torch._C._profiler._set_cuda_sync_enabled_val(True)

    # wait/warmup 不计入最终 active trace；active=2 会记录两个 step。
    wait, warmup, active = 1, 1, 2
    num_steps = wait + warmup + active
    rank = 0
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
        ),
    ) as prof:
        for step_idx in range(1, num_steps + 1):
            inner(streamed=streamed, pinned=pinned)
            if rank is None or rank == 0:
                prof.step()

    # 文件名编码两个开关，方便和四张 profiler 截图一一对应。
    prof.export_chrome_trace(f"trace_streamed{int(streamed)}_pinned{int(pinned)}.json")


# 生成四组对照 trace：
# 1. pageable + 默认 stream
# 2. pinned + 默认 stream
# 3. pageable + 独立 stream
# 4. pinned + 独立 stream
benchmark_with_profiler(streamed=False, pinned=False)
benchmark_with_profiler(streamed=False, pinned=True)
benchmark_with_profiler(streamed=True, pinned=False)
benchmark_with_profiler(streamed=True, pinned=True)
