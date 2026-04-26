import contextlib

import torch
from torch.cuda import Stream


assert torch.cuda.is_available(), "A CUDA device is required to run this script"

s = Stream()

torch.manual_seed(42)
t1_cpu_pinned = torch.randn(1024**2 * 5, pin_memory=True)
t2_cpu_paged = torch.randn(1024**2 * 5, pin_memory=False)
t3_cuda = torch.randn(1024**2 * 5, device="cuda:0")

device = torch.device("cuda", torch.cuda.current_device())


# The function we want to profile
def inner(pinned: bool, streamed: bool):
    with torch.cuda.stream(s) if streamed else contextlib.nullcontext():
        if pinned:
            t1_cuda = t1_cpu_pinned.to(device, non_blocking=True)
        else:
            t2_cuda = t2_cpu_paged.to(device, non_blocking=True)
        t_star_cuda_h2d_event = s.record_event()
    # This operation can be executed during the CPU to GPU copy if and only if the tensor is pinned and the copy is
    #  done in the other stream
    t3_cuda_mul = t3_cuda * t3_cuda * t3_cuda
    t3_cuda_h2d_event = torch.cuda.current_stream().record_event()
    t_star_cuda_h2d_event.synchronize()
    t3_cuda_h2d_event.synchronize()


# Our profiler: profiles the `inner` function and stores the results in a .json file
def benchmark_with_profiler(
    pinned,
    streamed,
) -> None:
    torch._C._profiler._set_cuda_sync_enabled_val(True)
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
    prof.export_chrome_trace(f"trace_streamed{int(streamed)}_pinned{int(pinned)}.json")


benchmark_with_profiler(streamed=False, pinned=False)
benchmark_with_profiler(streamed=False, pinned=True)
benchmark_with_profiler(streamed=True, pinned=False)
benchmark_with_profiler(streamed=True, pinned=True)
