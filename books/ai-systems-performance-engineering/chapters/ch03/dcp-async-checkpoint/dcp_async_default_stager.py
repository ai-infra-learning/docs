"""使用 PyTorch DCP async_save() + DefaultStager 的 checkpoint 示例。

在 books/ai-systems-performance-engineering/chapters/ch03 目录下运行：

    uv run python dcp-async-checkpoint/dcp_async_default_stager.py

这个脚本在 dcp.async_save() 的基础上使用 DefaultStager：
- 单节点多 GPU，一张 GPU 对应一个进程。
- 模型用 FSDP 分片，因此 checkpoint 也按 rank 保存为分布式分片。
- DefaultStager 会把 state dict 创建和 GPU -> CPU staging 放到后台线程。

普通 async_save() 主要异步化 checkpoint 写入；DefaultStager 进一步把
staging 阶段也从训练主线程挪出去。代价是额外 CPU 线程和 CPU memory 压力。
"""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.staging import DefaultStager
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard


@dataclass(frozen=True)
class RunConfig:
    # 四个脚本使用同一组模型和训练参数，保证 checkpoint 大小可比。
    steps: int
    batch_size: int
    hidden_size: int
    layers: int
    checkpoint_dir: str
    master_port: str


class AppState(Stateful):
    """把 model 和 optimizer 包装成 DCP 可保存的 Stateful 对象。

    DCP 会调用 state_dict()/load_state_dict()。这里通过
    get_state_dict()/set_state_dict() 让 FSDP 自动产出和恢复分布式
    state dict。
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # 对 FSDP 模型来说，这里返回的是适合 DCP 保存的分布式 state dict。
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model,
            self.optimizer,
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        # 示例主要测保存时间，但实现 load_state_dict 可以让 AppState 完整符合
        # Stateful 协议，方便扩展成恢复示例。
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class ToyModel(nn.Module):
    """用于制造较大 checkpoint 的简单 MLP。"""

    def __init__(self, hidden_size: int, layers: int):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        # 保留 net1 -> relu -> net2 的主结构，中间按 layers 放大隐藏层数量。
        self.net1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(layers - 1)
        )
        self.net2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        x = self.relu(self.net1(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.net2(x)


def setup(rank: int, world_size: int, config: RunConfig) -> None:
    """初始化单节点多进程分布式环境。

    MASTER_ADDR=localhost 表示这个示例只覆盖单节点；这里使用 Gloo backend，
    和 PyTorch DCP async checkpoint 示例保持一致。
    """

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", config.master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # 每个 rank 绑定到同编号 CUDA device。
    torch.cuda.set_device(rank)


def cleanup() -> None:
    """销毁当前进程的 process group。"""

    dist.destroy_process_group()


def directory_size(path: str) -> int:
    """统计 DCP checkpoint 目录大小。"""

    root = Path(path)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def format_bytes(num_bytes: int) -> str:
    """把 checkpoint 目录大小格式化成人容易读的单位。"""

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def remove_old_checkpoints(rank: int, config: RunConfig) -> None:
    """清理上一次运行留下的 checkpoint 目录。

    只有 rank 0 负责删除目录；dist.barrier() 保证其他 rank 不会在删除完成前
    开始写新的 checkpoint。
    """

    if rank == 0:
        for step in range(config.steps):
            shutil.rmtree(f"{config.checkpoint_dir}_step{step}", ignore_errors=True)
    dist.barrier()


def log_staging_done(
    rank: int,
    checkpoint_step: int,
    submit_time: float,
    wait_start: float,
) -> None:
    """打印 staging 阶段的耗时。

    staging 表示从提交 async_save() 到 state dict 创建和 GPU -> CPU copy 完成
    的时间。staging_wait 表示训练 loop 在这里实际等待的时间。
    """

    wait_seconds = time.perf_counter() - wait_start
    staging_seconds = time.perf_counter() - submit_time
    if rank == 0:
        print(
            f"[dcp_async_default_stager] step={checkpoint_step} "
            f"staging={staging_seconds:.3f}s staging_wait={wait_seconds:.3f}s",
            flush=True,
        )


def log_upload_done(
    rank: int,
    checkpoint_step: int,
    checkpoint_id: str,
    submit_time: float,
    wait_start: float,
) -> None:
    """打印 upload 阶段完成后的总耗时和目录大小。

    total 表示从提交 async_save() 到 checkpoint 完成的总 wall time。
    upload_wait 表示训练 loop 等待 checkpoint 写入完成的实际时间。
    """

    wait_seconds = time.perf_counter() - wait_start
    total_seconds = time.perf_counter() - submit_time
    if rank == 0:
        size = format_bytes(directory_size(checkpoint_id))
        print(
            f"[dcp_async_default_stager] step={checkpoint_step} "
            f"total={total_seconds:.3f}s upload_wait={wait_seconds:.3f}s "
            f"size={size}",
            flush=True,
        )


def run(rank: int, world_size: int, config: RunConfig) -> None:
    setup(rank, world_size, config)
    remove_old_checkpoints(rank, config)
    device = torch.device("cuda", rank)

    # 固定随机种子，保证每次运行模型初始化一致。
    torch.manual_seed(0)
    model = ToyModel(config.hidden_size, config.layers).to(device)
    # fully_shard 会把模型参数分片到多个 rank 上，模拟 FSDP 训练场景。
    model = fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # DefaultStager 返回的 future 拆成 staging_completion 和 upload_completion。
    checkpoint_future = None
    checkpoint_step = -1
    checkpoint_id = ""
    checkpoint_submit_time = 0.0

    for step in range(config.steps):
        # 做一个极简训练 step，让 optimizer state 被初始化并持续更新。
        optimizer.zero_grad(set_to_none=True)
        x = torch.rand(config.batch_size, config.hidden_size, device=device)
        model(x).sum().backward()

        # 上一个 checkpoint 可能还在把 GPU 状态 staging 到 CPU memory。
        # 必须在 optimizer.step() 之前等待 staging 完成，否则 optimizer 会修改
        # 参数，checkpoint 看到的权重就不再是提交保存时的稳定状态。
        #
        # 这个等待点放在 backward 之后，允许上一个 checkpoint 的 staging 与当前
        # step 的 forward/backward 尽量重叠。
        if checkpoint_future is not None:
            wait_start = time.perf_counter()
            checkpoint_future.staging_completion.result()
            log_staging_done(
                rank,
                checkpoint_step,
                checkpoint_submit_time,
                wait_start,
            )

        optimizer.step()

        # 限制同一时刻只有一个 checkpoint upload：
        # - 便于和其他脚本对比；
        # - 避免磁盘写入和 CPU buffer 在多个 checkpoint 请求之间堆积。
        if checkpoint_future is not None:
            wait_start = time.perf_counter()
            checkpoint_future.upload_completion.result()
            log_upload_done(
                rank,
                checkpoint_step,
                checkpoint_id,
                checkpoint_submit_time,
                wait_start,
            )

        checkpoint_step = step
        checkpoint_id = f"{config.checkpoint_dir}_step{step}"
        checkpoint_submit_time = time.perf_counter()
        # DefaultStager 会把 state dict 创建和 GPU -> CPU staging 放到后台线程。
        checkpoint_future = dcp.async_save(
            {"app": AppState(model, optimizer)},
            checkpoint_id=checkpoint_id,
            async_stager=DefaultStager(),
        )

    if checkpoint_future is not None:
        # 训练结束前必须等待最后一个 checkpoint 的 staging 和 upload 都完成。
        wait_start = time.perf_counter()
        checkpoint_future.staging_completion.result()
        log_staging_done(rank, checkpoint_step, checkpoint_submit_time, wait_start)

        wait_start = time.perf_counter()
        checkpoint_future.upload_completion.result()
        log_upload_done(
            rank,
            checkpoint_step,
            checkpoint_id,
            checkpoint_submit_time,
            wait_start,
        )

    cleanup()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "A CUDA device is required to run this script"
    # 参数固定在 main 中，四个脚本保持一致，运行时不需要手动传参。
    config = RunConfig(
        steps=20,
        batch_size=8,
        hidden_size=8192,
        layers=4,
        checkpoint_dir="checkpoint_default_stager",
        master_port="12355",
    )
    world_size = torch.cuda.device_count()
    print(
        f"Running DCP async_save with DefaultStager on {world_size} devices. "
        f"steps={config.steps}, batch_size={config.batch_size}, "
        f"hidden_size={config.hidden_size}, layers={config.layers}",
        flush=True,
    )
    mp.spawn(run, args=(world_size, config), nprocs=world_size, join=True)
