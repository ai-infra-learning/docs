"""直接使用 torch.save() 的 checkpoint 基线示例。

在 books/ai-systems-performance-engineering/chapters/ch03 目录下运行：

    uv run python dcp-async-checkpoint/torch_save.py

这个脚本是最简单的单进程基线：
- 只使用 1 张 CUDA GPU。
- 不使用 PyTorch Distributed Checkpoint（DCP）。
- 不做 checkpoint 分片。
- 不做异步写入。

它每一步都会把完整的 model state_dict 和 optimizer state_dict 写入单个
.pt 文件，用来和 DCP 同步/异步分布式 checkpoint 做性能对比。
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RunConfig:
    # 四个脚本使用同一组模型和训练参数，保证 checkpoint 大小可比。
    steps: int
    batch_size: int
    hidden_size: int
    checkpoint_prefix: str


class ToyModel(nn.Module):
    """用于制造 checkpoint 的简单 MLP，结构保持接近 PyTorch DCP 示例。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        # 结构对应原始示例中的 net1 -> relu -> net2，只把维度放大。
        self.net1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def format_bytes(num_bytes: int) -> str:
    """把 checkpoint 文件大小格式化成人容易读的单位。"""

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def remove_old_checkpoints(config: RunConfig) -> None:
    """清理上一次运行留下的 .pt 文件，避免文件大小和耗时统计被干扰。"""

    for step in range(config.steps):
        Path(f"{config.checkpoint_prefix}_step{step}.pt").unlink(missing_ok=True)


def run(config: RunConfig) -> None:
    # torch.save 基线只使用 cuda:0。它保存的是完整 checkpoint，不是分片。
    device = torch.device("cuda", 0)

    # 固定随机种子，保证每次运行模型初始化一致。
    torch.manual_seed(0)
    model = ToyModel(config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    remove_old_checkpoints(config)

    for step in range(config.steps):
        # 做一个极简训练 step，让 optimizer state 被初始化并持续更新。
        optimizer.zero_grad(set_to_none=True)
        x = torch.rand(config.batch_size, config.hidden_size, device=device)
        model(x).sum().backward()
        optimizer.step()

        checkpoint_path = Path(f"{config.checkpoint_prefix}_step{step}.pt")
        # 这是最朴素的 checkpoint 内容：完整模型参数、完整 optimizer 状态、
        # 当前 step。所有内容都会写进一个 .pt 文件。
        checkpoint = {
            "step": step,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
        }

        # 计时范围只覆盖 torch.save() 本身，也就是训练主线程被同步写盘阻塞的时间。
        start_time = time.perf_counter()
        torch.save(checkpoint, checkpoint_path)
        total_seconds = time.perf_counter() - start_time

        # size 是单个 .pt checkpoint 文件的大小。
        print(
            f"[torch_save] step={step} total={total_seconds:.3f}s "
            f"size={format_bytes(checkpoint_path.stat().st_size)}",
            flush=True,
        )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "A CUDA device is required to run this script"
    # 参数固定在 main 中，四个脚本保持一致，运行时不需要手动传参。
    config = RunConfig(
        steps=3,
        batch_size=8,
        hidden_size=2048,
        checkpoint_prefix="checkpoint_torch_save",
    )
    print(
        "Running plain torch.save baseline on one CUDA device. "
        f"hidden_size={config.hidden_size}",
        flush=True,
    )
    run(config)
