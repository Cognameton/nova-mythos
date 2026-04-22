#!/usr/bin/env python3
"""
Training throughput benchmark for the nova-mythos 1B model.

Uses synthetic (random token) data — no dataset download needed.
Measures tokens/sec, VRAM per GPU, and step time across multiple
batch configs to find the viable sweet spot on dual RTX 3060 12GB.

Single GPU:
    python training/benchmark_1b.py

Multi-GPU (both cards):
    torchrun --nproc_per_node=2 training/benchmark_1b.py
"""

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nova_mythos.model.architecture import OpenMythos, TransformerBlock, RecurrentBlock
from nova_mythos.model.variants import mythos_1b

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WARMUP_STEPS   = 5
MEASURE_STEPS  = 20
TARGET_TOKENS  = 100_000_000_000   # 100B — realistic 1B pretraining target
VOCAB_SIZE     = 50257             # gpt2 vocab; matches our tokenizer

# ---------------------------------------------------------------------------
# Configs to sweep
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    seq_len: int
    micro_batch: int
    grad_accum: int
    grad_checkpointing: bool = False

    @property
    def label(self) -> str:
        gc = "+gc" if self.grad_checkpointing else "   "
        return f"seq={self.seq_len:<5} mb={self.micro_batch} acc={self.grad_accum:<3} {gc}"


BENCH_CONFIGS = [
    BenchConfig(seq_len=512,  micro_batch=4, grad_accum=8),
    BenchConfig(seq_len=512,  micro_batch=4, grad_accum=8,  grad_checkpointing=True),
    BenchConfig(seq_len=1024, micro_batch=2, grad_accum=8),
    BenchConfig(seq_len=1024, micro_batch=2, grad_accum=8,  grad_checkpointing=True),
    BenchConfig(seq_len=2048, micro_batch=1, grad_accum=8),
    BenchConfig(seq_len=2048, micro_batch=1, grad_accum=8,  grad_checkpointing=True),
    BenchConfig(seq_len=2048, micro_batch=2, grad_accum=4,  grad_checkpointing=True),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(seq_len: int, micro_batch: int, device: str):
    ids = torch.randint(0, VOCAB_SIZE, (micro_batch, seq_len + 1), device=device)
    return ids[:, :-1], ids[:, 1:]


def apply_grad_checkpointing(model: nn.Module) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    check_fn = lambda m: isinstance(m, (TransformerBlock, RecurrentBlock))
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
            m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=check_fn,
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    label: str
    seq_len: int
    micro_batch: int
    grad_accum: int
    grad_checkpointing: bool
    world_size: int
    global_tokens_per_step: int
    mean_step_ms: float
    std_step_ms: float
    tokens_per_sec: float
    peak_vram_gb: float
    oom: bool
    estimated_days_100b: float
    note: str = ""

    def row(self) -> str:
        if self.oom:
            return f"  {self.label}  OOM"
        return (
            f"  {self.label}  "
            f"{self.mean_step_ms:7.0f}ms ±{self.std_step_ms:5.0f}  "
            f"{self.tokens_per_sec/1000:7.1f}k tok/s  "
            f"VRAM {self.peak_vram_gb:.2f}GB  "
            f"~{self.estimated_days_100b:.1f}d"
        )


# ---------------------------------------------------------------------------
# Single config run
# ---------------------------------------------------------------------------

def run_config(
    cfg: BenchConfig,
    ddp: bool,
    rank: int,
    local_rank: int,
    world_size: int,
    device: str,
) -> BenchResult:

    common = dict(
        label=cfg.label,
        seq_len=cfg.seq_len,
        micro_batch=cfg.micro_batch,
        grad_accum=cfg.grad_accum,
        grad_checkpointing=cfg.grad_checkpointing,
        world_size=world_size,
        global_tokens_per_step=cfg.seq_len * cfg.micro_batch * cfg.grad_accum * world_size,
    )

    try:
        model_cfg = mythos_1b()
        model_cfg.vocab_size = VOCAB_SIZE

        model = OpenMythos(model_cfg)

        if ddp:
            mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mp,
                auto_wrap_policy=ModuleWrapPolicy({TransformerBlock, RecurrentBlock}),
                device_id=local_rank,
            )
            if cfg.grad_checkpointing:
                apply_grad_checkpointing(model)
        else:
            model = model.to(device=device, dtype=torch.bfloat16)
            if cfg.grad_checkpointing:
                apply_grad_checkpointing(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.1,
            fused=torch.cuda.is_available(),
        )
        loss_fn = nn.CrossEntropyLoss()

        torch.cuda.reset_peak_memory_stats(local_rank)
        step_times: list[float] = []

        for step in range(WARMUP_STEPS + MEASURE_STEPS):
            t0 = time.perf_counter()
            optimizer.zero_grad()

            for _ in range(cfg.grad_accum):
                x, y = make_batch(cfg.seq_len, cfg.micro_batch, device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = loss_fn(
                        logits.reshape(-1, VOCAB_SIZE),
                        y.reshape(-1),
                    ) / cfg.grad_accum
                loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ddp:
                dist.barrier()

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if step >= WARMUP_STEPS:
                step_times.append(elapsed_ms)

        peak_vram_gb = torch.cuda.max_memory_allocated(local_rank) / 1e9
        mean_ms = sum(step_times) / len(step_times)
        std_ms = math.sqrt(sum((t - mean_ms) ** 2 for t in step_times) / len(step_times))
        global_tps = (cfg.seq_len * cfg.micro_batch * cfg.grad_accum * world_size) / (mean_ms / 1000)
        est_days = TARGET_TOKENS / (global_tps * 86400)

        del model, optimizer
        torch.cuda.empty_cache()
        if ddp:
            dist.barrier()

        return BenchResult(
            **common,
            mean_step_ms=mean_ms,
            std_step_ms=std_ms,
            tokens_per_sec=global_tps,
            peak_vram_gb=peak_vram_gb,
            oom=False,
            estimated_days_100b=est_days,
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if ddp:
            try:
                dist.barrier()
            except Exception:
                pass
        return BenchResult(
            **common,
            mean_step_ms=0,
            std_step_ms=0,
            tokens_per_sec=0,
            peak_vram_gb=12.0,
            oom=True,
            estimated_days_100b=float("inf"),
            note="OOM",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device     = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        model_cfg = mythos_1b()
        model_cfg.vocab_size = VOCAB_SIZE
        param_count = sum(p.numel() for p in OpenMythos(model_cfg).parameters())
        del model_cfg

        print(f"\nnova-mythos 1B Training Benchmark")
        print(f"{'='*70}")
        print(f"  GPUs          : {world_size}x RTX 3060 12GB")
        print(f"  Parameters    : {param_count/1e9:.3f}B")
        print(f"  Vocab size    : {VOCAB_SIZE:,}")
        print(f"  Warmup steps  : {WARMUP_STEPS}")
        print(f"  Measure steps : {MEASURE_STEPS}")
        print(f"  Target        : {TARGET_TOKENS/1e9:.0f}B tokens")
        print(f"{'='*70}\n")

    results = []
    for cfg in BENCH_CONFIGS:
        if master:
            print(f"  [{BENCH_CONFIGS.index(cfg)+1}/{len(BENCH_CONFIGS)}] {cfg.label} ...", flush=True)

        result = run_config(
            cfg=cfg,
            ddp=ddp,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
        results.append(result)

        if master:
            print(f"         {result.row()}", flush=True)

    if master:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Config':<40} {'Step':>9}  {'Tok/s':>9}  {'VRAM':>9}  {'100B est':>8}")
        print(f"  {'-'*40} {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}")
        for r in results:
            print(r.row())

        viable = [r for r in results if not r.oom]
        if viable:
            best = max(viable, key=lambda r: r.tokens_per_sec)
            print(f"\n  Best config   : {best.label.strip()}")
            print(f"  Throughput    : {best.tokens_per_sec/1000:.1f}k global tok/s")
            print(f"  VRAM peak     : {best.peak_vram_gb:.2f} GB per GPU")
            print(f"  Est. 100B     : ~{best.estimated_days_100b:.1f} days")
            print(f"  Est. 10B      : ~{best.estimated_days_100b/10:.1f} days (FineWeb-Edu sample)")

        out = Path("benchmark_results.json")
        out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\n  Results saved : {out}\n")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
