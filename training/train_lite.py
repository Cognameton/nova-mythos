#!/usr/bin/env python3
"""
Train nova-mythos lite (93.5M).

Datasets:
    fineweb   -- FineWeb-Edu sample-10BT, streamed from HuggingFace (default)
    corpus    -- Local binary dataset prepared by prepare_corpus.py
    mixed     -- Corpus interleaved with FineWeb-Edu at a configurable ratio

Single GPU:
    python training/train_lite.py --output-dir runs/lite-001

Multi-GPU (both cards):
    .venv/bin/torchrun --nproc_per_node=2 training/train_lite.py \
        --output-dir runs/lite-001

Local corpus only:
    .venv/bin/torchrun --nproc_per_node=2 training/train_lite.py \
        --output-dir runs/corpus-001 \
        --dataset corpus \
        --corpus-path data/corpus/corpus.bin

Mixed (10% corpus, 90% FineWeb-Edu):
    .venv/bin/torchrun --nproc_per_node=2 training/train_lite.py \
        --output-dir runs/mixed-001 \
        --dataset mixed \
        --corpus-path data/corpus/corpus.bin \
        --corpus-ratio 0.1

Resume from latest checkpoint:
    .venv/bin/torchrun --nproc_per_node=2 training/train_lite.py \
        --output-dir runs/lite-001 --resume

Hardware target: dual RTX 3060 12GB
Best config from benchmark: seq=512 mb=8 acc=4 → 11.4k global tok/s
10B token run ≈ 10 days on FineWeb-Edu
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nova_mythos.model.architecture import OpenMythos
from nova_mythos.model.variants import mythos_lite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME   = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
VOCAB_SIZE     = 50257   # gpt2

# ---------------------------------------------------------------------------
# Defaults — tuned for dual RTX 3060 benchmark sweet spot
# ---------------------------------------------------------------------------

SEQ_LEN         = 512
MICRO_BATCH     = 8
GRAD_ACCUM      = 4
PEAK_LR         = 3e-4
MIN_LR          = 3e-5
WEIGHT_DECAY    = 0.1
GRAD_CLIP       = 1.0
WARMUP_STEPS    = 2_000
TOTAL_STEPS     = 305_176   # 10B tokens / (512*8*4*2)
SAVE_EVERY      = 5_000
KEEP_CHECKPOINTS = 3
LOG_EVERY       = 50


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FineWebDataset(IterableDataset):
    """Streams FineWeb-Edu, tokenises with gpt2, yields packed (x, y) pairs."""

    def __init__(self, seq_len: int, rank: int, world_size: int, seed: int = 42):
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size
        self.seed       = seed

    def __iter__(self):
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        eos = tokenizer.eos_token_id

        ds = load_dataset(
            DATASET_NAME,
            name=DATASET_CONFIG,
            split="train",
            streaming=True,
        ).shuffle(seed=self.seed, buffer_size=10_000)

        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)

        buf: list[int] = []
        for example in ds:
            ids = tokenizer.encode(example["text"], add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf   = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:],  dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# Local corpus dataset (from prepare_corpus.py binary output)
# ---------------------------------------------------------------------------

class CorpusDataset(IterableDataset):
    """Reads a flat uint16 binary corpus file and yields packed (x, y) pairs.

    The binary file is a flat array of token IDs written by prepare_corpus.py.
    Documents are separated by EOS tokens — the dataset streams sequentially
    through the file, wrapping back to the start when exhausted.

    With multiple ranks each rank starts at a different offset so they see
    different data within each epoch.
    """

    def __init__(self, bin_path: Path, seq_len: int, rank: int, world_size: int):
        self.bin_path   = Path(bin_path)
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size

    def __iter__(self):
        import numpy as np
        data = np.fromfile(self.bin_path, dtype=np.uint16).astype(np.int64)
        n    = len(data)

        # Each rank starts at a different position so they don't overlap
        offset = (self.rank * (n // self.world_size)) % n

        while True:
            start = offset
            end   = start + self.seq_len + 1
            if end > n:
                # wrap around
                chunk = np.concatenate([data[start:], data[: end - n]])
            else:
                chunk = data[start: end]
            offset = (offset + self.seq_len + 1) % n
            yield (
                torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long),
            )


# ---------------------------------------------------------------------------
# Mixed dataset — interleaves corpus and FineWeb-Edu at a given ratio
# ---------------------------------------------------------------------------

class MixedDataset(IterableDataset):
    """Yields from CorpusDataset with probability `corpus_ratio`,
    otherwise from FineWebDataset.  Mixing is per-batch, not per-token,
    so the actual ratio is approximate but consistent over long runs."""

    def __init__(
        self,
        corpus_path: Path,
        seq_len: int,
        rank: int,
        world_size: int,
        corpus_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.corpus  = CorpusDataset(corpus_path, seq_len, rank, world_size)
        self.fineweb = FineWebDataset(seq_len, rank, world_size, seed)
        self.corpus_ratio = corpus_ratio

    def __iter__(self):
        import random
        rng         = random.Random(42)
        corpus_iter = iter(self.corpus)
        fw_iter     = iter(self.fineweb)
        while True:
            if rng.random() < self.corpus_ratio:
                yield next(corpus_iter)
            else:
                yield next(fw_iter)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def make_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return lr_lambda


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    output_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    model_cfg,
    ddp: bool,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if ddp else model
    torch.save(
        {
            "step":      step,
            "model":     raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        ckpt_dir / "state.pt",
    )
    (ckpt_dir / "config.json").write_text(json.dumps(asdict(model_cfg)))
    return ckpt_dir


def load_checkpoint(ckpt_dir: Path, model: nn.Module, optimizer, scheduler, ddp: bool):
    state = torch.load(ckpt_dir / "state.pt", map_location="cpu", weights_only=True)
    raw_model = model.module if ddp else model
    raw_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["step"]


def cleanup_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    for old in checkpoints[:-keep]:
        shutil.rmtree(old)


def find_latest_checkpoint(output_dir: Path):
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",        type=Path, required=True)
    p.add_argument("--resume",            action="store_true")
    p.add_argument("--seq-len",           type=int,   default=SEQ_LEN)
    p.add_argument("--micro-batch",       type=int,   default=MICRO_BATCH)
    p.add_argument("--grad-accum",        type=int,   default=GRAD_ACCUM)
    p.add_argument("--total-steps",       type=int,   default=TOTAL_STEPS)
    p.add_argument("--warmup-steps",      type=int,   default=WARMUP_STEPS)
    p.add_argument("--peak-lr",           type=float, default=PEAK_LR)
    p.add_argument("--min-lr",            type=float, default=MIN_LR)
    p.add_argument("--save-every",        type=int,   default=SAVE_EVERY)
    p.add_argument("--keep-checkpoints",  type=int,   default=KEEP_CHECKPOINTS)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--dataset",           default="fineweb",
                   choices=["fineweb", "corpus", "mixed"],
                   help="fineweb=FineWeb-Edu stream, corpus=local binary, mixed=both")
    p.add_argument("--corpus-path",       type=Path,  default=None,
                   help="Path to corpus.bin from prepare_corpus.py (required for corpus/mixed)")
    p.add_argument("--corpus-ratio",      type=float, default=0.1,
                   help="Fraction of batches drawn from corpus in mixed mode (default 0.1)")
    return p.parse_args()


def main():
    args = parse_args()

    # --- distributed setup ---
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
        args.output_dir.mkdir(parents=True, exist_ok=True)

    global_batch = args.seq_len * args.micro_batch * args.grad_accum * world_size

    if master:
        dataset_desc = args.dataset
        if args.dataset == "mixed":
            dataset_desc = f"mixed ({args.corpus_ratio:.0%} corpus + {1-args.corpus_ratio:.0%} FineWeb-Edu)"
        elif args.dataset == "corpus":
            dataset_desc = f"corpus ({args.corpus_path})"

        print(f"\nnova-mythos Lite Training")
        print(f"{'='*60}")
        print(f"  Output dir   : {args.output_dir}")
        print(f"  Dataset      : {dataset_desc}")
        print(f"  GPUs         : {world_size}")
        print(f"  Global batch : {global_batch:,} tokens/step")
        print(f"  Total steps  : {args.total_steps:,}")
        print(f"  Warmup steps : {args.warmup_steps:,}")
        print(f"  Peak LR      : {args.peak_lr}")
        print(f"  Total tokens : {global_batch * args.total_steps / 1e9:.1f}B")
        print(f"{'='*60}\n")

    # --- model ---
    model_cfg = mythos_lite()
    model_cfg.vocab_size = VOCAB_SIZE
    model = OpenMythos(model_cfg).to(device=device, dtype=torch.bfloat16)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # --- optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        weight_decay=WEIGHT_DECAY,
        fused=torch.cuda.is_available(),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_lr_lambda(args.warmup_steps, args.total_steps, args.min_lr / args.peak_lr),
    )

    # --- resume ---
    start_step = 0
    if args.resume:
        ckpt = find_latest_checkpoint(args.output_dir)
        if ckpt is None:
            if master:
                print("No checkpoint found — starting from scratch.")
        else:
            if master:
                print(f"Resuming from {ckpt.name}")
            start_step = load_checkpoint(ckpt, model, optimizer, scheduler, ddp)
            if master:
                print(f"Resumed at step {start_step:,}\n")

    # --- data ---
    if args.dataset == "fineweb":
        dataset = FineWebDataset(
            seq_len=args.seq_len, rank=rank,
            world_size=world_size, seed=args.seed,
        )
    elif args.dataset == "corpus":
        if args.corpus_path is None:
            raise ValueError("--corpus-path required when --dataset=corpus")
        dataset = CorpusDataset(
            bin_path=args.corpus_path, seq_len=args.seq_len,
            rank=rank, world_size=world_size,
        )
    else:  # mixed
        if args.corpus_path is None:
            raise ValueError("--corpus-path required when --dataset=mixed")
        dataset = MixedDataset(
            corpus_path=args.corpus_path, seq_len=args.seq_len,
            rank=rank, world_size=world_size,
            corpus_ratio=args.corpus_ratio, seed=args.seed,
        )

    loader = DataLoader(dataset, batch_size=args.micro_batch, num_workers=0)

    loss_fn = nn.CrossEntropyLoss()

    # --- training ---
    model.train()
    step         = start_step
    tokens_seen  = step * global_batch
    running_loss = 0.0
    t_step_start    = time.perf_counter()
    t_run_start     = time.perf_counter()
    steps_in_window = 0

    data_iter = iter(loader)

    while step < args.total_steps:
        optimizer.zero_grad()

        batch_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                # restart the dataset stream (happens if < 10B tokens consumed)
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss   = loss_fn(
                    logits.reshape(-1, VOCAB_SIZE),
                    y.reshape(-1),
                ) / args.grad_accum
            loss.backward()
            batch_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        step             += 1
        tokens_seen      += global_batch
        steps_in_window  += 1
        running_loss = 0.9 * running_loss + 0.1 * batch_loss   # EMA

        # --- log ---
        if master and step % LOG_EVERY == 0:
            elapsed      = time.perf_counter() - t_step_start
            tps          = global_batch * steps_in_window / elapsed
            t_step_start    = time.perf_counter()
            steps_in_window = 0

            total_elapsed = time.perf_counter() - t_run_start
            steps_left    = args.total_steps - step
            steps_done    = max(step - start_step, 1)
            eta_s         = steps_left * (total_elapsed / steps_done)
            eta_h         = eta_s / 3600

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  step {step:>7,}/{args.total_steps:,} | "
                f"loss {running_loss:.4f} | "
                f"lr {current_lr:.2e} | "
                f"tok/s {tps/1000:.1f}k | "
                f"tokens {tokens_seen/1e9:.2f}B | "
                f"eta {eta_h:.1f}h",
                flush=True,
            )

        # --- checkpoint ---
        if master and step % args.save_every == 0:
            raw_cfg = model_cfg
            ckpt_path = save_checkpoint(
                args.output_dir, step, model, optimizer, scheduler, raw_cfg, ddp
            )
            cleanup_checkpoints(args.output_dir, args.keep_checkpoints)
            print(f"  [checkpoint] saved {ckpt_path.name}", flush=True)

        if ddp and step % args.save_every == 0:
            dist.barrier()

    # --- final checkpoint ---
    if master:
        save_checkpoint(
            args.output_dir, step, model, optimizer, scheduler, model_cfg, ddp
        )
        print(f"\nTraining complete. {tokens_seen/1e9:.1f}B tokens seen.")
        print(f"Final checkpoint: {args.output_dir}/checkpoint-{step:07d}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
