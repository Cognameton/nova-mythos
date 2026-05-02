"""Diagnose whether the recurrent / latent-reasoning machinery is working.

Loads a checkpoint, runs a held-out batch through it, and reports five
independent signals about the recurrent block:

    1. Loss vs n_loops  — does extra recurrence reduce loss at all?
    2. ACT halting       — when do tokens halt? mean halt step?
    3. MoE routing       — is one expert dominating, or is load balanced?
    4. LTI A diagonal    — is the recurrent state over-damped or near-identity?
    5. LoRA per-loop     — are loop iterations actually differentiated?

Usage:
    .venv/bin/python -m scripts.diagnose_lite \\
        --checkpoint runs/lite-corpus/checkpoint-0042300
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.generate import load_model

CORPUS_BIN = Path("data/corpus/corpus.bin")
SEQ_LEN = 512
BATCH_SIZE = 4


def get_eval_batch(seq_len: int, batch: int, device: torch.device):
    raw = np.memmap(CORPUS_BIN, dtype=np.uint16, mode="r")
    needed = batch * (seq_len + 1)
    start = max(0, len(raw) - needed * 5)
    chunk = np.asarray(raw[start : start + needed], dtype=np.int64)
    data = torch.from_numpy(chunk).view(batch, seq_len + 1).to(device)
    return data[:, :-1], data[:, 1:]


def compute_loss(model, x, y, n_loops: int) -> float:
    with torch.no_grad():
        logits = model(x, n_loops=n_loops)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        ).item()


def test_loss_vs_loops(model, x, y, candidates):
    print("\n[1] Loss vs n_loops")
    print("    (lower = better; a flat curve means recurrence isn't earning compute)")
    print("    n_loops |   loss")
    print("    --------+--------")
    losses = []
    for n in candidates:
        loss = compute_loss(model, x, y, n)
        losses.append(loss)
        print(f"    {n:7d} | {loss:.4f}")
    span = max(losses) - min(losses)
    print(f"    → span across n_loops: {span:.4f}")
    if span < 0.01:
        print("    ⚠ recurrence appears to do nothing — loss is flat across loop counts")
    elif span < 0.05:
        print("    ⚠ recurrence has marginal effect — extra loops barely help")


def test_act(model, x):
    captured: list[torch.Tensor] = []

    original = model.recurrent.act.forward

    def hooked(h):
        p = original(h)
        captured.append(p.detach().cpu())
        return p

    model.recurrent.act.forward = hooked
    try:
        with torch.no_grad():
            _ = model(x, n_loops=model.cfg.max_loop_iters)
    finally:
        model.recurrent.act.forward = original

    threshold = model.cfg.act_threshold
    max_loops = model.cfg.max_loop_iters
    cumulative = torch.zeros_like(captured[0])
    weight_sum = torch.zeros_like(captured[0])
    halt_step = torch.full(cumulative.shape, max_loops, dtype=torch.long)
    halted = torch.zeros_like(captured[0], dtype=torch.bool)
    for step, p in enumerate(captured):
        is_final = step == len(captured) - 1
        unset = halt_step == max_loops
        new_halts = (cumulative + p >= threshold) & unset
        halt_step[new_halts] = step
        # Mirror the model's weight assignment: remainder when threshold
        # is crossed OR on the final loop (architectural fix).
        still_running = ~halted
        remainder = (1.0 - cumulative).clamp(min=0)
        weight = torch.where(
            (cumulative + p >= threshold) | is_final,
            remainder,
            p,
        )
        weight = weight * still_running.to(dtype=p.dtype)
        weight_sum = weight_sum + weight
        cumulative = cumulative + p * still_running.to(dtype=p.dtype)
        halted = halted | (cumulative >= threshold) | is_final

    flat = halt_step.flatten().tolist()
    counter = Counter(flat)
    total = len(flat)

    print(f"\n[2] ACT halting distribution (threshold={threshold}, max_loops={max_loops})")
    print("    halt step | tokens | %")
    print("    ----------+--------+------")
    for step in sorted(counter):
        n = counter[step]
        label = "never" if step == max_loops else str(step)
        print(f"    {label:>9} | {n:6d} | {100 * n / total:5.1f}%")

    avg = sum(s * n for s, n in counter.items()) / total
    print(f"    → mean halt step: {avg:.2f}")
    print(
        "    → cumulative halt p (sum over loops, halting-head behaviour): "
        f"mean={cumulative.mean().item():.4f} "
        f"min={cumulative.min().item():.4f} "
        f"max={cumulative.max().item():.4f}"
    )
    print(
        "    → h_out weight mass (sum of weights, output-correctness check): "
        f"mean={weight_sum.mean().item():.4f} "
        f"min={weight_sum.min().item():.4f} "
        f"max={weight_sum.max().item():.4f}"
    )
    p_all = torch.stack(captured)
    print(
        "    → halt p per iteration: "
        f"mean={p_all.mean().item():.4f} "
        f"min={p_all.min().item():.4f} "
        f"max={p_all.max().item():.4f}"
    )
    if avg < 1.5:
        print("    ⚠ ACT collapsed — most tokens halt at the first iteration (no recurrent reasoning)")
    elif avg > max_loops - 0.5:
        print("    ⚠ ACT halting head not useful — every token relies on the final-loop remainder forcing (deterministic depth)")
    if weight_sum.mean().item() < 0.99 or weight_sum.max().item() > 1.01:
        print("    ⚠ h_out weight mass is not unity — architectural unit-mass bug present (remainder not forced at final loop)")


def test_moe(model, x):
    moe = model.recurrent.block.ffn
    counts = Counter()
    original = moe.forward

    def hooked(x_in):
        with torch.no_grad():
            B, T, D = x_in.shape
            flat = x_in.reshape(B * T, D)
            logits = moe.router(flat)
            _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
            for eid in topk_idx.reshape(-1).tolist():
                counts[eid] += 1
        return original(x_in)

    moe.forward = hooked
    try:
        with torch.no_grad():
            _ = model(x, n_loops=model.cfg.max_loop_iters)
    finally:
        moe.forward = original

    n_exp = model.cfg.n_experts
    topk = model.cfg.n_experts_per_tok
    total = sum(counts.values())
    expected = 100.0 / n_exp

    print(f"\n[3] MoE routing distribution ({n_exp} experts, top-{topk})")
    print("    expert | routes |   %  | bar")
    print("    -------+--------+------+----")
    max_count = max(counts.values()) if counts else 1
    for eid in sorted(range(n_exp), key=lambda e: -counts.get(e, 0)):
        n = counts.get(eid, 0)
        bar = "#" * int(40 * n / max_count) if max_count else ""
        print(f"    {eid:6d} | {n:6d} | {100*n/total:4.1f}% | {bar}")

    used = sum(1 for c in counts.values() if c > 0)
    top_share = max_count / total
    print(f"    → balanced load = {expected:.1f}%/expert; top expert = {100*top_share:.1f}%; used = {used}/{n_exp}")
    if top_share > 0.5:
        print("    ⚠ MoE collapsed — one expert dominates routing")
    elif used < n_exp / 2:
        print("    ⚠ MoE underused — fewer than half of experts ever fire")


def test_lti(model):
    A = model.recurrent.injection.get_A().detach().cpu().numpy()
    B = model.recurrent.injection.B.detach().cpu().numpy()
    print("\n[4] LTI injection: h_{t+1} = A·h_t + B·e + transformer_out")
    print(f"    A diag — min={A.min():.4f}  median={np.median(A):.4f}  max={A.max():.4f}")
    print(f"    A < 0.05 (state forgotten each step) : {(A < 0.05).mean()*100:5.1f}% of channels")
    print(f"    A > 0.95 (state nearly preserved)    : {(A > 0.95).mean()*100:5.1f}% of channels")
    print(f"    B    — min={B.min():.4f}  median={np.median(B):.4f}  max={B.max():.4f}")
    if np.median(A) < 0.05:
        print("    ⚠ A is heavily damped — recurrent state h is wiped each loop; only e and trans_out flow forward")
    elif np.median(A) > 0.95:
        print("    ⚠ A is near identity — h carries forward strongly, transformer barely modifies it")


def test_lora(model):
    scale = model.recurrent.lora.scale.weight.detach().cpu().numpy()
    n_loops, rank = scale.shape
    print(f"\n[5] LoRA per-loop scale ({n_loops} loops × rank {rank})")
    print("    loop | ‖scale‖ | cos(loop_0)")
    print("    -----+---------+------------")
    s0 = scale[0]
    sims = []
    for t in range(n_loops):
        st = scale[t]
        norm = np.linalg.norm(st)
        denom = np.linalg.norm(s0) * norm + 1e-9
        cos = float(np.dot(s0, st) / denom)
        sims.append(cos)
        print(f"    {t:4d} | {norm:7.4f} | {cos:+.4f}")
    if min(sims[1:], default=1.0) > 0.99:
        print("    ⚠ LoRA scales are near-identical across loops — adapter is not differentiating iterations")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"loading {args.checkpoint} ...")
    model, cfg = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params  max_loop_iters={cfg.max_loop_iters}  "
          f"n_experts={cfg.n_experts}  topk={cfg.n_experts_per_tok}  lora_rank={cfg.lora_rank}")

    x, y = get_eval_batch(args.seq_len, args.batch, device)
    print(f"eval batch: {tuple(x.shape)} drawn from end of {CORPUS_BIN}")

    test_loss_vs_loops(model, x, y, [1, 2, 4, 8, 16])
    test_act(model, x)
    test_moe(model, x)
    test_lti(model)
    test_lora(model)

    print("\ndone.")


if __name__ == "__main__":
    main()
