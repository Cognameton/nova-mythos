"""Single-prompt generation from a mythos_lite checkpoint.

Usage:
    python scripts/generate.py \\
        --checkpoint runs/lite-corpus/checkpoint-0035250 \\
        --prompt "Once upon a time," \\
        --max-new-tokens 120

The checkpoint directory must contain config.json and state.pt as written by
training/train_lite.py:save_checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from nova_mythos.model.architecture import MythosConfig, OpenMythos


def load_model(ckpt_dir: Path, device: torch.device) -> tuple[OpenMythos, MythosConfig]:
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text())
    valid_keys = {f.name for f in fields(MythosConfig)}
    cfg = MythosConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    model = OpenMythos(cfg).to(device)
    state = torch.load(ckpt_dir / "state.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def get_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def generate(
    model: OpenMythos,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    n_loops: int,
    device: torch.device,
    seed: int | None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        n_loops=n_loops,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a checkpoint dir (e.g. runs/lite-corpus/checkpoint-0035250)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--n-loops", type=int, default=8,
                   help="Recurrent loop depth at inference (training default = 8)")
    p.add_argument("--seed", type=int, default=None,
                   help="If set, makes sampling reproducible")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, cfg = load_model(args.checkpoint, device)
    tokenizer = get_tokenizer()

    print(f"checkpoint : {args.checkpoint}")
    print(f"params     : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"n_loops    : {args.n_loops}  (cfg.max_loop_iters={cfg.max_loop_iters})")
    print(f"temp/top_k : {args.temperature} / {args.top_k}")
    print("-" * 60)
    print(generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        n_loops=args.n_loops,
        device=device,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
