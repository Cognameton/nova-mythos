"""Run a fixed probe suite against a mythos_lite checkpoint.

Generates samples for a battery of prompts at fixed seed and saves them to
results/probes/<checkpoint-name>.txt so behavior can be diffed across epochs.

Usage:
    python scripts/probe_lite.py --checkpoint runs/lite-corpus/checkpoint-0035250
    python scripts/probe_lite.py --checkpoint runs/lite-corpus/checkpoint-0035250 \\
        --temperature 0.6 --max-new-tokens 80

To compare two checkpoints:
    diff results/probes/checkpoint-0028200.txt \\
         results/probes/checkpoint-0035250.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from scripts.generate import load_model, get_tokenizer, generate

PROBES = [
    ("story",    "Once upon a time,"),
    ("history",  "The Industrial Revolution was a period when"),
    ("science",  "Photosynthesis is the process by which"),
    ("qa",       "Q: What is the largest planet in our solar system?\nA:"),
    ("code",     "def fibonacci(n):\n    "),
    ("formal",   "Dear Sir or Madam,\n\nI am writing to"),
    ("sequence", "Monday, Tuesday, Wednesday,"),
    ("openended","The most important thing to remember about learning is"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--n-loops", type=int, default=8)
    p.add_argument("--seed", type=int, default=42,
                   help="Fixed by default so probes are reproducible across runs")
    p.add_argument("--output-dir", type=Path, default=Path("results/probes"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--stdout-only", action="store_true",
                   help="Print to stdout, do not write a file")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, cfg = load_model(args.checkpoint, device)
    tokenizer = get_tokenizer()

    header = [
        f"# probe results for {args.checkpoint}",
        f"# date         : {datetime.now().isoformat(timespec='seconds')}",
        f"# params       : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M",
        f"# temperature  : {args.temperature}   top_k: {args.top_k}",
        f"# n_loops      : {args.n_loops}   (cfg.max_loop_iters={cfg.max_loop_iters})",
        f"# max_new_toks : {args.max_new_tokens}   seed: {args.seed}",
        "",
    ]

    sections = []
    for tag, prompt in PROBES:
        text = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            n_loops=args.n_loops,
            device=device,
            seed=args.seed,
        )
        sections.append(f"=== [{tag}] ===\n{text}\n")
        print(f"=== [{tag}] ===")
        print(text)
        print()

    if not args.stdout_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{args.checkpoint.name}.txt"
        out_path.write_text("\n".join(header) + "\n".join(sections))
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
