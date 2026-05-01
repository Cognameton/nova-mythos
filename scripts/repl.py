"""Interactive inference REPL for mythos_lite checkpoints.

Loads a checkpoint once, then lets you type prompts in a loop. Generation
parameters can be tweaked on the fly via slash commands without reloading.

Usage:
    .venv/bin/python -m scripts.repl --checkpoint runs/lite-corpus/checkpoint-0038775

Slash commands (typed at the prompt):
    /help                  show this help
    /params                show current generation params
    /temp <float>          set temperature (e.g. /temp 0.6)
    /topk <int>            set top_k (0 disables)
    /loops <int>           set recurrent loop depth (training default 8)
    /maxtok <int>          set max new tokens
    /seed <int|off>        set sampling seed; 'off' for nondeterministic
    /multi                 enter multi-line mode (end with '/send' on its own line)
    /load <path>           swap to a different checkpoint (slower)
    /quit, /exit           leave the REPL

Anything not starting with '/' is treated as a prompt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.generate import load_model, get_tokenizer, generate

DEFAULTS = {
    "temperature": 0.8,
    "top_k": 40,
    "n_loops": 8,
    "max_new_tokens": 120,
    "seed": 42,
}


def show_help():
    print(__doc__)


def show_params(state: dict, ckpt: Path):
    print(f"  checkpoint     : {ckpt}")
    print(f"  temperature    : {state['temperature']}")
    print(f"  top_k          : {state['top_k']}")
    print(f"  n_loops        : {state['n_loops']}")
    print(f"  max_new_tokens : {state['max_new_tokens']}")
    print(f"  seed           : {state['seed']}")


def parse_slash(line: str) -> tuple[str, str]:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower().lstrip("/")
    arg = parts[1] if len(parts) > 1 else ""
    return cmd, arg


def read_multi() -> str:
    print("(multi-line mode — finish with '/send' on its own line, '/cancel' to abort)")
    lines: list[str] = []
    while True:
        try:
            line = input("... ")
        except EOFError:
            return ""
        stripped = line.strip()
        if stripped == "/send":
            return "\n".join(lines)
        if stripped == "/cancel":
            return ""
        lines.append(line)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = args.checkpoint
    print(f"loading {ckpt} ...")
    model, cfg = load_model(ckpt, device)
    tokenizer = get_tokenizer()
    state = dict(DEFAULTS)
    print(f"ready — {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params on {device}")
    print("type a prompt, or /help for commands.")

    while True:
        try:
            line = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line.strip():
            continue

        if line.startswith("/"):
            cmd, arg = parse_slash(line)
            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                show_help()
            elif cmd == "params":
                show_params(state, ckpt)
            elif cmd == "temp":
                state["temperature"] = float(arg)
            elif cmd == "topk":
                state["top_k"] = int(arg)
            elif cmd == "loops":
                state["n_loops"] = int(arg)
            elif cmd == "maxtok":
                state["max_new_tokens"] = int(arg)
            elif cmd == "seed":
                state["seed"] = None if arg.lower() == "off" else int(arg)
            elif cmd == "multi":
                prompt = read_multi()
                if prompt:
                    run_one(model, tokenizer, prompt, state, device)
            elif cmd == "load":
                new_ckpt = Path(arg)
                if not (new_ckpt / "state.pt").exists():
                    print(f"  no state.pt at {new_ckpt}")
                    continue
                print(f"loading {new_ckpt} ...")
                model, cfg = load_model(new_ckpt, device)
                ckpt = new_ckpt
                print(f"ready — {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
            else:
                print(f"  unknown command: /{cmd}  (try /help)")
            continue

        run_one(model, tokenizer, line, state, device)


def run_one(model, tokenizer, prompt: str, state: dict, device):
    text = generate(
        model, tokenizer, prompt,
        max_new_tokens=state["max_new_tokens"],
        temperature=state["temperature"],
        top_k=state["top_k"],
        n_loops=state["n_loops"],
        device=device,
        seed=state["seed"],
    )
    print(text)


if __name__ == "__main__":
    main()
