"""Standalone probe runner.

Usage:
    .venv/bin/python scripts/run_probes.py [--config configs/nova_mythos.default.yaml] [--json out.json]

Loads NovaMythosBackend from config, runs the full probe battery, and prints
a formatted report. Optionally saves raw results as JSON for comparison with
Nova 2.0 llama_cpp baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nova_mythos.config import load_config, NovaMythosConfig, ModelConfig, TokenizerConfig
from nova_mythos.backend import NovaMythosBackend
from nova_mythos.harness import BackendProbeRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nova-mythos probe battery")
    parser.add_argument(
        "--config",
        default="configs/nova_mythos.default.yaml",
        help="Path to nova_mythos config YAML",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="FILE",
        help="Save full results as JSON to FILE",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per probe generation (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    print(f"Loading backend: {cfg.model.variant} on {cfg.model.device} ({cfg.model.dtype})")
    backend = NovaMythosBackend(cfg)
    backend.load()

    meta = backend.metadata()
    print(f"Backend ready: {meta}\n")

    runner = BackendProbeRunner(
        backend,
        persona_name="Nova",
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("Running probe battery...")
    report = runner.run_all()

    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend_name": report.backend_name,
            "model_id": report.model_id,
            "timestamp": report.timestamp,
            "pass_rate": report.pass_rate,
            "results": [asdict(r) for r in report.results],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nResults saved to: {out_path}")

    backend.unload()


if __name__ == "__main__":
    main()
