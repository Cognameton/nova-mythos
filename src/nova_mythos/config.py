"""Configuration for the nova-mythos backend."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    variant: str = "1b"
    checkpoint_path: Optional[str] = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_loop_iters: Optional[int] = None  # None = use variant default
    vocab_size: Optional[int] = None      # None = use variant default


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=lambda: ["User:", "user:"])


@dataclass
class TokenizerConfig:
    name: str = "gpt2"


@dataclass
class NovaMythosConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)


def load_config(path: str | Path) -> NovaMythosConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    gen_raw = raw.get("generation", {})
    tok_raw = raw.get("tokenizer", {})

    return NovaMythosConfig(
        model=ModelConfig(
            variant=model_raw.get("variant", "1b"),
            checkpoint_path=model_raw.get("checkpoint_path"),
            device=model_raw.get("device", "cuda:0"),
            dtype=model_raw.get("dtype", "bfloat16"),
            max_loop_iters=model_raw.get("max_loop_iters"),
        ),
        generation=GenerationConfig(
            max_new_tokens=gen_raw.get("max_new_tokens", 512),
            temperature=gen_raw.get("temperature", 0.7),
            top_p=gen_raw.get("top_p", 0.9),
            stop_sequences=gen_raw.get("stop_sequences", ["User:", "user:"]),
        ),
        tokenizer=TokenizerConfig(
            name=tok_raw.get("name", "gpt2"),
        ),
    )
