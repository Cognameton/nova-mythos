"""NovaMythosBackend — implements Nova 2.0's InferenceBackend protocol.

Drop-in replacement for LlamaCppBackend. Nova 2.0's runtime only calls:
    load(), unload(), metadata(), tokenize(), generate()

Phase 2: hardware validation (random weights, forward pass, VRAM check)
Phase 3: full implementation with checkpoint loading and generation
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Types mirrored from Nova 2.0 src/nova/types.py — kept local to avoid
# coupling the two projects at import time.
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_seconds: float
    backend: str = "nova_mythos"
    truncated: bool = False


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class NovaMythosBackend:
    """OpenMythos RDT inference backend for Nova 2.0."""

    def __init__(self, config):
        """
        config: NovaMythosConfig instance (from nova_mythos.config)
        """
        self._config = config
        self._model = None
        self._tokenizer = None
        self._device = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Instantiate model and tokenizer, move to target device."""
        import torch
        from nova_mythos.model import OpenMythos, MythosTokenizer
        from nova_mythos.model.variants import _VARIANTS

        from nova_mythos.model.variants import (
            mythos_1b, mythos_3b, mythos_10b, mythos_50b,
            mythos_100b, mythos_500b, mythos_1t,
        )
        _VARIANTS = {
            "1b": mythos_1b,
            "3b": mythos_3b,
            "10b": mythos_10b,
            "50b": mythos_50b,
            "100b": mythos_100b,
            "500b": mythos_500b,
            "1t": mythos_1t,
        }

        variant_name = self._config.model.variant
        if variant_name not in _VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant_name}'. "
                f"Available: {list(_VARIANTS.keys())}"
            )

        mythos_cfg = _VARIANTS[variant_name]()

        # Override loop count if explicitly configured
        if self._config.model.max_loop_iters is not None:
            mythos_cfg.max_loop_iters = self._config.model.max_loop_iters

        dtype = getattr(torch, self._config.model.dtype)
        self._device = torch.device(self._config.model.device)

        self._model = OpenMythos(mythos_cfg).to(dtype=dtype, device=self._device)

        checkpoint_path = self._config.model.checkpoint_path
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location=self._device)
            self._model.load_state_dict(state)
        # If no checkpoint, model runs with random weights (Phase 2 validation)

        self._model.eval()

        self._tokenizer = MythosTokenizer(self._config.tokenizer.name)

    def unload(self) -> None:
        """Free model from memory and clear CUDA cache."""
        import torch
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        if self._model is None:
            return {"backend": "nova_mythos", "loaded": False}

        import torch
        param_count = sum(p.numel() for p in self._model.parameters())
        return {
            "backend": "nova_mythos",
            "loaded": True,
            "variant": self._config.model.variant,
            "device": str(self._device),
            "dtype": self._config.model.dtype,
            "parameters": param_count,
            "checkpoint": self._config.model.checkpoint_path,
        }

    def tokenize(self, text: str) -> int:
        """Return token count for text."""
        if self._tokenizer is None:
            raise RuntimeError("Backend not loaded. Call load() first.")
        return len(self._tokenizer.encode(text))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run autoregressive generation and return result."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Backend not loaded. Call load() first.")

        import torch

        prompt_ids = self._tokenizer.encode(request.prompt)
        prompt_tokens = len(prompt_ids)

        input_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self._device
        )

        t0 = time.perf_counter()

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_tensor,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

        elapsed = time.perf_counter() - t0

        # Decode only the newly generated tokens
        new_ids = output_ids[0][prompt_tokens:].tolist()
        text = self._tokenizer.decode(new_ids)

        # Apply stop sequences
        truncated = False
        for stop in request.stop_sequences:
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
                truncated = True

        text = text.strip()
        completion_tokens = len(new_ids)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_seconds=elapsed,
            backend="nova_mythos",
            truncated=truncated,
        )
