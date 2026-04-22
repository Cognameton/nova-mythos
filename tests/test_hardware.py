"""Phase 2 — Hardware validation.

Verifies the 1B RDT variant instantiates and runs a forward pass on the
available CUDA device without OOM. Uses random weights (no checkpoint needed).

Run with: pytest tests/test_hardware.py -v -s
"""

import pytest
import torch

from nova_mythos.config import NovaMythosConfig, ModelConfig, GenerationConfig, TokenizerConfig
from nova_mythos.backend import NovaMythosBackend, GenerationRequest


@pytest.fixture
def backend_1b():
    cfg = NovaMythosConfig(
        model=ModelConfig(variant="1b", checkpoint_path=None, device="cuda:0", vocab_size=50257),
        generation=GenerationConfig(max_new_tokens=32),
        tokenizer=TokenizerConfig(name="gpt2"),
    )
    backend = NovaMythosBackend(cfg)
    backend.load()
    yield backend
    backend.unload()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_loads(backend_1b):
    meta = backend_1b.metadata()
    assert meta["loaded"] is True
    assert meta["variant"] == "1b"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vram_within_budget(backend_1b):
    """1B model should fit within 12GB on a single RTX 3060."""
    allocated_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"\nVRAM allocated: {allocated_gb:.2f} GB")
    assert allocated_gb < 12.0, f"VRAM usage {allocated_gb:.2f}GB exceeds 12GB budget"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_pass(backend_1b):
    """Model should produce output without error."""
    req = GenerationRequest(
        prompt="Hello, I am Nova.",
        max_tokens=16,
        temperature=0.7,
        top_p=0.9,
        stop_sequences=[],
    )
    result = backend_1b.generate(req)
    assert isinstance(result.text, str)
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.elapsed_seconds > 0
    print(f"\nGenerated: {repr(result.text)}")
    print(f"Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tokenize(backend_1b):
    count = backend_1b.tokenize("Hello, I am Nova.")
    assert count > 0
    print(f"\nToken count: {count}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unload_clears_cache():
    cfg = NovaMythosConfig(
        model=ModelConfig(variant="1b", checkpoint_path=None, device="cuda:0", vocab_size=50257),
        tokenizer=TokenizerConfig(name="gpt2"),
    )
    backend = NovaMythosBackend(cfg)
    backend.load()
    assert backend.metadata()["loaded"] is True
    backend.unload()
    assert backend.metadata()["loaded"] is False
