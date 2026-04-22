"""Phase 3 — Backend protocol compliance.

Verifies NovaMythosBackend satisfies Nova 2.0's InferenceBackend contract:
    load(), unload(), metadata(), tokenize(), generate()

These tests run on CPU with random weights to be CI-friendly.
"""

import pytest
from nova_mythos.config import NovaMythosConfig, ModelConfig, GenerationConfig, TokenizerConfig
from nova_mythos.backend import NovaMythosBackend, GenerationRequest


@pytest.fixture
def cpu_backend():
    cfg = NovaMythosConfig(
        model=ModelConfig(variant="1b", checkpoint_path=None, device="cpu", dtype="float32", vocab_size=50257),
        generation=GenerationConfig(max_new_tokens=8),
        tokenizer=TokenizerConfig(name="gpt2"),
    )
    backend = NovaMythosBackend(cfg)
    backend.load()
    yield backend
    backend.unload()


def test_metadata_before_load():
    cfg = NovaMythosConfig()
    backend = NovaMythosBackend(cfg)
    meta = backend.metadata()
    assert meta["loaded"] is False
    assert meta["backend"] == "nova_mythos"


def test_metadata_after_load(cpu_backend):
    meta = cpu_backend.metadata()
    assert meta["loaded"] is True
    assert "parameters" in meta
    assert meta["parameters"] > 0


def test_tokenize_returns_int(cpu_backend):
    count = cpu_backend.tokenize("Hello.")
    assert isinstance(count, int)
    assert count > 0


def test_generate_returns_result(cpu_backend):
    req = GenerationRequest(
        prompt="Nova:",
        max_tokens=8,
        temperature=1.0,
        top_p=1.0,
        stop_sequences=[],
    )
    result = cpu_backend.generate(req)
    assert hasattr(result, "text")
    assert hasattr(result, "prompt_tokens")
    assert hasattr(result, "completion_tokens")
    assert hasattr(result, "total_tokens")
    assert hasattr(result, "elapsed_seconds")
    assert result.backend == "nova_mythos"
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens


def test_stop_sequence_applied(cpu_backend):
    req = GenerationRequest(
        prompt="Begin. User: test",
        max_tokens=32,
        temperature=1.0,
        top_p=1.0,
        stop_sequences=["User:"],
    )
    result = cpu_backend.generate(req)
    assert "User:" not in result.text


def test_generate_after_unload_raises():
    cfg = NovaMythosConfig(
        model=ModelConfig(variant="1b", checkpoint_path=None, device="cpu", dtype="float32"),
    )
    backend = NovaMythosBackend(cfg)
    backend.load()
    backend.unload()
    req = GenerationRequest(prompt="test", max_tokens=4)
    with pytest.raises(RuntimeError):
        backend.generate(req)


def test_invalid_variant_raises():
    cfg = NovaMythosConfig(
        model=ModelConfig(variant="999b", device="cpu", dtype="float32"),
    )
    backend = NovaMythosBackend(cfg)
    with pytest.raises(ValueError):
        backend.load()
