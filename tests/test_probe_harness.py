"""Phase 4 — Probe harness tests.

Verifies the BackendProbeRunner infrastructure works correctly.
Run on CPU with random weights — tests harness correctness, not generation quality.
"""

import pytest
from nova_mythos.config import NovaMythosConfig, ModelConfig, GenerationConfig, TokenizerConfig
from nova_mythos.backend import NovaMythosBackend
from nova_mythos.harness import BackendProbeRunner, ProbeResult, ProbeReport


@pytest.fixture(scope="module")
def harness():
    cfg = NovaMythosConfig(
        model=ModelConfig(variant="1b", checkpoint_path=None, device="cpu", dtype="float32", vocab_size=50257),
        generation=GenerationConfig(max_new_tokens=32),
        tokenizer=TokenizerConfig(name="gpt2"),
    )
    backend = NovaMythosBackend(cfg)
    backend.load()
    runner = BackendProbeRunner(backend, persona_name="Nova", max_new_tokens=32)
    yield runner
    backend.unload()


# ------------------------------------------------------------------
# Structural tests — harness returns correct types regardless of quality
# ------------------------------------------------------------------

def test_probe_result_is_dataclass(harness):
    results = harness.run_no_think_compliance(["Hello."])
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, ProbeResult)
    assert isinstance(r.probe_id, str) and len(r.probe_id) > 0
    assert isinstance(r.timestamp, str)
    assert isinstance(r.score, float)
    assert isinstance(r.passed, bool)
    assert isinstance(r.notes, dict)


def test_no_think_compliance_runs(harness):
    prompts = ["Hello.", "Who are you?"]
    results = harness.run_no_think_compliance(prompts)
    assert len(results) == len(prompts)
    assert all(r.probe_type == "no_think_compliance" for r in results)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_no_reasoning_leak_runs(harness):
    results = harness.run_no_reasoning_leak(["Explain something."])
    assert len(results) == 1
    assert results[0].probe_type == "no_reasoning_leak"


def test_no_prompt_echo_runs(harness):
    results = harness.run_no_prompt_echo(["[Persona] Hello."])
    assert len(results) == 1
    assert results[0].probe_type == "no_prompt_echo"


def test_identity_presence_runs(harness):
    results = harness.run_identity_presence(n=2)
    assert len(results) == 2
    assert all(r.probe_type == "identity_presence" for r in results)


def test_generation_stability_runs(harness):
    results = harness.run_generation_stability(prompt="Hello.", n=2)
    assert len(results) == 1
    r = results[0]
    assert r.probe_type == "generation_stability"
    assert "lengths" in r.notes
    assert "spread" in r.notes
    assert len(r.notes["lengths"]) == 2


def test_run_all_returns_report(harness):
    report = harness.run_all(prompts=["Hello.", "Who are you?"])
    assert isinstance(report, ProbeReport)
    assert len(report.results) > 0
    assert 0.0 <= report.pass_rate <= 1.0


def test_report_has_all_probe_types(harness):
    report = harness.run_all(prompts=["Hello."])
    found_types = set(report.by_type.keys())
    expected = {
        "no_think_compliance",
        "no_reasoning_leak",
        "no_prompt_echo",
        "identity_presence",
        "generation_stability",
    }
    assert expected == found_types


def test_report_summary_is_string(harness):
    report = harness.run_all(prompts=["Hello."])
    summary = report.summary()
    assert isinstance(summary, str)
    assert "Backend" in summary
    assert "pass" in summary.lower() or "%" in summary


# ------------------------------------------------------------------
# Contract tests — these MUST hold even with random weights
# ------------------------------------------------------------------

def test_random_weights_never_produce_think_tags(harness):
    """Random weights should never produce <think> tags — if they do it's a bug."""
    results = harness.run_no_think_compliance(
        ["Hello.", "Who are you?", "Tell me something."]
    )
    for r in results:
        assert r.passed, (
            f"<think> tag found in random-weight output — "
            f"this should be structurally impossible.\nOutput: {r.answer}"
        )


def test_random_weights_never_echo_prompt_markers(harness):
    """Random weights should not produce [Persona], [Memory] etc. markers."""
    results = harness.run_no_prompt_echo(
        ["Hello.", "Who are you?", "Tell me something."]
    )
    for r in results:
        assert r.passed, (
            f"Prompt marker echoed in output.\nOutput: {r.answer}"
        )
