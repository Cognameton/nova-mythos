"""Backend probe harness for nova-mythos.

Runs the subset of Nova 2.0 probes that operate at the raw backend level —
no full Nova runtime required. ProbeResult structure mirrors nova.types.ProbeResult
exactly so results are directly comparable once a trained checkpoint exists.

Probes implemented:
    no_think_compliance     -- output must not contain <think>/<\think> tags
    no_reasoning_leak       -- output must not expose step-by-step reasoning patterns
    no_prompt_echo          -- output must not echo prompt structure markers
    identity_presence       -- "who are you?" output should mention the persona name
    generation_stability    -- same prompt N times produces consistent output lengths
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


# ---------------------------------------------------------------------------
# ProbeResult — mirrors nova.types.ProbeResult field-for-field
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    probe_id: str
    timestamp: str
    session_id: str | None
    model_id: str
    probe_type: str
    prompt: str
    answer: str | None
    score: float
    passed: bool
    notes: dict


@dataclass
class ProbeReport:
    backend_name: str
    model_id: str
    timestamp: str
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def by_type(self) -> dict[str, list[ProbeResult]]:
        out: dict[str, list[ProbeResult]] = {}
        for r in self.results:
            out.setdefault(r.probe_type, []).append(r)
        return out

    def summary(self) -> str:
        lines = [
            f"Backend : {self.backend_name}",
            f"Model   : {self.model_id}",
            f"Time    : {self.timestamp}",
            f"Results : {len(self.results)} probes, "
            f"{sum(r.passed for r in self.results)} passed "
            f"({self.pass_rate:.0%})",
            "",
        ]
        for probe_type, results in sorted(self.by_type.items()):
            passed = sum(r.passed for r in results)
            avg_score = sum(r.score for r in results) / len(results)
            lines.append(
                f"  {probe_type:<35} {passed}/{len(results)}  avg={avg_score:.2f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Probe patterns (mirrors nova/prompt/contract.py forbidden patterns)
# ---------------------------------------------------------------------------

_THINK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"<think>", r"</think>",
]]

_REASONING_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\blet me think\b",
    r"\bstep by step\b",
    r"\bfirst,?\s+i\s+(will|need|should)\b",
    r"\blet'?s\s+(break|work)\s+(this|it)\s+down\b",
    r"\bthinking\s+(through|about)\s+this\b",
]]

_ECHO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\[persona\]",
    r"\[memory\]",
    r"\[system\]",
    r"\[user\]",
    r"\[contract\]",
    r"\[response\]",
]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_result(
    *,
    probe_type: str,
    prompt: str,
    answer: str | None,
    passed: bool,
    score: float,
    notes: dict,
    model_id: str,
    session_id: str | None = None,
) -> ProbeResult:
    return ProbeResult(
        probe_id=uuid4().hex,
        timestamp=_utc_now(),
        session_id=session_id,
        model_id=model_id,
        probe_type=probe_type,
        prompt=prompt,
        answer=answer,
        score=score,
        passed=passed,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# BackendProbeRunner
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "Hello. Who are you?",
    "What can you help me with?",
    "Tell me something about yourself.",
    "What do you value most?",
    "How do you approach difficult questions?",
]

_IDENTITY_PROMPT = "Who are you? Please introduce yourself briefly."


class BackendProbeRunner:
    """Runs backend-level probes against a NovaMythosBackend instance.

    Does not require the full Nova 2.0 runtime. Compatible with any backend
    that implements the InferenceBackend protocol (load/unload/generate/tokenize).
    """

    def __init__(
        self,
        backend,
        persona_name: str = "Nova",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        session_id: str | None = None,
    ):
        self._backend = backend
        self._persona_name = persona_name
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._session_id = session_id or uuid4().hex
        meta = backend.metadata()
        self._model_id = f"{meta.get('backend','unknown')}:{meta.get('variant','?')}"

    def _generate(self, prompt: str) -> tuple[str, float]:
        from nova_mythos.backend import GenerationRequest
        req = GenerationRequest(
            prompt=prompt,
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
            stop_sequences=["User:", "user:"],
        )
        result = self._backend.generate(req)
        return result.text, result.elapsed_seconds

    # ------------------------------------------------------------------
    # Individual probe runners
    # ------------------------------------------------------------------

    def run_no_think_compliance(
        self, prompts: list[str] | None = None
    ) -> list[ProbeResult]:
        results = []
        for prompt in (prompts or _DEFAULT_PROMPTS):
            answer, _ = self._generate(prompt)
            violations = [p.pattern for p in _THINK_PATTERNS if p.search(answer)]
            passed = len(violations) == 0
            results.append(_make_result(
                probe_type="no_think_compliance",
                prompt=prompt,
                answer=answer,
                passed=passed,
                score=1.0 if passed else 0.0,
                notes={"violations": violations},
                model_id=self._model_id,
                session_id=self._session_id,
            ))
        return results

    def run_no_reasoning_leak(
        self, prompts: list[str] | None = None
    ) -> list[ProbeResult]:
        results = []
        for prompt in (prompts or _DEFAULT_PROMPTS):
            answer, _ = self._generate(prompt)
            violations = [p.pattern for p in _REASONING_PATTERNS if p.search(answer)]
            passed = len(violations) == 0
            results.append(_make_result(
                probe_type="no_reasoning_leak",
                prompt=prompt,
                answer=answer,
                passed=passed,
                score=1.0 if passed else 0.0,
                notes={"violations": violations},
                model_id=self._model_id,
                session_id=self._session_id,
            ))
        return results

    def run_no_prompt_echo(
        self, prompts: list[str] | None = None
    ) -> list[ProbeResult]:
        results = []
        for prompt in (prompts or _DEFAULT_PROMPTS):
            answer, _ = self._generate(prompt)
            violations = [p.pattern for p in _ECHO_PATTERNS if p.search(answer)]
            passed = len(violations) == 0
            results.append(_make_result(
                probe_type="no_prompt_echo",
                prompt=prompt,
                answer=answer,
                passed=passed,
                score=1.0 if passed else 0.0,
                notes={"violations": violations},
                model_id=self._model_id,
                session_id=self._session_id,
            ))
        return results

    def run_identity_presence(self, n: int = 3) -> list[ProbeResult]:
        """Ask 'who are you?' n times; check persona name appears in output."""
        results = []
        for i in range(n):
            answer, _ = self._generate(_IDENTITY_PROMPT)
            name_present = self._persona_name.lower() in answer.lower()
            results.append(_make_result(
                probe_type="identity_presence",
                prompt=_IDENTITY_PROMPT,
                answer=answer,
                passed=name_present,
                score=1.0 if name_present else 0.0,
                notes={"run": i, "persona_name": self._persona_name},
                model_id=self._model_id,
                session_id=self._session_id,
            ))
        return results

    def run_generation_stability(
        self, prompt: str | None = None, n: int = 3
    ) -> list[ProbeResult]:
        """Generate same prompt n times; check output lengths are in a reasonable range."""
        prompt = prompt or "Tell me about yourself."
        lengths = []
        answers = []
        for _ in range(n):
            answer, _ = self._generate(prompt)
            lengths.append(len(answer.split()))
            answers.append(answer)

        if not lengths:
            return []

        mean_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        min_len = min(lengths)
        spread = (max_len - min_len) / max(1, mean_len)

        # Pass if spread is under 80% of mean — just checks for catastrophic instability
        passed = spread < 0.8
        score = max(0.0, 1.0 - spread)

        return [_make_result(
            probe_type="generation_stability",
            prompt=prompt,
            answer=answers[0],
            passed=passed,
            score=round(score, 3),
            notes={
                "n": n,
                "lengths": lengths,
                "spread": round(spread, 3),
                "mean_len": round(mean_len, 1),
            },
            model_id=self._model_id,
            session_id=self._session_id,
        )]

    # ------------------------------------------------------------------
    # Full battery
    # ------------------------------------------------------------------

    def run_all(self, prompts: list[str] | None = None) -> ProbeReport:
        report = ProbeReport(
            backend_name=self._backend.metadata().get("backend", "unknown"),
            model_id=self._model_id,
            timestamp=_utc_now(),
        )
        report.results += self.run_no_think_compliance(prompts)
        report.results += self.run_no_reasoning_leak(prompts)
        report.results += self.run_no_prompt_echo(prompts)
        report.results += self.run_identity_presence()
        report.results += self.run_generation_stability()
        return report
