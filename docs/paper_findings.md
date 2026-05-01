# nova-mythos / mythos_lite — Research Findings Log

**Status:** working draft, in-flight. Findings are accumulated across training epochs.
**Updated through:** end of epoch 16 of run v1 (step 56,400; 1.85B tokens — Chinchilla target hit), plus epoch 1 of run v2 (clean restart, step 3,525) which exposed an architectural bug; v3 launches after the fix.
**Purpose:** preserve experimental observations and decisions across Claude Code sessions so they don't get lost.

This is a research log, not a paper draft. Sections are structured so they can be lifted into a paper if/when one is written.

---

## 0. Summary

A 93.5M-parameter Recurrent-Depth Transformer (mythos_lite, scaled-down variant of openmythos) plateaued at training loss ~5.0 after 12 epochs of pretraining (~1.4B tokens, FineWeb-Edu + local corpus). We hypothesized the plateau was learning-rate-bound (cosine schedule had bottomed at min_lr = 3e-5 for 5 consecutive epochs). We ran a five-signal diagnostic to confirm the recurrent / latent-reasoning machinery was functioning before intervening. The diagnostic showed the architecture was healthy; the plateau was not architectural. We then ran a combined intervention: LR warm-restart (peak 1.5e-4) and a data-distribution shift introducing Cosmopedia at 30% (later 60%). Loss dropped from 5.0 → 4.37 → 3.99 over two epochs (~0.6 nats and ~0.4 nats respectively).

We initially attributed the ACT halting collapse and depth-extrapolation loss observed at epochs 13–16 to the optimizer reset performed during the warm-restart. **A fresh-start run (v2) with no warm-restart contradicted that diagnosis: ACT collapsed to mass 0.083 within a single epoch.** The actual cause is an architectural bug in `RecurrentBlock.forward` — the per-loop ACT remainder is only assigned when cumulative mass crosses threshold mid-loop; for tokens that never cross threshold, no remainder is forced at the loop end, so `h_out` has mass < 1.0. The coda adapts by amplifying internally, creating a degenerate basin where "kill ACT" is the path of least resistance. Bug fixed; v3 run launching with the fix.

---

## 1. Model and training setup

### Architecture (mythos_lite, 93.5M params)
- Recurrent-Depth Transformer: prelude(2 layers) → recurrent block(8 loop iterations) → coda(2 layers)
- Multi-Latent Attention (MLA), n_heads=8, n_kv_heads=2
- MoE FFN inside the recurrent block: 8 routed experts (top-2) + 2 shared experts, expert_dim=512
- LTI-stable input injection: h_{t+1} = A·h_t + B·e + Transformer(h_t, e), spectral radius ρ(A) < 1 by construction
- ACT (Adaptive Computation Time) halting predictor: per-token sigmoid head, threshold 0.99
- Per-loop LoRA adapter (rank 4) for depth differentiation
- Tokenizer: GPT-2 (vocab 50,257)
- Architectural constants (n_experts=8, max_loop_iters=8, etc.) inherited from openmythos and not changed

### Hardware / throughput
- 2× NVIDIA RTX 3060 12GB, DDP via torchrun
- seq_len=512, micro_batch=8, grad_accum=4 → global batch 32,768 tokens/step
- Sustained ~11.6k tok/s
- ~2.8 hours per 3,525-step epoch

### Data
- **Phase 1 (epochs 1–5):** local corpus only (~115M tokens, 894 .txt files of educational/political documents)
- **Phase 2 (epochs 6–12):** mixed corpus + FineWeb-Edu, ratio dropping from 30% → 15% corpus over time
- **Phase 3 (epochs 13+):** mixed corpus + Cosmopedia (web_samples_v2) + FineWeb-Edu
  - Epoch 13: 10% corpus / 30% Cosmopedia / 60% FineWeb-Edu
  - Epoch 14+: 5% corpus / 60% Cosmopedia / 35% FineWeb-Edu

### LR schedule
- AdamW with linear warmup → cosine decay
- Phase 1–2: peak_lr=3e-4, min_lr=3e-5, warmup=2000 steps, cosine over total_steps
- Phase 3: warm-restart to peak_lr=1.5e-4, warmup=200, optimizer state reset (`--restart-lr`)

---

## 2. Phase 1 Observation: the plateau

Final-window loss per epoch:

| Epoch | Loss floor | Data mix | LR (end) |
|------:|:-----------|:---------|:---------|
| 1     | ~4.2       | corpus only | warmup |
| 2     | ~4.2       | corpus only | full |
| 3     | ~4.7       | mixed 30/70 | full |
| 4     | ~3.9       | corpus only | decaying |
| 5     | ~3.1       | corpus only | decaying |
| 6     | ~4.63      | mixed 30/70 | decaying |
| 7     | ~4.67      | mixed 30/70 | decaying |
| 8     | ~4.86      | mixed 15/85 | low |
| 9     | ~4.86      | mixed 15/85 | low |
| 10    | ~5.00      | mixed 15/85 | min |
| 11    | ~5.00      | mixed 15/85 | min |
| 12    | ~5.00      | mixed 15/85 | min |

(Note: loss is not directly comparable across data mixes — corpus-only is harder, mixed is easier, etc.)

The salient observation: from epoch 8 onward the loss stopped descending. By epoch 10–12 it had a tight 4.95–5.10 band with no progress. The cosine schedule had bottomed out at min_lr (3e-5) in this region.

---

## 3. Phase 2: Five-signal diagnostic

To distinguish "architecture is broken" from "optimizer is starved," we built `scripts/diagnose_lite.py`. It loads a checkpoint and runs five independent probes against a held-out batch from the corpus binary.

### Tests
1. **Loss vs n_loops at inference.** Compute cross-entropy at n_loops ∈ {1, 2, 4, 8, 16}. Flat curve = recurrent block earning no compute.
2. **ACT halting distribution.** Forward a batch; capture per-iteration halting probabilities; compute halt step per token. Mean halt step << max_loops = ACT collapsed (predictor untrained).
3. **MoE routing distribution.** Hook the MoE forward pass; count which experts get top-K selected. One expert >50% = collapse.
4. **LTI A diagonal.** Inspect the discretized state matrix. Median ≈ 0 = state forgotten each loop. Median ≈ 1 = state never updated.
5. **LoRA per-loop scale.** Compare per-loop scale vector cosine similarity to loop-0. All ≈ 1 = loops are identical.

### Findings on `checkpoint-0042300` (end of epoch 12)

| Test | Result | Interpretation |
|---|---|---|
| 1. Loss vs n_loops | n_loops=1 → 6.23, n_loops=8 → 4.88, n_loops=16 → 4.88 | Recurrence does 1.34 nats of work; trained depth is fully exploited; no extrapolation past 8 |
| 2. ACT halting | mean halt step 6.91; 78% halt at step 7; 12% never halt | Healthy — uses near-full depth on most tokens |
| 3. MoE routing | Experts 0,3 ~21%; experts 4,5,1,7 ~13–17%; **experts 2,6 ~0.4%** (dead) | Partial collapse: 6/8 experts active, 2 dead |
| 4. LTI A diagonal | median 0.35, range 0.34–0.37 | Healthy moderate-decay regime |
| 5. LoRA per-loop | strong cos differentiation; loops 3–5 anti-correlated with loop 0 (≤ -0.9) | Healthy — loop iterations are functionally distinct |

**Conclusion:** the architecture is functioning. The recurrent block contributes 1.34 nats of loss reduction; ACT is using near-full depth; LoRA differentiates iterations; LTI is in a stable regime. Two MoE experts are dead but six are active and balanced. The plateau is not architectural.

This led to the working hypothesis: the plateau is **optimization-side**, specifically that min_lr=3e-5 is too low to drive further weight updates.

---

## 4. Phase 3: Combined intervention (epoch 13)

We executed two interventions simultaneously (acknowledging the variable-confound cost in exchange for speed):

### Interventions
1. **LR warm-restart** via new `--restart-lr` flag in `train_lite.py`. Loads model weights from checkpoint, but skips loading optimizer and scheduler state. New peak_lr=1.5e-4, warmup=200 steps. This unfreezes weights without trashing them.
2. **Cosmopedia introduction.** Extended `MixedDataset` to support a third stream (`HuggingFaceTB/cosmopedia`, config `web_samples_v2`). Mix changed from 15/85 corpus/FineWeb-Edu to **10% corpus / 30% Cosmopedia / 60% FineWeb-Edu**.

### Code changes
- `training/train_lite.py`:
  - Added `CosmopediaDataset(IterableDataset)`
  - Extended `MixedDataset.__init__` and `__iter__` to handle 3-way mixing via `cosmopedia_ratio`
  - Added `--cosmopedia-ratio` CLI flag (default 0.0 = disabled)
  - Added `--restart-lr` CLI flag — modifies `load_checkpoint()` to skip optimizer/scheduler state restoration
  - Updated `dataset_desc` print to reflect 3-way mix

### Result over 3,525 steps (epoch 13)

| Step | Loss | LR | Note |
|---:|:---|:---|:---|
| 42,350 | 5.25 | 3.75e-5 | warmup; first reading after restart, small bump |
| 42,500 | 5.10 | 1.50e-4 | warmup complete |
| 42,650 | 4.99 | 1.50e-4 | crossed below the 5.0 floor |
| 43,000 | 4.78 | 1.50e-4 | actively descending |
| 44,000 | 4.68 | 1.50e-4 | trend continues |
| 45,000 | 4.50 | 1.49e-4 | (cosine begins gentle decay) |
| 45,825 | **4.37** | 1.49e-4 | end of epoch 13 |

**Net Δ:** 5.00 → 4.37 in one epoch (~0.63 nats). For comparison, epochs 4–12 combined had moved loss less than this.

---

## 5. Phase 4: Cosmopedia ramp (epoch 14)

After epoch 13, with no LR restart, we ramped Cosmopedia to 60% (target mix from the original Phase-3 plan):

- Mix: **5% corpus / 60% Cosmopedia / 35% FineWeb-Edu**
- LR continues natural cosine decay from 1.49e-4
- `--total-steps 49350` (one more epoch)
- Optimizer and scheduler state preserved (no `--restart-lr`)

### Result over 3,525 steps (epoch 14)

| Step | Loss | LR | Note |
|---:|:---|:---|:---|
| 45,850 | **3.96** | 1.49e-4 | first reading; immediate drop from ~4.37 |
| 46,000 | 4.19 | 1.49e-4 | settling |
| 47,000 | 4.02 | 1.48e-4 | descending |
| 48,000 | 4.03 | 1.46e-4 | tight band ~4.0 |
| 49,000 | 3.96 | 1.45e-4 | floor visible |
| 49,350 | **3.99** | 1.44e-4 | end of epoch 14 |

**Net Δ end-of-epoch:** 4.37 → 3.99 (~0.38 nats).

### ⚠ Important caveat: data-shift confound

The very first measurement of epoch 14 (step 45,850, before any gradient updates on the new mix) was **already 3.96**, lower than anything in epoch 13. This is not learning — the model is identical to checkpoint-0045825. It is the data-distribution shift: Cosmopedia text is more structured / lower-entropy than FineWeb-Edu, so per-token cross-entropy is mechanically lower.

Decomposing the ~0.4 nat drop attributable to epoch 14:
- ~0.1 nats: data shift (mechanical, no learning)
- ~0.3 nats: actual within-epoch learning

For future epochs, **a held-out fixed-distribution eval set is needed** to get clean across-epoch loss comparisons. Planned but not yet built (`scripts/eval_lite.py`).

### Epochs 15 and 16 (clean extension of same regime)

Same data mix (5/60/35), no LR restart, natural cosine decay continuing from epoch 14.

| Epoch | End loss | Δ (clean) | Tokens | LR end |
|---:|:---|:---|:---|:---|
| 15 | **3.87** | −0.12 | 1.73B | 1.39e-4 |
| 16 | **3.79** | −0.08 | 1.85B (Chinchilla) | 1.33e-4 |

Both epochs ran without data-mix or LR changes — the deltas are pure learning. The diminishing returns pattern (0.12 → 0.08) suggests a plateau in the 3.5–3.7 range over the next several epochs if the trend continues.

**Loss table consolidated post-warm-restart (clean trajectory):**

| Epoch | End loss | Δ vs prior | Confound |
|---:|:---|:---|:---|
| 12 | 5.00 | flat | — |
| 13 | 4.37 | −0.63 | LR restart + data shift |
| 14 | 3.99 | −0.38 | data shift (~0.1 mechanical) |
| 15 | 3.87 | −0.12 | none — clean |
| 16 | 3.79 | −0.08 | none — clean |

---

## 6. Side effects observed in diagnostics

Diagnostic series across checkpoints (epoch 12, 13, 16):

| Signal | Ep 12 | Ep 13 | Ep 16 | Trajectory |
|---|---|---|---|---|
| Loss @ n_loops=8 | 4.88 | 4.70 | **4.56** | descending |
| Loss @ n_loops=16 | 4.88 | 4.72 | **4.82** | ⚠ now *worse* than n=8 |
| Recurrence span (1→8) | 1.34 | 1.36 | **1.47** | block doing more work |
| **ACT mean halt step** | 6.91 | 7.86 | **8.00** | ⚠ no recovery; full collapse |
| **% tokens never halt** | 11.7% | 91.2% | **99.9%** | ⚠ ACT dead |
| Top MoE expert load | 22.0% | 21.2% | 21.2% | stable |
| Expert 2 routing | 0.4% | 1.6% | **4.2%** | ✓ recovering |
| Expert 6 routing | 0.2% | 1.4% | **2.8%** | ✓ recovering |
| LTI A median | 0.350 | 0.337 | 0.335 | stable, healthy |
| LoRA loop differentiation | strong | strong | strong (frozen) | stabilized |

### ACT collapse — initial misdiagnosis, then the real cause

**Initial diagnosis (later corrected):** When `--restart-lr` reset the optimizer, the ACT halting predictor's Adam moments were zeroed along with everything else. Under the high LR (1.5e-4), the halting head was pushed toward predicting near-zero per-iteration halt probabilities. As of checkpoint-0045825, **91.2% of tokens never reach the cumulative threshold within 8 iterations.**

Implication: the model now effectively runs full recurrent depth on every token (no adaptive compute). This costs ~12% extra compute per forward pass but is functionally fine — the per-loop weighting just becomes near-uniform across iterations. Loss did not regress (it dropped sharply), so this is an artifact of the warm-restart, not active harm.

**Update (epoch 16):** ACT did *not* self-recover under cosine LR decay. The mean halt step drifted further to 8.00 (99.9% of tokens never halt). The epoch-13 prediction of slow recovery was wrong — the implicit gradient pressure from under-magnitude hidden states is not strong enough to overcome the high-LR-induced bias.

### Real diagnosis: architectural bug (run v2 epoch 1)

After three epochs of inability to recover ACT, we ran a fresh "clean" pretraining run (v2) with no warm-restart, no `--restart-lr`, fresh weights, and a 20% corpus / 60% Cosmopedia / 20% FineWeb-Edu mix. Hypothesis: ACT collapse was caused by `--restart-lr` zeroing the halting head's Adam moments, and a fresh run would show normal ACT behaviour.

**The hypothesis was wrong.** Diagnostic on `runs/lite-v2/checkpoint-0003525` (end of v2 epoch 1) showed:

| Signal | v2 ep 1 | v1 ep 16 (broken) |
|---|---|---|
| Final ACT mass mean | 0.083 | 0.327 |
| % tokens never halt | 100% | 99.9% |
| Mean halt step | 8.00 | 8.00 |
| Loss @ n_loops=8 | 5.00 | 4.56 |
| Loss @ n_loops=16 | 5.43 | 4.82 |
| Depth extrap gap | +0.43 nats | +0.26 nats |

**ACT collapsed within a single epoch of clean training.** The warm-restart was not the cause — it was an accelerant on top of an underlying architectural failure mode.

The actual cause is in `RecurrentBlock.forward` — the per-loop ACT weighting code:

```python
remainder = (1.0 - cumulative_p).clamp(min=0)
weight = torch.where(
    cumulative_p + p >= self.cfg.act_threshold,
    remainder,    # final mass when threshold crossed mid-loop
    p,            # otherwise, just contribute p
)
weight = weight * still_running.to(dtype=h.dtype)
h_out = h_out + weight.unsqueeze(-1) * h
```

The remainder trick only fires when threshold is crossed *mid-loop*. If a token's cumulative mass stays below threshold through all `max_loop_iters` iterations, no remainder is ever assigned. `h_out` ends up with mass equal to `sum(p_t)` < threshold, which can be arbitrarily small (we observed 0.025 minimum, 0.083 mean).

The coda is downstream of `h_out`. Underweighted inputs are systematically smaller in magnitude, but the coda's RMSNorm + linear transformations can learn to compensate by increasing internal scale. The model ends up in a stable equilibrium:

- Halting head learns to output near-zero `p`
- ACT mass stays near zero
- `h_out` has tiny magnitude
- Coda has internalized a 12× amplification (1/0.083) to recover

This is a **degenerate basin** where adaptive halting has been quietly disabled, but loss has been minimized. The gradient signal that *should* push the halting head to fire (under-magnitude `h_out`) is gradient-equivalent to the coda's internal amplification, so the optimizer has no preference between "use ACT properly" and "kill ACT and let coda compensate." It picks the simpler path.

This is a real architectural bug, not a training-time anti-pattern. Original ACT formulation (Graves 2016) requires unit mass; this implementation provided no mechanism to enforce it at loop termination.

### The fix

In `RecurrentBlock.forward`, force the remainder assignment on the final loop iteration regardless of whether threshold is crossed:

```python
is_final_loop = t == n_loops - 1
remainder = (1.0 - cumulative_p).clamp(min=0)
weight = torch.where(
    (cumulative_p + p >= self.cfg.act_threshold) | is_final_loop,
    remainder,
    p,
)
```

Verified via smoke test: with this fix, sum of weights contributing to `h_out` is exactly 1.0 for every token, regardless of halting head behaviour. This removes the architectural escape hatch.

**What the fix does and does not solve:**
- ✓ Solves: unit-mass correctness — `h_out` always has mass 1.0
- ✓ Solves: the "kill ACT" gradient escape hatch
- ✗ Does not solve: ACT may still learn "always halt at the final iteration" (which is fine functionally — depth-8 deterministic with proper output scaling — but doesn't recover *adaptive* halting)

Adaptive halting usefulness is a separate problem, addressable later via ponder loss or training-time depth randomization. Fix the bug first, observe whether useful halting emerges naturally, intervene only if it doesn't.

### Loss of depth extrapolation — a follow-on consequence of ACT collapse

A previously-unanticipated finding emerged at epoch 16. With ACT firing normally, the recurrent block sees a distribution of effective depths during training (most tokens at full depth, some at fewer). When ACT collapses, every token uses exactly `max_loop_iters` depth, every step.

The model then over-specializes to depth=8:

| Checkpoint | Loss @ n_loops=8 | Loss @ n_loops=16 | Δ |
|---|---|---|---|
| Epoch 12 (ACT healthy) | 4.88 | 4.88 | 0.00 (neutral) |
| Epoch 13 (post-restart) | 4.70 | 4.72 | +0.02 (slight degradation) |
| Epoch 16 (ACT dead) | 4.56 | 4.82 | **+0.26 (significant)** |

This is a real architectural cost: **ACT collapse loses the depth-extrapolation property** that motivated the Recurrent-Depth Transformer design. The model still benefits from depth at training depth but no longer scales up at inference. For the paper, this strengthens the argument that ACT (or some replacement form of depth-randomization during training) is structurally important to RDT models — not optional.

We have not intervened to restore ACT. Whether the remaining pretraining (epochs 17–21) sees this degrade further or stabilize is a tracked research observation.

### MoE expert recovery — confirmed

Experts 2 and 6 trajectory:

| Checkpoint | Expert 2 | Expert 6 | Combined |
|---|---|---|---|
| Epoch 12 | 0.4% | 0.2% | **0.6%** |
| Epoch 13 | 1.6% | 1.4% | **3.0%** |
| Epoch 16 | 4.2% | 2.8% | **7.0%** |

Both dead experts are climbing steadily. The aux-loss-free load-balancing mechanism (DeepSeek-V3 style) is doing its job — given a fresh optimizer state and continued training, it pulls under-utilized experts back toward the balanced 12.5% target. Expert 2 may reach near-balance by epoch 20–22 if the trajectory holds.

**Paper-relevant takeaway:** MoE collapse is *not* permanent under aux-loss-free balancing. The mechanism is robust to optimizer perturbations and recovers from soft collapses. This contrasts with the ACT collapse, which did not self-heal — suggesting the load-balancing bias is structurally more robust than the implicit gradient pressure on the ACT halting head.

---

## 7. Probe outputs — qualitative coherence

`scripts/probe_lite.py` runs a fixed 8-prompt suite at fixed seed, saving to `results/probes/<checkpoint>.txt`. Selected comparisons across the intervention:

### "Q: What is the largest planet in our solar system?"
- **Epoch 12** (loss ~5.0): "Q: What you can do is that the world?" — does not engage with the topic.
- **Epoch 13** (loss ~4.4): "A: What you can't do about the solar solar system... The space at the start of the Universe." — engages with topic (planets, solar system, Earth, Universe).
- **Epoch 16** (loss ~3.8): produces a coherent list of follow-up questions — "What is the source of the human mind? What are the functions of the solar system? What are the components of the solar system?..." — recognizes the question structure and stays on-topic for ~10 lines, but still does not answer "Jupiter."

### "Photosynthesis is the process by which"
- **Epoch 13**: "the system of the material can be used? It is hard to see how it could be done with the data on the field of the system" — generic word salad.
- **Epoch 16**: "electrons to be transmitted to the mitochondria are formed and the molecules can be altered and the molecules move through the cell. This results in the formation and formation of the cells into smaller regions of the cells. The cell has an essential function..." — actual scientific vocabulary, topic-coherent for a full paragraph. Notably wrong about photosynthesis (it confuses it with cellular respiration / mitochondria), but it is producing biology textbook prose.

### "The most important thing to remember about learning is"
- **Epoch 12**: generic life-advice word salad ("learn how you can help you to understand the facts").
- **Epoch 13**: textbook prose about teaching/lessons/students. Coherent, register matches Cosmopedia.

### "def fibonacci(n):"
- **Epoch 12**: garbled tokens mixed with prose.
- **Epoch 13**: collapsed entirely — only whitespace produced. Cosmopedia has minimal code; the small share of code in FineWeb-Edu is no longer enough to maintain the capability.

### Style observation
The Cosmopedia introduction is visibly shifting the model's output register toward textbook/explanatory English. For an agent-brain target use case (Nova2), this is a desired direction: agents primarily generate explanatory and instructional text. The code regression is an expected and acceptable tradeoff.

---

## 8. Architecture lineage and intentional constants

mythos_lite is a directly scaled-down variant of openmythos. The following are inherited from the parent architecture and **not** treated as free hyperparameters:
- n_experts = 8
- n_shared_experts = 2
- n_experts_per_tok = 2 (top-K)
- max_loop_iters = 8
- prelude_layers = 2, coda_layers = 2

When dead experts were diagnosed, pruning was considered and rejected to preserve architectural fidelity across the variants family (lite → 1b → 3b → 10b → 50b → 100b → 500b → 1t in `variants.py`).

---

## 9. Open questions / next experiments

1. **Held-out eval set.** Build `scripts/eval_lite.py` with fixed batches from FineWeb-Edu, Cosmopedia, and corpus. Run at every checkpoint. Required for clean across-epoch loss tracking.
2. **~~ACT recovery trajectory.~~** **Resolved (negative result):** ACT did *not* self-recover. By epoch 16 it was fully collapsed (99.9% never halt) and depth extrapolation degraded by 0.26 nats. Recovery now requires explicit intervention.
3. **~~Dead expert recovery.~~** **Resolved (positive result):** experts 2 and 6 climbed from 0.6% combined → 7.0% combined over epochs 12–16. The aux-loss-free load balancing is robust. Expect near-balance by epoch ~20.
4. **Genuine loss floor.** Clean trajectory shows 0.12 → 0.08 nat-per-epoch gains. Extrapolation suggests plateau in **3.5–3.7** range over epochs 17–21. Track via clean-epoch deltas (no LR or data changes through ep 21).
5. **Phase 4 plan: post-Chinchilla extension (epochs 17–21).** Drop corpus from 5% → 1% (token exposure already saturated; 6+ passes baked in). Mix becomes **1% corpus / 65% Cosmopedia / 34% FineWeb-Edu**. No LR restart. Watch for floor stall over 5 epochs. Re-warm only if loss flattens for 3 consecutive epochs.
6. **SFT phase.** Once pretraining plateaus, fine-tune on instruction-following data (mix of public Alpaca-style + synthetic Q&A from corpus). This is the phase that turns "coherent text completer" into "Nova2 agent brain."
7. **Counterfactual single-variable runs (paper completeness).** The combined intervention conflates LR and data. Could re-run from `checkpoint-0042300` with only the LR change (or only the data change) on a separate output dir to disentangle. Not strictly required, but would tighten the ablation story.
8. **ACT remediation (deferred).** If by end of pretraining we want depth extrapolation back, options are: (a) auxiliary ponder loss on the halting head, (b) put halting head in its own optimizer param group with 10–100× lower LR, (c) cold-reinitialize the halting head and let it relearn. Each is a small intervention; deferred until pretraining is otherwise complete.

---

## 10. File map

| Artifact | Path | Purpose |
|---|---|---|
| Training script | `training/train_lite.py` | Pretraining; supports mixed 3-way data, --restart-lr, --cosmopedia-ratio |
| Probe suite | `scripts/probe_lite.py` | 8 fixed prompts; saves to `results/probes/<ckpt>.txt` |
| Diagnostic | `scripts/diagnose_lite.py` | 5-signal architectural health check; saves to `results/diagnostics/<ckpt>.txt` |
| REPL | `scripts/repl.py` | Interactive inference (currently premature — output not yet coherent) |
| Generate | `scripts/generate.py` | Single-prompt CLI generation |
| Probes (per ckpt) | `results/probes/checkpoint-XXXXX.txt` | Reproducible output samples |
| Diagnostics (per ckpt) | `results/diagnostics/checkpoint-XXXXX.txt` | Recurrent/MoE/ACT/LTI/LoRA health |
| Session logs | `session_log_2026-04-28.txt`, etc. | Day-by-day training notes |
| This document | `docs/paper_findings.md` | Persistent research log |

---

## 11. Reproducibility notes

- Random seed = 42 throughout (training, sampling, eval batches)
- All checkpoints save model + optimizer + scheduler state + config; warm-restart uses model only via `--restart-lr`
- Probe suite uses temperature=0.8, top_k=40, n_loops=8, max_new_tokens=120
- Diagnostic uses 4×512 batch from end of `data/corpus/corpus.bin` (held-out region not seen frequently in early epochs but no formal split)
- Hardware: 2× RTX 3060 12GB, PCIe (no NVLink), no special tuning beyond `find_unused_parameters=True` for ACT-driven flow control in DDP
