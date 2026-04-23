# mythos_lite — Design, Experimentation, and Results

This document records the full arc of the `mythos_lite` experiment: why the 1B model could not train on the target hardware, how the lite variant was designed, the benchmark methodology, the failures encountered along the way, and the final measured results.

---

## 1. Context — the hardware target

The goal of nova-mythos is to provide a training-capable RDT backend for consumer hardware. The target is a dual **NVIDIA RTX 3060 12GB** system — two consumer cards connected over PCIe, 24 GB total VRAM, no NVLink.

This is the hardware available to researchers and developers who cannot afford cloud GPU clusters or professional cards.

---

## 2. Why the 1B model cannot train here

The starting point was `mythos_1b` — the smallest variant in the OpenMythos family at 1.101B parameters. All training benchmark configurations (10 total, sweeping sequence lengths 512–2048 and both AdamW fp32 and bitsandbytes 8-bit Adam, with and without gradient checkpointing) failed with out-of-memory errors on both single and dual GPU.

### The parameter distribution problem

```
Total params : 1.101B
  MoE/experts: 928M  (84%)
  Attention  : 25M   (2%)
  Embed/head : 103M  (9%)
  LTI/other  : 45M   (4%)
```

84% of the 1B model's parameters live in MoE expert FFNs. This creates an asymmetric memory problem: the MoE experts are cheap at *inference* (only top-K fire per token) but expensive at *training* (all expert weights must be held in the optimizer state regardless of which ones activated).

### Memory math for training (2-way FSDP FULL_SHARD)

With FSDP sharding the model and optimizer across 2 GPUs:

| Item | Per GPU |
|---|---|
| Sharded weights (bf16) | 1.10 GB |
| Sharded AdamW (fp32 m+v+master) | 6.61 GB |
| Static total | **7.71 GB** |
| All-gather peak (temp full-model gather) | +2.20 GB |
| Peak before activations | **9.91 GB** |
| Headroom for activations | **2.09 GB** |

With 8 recurrent loop iterations and sequence lengths of 512+, activation memory consistently exceeded the 2.09 GB headroom. Even the most conservative configurations (seq=512, micro_batch=1) failed because the optimizer state alone consumed 7.71 GB, leaving insufficient space for gradient accumulation buffers and NCCL communication overhead.

### Why 8-bit Adam did not help

bitsandbytes 8-bit Adam should theoretically halve optimizer state from 6.61 GB to 3.31 GB per GPU. In practice it failed for a different reason: bitsandbytes stores optimizer state in custom 8-bit quantized format, but FSDP flattens all parameters into a contiguous "flat parameter" tensor before sharding. The 8-bit optimizer state and FSDP's flat-param representation are incompatible — bitsandbytes silently falls back to fp32, producing `"Deallocating Tensor that still has live PyObject references"` warnings and giving no memory benefit. All 10 configs still OOM'd.

### Conclusion on the 1B model

The 1B OpenMythos architecture is not trainable on dual RTX 3060 12GB. The constraint is structural, not tunable: 928M parameters in MoE experts produce optimizer state that exceeds the available VRAM before a single token is processed. A different scale of model is needed.

---

## 3. Design of mythos_lite

The goal was to produce the smallest model that preserves the essential RDT character — recurrent depth, MLA attention, MoE FFN, ACT halting — while fitting comfortably in training memory on the target hardware.

### Design constraints

- Static memory (weights + AdamW optimizer) must be well under 6 GB per GPU on a single card, leaving ≥6 GB for activations
- The recurrent structure (prelude → RecurrentBlock → coda) must be preserved
- MLA attention must be preserved (not replaced with GQA) — it is architecturally central to the RDT design
- The MoE FFN must be preserved — even at reduced scale, the sparse routing is a key property

### Parameter reduction strategy

| Hyperparameter | `mythos_1b` | `mythos_lite` | Effect |
|---|---|---|---|
| `dim` | 2048 | **1024** | 4× fewer parameters in attention projections and embeddings |
| `n_experts` | 64 | **8** | 8× fewer expert FFNs |
| `expert_dim` | 2048 | **512** | 4× smaller per-expert hidden dimension |
| `n_experts_per_tok` | 4 | **2** | Half the expert activation breadth per token |
| `max_loop_iters` | 16 | **8** | Half the recurrent depth; halves activation accumulation |
| `n_heads` | 16 | **8** | Proportional to dim reduction |
| `n_kv_heads` | 4 | **2** | Proportional to dim reduction |
| `kv_lora_rank` | 256 | **128** | Proportional MLA compression |
| `q_lora_rank` | 512 | **256** | Proportional MLA compression |
| `lora_rank` | 8 | **4** | Smaller depth-wise LoRA adapters |

What was deliberately kept:
- `attn_type = "mla"` — Multi-Latent Attention preserved in all blocks
- `prelude_layers = 2`, `coda_layers = 2` — same framing structure
- `n_shared_experts = 2` — always-active shared experts kept
- `act_threshold = 0.99` — same ACT halting threshold

### Resulting parameter budget

With vocabulary overridden to 50257 (gpt2 tokenizer) for training:

```
Total params : 93.5M
  MoE/experts: 18.9M  (20%)
  Attention  : 6.4M   (7%)
  Embed/head : 51.5M  (55%)
  LTI/other  : 16.8M  (18%)
```

The dominant cost shifted from MoE experts (84% → 20%) to the embedding and LM head layers (9% → 55%). This is an artefact of using the gpt2 vocabulary (50,257 tokens) with a reduced model dimension (1024): the two linear layers that map between token space and model space account for `50257 × 1024 × 2 = 103M` weights — more than the entire rest of the model.

### Memory projection (2-way DDP, AdamW fp32)

| Item | Per GPU |
|---|---|
| Weights (bf16) | 0.19 GB |
| AdamW optimizer (fp32 m+v) | 1.12 GB |
| DDP gradient bucket (bf16) | 0.19 GB |
| Static total | **1.50 GB** |
| Headroom for activations | **10.50 GB** |

This is a comfortable training budget. The entire static overhead is smaller than what the 1B model's all-gather buffer alone required.

---

## 4. Benchmark methodology

All benchmarks use synthetic random token data (no dataset required). Each configuration runs 3 warmup steps followed by 10 measured steps. Results report mean step time, standard deviation, global tokens per second, peak VRAM, and estimated days to reach training targets.

Hardware: 2× NVIDIA RTX 3060 12GB, PCIe (no NVLink), Ubuntu, CUDA 12.x, PyTorch 2.x.

---

## 5. Single GPU benchmark

### Results

| Config | Step time | Throughput | VRAM | Est. 10B tokens |
|---|---|---|---|---|
| seq=512  mb=8  acc=4 | 2605ms ± 27 | **6.3k tok/s** | 8.47 GB | 18.4 days |
| seq=1024 mb=4  acc=4 | 2847ms ± 15 | 5.8k tok/s | 9.65 GB | 20.1 days |
| seq=2048 mb=1  acc=8 | 4062ms ± 23 | 4.0k tok/s | 6.43 GB | 28.7 days |
| seq=2048 mb=2  acc=4 +gc | 5581ms ± 54 | 2.9k tok/s | 3.36 GB | 39.4 days |
| seq=4096 mb=1  acc=8 +gc | 13809ms ± 78 | 2.4k tok/s | 4.35 GB | 48.8 days |

### Key observations

**Throughput leader: seq=512, mb=8** at 6.3k tok/s using 8.47 GB VRAM.

**The quadratic attention wall:** seq=512 mb=8 and seq=1024 mb=4 process the same number of tokens per step (16,384) but seq=1024 uses 9.65 GB vs 8.47 GB. The extra 1.18 GB comes from attention score matrices, which scale O(T²) per head per layer — and the RecurrentBlock runs 8 iterations. Going to seq=2048 with the same total token count would push past 12 GB, which is why seq=2048 requires mb=1 (halving the batch to halve the attention memory).

**Gradient checkpointing tradeoff:** At seq=2048, disabling grad-checkpointing (mb=1, no GC) gives 4.0k tok/s at 6.43 GB. Enabling it (mb=2, +gc) allows doubling the batch but cuts throughput to 2.9k tok/s and only uses 3.36 GB — a poor trade unless VRAM is critically constrained. GC recomputes activations during the backward pass instead of storing them, at a ~33% compute overhead.

**seq=4096 is viable** but slow at 2.4k tok/s. The 8 recurrent iterations × O(T²) attention at 4096 context length is computationally expensive even for a 93.5M model.

---

## 6. Dual GPU — First attempt: FSDP

The initial dual-GPU benchmark used Fully Sharded Data Parallel (FSDP), consistent with the 1B benchmark approach. FSDP shards model weights, gradients, and optimizer state across GPUs, reducing per-GPU static memory at the cost of all-gather communication during each forward pass.

### Failure mode

After approximately 420 all-gather operations (deep into config 1 of 9), both ranks hit a 600-second NCCL timeout:

```
[Rank 1] Watchdog caught collective operation timeout:
WorkNCCL(SeqNum=423, OpType=_ALLGATHER_BASE, ...) ran for 600031ms before timing out.
[Rank 0] WorkNCCL(SeqNum=425, OpType=_ALLGATHER_BASE, ...) ran for 600098ms before timing out.
```

Rank 0 was at sequence 425 and rank 1 at sequence 423 — they had desynchronised by 2 collective operations, causing an all-gather deadlock that neither rank could recover from.

### Root cause

FSDP FULL_SHARD issues an all-gather before each wrapped module's forward pass and a reduce-scatter after each backward pass. For a model with `ModuleWrapPolicy({TransformerBlock, RecurrentBlock})`, this means:

- 2 prelude TransformerBlocks × forward all-gather = 2 collectives
- RecurrentBlock forward = 1 all-gather, then *the ACT loop runs 8 iterations* of shared-weight operations
- 2 coda TransformerBlocks × forward all-gather = 2 collectives
- Backward mirrors this in reverse

With grad_accum=4, a single training step issues on the order of 40–50 collective operations. At 13+ steps of warmup + measure, the total reaches 500+. Any subtle timing asymmetry between ranks — from CUDA kernel scheduling, PCIe bus contention, or ACT halting producing slightly different execution graphs — accumulates over hundreds of collectives and eventually causes desynchronisation.

### Why FSDP is the wrong tool here

FSDP is designed for models where per-GPU static memory is the binding constraint. For `mythos_lite` at 1.50 GB static memory, FSDP provides no meaningful benefit: the model fits entirely on a single 12 GB card with 10.5 GB to spare. The communication overhead and synchronisation complexity of FSDP are entirely unnecessary.

---

## 7. Dual GPU — Second attempt: DDP

DistributedDataParallel (DDP) keeps a full copy of the model on each GPU and issues a single gradient all-reduce after each backward pass. There are no per-module all-gathers, no flat-parameter complications, and no synchronisation within the forward pass. Each GPU sees different data (different shard of the training set) and gradients are averaged across ranks once per optimizer step.

Memory per GPU with DDP:

| Item | Per GPU |
|---|---|
| Weights (bf16) | 0.19 GB |
| AdamW optimizer (fp32 m+v) | 1.12 GB |
| DDP gradient bucket (bf16) | 0.19 GB |
| **Static total** | **1.50 GB** |

This is identical to single-GPU training. The benefit of two GPUs is throughput (2× data processed per step, same wall-clock time), not memory.

### Results

| Config | Step time | Global tok/s | VRAM/GPU | Est. 10B tokens |
|---|---|---|---|---|
| seq=512  mb=8  acc=4 | 2868ms ± 57 | **11.4k** | 8.65 GB | **10.1 days** |
| seq=512  mb=16 acc=4 | OOM | — | — | — |
| seq=1024 mb=4  acc=4 | 3130ms ± 28 | 10.5k | 9.85 GB | 11.1 days |
| seq=1024 mb=8  acc=4 | OOM | — | — | — |
| seq=2048 mb=1  acc=8 | 4602ms ± 45 | 7.1k | 6.65 GB | 16.3 days |
| seq=2048 mb=2  acc=4 +gc | 6026ms ± 152 | 5.4k | 3.55 GB | 21.3 days |
| seq=2048 mb=4  acc=4 +gc | 10045ms ± 221 | 6.5k | 6.12 GB | 17.7 days |
| seq=4096 mb=1  acc=8 +gc | 14362ms ± 392 | 4.6k | 4.53 GB | 25.4 days |
| seq=4096 mb=2  acc=4 +gc | 13097ms ± 169 | 5.0k | 7.89 GB | 23.1 days |

### Key observations

**DDP scaling efficiency: 1.81×** on 2 cards (11.4k vs 6.3k single GPU). A perfect 2× would require zero communication overhead; the gradient all-reduce on 93.5M parameters across PCIe accounts for the ~10% deficit.

**OOM configs mirror single GPU.** seq=512 mb=16 and seq=1024 mb=8 both OOM at the same activation memory limits as single GPU — DDP does not help with activation memory, only with static memory (which was already fine). The quadratic attention constraint is unchanged.

**seq=2048 mb=4 +gc (6.5k tok/s)** outperforms seq=2048 mb=2 +gc (5.4k tok/s) despite processing twice as many tokens per step. At 10045ms vs 6026ms, the larger batch is only 1.67× slower — larger batches amortise the gradient checkpointing recompute overhead better, giving higher GPU utilisation per second.

**seq=4096 mb=2 +gc (5.0k tok/s)** is faster than seq=4096 mb=1 +gc (4.6k tok/s) for the same reason — and uses 7.89 GB vs 4.53 GB, a worthwhile trade.

---

## 8. Training viability verdict

| Claim | Status |
|---|---|
| mythos_lite fits in 12 GB VRAM for training | ✓ Confirmed (1.50 GB static, up to 9.85 GB peak) |
| mythos_lite trains on a single RTX 3060 | ✓ Confirmed (6.3k tok/s, ~18 days for 10B tokens) |
| mythos_lite trains on dual RTX 3060 (DDP) | ✓ Confirmed (11.4k tok/s, ~10 days for 10B tokens) |
| mythos_1b trains on dual RTX 3060 | ✗ All configs OOM — 9.91 GB static leaves no room for activations |
| FSDP is appropriate for mythos_lite | ✗ Deadlocks after ~420 collectives — DDP is correct at this scale |
| 8-bit Adam helps the 1B on FSDP | ✗ Incompatible with FSDP flat params; silently falls back to fp32 |

---

## 9. Hardware reference table

For reference when choosing a training configuration:

| Sequence length | Micro-batch/GPU | Grad accum | GC | VRAM/GPU | Global tok/s |
|---|---|---|---|---|---|
| 512 | 8 | 4 | No | 8.65 GB | 11,400 |
| 1024 | 4 | 4 | No | 9.85 GB | 10,500 |
| 2048 | 1 | 8 | No | 6.65 GB | 7,100 |
| 2048 | 4 | 4 | Yes | 6.12 GB | 6,500 |
| 4096 | 2 | 4 | Yes | 7.89 GB | 5,000 |

All measurements on 2× RTX 3060 12GB with DDP. The recommended starting configuration for a 10B token run is `seq=512, mb=8, acc=4` — best throughput, no grad-checkpointing overhead, 2.1 GB VRAM headroom.

---

## 10. Recommended training invocation

```bash
.venv/bin/torchrun --nproc_per_node=2 training/train_lite.py \
    --output-dir runs/lite-001 \
    --seq-len 512 \
    --micro-batch 8 \
    --grad-accum 4 \
    --total-steps 305176 \
    --warmup-steps 2000
```

For a longer-context variant (accepting lower throughput):

```bash
.venv/bin/torchrun --nproc_per_node=2 training/train_lite.py \
    --output-dir runs/lite-2048 \
    --seq-len 2048 \
    --micro-batch 1 \
    --grad-accum 8 \
    --total-steps 305176 \
    --warmup-steps 2000
```

Full training script documentation: [`training/train_lite.py`](../training/train_lite.py).
