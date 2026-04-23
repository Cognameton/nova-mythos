# nova-mythos

An implementation of the [OpenMythos](https://github.com/kyegomez/OpenMythos) Recurrent-Depth Transformer (RDT) architecture, packaged as a drop-in inference backend for [Nova 2.0](https://github.com/Cognameton/nova). Includes a consumer-hardware-viable training variant (`mythos_lite`), a full benchmark suite, and a production training pipeline targeting FineWeb-Edu.

> **Assisted by:** Claude (Anthropic) and Codex (OpenAI) were used throughout this project for architecture analysis, debugging, and implementation.

---

## What is the Recurrent-Depth Transformer?

Standard transformers process a sequence through a fixed stack of layers — one pass, one representation. The RDT replaces the stack with a single recurrent block that iterates over itself, governed by a continuous-depth update rule:

```
h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)
```

Where `h_t` is the hidden state at iteration `t`, `e` is the embedded input, `A` is a learnable state-transition matrix (spectral radius kept below 1 for stability), and `B` is a learnable input-injection matrix.

Rather than adding more layers, the model adds more *thinking time* — the same weights run repeatedly, each pass refining the representation. An Adaptive Computation Time (ACT) halting mechanism lets the model exit early on easy tokens and run the full depth on hard ones.

### Core components

| Component | Role |
|---|---|
| **Prelude blocks** | Standard transformer layers that build the initial representation before the loop |
| **RecurrentBlock** | Single shared-weight block iterated up to `max_loop_iters` times |
| **MLAttention** | Multi-Latent Attention (DeepSeek-V2): compresses KV through a low-rank latent, reducing cache memory 10–20× vs standard GQA |
| **MoEFFN** | Mixture-of-Experts feed-forward: routed experts + always-active shared experts, only top-K fire per token |
| **LTIInjection** | Linear Time-Invariant state update — the `A · h_t + B · e` term with guaranteed stability |
| **ACTHalting** | Per-position probabilistic early exit, accumulating confidence until a threshold is reached |
| **LoRAAdapter** | Depth-wise per-iteration low-rank adaptation so each loop pass can behave differently |
| **Coda blocks** | Standard transformer layers that post-process the final recurrent state |

---

## Project goals

1. **Verify architectural compatibility** — can OpenMythos run as a Nova 2.0 `InferenceBackend`?
2. **Validate on consumer hardware** — RTX 3060 12GB single card and dual-card setup
3. **Characterise training viability** — what model scale can actually train on a two-card consumer rig?
4. **Provide a working training pipeline** — from dataset download to checkpoint, ready to run

---

## Variants

| Variant | Parameters | MoE share | Target use |
|---|---|---|---|
| `mythos_lite` | 93.5M | 20% | **Training on consumer GPU** — designed and benchmarked for 2× RTX 3060 12GB |
| `mythos_1b` | 1.1B | 84% | Inference on consumer GPU (2.2 GB VRAM); training requires ≥24 GB VRAM |
| `mythos_3b` | ~3B | — | Inference |
| `mythos_10b` | ~10B | — | Inference |
| `mythos_50b` | ~50B | — | Inference |
| `mythos_100b` | ~100B | — | Inference |
| `mythos_500b` | ~500B | — | Inference |
| `mythos_1t` | ~1T | — | Inference |

See [`docs/MYTHOS_LITE.md`](docs/MYTHOS_LITE.md) for the full design rationale, parameter budget, and benchmark results behind `mythos_lite`.

---

## Hardware requirements

### Inference

| Model | Min VRAM | Measured VRAM | Notes |
|---|---|---|---|
| `mythos_lite` | 2 GB | ~0.6 GB (bf16) | Runs on anything modern |
| `mythos_1b` | 4 GB | 2.20 GB (bf16) | Measured on RTX 3060 12GB, ~260ms/token |

### Training (`mythos_lite` only)

| Setup | Throughput | Est. 10B tokens |
|---|---|---|
| 1× RTX 3060 12GB | 6.3k tok/s | ~18 days |
| 2× RTX 3060 12GB (DDP) | **11.4k tok/s** | **~10 days** |

The 1B model cannot train on consumer hardware — see [`docs/MYTHOS_LITE.md`](docs/MYTHOS_LITE.md) for the full analysis.

---

## Project structure

```
nova-mythos/
├── src/nova_mythos/
│   ├── model/
│   │   ├── architecture.py   # OpenMythos RDT implementation (vendored, MIT)
│   │   ├── variants.py       # Config factory functions for each scale
│   │   └── tokenizer.py
│   ├── backend.py            # NovaMythosBackend — Nova 2.0 InferenceBackend protocol
│   ├── config.py             # NovaMythosConfig, load_config()
│   └── harness.py            # BackendProbeRunner, ProbeResult, ProbeReport
├── training/
│   ├── train_lite.py         # Full training script: FineWeb-Edu + DDP + checkpointing
│   ├── benchmark_lite.py     # Throughput benchmark for mythos_lite
│   └── benchmark_1b.py       # Throughput benchmark for mythos_1b (documents OOM)
├── scripts/
│   └── run_probes.py         # Standalone probe runner (loads backend from config)
├── tests/
│   ├── test_hardware.py      # Phase 2: GPU forward pass, VRAM validation
│   ├── test_backend.py       # Phase 3: InferenceBackend protocol compliance
│   └── test_probe_harness.py # Phase 4: Probe harness structural + contract tests
├── configs/
│   └── nova_mythos.default.yaml
├── results/
│   ├── benchmark_lite_results.json   # Measured dual-GPU DDP results
│   └── benchmark_1b_results.json     # Measured OOM analysis for 1B
└── docs/
    └── MYTHOS_LITE.md        # Full mythos_lite design, experiment, and results
```

---

## Installation

```bash
git clone git@github.com:Cognameton/nova-mythos.git
cd nova-mythos
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11+ required. PyTorch must be installed separately with the correct CUDA version for your system — see [pytorch.org](https://pytorch.org).

---

## Usage

### Running the probe harness

```bash
# Against the default config (mythos_1b, requires a checkpoint)
python scripts/run_probes.py

# Override config
python scripts/run_probes.py --config configs/nova_mythos.default.yaml

# Save results as JSON for comparison
python scripts/run_probes.py --json results/probe_run_001.json
```

### Running the benchmarks

```bash
# mythos_lite benchmark (recommended — confirms hardware viability)
python training/benchmark_lite.py                          # single GPU
.venv/bin/torchrun --nproc_per_node=2 training/benchmark_lite.py  # dual GPU

# 1B benchmark (documents OOM on consumer hardware)
python training/benchmark_1b.py
.venv/bin/torchrun --nproc_per_node=2 training/benchmark_1b.py
```

### Training mythos_lite

```bash
# Single GPU
python training/train_lite.py --output-dir runs/lite-001

# Dual GPU (recommended)
.venv/bin/torchrun --nproc_per_node=2 training/train_lite.py \
    --output-dir runs/lite-001

# Resume after interruption
.venv/bin/torchrun --nproc_per_node=2 training/train_lite.py \
    --output-dir runs/lite-001 --resume
```

Downloads FineWeb-Edu `sample-10BT` automatically via HuggingFace streaming. No manual dataset preparation needed. Checkpoints saved every 5,000 steps to `runs/lite-001/checkpoint-XXXXXXX/`.

### Using NovaMythosBackend in code

```python
from nova_mythos.config import load_config
from nova_mythos.backend import NovaMythosBackend, GenerationRequest

cfg = load_config("configs/nova_mythos.default.yaml")
backend = NovaMythosBackend(cfg)
backend.load()

result = backend.generate(GenerationRequest(
    prompt="Hello. Who are you?",
    max_tokens=128,
    temperature=0.7,
    stop_sequences=["User:"],
))
print(result.text)

backend.unload()
```

### Running the test suite

```bash
# CPU tests (backend protocol + probe harness) — no GPU required
pytest tests/test_backend.py tests/test_probe_harness.py -v

# GPU tests — requires CUDA
pytest tests/test_hardware.py -v -s
```

---

## The Nova 2.0 InferenceBackend protocol

`NovaMythosBackend` implements the five-method interface that Nova 2.0 uses to communicate with any inference backend:

```python
backend.load()                          # load weights to device
backend.unload()                        # free device memory
backend.metadata() -> dict              # variant, device, loaded state
backend.tokenize(text: str) -> int      # token count
backend.generate(req) -> GenerationResult
```

This makes nova-mythos a drop-in replacement for Nova's existing llama_cpp backend — swap the config, not the runtime.

---

## Config reference

`configs/nova_mythos.default.yaml`:

```yaml
model:
  variant: lite            # lite | 1b | 3b | 10b | ...
  checkpoint_path: ./checkpoints/nova_mythos_lite.pt
  device: cuda:0
  dtype: bfloat16
  # max_loop_iters: 8      # override variant default (optional)

generation:
  max_new_tokens: 512
  temperature: 0.7
  stop_sequences:
    - "User:"
    - "user:"

tokenizer:
  name: gpt2
```

---

## Development phases

This project was built in four validated phases before the training work:

| Phase | Description | Tests |
|---|---|---|
| 1 — Foundation | Project scaffold, vendored architecture, dtype bug fixes | — |
| 2 — Hardware Validation | GPU forward pass, VRAM measurement | 5 passing |
| 3 — Protocol Compliance | NovaMythosBackend, GenerationRequest/Result | 7 passing |
| 4 — Probe Harness | BackendProbeRunner, all probe types, contract tests | 11 passing |

All 23 tests pass on CPU (phases 3–4) and GPU (phase 2).

---

## Attribution

The RDT architecture is from [OpenMythos](https://github.com/kyegomez/OpenMythos) by Kye Gomez, used under the MIT License.

This project was developed by [Cognameton / Everyman AI Lab](https://github.com/Cognameton) with assistance from **Claude** (Anthropic) and **Codex** (OpenAI).

---

## License

MIT — see [OpenMythos LICENSE](https://github.com/kyegomez/OpenMythos/blob/main/LICENSE) for the vendored architecture component.
