# Hypura

Storage-tier-aware GGUF runtime and KoboldCpp-compatible server product.

Hypura ships two user-facing entrypoints:

- `hypura serve` for the native Hypura runtime and HTTP API
- `hypura koboldcpp` for the KoboldCpp-compatible supervisor/worker stack with vendored Kobold Lite, savedata bridges, and probe-gated optional multimodal surfaces

## Why Hypura

- Keep oversized GGUF models moving by placing tensors across GPU, RAM, pinned host memory, and NVMe instead of treating VRAM as the only useful tier.
- Expose a real KoboldCpp-compatible profile without adding a second proxy product beside the runtime.
- Carry TurboQuant and Triality metadata end-to-end in GGUF workflows while keeping the public CLI and release surface simple.
- Build Windows CUDA releases against an explicit toolkit such as CUDA 12.8 instead of drifting to whichever Visual Studio CUDA integration was installed last.

## What ships in v0.12.0

### Core runtime

- Tier-aware placement across GPU, RAM, pinned host memory, and NVMe
- `inspect`, `bench`, `run`, and `optimize` workflows for analysis, benchmarking, and layout work
- TurboQuant and Triality-aware runtime metadata handling
- Vendored `llama.cpp` main sync with `tq4_1s` GGML CPU support, staged CUDA dequant support, fail-closed metadata handling, and two-layer Triality TurboQuant compatibility
- Vendored `Turboquant-CUDA` main sync with TheTom-compatible Triality bridge metadata and refreshed GGUF profile tooling
- Apple Silicon Metal and Windows CUDA in the same workspace

### KoboldCpp compatibility

- `hypura koboldcpp <model.gguf>` with KoboldCpp-style defaults such as port `5001`
- Vendored Kobold Lite
- Kobold extra/admin routes, state save/load, preload story, `.jsondb` bridge, and `.kcpps` launcher config bridge
- OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, and `/v1/embeddings`
- Built-in websearch route and supervisor-managed feature probing
- Supervisor/worker split so compat reloads and feature-state changes stay out of the native `serve` path

### Windows packaged path

- Desktop-owned first-run asset bootstrap manifest
- Probe-gated embeddings plus STT/TTS packaged path
- Structured unavailable responses when optional multimodal backends are not ready instead of optimistic success flags

## Compatibility snapshot

The pinned KoboldCpp baseline is `v1.111.2`. The current implementation ledger lives in [docs/compat/koboldcpp-v1.111.2-parity-manifest.json](docs/compat/koboldcpp-v1.111.2-parity-manifest.json).

Implemented compatibility areas:

- Kobold generation routes and admin/state endpoints
- Vendored Kobold Lite
- OpenAI chat, completions, and embeddings
- Savedata and launcher config bridges
- Probe-gated multimodal proxy routes
- Windows packaged asset bootstrap for embeddings plus audio

Still honest about current limits:

- Packaged Stable Diffusion payloads are still optional rather than part of the default packaged-ready set
- Multimodal feature flags depend on actual local assets or helper availability
- Ollama parity still needs a full audit pass even though compatibility surfaces exist

## Benchmark evidence

Benchmark output is computed from the JSON corpus in `benchmarks/results/` and summarized with mean +/- SD, error-bar charts, and multi-group comparison tables in [benchmarks/CHARTS.md](benchmarks/CHARTS.md).

Current measured hardware corpus:

- `AMD Ryzen 5 4500 6-Core Processor / NVIDIA GeForce RTX 3060 / 31.9 GB RAM`

Best observed Hypura score per model in the current corpus:

| Model | Score group | Benchmark score (tok/s) | Samples | Notes |
| --- | --- | --- | ---: | --- |
| Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M | `hypura four-tier + auto` | `51.835 +/- 2.293` | 2 | Repeated Windows CUDA runs; paired `mmproj` projector was inspect-validated separately |
| Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M | `hypura four-tier + auto` | `0.041 +/- 0.034` | 2 | Sparse MoE mmap path fell back to CPU-only on this machine; baseline remained faster in this corpus |
| Shadows-MoE-Q6 | `hypura four-tier + off` | `1.158 +/- 0.111` | 2 | Includes repeated runs and a baseline comparator in `benchmarks/results/` |
| supergemma4-Q8_0 | `hypura legacy-3tier + off` | `29.851 +/- 0.000` | 1 | Single-run exploratory datapoint; GPU-resident and not yet a stable replicated estimate |

Multi-group summary for the same corpus:

| Model | baseline | legacy-3tier + off | four-tier + off | four-tier + auto |
| --- | ---: | ---: | ---: | ---: |
| Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M | `41.681 +/- 4.357` | `50.390 +/- 3.980` | `29.407 +/- 34.425` | `51.835 +/- 2.293` |
| Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M | `0.059 +/- 0.050` | `0.020 +/- 0.001` | `0.038 +/- 0.014` | `0.041 +/- 0.034` |
| Shadows-MoE-Q6 | `1.121 +/- 0.023` | `1.086 +/- 0.029` | `1.158 +/- 0.111` | `0.984 +/- 0.334` |
| supergemma4-Q8_0 | `N/A` | `29.851 +/- 0.000` | `0.173 +/- 0.000` | `0.167 +/- 0.000` |

Read these numbers with the run count in mind:

- `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M` has `n=2`; `four-tier + auto` is currently the strongest replicated group, while `four-tier + off` shows very high variance.
- `Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M` has `n=2`, but all groups are very slow on this hardware because Hypura's sparse MoE mmap path fell back to CPU-only (`ngl=0`) once the 19.7 GB model exceeded the RTX 3060 GPU budget.
- `Shadows-MoE-Q6` has `n=2` for every reported group, so SD reflects actual repetition.
- `supergemma4-Q8_0` currently has `n=1`, so `+/- 0.000` means "only one observation", not "perfectly stable".
- The `supergemma4-Q8_0` run is a full GPU-resident Windows CUDA datapoint, not an NVMe spill benchmark.

## Quick start

### Native runtime

```powershell
hypura serve .\model.gguf
```

### One-shot local generation

```powershell
hypura run .\model.gguf --prompt "Hello"
```

### KoboldCpp-compatible profile

```powershell
hypura koboldcpp .\model.gguf
```

Useful follow-up routes:

- native server default: `http://127.0.0.1:8080`
- KoboldCpp profile default: `http://127.0.0.1:5001`
- Kobold Lite: `http://127.0.0.1:5001/kobold-lite`

## Build

### Standard build

```powershell
.\scripts\stop-cargo.ps1
cargo build --release
```

### Windows CUDA 12.8 build

```powershell
.\scripts\stop-cargo.ps1
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:HYPURA_CUDA = "1"
$env:HYPURA_CUDA_ARCHITECTURES = "86"
cargo build --release
```

If you need a fully isolated CUDA build tree:

```powershell
$env:CARGO_TARGET_DIR = ".\target-cuda128"
cargo build --bin hypura --message-format short
```

## Repo map

- `src/compute/` - runtime, inference, and storage-tier execution
- `src/scheduler/` - placement and estimation logic
- `src/server/` - native HTTP surface plus compat supervisor/worker layers
- `hypura-sys/` - vendored `llama.cpp` FFI build
- `vendor/llama.cpp/` - upstream runtime dependency
- `vendor/turboquant-cuda/` - vendored TurboQuant reference implementation and metadata tooling
- `docs/compat/` - pinned compatibility manifests and packaged asset manifests
- `hypura-desktop/` - packaged desktop bootstrap shell

## Release flow

- Use [RELEASING.md](RELEASING.md) for version alignment, versioned stable branch flow, tagging, and GitHub CLI release steps.
- On Windows, stop concurrent `cargo` and `rustc` processes before builds to avoid stale file locks.
- After `llama.cpp` or FFI changes, prefer cleaning `hypura-sys` outputs rather than wiping the entire workspace by default.

## Related repositories

- [zapabob/Turboquant-CUDA](https://github.com/zapabob/Turboquant-CUDA)
- [zapabob/llama.cpp](https://github.com/zapabob/llama.cpp)
