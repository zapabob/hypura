# Hypura

Storage-tier-aware GGUF runtime plus a KoboldCpp-compatible server profile.

Hypura has two shipped product surfaces:

- `hypura serve`: the native Hypura runtime and HTTP server
- `hypura koboldcpp`: a KoboldCpp-compatible supervisor/worker profile with vendored Kobold Lite, savedata bridges, OpenAI-compatible endpoints, and probe-gated multimodal surfaces

## TL;DR

- Run models that do not fit cleanly in GPU memory by placing tensors across GPU, RAM, pinned host memory, and NVMe.
- Serve a KoboldCpp-compatible stack without standing up a separate proxy layer.
- Keep TurboQuant and Triality metadata inside GGUF workflows while preserving a plain `hypura` CLI surface.
- Build Windows CUDA releases against an explicit toolkit version such as CUDA 12.8 without silently drifting to the newest installed Visual Studio CUDA integration.

## What ships in v0.9.1

### Native runtime

- Tier-aware tensor placement across GPU, RAM, pinned host memory, and NVMe
- `inspect`, `bench`, and `optimize` workflows for real model analysis and layout work
- TurboQuant and Triality-aware runtime metadata handling
- Apple Silicon Metal path and Windows CUDA path in the same workspace

### KoboldCpp compatibility profile

- `hypura koboldcpp <model.gguf>` with KoboldCpp-style defaults such as port `5001`
- Vendored Kobold Lite surface
- Kobold extra/admin routes, state save/load, preload story, `.jsondb` bridge, and `.kcpps` launcher config bridge
- OpenAI-compatible `/v1/completions`, `/v1/chat/completions`, and `/v1/embeddings`
- Built-in websearch route and supervisor-managed feature probing
- Supervisor/worker split so compat reloads and feature state changes do not mutate the native `serve` path

### Packaged Windows bootstrap path

- Desktop-owned first-run asset bootstrap manifest
- Probe-gated embeddings plus STT/TTS packaged path
- Structured unavailable responses when optional multimodal backends are not ready instead of optimistic success flags

## Compatibility snapshot

The pinned KoboldCpp baseline is `v1.111.2`. Current manifest status lives in [docs/compat/koboldcpp-v1.111.2-parity-manifest.json](docs/compat/koboldcpp-v1.111.2-parity-manifest.json).

Shipped compatibility areas:

- Kobold generation routes and admin/state endpoints
- Vendored Kobold Lite
- OpenAI chat, completions, and embeddings
- Savedata and launcher config bridges
- Probe-gated multimodal proxy routes
- Windows packaged asset bootstrap for embeddings plus audio

Known limits that remain honest release notes:

- Packaged Stable Diffusion payloads are still optional rather than part of the default packaged-ready set
- Multimodal feature flags depend on actual local assets or helper availability
- Ollama parity still needs a full audit pass even though compatibility surfaces exist

## Bench snapshot

These are repository snapshot numbers from the current project status notes, not a fresh benchmark run for this release.

| Model | Mode | Throughput | Notes |
| --- | --- | --- | --- |
| Mixtral 8x7B Q5_K_M (30.9 GB) | expert-streaming | 2.19 tok/s | M1 Max 32 GB snapshot |
| Llama 3.3 70B Q4_K_M (39.6 GB) | dense-FFN-streaming | 0.30 tok/s | M1 Max 32 GB snapshot |

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
- `docs/compat/` - pinned compatibility manifests and packaged asset manifests
- `hypura-desktop/` - packaged desktop bootstrap shell

## Release notes for operators

- Use [RELEASING.md](RELEASING.md) for version alignment, stable branch flow, tagging, and GitHub CLI release steps.
- On Windows, stop concurrent `cargo` and `rustc` processes before builds to avoid stale file locks.
- After `llama.cpp` or FFI changes, prefer cleaning `hypura-sys` outputs rather than wiping the entire workspace by default.

## Related repositories

- [zapabob/Turboquant-CUDA](https://github.com/zapabob/Turboquant-CUDA)
- [zapabob/llama.cpp](https://github.com/zapabob/llama.cpp)
