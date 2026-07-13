# Hypura v1.0.0

Hypura is a storage-tier-aware GGUF runtime with a Triality Council execution path, native HTTP API, and KoboldCpp-compatible server profile.

Version 1.0.0 is the first stable Council release. Its Windows binary is built for NVIDIA RTX 50-series GPUs with CUDA 12.8 and compute capability `sm_120`. The release carries a CLI executable, recursive fullsource archive, MSI, NSIS installer, and SHA-256 manifests on both the stable and main release tracks.

## What v1.0.0 adds

### Triality Council

The dedicated Answer Council evaluates three Triality views of the same prompt and selects one response through deterministic 3 x 3 teacher-forced cross-scoring. It deliberately rejects attention-consensus and synthesis requests instead of silently treating them as Answer Council work. Native per-layer Triality execution remains available on the ordinary `run`, `serve`, and compatible worker paths when the loaded schema-v2 model advertises the requested capability.

Sequential execution is the safe default. `auto` permits parallel answer contexts only when the scheduler can preserve the configured VRAM headroom; an explicitly requested parallel run fails closed when its memory budget is not admissible.

The trace-enabled Council API response records the selected candidate, scoring matrix, memory decision, NC-KA gate outcome, URT report, capabilities, and Aha evaluation. The default persistent record keeps selection and cross-score evidence, an Aha event when available, and content only according to the explicit storage policy; it does not persist the complete trace. Prompt text and private candidate text are not written to persistent telemetry by default.

### Triality schema v2

`zapabob/Turboquant-CUDA` is the schema producer and verifier. The v2 contract adds deterministic bundle metadata, short canonical tensor identifiers, manifest hashing, strict dtype and shape validation, NC-KA and URT descriptors, and negative validation for incomplete or contradictory bundles. Schema v1 remains readable for the legacy single-view path.

`zapabob/llama.cpp` is the low-level runtime authority. It owns strict GGUF contract parsing, rotation-tensor structural and SO(8) validation, caller-owned public C ABI buffers, context configuration, capability reporting, fused attention consensus, and low-level finite-metric capture. Hypura verifies the embedded controller's physical-byte hash, evaluates request-level NC-KA, projects memory across storage tiers, applies fail-closed admission, and owns privacy-safe Council and URT persistence. Storage-changing context configuration is reserved before allocation; it is not silently changed after decode begins.

### NC-KA, URT, and Aha

NC-KA checks the numerical rank and conditioning evidence required by a configured Council policy. URT records reproducible transformation and selection evidence under a configured data root: the HTTP service uses its application-data root, while the CLI uses the parent of its explicit `--output-dir`. Aha activation requires both an explicit safety comparator and calibration evidence; without them, Aha remains disabled rather than reporting an unsupported safety claim.

### Existing runtime surfaces

Hypura continues to provide:

- tier-aware tensor placement across GPU, pinned host memory, pageable RAM, and NVMe
- native `run`, `serve`, `inspect`, `estimate`, `bench`, `profile`, `iobench`, and `optimize` workflows
- a KoboldCpp-compatible supervisor/worker profile with vendored Kobold Lite
- OpenAI-compatible completions, chat completions, and embeddings routes
- savedata, preload story, launcher configuration, and probe-gated optional multimodal bridges
- ELT loop metadata detection with a fail-closed runtime gate

## Source authority and reproducibility

The canonical runtime inputs are:

- [zapabob/llama.cpp](https://github.com/zapabob/llama.cpp)
- [zapabob/Turboquant-CUDA](https://github.com/zapabob/Turboquant-CUDA)

Hypura pins tested commits from both repositories. Every fullsource release recursively materializes direct and nested submodules and includes `SOURCE-MANIFEST.json` with the exact Hypura, llama.cpp, Turboquant-CUDA, and nested llama.cpp commits. Release checksums are listed in `SHA256SUMS.txt`.

## Windows stable release requirements

The v1.0.0 prebuilt CLI and Desktop artifacts target:

- Windows 11 x86-64
- NVIDIA RTX 50-series GPU with compute capability 12.0
- NVIDIA CUDA 12.8 runtime
- Microsoft Visual C++ runtime compatible with the Visual Studio 2022 build

The release binary is intentionally rebuilt with `CUDA_PATH`, `HYPURA_CUDA_ARCHITECTURES`, and `CMAKE_CUDA_ARCHITECTURES` pinned to CUDA 12.8 and architecture 120. A binary importing CUDA 13 libraries does not pass the v1.0.0 release gate.

Source builds for other platforms remain available, but they are not represented by the Windows `sm_120` asset and must pass their own backend validation.

## Install

Download the stable assets from the `v1.0.0` GitHub release and verify them against `SHA256SUMS.txt`. The main snapshot uses the `main-v1.0.0` tag and contains the same tested source commit under channel-specific asset names.

CLI users may place the verified executable in `%USERPROFILE%\.cargo\bin` or another directory on `PATH`. Desktop users may install either the MSI or NSIS package. Check the release notes for the code-signing status before running an installer.

```powershell
hypura --version
hypura --help
hypura council --help
```

## Quick start

### Council generation

```powershell
hypura council .\model.gguf `
  --prompt 'Explain why deterministic evaluation matters.' `
  --parallelism auto `
  --cross-score `
  --max-tokens 128
```

Use `--parallelism sequential` when reproducibility and minimum peak memory are more important than latency. `parallel` is never treated as permission to exceed the configured headroom.

### Native one-shot generation

```powershell
hypura run .\model.gguf --prompt 'Hello' --max-tokens 128
```

### Native HTTP server

```powershell
hypura serve .\model.gguf --host 127.0.0.1 --port 8080
```

```powershell
$body = @{
  prompt = 'Explain Triality Council in one paragraph.'
  parallelism = 'auto'
  cross_score = $true
  max_tokens = 128
  stream = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:8080/api/extra/triality/council' `
  -ContentType 'application/json' `
  -Body $body
```

The response includes an `id`. Retrieve the persisted privacy-safe trace with:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/extra/triality/council/<id>
```

### KoboldCpp-compatible profile

```powershell
hypura koboldcpp .\model.gguf
```

The default native endpoint is `http://127.0.0.1:8080`. The compatibility profile defaults to `http://127.0.0.1:5001`, with Kobold Lite at `/kobold-lite`.

## Build from source

Initialize every submodule before building:

```powershell
git submodule update --init --recursive
```

For the supported Windows CUDA 12.8 release configuration:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
$env:HYPURA_CUDA = '1'
$env:HYPURA_CUDA_ARCHITECTURES = '120'
$env:CMAKE_CUDA_ARCHITECTURES = '120'
$env:CARGO_TARGET_DIR = 'H:\hypura-cargo-target-v1.0.0'

$activeRustBuilds = Get-CimInstance Win32_Process | Where-Object {
  $_.Name -in @('cargo.exe', 'rustc.exe')
}
if ($activeRustBuilds) {
  $activeRustBuilds | Select-Object ProcessId,ParentProcessId,Name,CommandLine
  throw 'Rust builds are active. Identify ownership and wait for or stop only release-owned processes before cleaning.'
}

cargo clean -p hypura-sys
cargo build --release --locked --bin hypura
```

Use a target directory with sufficient free space. On Windows, clean `hypura-sys` whenever the pinned llama.cpp commit, public FFI header, CUDA toolkit, or CUDA architecture changes.

Desktop installers are built separately:

```powershell
Set-Location .\hypura-desktop
npm ci
$env:CARGO_TARGET_DIR = 'H:\hypura-desktop-release-target-v1.0.0'
npm run tauri -- build
```

## Validation boundaries

The Turboquant schema fixture proves serialization, parser, manifest, and fail-closed behavior. It is not a runnable language model. Live Council QA uses a runnable GGUF whose original tensors are preserved byte-for-byte while a complete, deterministic identity-view schema-v2 bundle is materialized into the file. That fixture proves runtime wiring and view isolation; it is not evidence that identity views improve model quality.

The Aha false-positive target cannot be established from source code or an unlabeled smoke test. It requires a versioned labeled evaluation set, declared sample count, safety comparator, and recorded calculation. Until that evidence is present, Hypura reports Aha as disabled or uncalibrated.

ELT loop execution has a separate gate. An `elt.loop.required=true` model is rejected unless the selected zapabob runtime has been verified for loop-aware decode and graph execution.

## Repository map

- `src/council/`: Council types, scoring, selection, NC-KA, and Aha policy
- `src/urt/`: URT registry, reports, and privacy-safe persistence
- `src/compute/`: llama.cpp FFI, inference contexts, and storage-tier execution
- `src/scheduler/`: placement, VRAM headroom, and Council admission
- `src/server/`: native API and compatibility supervisor/worker surfaces
- `hypura-sys/`: vendored llama.cpp build and generated FFI boundary
- `vendor/llama.cpp/`: pinned canonical runtime
- `vendor/turboquant-cuda/`: pinned canonical schema producer and verifier
- `hypura-desktop/`: Tauri Desktop application and installers
- `scripts/package_fullsource.py`: deterministic recursive source packaging
- `scripts/stage_release_assets.py`: channel-specific release staging and checksums

## Release integrity

The complete stable and main publication procedure is documented in [RELEASING.md](RELEASING.md). It includes source gates, CUDA dependency inspection, CLI and HTTP manual QA, Desktop builds, deterministic fullsource verification, overwrite installation, branches, tags, GitHub releases, and post-publication download checks.

Security-sensitive or private Council inputs should not be attached to bug reports. Provide the source manifest, trace identifier, capability summary, and redacted telemetry instead.
