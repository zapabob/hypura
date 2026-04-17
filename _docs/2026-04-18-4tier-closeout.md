# 4-Tier Residency Closeout

Date: 2026-04-18
Workspace: `C:\Users\downl\Desktop\hypura-main\hypura-main`
Branch: `codex/4-tier-closeout`

## Scope

Scoped closeout for the Windows/CUDA 4-tier residency wave only.

Included:
- 4-tier scheduler/runtime wiring
- CLI residency controls and summaries
- runtime residency state tests
- benchmark telemetry/result schema updates
- focused benchmark evidence for Windows/CUDA

Excluded:
- vendored `vendor/*` drift
- upstream merge tooling
- unrelated release and Rust2024 follow-up files

## Verification

### Fresh compile and tests

```powershell
cargo +1.93.0-x86_64-pc-windows-msvc check -p hypura -j 1 --message-format short
$env:HYPURA_NO_CUDA='1'; cargo +1.93.0-x86_64-pc-windows-msvc test -p hypura scheduler::placement --lib -j 1
$env:HYPURA_NO_CUDA='1'; cargo +1.93.0-x86_64-pc-windows-msvc test -p hypura compute::nvme_backend --lib -j 1
cargo +1.93.0-x86_64-pc-windows-msvc build -p hypura -j 1
```

Observed:
- `cargo check`: passed
- `scheduler::placement`: 7 passed
- `compute::nvme_backend`: 8 passed
- `cargo build`: passed

### CLI behavior checks

```powershell
.\target\debug\hypura.exe run --context 32768 --max-tokens 1 --prompt "Hello" --residency-profile four-tier --host-pinned auto "H:\from_D\OrganiZen\オールインワン 08-09-2024\EasyNovelAssistant-main\EasyNovelAssistant-main\EasyNovelAssistant\setup\KoboldCpp\Shadows-MoE-Q6.gguf"
.\target\debug\hypura.exe bench --baseline --context 32768 --max-tokens 4 "H:\from_D\OrganiZen\オールインワン 08-09-2024\EasyNovelAssistant-main\EasyNovelAssistant-main\EasyNovelAssistant\setup\KoboldCpp\Shadows-MoE-Q6.gguf"
```

Observed:
- `run` printed four residency buckets and `Residency: mode=four-tier, pinned_tier=collapsed, pinned_policy=auto`
- `bench` ran `legacy-3tier+off`, `four-tier+off`, and `four-tier+auto`
- `bench` printed per-run residency status and telemetry summary

## Benchmark evidence

Artifact:
- `benchmarks/results/2026-04-17T16-01-04_Shadows-MoE-Q6.json`

Model:
- `Shadows-MoE-Q6.gguf`
- architecture: `llama`
- params: `12.9B`
- quant: `Q6K`

Hardware:
- CPU: `AMD Ryzen 5 4500 6-Core Processor`
- GPU: `NVIDIA GeForce RTX 3060`
- RAM: `31.9 GB`
- NVMe sequential read: `2.5 GB/s`

Result summary:
- baseline `llama.cpp`: `1.1 tok/s`
- `hypura legacy-3tier + off`: `1.1 tok/s`
- `hypura four-tier + off`: `1.2 tok/s`
- `hypura four-tier + auto`: `1.2 tok/s`
- primary speedup: `1.1x`

Notes:
- The model spilled `2.4 GB` to NVMe in all three Hypura runs, so the comparison exercised the NVMe-backed path.
- `pinned_tier` remained collapsed on this machine/profile, so pinned host residency did not activate in this evidence run.
- Telemetry fields are now present in the JSON output even when the measured values remain `0.0` for this workload.
