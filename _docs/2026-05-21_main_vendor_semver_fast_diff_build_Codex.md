# 2026-05-21 Main Vendor Semver Fast Diff Build

## Overview

Advanced Hypura to `0.15.0`, aligned the top-level vendored runtime pins with
the current `zapabob/llama.cpp` and `zapabob/Turboquant-CUDA` `main` branches,
and completed a 6-core fast differential Windows CUDA release build.

## Background / Requirements

- User requested a fast differential build using 6 cores.
- User requested a semantic version bump aligned to `Turboquant-CUDA` and
  `llama.cpp` on the `main` branch.
- The active product boundary is Hypura plus vendored `zapabob/llama.cpp` plus
  vendored `zapabob/Turboquant-CUDA`.
- The previous ELT loop metadata work remains a product gate: Hypura can detect
  and fail closed for `elt.loop.required=true`, but actual L>=2 execution still
  requires loop-aware decode/graph support in the vendored llama runtime.

## Assumptions / Decisions

- Bumped Hypura from the in-tree `0.14.0` preparation state to `0.15.0` because
  the top-level `llama.cpp` runtime moved again on `main`.
- Updated `.gitmodules` so the top-level `vendor/llama.cpp` branch anchor is
  `main`, matching the user's requested branch.
- Advanced `vendor/turboquant-cuda/zapabob/llama.cpp` to the same latest
  `zapabob/llama.cpp` `main` checkout as the top-level runtime after the user
  requested nested alignment too.
- Committed and pushed the nested submodule pointer update to `Turboquant-CUDA`
  `main`, then advanced the Hypura parent pointer to that clone-reproducible
  `Turboquant-CUDA` commit.
- Treated the fast build as the Hypura embedded vendored runtime proof. The
  external standalone `llama-turboquant` install slot was not rebuilt in this
  pass.

## Changed Files

- `.gitmodules`
- `Cargo.toml`
- `Cargo.lock`
- `hypura-desktop/Cargo.toml`
- `hypura-desktop/Cargo.lock`
- `hypura-desktop/package.json`
- `hypura-desktop/package-lock.json`
- `hypura-desktop/src-tauri/tauri.conf.json`
- `README.md`
- `RELEASING.md`
- `vendor/llama.cpp`
- `vendor/turboquant-cuda`
- `vendor/turboquant-cuda/zapabob/llama.cpp`

The working tree also still includes the earlier ELT loop runtime gate files
from the same release-preparation branch.

## Implementation Details

- Updated the top-level `vendor/llama.cpp` checkout to:
  `a9cebe03e8b0df6c72a2fa3a86c1a6b9c648abe6`
  (`docs: record attn rotation install verification`).
- Confirmed the top-level `vendor/turboquant-cuda` checkout is on:
  `df771a2887067f3996a47cf7a95e72e52a0ec649`
  (`chore(vendor): align nested llama main`).
- Advanced the nested `vendor/turboquant-cuda/zapabob/llama.cpp` checkout from:
  `31b900be63367c5065655d2ff4a77ca45f053e3f`
  to:
  `a9cebe03e8b0df6c72a2fa3a86c1a6b9c648abe6`.
- Pushed the `Turboquant-CUDA` child commit to `origin/main`:
  `4fd30ea..df771a2`.
- Updated user-facing release surfaces to `0.15.0`.
- Updated README release notes to state the exact vendor boundaries and the
  looped ELT L>=2 claim boundary.

## Commands Run

```powershell
git -C vendor\llama.cpp fetch origin main
git -C vendor\turboquant-cuda fetch origin main
git -C vendor\turboquant-cuda\zapabob\llama.cpp fetch origin main
git -C vendor\llama.cpp checkout --detach origin/main
git -C vendor\turboquant-cuda checkout --detach origin/main
git -C vendor\turboquant-cuda\zapabob\llama.cpp checkout --detach origin/main
git -C vendor\turboquant-cuda switch -c codex/nested-llama-main-20260521
git -C vendor\turboquant-cuda add zapabob/llama.cpp
git -C vendor\turboquant-cuda commit -m "chore(vendor): align nested llama main"
git -C vendor\turboquant-cuda push origin HEAD:main
git -C vendor\turboquant-cuda fetch origin main
git submodule sync --recursive
cargo metadata --no-deps --format-version 1

$env:CARGO_BUILD_JOBS='6'
$env:CMAKE_BUILD_PARALLEL_LEVEL='6'
$env:HYPURA_CUDA_ARCHITECTURES='86'
$env:LLAMA_BUILD_UI='OFF'
$env:LLAMA_BUILD_WEBUI='OFF'
.\scripts\build-hypura-hf.ps1 -StopOtherCargo -InstallTo "$env:USERPROFILE\.cargo\bin"

.\target\release\Hypura.exe --version
& "$env:USERPROFILE\.cargo\bin\Hypura.exe" --version
.\target\release\Hypura.exe koboldcpp --help
git submodule status --recursive
git diff --check
Get-Process cargo,rustc,link,cl,cmake,ninja -ErrorAction SilentlyContinue
```

## Test / Verification Results

- `cargo metadata --no-deps --format-version 1`: passed and confirmed
  `hypura` and `hypura-sys` resolve as `0.15.0`.
- 6-core fast differential build completed successfully with:
  - `CARGO_BUILD_JOBS=6`
  - `CMAKE_BUILD_PARALLEL_LEVEL=6`
  - `HYPURA_CUDA_ARCHITECTURES=86`
  - `LLAMA_BUILD_UI=OFF`
  - `LLAMA_BUILD_WEBUI=OFF`
- Build script skipped `F:\` because free space was `14.52 GiB`, below the
  script's `15 GiB` safety threshold, and used `H:\hypura-cargo-target`.
- Release build completed in `33m 24s`.
- Built `hypura-sys v0.15.0` and `hypura v0.15.0`.
- Installed `H:\hypura-cargo-target\release\Hypura.exe` over:
  - `C:\Users\downl\Desktop\hypura-main\hypura-main\target\release\Hypura.exe`
  - `C:\Users\downl\.cargo\bin\Hypura.exe`
- `.\target\release\Hypura.exe --version`: `hypura 0.15.0`.
- `C:\Users\downl\.cargo\bin\Hypura.exe --version`: `hypura 0.15.0`.
- `.\target\release\Hypura.exe koboldcpp --help`: passed and printed the
  KoboldCpp-compatible server profile help.
- `git submodule status --recursive`:
  - `vendor/llama.cpp`: `a9cebe03e8b0df6c72a2fa3a86c1a6b9c648abe6`
  - `vendor/turboquant-cuda`: `df771a2887067f3996a47cf7a95e72e52a0ec649`
  - `vendor/turboquant-cuda/zapabob/llama.cpp`:
    `a9cebe03e8b0df6c72a2fa3a86c1a6b9c648abe6`
- `git -C vendor\turboquant-cuda rev-parse origin/main`:
  `df771a2887067f3996a47cf7a95e72e52a0ec649`.
- `git diff --check`: passed with only CRLF conversion warnings.
- Final process check: `NO_BUILD_PROCESSES`.

## Residual Risks

- The external standalone runtime slot
  `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin` was not rebuilt
  during this pass. The verified artifact is the Hypura release binary built
  against the vendored runtime.
- `Turboquant-CUDA` now has a committed `main` revision for the nested
  `zapabob/llama.cpp` pointer, and Hypura points at that commit. The remaining
  risk is normal submodule pin maintenance if either upstream moves again.
- Looped ELT L>=2 correctness remains gated on actual loop-aware decode/graph
  execution in `zapabob/llama.cpp`; Hypura's current responsibility is
  detection, product selection, and fail-closed distribution behavior.

## Recommended Next Actions

- If the local standalone llama runtime slot must also be refreshed, build
  `vendor/llama.cpp` as a separate CUDA runtime artifact and overwrite
  `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`.
- Re-run the Hypura fast build if a binary produced after `df771a2` is required
  as an explicit release artifact; the previous Hypura build already used the
  same top-level `vendor/llama.cpp` runtime commit.
- After loop-aware decode/graph execution lands, run a real
  `elt.loop.required=true` GGUF smoke test through Hypura before making any
  L>=2 runtime claim.
