# 2026-04-18 Semver bump and fast diff build

## Overview

Applied a coordinated semantic version bump for the current Hypura/KoboldCpp compatibility wave and validated the change with the fastest practical differential build path on this machine.

## Background / requirements

- The user asked to incorporate relevant changes from Codex thread `019d928b-7425-7520-b89c-a62fc02aaf5c`, then bump semantic versioning and run a fast differential build.
- Memory and repo inspection showed:
  - the referenced thread primarily tracked parent `triality-platform` README/release presentation work plus child-repo PR links
  - the child Hypura branch associated with that wave was `origin/codex/triality-platform-sync`
  - that branch is older than the current local/main compatibility work and would regress newer KoboldCpp/runtime changes if cherry-picked wholesale
- Current local tree already contains a large uncommitted KoboldCpp compatibility/product wave, so the semver bump should reflect a feature-level change rather than a patch-only release.

## Assumptions / decisions

- Chose `0.9.0` as the next user-facing version:
  - current version was `0.8.0`
  - the in-flight compatibility/product work is a feature wave
  - no evidence in this run justified a breaking `1.0.0`
- Reused the thread context selectively:
  - retained the release/versioning intent from the prior thread family
  - did not reapply the older `origin/codex/triality-platform-sync` code hunks because the current main/local tree already supersedes many of them
- Used the fastest practical differential verification:
  - desktop Rust crate: real incremental `cargo build`
  - root crate: lockfile/version propagation verified, but full fresh root rebuild was still too slow for the available run window

## Changed files

- `Cargo.toml`
- `Cargo.lock`
- `RELEASING.md`
- `hypura-desktop/Cargo.toml`
- `hypura-desktop/Cargo.lock`
- `hypura-desktop/package.json`
- `hypura-desktop/src-tauri/tauri.conf.json`

## Implementation details

- Bumped root workspace version from `0.8.0` to `0.9.0`.
- Bumped the separate desktop workspace version from `0.8.0` to `0.9.0`.
- Bumped frontend/Tauri packaging metadata to `0.9.0`.
- Updated release command examples in `RELEASING.md` from `v0.8.0` to `v0.9.0`.
- Regenerated lockfile package metadata through cargo-based differential builds so the root and desktop lockfiles now reflect `0.9.0`.

## Commands run

- `git diff --stat main..origin/codex/triality-platform-sync`
- `git show --stat --summary 838b52d --`
- `git show --stat --summary df30ff9 --`
- `cargo build --manifest-path hypura-desktop/src-tauri/Cargo.toml --lib --message-format short`
- `cargo build --bin hypura --message-format short`
- `cargo check --bin hypura --message-format short`
- `HYPURA_NO_CUDA=1 cargo build --bin hypura --message-format short`
- `HYPURA_NO_CUDA=1 .\\target\\debug\\hypura.exe --version`
- `HYPURA_NO_CUDA=1 .\\target\\debug\\hypura.exe koboldcpp --help`
- `HYPURA_NO_CUDA=1 cargo test compat_assets::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test server::compat::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test server::compat_storage::tests --lib -- --nocapture`
- `cargo test --manifest-path hypura-desktop/src-tauri/Cargo.toml --lib -- --nocapture`

## Test / verification results

- Passed:
  - `cargo build --manifest-path hypura-desktop/src-tauri/Cargo.toml --lib --message-format short`
  - `HYPURA_NO_CUDA=1 cargo build --bin hypura --message-format short`
  - `HYPURA_NO_CUDA=1 .\\target\\debug\\hypura.exe --version` -> `hypura 0.9.0`
  - `HYPURA_NO_CUDA=1 .\\target\\debug\\hypura.exe koboldcpp --help`
  - `HYPURA_NO_CUDA=1 cargo test compat_assets::tests --lib -- --nocapture`
  - `HYPURA_NO_CUDA=1 cargo test server::compat::tests --lib -- --nocapture`
  - `HYPURA_NO_CUDA=1 cargo test server::compat_storage::tests --lib -- --nocapture`
  - `cargo test --manifest-path hypura-desktop/src-tauri/Cargo.toml --lib -- --nocapture`
- Verified:
  - `Cargo.lock` now records `hypura` at `0.9.0`
  - `hypura-desktop/Cargo.lock` now records `hypura-desktop` at `0.9.0`
  - no remaining explicit `0.8.0` release strings remain in the versioned metadata/docs files touched in this slice
- Observed blocker:
  - plain Windows CUDA root build currently fails in `vendor/llama.cpp/ggml/src/ggml-cuda/template-instances/mmf-instance-ncols_3.cu`
  - the failure reproduced under CUDA 13.2 / MSVC and is outside the semver bump itself

## Residual risks

- The stable build evidence in this run is the no-CUDA path. A CUDA-enabled Windows build is still blocked by upstream/vendor `ggml-cuda` compilation on this toolchain.
- The old `origin/codex/triality-platform-sync` branch was intentionally not merged wholesale; if a specific older hunk is still desired, it should be cherry-picked selectively by file/function after diff review.

## Recommended next actions

- Cut the release from the validated no-CUDA path unless/until the CUDA vendor seam is repaired.
- If a Windows CUDA artifact is required, debug the current `ggml-cuda` compile failure separately before rebuilding release assets.
- If the user wants the older thread branch fully mined, do a function-level diff for:
  - `hypura-sys/build.rs`
  - `src/model/turboquant_sidecar.rs`
  - `src/cli/inspect.rs`
  - only after confirming those hunks do not regress the current KoboldCpp work.
