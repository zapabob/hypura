# 2026-05-21 ELT Loop Runtime Gate

## Overview

Added a Hypura-side runtime gate for looped ELT GGUF metadata. Hypura now
detects `elt.loop.*` metadata, reports it through CLI/API surfaces, bridges the
loop parameters into `LLAMA_ELT_LOOP_*` environment variables, and refuses to
run `elt.loop.required=true` models as plain L=1 unless a verified loop-aware
`zapabob/llama.cpp` runtime is explicitly selected.

## Background / Requirements

- The implementation line is `zapabob/Turboquant-CUDA` plus `zapabob/llama.cpp`
  plus `zapabob/hypura`.
- `Turboquant-CUDA` preserves and classifies looped ELT metadata such as
  `ELT/Qwen3.5-looped`.
- `zapabob/llama.cpp` owns actual loop-aware decode/graph execution.
- Hypura owns the product runtime, distribution surface, and verification gate.
- Hypura must not imply that metadata alone makes an L>=2 model runnable.

## Assumptions / Decisions

- `elt.loop.required=true`, `L_min > 1`, `L_default > 1`,
  `ELT/Qwen3.5-looped`, or `requires_looped_qwen35_runtime` means loop-aware
  runtime support is required.
- `HYPURA_ELT_LOOP_RUNTIME_SUPPORTED=1` is an explicit deployment assertion for
  a verified loop-aware vendored runtime. It is not itself the loop
  implementation.
- Optional metadata with `L_default=1` can be reported without blocking.

## Changed Files

- `src/model/elt_loop.rs`
- `src/model/mod.rs`
- `src/compute/inference.rs`
- `src/cli/fmt_util.rs`
- `src/cli/inspect.rs`
- `src/cli/run.rs`
- `src/cli/bench.rs`
- `src/cli/serve.rs`
- `src/server/ollama_types.rs`
- `src/server/routes.rs`
- `README.md`
- `RELEASING.md`

## Implementation Details

- Added `EltLoopMetadata` parsing for the current `elt.loop.*` key family,
  including both `L_*` and lowercase aliases.
- Added fail-closed validation in runtime setup, server model loading, and
  one-shot generation paths.
- Added `LLAMA_ELT_LOOP_*` environment bridging so a loop-aware vendored
  `llama.cpp` build has a stable parameter surface.
- Added `inspect`, `run`, `bench`, and `serve` status lines with a gate label.
- Added `/api/show` model metadata fields for ELT loop status and Hypura gate
  status.
- Updated release docs to require explicit looped ELT verification before any
  release claim.

## Commands Run

- `rustfmt src\model\elt_loop.rs`
- `cargo test -p hypura elt_loop --lib`
- `cargo check -p hypura --lib --message-format short`
- `$env:CARGO_TARGET_DIR='H:\hypura-cargo-target'; $env:CARGO_BUILD_JOBS='4'; $env:LLAMA_BUILD_UI='OFF'; $env:LLAMA_BUILD_WEBUI='OFF'; cargo check -p hypura --lib --message-format short`
- `cargo metadata --no-deps --format-version 1`
- `git diff --check`
- `Get-Process cargo,rustc,link,cl,cmake,ninja ...`

## Test / Verification Results

- `cargo metadata --no-deps --format-version 1`: passed and confirmed the
  workspace/package metadata resolves.
- `git diff --check`: passed with only existing CRLF conversion warnings.
- `cargo test -p hypura elt_loop --lib`: timed out after 20 minutes while
  building vendored runtime dependencies.
- `cargo check -p hypura --lib --message-format short`: timed out after 15
  minutes on the default target directory.
- `cargo check` with `CARGO_TARGET_DIR=H:\hypura-cargo-target` and UI build
  disabled also timed out after 15 minutes in the vendored CMake/MSVC path.
- Leftover `cargo`, `cl`, and `cmake` processes from the timed-out verification
  attempts were stopped and rechecked.

## Residual Risks

- Full Rust typecheck/test completion is still pending because the local
  vendored `hypura-sys` build path exceeded the practical turn budget.
- The Hypura gate selects and validates the product boundary, but actual L>=2
  correctness still requires loop-aware decode/graph work in `zapabob/llama.cpp`.
- `HYPURA_ELT_LOOP_RUNTIME_SUPPORTED=1` must only be set by release automation
  or operators after that vendored runtime proof exists.

## Recommended Next Actions

- Add or consume a concrete `zapabob/llama.cpp` runtime capability marker once
  loop-aware decode/graph execution lands.
- Re-run `cargo test -p hypura elt_loop --lib` from a warmed build shell or
  after `hypura-sys` has completed in the shared target directory.
- Add a tiny fixture GGUF with `elt.loop.required=true` for CLI-level
  fail-closed smoke tests.
