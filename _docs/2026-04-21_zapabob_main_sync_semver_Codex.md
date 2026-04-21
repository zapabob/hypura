# 2026-04-21 - zapabob llama.cpp main sync + Hypura semver bump

## Summary

- advanced `vendor/llama.cpp` from `029890e6d` to `00459dd29` (`zapabob/main`)
- bumped Hypura version surfaces from `0.10.0` to `0.11.0`
- updated release/docs references to the new version

## Upstream slice carried in

- `tq4_1s` runtime line merged into `zapabob/main`
- Triality ABI hardening
- fail-closed metadata handling
- `tq4_1s` CUDA closeout

## Hypura-side build seam

Windows CUDA 12.8 builds failed while linking `llama-turboquant` because upstream
tool CMake still referenced `common.lib` after the `llama-common` rename landed.
Hypura does not consume vendored tool executables, so `hypura-sys/build.rs` now
sets `LLAMA_BUILD_TOOLS=OFF` and keeps the vendored runtime/library build only.

## Verification target

- `cargo build --bin hypura --message-format short` with `CUDA_PATH=v12.8`
- `target/debug/hypura.exe --version`
- `target/debug/hypura.exe koboldcpp --help`
