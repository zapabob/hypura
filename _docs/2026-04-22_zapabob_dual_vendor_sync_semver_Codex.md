# 2026-04-22 - zapabob llama.cpp + Turboquant-CUDA sync and Hypura semver bump

## Summary

- advanced `vendor/llama.cpp` from `00459dd29` to `2b4bb412f`
- advanced `vendor/turboquant-cuda` from `e3e7333492ef` to `72586ec98dab`
- bumped Hypura version surfaces from `0.11.0` to `0.12.0`

## Upstream slice carried in

- `llama.cpp`: two-layer Triality TurboQuant compatibility on top of the prior `tq4_1s` / ABI hardening line
- `Turboquant-CUDA`: TheTom-compatible Triality bridge metadata and refreshed GGUF profile/triality contract coverage

## Hypura impact

- no additional Rust-side compatibility patch was required beyond the existing
  `LLAMA_BUILD_TOOLS=OFF` guard in `hypura-sys/build.rs`
- Windows CUDA 12.8 root build continued to pass after both vendor pins moved

## Verification target

- `cargo build --bin hypura --message-format short` with `CUDA_PATH=v12.8`
- `cargo build --manifest-path hypura-desktop/src-tauri/Cargo.toml --lib --message-format short`
- `target/debug/hypura.exe --version`
- `target/debug/hypura.exe koboldcpp --help`
