# 2026-04-20 tq4_1s vendor sync + semver bump

## Summary

- Updated `vendor/llama.cpp` from `9276d42f4` to `029890e6d`.
- Source branch for the vendor delta: `codex/tq4-1s-runtime-reference` in `C:\Users\downl\Desktop\Turboquant-CUDA\zapabob\llama.cpp`.
- Scope of the vendor sync is the staged `tq4_1s` GGML runtime slice:
  - GGML type registration
  - CPU quant/dequant and vec-dot wiring
  - staged CUDA dequant path
  - runtime reference test coverage
- Coordinated Hypura version bump from `0.9.1` to `0.10.0`.

## Notes

- This slice intentionally tracks the runtime-support commit and not the separate metadata-only `convert_hf_to_gguf.py` changes from other local mirrors.
- Verification target is a fast differential build of the root crate and desktop crate after the submodule SHA and version surfaces move together.
