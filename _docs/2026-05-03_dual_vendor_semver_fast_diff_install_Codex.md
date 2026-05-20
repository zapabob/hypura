# 2026-05-03 dual vendor semver fast diff install

## Summary

- Advanced `vendor/llama.cpp` from `2b4bb412f` to `0eaf4a667`.
- Advanced `vendor/turboquant-cuda` from `72586ec98dab` to `76a90fbd1a6`.
- Bumped Hypura user-facing version surfaces from `0.12.0` to `0.13.0`.
- Restored the no-op `kobold-gui` feature expected by the existing fast build scripts.
- Built Hypura release with CUDA enabled and `HYPURA_CUDA_ARCHITECTURES=86`.
- Overwrote the local Hypura install slot at `C:\Users\downl\.cargo\bin\Hypura.exe`.
- Refreshed the local llama.cpp CUDA install slot by copying `H:\llama-cuda-tq4-build\bin\Release` over `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`.

## Semver decision

`llama.cpp` moved to a newer merge head, but the tree diff against the prior pinned commit was empty. `Turboquant-CUDA` carried user-visible GGUF export/profile work for Qwen3.5-4B TQ4 and looped ELT metadata inference. That makes this a feature-level Hypura release bump, so the next version is `0.13.0`.

## Verification

- `cargo build -p hypura --release --features kobold-gui`
  - target dir: `H:\hypura-cargo-target`
  - CUDA arch: `CMAKE_CUDA_ARCHITECTURES=86`
  - CUDA: `GGML_CUDA=ON`
  - toolkit: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe`
  - result: `Finished release profile [optimized] target(s) in 16m 47s`
- `C:\Users\downl\.cargo\bin\Hypura.exe --version`
  - `hypura 0.13.0`
- `C:\Users\downl\.cargo\bin\Hypura.exe koboldcpp --help`
  - passed, with `research-kv-split` and `triality-vector` defaults visible.
- `cargo metadata --manifest-path hypura-desktop\Cargo.toml --no-deps --format-version 1`
  - reported `hypura-desktop@0.13.0`.
- `git diff --check`
  - passed; only CRLF conversion warnings were reported.
- `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe --version`
  - `version: 8876 (a18d78a9d)`
  - CUDA device init succeeded.
- `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe --list-devices`
  - reported `CUDA0: NVIDIA GeForce RTX 3060`.

## Notes

The external `Turboquant-CUDA` runtime submodule used by the local llama.cpp install remains `a18d78a9d`, so the refreshed llama.cpp install is an overwrite of the current proven CUDA runtime rather than a new C++ runtime revision.
