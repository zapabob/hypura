# 2026-04-19 zapabob llama.cpp sync + CUDA 12.8 six-core diff build

## Summary

- Updated `vendor/llama.cpp` from `df3913e92` to `9276d42f4` (`zapabob/master`).
- Kept the existing release-prep working tree changes intact.
- Fixed Windows CUDA rebuild stability in `hypura-sys/build.rs` by forcing `GGML_CCACHE=OFF` at CMake configure time.
- Rebuilt `hypura` as a CUDA-enabled Windows debug binary with a six-core incremental build.

## Why the build needed extra work

The initial retry after the submodule update exposed three Windows-specific issues:

1. `C:` and temp-directory exhaustion during `nvcc` compiler-id checks.
2. Rust linking used a mismatched Visual Studio linker from the VS 18 Insiders install while CMake used VS 2022 BuildTools.
3. `llama.cpp` CMake still injected `sccache` into CUDA compile rules, and that path caused `nvcc` to drift back to CUDA 13.2 even when the generated rule targeted CUDA 12.8.

The final successful path was:

- target dir on `F:\`
- `TEMP/TMP/TMPDIR` on `F:\tmp`
- `VsDevCmd.bat` from VS 2022 BuildTools
- `CMAKE_GENERATOR=Ninja`
- `GGML_CCACHE=OFF` defined in `hypura-sys/build.rs`
- `HYPURA_CUDA=1`
- `HYPURA_CUDA_ARCHITECTURES=86`
- `NUM_JOBS=6`
- `CMAKE_BUILD_PARALLEL_LEVEL=6`

After that, the final remaining linker mismatch came from a stale `highs-sys` native build compiled under the old toolchain state. Rebuilding `highs-sys` under the corrected environment resolved the final link failure.

## Verification

Build:

```powershell
cargo build -j 6 --bin hypura --message-format short
```

Runtime smoke:

```powershell
F:\hypura-cuda128-latest\debug\hypura.exe --version
F:\hypura-cuda128-latest\debug\hypura.exe koboldcpp --help
```

Observed:

- `vendor/llama.cpp` HEAD: `9276d42f4`
- `Finished 'dev' profile [unoptimized + debuginfo] target(s) in 22m 10s`
- `hypura 0.9.1`
- `koboldcpp --help` completed successfully from the built binary

## Artifact

- Built binary: `F:\hypura-cuda128-latest\debug\hypura.exe`
