# CUDA 12.8 Windows build fix for Hypura

## Overview

Built a Windows CUDA-enabled `hypura` binary against CUDA 12.8 and fixed the
build-script seam that was silently mixing CUDA 12.8 toolkit paths with the
newest installed Visual Studio CUDA integration.

## Background / requirements

- User requested a sync pass against thread `019da01c-1a2f-78e2-97a4-b75d9631a4c7`
  and a CUDA 12.8 build attempt for this repo.
- The referenced thread was not a `hypura-main` workspace run. It was a
  `Turboquant-CUDA` session in a different workspace, so there was no direct
  patch stack to import into this repo.
- Existing local environment had both CUDA 12.8 and CUDA 13.2 installed.

## Assumptions / decisions

- Treat the thread as external context only and perform the actual build work in
  `C:\Users\downl\Desktop\hypura-main\hypura-main`.
- Use an isolated target directory, `target-cuda128`, so the CUDA build would
  not mix with prior CPU/no-CUDA artifacts.
- Keep the change minimal and local to `hypura-sys/build.rs`.

## Changed files

- `hypura-sys/build.rs`

## Implementation details

1. Ran a fresh CUDA build with:
   - `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
   - `HYPURA_CUDA=1`
   - `HYPURA_CUDA_ARCHITECTURES=86`
   - `CARGO_TARGET_DIR=target-cuda128`
2. First fresh build succeeded, but investigation showed a mixed toolchain:
   - `CUDAToolkit_ROOT` pointed at CUDA 12.8
   - Visual Studio generator still imported `CUDA 13.2.props/targets`
   - `CMakeCUDACompiler.cmake` still resolved `nvcc` to CUDA 13.2
3. Fixed the first seam by preferring `CUDA_PATH/bin/nvcc(.exe)` in
   `find_nvcc()`.
4. Fixed the actual Windows generator seam by setting Visual Studio toolset
   selection to `cuda=<CUDA_PATH>` when CUDA is enabled, so CMake/VS picks the
   requested toolkit version instead of the newest installed CUDA integration.
5. Rebuilt after cleaning only the generated `hypura-sys` build artifacts inside
   `target-cuda128` to keep the rebuild incremental.

## Commands run

```powershell
git status --short
git branch --show-current
Get-Content C:\Users\downl\.codex\sessions\2026\04\18\rollout-2026-04-18T19-21-29-019da01c-1a2f-78e2-97a4-b75d9631a4c7.jsonl -TotalCount 120
Get-Content hypura-sys\build.rs -TotalCount 260

$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp;' + $env:PATH
$env:CARGO_TARGET_DIR='C:\Users\downl\Desktop\hypura-main\hypura-main\target-cuda128'
$env:HYPURA_CUDA='1'
$env:HYPURA_CUDA_ARCHITECTURES='86'
$env:NUM_JOBS='1'
$env:CMAKE_BUILD_PARALLEL_LEVEL='1'
cargo build --bin hypura --message-format short

Get-ChildItem target-cuda128\debug\build\hypura-sys-* -Recurse -Filter ggml-cuda.lib
Select-String target-cuda128\debug\build\hypura-sys-*\out\build\CMakeCache.txt -Pattern 'CUDAToolkit_ROOT|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES'
Get-ChildItem target-cuda128\debug\build\hypura-sys-*\out\build\CMakeFiles -Recurse -Filter CMakeCUDACompiler.cmake | Get-Content -TotalCount 90
Get-ChildItem target-cuda128\debug\build\hypura-sys-*\out\build -Recurse -Filter *.vcxproj | Select-String -Pattern 'CUDA 12.8|CUDA 13.2'

.\\target-cuda128\\debug\\hypura.exe --version
.\\target-cuda128\\debug\\hypura.exe koboldcpp -h
```

## Test / verification results

- `cargo build --bin hypura --message-format short`
  - Passed with CUDA enabled in `target-cuda128`
- `ggml-cuda.lib`
  - Generated under `target-cuda128\debug\build\hypura-sys-...\out\lib\ggml-cuda.lib`
- `CMakeCache.txt`
  - Confirmed `CMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe`
  - Confirmed `CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- `CMakeCUDACompiler.cmake`
  - Confirmed compiler/toolkit version `12.8.61`
- generated `.vcxproj`
  - Confirmed `CUDA 12.8.props/targets` imports
- `.\\target-cuda128\\debug\\hypura.exe --version`
  - Returned `hypura 0.9.0`
- `.\\target-cuda128\\debug\\hypura.exe koboldcpp -h`
  - Returned help successfully

## Residual risks

- This run verified build-time CUDA 12.8 selection and basic binary startup, but
  did not run a real model generation smoke because no concrete model path was
  provided in this task.
- `target-cuda128/` is left as an untracked generated artifact so the built
  binary remains available locally.
- The referenced `uv sync` thread did not map directly onto this repo. If a
  parent or sibling workspace needs to be synchronized as well, that should be
  done in the matching repository rather than here.

## Recommended next actions

1. Run `target-cuda128\debug\hypura.exe inspect <model.gguf>` with a real local
   GGUF to verify runtime loading under the CUDA 12.8 build.
2. Run a short `hypura run` or `hypura koboldcpp <model.gguf> --dry-run` smoke
   in the same CUDA 12.8 environment.
3. If this build should become the default Windows CUDA path, follow up with a
   release-oriented smoke script or a documented `target-cuda128` build recipe.
