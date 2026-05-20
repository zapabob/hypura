# 2026-05-18 Dual Vendor Semver Fast Diff Install

## Scope

- Advanced Hypura from `0.13.0` to `0.14.0` because both vendor runtime surfaces moved after the previous release preparation.
- Updated the vendored `llama.cpp` checkout from `0eaf4a6670c86867b50f8579136e0cba816a768d` to `31b900be63367c5065655d2ff4a77ca45f053e3f`.
- Updated the vendored `Turboquant-CUDA` checkout from `76a90fbd1a6e6a7fee32d4c384165338ae5d41c6` to `4fd30ea249c82ee57eb770244585c5604e68a599`.
- Aligned `vendor/turboquant-cuda/zapabob/llama.cpp` to `31b900be63367c5065655d2ff4a77ca45f053e3f`.

## Rationale

- `llama.cpp` includes the May 2026 upstream sync, server/router CUDA context handling, MTP support, conversion module split, and TurboQuant CUDA refreshes.
- `Turboquant-CUDA` includes default-branch dependency remediation, Triality SO8 audit artifacts, Qwen3.5-4B TQ4 export support, and a nested runtime pin matching the top-level `llama.cpp` pin.
- The change is a feature-level vendor/runtime refresh, so the Hypura semantic version moved to `0.14.0`.

## Local Install Target

- Hypura CLI: `C:\Users\downl\.cargo\bin\Hypura.exe`
- llama.cpp/TurboQuant runtime slot: `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`
- CUDA architecture for this PC: `86` for NVIDIA GeForce RTX 3060.

## Verification Plan

- Run the fast Windows CUDA build through `scripts/build-hypura-hf.ps1` with `HYPURA_CUDA_ARCHITECTURES=86`.
- Overwrite the Hypura CLI install slot and the local llama.cpp/TurboQuant bin slot.
- Verify `Hypura.exe --version`, `Hypura.exe koboldcpp --help`, `llama-server.exe --version`, `llama-server.exe --list-devices`, and package metadata.

## Results

- Hypura fast build completed with `hypura-sys v0.14.0` and `hypura v0.14.0` in `22m 44s`.
- The Hypura build script skipped `F:\` because free space was below the 15 GiB safety threshold and used `H:\hypura-cargo-target`.
- Installed `H:\hypura-cargo-target\release\Hypura.exe` over:
  - `C:\Users\downl\Desktop\hypura-main\hypura-main\target\release\Hypura.exe`
  - `C:\Users\downl\.cargo\bin\Hypura.exe`
- Standalone llama.cpp/TurboQuant runtime was configured for RTX 3060 with `CMAKE_CUDA_ARCHITECTURES=86`.
- First standalone `llama-server.exe` build hit server UI asset provisioning failure (`npm` build failure and Hugging Face TLS download failure). The runtime was reconfigured with UI embedding disabled and rebuilt successfully.
- Built and installed `llama-server.exe` version `9458 (31b900be6)` over `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin`.
- Installed runtime help exposes TurboQuant cache types `turbo2`, `turbo3`, and `turbo4`.
- Installed `llama-server.exe --list-devices` and `llama-cli.exe --list-devices` both print `CUDA0: NVIDIA GeForce RTX 3060`; the current runtime process does not exit cleanly after listing devices, so verification used a controlled timeout and kill.
- No active `cargo`, `rustc`, `cmake`, `ninja`, `cl`, `link`, `nvcc`, `MSBuild`, `llama-server`, or `llama-cli` processes remained after cleanup.
