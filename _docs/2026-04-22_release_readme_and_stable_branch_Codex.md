# 2026-04-22 - release README rewrite and stable branch packaging

## Overview

- rewrote the top-level README as a release-facing landing page for `v0.12.0`
- updated release guidance to use a versioned stable branch flow (`stable/v0.12.0`)
- fixed the packaged desktop Tauri import seam so release bundling succeeds
- built fresh release artifacts for the CLI plus Windows desktop installers

## Background / requirements

The repo already carried the `0.12.0` semver bump and dual vendor sync for
`zapabob/llama.cpp` plus `zapabob/Turboquant-CUDA`. The follow-up ask was to:

- rewrite `README.md`
- cut a stable release branch
- create tagged GitHub releases with built artifacts
- keep `main` aligned with the release commit

## Assumptions / decisions

- the release commit should be shared by `main` and `stable/v0.12.0`
- the GitHub tag should remain `v0.12.0`
- release assets should include the native Windows CLI binary and the packaged
  desktop installer outputs when they build successfully
- benchmark evidence should remain visible in the top-level README instead of
  being pushed down into artifact-only paths

## Changed files

- `README.md`
- `RELEASING.md`
- `hypura-desktop/src/main.js`
- this log file

## Implementation details

### README rewrite

- tightened the product framing around the two shipped entrypoints:
  - `hypura serve`
  - `hypura koboldcpp`
- preserved explicit `v0.12.0` shipped-surface bullets for:
  - core runtime
  - KoboldCpp compatibility
  - Windows packaged path
- kept benchmark evidence, compatibility honesty, quick start, build commands,
  repo map, and release flow visible in the top-level README

### Release workflow update

- switched `RELEASING.md` from a generic `stable` branch description to a
  versioned stable branch flow using `stable/v0.12.0`
- documented the intended flow:
  - verify on `main`
  - branch `stable/vX.Y.Z`
  - push both branches
  - tag the shared release commit

### Desktop packaging seam

- changed `hypura-desktop/src/main.js` to import `getCurrentWindow` from
  `@tauri-apps/api/window` instead of `@tauri-apps/api/webview`
- this was required for the current Tauri API surface so `npx tauri build`
  could complete

## Commands run

```powershell
git status --short --branch
gh auth status
Get-PSDrive -PSProvider FileSystem

$env:TMP = 'H:\hypura-tmp'
$env:TEMP = 'H:\hypura-tmp'
$env:RUSTC_WRAPPER = ''
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:HYPURA_CUDA = '1'
$env:HYPURA_CUDA_ARCHITECTURES = '86'
cargo build --release --bin hypura --target-dir H:\hypura-release-target-20260422 --message-format short

& 'H:\hypura-release-target-20260422\release\hypura.exe' --version
& 'H:\hypura-release-target-20260422\release\hypura.exe' koboldcpp --help

$env:TMP = 'H:\hypura-tmp'
$env:TEMP = 'H:\hypura-tmp'
$env:RUSTC_WRAPPER = ''
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:HYPURA_CUDA = '1'
$env:HYPURA_CUDA_ARCHITECTURES = '86'
cargo test server::compat::tests --lib --target-dir H:\hypura-test-target-20260422 -- --nocapture

cd hypura-desktop
$env:TMP = 'H:\hypura-tmp'
$env:TEMP = 'H:\hypura-tmp'
$env:npm_config_cache = 'H:\npm-cache'
npm install

$env:TMP = 'H:\hypura-tmp'
$env:TEMP = 'H:\hypura-tmp'
$env:RUSTC_WRAPPER = ''
$env:CARGO_TARGET_DIR = 'H:\hypura-desktop-release-target-20260422'
npx tauri build
```

## Test / verification results

- `cargo build --release --bin hypura` succeeded with CUDA `12.8`
- `hypura.exe --version` returned `hypura 0.12.0`
- `hypura.exe koboldcpp --help` completed successfully
- `cargo test server::compat::tests --lib` passed (`3 passed; 0 failed`)
- `npx tauri build` succeeded after the Tauri API import fix
- produced desktop bundles:
  - `H:\hypura-desktop-release-target-20260422\release\bundle\msi\Hypura Desktop_0.12.0_x64_en-US.msi`
  - `H:\hypura-desktop-release-target-20260422\release\bundle\nsis\Hypura Desktop_0.12.0_x64-setup.exe`

## Residual risks

- `npm install` created a local `package-lock.json`; keep it because it now
  reflects the release-verified desktop dependency set
- `hypura-desktop/node_modules/` is generated runtime state and should not be
  committed
- the release/tag/push operations still need to be executed after the release
  commit is created

## Recommended next actions

- commit the release-facing doc and desktop seam updates
- create and push `stable/v0.12.0`
- create tag `v0.12.0`
- publish the GitHub release with the built CLI and desktop assets
