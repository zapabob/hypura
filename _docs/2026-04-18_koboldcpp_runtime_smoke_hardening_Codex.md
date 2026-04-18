# 2026-04-18 KoboldCpp packaged runtime smoke hardening

## Overview

Hardened the packaged `hypura-desktop` launcher so optional STT/TTS helpers are only advertised when they survive real runtime probes instead of being assumed ready from manifest wiring alone.

## Background / requirements

- Prior packaged payload work had concrete manifests and bootstrap wiring, but runtime smoke on this machine still had three residuals:
  - root `compat_assets` tests had previously timed out
  - `cargo run --quiet --bin hypura -- koboldcpp --help` had previously timed out
  - helper runtime behavior for packaged STT/TTS had not been validated end-to-end
- Fresh repro on 2026-04-18 showed:
  - `compat_assets` root tests now pass
  - `koboldcpp --help` now completes
  - `gowhisper` release asset `v0.0.39` for Windows is client-only and does not expose `run` or `server`
  - `sherpa-onnx-non-streaming-tts-x64-v1.12.39.exe` can hang, so the desktop launcher must not mark TTS ready without a bounded probe

## Assumptions / decisions

- Keep the current public packaged API surface unchanged.
- Do not claim packaged STT/TTS readiness from asset presence alone.
- Treat helper executables as untrusted until they pass bounded startup or generation probes.
- Preserve text-mode startup even when optional packaged helpers fail or hang.

## Changed files

- `hypura-desktop/src-tauri/src/packaged.rs`

## Implementation details

- Added bounded process execution helpers for packaged runtime probes:
  - timeout-aware command capture
  - concise stdout/stderr summarization
  - WAV stdout fallback detection
- Added `gowhisper` help parsing to detect whether a local server command exists.
- Changed packaged STT startup to fail fast with a clear error when the current Windows `gowhisper` release asset is client-only.
- Changed packaged TTS handling to:
  - run a short startup probe using real generation
  - disable TTS if the helper times out or produces no WAV output
  - use timeout-bounded generation for actual `/api/extra/tts` and `/v1/audio/speech` requests
- Updated packaged runtime status text so the desktop shell can report "started with some optional packaged features disabled" instead of implying all optional helpers are available.
- Added unit tests covering:
  - `gowhisper` help parsing without server commands
  - WAV magic detection for stdout fallback

## Commands run

- `cargo test --manifest-path hypura-desktop/src-tauri/Cargo.toml --lib packaged::tests -- --nocapture`
- `cargo check --manifest-path hypura-desktop/src-tauri/Cargo.toml --message-format short`
- `cargo test compat_assets::tests --lib -- --nocapture`
- `cargo run --quiet --bin hypura -- koboldcpp --help`
- `cargo test --manifest-path hypura-desktop/src-tauri/Cargo.toml --lib -- --nocapture`

## Test / verification results

- Passed: desktop packaged tests, including new runtime-hardening tests
- Passed: desktop crate compile check
- Passed: root `compat_assets` tests
- Passed: root `hypura koboldcpp --help`
- Confirmed with live repro:
  - `gowhisper-windows-amd64.exe --help` on `v0.0.39` exposes client commands only
  - no `run` or `server` command is present in that release asset

## Residual risks

- Packaged STT is still disabled in practice until a Windows helper with a real local server or direct offline transcription contract is supplied.
- Packaged TTS is now honest and bounded, but end-to-end success still depends on the chosen `sherpa-onnx` executable matching the current Kitten flag contract on the target machine.
- The manifest still points at the previously selected helper assets; this change hardens runtime behavior but does not replace upstream binaries.

## Recommended next actions

- Replace packaged STT helper selection with a release asset that actually supports local offline transcription on Windows, or switch the packaged bridge to a direct CLI model that does not require a private HTTP server.
- Revisit packaged TTS asset selection if the current `sherpa-onnx` non-streaming executable continues to fail bounded real-generation probes.
- Once helper selection is updated, rerun a full clean-machine packaged bootstrap smoke.
