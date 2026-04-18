# 2026-04-18 KoboldCpp packaged payload follow-up

## Scope

Implemented the Windows-first packaged payload follow-up for `hypura koboldcpp`:

- concrete asset manifest entries for embeddings + STT/TTS
- desktop-owned bootstrap shell for `hypura-desktop`
- local loopback audio bridge for packaged STT/TTS injection
- auto-restart flow after optional asset placement

## Main changes

- `src/compat_assets.rs`
  - added recursive Hugging Face tree bootstrap support for `*_tree` asset kinds
  - writes a local tree-ready marker after successful direct-file placement
  - keeps the existing manifest schema and file-based bootstrap for single-file assets
- `docs/compat/koboldcpp-assets.json`
  - replaced placeholder entries with concrete URLs, sizes, and checksums for:
    - embeddings GGUF
    - `gowhisper` Windows helper + Whisper model
    - `sherpa-onnx` Windows helper + Kitten TTS payload
  - modeled `espeak-ng-data` as a recursive HF tree bootstrap entry
- `hypura-desktop/src-tauri/src/packaged.rs`
  - added packaged launch orchestration for `hypura koboldcpp`
  - starts text mode immediately
  - downloads pending optional assets in the background
  - restarts the managed runtime when new features become ready
  - injects `HYPURA_KCPP_TRANSCRIBE_URL` / `HYPURA_KCPP_TTS_URL`
  - exposes a local loopback bridge for `/api/extra/transcribe`, `/v1/audio/transcriptions`, `/api/extra/tts`, `/v1/audio/speech`, and `/speakers_list`
- `hypura-desktop/src-tauri/src/lib.rs`
  - rewired commands from the old `serve` launcher to the packaged `koboldcpp` launcher flow
- `hypura-desktop/index.html`
- `hypura-desktop/src/main.js`
  - replaced the old `serve` UI with a packaged bootstrap UI showing runtime status and feature readiness
- `hypura-desktop/src-tauri/build.rs`
  - copies `docs/compat/koboldcpp-assets.json` into the Tauri resources directory at build time
- `docs/compat/koboldcpp-v1.111.2-parity-manifest.json`
  - updated packaged asset bootstrap status text to reflect the concrete payload flow

## Verification

Succeeded:

- `HYPURA_NO_CUDA=1 cargo check --bin hypura --message-format short`
- `cargo check --manifest-path hypura-desktop/src-tauri/Cargo.toml --message-format short`
- `cargo test --manifest-path hypura-desktop/src-tauri/Cargo.toml --lib -- --nocapture`

Not completed:

- `cargo test compat_assets::tests --lib -- --nocapture`
  - timed out in the root crate on this machine, but the same shared `compat_assets.rs` tests passed via the desktop crate include path
- `cargo run --quiet --bin hypura -- koboldcpp --help`
  - timed out before completion on this machine

## Known residuals

- The packaged TTS path is wired for `sherpa-onnx` + Kitten payloads, but it has not yet been smoke-tested end-to-end on this machine with the full downloaded payload set.
- The STT helper launch path assumes the current `gowhisper run --http.addr ... --models ...` contract; compile-time and manifest validation passed, but runtime helper invocation was not exercised end-to-end here.
- SD remains outside the v1 packaged-ready path by design.
