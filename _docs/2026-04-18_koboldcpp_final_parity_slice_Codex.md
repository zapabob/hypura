# 2026-04-18 KoboldCpp Final Parity Slice (Codex)

## Scope
- Finish the remaining KoboldCpp product-parity slice on top of the already-landed savedata bridge and vendored Kobold Lite surface.
- Replace the same-process compat control path with a real supervisor/worker split for `hypura koboldcpp`.
- Remove prefix-match-based compat restore and move compat state save/load onto a live llama session.
- Replace multimodal `501` stubs with supervisor-probed proxy routes that surface structured availability.

## Implemented
- Added `src/cli/koboldcpp.rs` as the supervisor entrypoint for `hypura koboldcpp`.
- Added hidden worker bootstrap support through:
  - `src/main.rs`
  - `src/cli/serve.rs`
  - `src/server/supervisor.rs`
- `hypura koboldcpp` now starts a private loopback control plane and launches a hidden `__koboldcpp_worker` child that owns:
  - the public HTTP bind
  - vendored Kobold Lite
  - compat HTTP routes
- The supervisor now owns:
  - current launcher profile state
  - savedata DB path and preload story settings
  - worker lifecycle
  - multimodal feature probe state
  - private control-plane authentication
- Added `CompatWorkerBootstrap` JSON handoff so the worker inherits:
  - public bind
  - compat profile state
  - savedata bridge paths
  - preload story path
  - control-plane URL/token
  - feature advertisement state
- Added `CompatSupervisorCommand` with:
  - `ReloadConfig`
  - `ReprobeBundles`
  - `ShutdownWorker`
- `POST /api/admin/reload_config` now dispatches to the supervisor control plane and performs worker restart semantics instead of same-process model reload.

## Compat Runtime Session
- Added `CompatRuntimeSession` in `src/compute/inference.rs`.
- Compat mode now keeps a live llama context for:
  - generation
  - `save_state`
  - `load_state`
  - runtime metadata capture
- `POST /api/admin/save_state` now snapshots raw llama state bytes plus token metadata from the live compat session.
- `POST /api/admin/load_state` now restores directly into the live compat session instead of relying on prompt-prefix replay.
- Generation, save/load, and reload now share one compat-session lock path so these operations do not race each other inside the worker.

## Multimodal Surface
- Added supervisor-managed proxy handlers for:
  - `POST /api/extra/transcribe`
  - `POST /v1/audio/transcriptions`
  - `POST /api/extra/tts`
  - `POST /v1/audio/speech`
  - `GET /speakers_list`
  - `POST /sdapi/v1/txt2img`
  - `POST /sdapi/v1/img2img`
  - `POST /sdapi/v1/interrogate`
  - `POST /sdapi/v1/upscale`
  - `GET /sdapi/v1/options`
  - `GET /sdapi/v1/sd-models`
- Feature advertisement is now driven from supervisor probe state, not static stubs.
- In this repo snapshot, probe state is sourced from configured backend endpoints:
  - `HYPURA_KCPP_TRANSCRIBE_URL`
  - `HYPURA_KCPP_TTS_URL`
  - `HYPURA_KCPP_SD_URL`
- When a backend is not configured, compat mode now returns structured `503` unavailable responses instead of long-lived `501` placeholder stubs.

## Files Changed
- `Cargo.toml`
- `src/main.rs`
- `src/cli/mod.rs`
- `src/cli/serve.rs`
- `src/cli/koboldcpp.rs`
- `src/compute/inference.rs`
- `src/server/compat.rs`
- `src/server/mod.rs`
- `src/server/routes.rs`
- `src/server/supervisor.rs`
- `docs/compat/koboldcpp-v1.111.2-parity-manifest.json`

## Notes
- `hypura serve` remains the native same-process server path.
- `hypura koboldcpp` is now the compat-only supervisor path.
- The previous compat slice document is still accurate for savedata/storage bridge details, but this slice supersedes its remaining "pending" notes for:
  - supervisor/worker split
  - `reload_config`
  - live-session state restore semantics
- Multimodal parity is implemented as a supervisor-managed proxy contract in this snapshot. Shipping bundled STT/TTS/SD helper binaries and first-run asset placement remains a packaging follow-up.

## Verification
- `HYPURA_NO_CUDA=1 cargo check --bin hypura --message-format short`
- `HYPURA_NO_CUDA=1 cargo test server::supervisor::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test server::compat_storage::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test server::routes::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test compute::inference::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test server::compat::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo run --quiet --bin hypura -- koboldcpp --help`

## Residual Gaps
- Embeddings and websearch remain explicit compatibility stubs.
- Multimodal execution currently depends on configured local backend endpoints rather than bundled helper processes shipped by this repo alone.
- Full installer-level Windows packaging for first-run multimodal asset placement is still a follow-up task.
