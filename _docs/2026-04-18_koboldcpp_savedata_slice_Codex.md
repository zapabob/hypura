# 2026-04-18 KoboldCpp Savedata Slice (Codex)

## Scope
- Implement the savedata and product-save compatibility slice for the pinned KoboldCpp `v1.111.2` target.
- Keep `hypura serve` on the native inline GUI.
- Switch `hypura koboldcpp` to vendored upstream Kobold Lite assets and storage-backed save/load routes.

## Implemented
- Added `src/server/compat_storage.rs` as the canonical Hypura compat storage layer.
- Canonical storage now persists:
  - preload story bundle
  - remote save slots
  - launcher profiles (`.kcpps` JSON payloads)
  - GUI presets
  - GUI history and GUI events
  - UI theme
- Added SQLite-backed persistence with a KoboldCpp `.jsondb` bridge export/import path.
- Added CLI support on `hypura koboldcpp` for:
  - `--savedatafile`
  - `--preloadstory`
  - `--admindir`
  - `--config`
  - `--exportconfig`
  - `--migration-dir`
- Added Kobold Lite save/config endpoints:
  - `GET /api/extra/preloadstory`
  - `POST /api/extra/data/list`
  - `POST /api/extra/data/save`
  - `POST /api/extra/data/load`
  - `GET /api/admin/list_options`
  - `POST /api/admin/reload_config`
- Added runtime-state slot persistence for:
  - `POST /api/admin/save_state`
  - `POST /api/admin/load_state`
- Added a private compat control channel so `POST /api/admin/reload_config` now dispatches real in-place model/context reload work instead of only updating stored profile state.
- Added request-scoped llama runtime snapshot capture and prefix-aware restore wiring in `src/compute/inference.rs` plus `src/compute/ffi.rs`.
- Added canonical SQLite metadata plus sidecar binary files for runtime state slots.
- Vendored upstream Lite/docs assets under `vendor/kobold-lite/` and serve them from:
  - `/kobold-lite`
  - `/api`
  - `/koboldcpp_api.html`
  - `/koboldcpp_api.json`
  - `/manifest.json`
  - `/sw.js`
  - `/niko.png`

## Notes
- The canonical store is Hypura-owned SQLite. The KoboldCpp `savedatafile` remains a bridge artifact, not the source of truth.
- `reload_config` now updates the stored compat launcher profile and triggers an in-place model/context reload through a private control channel. The planned supervisor/worker child-process split is still pending.
- Runtime state save/load now works through Hypura-owned slot metadata plus sidecar files. Because Hypura contexts are request-scoped, a loaded state is reused when the next prompt tokenizes with the saved prefix.

## Verification
- `HYPURA_NO_CUDA=1 cargo check --bin hypura --message-format short`
- `HYPURA_NO_CUDA=1 cargo test server::compat::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test server::compat_storage::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test server::routes::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo test compute::inference::tests --lib -- --nocapture`
- `HYPURA_NO_CUDA=1 cargo run --quiet --bin hypura -- koboldcpp --help`
