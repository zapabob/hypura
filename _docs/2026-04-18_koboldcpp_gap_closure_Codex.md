# 2026-04-18 KoboldCpp Remaining Gap Closure

## Scope

- Close the last compatibility stubs for `/v1/embeddings`, `/api/extra/embeddings`, and `/api/extra/websearch`.
- Add a supervisor-managed secondary embeddings runtime path.
- Add built-in DuckDuckGo HTML websearch.
- Add Windows-first asset bootstrap scaffolding with `%LOCALAPPDATA%\\Hypura\\koboldcpp\\assets` default placement and packaged manifest resources.

## Implemented

- Added `--embeddings-model` and `--asset-root` to `hypura koboldcpp`.
- Persisted `embeddings_model` and `asset_root` inside compat launcher profiles.
- Added `src/compat_assets.rs` for manifest discovery, verification, bootstrap staging, and background materialization hooks.
- Added `src/server/embeddings.rs` for OpenAI-shaped embeddings responses backed by a dedicated GGUF runtime.
- Added `src/server/websearch.rs` for DuckDuckGo HTML search, top-result extraction, page fetch, and readable-text fallback.
- Extended the supervisor control plane with:
  - `/embeddings/v1/embeddings`
  - `/embeddings/api/extra/embeddings`
  - `/builtin/api/extra/websearch`
- Replaced public worker stubs with control-plane proxy routing.
- Added packaged asset manifests at:
  - `docs/compat/koboldcpp-assets.json`
  - `hypura-desktop/src-tauri/resources/koboldcpp-assets.json`
- Added Tauri bundle resource wiring for the packaged manifest.

## Notes

- `websearch` is now always advertised in compat mode because it is a built-in worker capability.
- `embeddings` is advertised only when a secondary embeddings model is configured and the runtime loads successfully.
- The repo snapshot includes the bootstrap mechanism and packaged manifest resources, but it does not yet ship the large embeddings or multimodal binaries themselves. Packaging can now fill those through bundled resources or manifest download URLs without changing runtime code.
