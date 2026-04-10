# Hypura Desktop (Tauri)

Native shell: **file picker**, **drag-and-drop** of `.gguf`, **recent paths**, **start/stop** `hypura serve`, **HTTP model switch** (`POST /api/extra/model/switch`) when the server is already running, and **copy helpers** for Kobold / Kobold-lite URLs.

## Prerequisites

- Node.js 18+ and npm
- Rust toolchain (same as main Hypura repo)
- Windows: WebView2
- `hypura` on `PATH`, or place `hypura.exe` next to the desktop binary, or set `HYPURA_EXE`
- `src-tauri/icons/icon.ico` is a minimal placeholder so `tauri-build` succeeds; run `npm run tauri icon <your.png>` from `hypura-desktop/` before a branded release

## Commands

```sh
cd hypura-desktop
npm install
npm run tauri dev
```

Release:

```sh
npm run tauri build
```

This folder is a **separate Cargo workspace** from the repo root (`exclude` in the parent workspace). When bumping the project version, follow the **four-file checklist** in [RELEASING.md](../RELEASING.md).

If the server uses `HYPURA_API_KEY`, fill the optional API key field in the UI before **Switch model** (Bearer header).
