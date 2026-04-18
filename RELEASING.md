# Hypura releases (GitHub CLI)

Tagging, version alignment, and uploading binaries with [GitHub CLI](https://cli.github.com/) (`gh`).

## 1. Before you build (required)

- Run **`gh auth status`** and complete `gh auth login` if needed.
- **Stop** running `cargo` / `rustc` (see README 「ビルド前（必須）」 or [`scripts/stop-cargo.ps1`](scripts/stop-cargo.ps1) / [`scripts/stop-cargo.sh`](scripts/stop-cargo.sh)). Avoid `target/` file locks.
- Optional faster rebuilds: merge [`.cargo/config.toml.example`](.cargo/config.toml.example) into `%USERPROFILE%\.cargo\config.toml` for **sccache**. If NVCC + sccache misbehaves, unset `CMAKE_*_COMPILER_LAUNCHER`.
- Parallelism (example, 6 cores): PowerShell  
  `$env:CARGO_BUILD_JOBS=6; $env:CMAKE_BUILD_PARALLEL_LEVEL=6`

## 2. Incremental vs clean

- Default: **`cargo build --release`** uses Cargo incremental compilation (no clean).
- After **llama.cpp / FFI** changes: `cargo clean -p hypura-sys` (or full `cargo clean`) before release.

## 3. Semver: bump checklist (keep all in sync)

When releasing **vX.Y.Z**, update every item before tagging:

| # | File | Field |
|---|------|--------|
| 1 | Root [`Cargo.toml`](Cargo.toml) | `[workspace.package] version = "X.Y.Z"` |
| 2 | [`hypura-desktop/Cargo.toml`](hypura-desktop/Cargo.toml) | `[workspace.package] version = "X.Y.Z"` |
| 3 | [`hypura-desktop/src-tauri/tauri.conf.json`](hypura-desktop/src-tauri/tauri.conf.json) | `"version": "X.Y.Z"` |
| 4 | [`hypura-desktop/package.json`](hypura-desktop/package.json) | `"version": "X.Y.Z"` |

`hypura` and `hypura-sys` use `version.workspace = true` at the repo root. `hypura-desktop` is a **separate** workspace (`exclude` in the root workspace) but must carry the **same** user-facing version for installers and docs.

Confirm:

```sh
hypura --version
# must match [workspace.package] version in root Cargo.toml
```

## 4. Install to cargo bin (optional)

```sh
./scripts/stop-cargo.sh   # or .\scripts\stop-cargo.ps1 on Windows
cargo install --path . --locked --force
```

Copy `target/release/hypura` or `hypura.exe` to your staging / `dist/` folder if you maintain one.

## 5. Build Hypura Desktop (optional asset)

The repo ships a minimal `hypura-desktop/src-tauri/icons/icon.ico` for Windows resource generation. Replace it with `npm run tauri icon path/to/branding.png` (from `hypura-desktop/`) before a public release.

From repo root:

```sh
cd hypura-desktop
npm install
npm run tauri build
```

Typical Windows bundle outputs (paths vary slightly by Tauri version):

- `hypura-desktop/src-tauri/target/release/bundle/msi/*.msi`
- `hypura-desktop/src-tauri/target/release/bundle/nsis/*.exe`

macOS: `.dmg` / `.app` under `bundle/`. Attach what you built to the GitHub Release.

## 6. Stable branch workflow

Create **`stable`** once (optional but useful for hotfixes):

```sh
git checkout main
git pull
git checkout -b stable
git push -u origin stable
```

**Normal release:** merge `main` into `stable`, then tag the `stable` tip:

```sh
git checkout stable
git merge main
git push origin stable
```

**Hotfix:** commit on `stable`, push, then tag from `stable` (cherry-pick back to `main` as needed).

## 7. Tag and `gh release create`

Use tag **`vX.Y.Z`** (with `v` prefix) matching the bumped version.

```sh
git tag -a v0.9.0 -m "release v0.9.0"
git push origin v0.9.0
```

**Windows example** — attach CLI binary + Tauri installer (adjust paths to your build outputs):

```powershell
gh release create v0.9.0 `
  --title "Hypura v0.9.0" `
  --notes "See README and git log for changes." `
  "target/release/hypura.exe#hypura-0.9.0-windows-x86_64.exe" `
  "hypura-desktop/src-tauri/target/release/bundle/nsis/Hypura Desktop_0.9.0_x64-setup.exe#hypura-desktop-0.9.0-windows-x64-setup.exe"
```

Syntax: `"local/path#DisplayName"` renames the asset on GitHub. If your NSIS/MSI filename differs, tab-complete the path.

**Linux / macOS** — attach `target/release/hypura` and/or zip:

```sh
gh release create v0.9.0 \
  --title "Hypura v0.9.0" \
  --notes "See README." \
  target/release/hypura
```

## 8. Verify

```sh
gh release view v0.9.0
```

## 9. CI tokens

For automation, use a **`GH_TOKEN`** (or `GITHUB_TOKEN`) with `contents: write` on the repository; do not reuse personal tokens in shared runners without scoping.
