# build-hypura-c.ps1 と HF スクリプトのプロセス終了（実装ログ）

**日付**: 2026-04-04（worktree: `main`）

## 目的

- **C: 上で高速差分ビルド**: `CARGO_TARGET_DIR` を `%LOCALAPPDATA%\hypura-cargo-target` に固定し `CARGO_INCREMENTAL=1`（インクリメンタル再利用）。
- **警告ゼロで機能維持**: 既定 `RUSTFLAGS=-D warnings`（型まわり含むコンパイラ警告をビルド失敗にする）。依存側で止まるときだけ `-AllowWarnings`。
- **上書きインストール**: ビルド後に `Hypura.exe` / `hypura.exe` をコピー。**ロック回避**のため、コピー前に `Hypura` / `hypura` プロセスを終了（`-NoKill` でスキップ）。
- **コピペ用**: `-PasteCommands` で手動上書き用の `Copy-Item` ブロックを表示、`-Clipboard` でクリップボードへ同ブロックを入れる。`%USERPROFILE%\.cargo\bin\cargo.exe` を優先して呼び出し（PATH に無くてもビルド可）。

## 使い方（リポジトリルート）

```powershell
.\scripts\build-hypura-c.ps1
.\scripts\build-hypura-c.ps1 -InstallTo "C:\Tools\Hypura"
.\scripts\build-hypura-c.ps1 -PasteCommands
.\scripts\build-hypura-c.ps1 -Clipboard
```

## F:/H: 向け（`build-hypura-hf.ps1`）

C 版と**同じオプション**に揃えた（2026-04-04 更新）。

- `CARGO_TARGET_DIR`: F: の `F:\hypura-cargo-target` を優先、無理なら H: の `H:\hypura-cargo-target`。
- 既定 `RUSTFLAGS=-D warnings`、`-AllowWarnings` で解除。
- `-InstallTo`（複数可）、`-AlsoCopyTo`、`-NoCopyToRepo`、`-PasteCommands`、`-Clipboard`。
- 各インストール先に **`Hypura.exe` と `hypura.exe` の両方**を上書き（旧 HF は `AlsoCopyTo` で `hypura.exe` だけ出してなかったのを修正）。
- `Get-CargoExe` で `%USERPROFILE%\.cargo\bin\cargo.exe` を優先。

```powershell
.\scripts\build-hypura-hf.ps1
.\scripts\build-hypura-hf.ps1 -AlsoCopyTo "F:\Hypura" -PasteCommands
.\scripts\build-hypura-hf.ps1 -InstallTo "F:\Tools\Hypura" -Clipboard
```

## Cargo: `Blocking waiting for file lock on artifact directory`

同じ `CARGO_TARGET_DIR` を別の `cargo` が掴んでいるときに出る（別ターミナル、rust-analyzer、前回のビルドなど）。

1. **しばらく待つ** — 先に動いているビルドが終われば進むことが多い。
2. **他のターミナル**で `cargo` が動いていないか確認し、不要なら終了する。
3. **タスクマネージャー**で `cargo.exe` / `rustc.exe` が複数いないか見る。止めてよいものだけ終了する。
4. **ずっと止まる**ときは PC 再起動するか、Cargo 関連プロセスを終了してから再実行する。

### スクリプト側（自動化）

- **`build-hypura-c.ps1` / `build-hypura-hf.ps1`**: 既定で他に `cargo.exe` がいると **警告**を出す。ビルド前に他 `cargo` を落とすなら **`-StopOtherCargo`**（rust-analyzer 等も止まるので注意）。
- **`scripts/stop-cargo-builds.ps1`**: 実行中の **すべての `cargo.exe`** を終了（手動・詰まったとき用）。

## 注意

- `-D warnings` は依存クレート由来の警告でもビルドが止まる場合がある。そのときは `-AllowWarnings` を使う（C/HF 共通）。
