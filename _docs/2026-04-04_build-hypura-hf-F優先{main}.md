# 実装ログ: build-hypura-hf.ps1 を F: 優先に変更

- **日付**: 2026-04-04（参照 UTC: `2026-04-04T10:07:21+00:00`）
- **worktree**: main（リポジトリルート `hypura-main`）

## 変更内容

- `scripts/build-hypura-hf.ps1` の `CARGO_TARGET_DIR` 候補順を **`F:\hypura-cargo-target` → `H:\hypura-cargo-target`** に変更。
- コメント・Usage・エラーメッセージを F 優先の説明に合わせて更新。

## 検証

- 手元で `cargo build` は未実行（ユーザー環境のディスク空きに依存）。

## メモ

- README: Windows ネイティブ節に `build-hypura-hf.ps1` の短い説明を追記（F 優先・`Hypura.exe` / `hypura.exe` コピー）。
