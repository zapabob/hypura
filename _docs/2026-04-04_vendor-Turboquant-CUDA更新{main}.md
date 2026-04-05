# vendor/turboquant-cuda 最新版への更新

- **実施日時（ローカル）**: 2026-04-04（作業ツリー: `main`）
- **リモート**: `https://github.com/zapabob/Turboquant-CUDA`
- **ブランチ**: `main`

## 変更内容

| 項目 | 更新前 | 更新後 |
|------|--------|--------|
| コミット | `3d39e7d` | `e3e7333492ef17ea2443010ec77516ffa4f69680` |

- `git pull --ff-only origin main` で fast-forward（Git LFS による `Filtering content` を含む）。
- 上流の主な増分: Multiscreen KV、Triality SO(8)、VRAM スクリプト、README 刷新、Rust workspace など（先頭コミットメッセージ: `feat: multiscreen KV, Triality SO(8), VRAM scripts; README rewrite; rust workspace`）。

## CoT（仮説→検証）

1. **仮説**: サブモジュールは `.gitmodules` の URL から `origin/main` が正本。
2. **検証**: `git fetch` → `HEAD..origin/main` に複数コミット → `pull --ff-only` で衝突なし完了、`rev-parse HEAD` が `e3e7333` と一致。

## 親リポジトリ側

- `git add vendor/turboquant-cuda` でサブモジュール指し示しコミットをステージ済み（マージ・プッシュはユーザー判断）。

## 備考

- Hypura 本体の `vendor/turboquant-cuda` 相対パス（例: `kv_codec_python.rs`）は変更不要。
- 上流に `vendor/turboquant-cuda/rust/hypura/` のミラーが増えているため、将来の同期方針は別タスクで検討可。
