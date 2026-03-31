# 2026-03-31 semverリリース固め{main}

## 実施内容
- `hypura` / `hypura-sys` のバージョンを `0.1.2` へ更新。
- `README.md` を日英併記のリリース情報中心にリライト（重複英語ブロックを整理）。
- `ggml_map_custom` のリンクエラーを `ggml_map_custom1` へ置換して解消。
- `cargo build --release -j 2` を通し、`target_release_rtx/release/hypura.exe` を生成。
- RTX 3060/3080 安定版として成果物を tarball 化:
  - `dist/hypura-rtx3060-3080-stable-v0.1.2/`
  - `dist/hypura-rtx3060-3080-stable-v0.1.2.tar.gz`

## ビルド結果
- Release build: 成功（約 1m37s）
- 出力バイナリ:
  - `target_release_rtx/release/hypura.exe`

## 補足
- MCP 経由の現在時刻取得は `plugin-meta-quest-agentic-tools-hzdb` が未接続で取得不可。
- 代替としてシステム時刻に基づく日付でログを作成。
- 追補: `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` で段階運用検証（4096/64, 4096/256, 8192/512）と `serve + proxy` 3連続疎通を確認。README へ最新実測値を反映済み。
