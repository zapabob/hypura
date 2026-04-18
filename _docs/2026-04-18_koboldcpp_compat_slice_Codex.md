# 2026-04-18 KoboldCpp Compat Slice Codex

## 概要
- `hypura` に KoboldCpp 互換の専用 CLI 入口 `hypura koboldcpp` を追加した。
- 既存サーバーへ KoboldCpp 互換のデフォルト値、Kobold-lite 自動起動、compat manifest を接続した。
- 互換面は pinned snapshot `v1.111.2` の第一段として、実装済み API と 501 stub を明示した。

## 背景・要求
- KoboldCpp 完全互換に向けて、製品互換レイヤーを段階的に進める。
- 既存の Hypura サーバーを壊さず、KoboldCpp 指向の起動導線と HTTP 互換面を整備する。
- 未実装面は隠さず manifest に残し、受け入れ範囲を明示する。

## 前提・判断
- pinned snapshot は `KoboldCpp v1.111.2 / API schema 2025.06.03` を基準にした。
- 今回は「完全互換の土台を通す」ことを優先し、単一 exe packaging や savedata bridge、multimodal bundling までは入れていない。
- 検証は `HYPURA_NO_CUDA=1` を使った CPU-only check を採用した。Rust 差分確認としては有効だが、CUDA 経路の新鮮な証跡にはなっていない。

## 変更ファイル
- `C:\Users\downl\Desktop\hypura-main\hypura-main\src\main.rs`
- `C:\Users\downl\Desktop\hypura-main\hypura-main\src\cli\serve.rs`
- `C:\Users\downl\Desktop\hypura-main\hypura-main\src\server\routes.rs`
- `C:\Users\downl\Desktop\hypura-main\hypura-main\docs\compat\koboldcpp-v1.111.2-parity-manifest.json`

## 実装詳細
- `src/main.rs`
  - `koboldcpp` / `compat` サブコマンドを追加した。
  - KoboldCpp プロファイル用に `port=5001`、`max_length`、`show_gui` を環境変数へ橋渡しするようにした。
- `src/cli/serve.rs`
  - 互換モード判定、`HYPURA_DEFAULT_GEN_AMT` 読み取り、Kobold-lite 自動オープン処理を追加した。
  - 互換 feature advertisement を実装済み機能だけに絞り、未実装 websearch/embeddings などを false で返すようにした。
  - 起動ログに Kobold 互換 API と GUI URL を出すようにした。
- `src/server/routes.rs`
  - 既存の compat API 面を KoboldCpp CLI プロファイルから使いやすいように整理した。
  - `/api/v1/config/max_length` が compat default max length を返す前提を server state へ接続した。
- `docs/compat/koboldcpp-v1.111.2-parity-manifest.json`
  - pinned snapshot、CLI profile、HTTP surface、stub 状態、残ギャップを manifest 化した。

## 実行コマンド
```text
rustfmt --edition 2021 src\main.rs src\cli\serve.rs src\server\mod.rs src\server\compat.rs src\server\ollama_types.rs src\server\routes.rs
$env:HYPURA_NO_CUDA='1'; cargo check --bin hypura --message-format short
$env:HYPURA_NO_CUDA='1'; cargo test server::compat::tests --lib -- --nocapture
$env:HYPURA_NO_CUDA='1'; cargo run --quiet --bin hypura -- koboldcpp --help
```

## テスト・検証結果
- `cargo check --bin hypura --message-format short`: 成功
  - `Finished 'dev' profile [unoptimized + debuginfo] target(s) in 2m 57s`
- `cargo test server::compat::tests --lib -- --nocapture`: 成功
  - 3 tests passed
- `cargo run --quiet --bin hypura -- koboldcpp --help`: 成功
  - `koboldcpp` サブコマンドが表示され、default port `5001` / `max-length` / `--no-show-gui` を確認

## 残リスク
- CUDA/実機 GPU 経路での fresh build 証跡は未取得。
- `/v1/embeddings`, `/api/extra/websearch`, `/sdapi/v1/txt2img`, `/api/extra/transcribe` は 501 stub のまま。
- savedata import/export bridge、launcher 完全互換、multimodal bundle は未実装。

## 次の推奨アクション
- `koboldcpp` プロファイルで実モデルを使った `--dry-run` と実サーバースモークを追加する。
- savedata bridge の JSON contract を manifest へ追記し、round-trip test を入れる。
- multimodal/TTS/STT/SD surface を順に 501 stub から実装へ置き換える。
