# 2026-03-31 ggufモデル切替GUI追加{main}

## 目的
- `hypura serve` 実行中に、`/kobold-lite` から GGUF を選択して切替できるようにする。
- EasyNovelAssistant / Koboldcpp 風の「モデル選択して切替」運用を Hypura 側で可能にする。

## 実装内容
- サーバー状態を拡張:
  - `AppState` に `model_path`, `model_dir`, `default_context` を追加。
  - `model_name`, `gguf_info` を `Mutex` 保護に変更し、切替後に更新可能化。
- 新規 API:
  - `GET /api/extra/models`
    - `model_dir` 直下の `.gguf` 一覧を返却。
    - 現在ロード中モデルに `selected=true` を付与。
  - `POST /api/extra/model/switch`
    - 指定パスの GGUF を検証し、モデルをロードし直して差し替え。
    - 応答に `model`, `context` を返却。
    - 生成中は `409 conflict` で拒否。
- `kobold-lite` UI 拡張:
  - `GGUF Model Switcher` パネルを追加。
  - `modelList` ドロップダウン + `Refresh Models` + `Switch Model` を追加。
  - 切替成功時に `Connection`/`Model` 表示を更新。
  - 生成中ロックの除外ボタンにモデル関連ボタンを追加。

## 動作確認
- `cargo check` 成功。
- 変更ファイル:
  - `src/server/routes.rs`
  - `src/server/ollama_types.rs`
  - `src/cli/serve.rs`

## 注意点
- モデル探索は `serve` 起動時に指定したモデルの親ディレクトリ直下のみ対象。
- 切替時の TurboQuant モードは、既存互換を優先して `exact` で再ロードしている。
