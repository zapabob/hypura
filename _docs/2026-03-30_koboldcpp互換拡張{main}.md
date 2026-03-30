# 実装ログ: KoboldCpp互換拡張

- 実装日時 (MCP取得 UTC): `2026-03-30T13:02:08+00:00`
- 対象ブランチ: `main`
- 目的:
  - `EasyNovelAssistant` が利用する KoboldCpp API を `Hypura` でも受けられるようにする
  - TurboQuant対応済み KoboldCpp 成果物を EasyNovelAssistant 配下へ同期する手順を自動化する

## 変更内容

### 1) Hypura に KoboldCpp 互換 API を追加

対象:
- `src/server/routes.rs`
- `src/server/ollama_types.rs`

追加したエンドポイント:
- `GET /api/v1/model`
- `POST /api/v1/generate`
- `GET /api/extra/generate/check`
- `POST /api/extra/abort`

互換レスポンス:
- `/api/v1/model` -> `{ "result": "<model_name>" }`
- `/api/v1/generate` -> `{ "results": [ { "text": "<generated>" } ] }`
- `/api/extra/generate/check` -> 空テキストを返すスタブ
- `/api/extra/abort` -> `{ "success": true }` を返すスタブ

備考:
- `rep_pen` / `stop_sequence` は現時点では受理のみで、推論エンジンへの適用は未実装

### 2) EasyNovelAssistant 側の TurboQuant KoboldCpp 同期スクリプト追加

対象:
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\Sync-KoboldCpp-TurboQuant.ps1`

役割:
- `koboldcpp-turboquant-integration\koboldcpp` から `EasyNovelAssistant\KoboldCpp` へ主要ファイルをコピー
- コピー対象:
  - `koboldcpp.py`
  - `koboldcpp_cublas.dll`
  - `ggml.dll`
  - `ggml-base.dll`
  - `ggml-cpu.dll`

## 検証

- `cargo check` を実行して `Hypura` の型・ビルド整合を確認 (成功)
- 既存の `cargo test` は別件リンクエラー (`ggml_map_custom` 未解決) が残存しているが、今回追加差分起因ではない

## 次フェーズ案

1. `rep_pen`・`stop_sequence` を `SamplingParams` と decode ループに反映  
2. `/api/extra/true_max_context_length` など Kobold 追加互換エンドポイントを拡張  
3. TurboQuantランタイム設定 (`--tq-*`) を Hypura CLI/HTTP オプションへ露出し、Kobold側と同等操作性に揃える
