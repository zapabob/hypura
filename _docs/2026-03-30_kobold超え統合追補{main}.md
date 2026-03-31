# 2026-03-30 Kobold超え統合 追補ログ

## 実装目的
- OpenClaw + EasyNovelAssistant 同時互換を維持しつつ、Hypura の GUI/CLI 同等化と運用強化を進める。

## 実装内容
- `src/server/ollama_types.rs`
  - `ChatRequest` に `think` 受理を追加（互換受理、現時点は無害化）。
  - `GenerateOptions` に `num_ctx` を受理追加。
  - `ShowRequest` を `model` / `name` 両受理へ変更（OpenClaw `/api/show` 契約対策）。
  - `KoboldGenerateRequest` に TurboQuant ランタイム上書き項目を追加。

- `src/server/routes.rs`
  - `/api/show` で `name` または `model` を受理し、`model_info.general.name` に反映。
  - Kobold generate 系 (`/api/v1/generate`, `/api/extra/generate/stream`) で
    TurboQuant 上書き (`tq_*`) を適用。
  - Kobold stream 最終行に `tok_per_sec_avg`, `prompt_eval_ms` を追加。
  - `/kobold-lite` GUI を Parity+ 化:
    - TurboQuant パラメータ入力欄
    - Preset Diff
    - CLI Export/Import ブリッジ
    - Retry Last
    - Runtime Metrics 表示
    - preset format version (`hypura_kobold_presets_v2`)

## 互換検証
- Rust ビルド検証:
  - `cargo check` 成功（Hypura 側）
- OpenClaw 側互換検証:
  - `pnpm vitest extensions/ollama/index.test.ts extensions/ollama/provider-discovery.contract.test.ts --run`
  - `2 files / 6 tests passed`
- EasyNovelAssistant 側導線検証:
  - `py -3 -m py_compile .../EasyNovelAssistant/src/kobold_cpp.py` 成功
  - 呼び出し先が `kobold: /api/v1/generate + /api/extra/abort`、`hypura: /api/generate` であることを再確認。

## 備考
- 既存作業中の差分（sampler/ffi/TurboQuant/既存docs など）は本ログの対象外として保持。
- MCP 経由時刻取得は利用可能ツール制約のため UTC 換算フローを併用して記録。
- 2026-03-31 時点で `/kobold-lite` の UIUX は Parity++ へ拡張（段階プリセット、生成中ロック、接続ステータス、メトリクスカード、進捗表示）。詳細は `2026-03-31_kobold-lite段階運用UIUX強化{main}.md` を参照。
