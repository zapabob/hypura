# 実装ログ: rep_pen / stop_sequence + Kobold互換拡張

- 日時: 2026-03-30 (UTC基準)
- ブランチ: `main`
- 対象: `hypura-main`

## 今回の実装

1. `rep_pen` を推論ループへ反映
   - `SamplingParams` に `repeat_penalty` / `repeat_last_n` を追加
   - sampler chain に `llama_sampler_init_penalties(...)` を追加
   - Koboldリクエストの `rep_pen`, `rep_pen_range` を実際にサンプリングへ反映

2. `stop_sequence` を推論ループへ反映
   - `GenerateFromLoadedParams` に `stop_sequences` を追加
   - 生成中に `generated_text` の末尾一致を検出し停止
   - stop文字列一致時は返却テキストから停止文字列をトリム

3. Kobold互換エンドポイントを拡張
   - 追加: `GET /api/extra/true_max_context_length`
   - 返却: `{ "value": <context_length> }`

4. TurboQuant `--tq-*` パラメータを `serve` CLI へ露出
   - `--tq-so8-off`
   - `--tq-so8-learned`
   - `--tq-triality-off`
   - `--tq-triality-mix`
   - `--tq-rotation-seed`
   - `--tq-artifact`
   - 起動時に対応する `LLAMA_TURBOQUANT_*` 環境変数へ反映

5. API側にも TurboQuant ランタイム上書き項目を追加
   - `options.tq_so8_off`
   - `options.tq_so8_learned`
   - `options.tq_triality_off`
   - `options.tq_triality_mix`
   - `options.tq_rotation_seed`
   - `options.tq_artifact`

6. GUI (Kobold寄せの簡易版) を追加
   - `GET /kobold-lite`
   - Prompt / max_length / temperature / top_k / top_p / rep_pen / stop_sequence を入力して
     `POST /api/v1/generate` を直接叩ける

## 変更ファイル

- `src/compute/ffi.rs`
- `src/compute/inference.rs`
- `src/server/ollama_types.rs`
- `src/server/routes.rs`
- `src/cli/serve.rs`
- `src/main.rs`

## 検証

- `cargo check` 成功
- 変更ファイルのLintsエラーなし
