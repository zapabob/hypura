# 実装ログ: Kobold GUI 本格寄せ

- ブランチ: `main`
- 対象: `hypura-main`

## 実装内容

1. `/kobold-lite` を Kobold運用向けUIへ置換
   - プリセット保存/読込/削除（`localStorage`）
   - sampler_order 入力
   - 生成 (`/api/v1/generate`)
   - ストリーミング生成 (`/api/extra/generate/stream`)
   - 中断 (`/api/extra/abort`)
   - 状態確認 (`/api/extra/generate/check`)

2. Kobold互換 API 拡張
   - 追加: `POST /api/extra/generate/stream` (NDJSON)
   - `check` は実行中 `"busy"` を返却
   - `abort` は実際のキャンセルフラグを立てる動作へ変更

3. sampler_order 厳密反映
   - `KoboldGenerateRequest` / `GenerateOptions` に `sampler_order` 追加
   - sampler chain を順序指定対応 (`new_with_order`)
   - 対応ID:
     - `0=top_k`
     - `2=top_p`
     - `5=temperature`
     - `6=repetition_penalty`
     - `7=min_p`
   - 未指定項目は既定順で補完

4. 推論ループ中断
   - `GenerateFromLoadedParams` に `cancel_flag` を追加
   - decode ループで token ごとにキャンセル確認

## 変更ファイル

- `src/server/routes.rs`
- `src/server/ollama_types.rs`
- `src/compute/ffi.rs`
- `src/compute/inference.rs`
- `src/cli/serve.rs`

## 検証

- `cargo check` 成功
- Lint エラーなし
