# 2026-03-31 Kobold-lite段階運用UIUX強化{main}

## 目的
- Koboldcpp準拠GUI (`/kobold-lite`) を実運用向けに強化する。
- `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` で
  4096 -> 8192 の段階運用プリセットを安定確認する。

## UIUX 強化内容
- 段階プリセットボタンを追加:
  - Short (4096/64)
  - Medium (4096/256)
  - Long (8192/512)
- 生成中ロック:
  - 多重送信防止（Generate系を生成中に無効化）
  - Abortボタンは生成中のみ有効
- 接続ステータス表示:
  - `/api/v1/model` で接続確認
  - 最終成功時刻 / 最終エラー表示
- メトリクス表示をカード化:
  - tok/s
  - prompt ms
  - eval count
  - total ms
- 進捗表示:
  - token count / elapsed ms

## 回帰確認（API）
- `POST /api/show`: 成功（`name` 反映）
- `GET /api/v1/model` (proxy): 成功
- `POST /api/v1/generate` (proxy): 成功
- `POST /api/extra/generate/stream` (direct): 成功
  - final line: `done=true`
  - `tok_per_sec_avg`, `prompt_eval_ms` を確認
- `GET /api/extra/generate/check` (proxy): 成功
- `POST /api/extra/abort` (proxy): 成功

## 段階運用実測
- run short:
  - `--context 4096 --max-tokens 64` 成功
- run medium:
  - `--context 4096 --max-tokens 256` 成功
- run long:
  - `--context 8192 --max-tokens 512` 成功
- serve + proxy 3連続:
  - `iter=1 chars=89`
  - `iter=2 chars=35`
  - `iter=3 chars=56`
  - 全試行でモデル取得と生成成功

## 運用方針
- 常用は `serve --context 4096` を基準。
- 生成長だけ `64 -> 256 -> 512` の順で昇格。
- 不安定時は直前の成功段階へ即ロールバック。
