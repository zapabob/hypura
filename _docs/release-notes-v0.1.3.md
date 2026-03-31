# Hypura v0.1.3 - stable RTX 3060/3080 release

## 日本語

### このリリースについて

`v0.1.3` は Windows + RTX 3060/3080 を主対象にした安定運用継続リリースです。
SemVer を `0.1.3` に更新し、Kobold/OpenClaw 系の実運用ラインを維持したまま、
今回の段階運用実測（4096/8192）を反映して運用ガイドを整理しています。

### 主な変更

- SemVer 更新: `hypura` / `hypura-sys` を `0.1.3` へ更新。
- 配布更新: Windows 向け安定アーティファクト `hypura-rtx3060-3080-stable-v0.1.3.tar.gz` を提供。
- Kobold-lite UIUX 強化（Parity++）:
  - 段階プリセット（Short 4096/64, Medium 4096/256, Long 8192/512）
  - 生成中ロック（多重送信防止）と Abort 活性制御
  - 接続ステータス、最終成功時刻、最終エラー表示
  - メトリクスカード（tok/s, prompt ms, eval count, total ms）と進捗表示
- 互換維持:
  - Kobold/Ollama 互換 API を維持
  - stream終端メトリクス（`tok_per_sec_avg`, `prompt_eval_ms`）を維持

### 実測（latest session: 4096/8192 段階運用）

- `hypura run --context 4096 --max-tokens 64`: 成功
- `hypura run --context 4096 --max-tokens 256`: 成功
- `hypura run --context 8192 --max-tokens 512`: 成功
- `hypura serve --context 4096` + proxy (`/api/v1/*`) 3連続:
  - `iter=1 chars=89`
  - `iter=2 chars=35`
  - `iter=3 chars=56`
  - 全試行でモデル取得・生成ともに成功
- 追加確認:
  - `POST /api/show` 成功
  - `POST /api/extra/generate/stream` final line `done=true` + metrics keys 確認
  - `GET /api/extra/generate/check`, `POST /api/extra/abort` 成功

### 運用推奨（安全昇格）

- 常用ベース: `serve --context 4096`
- 段階昇格: `max_tokens 64 -> 256 -> 512`
- 昇格条件: 各段階で 3連続成功（`/api/tags`, `/api/v1/model`, `/api/v1/generate`）
- 異常時: 直前成功段階へ即ロールバック

## English

### About this release

`v0.1.3` continues the stability-focused Windows line for RTX 3060/3080.
It bumps SemVer to `0.1.3` and updates operational guidance using the latest
staged validation results (4096/8192) without breaking Kobold/OpenClaw flows.

### Highlights

- SemVer bump: `hypura` / `hypura-sys` -> `0.1.3`.
- Updated stable artifact:
  - `hypura-rtx3060-3080-stable-v0.1.3.tar.gz`
- Kobold-lite UIUX (Parity++) improvements:
  - One-click staged presets (Short/Medium/Long)
  - Generation lock to prevent duplicate submits
  - Connection status + last success/error visibility
  - Runtime metric cards + progress indicator
- Compatibility continuity:
  - Kobold/Ollama-compatible API behavior maintained
  - Stream-final metrics kept (`tok_per_sec_avg`, `prompt_eval_ms`)

### Latest measured operations (staged 4096/8192)

- `run` checks:
  - `--context 4096 --max-tokens 64`: pass
  - `--context 4096 --max-tokens 256`: pass
  - `--context 8192 --max-tokens 512`: pass
- `serve --context 4096` + proxy, 3 consecutive generate calls:
  - `iter=1 chars=89`
  - `iter=2 chars=35`
  - `iter=3 chars=56`
  - model discovery + generation succeeded on all runs
- Additional endpoint checks:
  - `/api/show` pass
  - `/api/extra/generate/stream` final `done=true` with metrics keys
  - `/api/extra/generate/check` and `/api/extra/abort` pass

### Recommended safe promotion path

- Baseline: `serve --context 4096`
- Increase generation budget gradually: `64 -> 256 -> 512`
- Promote only after 3 consecutive healthy passes per stage
- Roll back immediately to the previous stable stage on instability
