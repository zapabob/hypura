# 実装ログ: OpenClaw contextWindow/maxTokens 折衷設定

- 日付: 2026-03-24
- 対象: OpenClaw の Hypura プロバイダ設定
- 目的: `low context window` 警告を回避しつつ、生成安定性を維持する

## 変更内容

以下 2 ファイルの `models.providers.hypura.models[0]` を更新:

- `C:\Users\downl\.openclaw\openclaw.json`
- `C:\Users\downl\Desktop\clawdbot-main3\clawdbot-main\.openclaw-desktop\openclaw.json`

更新値:

- `contextWindow: 32768`（警告回避のため）
- `maxTokens: 1024`（実生成は小さめで安定寄り）

## 検証

`C:\Users\downl\.openclaw\openclaw.json` を読み出して以下を確認:

- `contextWindow = 32768`
- `maxTokens = 1024`

## 備考

- 実運用の OOM 回避を優先する場合、`maxTokens` は 512 まで下げるとさらに安全。
- 必要なら後続で、ワークロード別に `1024/768/512` のプリセット化を行う。
