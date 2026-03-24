# 実装ログ: serve 汎用化

## 実施日時
- 2026-03-25

## 目的
- デスクトップの `Hypura 中枢 (Ollama API).lnk` と文字化け名ショートカットの両方から、環境依存を減らして `serve` できるようにする。

## 変更内容
1. `scripts/hypura-central-smart.ps1`
   - 固定 `hypura.exe` パスを廃止。
   - 解決順序を `HYPURA_EXE` → `target/release/hypura.exe` → `dist/**/hypura.exe` に変更。
   - 固定 GGUF パスを廃止。
   - 解決順序を `HYPURA_MODEL` → `test-models/*.gguf` → `models/*.gguf` に変更。
   - 未解決時エラーを、次に取るべき行動が分かる文言へ改善。

2. `scripts/hypura-central-serve.ps1`
   - `smart` と同じ実行ファイル/モデル自動解決ロジックを適用。
   - PowerShell 自動変数衝突を避けるため `Host` を `BindHost` に変更。

3. `scripts/Configure-HypuraCentralShortcut.ps1`
   - 単一ショートカット名指定から、複数名 (`string[]`) を常時生成する方式に変更。
   - `Hypura 中枢 (Ollama API).lnk` と `Hypura 荳ｭ譫｢ (Ollama API).lnk` の両方を必ず再生成。

## 実行確認
- `hypura-central-smart.ps1 -ShowState` が成功（状態 JSON を出力）。
- `Configure-HypuraCentralShortcut.ps1` が成功し、対象2ショートカットを `ensured` で出力。
- PSScriptAnalyzer 警告は既存の `Ensure-Dir` 命名のみ（今回の変更起因ではない）。

## 補足
- MCP で日時取得可能な専用ツールがこの環境では確認できなかったため、ログ日付は PowerShell `Get-Date` で記録。
