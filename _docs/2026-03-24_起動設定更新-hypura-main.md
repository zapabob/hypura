# 実装ログ: Hypura 起動設定更新（Q4_K_M / 16K / ヘッドレス / ログオン自動起動）

- 日付: 2026-03-24
- worktree: hypura-main

## 変更内容

1. `scripts/hypura-central-smart.ps1`
   - 既定モデルを `C:\Users\downl\Downloads\Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` に変更
   - 状態ファイル未作成時の既定 context を `16384` に変更
   - 状態ファイルが不正値の際のフォールバックも `16384` に変更

2. デスクトップショートカット更新
   - `C:\Users\downl\Desktop\Hypura 中枢 (Ollama API).lnk`
   - 引数を `-WindowStyle Hidden` に更新（ヘッドレス）

3. Startup 自動起動更新
   - `C:\Users\downl\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\HypuraAutoStart.cmd`
   - `powershell.exe ... -WindowStyle Hidden -File ...\hypura-central-smart.ps1` に更新

4. 接続整合の補正
   - `C:\Users\downl\.openclaw\openclaw.json`
   - `C:\Users\downl\Desktop\clawdbot-main3\clawdbot-main\.openclaw-desktop\openclaw.json`
   - `agents.defaults.model.primary` を `/api/tags` と一致する `hypura/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive` に修正

## 検証結果

`debug-verify-hypura-openclaw.ps1` 実行結果:

- `health OK: True`
- `/api/tags name: Qwen3.5-27B-Uncensored-HauhauCS-Aggressive`
- `primary vs tags match: True`
- `EXIT=0`

## 備考

- ログオン時自動起動は Startup フォルダ方式（管理者権限不要）
- ログイン前（OS起動直後）での起動は管理者権限付きタスクスケジューラが別途必要
