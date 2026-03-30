# KoboldCpp対応 実装ログ

- Date: 2026-03-30
- Scope: 既存 KoboldCpp 起動時に TurboQuant 設定を渡せるよう対応

## 追加ファイル

- `scripts/run_koboldcpp_turboquant.ps1`
  - KoboldCpp 実行ファイルの自動検出 or `-KoboldCppExe` 明示指定
  - TurboQuant 環境変数を設定して KoboldCpp を起動
  - 受け取った引数をそのまま KoboldCpp に透過転送
- `scripts/koboldcpp-tq.cmd`
  - PowerShell ランチャーの簡易ラッパー

## 対応した環境変数

- `LLAMA_TURBOQUANT`
- `LLAMA_TURBOQUANT_SO8`
- `LLAMA_TURBOQUANT_TRIALITY`
- `LLAMA_TURBOQUANT_TRIALITY_MIX`
- `LLAMA_TURBOQUANT_ROTATION_SEED`

## 使い方

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_koboldcpp_turboquant.ps1 -KoboldCppExe "C:\path\to\koboldcpp.exe" -- --model "C:\path\to\model.gguf"
```

または:

```cmd
scripts\koboldcpp-tq.cmd -KoboldCppExe "C:\path\to\koboldcpp.exe" -- --model "C:\path\to\model.gguf"
```

## 検証

- KoboldCpp 未検出時に、明確なエラーメッセージを返すことを確認
