# TurboQuant Global Install 実装ログ

- Date: 2026-03-30
- Scope: このPC上で `llama-turboquant` をどのディレクトリからでも起動可能にする

## 実施内容

- 追加: `scripts/install_turboquant_global.ps1`
  - `vendor/llama.cpp/build-turboquant-cpu/bin/Release` の成果物を収集
  - `C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin` へ配置
  - `tqllama.cmd` シムを生成
  - User PATH へ `...\llama-turboquant\bin` を自動登録

## 検証

- インストーラ実行成功
  - `powershell -ExecutionPolicy Bypass -File scripts/install_turboquant_global.ps1`
- グローバル配置バイナリ起動確認
  - `C:/Users/downl/AppData/Local/Programs/llama-turboquant/bin/llama-turboquant.exe ...`
- シム起動確認
  - `tqllama.cmd` 実行で usage 表示を確認

## 備考

- 既存シェルでは PATH 反映前のことがあるため、新しい PowerShell を開くと確実に `llama-turboquant` が通る。
