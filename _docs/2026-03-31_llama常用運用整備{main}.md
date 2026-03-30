# llama.cpp 常用運用整備ログ

- 実施日: 2026-03-31
- 対象モデル: `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
- モデルパス: `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\KoboldCpp\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`

## 1. 起動前提の固定

- 実行: `chcp 65001`
- 実行: `llama-cli --version`
- 確認: `version: 8483 (abe37ec28)` / CUDAバックエンド検出あり

## 2. 基準設定の安定性確認

以下の設定を固定して3回連続で単発推論を実行:

- `-ngl 99`
- `-c 8192`
- `-st`（単発ターンで自動終了）

結果:

- 3/3 回で起動・生成・終了まで成功
- 各回で CUDA 検出とVRAMメモリ内訳ログを確認
- 生成速度（Generation）は約 `39.7 ~ 45.5 t/s`

## 3. 常用ランチャー作成

`run-qwen35.cmd` を作成:

- `chcp 65001` を毎回自動実行
- 既定モデルパスを固定
- 通常起動: `llama-cli -m <model> -ngl 99 -c 8192 -i`
- `--version` 引数時はバージョン確認のみ実行

## 4. 常用コマンド

- 対話常用: `run-qwen35.cmd`
- バージョン確認: `run-qwen35.cmd --version`
- 単発テスト（対話終了まで自動）:
  - `llama-cli -m "<model_path>" -ngl 99 -c 8192 -st -p "テスト文" -n 32 --temp 0.7 --simple-io`
