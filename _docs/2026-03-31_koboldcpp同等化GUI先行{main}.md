# 2026-03-31 koboldcpp同等化GUI先行{main}

## 目的
- `/kobold-lite` を KoboldCpp 運用導線へ寄せ、GUI先行で同等化を進める。

## 実装概要
- GUIレイアウト再編:
  - 右カラムに `UI Mode` / `Server Presets` / `Generation History` / `Event Log` を追加。
  - `Chat/Instruct/Storywriter/Adventure` のモード切替ボタンを追加。
- 生成パラメータ拡張:
  - `mirostat`, `mirostat_tau`, `mirostat_eta`, `dynatemp_*`, `smoothing_factor`,
    `presence_penalty`, `frequency_penalty` をGUIとJSON受理型に追加。
  - CLIブリッジの生成/逆変換対象も拡張。
- サーバーAPI追加:
  - `GET /api/extra/presets/list`
  - `POST /api/extra/presets/save`
  - `POST /api/extra/presets/delete`
  - `GET /api/extra/history`
  - `GET /api/extra/events`
- 可観測性:
  - 生成成功時に履歴を記録。
  - 失敗/切替時にイベントログを記録。
- CLI追従:
  - `serve` に `--model-dir` と `--ui-theme` を追加。
  - `model_dir` を AppState へ反映し、モデル一覧スキャン範囲を制御可能にした。

## 変更ファイル
- `src/server/routes.rs`
- `src/server/ollama_types.rs`
- `src/main.rs`
- `src/cli/serve.rs`
- `README.md`

## 検証
- `cargo check` 成功。
