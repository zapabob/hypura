# TurboQuant-CUDA統合 実装ログ

- Date (UTC): 2026-03-30
- Workspace: `hypura-main`
- Scope: `vendor/llama.cpp` への TurboQuant 統合（SO8/Triality・学習/評価CLI・KV経路フラグ統合）

## 実施内容

1. `vendor/turboquant-cuda` を参照し、llama.cpp 側で必要な責務を分離
   - ランタイム設定: TurboQuant on/off、SO8/Trialityフラグ
   - 学習/評価: Trialityコードブック生成・評価
   - アーティファクトI/O: 保存/再読込
2. TurboQuant コアモジュールを追加
   - `vendor/llama.cpp/src/llama-turboquant.h`
   - `vendor/llama.cpp/src/llama-turboquant.cpp`
3. llama 本体への接続
   - `vendor/llama.cpp/src/CMakeLists.txt` に `llama-turboquant.cpp` を追加
   - `vendor/llama.cpp/src/llama-kv-cache.h` に runtime config を追加
   - `vendor/llama.cpp/src/llama-kv-cache.cpp` に TurboQuant 設定読込と K/V 経路フックを追加
4. ビルド/ツール配線
   - `vendor/llama.cpp/CMakeLists.txt` に `LLAMA_TURBOQUANT` オプション追加
   - `vendor/llama.cpp/tools/CMakeLists.txt` に `tools/turboquant` を追加
   - `vendor/llama.cpp/tools/turboquant/CMakeLists.txt` を新規追加
   - `vendor/llama.cpp/tools/turboquant/turboquant.cpp` を新規追加（train/eval CLI）

## 検証

- CPUビルド（TurboQuant tool target）:
  - `cmake --build build-turboquant-cpu --config Release --target llama-turboquant`
  - 成功: `llama-turboquant.exe` 生成
- ツール実行スモーク:
  - `llama-turboquant.exe train --out turboquant_smoke.tq --head-dim 128 --vecs 128 --seed 7`
  - `llama-turboquant.exe eval --artifact turboquant_smoke.tq --vecs 64 --seed 9`
  - 成功: `triality_exact_mse`, `triality_proxy_mse`, `relative_mse_reduction` 出力を確認
- CUDAビルド:
  - `cmake -S . -B build-turboquant -DGGML_CUDA=ON -DLLAMA_TURBOQUANT=ON` で configure 成功
  - `cmake --build build-turboquant --config Release --target llama llama-turboquant` は CUDAコンパイル進行を確認

## 補足

- 既存 upstream 構造への影響を抑えるため、TurboQuant 実装は独立モジュール化して段階接続した。
- `vendor/llama.cpp` はサブモジュール差分として管理されるため、ルート `git status` では `m vendor/llama.cpp` 表示となる。
