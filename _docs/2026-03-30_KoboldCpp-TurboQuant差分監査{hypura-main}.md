# KoboldCpp TurboQuant 差分監査

- Date: 2026-03-30
- Target:
  - `C:/Users/downl/Desktop/koboldcpp-turboquant-integration/koboldcpp`
  - `C:/Users/downl/Desktop/koboldcpp-turboquant-integration/Turboquant-CUDA`
  - `C:/Users/downl/Desktop/hypura-main/hypura-main/vendor/llama.cpp`

## 監査結果サマリ

- `Turboquant-CUDA` は Python 実装が正本で、SO8 学習・Triality・評価統計が最も充実。
- 既存 `vendor/llama.cpp` の TurboQuant 実装は C++ 側の最小実装で、現状は推論経路の有効化ログと簡易 Triality コードブック評価が中心。
- `koboldcpp` 側には TurboQuant 関連コードが未実装のため、`vendor/llama.cpp` 差分を土台にしつつ、`Turboquant-CUDA` の学習/評価ロジックを段階移植する方針が妥当。

## 差分マトリクス

| 領域 | Turboquant-CUDA | vendor/llama.cpp | 採用判断 |
|---|---|---|---|
| SO8回転学習 | `turboquant_mse.py` で `block_so8_learned` 最適化あり | `llama-turboquant.cpp` は static/簡易回転処理 | KoboldCppには `Turboquant-CUDA` 仕様優先で移植 |
| Triality | `research_extension/k_triality.py` に vector/spinor proxy + 統計 | 3-centroid 簡易コードブック評価 | KoboldCppは Triality mode 名・I/O を Python仕様に寄せる |
| Artifact I/O | `.pt` ベース + JSON/CSV統計 | 独自 `TQCUDA1` テキスト形式 | 互換重視で当面 `TQCUDA1` 維持 + 将来 `pt` 読込拡張 |
| llama 統合点 | なし（参照用） | `llama-kv-cache.*` / envフラグ接続あり | KoboldCppへ `llama-kv-cache` 経路を優先移植 |
| 学習/評価CLI | scripts 群が豊富 | `tools/turboquant/turboquant.cpp` は smoke 用 | KoboldCppに `llama-turboquant` CLI を移植し拡張 |
| CUDA13.x | Python CUDA runtime 前提 | C++ CUDAカーネル未追加 | KoboldCpp CMake（既存 CUDA13 分岐）を活用し統合 |

## 具体的採用対象

- `vendor/llama.cpp` から採用:
  - `src/llama-turboquant.h`
  - `src/llama-turboquant.cpp`
  - `src/llama-kv-cache.h` / `src/llama-kv-cache.cpp` の TurboQuant runtime hook
  - `tools/turboquant/*`
- `Turboquant-CUDA` から採用:
  - SO8 学習ロジック仕様（`turboquant_mse.py`）
  - Triality mode 定義・命名（`k_triality.py`）
  - 評価軸（hidden/logit/attention error + 統計）

## リスク

- KoboldCpp は upstream llama.cpp から独自拡張が大きく、直接パッチ適用では衝突しやすい。
- `koboldcpp` の CMake は通常向けではなく CUBLAS 用警告付き構成のため、ビルド検証は限定ターゲットから段階実施が必要。
