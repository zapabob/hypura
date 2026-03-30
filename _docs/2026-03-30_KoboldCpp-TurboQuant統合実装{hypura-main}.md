# KoboldCpp TurboQuant 統合実装ログ

- Date: 2026-03-30
- Workspace: `hypura-main`
- Integration workspace: `C:/Users/downl/Desktop/koboldcpp-turboquant-integration`

## 実施内容

1. 統合リポ準備
   - `koboldcpp` と `Turboquant-CUDA` を Desktop 配下にクローン。
2. 差分監査
   - `Turboquant-CUDA` / `vendor/llama.cpp` / `koboldcpp` を比較し、採用方針を `_docs/2026-03-30_KoboldCpp-TurboQuant差分監査{hypura-main}.md` に記録。
3. KoboldCpp への移植
   - 追加: `koboldcpp/src/llama-turboquant.h`
   - 追加: `koboldcpp/src/llama-turboquant.cpp`
   - 変更: `koboldcpp/src/llama-kv-cache.h`
     - TurboQuant runtime config と K/V path ログフラグを追加。
   - 変更: `koboldcpp/src/llama-kv-cache.cpp`
     - env から `LLAMA_TURBOQUANT*` を読込。
     - K/V 書き込み経路で TurboQuant 有効ログを追加。
   - 変更: `koboldcpp/src/llama.cpp`
     - `#include "llama-turboquant.cpp"` を追加。
4. Runtime switch 配線
   - 変更: `koboldcpp/koboldcpp.py`
     - 新規 CLI オプション追加:
       - `--turboquant`
       - `--tq-so8-off`
       - `--tq-so8-learned`
       - `--tq-triality-off`
       - `--tq-triality-mix`
       - `--tq-rotation-seed`
       - `--tq-artifact`
     - `apply_turboquant_env()` を追加し、モデルロード時に env を適用。
5. EasyNovelAssistant 連携
   - 追加: `C:/Users/downl/Desktop/EasyNovelAssistant/EasyNovelAssistant/Run-EasyNovelAssistant-KoboldCpp-TurboQuant.bat`
     - KoboldCpp endpoint チェック
     - 未起動時に TurboQuant 付きで `koboldcpp.py` 起動
     - EasyNovelAssistant 本体を起動
6. パッケージ化
   - 追加: `koboldcpp/scripts/package_turboquant_koboldcpp.ps1`
   - `packages/koboldcpp-turboquant-cuda132` と ZIP を生成。
   - パッケージ内 `koboldcpp.py --help` スモーク成功（TurboQuant オプション表示確認）。

## CUDA 13.2 ビルド状況

- Configure:
  - `CUDAToolkit 13.2.51` を検出
  - `CMAKE_CUDA_ARCHITECTURES=75-virtual;80-virtual;86-virtual`
- Build:
  - `koboldcpp_cublas` ターゲットで NVCC コンパイル進行を確認
  - 既知警告:
    - `C4819`（コードページ警告）
    - `#221-D`（float 警告）
  - 警告は大量だが、CUDA 13.2 toolchain でコンパイルフェーズが進行することを確認。

## 備考

- パッケージスクリプトは DLL が存在すれば自動同梱する設計。
- ビルド完了後に同スクリプトを再実行すると、配布 ZIP に実バイナリが反映される。

## 追加実装（HF→GGUF TurboQuant コンバータ）

- 追加: `C:/Users/downl/Desktop/koboldcpp-turboquant-integration/koboldcpp/convert_hf_to_gguf_turboquant.py`
  - 既存 `convert_hf_to_gguf.py` ラッパーとして実装。
  - SO8学習（TurboQuantMSE）・Triality proxy・`TQCUDA1` artifact出力・GGUF TurboQuantメタデータ埋め込みを実装。
  - 詳細は `_docs/2026-03-30_TurboQuant-HF2GGUF{hypura-main}.md` を参照。
