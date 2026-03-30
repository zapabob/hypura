# TurboQuant HF→GGUF 実装ログ

- Date: 2026-03-30
- Workspace: `hypura-main`
- Converter repo: `C:/Users/downl/Desktop/koboldcpp-turboquant-integration/koboldcpp`

## 追加実装

- 新規: `convert_hf_to_gguf_turboquant.py`
  - 既存 `convert_hf_to_gguf.py` をラップし、TurboQuant拡張を付与。
  - 追加CLI:
    - `--tq-enable`
    - `--tq-so8-learned`
    - `--tq-triality-enable`
    - `--tq-triality-mix`
    - `--tq-artifact-out`
    - `--tq-seed`
    - `--tq-train-rows`
    - `--tq-so8-steps`
    - `--tq-so8-lr`

## 実装内容

1. SO8学習統合
   - Turboquant-CUDA の `TurboQuantMSE` を利用し、8次元ブロックの SO8 回転を学習。
   - `--tq-so8-learned` 無効時は static SO8 回転を使用。

2. Triality codebook生成
   - 学習サンプルから 3-centroid 近似コードブックを生成。
   - 変換時のテンソル処理で triality proxy を mix 係数付きで適用。

3. 量子化パス組み込み
   - `modify_tensors` フックで対象テンソル（attention 系、末尾次元8の倍数）に blockwise 変換を適用。
   - TurboQuant無効時は既存変換パスをそのまま通過。

4. GGUFメタデータ埋め込み (`TQCUDA1` 互換)
   - `set_gguf_parameters` フックで以下を追加:
     - `turboquant.enabled`
     - `turboquant.so8_enabled`
     - `turboquant.so8_learned`
     - `turboquant.triality_enabled`
     - `turboquant.triality_mix`
     - `turboquant.rotation_seed`
     - `turboquant.artifact_ref`
     - `turboquant.rotation8`
     - `turboquant.triality_codebook`
   - `TQCUDA1` artifact をテキスト形式で出力。

## 検証結果

- `py -3 convert_hf_to_gguf_turboquant.py --print-supported-models` 成功。
- `py -3 -c "import convert_hf_to_gguf_turboquant as m; print('import-ok')"` 成功。
- `py -3 convert_hf_to_gguf_turboquant.py --tq-enable C:/__not_exists__` で期待どおり入力ディレクトリエラーを確認。

## 既知制約

- 本ログ時点では「実モデルを使ったHF→GGUF変換のフルE2E（生成GGUFをKoboldCppで実ロード）」は未実施。
- Triality は現段階で lightweight proxy（3-centroid）実装。
- IDE lint では動的 import (`turboquant.*`) が未解決警告になるが、実行時 import は成功。
