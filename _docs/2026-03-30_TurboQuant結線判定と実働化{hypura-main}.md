# TurboQuant 結線判定と実働化ログ

- DateTime: 2026-03-30 22:02:09 +09:00
- Workspace: `hypura-main`
- Scope: TurboQuant runtime 経路の可視化、C codec FFI 実運用接続、検証結果の整理

## 実施内容

1. TurboQuant runtime 経路可視化を追加
   - 実行経路を `RustDirectCallback` / `CCodecFfi` の2系統で明示。
   - `HYPURA_TURBOQUANT_RUNTIME` で経路を切替可能化（既定は Rust 直結）。
   - セッション完了ログ（request/blocking/NVMe）を INFO で出力し、`kq_overwrites` / `kqv_overwrites` と併せて実働証拠を残せる状態にした。

2. C codec runtime への接続を追加
   - `KvCodecRuntimeBridge` を追加し、`hypura_kv_codec_runtime_create/free` を RAII 管理。
   - C runtime 経由で以下を呼べるようにした:
     - `compress_k`
     - `compress_v`
     - `score_k`
     - `read_v`
   - `paper-full-kv` の場合に `compress_values=1` を設定。

3. C API の不足関数を補完
   - `hypura-sys/src/hypura_kv_codec.h` と `hypura-sys/src/hypura_kv_codec.c` に以下を追加:
     - `hypura_kv_codec_compress_v_vec`
     - `hypura_kv_codec_read_v_vec`

4. TurboQuant-CUDA 直接リンク方針
   - 今回は `vendor/turboquant-cuda` の直接リンクは追加しない方針を維持。
   - 理由: 現行実装は Rust codec と C shim の接続確認を主目的とし、`build.rs` のリンク変更は検証軸を増やすため段階分離した。

## 検証結果

- `cargo check`
  - 成功。
- `cargo test --lib kv_codec_ffi::tests::ffi_bridge_creation -- --nocapture`
  - 成功（1 passed）。
- `cargo test --lib runtime_tensor_component_maps_attention_nodes -- --nocapture`
  - 成功（1 passed）。

## 追加で判明した事項（既存課題）

- `cargo test`（bin test を含む）では Windows リンク段で以下エラーを確認:
  - `ggml_map_custom` の未解決外部シンボル（`hypura_op_compress_k` 参照）
  - `LINK1120`
- これは今回追加の runtime 経路可視化とは別の、Windows テストリンク構成側の既存課題として切り分け。

## 判定（今回の結論）

- 「受け口のみ」ではなく、Rust 推論経路から C codec runtime を使う実運用経路を選択可能な状態にした。
- ただし `vendor/turboquant-cuda` の直接リンク E2E は今回対象外（段階3で実施予定）。

