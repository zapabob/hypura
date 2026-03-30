# 実装ログ: sampler 完全互換拡張

- 日付: 2026-03-30
- ブランチ: `main`

## 目的
Kobold寄せで未対応だった `top_a / tfs / typical` を実演算で反映し、`sampler_order` の互換性を引き上げる。

## 変更内容

1. FFI 拡張 (`hypura-sys`)
   - 追加ファイル:
     - `hypura-sys/src/hypura_sampler_ext.h`
     - `hypura-sys/src/hypura_sampler_ext.c`
   - 追加関数:
     - `hypura_sampler_init_top_a(float a, size_t min_keep)`
     - `hypura_sampler_init_tfs_z(float z, size_t min_keep)`
   - `wrapper.h` にヘッダ追加
   - `build.rs` に C ソース追加・rerun-if-changed 追加

2. SamplingParams 拡張
   - `src/compute/ffi.rs`:
     - `top_a`
     - `tfs`
     - `typical`
   - 既定値:
     - `top_a = 0.0`
     - `tfs = 1.0`
     - `typical = 1.0`

3. sampler_order 厳密拡張
   - `LlamaSampler::new_with_order` の対応IDを拡張:
     - `0=top_k`
     - `1=top_a`
     - `2=top_p`
     - `3=tfs`
     - `4=typical`
     - `5=temperature`
     - `6=repetition_penalty`
     - `7=min_p`
   - 未指定項目は既定順で補完

4. API/GUI 受け渡し更新
   - `src/server/ollama_types.rs`
     - `GenerateOptions` に `top_a / tfs / typical` を追加
   - `src/server/routes.rs`
     - Kobold generate handlers と `build_sampling()` で反映
     - GUI default sampler order を `6,0,1,3,4,2,5` に更新

## 検証

- `cargo check` 成功
- Lint エラーなし
