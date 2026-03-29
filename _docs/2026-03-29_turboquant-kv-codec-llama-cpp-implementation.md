# TurboQuant KV Codec Implementation Log

**Date:** 2026-03-29  
**Author:** zapabob  
**Status:** ✅ Complete (Phase B+)

---

## Overview

TurboQuant KV cache compression integration with llama.cpp, following the paper:
"TurboQuant: A Fast and Efficient Post-Training KV Cache Compression" (arXiv 2504.19874)

## Architecture

### Paper Implementation (2-stage)

**Stage 1 (Key Compression):**
1. 明示的ノルム (Explicit normalization)
2. 回転行列 (Rotation matrix R)
3. スカラー量子化 (Scalar quantization with Lloyd-Max centroids)
4. QJL残差補正 (Residual QJL estimator)

**Stage 2 (Value Handling):**
- `paper-key-only`: V = exact保持
- `paper-full-kv`: V = 同じコーデックで圧縮

### Implementation Stack

```
llama.cpp (C++)
    │
    ├── include/llama-kv-codec.h     # KV codec C API
    ├── src/llama-kv-codec.cpp       # Default params, set/get
    ├── src/llama-context.h          # kv_codec_params追加
    └── src/CMakeLists.txt           # ビルド設定
            │
            ▼
hypura-sys (FFI Bridge)
    │
    ├── hypura_kv_codec.h            # C API定義
    ├── hypura_kv_codec.c            # GGMLカスタムOP
    └── build.rs                     # コンパイル設定
            │
            ▼
hypura (Rust)
    │
    ├── src/cache/kv_codec.rs        # TurboQuantコーデック実装
    ├── src/cache/kv_codec_ffi.rs    # FFIブリッジ
    ├── src/model/turboquant_sidecar.rs  # sidecar config
    └── src/compute/inference.rs     # eval_callback統合
```

## Changes

### llama.cpp (zapabob/llama.cpp)

| File | Change | Description |
|------|--------|-------------|
| `include/llama-kv-codec.h` | 新規 | KV codec C API定義 |
| `src/llama-kv-codec.cpp` | 新規 | デフォルトパラメータ、set/get関数 |
| `src/llama-context.h` | 変更 | kv_codec_params追加 |
| `include/llama.h` | 変更 | 新ヘッダーinclude |
| `src/CMakeLists.txt` | 変更 | 新ソースファイル追加 |

**PR:** https://github.com/ggml-org/llama.cpp/pull/21142

### hypura-sys

| File | Change | Description |
|------|--------|-------------|
| `hypura_kv_codec.h` | 新規 | KVコーデックC API |
| `hypura_kv_codec.c` | 新規 | GGMLカスタムOP実装 |
| `wrapper.h` | 変更 | 新ヘッダー追加 |
| `build.rs` | 変更 | 新Cファイルのコンパイル追加 |

### hypura (Rust)

| File | Change | Description |
|------|--------|-------------|
| `src/cache/kv_codec.rs` | 新規 | TurboQuant論文準拠コーデック |
| `src/cache/kv_codec_ffi.rs` | 新規 | Rust FFIブリッジ |
| `src/cache/mod.rs` | 変更 | 新モジュール追加 |
| `src/model/turboquant_sidecar.rs` | 新規 | sidecar config読込・検証 |
| `src/compute/inference.rs` | 変更 | eval_callbackによるK/Vインターセプト |
| `src/cli/serve.rs` | 変更 | state.clone()修正 |

**PR:** https://github.com/zapabob/hypura/pull/1

## API Design

### C API (llama-kv-codec.h)

```c
// コーデックパラメータ
struct llama_kv_codec_params {
    llama_kv_codec * codec;
    llama_kv_codec_compress_k_fn compress_k;
    llama_kv_codec_compress_v_fn compress_v;
    llama_kv_codec_score_k_fn    score_k;
    llama_kv_codec_read_v_fn     read_v;
    llama_kv_codec_reset_fn      reset;
    llama_kv_codec_free_fn       free_codec;
    uint32_t n_layer;
    uint32_t n_kv_head;
    uint32_t head_dim;
    bool compress_keys;
    bool compress_values;
    bool use_codec_score;
    bool use_codec_readv;
};

// 使用例
struct llama_kv_codec_params codec_params = llama_kv_codec_default_params();
codec_params.codec = my_codec;
codec_params.compress_k = my_compress_k;
codec_params.score_k = my_score_k;
codec_params.compress_keys = true;
llama_set_kv_codec(ctx, &codec_params);
```

### Rust API (kv_codec.rs)

```rust
pub trait KvCodec {
    fn name(&self) -> &'static str;
    fn ingest_k(&self, layer: u32, head: u32, token: u32, k: &[f32]) -> Result<Vec<f32>>;
    fn ingest_v(&self, layer: u32, head: u32, token: u32, v: &[f32]) -> Result<Vec<f32>>;
    fn score_k(&self, layer: u32, head: u32, query: &[f32], tokens: Range<u32>) -> Result<Vec<f32>>;
    fn read_v(&self, layer: u32, head: u32, tokens: Range<u32>) -> Result<Vec<f32>>;
    fn fork_session(&self) -> Box<dyn KvCodec + Send>;
}
```

## Integration Flow

### eval_callback Integration (inference.rs)

```
llama.cpp GGML Graph
      │
      ▼
eval_callback_with_runtime()
      │
      ├─ Qcur → query_vectors保存
      ├─ Kcur → codec.ingest_k() → 圧縮Kで上書き
      ├─ Vcur → codec.ingest_v() → 圧縮Vで上書き
      ├─ kq   → codec.score_k() → スコア上書き
      └─ kqv  → codec.read_v()  → 出力上書き
```

## Testing

### Test Results

```
test result: ok. 50 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### TurboQuant関連テスト

| テスト | 結果 |
|-------|------|
| `cache::kv_codec::exact_codec_round_trips_values` | ✅ |
| `cache::kv_codec::paper_key_only_keeps_values_exact_and_scores_from_state` | ✅ |
| `cache::kv_codec::paper_full_kv_reconstructs_values` | ✅ |
| `cache::kv_codec_ffi::ffi_exact_mode` | ✅ |
| `cache::kv_codec_ffi::ffi_bridge_creation` | ✅ |
| `model::turboquant_sidecar::paper_config_parse_and_validate` | ✅ |
| `model::turboquant_sidecar::schema_mode_mismatch_fails` | ✅ |
| `model::turboquant_sidecar::exact_mode_ignores_sidecar` | ✅ |

## Build Commands

```powershell
# hypuraビルド
cd C:\Users\downl\Desktop\hypura-main\hypura-main
$env:CARGO_INCREMENTAL = "0"
cargo build --release

# テスト実行
cargo test --release --lib

# KVコーデックテストのみ
cargo test --release --lib -- cache::kv_codec

# TurboQuantテストのみ
cargo test --release --lib -- model::turboquant
```

## Configuration

### Sidecar Config Format

```json
{
  "schema": "paper",
  "codec": "turboquant-prod",
  "num_layers": 32,
  "num_kv_heads": 8,
  "head_dim": 128,
  "key_bits": 3.5,
  "value_exact": true,
  "rotation": {
    "seed": 7,
    "matrix": [1.0, 0.0, ...]
  },
  "scalar_quantizer": {
    "centroids": [-1.0, -0.25, 0.25, 1.0],
    "decision_boundaries": [-0.5, 0.0, 0.5]
  }
}
```

## Future Work

1. **Stage 3**: 動的ビット割り当て (Dynamic bit allocation)
2. **Stage 4**: アウトライア検出 (Outlier channel detection)
3. **GPU最適化**: CUDA/Metalカーネルでの量子化
4. **ベンチマーク**: 既存手法との比較評価

## References

- Paper: https://arxiv.org/abs/2504.19874
- llama.cpp PR: https://github.com/ggml-org/llama.cpp/pull/21142
- hypura PR: https://github.com/zapabob/hypura/pull/1
