# supergemma4-Q8_0 measurement summary

## Model file
- Path: `C:\Users\downl\Desktop\SO8T\gguf_models\Abiray\supergemma4-e4b-abliterated-GGUF\supergemma4-Q8_0.gguf`
- Size: `7.48 GB`
- SHA256 (first 16777216 bytes): `0f2bdcbf746adaf6d439d4eccdc820871c4c39be51f26162b3c73c7d9446eb4e`

## Metadata
- Format: `GGUF v3`
- Architecture: `gemma4`
- Parameters: `7.5B`
- Layers: `42`
- Embedding dim: `2560`
- Attention heads: `8`
- KV heads: `2`
- Context length: `131072`
- Quantization: `Q8_0`
- Total size: `7.5 GB`
- Tensors: `720`

## Tensor role stats
- Embedding: `2` tensors, `3.46 GB ` (46.32%)
- FfnDown: `42` tensors, `1.09 GB ` (14.59%)
- FfnGate: `42` tensors, `1.09 GB ` (14.59%)
- FfnUp: `42` tensors, `1.09 GB ` (14.59%)
- AttentionOutput: `42` tensors, `259.70 MB ` (3.39%)
- AttentionQuery: `42` tensors, `259.70 MB ` (3.39%)
- AttentionKey: `42` tensors, `64.40 MB ` (0.84%)
- AttentionValue: `42` tensors, `64.40 MB ` (0.84%)
- Other("per_layer_model_proj.weight"): `1` tensors, `52.50 MB ` (0.69%)
- Norm: `296` tensors, `2.16 MB ` (0.03%)
- Other("blk.0.inp_gate.weight"): `1` tensors, `680.00 KB ` (0.01%)
- Other("blk.0.proj.weight"): `1` tensors, `680.00 KB ` (0.01%)

## Tensor type stats
- Q8_0: `380` tensors, `7.42 GB ` (99.29%)
- F16: `1` tensors, `52.50 MB ` (0.69%)
- F32: `339` tensors, `2.16 MB ` (0.03%)

## Largest tensors
- `per_layer_token_embd.weight`: `2.8 GB` `Q8_0` `Embedding`
- `token_embd.weight`: `680.0 MB` `Q8_0` `Embedding`
- `per_layer_model_proj.weight`: `52.5 MB` `F16` `Other("per_layer_model_proj.weight")`
- `blk.0.ffn_down.weight`: `26.6 MB` `Q8_0` `FfnDown`
- `blk.0.ffn_gate.weight`: `26.6 MB` `Q8_0` `FfnGate`
- `blk.0.ffn_up.weight`: `26.6 MB` `Q8_0` `FfnUp`
- `blk.1.ffn_down.weight`: `26.6 MB` `Q8_0` `FfnDown`
- `blk.1.ffn_gate.weight`: `26.6 MB` `Q8_0` `FfnGate`
- `blk.1.ffn_up.weight`: `26.6 MB` `Q8_0` `FfnUp`
- `blk.2.ffn_down.weight`: `26.6 MB` `Q8_0` `FfnDown`

## Benchmark runs
- hypura legacy-3tier + off: generation `29.851 tok/s`, prompt `0.221 tok/s`, wall `208.4s`
- hypura four-tier + off: generation `0.173 tok/s`, prompt `1.098 tok/s`, wall `99.7s`
- hypura four-tier + auto: generation `0.167 tok/s`, prompt `1.096 tok/s`, wall `68.2s`

## Benchmark aggregate
- Generation tok/s mean: `10.063`
- Generation tok/s median: `0.173`
- Generation tok/s min/max: `0.167` / `29.851`
- Wall time mean: `125.4s`
