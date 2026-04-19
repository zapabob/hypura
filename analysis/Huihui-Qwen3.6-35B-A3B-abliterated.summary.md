# Huihui-Qwen3.6-35B-A3B-abliterated

## Model
- GGUF: `Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf`
- Architecture: `qwen35moe`
- Parameters: `34.7B`
- Quantization: `Q4K`
- Size: `19.70 GB`
- Layers: `40`
- MoE: `256 experts, 8 active per token`
- Context: `262144`

## Benchmark summary
- Hardware: `AMD Ryzen 5 4500 6-Core Processor / NVIDIA GeForce RTX 3060 / 31.9 GB RAM`
- Config: `context=1024`, `max_tokens=2`, prompt `Hello.`
- Baseline: `0.058528 +/- 0.050199 tok/s` (`n=2`)
- Hypura legacy-3tier + off: `0.020106 +/- 0.000687 tok/s` (`n=2`)
- Hypura four-tier + off: `0.037910 +/- 0.014356 tok/s` (`n=2`)
- Hypura four-tier + auto: `0.041226 +/- 0.033923 tok/s` (`n=2`)
- Best observed Hypura group: `hypura four-tier + auto`

## Notes
- `estimate` predicted a fast path, but the runtime logs reported `Sparse MoE mmap: model (19.7 GB) exceeds GPU budget (10.0 GB), using CPU-only (ngl=0)`.
- Both runs stayed on the `SparseMoeMmap` path and did not use host pinned memory or NVMe transfers.
- In the current corpus, baseline remained faster than every Hypura group for this model on this machine.
- Raw inspect output is in `analysis/Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.inspect.txt`.

