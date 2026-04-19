# Gemma-4-E4B-Uncensored-HauhauCS-Aggressive

## Text model
- GGUF: `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
- Architecture: `gemma4`
- Size: `4.95 GB`
- Quantization: `Q4K`
- Parameters: `7.5B`
- Context: `131072`

## mmproj
- GGUF: `mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf`
- Architecture: `clip`
- Size: `0.92 GB`
- Quantization: `F16`
- Parameters: `478.1M`
- Tensors: `483`
- Status: inspect verified; current Hypura CLI build does not accept `mmproj` as a direct `bench` input

## Benchmark summary
- Hardware: `AMD Ryzen 5 4500 6-Core Processor / NVIDIA GeForce RTX 3060 / 31.9 GB RAM`
- Baseline: `41.681 +/- 4.357 tok/s` (`n=2`)
- Hypura legacy-3tier + off: `50.390 +/- 3.980 tok/s` (`n=2`)
- Hypura four-tier + off: `29.407 +/- 34.425 tok/s` (`n=2`)
- Hypura four-tier + auto: `51.835 +/- 2.293 tok/s` (`n=2`)
- Best observed Hypura group: `hypura four-tier + auto`

## Notes
- This model is fully GPU-resident on the RTX 3060 in these runs; no host or NVMe spill was used.
- `four-tier + off` showed very high variance because one run collapsed to `5.065 tok/s` while the other reached `53.749 tok/s`.
- Raw inspect outputs live beside this summary in `analysis/`.
