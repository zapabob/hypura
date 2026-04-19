# Overview

Tested the `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` GGUF pair on the Windows CUDA 12.8 `hypura.exe` benchmark path, added repeated benchmark results for the text model, refreshed the aggregate benchmark statistics, and updated the top-level README benchmark score table.

# Background / requirements

- The requested artifacts were:
  - `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
  - `mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf`
- The user wanted the pair tested and incorporated into the repository's multi-group comparison and summary-statistics workflow.

# Assumptions / decisions

- Used the existing Windows CUDA 12.8 build at `F:\hypura-cuda128-latest\debug\hypura.exe`.
- Benchmarked the text GGUF directly with repeated runs and baseline comparison.
- Treated the `mmproj` GGUF as inspect-only for this slice because the current CLI build does not expose a direct projector argument for `bench`.
- Used `n=2` repeated runs for the text model so mean +/- SD can be computed.

# Changed files

- `benchmarks/results/2026-04-19T09-42-49_Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.json`
- `benchmarks/results/2026-04-19T09-44-26_Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.json`
- `analysis/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.inspect.txt`
- `analysis/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.inspect.txt`
- `analysis/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-combined.summary.json`
- `analysis/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-combined.summary.md`
- `benchmarks/CHARTS.md`
- `README.md`

# Implementation details

- Verified both files exist and inspected them:
  - text model: `gemma4`, `7.5B`, `Q4K`, `4.95 GB`, `720` tensors
  - projector: `clip`, `478.1M`, `F16`, `0.92 GB`, `483` tensors
- Ran two repeated `bench --baseline` passes on the text model with:
  - `--turboquant-mode exact`
  - `--context 2048`
  - `--max-tokens 8`
  - prompt: `Hello from Hypura benchmark.`
- Refreshed benchmark corpus aggregation and regenerated summary markdown/charts.
- Updated README benchmark score to include the new model's replicated score and variance note.

# Commands run

```powershell
& 'F:\hypura-cuda128-latest\debug\hypura.exe' inspect 'C:\Users\downl\Desktop\SO8T\gguf_models\stronman\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
& 'F:\hypura-cuda128-latest\debug\hypura.exe' inspect 'C:\Users\downl\Desktop\SO8T\gguf_models\stronman\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf'
& 'F:\hypura-cuda128-latest\debug\hypura.exe' estimate 'C:\Users\downl\Desktop\SO8T\gguf_models\stronman\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
& 'F:\hypura-cuda128-latest\debug\hypura.exe' bench --baseline --turboquant-mode exact --max-tokens 8 --context 2048 --prompt 'Hello from Hypura benchmark.' 'C:\Users\downl\Desktop\SO8T\gguf_models\stronman\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf'
python benchmarks\gen_charts.py --charts-dir benchmarks\charts --output-md benchmarks\CHARTS.md benchmarks\results\2026-04-17T15-51-46_Shadows-MoE-Q6.json benchmarks\results\2026-04-17T16-01-04_Shadows-MoE-Q6.json benchmarks\results\2026-04-19T08-47-24_supergemma4-Q8_0.json benchmarks\results\2026-04-19T09-42-49_Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.json benchmarks\results\2026-04-19T09-44-26_Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.json
```

# Test / verification results

- Text model inspect: passed
- Projector inspect: passed
- Text model estimate: passed, predicted `59.3 tok/s`, GPU-resident plan
- Benchmark run 1:
  - baseline `38.600 tok/s`
  - legacy-3tier + off `53.204 tok/s`
  - four-tier + off `53.749 tok/s`
  - four-tier + auto `50.213 tok/s`
- Benchmark run 2:
  - baseline `44.762 tok/s`
  - legacy-3tier + off `47.576 tok/s`
  - four-tier + off `5.065 tok/s`
  - four-tier + auto `53.456 tok/s`
- Aggregated stats now report:
  - baseline `41.681 +/- 4.357 tok/s`
  - legacy-3tier + off `50.390 +/- 3.980 tok/s`
  - four-tier + off `29.407 +/- 34.425 tok/s`
  - four-tier + auto `51.835 +/- 2.293 tok/s`

# Residual risks

- The current CLI build still does not expose a direct `mmproj` input path for `bench`, so the projector was validated structurally rather than end-to-end in a multimodal inference run.
- `four-tier + off` showed extremely high variance for this model, so additional repeats would be useful before treating that group as stable.
- The aggregate chart wrapper in Git Bash previously required explicit Python selection; direct generator invocation was used for the authoritative refresh.

# Recommended next actions

- Add one or more additional repeats for `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M` to tighten the SD, especially for `four-tier + off`.
- If multimodal end-to-end testing is needed, add a CLI or scripted path that accepts `mmproj` and a sample image payload.
- Fold projector-aware multimodal smoke tests into the benchmark or analysis flow once the CLI surface exists.
