# Overview

Tested `Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf`, captured two repeated benchmark runs with baseline comparison, and updated the repository benchmark score summaries to include the new model.

# Background / requirements

- The user asked to test `C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf` in the same style as the earlier models.
- The benchmark corpus and README score table therefore needed a repeated-run mean +/- SD entry and multi-group comparison row.

# Assumptions / decisions

- Reused the Windows CUDA 12.8 benchmark binary at `F:\hypura-cuda128-latest\debug\hypura.exe`.
- Reduced the benchmark config to `context=1024`, `max_tokens=2`, prompt `Hello.` because the full-size run was too slow and had already been interrupted once.
- Accepted the reduced benchmark config as the practical way to obtain a replicated `n=2` datapoint for this model on current hardware.

# Changed files

- `benchmarks/results/2026-04-19T10-21-25_Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.json`
- `benchmarks/results/2026-04-19T10-31-54_Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.json`
- `analysis/Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.inspect.txt`
- `analysis/Huihui-Qwen3.6-35B-A3B-abliterated.summary.md`
- `analysis/Huihui-Qwen3.6-35B-A3B-abliterated.summary.json`
- `benchmarks/CHARTS.md`
- `README.md`

# Implementation details

- Verified the GGUF exists and inspected it:
  - architecture `qwen35moe`
  - `34.7B` params
  - `Q4K`
  - `19.7 GB`
  - `256` experts with `8` active
- Saved raw tensor inspection output to `analysis/`.
- Ran `estimate`, which predicted a fast path, but the actual runtime logs showed the sparse MoE mmap path falling back to CPU-only because the model exceeded the effective GPU budget.
- Ran two repeated benchmark passes with baseline enabled.
- Updated the benchmark score summary in README to include the new model and explain the CPU-only fallback behavior.

# Commands run

```powershell
& 'F:\hypura-cuda128-latest\debug\hypura.exe' inspect 'C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf'
& 'F:\hypura-cuda128-latest\debug\hypura.exe' estimate 'C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf'
& 'F:\hypura-cuda128-latest\debug\hypura.exe' bench --baseline --turboquant-mode exact --max-tokens 2 --context 1024 --prompt 'Hello.' 'C:\Users\downl\Desktop\SO8T\gguf_models\mradermacher\Huihui-Qwen3.6-35B-A3B-abliterated-GGUF\Huihui-Qwen3.6-35B-A3B-abliterated.Q4_K_M.gguf'
```

# Test / verification results

- `inspect`: passed
- `estimate`: passed
- repeated benchmarks: both passed and wrote JSON results
- aggregated statistics:
  - baseline `0.058528 +/- 0.050199 tok/s`
  - legacy-3tier + off `0.020106 +/- 0.000687 tok/s`
  - four-tier + off `0.037910 +/- 0.014356 tok/s`
  - four-tier + auto `0.041226 +/- 0.033923 tok/s`

# Residual risks

- This model is extremely slow on the current machine; even the shortened benchmark config took around 10 to 12 minutes per run.
- `estimate` and runtime behavior diverged materially because the runtime switched to CPU-only sparse MoE mmap; future estimator tuning may be needed.
- The current datapoint is valid for this smaller config, but it is not directly comparable to the earlier `max_tokens=8` Gemma-family measurements without that caveat.

# Recommended next actions

- If this model matters operationally, test it on a machine with substantially more usable GPU memory.
- Revisit the estimator so the predicted placement/performance better reflects the CPU-only fallback seen at runtime.
- If desired, add a separate README note or chart grouping for "reduced-config heavy-model smoke benchmarks" so these long-running models are clearly distinguished.
