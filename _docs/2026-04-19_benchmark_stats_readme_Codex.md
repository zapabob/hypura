# Overview

Updated benchmark aggregation so the repository can compute mean +/- SD, error-bar charts, and multi-group comparison tables from the current `hypura bench` JSON schema. Refreshed the top-level README benchmark section to use measured corpus data instead of stale snapshot prose.

# Background / requirements

- The repository had fresh Windows CUDA benchmark results, including `supergemma4-Q8_0`, but the chart generator still expected an older JSON schema.
- The user asked to start from the new result, measure summary statistics including mean +/- SD, generate error-bar graphs, and make the README use those benchmark scores.

# Assumptions / decisions

- Treated `benchmarks/results/*.json` as the benchmark corpus of record.
- Preserved compatibility with legacy single-result schema while making current `hypura_runs` the primary path.
- Reported single-run groups with `SD = 0.000` and documented that they are exploratory rather than stable replicated estimates.
- Used ASCII `+/-` and `N/A` in generated markdown to avoid Windows console encoding issues.

# Changed files

- `benchmarks/gen_charts.py`
- `benchmarks/test_gen_charts.py`
- `benchmarks/gen_charts.sh`
- `benchmarks/CHARTS.md`
- `benchmarks/charts/generation_mean_sd.png`
- `benchmarks/charts/primary_benchmark_score.png`
- `README.md`

# Implementation details

- Added a dedicated Python aggregation module for benchmark result loading, summary-stat computation, markdown rendering, and chart generation.
- Added regression tests that verify current-schema loading and mean/SD aggregation.
- Replaced the old shell-embedded chart logic with a thin wrapper that calls the Python module.
- Generated two new charts:
  - grouped model/profile throughput with error bars
  - primary benchmark score with error bars
- Rewrote the README benchmark section to report:
  - measured hardware corpus
  - per-model benchmark score
  - multi-group comparison table
  - explicit notes on sample counts and the exploratory nature of the single-run `supergemma4-Q8_0` datapoint

# Commands run

```powershell
python -m unittest benchmarks.test_gen_charts -v
python benchmarks\gen_charts.py --charts-dir benchmarks\charts --output-md benchmarks\CHARTS.md benchmarks\results\2026-04-17T15-51-46_Shadows-MoE-Q6.json benchmarks\results\2026-04-17T16-01-04_Shadows-MoE-Q6.json benchmarks\results\2026-04-19T08-47-24_supergemma4-Q8_0.json
python -c "from benchmarks.gen_charts import load_all_runs, summarize_runs; from pathlib import Path; files=sorted(Path('benchmarks/results').glob('*.json')); runs=load_all_runs(files); stats=summarize_runs(runs); [print(f'{k}|n={s.n}|mean={s.mean_tok_per_sec:.6f}|sd={s.sd_tok_per_sec:.6f}') for k,s in sorted(stats.items())]"
```

# Test / verification results

- `python -m unittest benchmarks.test_gen_charts -v`: passed
- `python benchmarks\gen_charts.py ...`: passed and regenerated `benchmarks/CHARTS.md`
- Verified computed summary statistics for the current corpus:
  - `Shadows-MoE-Q6 baseline`: `1.121 +/- 0.023`, `n=2`
  - `Shadows-MoE-Q6 four-tier + off`: `1.158 +/- 0.111`, `n=2`
  - `supergemma4-Q8_0 legacy-3tier + off`: `29.851 +/- 0.000`, `n=1`

# Residual risks

- The current benchmark corpus is small and hardware-specific.
- `supergemma4-Q8_0` still has only one observed run, so its SD is not a stability estimate.
- Existing older chart PNGs remain in `benchmarks/charts/`; they are no longer referenced from `benchmarks/CHARTS.md` but still exist in the directory.

# Recommended next actions

- Re-run `supergemma4-Q8_0` multiple times under the same settings so the README can report a replicated SD.
- Add more benchmark corpora from Apple Silicon and larger spill-heavy GGUFs so the README score table covers both GPU-resident and NVMe-tier workloads.
- If older chart artifacts are no longer needed, remove them in a follow-up cleanup change.
