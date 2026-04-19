from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.gen_charts import load_runs_from_result, summarize_runs


class GenChartsTests(unittest.TestCase):
    def _write_payload(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, tmp)
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def test_load_runs_from_current_schema_includes_baseline(self) -> None:
        payload = {
            "timestamp": "2026-04-19T00:00:00Z",
            "model": {"name": "demo", "architecture": "llama", "quant": "Q4", "size_gb": 1.5},
            "hardware": {"cpu": "CPU", "gpu": "GPU", "ram_gb": 8.0},
            "config": {"context": 2048},
            "baseline": {
                "tok_per_sec": 2.0,
                "prompt_eval_ms": 100.0,
                "wall_time_ms": 500.0,
                "prompt_tokens": 4,
                "tokens_generated": 8,
            },
            "hypura_runs": [
                {
                    "label": "hypura four-tier + auto",
                    "placement": {
                        "gpu_gb": 1.0,
                        "host_pageable_gb": 0.2,
                        "host_pinned_gb": 0.0,
                        "nvme_gb": 0.3,
                        "inference_mode": "FullStreaming",
                    },
                    "result": {
                        "tok_per_sec": 3.0,
                        "prompt_eval_ms": 50.0,
                        "wall_time_ms": 200.0,
                        "prompt_tokens": 4,
                        "tokens_generated": 8,
                    },
                }
            ],
            "primary_run_label": "hypura four-tier + auto",
        }
        path = self._write_payload(payload)
        runs = load_runs_from_result(path)
        self.assertEqual(len(runs), 2)
        self.assertEqual({run.label for run in runs}, {"baseline", "hypura four-tier + auto"})
        primary = next(run for run in runs if run.label == "hypura four-tier + auto")
        self.assertTrue(primary.is_primary)

    def test_summarize_runs_computes_mean_and_sd(self) -> None:
        payload = {
            "timestamp": "2026-04-19T00:00:00Z",
            "model": {"name": "demo", "architecture": "llama", "quant": "Q4", "size_gb": 1.5},
            "hardware": {"cpu": "CPU", "gpu": "GPU", "ram_gb": 8.0},
            "config": {"context": 2048},
            "baseline": None,
            "hypura_runs": [
                {
                    "label": "hypura four-tier + auto",
                    "placement": {
                        "gpu_gb": 1.0,
                        "host_pageable_gb": 0.0,
                        "host_pinned_gb": 0.0,
                        "nvme_gb": 0.0,
                        "inference_mode": "FullResident",
                    },
                    "result": {
                        "tok_per_sec": 3.0,
                        "prompt_eval_ms": 50.0,
                        "wall_time_ms": 200.0,
                        "prompt_tokens": 4,
                        "tokens_generated": 8,
                    },
                }
            ],
            "primary_run_label": "hypura four-tier + auto",
        }
        path1 = self._write_payload(payload)
        payload["hypura_runs"][0]["result"]["tok_per_sec"] = 5.0
        payload["hypura_runs"][0]["result"]["wall_time_ms"] = 300.0
        path2 = self._write_payload(payload)
        runs = load_runs_from_result(path1) + load_runs_from_result(path2)
        stats = summarize_runs(runs)
        key = ("CPU / GPU / 8.0 GB RAM", "demo", "hypura four-tier + auto")
        stat = stats[key]
        self.assertEqual(stat.n, 2)
        self.assertAlmostEqual(stat.mean_tok_per_sec, 4.0)
        self.assertAlmostEqual(stat.sd_tok_per_sec, 1.41421356237, places=6)


if __name__ == "__main__":
    unittest.main()
