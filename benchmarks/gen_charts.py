from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


LABEL_ORDER = [
    "baseline",
    "hypura legacy-3tier + off",
    "hypura four-tier + off",
    "hypura four-tier + auto",
]

LABEL_COLORS = {
    "baseline": "#8b949e",
    "hypura legacy-3tier + off": "#58a6ff",
    "hypura four-tier + off": "#f2cc60",
    "hypura four-tier + auto": "#3fb950",
}

BG = "#0d1117"
FG = "#c9d1d9"
GRID = "#21262d"
BORDER = "#30363d"


@dataclass(frozen=True)
class BenchmarkRun:
    source_file: str
    timestamp: str
    hardware_key: str
    hardware_display: str
    model_name: str
    architecture: str
    quant: str
    model_size_gb: float
    context: int
    label: str
    tok_per_sec: float
    prompt_eval_ms: float
    wall_time_ms: float
    prompt_tokens: int
    tokens_generated: int
    gpu_gb: float
    host_pageable_gb: float
    host_pinned_gb: float
    nvme_gb: float
    inference_mode: str
    is_primary: bool


@dataclass(frozen=True)
class SummaryStat:
    label: str
    n: int
    mean_tok_per_sec: float
    sd_tok_per_sec: float
    min_tok_per_sec: float
    max_tok_per_sec: float
    mean_prompt_eval_ms: float
    sd_prompt_eval_ms: float
    mean_wall_time_ms: float
    sd_wall_time_ms: float


def _sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _hardware_display(data: dict[str, Any]) -> str:
    cpu = str(data["hardware"]["cpu"]).strip()
    gpu = str(data["hardware"].get("gpu") or "unknown").strip()
    ram_gb = float(data["hardware"]["ram_gb"])
    return f"{cpu} / {gpu} / {ram_gb:.1f} GB RAM"


def _hardware_key(data: dict[str, Any]) -> str:
    cpu = str(data["hardware"]["cpu"]).strip()
    gpu = str(data["hardware"].get("gpu") or "unknown").strip()
    ram_gb = float(data["hardware"]["ram_gb"])
    return f"{cpu}|{gpu}|{ram_gb:.3f}"


def _normalize_legacy_rows(data: dict[str, Any], source_file: str) -> list[BenchmarkRun]:
    hypura = data.get("hypura")
    if not hypura:
        return []
    placement = data.get("placement", {})
    result = data.get("result", {})
    return [
        BenchmarkRun(
            source_file=source_file,
            timestamp=str(data.get("timestamp", "")),
            hardware_key=_hardware_key(data),
            hardware_display=_hardware_display(data),
            model_name=str(data["model"]["name"]),
            architecture=str(data["model"].get("architecture", "unknown")),
            quant=str(data["model"].get("quant", "unknown")),
            model_size_gb=float(data["model"].get("size_gb", 0.0)),
            context=int(data.get("context", 0)),
            label="hypura",
            tok_per_sec=float(hypura.get("tok_per_sec", 0.0)),
            prompt_eval_ms=float(result.get("prompt_eval_ms", 0.0)),
            wall_time_ms=float(result.get("wall_time_ms", 0.0)),
            prompt_tokens=int(result.get("prompt_tokens", 0)),
            tokens_generated=int(result.get("tokens_generated", 0)),
            gpu_gb=float(placement.get("gpu_gb", 0.0)),
            host_pageable_gb=float(placement.get("ram_gb", 0.0)),
            host_pinned_gb=0.0,
            nvme_gb=float(placement.get("nvme_gb", 0.0)),
            inference_mode=str(placement.get("inference_mode", "unknown")),
            is_primary=True,
        )
    ]


def load_runs_from_result(path: str | Path) -> list[BenchmarkRun]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source_file = str(Path(path))
    if payload.get("hypura_runs"):
        rows: list[BenchmarkRun] = []
        primary_label = payload.get("primary_run_label")
        for run in payload["hypura_runs"]:
            placement = run.get("placement", {})
            result = run.get("result", {})
            rows.append(
                BenchmarkRun(
                    source_file=source_file,
                    timestamp=str(payload.get("timestamp", "")),
                    hardware_key=_hardware_key(payload),
                    hardware_display=_hardware_display(payload),
                    model_name=str(payload["model"]["name"]),
                    architecture=str(payload["model"].get("architecture", "unknown")),
                    quant=str(payload["model"].get("quant", "unknown")),
                    model_size_gb=float(payload["model"].get("size_gb", 0.0)),
                    context=int(payload.get("config", {}).get("context", 0)),
                    label=str(run["label"]),
                    tok_per_sec=float(result.get("tok_per_sec", 0.0)),
                    prompt_eval_ms=float(result.get("prompt_eval_ms", 0.0)),
                    wall_time_ms=float(result.get("wall_time_ms", 0.0)),
                    prompt_tokens=int(result.get("prompt_tokens", 0)),
                    tokens_generated=int(result.get("tokens_generated", 0)),
                    gpu_gb=float(placement.get("gpu_gb", 0.0)),
                    host_pageable_gb=float(placement.get("host_pageable_gb", 0.0)),
                    host_pinned_gb=float(placement.get("host_pinned_gb", 0.0)),
                    nvme_gb=float(placement.get("nvme_gb", 0.0)),
                    inference_mode=str(placement.get("inference_mode", "unknown")),
                    is_primary=str(run["label"]) == str(primary_label),
                )
            )
        baseline = payload.get("baseline")
        if baseline and float(baseline.get("tok_per_sec", 0.0)) > 0.0:
            rows.append(
                BenchmarkRun(
                    source_file=source_file,
                    timestamp=str(payload.get("timestamp", "")),
                    hardware_key=_hardware_key(payload),
                    hardware_display=_hardware_display(payload),
                    model_name=str(payload["model"]["name"]),
                    architecture=str(payload["model"].get("architecture", "unknown")),
                    quant=str(payload["model"].get("quant", "unknown")),
                    model_size_gb=float(payload["model"].get("size_gb", 0.0)),
                    context=int(payload.get("config", {}).get("context", 0)),
                    label="baseline",
                    tok_per_sec=float(baseline.get("tok_per_sec", 0.0)),
                    prompt_eval_ms=float(baseline.get("prompt_eval_ms", 0.0)),
                    wall_time_ms=float(baseline.get("wall_time_ms", 0.0)),
                    prompt_tokens=int(baseline.get("prompt_tokens", 0)),
                    tokens_generated=int(baseline.get("tokens_generated", 0)),
                    gpu_gb=0.0,
                    host_pageable_gb=0.0,
                    host_pinned_gb=0.0,
                    nvme_gb=0.0,
                    inference_mode="baseline",
                    is_primary=False,
                )
            )
        return rows
    return _normalize_legacy_rows(payload, source_file)


def load_all_runs(files: list[str | Path]) -> list[BenchmarkRun]:
    rows: list[BenchmarkRun] = []
    for file in files:
        try:
            rows.extend(load_runs_from_result(file))
        except Exception:
            continue
    return rows


def summarize_runs(runs: list[BenchmarkRun]) -> dict[tuple[str, str, str], SummaryStat]:
    grouped: dict[tuple[str, str, str], list[BenchmarkRun]] = defaultdict(list)
    for run in runs:
        grouped[(run.hardware_display, run.model_name, run.label)].append(run)

    stats: dict[tuple[str, str, str], SummaryStat] = {}
    for key, items in grouped.items():
        speeds = [item.tok_per_sec for item in items]
        prompts = [item.prompt_eval_ms for item in items]
        walls = [item.wall_time_ms for item in items]
        stats[key] = SummaryStat(
            label=key[2],
            n=len(items),
            mean_tok_per_sec=mean(speeds),
            sd_tok_per_sec=_sample_sd(speeds),
            min_tok_per_sec=min(speeds),
            max_tok_per_sec=max(speeds),
            mean_prompt_eval_ms=mean(prompts),
            sd_prompt_eval_ms=_sample_sd(prompts),
            mean_wall_time_ms=mean(walls),
            sd_wall_time_ms=_sample_sd(walls),
        )
    return stats


def _ordered_labels(labels: set[str]) -> list[str]:
    ordered = [label for label in LABEL_ORDER if label in labels]
    ordered.extend(sorted(label for label in labels if label not in LABEL_ORDER))
    return ordered


def _format_mean_sd(mu: float, sigma: float) -> str:
    return f"{mu:.3f} +/- {sigma:.3f}"


def build_markdown(
    runs: list[BenchmarkRun],
    stats: dict[tuple[str, str, str], SummaryStat],
    chart_files: dict[str, str | None],
) -> str:
    hardware_order = sorted({run.hardware_display for run in runs})
    lines = [
        "# Hypura Benchmark Charts",
        "",
        "*Auto-generated by `benchmarks/gen_charts.sh`*",
        "",
    ]

    if chart_files.get("grouped"):
        lines.extend(
            [
                "## Generation Throughput With Error Bars",
                "",
                "Mean generation throughput per model/profile. Error bars show sample SD across repeated runs for the same hardware and model.",
                "",
                f"![Generation Mean and SD]({chart_files['grouped']})",
                "",
            ]
        )

    if chart_files.get("primary"):
        lines.extend(
            [
                "## Primary Benchmark Score",
                "",
                "Primary score uses each result file's `primary_run_label` when available and reports mean +/- SD across repeated runs.",
                "",
                f"![Primary Benchmark Score]({chart_files['primary']})",
                "",
            ]
        )

    for hardware in hardware_order:
        hardware_runs = [run for run in runs if run.hardware_display == hardware]
        model_order = sorted({run.model_name for run in hardware_runs})
        lines.extend([f"## {hardware}", ""])
        lines.extend(
            [
                "| Model | Group | n | Mean +/- SD (tok/s) | Min | Max | Mean Prompt Eval (ms) | Mean Wall (ms) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_name in model_order:
            labels = _ordered_labels({run.label for run in hardware_runs if run.model_name == model_name})
            for label in labels:
                stat = stats[(hardware, model_name, label)]
                lines.append(
                    f"| {model_name} | {label} | {stat.n} | {_format_mean_sd(stat.mean_tok_per_sec, stat.sd_tok_per_sec)} | "
                    f"{stat.min_tok_per_sec:.3f} | {stat.max_tok_per_sec:.3f} | {stat.mean_prompt_eval_ms:.1f} | {stat.mean_wall_time_ms:.1f} |"
                )
        lines.append("")

        comparison_labels = [label for label in LABEL_ORDER if label in {run.label for run in hardware_runs}]
        if comparison_labels:
            lines.extend(
                [
                    "### Multi-group Comparison",
                    "",
                    "| Model | " + " | ".join(comparison_labels) + " |",
                    "| --- | " + " | ".join(["---:" for _ in comparison_labels]) + " |",
                ]
            )
            for model_name in model_order:
                values = []
                for label in comparison_labels:
                    stat = stats.get((hardware, model_name, label))
                    values.append(_format_mean_sd(stat.mean_tok_per_sec, stat.sd_tok_per_sec) if stat else "N/A")
                lines.append(f"| {model_name} | " + " | ".join(values) + " |")
            lines.append("")

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `Mean +/- SD` is computed per hardware, model, and run label from the benchmark JSON files currently present in `benchmarks/results/`.",
            "- Single-run groups report `SD = 0.000`; treat those as exploratory datapoints rather than stable estimates.",
            "- `baseline` is the non-Hypura comparator from `hypura bench --baseline` when that result was captured.",
            "",
        ]
    )
    return "\n".join(lines)


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": FG,
            "text.color": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "grid.color": GRID,
            "font.family": "monospace",
            "font.size": 11,
        }
    )


def generate_grouped_error_bar_chart(
    runs: list[BenchmarkRun],
    stats: dict[tuple[str, str, str], SummaryStat],
    output_path: str | Path,
) -> str | None:
    if not runs:
        return None
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    selected_hardware = sorted({run.hardware_display for run in runs})[0]
    filtered_runs = [run for run in runs if run.hardware_display == selected_hardware]
    models = sorted({run.model_name for run in filtered_runs})
    labels = _ordered_labels({run.label for run in filtered_runs})
    x = np.arange(len(models))
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2.2), 6))
    for idx, label in enumerate(labels):
        means = []
        sds = []
        for model in models:
            stat = stats.get((selected_hardware, model, label))
            means.append(stat.mean_tok_per_sec if stat else math.nan)
            sds.append(stat.sd_tok_per_sec if stat else 0.0)
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=sds,
            capsize=5,
            color=LABEL_COLORS.get(label, "#58a6ff"),
            edgecolor=BORDER,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right")
    ax.set_ylabel("Generation throughput (tok/s)")
    ax.set_title(f"Mean ± SD by Model and Profile\n{selected_hardware}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output.name


def generate_primary_score_chart(
    runs: list[BenchmarkRun],
    stats: dict[tuple[str, str, str], SummaryStat],
    output_path: str | Path,
) -> str | None:
    primary_runs = [run for run in runs if run.is_primary]
    if not primary_runs:
        return None
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    selected_hardware = sorted({run.hardware_display for run in primary_runs})[0]
    primary_runs = [run for run in primary_runs if run.hardware_display == selected_hardware]
    models = sorted({run.model_name for run in primary_runs})
    means = []
    sds = []
    for model in models:
        label = next(run.label for run in primary_runs if run.model_name == model)
        stat = stats[(selected_hardware, model, label)]
        means.append(stat.mean_tok_per_sec)
        sds.append(stat.sd_tok_per_sec)

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(9, len(models) * 2.2), 5.5))
    ax.bar(x, means, yerr=sds, capsize=6, color="#3fb950", edgecolor=BORDER)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=10, ha="right")
    ax.set_ylabel("Primary benchmark score (tok/s)")
    ax.set_title(f"Primary Benchmark Score Mean ± SD\n{selected_hardware}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output = Path(output_path)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output.name


def render_outputs(files: list[str | Path], charts_dir: str | Path, output_md: str | Path) -> dict[str, Any]:
    runs = load_all_runs(files)
    if not runs:
        raise SystemExit("No valid benchmark results found.")

    stats = summarize_runs(runs)
    charts_path = Path(charts_dir)
    charts_path.mkdir(parents=True, exist_ok=True)
    grouped_name = generate_grouped_error_bar_chart(runs, stats, charts_path / "generation_mean_sd.png")
    primary_name = generate_primary_score_chart(runs, stats, charts_path / "primary_benchmark_score.png")
    chart_files = {
        "grouped": f"charts/{grouped_name}" if grouped_name else None,
        "primary": f"charts/{primary_name}" if primary_name else None,
    }
    markdown = build_markdown(runs, stats, chart_files)
    Path(output_md).write_text(markdown, encoding="utf-8")
    return {"runs": runs, "stats": stats, "charts": chart_files, "markdown": markdown}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark charts and markdown from Hypura JSON results.")
    parser.add_argument("--charts-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    render_outputs(args.files, args.charts_dir, args.output_md)


if __name__ == "__main__":
    main()
