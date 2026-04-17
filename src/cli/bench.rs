use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use hypura::compute::ffi::SamplingParams;
use hypura::compute::inference::*;
use hypura::model::turboquant_sidecar::{RotationPolicy, TurboQuantMode};
use hypura::scheduler::types::{HostPinnedPolicy, ResidencyPolicyConfig, ResidencyProfile};
use hypura::telemetry::metrics::{TelemetryEmitter, TelemetryEvent};
use indicatif::{ProgressBar, ProgressStyle};

use super::fmt_util::{cli_progress_enabled, format_bytes, format_params};

const DEFAULT_PROMPT: &str = "Write a short paragraph about artificial intelligence.";

pub fn run(
    model_path: &str,
    baseline: bool,
    context: u32,
    max_tokens: u32,
    prompt: Option<&str>,
    _force: bool,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    rotation_policy: RotationPolicy,
    rotation_seed: u32,
    residency_profile: Option<ResidencyProfile>,
    host_pinned: Option<HostPinnedPolicy>,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(
        model_path,
        baseline,
        context,
        max_tokens,
        prompt,
        turboquant_mode,
        turboquant_config,
        rotation_policy,
        rotation_seed,
        residency_profile,
        host_pinned,
    ))
}

async fn run_async(
    model_path: &str,
    run_baseline: bool,
    context: u32,
    max_tokens: u32,
    prompt: Option<&str>,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    rotation_policy: RotationPolicy,
    rotation_seed: u32,
    residency_profile: Option<ResidencyProfile>,
    host_pinned: Option<HostPinnedPolicy>,
) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    let prompt_text = prompt.unwrap_or(DEFAULT_PROMPT);
    let llama_bridge = LlamaTurboquantCliBridge {
        rotation_policy,
        llama_rotation_seed: rotation_seed,
        ..Default::default()
    };

    let residency_cases = benchmark_cases(residency_profile, host_pinned);
    let base_runtime = resolve_runtime_setup(
        path,
        context,
        turboquant_mode,
        turboquant_config.map(Path::new),
        llama_bridge.clone(),
        residency_cases[0],
    )?;

    let filename = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    println!();
    println!("Hypura Benchmark: {filename}");
    println!("{}", "笏".repeat(56));
    println!(
        "  Model: {} ({} params, {}, {})",
        base_runtime.metadata.architecture,
        format_params(base_runtime.metadata.parameter_count),
        base_runtime
            .metadata
            .quantization
            .as_deref()
            .unwrap_or("unknown"),
        format_bytes(base_runtime.gguf.total_tensor_bytes()),
    );
    println!(
        "  Hardware: {}, {} RAM, GPU={}, NVMe={:.1} GB/s",
        base_runtime.hardware.cpu.model_name,
        format_bytes(base_runtime.hardware.memory.total_bytes),
        base_runtime
            .hardware
            .gpu
            .as_ref()
            .map(|g| g.name.as_str())
            .unwrap_or("none"),
        base_runtime
            .hardware
            .storage
            .first()
            .map(|s| s.sequential_read.peak_sequential as f64 / 1e9)
            .unwrap_or(0.0),
    );
    println!(
        "  Matrix: {}",
        residency_cases
            .iter()
            .map(|case| format!(
                "{}+{}",
                case.residency_profile.label(),
                case.host_pinned_policy.label()
            ))
            .collect::<Vec<_>>()
            .join(", "),
    );
    println!(
        "  TurboQuant: mode={}, schema={}, config={}, runtime_status={}",
        base_runtime.turboquant.mode,
        base_runtime.turboquant.schema_label(),
        base_runtime.turboquant.source_label(),
        turboquant_runtime_status(
            base_runtime.turboquant.mode,
            base_runtime.turboquant.config.is_some()
        ),
    );
    println!(
        "  Config: context={}, max_tokens={}, baseline={}",
        context,
        max_tokens,
        if run_baseline { "on" } else { "off" },
    );
    println!();

    let model_total_bytes = base_runtime.gguf.total_tensor_bytes();
    let total_ram = base_runtime.hardware.memory.total_bytes;
    let min_headroom: u64 = 4 * (1 << 30);
    let baseline_safe = model_total_bytes <= total_ram.saturating_sub(min_headroom);

    let config = InferenceConfig {
        n_ctx: context,
        n_batch: 512,
        n_threads: InferenceConfig::default().n_threads,
        sampling: SamplingParams {
            max_tokens,
            seed: 42,
            ..SamplingParams::default()
        },
    };

    let baseline_result = if run_baseline {
        run_baseline_case(path, prompt_text, &config, &base_runtime, baseline_safe).await?
    } else {
        None
    };

    let mut hypura_runs = Vec::new();
    for (idx, residency_policy) in residency_cases.iter().copied().enumerate() {
        let runtime = resolve_runtime_setup(
            path,
            context,
            turboquant_mode,
            turboquant_config.map(Path::new),
            llama_bridge.clone(),
            residency_policy,
        )?;

        println!(
            "  Run {}: Hypura {} + {}",
            idx + 1,
            residency_policy.residency_profile.label(),
            residency_policy.host_pinned_policy.label(),
        );
        println!(
            "    Placement: {} GPU | {} host pageable | {} host pinned | {} NVMe",
            format_bytes(runtime.placement_summary.total_gpu_bytes),
            format_bytes(runtime.placement_summary.total_host_pageable_bytes),
            format_bytes(runtime.placement_summary.total_host_pinned_bytes),
            format_bytes(runtime.placement_summary.total_nvme_bytes),
        );
        println!(
            "    Residency: mode={:?}, pinned_tier={}, pinned_policy={}",
            runtime.plan.inference_mode,
            if runtime.placement_summary.host_pinned_active {
                "active"
            } else {
                "collapsed"
            },
            runtime.plan.residency_policy.host_pinned_policy.label(),
        );

        let run = run_hypura_case(path, prompt_text, &config, &runtime).await?;
        print_run_result(&run.label, &run.result, run.result.wall_time_ms);
        println!(
            "    Telemetry: gpu_slot_hit_rate={:.3}, pinned_slot_hit_rate={:.3}, pageable_fallback_rate={:.3}, nvme_mbps={:.1}, h2d_pinned_mbps={:.1}, h2d_pageable_mbps={:.1}, eviction_churn_per_token={:.3}, first_token_stall_ms={:.1}",
            run.telemetry.gpu_slot_hit_rate,
            run.telemetry.pinned_slot_hit_rate,
            run.telemetry.pageable_fallback_rate,
            run.telemetry.nvme_mbps,
            run.telemetry.h2d_pinned_mbps,
            run.telemetry.h2d_pageable_mbps,
            run.telemetry.eviction_churn_per_token,
            run.telemetry.first_token_stall_ms,
        );
        println!();
        hypura_runs.push(run);
    }

    let primary_idx = primary_hypura_run_index(&hypura_runs).unwrap_or(0);
    let primary_run = hypura_runs
        .get(primary_idx)
        .ok_or_else(|| anyhow::anyhow!("no Hypura benchmark runs recorded"))?;
    let primary_run_label = primary_run.label.clone();
    let primary_speedup_vs_baseline = baseline_result.as_ref().and_then(|baseline| {
        if baseline.tok_per_sec > 0.0 {
            Some(primary_run.result.tok_per_sec / baseline.tok_per_sec)
        } else {
            None
        }
    });

    if let Some(base) = baseline_result.as_ref() {
        if base.tok_per_sec > 0.0 {
            let speedup = primary_run.result.tok_per_sec / base.tok_per_sec;
            println!(
                "  Primary speedup: {:.1}x ({} vs baseline)",
                speedup, primary_run.label
            );
            println!();
        }
    }

    let bench_result = BenchmarkResult {
        timestamp: chrono::Utc::now().to_rfc3339(),
        model: ModelInfo {
            name: filename.trim_end_matches(".gguf").to_string(),
            architecture: base_runtime.metadata.architecture.clone(),
            params: format_params(base_runtime.metadata.parameter_count),
            quant: base_runtime.metadata.quantization.clone().unwrap_or_default(),
            size_gb: base_runtime.gguf.total_tensor_bytes() as f64 / (1u64 << 30) as f64,
        },
        hardware: HardwareInfo {
            cpu: base_runtime.hardware.cpu.model_name.clone(),
            ram_gb: base_runtime.hardware.memory.total_bytes as f64 / (1u64 << 30) as f64,
            gpu: base_runtime
                .hardware
                .gpu
                .as_ref()
                .map(|g| g.name.clone())
                .unwrap_or_default(),
            nvme_seq_gbps: base_runtime
                .hardware
                .storage
                .first()
                .map(|s| s.sequential_read.peak_sequential as f64 / 1e9)
                .unwrap_or(0.0),
        },
        config: BenchConfig {
            context,
            max_tokens,
            prompt: prompt_text.to_string(),
            baseline_requested: run_baseline,
            residency_matrix: residency_cases
                .iter()
                .map(|case| {
                    format!(
                        "{}+{}",
                        case.residency_profile.label(),
                        case.host_pinned_policy.label()
                    )
                })
                .collect(),
            turboquant_mode: base_runtime.turboquant.mode.to_string(),
            turboquant_schema: base_runtime.turboquant.schema_label().to_string(),
            turboquant_runtime_status: turboquant_runtime_status(
                base_runtime.turboquant.mode,
                base_runtime.turboquant.config.is_some(),
            )
            .to_string(),
        },
        baseline: baseline_result,
        hypura_runs,
        primary_run_label,
        primary_speedup_vs_baseline,
    };

    save_benchmark_result(&bench_result)?;
    println!("  Results saved to benchmarks/results/");

    Ok(())
}

fn benchmark_cases(
    residency_profile: Option<ResidencyProfile>,
    host_pinned: Option<HostPinnedPolicy>,
) -> Vec<ResidencyPolicyConfig> {
    let mut cases = if residency_profile.is_some() || host_pinned.is_some() {
        vec![ResidencyPolicyConfig::new(
            residency_profile.unwrap_or(ResidencyProfile::FourTier),
            host_pinned.unwrap_or(HostPinnedPolicy::Auto),
        )
        .normalized()]
    } else {
        vec![
            ResidencyPolicyConfig::new(ResidencyProfile::Legacy3Tier, HostPinnedPolicy::Off),
            ResidencyPolicyConfig::new(ResidencyProfile::FourTier, HostPinnedPolicy::Off),
            ResidencyPolicyConfig::new(ResidencyProfile::FourTier, HostPinnedPolicy::Auto),
        ]
    };
    cases.dedup();
    cases
}

async fn run_baseline_case(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    runtime: &RuntimeSetup,
    baseline_safe: bool,
) -> anyhow::Result<Option<RunResult>> {
    let model_total_bytes = runtime.gguf.total_tensor_bytes();
    let total_ram = runtime.hardware.memory.total_bytes;
    let baseline_ngl = if baseline_safe { runtime.n_gpu_layers } else { 0 };

    let baseline_label = if baseline_safe {
        "llama.cpp (full GPU)".to_string()
    } else {
        let model_gb = model_total_bytes as f64 / (1u64 << 30) as f64;
        let total_gb = total_ram as f64 / (1u64 << 30) as f64;
        println!(
            "  Baseline: model ({model_gb:.1} GB) exceeds RAM ({total_gb:.1} GB), using CPU-only (ngl=0)."
        );
        "llama.cpp CPU-only (ngl=0)".to_string()
    };

    println!("  Baseline: {baseline_label}");
    let tok_target = config.sampling.max_tokens as u64;
    let pb = benchmark_progress_bar(tok_target, "baseline run");

    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel();
    let path_c = model_path.to_path_buf();
    let prompt_c = prompt.to_string();
    let config_c = config.clone();
    let telemetry = Arc::new(TelemetryEmitter::new(64));

    let wall_start = Instant::now();
    let handle = tokio::task::spawn_blocking(move || {
        generate_blocking(
            &path_c,
            &prompt_c,
            &config_c,
            baseline_ngl,
            token_tx,
            telemetry,
        )
    });

    let mut n_tok: u64 = 0;
    while token_rx.recv().await.is_some() {
        n_tok = n_tok.saturating_add(1);
        if cli_progress_enabled() {
            pb.set_position(n_tok.min(tok_target));
        }
    }

    match handle.await {
        Ok(Ok(result)) => {
            if cli_progress_enabled() {
                pb.finish_and_clear();
            }
            let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
            let run = RunResult {
                prompt_eval_ms: result.prompt_eval_ms,
                prompt_tokens: result.prompt_tokens,
                tok_per_sec: result.tok_per_sec_avg,
                tokens_generated: result.tokens_generated,
                wall_time_ms: wall_ms,
            };
            print_run_result(&baseline_label, &run, wall_ms);
            println!();
            Ok(Some(run))
        }
        Ok(Err(e)) => {
            if cli_progress_enabled() {
                pb.finish_and_clear();
            }
            println!("    Baseline failed: {e}");
            println!("    Continuing with Hypura runs...");
            println!();
            Ok(None)
        }
        Err(e) => {
            if cli_progress_enabled() {
                pb.finish_and_clear();
            }
            println!("    Baseline panicked: {e}");
            println!("    Continuing with Hypura runs...");
            println!();
            Ok(None)
        }
    }
}

async fn run_hypura_case(
    model_path: &Path,
    prompt: &str,
    config: &InferenceConfig,
    runtime: &RuntimeSetup,
) -> anyhow::Result<HypuraBenchRun> {
    let tok_target = config.sampling.max_tokens as u64;
    let pb = benchmark_progress_bar(tok_target, "Hypura run");
    let telemetry = Arc::new(TelemetryEmitter::new(128));
    let (collector_handle, stop_tx) = spawn_telemetry_collector(telemetry.clone());

    let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel();
    let path_c = model_path.to_path_buf();
    let prompt_c = prompt.to_string();
    let config_c = config.clone();
    let plan_c = Arc::new(runtime.plan.clone());
    let gguf_c = Arc::new(runtime.gguf.clone());
    let turboquant_c = Arc::new(runtime.turboquant.clone());
    let n_gpu_layers = runtime.n_gpu_layers;

    let wall_start = Instant::now();
    let handle = tokio::task::spawn_blocking(move || {
        generate_with_nvme_scheduling(
            &path_c,
            &prompt_c,
            &config_c,
            n_gpu_layers,
            &plan_c,
            &gguf_c,
            &turboquant_c,
            token_tx,
            telemetry,
        )
    });

    let mut n_tok: u64 = 0;
    while token_rx.recv().await.is_some() {
        n_tok = n_tok.saturating_add(1);
        if cli_progress_enabled() {
            pb.set_position(n_tok.min(tok_target));
        }
    }
    let result = handle.await??;
    if cli_progress_enabled() {
        pb.finish_and_clear();
    }
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let _ = stop_tx.send(());
    let telemetry = collector_handle
        .await
        .unwrap_or_else(|_| BenchTelemetry::default());

    Ok(HypuraBenchRun {
        label: format!(
            "hypura {} + {}",
            runtime.plan.residency_policy.residency_profile.label(),
            runtime.plan.residency_policy.host_pinned_policy.label()
        ),
        residency_profile: runtime.plan.residency_policy.residency_profile.label().to_string(),
        host_pinned_policy: runtime.plan.residency_policy.host_pinned_policy.label().to_string(),
        n_gpu_layers: runtime.n_gpu_layers,
        placement: PlacementInfo {
            gpu_gb: runtime.placement_summary.total_gpu_bytes as f64 / (1u64 << 30) as f64,
            host_pageable_gb: runtime.placement_summary.total_host_pageable_bytes as f64
                / (1u64 << 30) as f64,
            host_pinned_gb: runtime.placement_summary.total_host_pinned_bytes as f64
                / (1u64 << 30) as f64,
            nvme_gb: runtime.placement_summary.total_nvme_bytes as f64 / (1u64 << 30) as f64,
            inference_mode: format!("{:?}", runtime.plan.inference_mode),
            pinned_tier_active: runtime.placement_summary.host_pinned_active,
            pinned_budget_gb: runtime.placement_summary.host_pinned_budget_bytes as f64
                / (1u64 << 30) as f64,
        },
        result: RunResult {
            prompt_eval_ms: result.prompt_eval_ms,
            prompt_tokens: result.prompt_tokens,
            tok_per_sec: result.tok_per_sec_avg,
            tokens_generated: result.tokens_generated,
            wall_time_ms: wall_ms,
        },
        telemetry,
    })
}

fn benchmark_progress_bar(tokens: u64, message: &str) -> ProgressBar {
    if cli_progress_enabled() {
        let pb = ProgressBar::new(tokens.max(1));
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:28.cyan/blue}] {pos}/{len} tok ETA {eta_precise} {msg}",
                )
                .unwrap()
                .progress_chars("##-"),
        );
        pb.set_message(message.to_string());
        pb
    } else {
        ProgressBar::hidden()
    }
}

fn spawn_telemetry_collector(
    telemetry: Arc<TelemetryEmitter>,
) -> (tokio::task::JoinHandle<BenchTelemetry>, oneshot::Sender<()>) {
    let mut rx = telemetry.subscribe();
    let (stop_tx, mut stop_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let mut latest = BenchTelemetry::default();
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                event = rx.recv() => {
                    match event {
                        Ok(TelemetryEvent::PrefetchStatus {
                            hit_rate,
                            nvme_mbps,
                            gpu_slot_hit_rate,
                            pinned_slot_hit_rate,
                            pageable_fallback_rate,
                            h2d_pinned_mbps,
                            h2d_pageable_mbps,
                            eviction_churn_per_token,
                            first_token_stall_ms,
                        }) => {
                            latest = BenchTelemetry {
                                hit_rate,
                                nvme_mbps,
                                gpu_slot_hit_rate,
                                pinned_slot_hit_rate,
                                pageable_fallback_rate,
                                h2d_pinned_mbps,
                                h2d_pageable_mbps,
                                eviction_churn_per_token,
                                first_token_stall_ms,
                            };
                        }
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
            }
        }
        latest
    });
    (handle, stop_tx)
}

fn primary_hypura_run_index(runs: &[HypuraBenchRun]) -> Option<usize> {
    runs.iter()
        .position(|run| {
            run.residency_profile == ResidencyProfile::FourTier.label()
                && run.host_pinned_policy == HostPinnedPolicy::Auto.label()
        })
        .or_else(|| {
            runs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.result
                        .tok_per_sec
                        .partial_cmp(&b.result.tok_per_sec)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
        })
}

fn print_run_result(label: &str, result: &RunResult, wall_ms: f64) {
    let prompt_tps = if result.prompt_eval_ms > 0.0 {
        result.prompt_tokens as f64 / (result.prompt_eval_ms / 1000.0)
    } else {
        0.0
    };
    println!("    Label:          {label}");
    println!(
        "    Prompt eval:    {:.1}s ({:.1} tok/s)",
        result.prompt_eval_ms / 1000.0,
        prompt_tps
    );
    println!(
        "    Generation:     {:.1} tok/s ({} tokens)",
        result.tok_per_sec, result.tokens_generated
    );
    println!("    Wall time:      {:.1}s", wall_ms / 1000.0);
}

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    timestamp: String,
    model: ModelInfo,
    hardware: HardwareInfo,
    config: BenchConfig,
    baseline: Option<RunResult>,
    hypura_runs: Vec<HypuraBenchRun>,
    primary_run_label: String,
    primary_speedup_vs_baseline: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct ModelInfo {
    name: String,
    architecture: String,
    params: String,
    quant: String,
    size_gb: f64,
}

#[derive(Serialize, Deserialize)]
struct HardwareInfo {
    cpu: String,
    ram_gb: f64,
    gpu: String,
    nvme_seq_gbps: f64,
}

#[derive(Serialize, Deserialize)]
struct PlacementInfo {
    gpu_gb: f64,
    host_pageable_gb: f64,
    host_pinned_gb: f64,
    nvme_gb: f64,
    inference_mode: String,
    pinned_tier_active: bool,
    pinned_budget_gb: f64,
}

#[derive(Serialize, Deserialize)]
struct BenchConfig {
    context: u32,
    max_tokens: u32,
    prompt: String,
    baseline_requested: bool,
    residency_matrix: Vec<String>,
    turboquant_mode: String,
    turboquant_schema: String,
    turboquant_runtime_status: String,
}

#[derive(Serialize, Deserialize)]
struct HypuraBenchRun {
    label: String,
    residency_profile: String,
    host_pinned_policy: String,
    n_gpu_layers: i32,
    placement: PlacementInfo,
    result: RunResult,
    telemetry: BenchTelemetry,
}

#[derive(Default, Serialize, Deserialize)]
struct BenchTelemetry {
    hit_rate: f64,
    nvme_mbps: f64,
    gpu_slot_hit_rate: f64,
    pinned_slot_hit_rate: f64,
    pageable_fallback_rate: f64,
    h2d_pinned_mbps: f64,
    h2d_pageable_mbps: f64,
    eviction_churn_per_token: f64,
    first_token_stall_ms: f64,
}

#[derive(Clone, Serialize, Deserialize)]
struct RunResult {
    prompt_eval_ms: f64,
    prompt_tokens: u32,
    tok_per_sec: f64,
    tokens_generated: u32,
    wall_time_ms: f64,
}

fn save_benchmark_result(result: &BenchmarkResult) -> anyhow::Result<()> {
    let dir = Path::new("benchmarks/results");
    std::fs::create_dir_all(dir)?;

    let ts = result.timestamp.replace(':', "-").replace('.', "-");
    let filename = format!("{}_{}.json", &ts[..19], result.model.name);
    let json_path = dir.join(&filename);
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(&json_path, &json)?;

    update_readme(result)?;

    Ok(())
}

fn update_readme(result: &BenchmarkResult) -> anyhow::Result<()> {
    let readme_path = Path::new("benchmarks/README.md");

    let header = "# Hypura Benchmarks\n\n\
        | Date | Model | Hardware | GPU | RAM | NVMe | Baseline tok/s | Hypura tok/s | Speedup |\n\
        |------|-------|----------|-----|-----|------|----------------|--------------|---------|";

    let existing = if readme_path.exists() {
        std::fs::read_to_string(readme_path)?
    } else {
        format!("{}\n", header)
    };

    let baseline_str = result
        .baseline
        .as_ref()
        .map(|b| format!("{:.1}", b.tok_per_sec))
        .unwrap_or_else(|| "-".to_string());

    let primary_run = result
        .hypura_runs
        .iter()
        .find(|run| run.label == result.primary_run_label)
        .or_else(|| result.hypura_runs.first())
        .ok_or_else(|| anyhow::anyhow!("missing primary Hypura run"))?;

    let speedup_str = result
        .primary_speedup_vs_baseline
        .map(|s| format!("{s:.1}x"))
        .unwrap_or_else(|| "-".to_string());

    let date = &result.timestamp[..10];
    let row = format!(
        "| {date} | {} {} | {} {}GB | {:.1} GB | {:.1} GB | {:.1} GB | {baseline_str} | {:.1} | {speedup_str} |",
        result.model.name,
        result.model.quant,
        result.hardware.cpu,
        result.hardware.ram_gb as u32,
        primary_run.placement.gpu_gb,
        primary_run.placement.host_pageable_gb + primary_run.placement.host_pinned_gb,
        primary_run.placement.nvme_gb,
        primary_run.result.tok_per_sec,
    );

    let content = format!("{}\n{}\n", existing.trim_end(), row);
    std::fs::write(readme_path, content)?;

    Ok(())
}

fn turboquant_runtime_status(mode: TurboQuantMode, has_config: bool) -> &'static str {
    if mode == TurboQuantMode::Exact {
        "inactive"
    } else if !has_config {
        "unresolved"
    } else if mode == TurboQuantMode::PaperFullKv {
        "experimental-full-kv"
    } else if has_config {
        "faithful-attached"
    } else {
        "unresolved"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_cases_default_matrix_matches_plan() {
        let cases = benchmark_cases(None, None);
        assert_eq!(cases.len(), 3);
        assert_eq!(cases[0].residency_profile, ResidencyProfile::Legacy3Tier);
        assert_eq!(cases[1].host_pinned_policy, HostPinnedPolicy::Off);
        assert_eq!(cases[2].host_pinned_policy, HostPinnedPolicy::Auto);
    }

    #[test]
    fn benchmark_cases_normalize_legacy_host_policy() {
        let cases = benchmark_cases(Some(ResidencyProfile::Legacy3Tier), Some(HostPinnedPolicy::Force));
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].host_pinned_policy, HostPinnedPolicy::Off);
    }
}
