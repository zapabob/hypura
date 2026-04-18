use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::time::Instant;

use hypura::compute::inference::{self, LlamaTurboquantCliBridge};
use hypura::model::turboquant_sidecar::{RotationPolicy, TurboQuantMode};
use hypura::scheduler::types::{HostPinnedPolicy, ResidencyPolicyConfig, ResidencyProfile};
use hypura::server::compat::{CompatFeatureFlags, CompatPerfState};
use hypura::server::compat_storage::{
    CompatStorage, CompatStorageOptions, DEFAULT_NET_SAVE_SLOTS, LauncherProfile,
};
use hypura::server::ollama_types::GgufInfo;
use hypura::server::routes::{self, AppState};
use hypura::server::supervisor::{
    COMPAT_CONTROL_TOKEN_ENV, COMPAT_CONTROL_URL_ENV, COMPAT_FEATURES_JSON_ENV,
    CompatControlPlaneClient, CompatWorkerBootstrap, compat_feature_flags_from_env,
};
use hypura::telemetry::metrics::TelemetryEmitter;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{Value, json};

use super::fmt_util::{cli_progress_enabled, format_bytes};

fn set_process_env_var<K: AsRef<std::ffi::OsStr>, V: AsRef<std::ffi::OsStr>>(key: K, value: V) {
    unsafe {
        std::env::set_var(key, value);
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn compat_mode_enabled() -> bool {
    std::env::var("HYPURA_COMPAT_PROFILE")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "koboldcpp" | "kobold" | "compat"
            )
        })
        .unwrap_or(false)
}

fn compat_default_max_length(default_value: u32, context: u32) -> u32 {
    std::env::var("HYPURA_DEFAULT_GEN_AMT")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_value)
        .min(context.max(1))
}

fn browser_visible_host(host: &str) -> &str {
    match host {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
        _ => host,
    }
}

fn maybe_open_kobold_lite_gui(host: &str, port: u16) {
    if !compat_mode_enabled() || !env_flag("HYPURA_COMPAT_SHOW_GUI") {
        return;
    }

    let url = format!("http://{}:{port}/kobold-lite", browser_visible_host(host));

    #[cfg(target_os = "windows")]
    let launch_result = std::process::Command::new("explorer").arg(&url).spawn();

    #[cfg(target_os = "macos")]
    let launch_result = std::process::Command::new("open").arg(&url).spawn();

    #[cfg(all(unix, not(target_os = "macos")))]
    let launch_result = std::process::Command::new("xdg-open").arg(&url).spawn();

    if let Err(error) = launch_result {
        tracing::warn!("failed to open Kobold-lite UI automatically: {error}");
    }
}

pub(crate) fn load_json_object(path: &Path) -> anyhow::Result<Value> {
    let text = fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&text)?;
    anyhow::ensure!(value.is_object(), "launcher config must be a JSON object");
    Ok(value)
}

fn json_string_field(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_string)
}

pub(crate) fn resolve_canonical_db_path(
    model_path: &Path,
    savedata_bridge_path: Option<&Path>,
) -> std::path::PathBuf {
    if let Some(bridge) = savedata_bridge_path {
        let parent = bridge
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        let stem = bridge
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("koboldcpp");
        return parent.join(format!("{stem}.hypura.sqlite3"));
    }

    let parent = model_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let stem = model_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("koboldcpp");
    parent.join(format!("{stem}.koboldcpp.sqlite3"))
}

fn compat_protected_mode() -> bool {
    std::env::var("HYPURA_API_KEY")
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

pub(crate) fn build_runtime_launcher_profile(
    name: &str,
    seed_config: Option<Value>,
    model_path: &str,
    host: &str,
    port: u16,
    savedatafile: Option<&str>,
    preloadstory: Option<&str>,
    admindir: Option<&str>,
    embeddings_model: Option<&str>,
    asset_root: Option<&str>,
    ui_theme: &str,
) -> anyhow::Result<LauncherProfile> {
    let mut root = seed_config.unwrap_or_else(|| json!({}));
    let obj = root
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("launcher config seed must be a JSON object"))?;
    obj.insert(
        "model_param".to_string(),
        Value::String(model_path.to_string()),
    );
    obj.insert("host".to_string(), Value::String(host.to_string()));
    obj.insert("port_param".to_string(), json!(port));
    obj.insert(
        "showgui".to_string(),
        Value::Bool(env_flag("HYPURA_COMPAT_SHOW_GUI")),
    );
    obj.insert("ui_theme".to_string(), Value::String(ui_theme.to_string()));

    if let Some(savedatafile) = savedatafile.filter(|value| !value.trim().is_empty()) {
        obj.insert(
            "savedatafile".to_string(),
            Value::String(savedatafile.to_string()),
        );
    }
    if let Some(preloadstory) = preloadstory.filter(|value| !value.trim().is_empty()) {
        obj.insert(
            "preloadstory".to_string(),
            Value::String(preloadstory.to_string()),
        );
    }
    if let Some(admindir) = admindir.filter(|value| !value.trim().is_empty()) {
        obj.insert("admindir".to_string(), Value::String(admindir.to_string()));
        obj.insert("admin".to_string(), Value::Bool(true));
    }
    if let Some(embeddings_model) = embeddings_model.filter(|value| !value.trim().is_empty()) {
        obj.insert(
            "embeddings_model".to_string(),
            Value::String(embeddings_model.to_string()),
        );
    }
    if let Some(asset_root) = asset_root.filter(|value| !value.trim().is_empty()) {
        obj.insert(
            "asset_root".to_string(),
            Value::String(asset_root.to_string()),
        );
    }

    let max_length = std::env::var("HYPURA_DEFAULT_GEN_AMT")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(256);
    obj.insert(
        "gendefaults".to_string(),
        json!({ "max_length": max_length }),
    );
    obj.insert("istemplate".to_string(), Value::Bool(false));

    LauncherProfile::from_kcpps_value(name.to_string(), root)
}

pub fn run(
    model_path: &str,
    host: &str,
    port: u16,
    context: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    rotation_policy: RotationPolicy,
    _rotation_seed: u32,
    tq_so8_off: bool,
    tq_so8_learned: bool,
    tq_triality_off: bool,
    tq_triality_mix: f32,
    tq_rotation_seed: u32,
    tq_artifact: Option<&str>,
    model_dir: Option<&str>,
    ui_theme: &str,
    dry_run: bool,
    residency_profile: ResidencyProfile,
    host_pinned: HostPinnedPolicy,
    savedatafile: Option<&str>,
    preloadstory: Option<&str>,
    admindir: Option<&str>,
    config: Option<&str>,
    exportconfig: Option<&str>,
    migration_dir: Option<&str>,
) -> anyhow::Result<()> {
    let llama_bridge = LlamaTurboquantCliBridge {
        rotation_policy,
        llama_rotation_seed: tq_rotation_seed,
        tq_so8_off,
        tq_triality_off,
        tq_so8_learned,
        tq_triality_mix,
        tq_artifact: tq_artifact.map(|s| s.to_string()),
    };
    if !ui_theme.trim().is_empty() {
        set_process_env_var("HYPURA_UI_THEME", ui_theme.trim());
    }
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(
        model_path,
        host,
        port,
        context,
        turboquant_mode,
        turboquant_config,
        model_dir,
        llama_bridge,
        ui_theme,
        dry_run,
        ResidencyPolicyConfig::new(residency_profile, host_pinned),
        savedatafile,
        preloadstory,
        admindir,
        config,
        exportconfig,
        migration_dir,
    ))
}

pub fn run_worker_bootstrap(bootstrap_file: &str) -> anyhow::Result<()> {
    let bootstrap = CompatWorkerBootstrap::read_from_path(Path::new(bootstrap_file))?;
    unsafe {
        std::env::set_var("HYPURA_COMPAT_PROFILE", "koboldcpp");
        std::env::set_var(
            "HYPURA_DEFAULT_GEN_AMT",
            bootstrap.default_max_length.to_string(),
        );
        std::env::set_var("HYPURA_COMPAT_SHOW_GUI", "0");
        std::env::set_var(COMPAT_CONTROL_URL_ENV, &bootstrap.control_plane.base_url);
        std::env::set_var(
            COMPAT_CONTROL_TOKEN_ENV,
            &bootstrap.control_plane.bearer_token,
        );
        std::env::set_var(
            COMPAT_FEATURES_JSON_ENV,
            serde_json::to_string(&bootstrap.feature_state)?,
        );
    }
    run(
        &bootstrap.model_path,
        &bootstrap.public_host,
        bootstrap.public_port,
        bootstrap.context,
        bootstrap.turboquant_mode,
        bootstrap.turboquant_config.as_deref(),
        bootstrap.rotation_policy,
        bootstrap.rotation_seed,
        bootstrap.tq_so8_off,
        bootstrap.tq_so8_learned,
        bootstrap.tq_triality_off,
        bootstrap.tq_triality_mix,
        bootstrap.tq_rotation_seed,
        bootstrap.tq_artifact.as_deref(),
        bootstrap.model_dir.as_deref(),
        &bootstrap.ui_theme,
        false,
        bootstrap.residency_profile,
        bootstrap.host_pinned,
        bootstrap.savedatafile.as_deref(),
        bootstrap.preloadstory.as_deref(),
        bootstrap.admindir.as_deref(),
        bootstrap.config.as_deref(),
        None,
        bootstrap.migration_dir.as_deref(),
    )
}
async fn run_async(
    model_path: &str,
    host: &str,
    port: u16,
    context: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    model_dir: Option<&str>,
    llama_bridge: LlamaTurboquantCliBridge,
    ui_theme: &str,
    dry_run: bool,
    residency_policy: ResidencyPolicyConfig,
    savedatafile: Option<&str>,
    preloadstory: Option<&str>,
    admindir: Option<&str>,
    config: Option<&str>,
    exportconfig: Option<&str>,
    migration_dir: Option<&str>,
) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    let seed_config = if compat_mode_enabled() {
        config
            .map(|path| load_json_object(Path::new(path)))
            .transpose()?
    } else {
        None
    };
    let seed_profile = seed_config
        .as_ref()
        .map(|value| LauncherProfile::from_kcpps_value("seed.kcpps".to_string(), value.clone()))
        .transpose()?;
    let effective_savedatafile = savedatafile.map(str::to_string).or_else(|| {
        seed_profile
            .as_ref()
            .and_then(|profile| profile.savedatafile.clone())
    });
    let effective_preloadstory = preloadstory.map(str::to_string).or_else(|| {
        seed_profile
            .as_ref()
            .and_then(|profile| profile.preloadstory.clone())
    });
    let effective_admindir = admindir.map(str::to_string).or_else(|| {
        seed_config
            .as_ref()
            .and_then(|config| json_string_field(config, "admindir"))
    });

    let pb_setup = if cli_progress_enabled() {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
        );
        pb.set_message("Analyzing GGUF, hardware profile, and placement…");
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
        pb
    } else {
        ProgressBar::hidden()
    };

    let runtime = inference::resolve_runtime_setup(
        path,
        context,
        turboquant_mode,
        turboquant_config.map(Path::new),
        llama_bridge.clone(),
        residency_policy,
    )?;
    if cli_progress_enabled() {
        pb_setup.finish_and_clear();
    }

    let compat_storage = if compat_mode_enabled() {
        let canonical_db_path =
            resolve_canonical_db_path(path, effective_savedatafile.as_deref().map(Path::new));
        let storage = Arc::new(CompatStorage::open(CompatStorageOptions {
            canonical_db_path,
            savedata_bridge_path: effective_savedatafile
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(std::path::PathBuf::from),
            preload_story_path: effective_preloadstory
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(std::path::PathBuf::from),
            admindir: effective_admindir
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(std::path::PathBuf::from),
            migration_dir: migration_dir
                .filter(|value| !value.trim().is_empty())
                .map(std::path::PathBuf::from),
            slot_count: DEFAULT_NET_SAVE_SLOTS,
            default_ui_theme: ui_theme.to_ascii_lowercase(),
        })?);
        let current_profile_name = config
            .and_then(|path| {
                Path::new(path)
                    .file_name()
                    .and_then(|file| file.to_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "current-session.kcpps".to_string());
        let current_profile = build_runtime_launcher_profile(
            &current_profile_name,
            seed_config.clone(),
            model_path,
            host,
            port,
            effective_savedatafile.as_deref(),
            effective_preloadstory.as_deref(),
            effective_admindir.as_deref(),
            None,
            None,
            ui_theme,
        )?;
        storage.save_launcher_profile(&current_profile)?;
        storage.set_current_launcher_profile(&current_profile.name)?;
        if let Some(exportconfig) = exportconfig.filter(|value| !value.trim().is_empty()) {
            storage.export_current_launcher_profile(Path::new(exportconfig))?;
        }
        Some(storage)
    } else {
        None
    };

    if dry_run {
        println!();
        println!("Hypura serve dry-run");
        println!("  Model: {}", path.display());
        println!("  Bind: http://{host}:{port}");
        println!(
            "  Placement: {} GPU | {} host pageable | {} host pinned | {} NVMe",
            format_bytes(runtime.placement_summary.total_gpu_bytes),
            format_bytes(runtime.placement_summary.total_host_pageable_bytes),
            format_bytes(runtime.placement_summary.total_host_pinned_bytes),
            format_bytes(runtime.placement_summary.total_nvme_bytes),
        );
        println!(
            "  Residency: mode={}, pinned_tier={}, pinned_policy={}, n_gpu_layers={}",
            runtime.plan.residency_policy.residency_profile.label(),
            if runtime.placement_summary.host_pinned_active {
                "active"
            } else {
                "collapsed"
            },
            runtime.plan.residency_policy.host_pinned_policy.label(),
            runtime.n_gpu_layers
        );
        println!(
            "  TurboQuant: mode={}, schema={}, config={}, runtime_status={}",
            runtime.turboquant.mode,
            runtime.turboquant.schema_label(),
            runtime.turboquant.source_label(),
            if runtime.turboquant.mode == hypura::model::turboquant_sidecar::TurboQuantMode::Exact {
                "inactive"
            } else if runtime.turboquant.mode
                == hypura::model::turboquant_sidecar::TurboQuantMode::PaperFullKv
            {
                "experimental-full-kv"
            } else {
                "faithful-attached"
            }
        );
        if let Some(ref gguf_cfg) = runtime.turboquant.gguf_metadata {
            println!(
                "  Triality profile: public_mode={}, runtime_mode={}, schema_version={}",
                gguf_cfg.public_mode_label, gguf_cfg.runtime_mode, gguf_cfg.schema_version
            );
            println!(
                "  Triality rotation: policy={}, seed={}, view={}, mix={:.3}",
                gguf_cfg
                    .rotation_policy
                    .map(|policy| policy.as_str().to_string())
                    .unwrap_or_else(|| "none".to_string()),
                gguf_cfg.rotation_seed,
                gguf_cfg.triality_view.as_deref().unwrap_or("none"),
                gguf_cfg.triality_mix.unwrap_or(0.0)
            );
            println!(
                "  Triality payload: format={}, bytes={}, paper_fidelity={}, k_bits={:.3}, v_bits={:.3}, inline_json={}",
                gguf_cfg.payload_format.as_deref().unwrap_or("none"),
                gguf_cfg.payload_bytes,
                gguf_cfg.paper_fidelity,
                gguf_cfg.k_bits,
                gguf_cfg.v_bits,
                if gguf_cfg.payload_json.is_some() {
                    "yes"
                } else {
                    "no"
                }
            );
        }
        if let Some(storage) = compat_storage.as_ref() {
            println!(
                "  Compat storage: {}",
                storage.canonical_db_path().display()
            );
            if let Some(savedata_bridge_path) = storage.savedata_bridge_path() {
                println!("  SaveData bridge: {}", savedata_bridge_path.display());
            }
            if let Some(admindir) = storage.admindir() {
                println!("  Admin dir: {}", admindir.display());
            }
        }
        println!("  Dry-run: runtime wiring resolved without loading model weights");
        return Ok(());
    }
    let file_size = std::fs::metadata(path)?.len();

    let gguf_info = GgufInfo {
        file_size,
        architecture: runtime.metadata.architecture.clone(),
        parameter_count: runtime.metadata.parameter_count,
        quantization: runtime
            .metadata
            .quantization
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        context_length: runtime.metadata.context_length,
    };

    let config = inference::InferenceConfig {
        n_ctx: context,
        ..inference::InferenceConfig::default()
    };

    let pb = if cli_progress_enabled() {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
        );
        pb.set_message("Loading model weights into memory (no ETA — size-dependent)…");
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
        pb
    } else {
        ProgressBar::hidden()
    };

    let load_start = Instant::now();

    let path_owned = path.to_path_buf();
    let plan_arc = Arc::new(runtime.plan.clone());
    let gguf_arc = Arc::new(runtime.gguf.clone());
    let turboquant = runtime.turboquant.clone();
    let plan_for_load = plan_arc.clone();
    let gguf_for_load = gguf_arc.clone();
    let n_gpu_layers = runtime.n_gpu_layers;

    let loaded = tokio::task::spawn_blocking(move || {
        inference::load_model(
            &path_owned,
            &config,
            n_gpu_layers,
            &plan_for_load,
            &gguf_for_load,
            &turboquant,
        )
    })
    .await??;

    if cli_progress_enabled() {
        pb.finish_and_clear();
    }
    let load_secs = load_start.elapsed().as_secs_f64();
    println!("Model loaded in {load_secs:.1}s");

    let load_duration_ns = load_start.elapsed().as_nanos() as u64;
    let model_name = std::env::var("HYPURA_MODEL_NAME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| loaded.model_name.clone());

    let telemetry = Arc::new(TelemetryEmitter::new(256));
    let init_theme = if let Some(storage) = compat_storage.as_ref() {
        storage.get_ui_theme()?
    } else {
        std::env::var("HYPURA_UI_THEME")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| matches!(s.as_str(), "light" | "dark" | "classic"))
            .unwrap_or_else(|| "classic".to_string())
    };

    let serve_turboquant_config_path = turboquant_config.map(|s| std::path::PathBuf::from(s));
    let compat_default_max_length_value = compat_default_max_length(256, context);
    let compat_feature_defaults = CompatFeatureFlags {
        savedata: compat_storage
            .as_ref()
            .map(|storage| storage.savedata_enabled())
            .unwrap_or(false),
        admin: if compat_storage
            .as_ref()
            .and_then(|storage| storage.admindir())
            .is_some()
        {
            if compat_protected_mode() { 2 } else { 1 }
        } else {
            0
        },
        ..CompatFeatureFlags::default()
    };
    let compat_features = if compat_mode_enabled() {
        compat_feature_flags_from_env().unwrap_or(compat_feature_defaults)
    } else {
        compat_feature_defaults
    };
    let loaded_model = Arc::new(std::sync::Mutex::new(loaded));
    let compat_control_client = if compat_mode_enabled() {
        CompatControlPlaneClient::from_env()
    } else {
        None
    };
    let compat_session = if compat_mode_enabled() {
        Some(Arc::new(std::sync::Mutex::new(
            inference::CompatRuntimeSession::new(loaded_model.clone())?,
        )))
    } else {
        None
    };

    let state = Arc::new(AppState {
        loaded_model,
        model_name: Arc::new(std::sync::Mutex::new(model_name.clone())),
        model_path: Arc::new(std::sync::Mutex::new(path.to_path_buf())),
        gguf_info: Arc::new(std::sync::Mutex::new(gguf_info)),
        model_dir: model_dir
            .map(std::path::PathBuf::from)
            .or_else(|| path.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from(".")),
        default_context: Arc::new(AtomicU32::new(context)),
        load_duration_ns,
        telemetry,
        turboquant: runtime.turboquant,
        serve_turboquant_mode: turboquant_mode,
        serve_turboquant_config_path,
        serve_llama_bridge: llama_bridge,
        serve_residency_policy: residency_policy.normalized(),
        active_cancel: Arc::new(std::sync::Mutex::new(None)),
        generation_in_progress: Arc::new(AtomicBool::new(false)),
        gui_presets: Arc::new(std::sync::Mutex::new(HashMap::new())),
        gui_history: Arc::new(std::sync::Mutex::new(VecDeque::new())),
        gui_events: Arc::new(std::sync::Mutex::new(VecDeque::new())),
        ui_theme: Arc::new(std::sync::Mutex::new(init_theme)),
        compat_storage: compat_storage.clone(),
        compat_started_at: Instant::now(),
        compat_default_max_length: Arc::new(AtomicU32::new(compat_default_max_length_value)),
        compat_features: Arc::new(std::sync::Mutex::new(compat_features)),
        compat_perf: Arc::new(std::sync::Mutex::new(CompatPerfState::default())),
        compat_control_client,
        compat_session,
    });

    if let Some(storage) = compat_storage.as_ref() {
        storage.migrate_gui_state(
            &state.gui_presets.lock().unwrap(),
            &state.gui_history.lock().unwrap(),
            &state.gui_events.lock().unwrap(),
        )?;
    }

    let app = routes::router(state.clone());
    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    println!();
    println!("Hypura serving {model_name}");
    println!("  Endpoint: http://{bind_addr}");
    println!("  Ollama-compatible API: /api/generate, /api/chat, /api/tags");
    if compat_mode_enabled() {
        println!(
            "  KoboldCpp-compatible API: /api/v1/*, /api/extra/*, /v1/completions, /v1/chat/completions"
        );
        println!(
            "  Kobold-lite UI: http://{}:{port}/kobold-lite",
            browser_visible_host(host)
        );
        println!(
            "  Kobold default max_length: {}",
            state
                .compat_default_max_length
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        if let Some(storage) = state.compat_storage.as_ref() {
            println!(
                "  Compat storage DB: {}",
                storage.canonical_db_path().display()
            );
            if let Some(savedata_bridge) = storage.savedata_bridge_path() {
                println!("  SaveData bridge: {}", savedata_bridge.display());
            }
            if let Some(admindir) = storage.admindir() {
                println!("  Admin dir: {}", admindir.display());
            }
        }
    }
    if std::env::var("HYPURA_API_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .is_some()
    {
        println!(
            "  API key: required (HYPURA_API_KEY); use Authorization: Bearer <key> or X-API-Key"
        );
    }
    println!(
        "  TurboQuant: mode={}, schema={}, config={}, runtime_status={}",
        state.turboquant.mode,
        state.turboquant.schema_label(),
        state.turboquant.source_label(),
        if state.turboquant.mode == hypura::model::turboquant_sidecar::TurboQuantMode::Exact {
            "inactive"
        } else if state.turboquant.mode
            == hypura::model::turboquant_sidecar::TurboQuantMode::PaperFullKv
        {
            "experimental-full-kv"
        } else {
            "faithful-attached"
        }
    );
    if let Some(ref gguf_cfg) = state.turboquant.gguf_metadata {
        println!(
            "  Triality profile: public_mode={}, runtime_mode={}, schema_version={}",
            gguf_cfg.public_mode_label, gguf_cfg.runtime_mode, gguf_cfg.schema_version
        );
        println!(
            "  Triality rotation: policy={}, seed={}, view={}, mix={:.3}",
            gguf_cfg
                .rotation_policy
                .map(|policy| policy.as_str().to_string())
                .unwrap_or_else(|| "none".to_string()),
            gguf_cfg.rotation_seed,
            gguf_cfg.triality_view.as_deref().unwrap_or("none"),
            gguf_cfg.triality_mix.unwrap_or(0.0)
        );
        println!(
            "  Triality payload: format={}, bytes={}, paper_fidelity={}, k_bits={:.3}, v_bits={:.3}, inline_json={}",
            gguf_cfg.payload_format.as_deref().unwrap_or("none"),
            gguf_cfg.payload_bytes,
            gguf_cfg.paper_fidelity,
            gguf_cfg.k_bits,
            gguf_cfg.v_bits,
            if gguf_cfg.payload_json.is_some() {
                "yes"
            } else {
                "no"
            }
        );
    }
    println!(
        "  Placement: {} GPU | {} host pageable | {} host pinned | {} NVMe",
        format_bytes(runtime.placement_summary.total_gpu_bytes),
        format_bytes(runtime.placement_summary.total_host_pageable_bytes),
        format_bytes(runtime.placement_summary.total_host_pinned_bytes),
        format_bytes(runtime.placement_summary.total_nvme_bytes),
    );
    println!(
        "  Residency: mode={}, pinned_tier={}, pinned_policy={}",
        runtime.plan.residency_policy.residency_profile.label(),
        if runtime.placement_summary.host_pinned_active {
            "active"
        } else {
            "collapsed"
        },
        runtime.plan.residency_policy.host_pinned_policy.label(),
    );
    println!();

    maybe_open_kobold_lite_gui(host, port);
    axum::serve(listener, app).await?;
    Ok(())
}
