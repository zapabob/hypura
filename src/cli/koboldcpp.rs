use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context;
use serde_json::Value;
use tokio::process::{Child, Command};

use hypura::compat_assets::{
    AssetBootstrapReport, bootstrap_assets, default_asset_root, discover_asset_manifest_path,
    materialize_pending_asset_entry,
};
use hypura::model::turboquant_sidecar::{RotationPolicy, TurboQuantMode};
use hypura::scheduler::types::{HostPinnedPolicy, ResidencyProfile};
use hypura::server::compat::CompatFeatureFlags;
use hypura::server::compat_storage::{
    CompatStorage, CompatStorageOptions, DEFAULT_NET_SAVE_SLOTS, LauncherProfile,
};
use hypura::server::embeddings::EmbeddingsRuntime;
use hypura::server::supervisor::{
    CompatControlPlaneClient, CompatControlPlaneClientInfo, CompatSupervisorCommand,
    CompatWorkerBootstrap, MultimodalBackendConfig,
    spawn_supervisor_control_plane,
};

use super::serve::{build_runtime_launcher_profile, load_json_object, resolve_canonical_db_path};

fn browser_visible_host(host: &str) -> &str {
    match host {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1",
        _ => host,
    }
}

fn maybe_open_kobold_lite_gui(host: &str, port: u16) {
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

fn compat_protected_mode() -> bool {
    std::env::var("HYPURA_API_KEY")
        .ok()
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

fn launcher_profile_context(profile: &LauncherProfile, fallback: u32) -> u32 {
    parse_u32_value(profile.raw_config.get("contextsize"))
        .or_else(|| parse_u32_value(profile.raw_config.get("max_context_length")))
        .or_else(|| parse_u32_value(profile.raw_config.get("context")))
        .or_else(|| parse_u32_value(profile.raw_config.get("n_ctx")))
        .unwrap_or(fallback)
        .max(256)
}

fn launcher_profile_max_length(profile: &LauncherProfile, fallback: u32) -> u32 {
    profile
        .gendefaults
        .as_ref()
        .and_then(|defaults| parse_u32_value(defaults.get("max_length")))
        .unwrap_or(fallback)
        .max(1)
}

fn parse_u32_value(value: Option<&Value>) -> Option<u32> {
    value.and_then(|value| {
        value
            .as_u64()
            .and_then(|raw| u32::try_from(raw).ok())
            .or_else(|| value.as_str().and_then(|raw| raw.trim().parse::<u32>().ok()))
    })
}

fn parse_string_value(value: Option<&Value>) -> Option<String> {
    value.and_then(|value| value.as_str().map(str::trim))
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn launcher_profile_string(profile: &LauncherProfile, key: &str) -> Option<String> {
    parse_string_value(profile.raw_config.get(key))
}

fn resolve_profile_path(storage: Option<&CompatStorage>, raw_path: &str) -> PathBuf {
    let candidate = PathBuf::from(raw_path);
    if candidate.is_absolute() {
        return candidate;
    }
    if let Some(admindir) = storage.and_then(|storage| storage.admindir()) {
        let joined = admindir.join(raw_path);
        if joined.exists() {
            return joined;
        }
    }
    candidate
}

fn resolve_asset_relative_path(
    storage: Option<&CompatStorage>,
    asset_root: &Path,
    raw_path: &str,
) -> PathBuf {
    let candidate = PathBuf::from(raw_path);
    if candidate.is_absolute() {
        return candidate;
    }
    let asset_candidate = asset_root.join(&candidate);
    if asset_candidate.exists() {
        return asset_candidate;
    }
    resolve_profile_path(storage, raw_path)
}

fn compute_base_feature_state(storage: &CompatStorage) -> CompatFeatureFlags {
    CompatFeatureFlags {
        savedata: storage.savedata_enabled(),
        admin: if storage.admindir().is_some() {
            if compat_protected_mode() { 2 } else { 1 }
        } else {
            0
        },
        ..CompatFeatureFlags::default()
    }
}

fn finalize_feature_state(
    storage: &CompatStorage,
    multimodal: &MultimodalBackendConfig,
    embeddings_enabled: bool,
) -> CompatFeatureFlags {
    let mut features = multimodal.apply_to_features(compute_base_feature_state(storage));
    features.websearch = true;
    features.embeddings = embeddings_enabled;
    features
}

fn resolve_embeddings_model_path(
    storage: Option<&CompatStorage>,
    asset_root: &Path,
    cli_embeddings_model: Option<&str>,
    profile: Option<&LauncherProfile>,
    fallback_model: Option<&Path>,
    asset_report: &AssetBootstrapReport,
) -> Option<PathBuf> {
    cli_embeddings_model
        .filter(|value| !value.trim().is_empty())
        .map(|value| resolve_asset_relative_path(storage, asset_root, value))
        .or_else(|| {
            profile
                .and_then(|profile| launcher_profile_string(profile, "embeddings_model"))
                .map(|value| resolve_asset_relative_path(storage, asset_root, &value))
        })
        .or_else(|| fallback_model.map(Path::to_path_buf))
        .or_else(|| asset_report.ready.get("embeddings_model").cloned())
}

fn load_embeddings_runtime(model_path: Option<&Path>) -> anyhow::Result<Option<EmbeddingsRuntime>> {
    let Some(model_path) = model_path else {
        return Ok(None);
    };
    if !model_path.exists() {
        tracing::warn!(
            "compat embeddings model path does not exist yet: {}",
            model_path.display()
        );
        return Ok(None);
    }
    Ok(Some(EmbeddingsRuntime::load(model_path)?))
}

fn spawn_pending_asset_downloads(
    client_info: CompatControlPlaneClientInfo,
    asset_root: PathBuf,
    pending: Vec<hypura::compat_assets::KoboldcppAssetEntry>,
) {
    let downloadable: Vec<_> = pending
        .into_iter()
        .filter(|entry| entry.download_url.is_some())
        .collect();
    if downloadable.is_empty() {
        return;
    }

    tokio::spawn(async move {
        let client = CompatControlPlaneClient::new(client_info);
        let mut downloaded_any = false;
        for entry in downloadable {
            let entry_id = entry.id.clone();
            let asset_root = asset_root.clone();
            let result = tokio::task::spawn_blocking(move || {
                materialize_pending_asset_entry(&asset_root, &entry)
            })
            .await;
            match result {
                Ok(Ok(path)) => {
                    downloaded_any = true;
                    tracing::info!(
                        "downloaded compat asset {} to {}",
                        entry_id,
                        path.display()
                    );
                }
                Ok(Err(error)) => {
                    tracing::warn!("failed to materialize compat asset {}: {error}", entry_id);
                }
                Err(error) => {
                    tracing::warn!("asset download task failed for {}: {error}", entry_id);
                }
            }
        }
        if downloaded_any {
            let _ = client
                .send_command(CompatSupervisorCommand::ReprobeBundles)
                .await;
        }
    });
}

#[derive(Debug, Clone)]
struct WorkerLaunchState {
    bootstrap: CompatWorkerBootstrap,
    bootstrap_path: PathBuf,
}

async fn spawn_worker(launch: &WorkerLaunchState) -> anyhow::Result<Child> {
    launch.bootstrap.write_to_path(&launch.bootstrap_path)?;
    let current_exe = std::env::current_exe()?;
    let mut command = Command::new(current_exe);
    command
        .arg("__koboldcpp_worker")
        .arg("--bootstrap-file")
        .arg(&launch.bootstrap_path)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    Ok(command.spawn()?)
}

async fn stop_worker(child: &mut Child) -> anyhow::Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }
    child.start_kill()?;
    let _ = child.wait().await?;
    Ok(())
}

async fn wait_for_worker_ready(host: &str, port: u16) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://{}:{port}/api/version", browser_visible_host(host));
    let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
    loop {
        if let Ok(response) = client.get(&url).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("worker did not become ready at {url}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    model_path: &str,
    host: &str,
    port: u16,
    context: u32,
    max_length: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    rotation_policy: RotationPolicy,
    rotation_seed: u32,
    tq_so8_off: bool,
    tq_so8_learned: bool,
    tq_triality_off: bool,
    tq_triality_mix: f32,
    tq_rotation_seed: u32,
    tq_artifact: Option<&str>,
    model_dir: Option<&str>,
    savedatafile: Option<&str>,
    embeddings_model: Option<&str>,
    preloadstory: Option<&str>,
    admindir: Option<&str>,
    config: Option<&str>,
    exportconfig: Option<&str>,
    migration_dir: Option<&str>,
    asset_root: Option<&str>,
    ui_theme: &str,
    no_show_gui: bool,
    dry_run: bool,
    residency_profile: ResidencyProfile,
    host_pinned: HostPinnedPolicy,
) -> anyhow::Result<()> {
    if dry_run {
        unsafe {
            std::env::set_var("HYPURA_COMPAT_PROFILE", "koboldcpp");
            std::env::set_var("HYPURA_DEFAULT_GEN_AMT", max_length.to_string());
            std::env::set_var("HYPURA_COMPAT_SHOW_GUI", "0");
        }
        return super::serve::run(
            model_path,
            None,
            host,
            port,
            context,
            turboquant_mode,
            turboquant_config,
            rotation_policy,
            rotation_seed,
            tq_so8_off,
            tq_so8_learned,
            tq_triality_off,
            tq_triality_mix,
            tq_rotation_seed,
            tq_artifact,
            model_dir,
            ui_theme,
            true,
            residency_profile,
            host_pinned,
            savedatafile,
            preloadstory,
            admindir,
            config,
            exportconfig,
            migration_dir,
        );
    }

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(run_async(
        model_path,
        host,
        port,
        context,
        max_length,
        turboquant_mode,
        turboquant_config,
        rotation_policy,
        rotation_seed,
        tq_so8_off,
        tq_so8_learned,
        tq_triality_off,
        tq_triality_mix,
        tq_rotation_seed,
        tq_artifact,
        model_dir,
        savedatafile,
        embeddings_model,
        preloadstory,
        admindir,
        config,
        exportconfig,
        migration_dir,
        asset_root,
        ui_theme,
        no_show_gui,
        residency_profile,
        host_pinned,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn run_async(
    model_path: &str,
    host: &str,
    port: u16,
    context: u32,
    max_length: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    rotation_policy: RotationPolicy,
    rotation_seed: u32,
    tq_so8_off: bool,
    tq_so8_learned: bool,
    tq_triality_off: bool,
    tq_triality_mix: f32,
    tq_rotation_seed: u32,
    tq_artifact: Option<&str>,
    model_dir: Option<&str>,
    savedatafile: Option<&str>,
    embeddings_model: Option<&str>,
    preloadstory: Option<&str>,
    admindir: Option<&str>,
    config: Option<&str>,
    exportconfig: Option<&str>,
    migration_dir: Option<&str>,
    asset_root: Option<&str>,
    ui_theme: &str,
    no_show_gui: bool,
    residency_profile: ResidencyProfile,
    host_pinned: HostPinnedPolicy,
) -> anyhow::Result<()> {
    let seed_config = config.map(|path| load_json_object(Path::new(path))).transpose()?;
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
            .and_then(|config| config.get("admindir"))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let mut current_asset_root = asset_root
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            seed_profile
                .as_ref()
                .and_then(|profile| launcher_profile_string(profile, "asset_root"))
                .map(PathBuf::from)
        })
        .unwrap_or_else(default_asset_root);
    let mut asset_manifest_path = discover_asset_manifest_path();
    let canonical_db_path = resolve_canonical_db_path(
        Path::new(model_path),
        effective_savedatafile.as_deref().map(Path::new),
    );
    let storage = Arc::new(CompatStorage::open(CompatStorageOptions {
        canonical_db_path,
        savedata_bridge_path: effective_savedatafile.clone().map(PathBuf::from),
        preload_story_path: effective_preloadstory.clone().map(PathBuf::from),
        admindir: effective_admindir.clone().map(PathBuf::from),
        migration_dir: migration_dir
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from),
        slot_count: DEFAULT_NET_SAVE_SLOTS,
        default_ui_theme: ui_theme.to_ascii_lowercase(),
    })?);

    if let Some(preloadstory) = effective_preloadstory.as_deref() {
        let preload_path = resolve_profile_path(Some(&storage), preloadstory);
        if preload_path.exists() {
            storage.import_preload_story_path(&preload_path)?;
        }
    }

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
            embeddings_model,
            Some(current_asset_root.to_string_lossy().as_ref()),
            ui_theme,
        )?;
    storage.save_launcher_profile(&current_profile)?;
    storage.set_current_launcher_profile(&current_profile.name)?;
    if let Some(exportconfig) = exportconfig.filter(|value| !value.trim().is_empty()) {
        storage.export_current_launcher_profile(Path::new(exportconfig))?;
    }

    let mut asset_report = bootstrap_assets(&current_asset_root, asset_manifest_path.as_deref())?;
    let mut current_embeddings_model = resolve_embeddings_model_path(
        Some(&storage),
        &current_asset_root,
        embeddings_model,
        Some(&current_profile),
        None,
        &asset_report,
    );
    let embeddings_runtime = Arc::new(Mutex::new(load_embeddings_runtime(
        current_embeddings_model.as_deref(),
    )?));
    let multimodal = MultimodalBackendConfig::from_env();
    let embeddings_available = embeddings_runtime.lock().unwrap().is_some();
    let feature_state = finalize_feature_state(&storage, &multimodal, embeddings_available);
    let control_plane =
        spawn_supervisor_control_plane(multimodal.clone(), embeddings_runtime.clone()).await?;
    spawn_pending_asset_downloads(
        control_plane.client_info.clone(),
        current_asset_root.clone(),
        asset_report.pending.clone(),
    );
    let bootstrap_path = std::env::temp_dir().join(format!(
        "hypura-koboldcpp-worker-{}.json",
        uuid::Uuid::new_v4().simple()
    ));
    let mut launch = WorkerLaunchState {
        bootstrap: CompatWorkerBootstrap {
            public_host: host.to_string(),
            public_port: port,
            model_path: model_path.to_string(),
            context,
            default_max_length: max_length.min(context.max(1)),
            turboquant_mode,
            turboquant_config: turboquant_config.map(str::to_string),
            rotation_policy,
            rotation_seed,
            tq_so8_off,
            tq_so8_learned,
            tq_triality_off,
            tq_triality_mix,
            tq_rotation_seed,
            tq_artifact: tq_artifact.map(str::to_string),
            model_dir: model_dir.map(str::to_string),
            ui_theme: ui_theme.to_string(),
            savedatafile: effective_savedatafile.clone(),
            preloadstory: effective_preloadstory.clone(),
            admindir: effective_admindir.clone(),
            config: config.map(str::to_string),
            migration_dir: migration_dir.map(str::to_string),
            residency_profile,
            host_pinned,
            control_plane: control_plane.client_info.clone(),
            feature_state,
        },
        bootstrap_path,
    };

    let mut child = spawn_worker(&launch).await?;
    wait_for_worker_ready(host, port).await?;
    if !no_show_gui {
        maybe_open_kobold_lite_gui(host, port);
    }

    let control_multimodal = control_plane.multimodal.clone();
    let control_embeddings = control_plane.embeddings.clone();
    let control_plane_client_info = control_plane.client_info.clone();
    let mut command_rx = control_plane.command_rx;
    loop {
        tokio::select! {
            maybe_command = command_rx.recv() => {
                let Some(command) = maybe_command else {
                    break;
                };
                match command {
                    CompatSupervisorCommand::ReloadConfig { filename, baseconfig } => {
                        let profile = storage
                            .reload_admin_profile(&filename, baseconfig.as_deref())
                            .with_context(|| format!("reloading admin profile {filename}"))?;
                        let next_context = launcher_profile_context(&profile, launch.bootstrap.context);
                        let next_max_length = launcher_profile_max_length(&profile, launch.bootstrap.default_max_length)
                            .min(next_context.max(1));
                        launch.bootstrap.preloadstory = profile.preloadstory.clone();
                        if let Some(preloadstory) = profile.preloadstory.as_deref() {
                            let preload_path = resolve_profile_path(Some(&storage), preloadstory);
                            if preload_path.exists() {
                                storage.import_preload_story_path(&preload_path)?;
                            }
                        }
                        if let Some(model_param) = profile.model_param.as_deref() {
                            let next_model_path = resolve_profile_path(Some(&storage), model_param);
                            launch.bootstrap.model_path = next_model_path.to_string_lossy().to_string();
                        }
                        let next_asset_root = launcher_profile_string(&profile, "asset_root")
                            .map(PathBuf::from)
                            .unwrap_or_else(|| current_asset_root.clone());
                        asset_manifest_path = discover_asset_manifest_path().or(asset_manifest_path.clone());
                        asset_report = bootstrap_assets(&next_asset_root, asset_manifest_path.as_deref())?;
                        spawn_pending_asset_downloads(
                            control_plane_client_info.clone(),
                            next_asset_root.clone(),
                            asset_report.pending.clone(),
                        );
                        let next_embeddings_model = resolve_embeddings_model_path(
                            Some(&storage),
                            &next_asset_root,
                            None,
                            Some(&profile),
                            current_embeddings_model.as_deref(),
                            &asset_report,
                        );
                        *control_embeddings.lock().unwrap() =
                            load_embeddings_runtime(next_embeddings_model.as_deref())?;
                        current_asset_root = next_asset_root;
                        current_embeddings_model = next_embeddings_model;
                        let multimodal = control_multimodal.lock().unwrap().clone();
                        launch.bootstrap.context = next_context;
                        launch.bootstrap.default_max_length = next_max_length;
                        launch.bootstrap.config = Some(resolve_profile_path(Some(&storage), &filename).to_string_lossy().to_string());
                        launch.bootstrap.feature_state = finalize_feature_state(
                            &storage,
                            &multimodal,
                            control_embeddings.lock().unwrap().is_some(),
                        );
                        stop_worker(&mut child).await?;
                        child = spawn_worker(&launch).await?;
                    }
                    CompatSupervisorCommand::ReprobeBundles => {
                        let multimodal = MultimodalBackendConfig::from_env();
                        *control_multimodal.lock().unwrap() = multimodal.clone();
                        asset_manifest_path = discover_asset_manifest_path().or(asset_manifest_path.clone());
                        asset_report = bootstrap_assets(&current_asset_root, asset_manifest_path.as_deref())?;
                        spawn_pending_asset_downloads(
                            control_plane_client_info.clone(),
                            current_asset_root.clone(),
                            asset_report.pending.clone(),
                        );
                        current_embeddings_model = resolve_embeddings_model_path(
                            Some(&storage),
                            &current_asset_root,
                            None,
                            storage.get_current_launcher_profile()?.as_ref(),
                            current_embeddings_model.as_deref(),
                            &asset_report,
                        );
                        *control_embeddings.lock().unwrap() =
                            load_embeddings_runtime(current_embeddings_model.as_deref())?;
                        launch.bootstrap.feature_state = finalize_feature_state(
                            &storage,
                            &multimodal,
                            control_embeddings.lock().unwrap().is_some(),
                        );
                        stop_worker(&mut child).await?;
                        child = spawn_worker(&launch).await?;
                    }
                    CompatSupervisorCommand::ShutdownWorker => {
                        stop_worker(&mut child).await?;
                        break;
                    }
                }
            }
            exit = child.wait() => {
                let status = exit?;
                anyhow::bail!("koboldcpp worker exited unexpectedly with status {status}");
            }
        }
    }

    Ok(())
}
