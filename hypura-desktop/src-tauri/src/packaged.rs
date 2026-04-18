use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Context;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::Engine;
use reqwest::blocking::Client as BlockingClient;
use serde::{Deserialize, Serialize};
use tauri::AppHandle;
use tauri::Emitter;
use tauri::State as TauriState;

#[allow(dead_code)]
#[path = "../../../src/compat_assets.rs"]
mod compat_assets;

use compat_assets::{
    AssetBootstrapReport, bootstrap_assets, default_asset_root, materialize_pending_asset_entry,
};

const DESKTOP_STATUS_EVENT: &str = "packaged-status";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackagedLaunchCmd {
    pub model_path: String,
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub asset_root: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStatus {
    pub launch_id: Option<String>,
    pub running: bool,
    pub base_url: Option<String>,
    pub lite_url: Option<String>,
    pub model_path: Option<String>,
    pub asset_root: Option<String>,
    pub embeddings_ready: bool,
    pub transcribe_ready: bool,
    pub tts_ready: bool,
    pub audio_ready: bool,
    pub pending_assets: Vec<String>,
    pub downloaded_assets: Vec<String>,
    pub last_error: Option<String>,
    pub status_text: String,
}

#[derive(Default)]
pub struct DesktopState {
    pub runtime: Option<ManagedRuntime>,
    pub status: RuntimeStatus,
}

pub struct ManagedRuntime {
    pub launch_id: String,
    pub request: PackagedLaunchCmd,
    pub child: Child,
    pub whisper_child: Option<Child>,
    pub bridge: Option<AudioBridgeHandle>,
}

pub struct AudioBridgeHandle {
    pub base_url: String,
    pub shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    pub thread: Option<std::thread::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
pub struct PreparedRuntime {
    pub report: AssetBootstrapReport,
    pub embeddings_model: Option<PathBuf>,
    pub transcribe: Option<TranscribeAssets>,
    pub tts: Option<TtsAssets>,
}

#[derive(Debug, Clone)]
pub struct TranscribeAssets {
    pub exe_path: PathBuf,
    pub model_path: PathBuf,
    pub model_id: String,
}

#[derive(Debug, Clone)]
pub struct TtsAssets {
    pub exe_path: PathBuf,
    pub model_path: PathBuf,
    pub voices_path: PathBuf,
    pub tokens_path: PathBuf,
    pub data_dir: PathBuf,
}

#[derive(Clone)]
struct AudioBridgeState {
    client: reqwest::Client,
    transcribe: Option<WhisperService>,
    tts: Option<TtsAssets>,
}

#[derive(Debug, Deserialize)]
struct KoboldTranscribeRequest {
    audio_data: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    langcode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct KoboldTtsRequest {
    input: String,
    #[serde(default)]
    voice: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiSpeechRequest {
    input: String,
    #[serde(default)]
    voice: Option<String>,
}

#[derive(Debug, Serialize)]
struct TranscribeResponse {
    text: String,
}

const KITTEN_VOICES: &[(&str, u32)] = &[
    ("expr-voice-2-m", 0),
    ("expr-voice-2-f", 1),
    ("expr-voice-3-m", 2),
    ("expr-voice-3-f", 3),
    ("expr-voice-4-m", 4),
    ("expr-voice-4-f", 5),
    ("expr-voice-5-m", 6),
    ("expr-voice-5-f", 7),
];

pub fn resolve_hypura_exe() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("HYPURA_EXE") {
        let candidate = PathBuf::from(path.trim());
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(exe_dir) = current_exe.parent() {
            for name in ["hypura.exe", "hypura"] {
                let candidate = exe_dir.join(name);
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }
    }
    Ok(PathBuf::from(if cfg!(windows) {
        "hypura.exe"
    } else {
        "hypura"
    }))
}

pub fn pick_asset_root(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let dir = app.dialog().file().blocking_pick_folder();
    Ok(dir.map(|value| value.to_string()))
}

pub fn get_runtime_status(
    state: TauriState<'_, Arc<Mutex<DesktopState>>>,
) -> Result<RuntimeStatus, String> {
    let guard = state.lock().map_err(|error| error.to_string())?;
    Ok(guard.status.clone())
}

pub fn stop_managed_runtime(
    state: TauriState<'_, Arc<Mutex<DesktopState>>>,
) -> Result<(), String> {
    let mut guard = state.lock().map_err(|error| error.to_string())?;
    stop_runtime_locked(&mut guard);
    guard.status.running = false;
    guard.status.status_text = "Stopped managed packaged runtime.".to_string();
    guard.status.pending_assets.clear();
    guard.status.launch_id = None;
    Ok(())
}

pub fn launch_packaged_koboldcpp(
    cmd: PackagedLaunchCmd,
    app: AppHandle,
    state: TauriState<'_, Arc<Mutex<DesktopState>>>,
) -> Result<String, String> {
    let asset_root = cmd
        .asset_root
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(default_asset_root);
    let manifest_path = resolve_manifest_path();
    let launch_id = uuid::Uuid::new_v4().simple().to_string();

    {
        let mut guard = state.lock().map_err(|error| error.to_string())?;
        start_runtime_locked(
            &mut guard,
            &launch_id,
            &cmd,
            &asset_root,
            manifest_path.as_deref(),
        )?;
        emit_status(&app, &guard.status);
    }

    let pending = {
        let guard = state.lock().map_err(|error| error.to_string())?;
        guard.status.pending_assets.clone()
    };
    if !pending.is_empty() {
        spawn_asset_bootstrap_thread(
            app,
            state.inner().clone(),
            launch_id,
            asset_root,
            manifest_path,
        );
    }

    Ok(format!("http://{}:{}", cmd.host, cmd.port))
}

fn resolve_manifest_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(exe_dir) = current_exe.parent() {
            candidates.push(exe_dir.join("resources").join("koboldcpp-assets.json"));
            candidates.push(exe_dir.join("koboldcpp-assets.json"));
        }
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("koboldcpp-assets.json"),
    );
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("docs")
            .join("compat")
            .join("koboldcpp-assets.json"),
    );
    candidates.into_iter().find(|path| path.exists())
}

fn stop_runtime_locked(guard: &mut DesktopState) {
    if let Some(mut runtime) = guard.runtime.take() {
        if let Some(bridge) = runtime.bridge.as_mut() {
            if let Some(shutdown_tx) = bridge.shutdown_tx.take() {
                let _ = shutdown_tx.send(());
            }
            if let Some(thread) = bridge.thread.take() {
                let _ = thread.join();
            }
        }
        if let Some(child) = runtime.whisper_child.as_mut() {
            stop_child(child);
        }
        stop_child(&mut runtime.child);
    }
}

fn stop_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn emit_status(app: &AppHandle, status: &RuntimeStatus) {
    let _ = app.emit(DESKTOP_STATUS_EVENT, status);
}

fn spawn_asset_bootstrap_thread(
    app: AppHandle,
    state: Arc<Mutex<DesktopState>>,
    launch_id: String,
    asset_root: PathBuf,
    manifest_path: Option<PathBuf>,
) {
    thread::spawn(move || {
        let pending_entries = match bootstrap_assets(&asset_root, manifest_path.as_deref()) {
            Ok(report) => report.pending,
            Err(error) => {
                update_status_error(
                    &app,
                    &state,
                    &launch_id,
                    format!("Asset manifest reload failed: {error}"),
                );
                return;
            }
        };

        for entry in pending_entries {
            update_status_message(
                &app,
                &state,
                &launch_id,
                format!("Downloading optional packaged asset {}...", entry.id),
            );
            match materialize_pending_asset_entry(&asset_root, &entry) {
                Ok(path) => update_download_success(&app, &state, &launch_id, &entry.id, &path),
                Err(error) => update_status_error(
                    &app,
                    &state,
                    &launch_id,
                    format!("Asset {} failed: {error}", entry.id),
                ),
            }
        }

        let request = {
            let guard = match state.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            let Some(runtime) = guard.runtime.as_ref() else {
                return;
            };
            if runtime.launch_id != launch_id {
                return;
            }
            runtime.request.clone()
        };

        let mut guard = match state.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        match start_runtime_locked(&mut guard, &launch_id, &request, &asset_root, manifest_path.as_deref()) {
            Ok(()) => {
                guard.status.status_text =
                    "Optional packaged assets became ready; restarted koboldcpp runtime.".to_string();
                emit_status(&app, &guard.status);
            }
            Err(error) => {
                guard.status.last_error = Some(error.clone());
                guard.status.status_text =
                    "Optional assets finished downloading, but the packaged runtime restart failed.".to_string();
                emit_status(&app, &guard.status);
            }
        }
    });
}

fn update_status_message(
    app: &AppHandle,
    state: &Arc<Mutex<DesktopState>>,
    launch_id: &str,
    message: String,
) {
    if let Ok(mut guard) = state.lock() {
        if guard.status.launch_id.as_deref() != Some(launch_id) {
            return;
        }
        guard.status.status_text = message;
        emit_status(app, &guard.status);
    }
}

fn update_status_error(
    app: &AppHandle,
    state: &Arc<Mutex<DesktopState>>,
    launch_id: &str,
    error: String,
) {
    if let Ok(mut guard) = state.lock() {
        if guard.status.launch_id.as_deref() != Some(launch_id) {
            return;
        }
        guard.status.last_error = Some(error.clone());
        guard.status.status_text = error;
        emit_status(app, &guard.status);
    }
}

fn update_download_success(
    app: &AppHandle,
    state: &Arc<Mutex<DesktopState>>,
    launch_id: &str,
    entry_id: &str,
    path: &Path,
) {
    if let Ok(mut guard) = state.lock() {
        if guard.status.launch_id.as_deref() != Some(launch_id) {
            return;
        }
        if !guard
            .status
            .downloaded_assets
            .iter()
            .any(|existing| existing == entry_id)
        {
            guard.status.downloaded_assets.push(entry_id.to_string());
        }
        guard.status.pending_assets.retain(|value| value != entry_id);
        guard.status.status_text =
            format!("Downloaded optional packaged asset {} to {}.", entry_id, path.display());
        emit_status(app, &guard.status);
    }
}

fn model_id_from_path(path: &Path) -> String {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bin");
    file_name
        .strip_suffix(".bin")
        .or_else(|| file_name.strip_suffix(".gguf"))
        .or_else(|| file_name.strip_suffix(".onnx"))
        .unwrap_or(file_name)
        .to_string()
}

#[derive(Debug, Clone)]
struct WhisperService {
    base_url: String,
    model_id: String,
}

struct StartedWhisperService {
    service: WhisperService,
    child: Child,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoWhisperLaunchMode {
    Run,
    Server,
}

fn start_go_whisper_server(config: &TranscribeAssets) -> anyhow::Result<StartedWhisperService> {
    let launch_mode = detect_go_whisper_launch_mode(config)?;
    let port = reserve_loopback_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let model_dir = config
        .model_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut command = Command::new(&config.exe_path);
    match launch_mode {
        GoWhisperLaunchMode::Run => {
            command
                .arg("run")
                .arg("--http.addr")
                .arg(format!("127.0.0.1:{port}"))
                .arg("--models")
                .arg(&model_dir);
        }
        GoWhisperLaunchMode::Server => {
            command
                .arg("server")
                .arg("--listen")
                .arg(format!("127.0.0.1:{port}"));
        }
    }
    command
        .env("GOWHISPER_DIR", &model_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = command.spawn().with_context(|| {
        format!(
            "starting STT helper {} with model dir {}",
            config.exe_path.display(),
            model_dir.display()
        )
    })?;
    if let Err(error) = wait_for_http_ready(&format!("{base_url}/api/whisper/model")) {
        stop_child(&mut child);
        return Err(error);
    }
    Ok(StartedWhisperService {
        service: WhisperService {
            base_url,
            model_id: config.model_id.clone(),
        },
        child,
    })
}

fn detect_go_whisper_launch_mode(config: &TranscribeAssets) -> anyhow::Result<GoWhisperLaunchMode> {
    let mut command = Command::new(&config.exe_path);
    command.arg("--help");
    let help = capture_command_text_with_timeout(
        command,
        Duration::from_secs(5),
        &format!("probing STT helper {}", config.exe_path.display()),
    )?;
    if help_lists_go_whisper_command(&help, "run") {
        return Ok(GoWhisperLaunchMode::Run);
    }
    if help_lists_go_whisper_command(&help, "server") {
        return Ok(GoWhisperLaunchMode::Server);
    }
    anyhow::bail!(
        "{} does not expose a local server command. The v0.0.39 Windows release asset is client-only and packaged STT remains disabled.",
        config.exe_path.display()
    );
}

fn help_lists_go_whisper_command(help: &str, command: &str) -> bool {
    help.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed == command
            || trimmed.starts_with(&format!("{command} "))
            || trimmed.starts_with(&format!("{command}\t"))
    })
}

fn wait_for_http_ready(url: &str) -> anyhow::Result<()> {
    let client = BlockingClient::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        if let Ok(response) = client.get(url).send() {
            if response.status().is_success() {
                return Ok(());
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    anyhow::bail!("service did not become ready at {url}")
}

fn reserve_loopback_port() -> anyhow::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

fn start_audio_bridge(
    transcribe: Option<WhisperService>,
    tts: Option<TtsAssets>,
) -> anyhow::Result<Option<AudioBridgeHandle>> {
    if transcribe.is_none() && tts.is_none() {
        return Ok(None);
    }

    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    listener.set_nonblocking(true)?;
    let addr = listener.local_addr()?;
    let base_url = format!("http://127.0.0.1:{}", addr.port());
    let state = AudioBridgeState {
        client: reqwest::Client::new(),
        transcribe,
        tts,
    };
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let thread = thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(_) => return,
        };
        runtime.block_on(async move {
            let listener = match tokio::net::TcpListener::from_std(listener) {
                Ok(listener) => listener,
                Err(_) => return,
            };
            let app = Router::new()
                .route("/api/extra/transcribe", post(kobold_transcribe_handler))
                .route("/v1/audio/transcriptions", post(openai_transcribe_handler))
                .route("/api/extra/tts", post(kobold_tts_handler))
                .route("/v1/audio/speech", post(openai_tts_handler))
                .route("/speakers_list", get(speakers_list_handler))
                .layer(DefaultBodyLimit::disable())
                .with_state(state);
            let shutdown = async {
                let _ = shutdown_rx.await;
            };
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(shutdown)
                .await;
        });
    });
    Ok(Some(AudioBridgeHandle {
        base_url,
        shutdown_tx: Some(shutdown_tx),
        thread: Some(thread),
    }))
}

async fn kobold_transcribe_handler(
    State(state): State<AudioBridgeState>,
    Json(request): Json<KoboldTranscribeRequest>,
) -> Response {
    let Some(config) = state.transcribe.as_ref() else {
        return unavailable_json("transcribe").into_response();
    };
    let bytes = match decode_data_url(&request.audio_data) {
        Ok(bytes) => bytes,
        Err(error) => return bad_request_json(error.to_string()).into_response(),
    };
    match run_whisper_transcribe(
        &state.client,
        config,
        bytes,
        "input.wav",
        request.prompt.as_deref(),
        request.langcode.as_deref(),
    )
    .await
    {
        Ok(text) => Json(TranscribeResponse { text }).into_response(),
        Err(error) => bad_gateway_json(error.to_string()).into_response(),
    }
}

async fn openai_transcribe_handler(
    State(state): State<AudioBridgeState>,
    mut multipart: Multipart,
) -> Response {
    let Some(config) = state.transcribe.as_ref() else {
        return unavailable_json("transcribe").into_response();
    };
    let mut audio = None;
    let mut filename = "audio.wav".to_string();
    let mut prompt = None;
    let mut language = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" | "audio" => {
                if let Some(value) = field.file_name() {
                    filename = value.to_string();
                }
                match field.bytes().await {
                    Ok(bytes) => audio = Some(bytes.to_vec()),
                    Err(error) => return bad_request_json(error.to_string()).into_response(),
                }
            }
            "prompt" => match field.text().await {
                Ok(text) => prompt = Some(text),
                Err(error) => return bad_request_json(error.to_string()).into_response(),
            },
            "language" => match field.text().await {
                Ok(text) => language = Some(text),
                Err(error) => return bad_request_json(error.to_string()).into_response(),
            },
            _ => {}
        }
    }
    let Some(audio) = audio else {
        return bad_request_json("missing file form field".to_string()).into_response();
    };
    match run_whisper_transcribe(
        &state.client,
        config,
        audio,
        &filename,
        prompt.as_deref(),
        language.as_deref(),
    )
    .await
    {
        Ok(text) => Json(TranscribeResponse { text }).into_response(),
        Err(error) => bad_gateway_json(error.to_string()).into_response(),
    }
}

async fn kobold_tts_handler(
    State(state): State<AudioBridgeState>,
    Json(request): Json<KoboldTtsRequest>,
) -> Response {
    let Some(config) = state.tts.as_ref() else {
        return unavailable_json("tts").into_response();
    };
    match generate_tts_audio(config, &request.input, request.voice.as_deref()) {
        Ok(bytes) => wav_response(bytes),
        Err(error) => bad_gateway_json(error.to_string()).into_response(),
    }
}

async fn openai_tts_handler(
    State(state): State<AudioBridgeState>,
    Json(request): Json<OpenAiSpeechRequest>,
) -> Response {
    let Some(config) = state.tts.as_ref() else {
        return unavailable_json("tts").into_response();
    };
    match generate_tts_audio(config, &request.input, request.voice.as_deref()) {
        Ok(bytes) => wav_response(bytes),
        Err(error) => bad_gateway_json(error.to_string()).into_response(),
    }
}

async fn speakers_list_handler(State(state): State<AudioBridgeState>) -> Response {
    if state.tts.is_none() {
        return unavailable_json("speakers_list").into_response();
    }
    let voices: Vec<_> = KITTEN_VOICES
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    Json(voices).into_response()
}

async fn run_whisper_transcribe(
    client: &reqwest::Client,
    config: &WhisperService,
    audio: Vec<u8>,
    filename: &str,
    prompt: Option<&str>,
    language: Option<&str>,
) -> anyhow::Result<String> {
    let audio_part = reqwest::multipart::Part::bytes(audio)
        .file_name(filename.to_string())
        .mime_str("audio/wav")?;
    let mut form = reqwest::multipart::Form::new()
        .part("audio", audio_part)
        .text("model", config.model_id.clone())
        .text("filename", filename.to_string());
    if let Some(prompt) = prompt.filter(|value| !value.trim().is_empty()) {
        form = form.text("prompt", prompt.to_string());
    }
    if let Some(language) = language.filter(|value| {
        let value = value.trim();
        !value.is_empty() && !value.eq_ignore_ascii_case("auto")
    }) {
        form = form.text("language", language.to_string());
    }
    let response = client
        .post(format!("{}{}", config.base_url, "/api/whisper/transcribe"))
        .header(header::ACCEPT, "text/plain")
        .multipart(form)
        .send()
        .await
        .with_context(|| format!("calling whisper helper at {}", config.base_url))?;
    anyhow::ensure!(
        response.status().is_success(),
        "whisper helper returned status {}",
        response.status()
    );
    Ok(response.text().await?.trim().to_string())
}

fn generate_tts_audio(
    config: &TtsAssets,
    text: &str,
    voice: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    generate_tts_audio_with_timeout(config, text, voice, Duration::from_secs(45))
}

fn generate_tts_audio_with_timeout(
    config: &TtsAssets,
    text: &str,
    voice: Option<&str>,
    timeout: Duration,
) -> anyhow::Result<Vec<u8>> {
    let sid = resolve_tts_voice_id(voice);
    let output_path = std::env::temp_dir().join(format!(
        "hypura-kcpp-tts-{}.wav",
        uuid::Uuid::new_v4().simple()
    ));
    let mut command = Command::new(&config.exe_path);
    command
        .arg(format!("--kitten-model={}", config.model_path.display()))
        .arg(format!("--kitten-voices={}", config.voices_path.display()))
        .arg(format!("--kitten-tokens={}", config.tokens_path.display()))
        .arg(format!("--kitten-data-dir={}", config.data_dir.display()))
        .arg(format!("--sid={sid}"))
        .arg(format!("--output-filename={}", output_path.display()))
        .arg(text);
    let output = run_command_capture_with_timeout(
        &mut command,
        timeout,
        &format!("spawning TTS helper {}", config.exe_path.display()),
    )?;
    anyhow::ensure!(
        output.status.success(),
        "TTS helper failed: {}",
        summarize_process_output(&output)
    );
    if output_path.exists() {
        let bytes = std::fs::read(&output_path)
            .with_context(|| format!("reading generated TTS audio {}", output_path.display()))?;
        let _ = std::fs::remove_file(&output_path);
        return Ok(bytes);
    }
    if looks_like_wav_bytes(&output.stdout) {
        return Ok(output.stdout);
    }
    anyhow::bail!(
        "TTS helper completed without producing a WAV file. {}",
        summarize_process_output(&output)
    );
}

fn resolve_tts_voice_id(voice: Option<&str>) -> u32 {
    let Some(voice) = voice.map(str::trim).filter(|value| !value.is_empty()) else {
        return 1;
    };
    if let Ok(numeric) = voice.parse::<u32>() {
        if numeric <= 7 {
            return numeric;
        }
    }
    KITTEN_VOICES
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(voice))
        .map(|(_, sid)| *sid)
        .unwrap_or_else(|| match voice.to_ascii_lowercase().as_str() {
            "alloy" | "ash" | "echo" => 0,
            "nova" | "shimmer" | "sage" | "fable" | "coral" => 1,
            "onyx" => 6,
            _ => 1,
        })
}

fn decode_data_url(data_url: &str) -> anyhow::Result<Vec<u8>> {
    let (_, encoded) = data_url
        .split_once(',')
        .ok_or_else(|| anyhow::anyhow!("invalid data URL payload"))?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded.as_bytes())
        .context("decoding base64 audio payload")
}

fn probe_tts_assets(config: &TtsAssets) -> anyhow::Result<()> {
    let bytes = generate_tts_audio_with_timeout(
        config,
        "Packaged runtime probe.",
        Some("alloy"),
        Duration::from_secs(15),
    )?;
    anyhow::ensure!(
        looks_like_wav_bytes(&bytes),
        "TTS probe did not yield a WAV payload"
    );
    Ok(())
}

fn looks_like_wav_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

fn capture_command_text_with_timeout(
    mut command: Command,
    timeout: Duration,
    description: &str,
) -> anyhow::Result<String> {
    let output = run_command_capture_with_timeout(&mut command, timeout, description)?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(if stderr.trim().is_empty() {
        stdout.into_owned()
    } else if stdout.trim().is_empty() {
        stderr.into_owned()
    } else {
        format!("{stdout}\n{stderr}")
    })
}

fn run_command_capture_with_timeout(
    command: &mut Command,
    timeout: Duration,
    description: &str,
) -> anyhow::Result<Output> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| description.to_string())?;
    let deadline = Instant::now() + timeout;
    loop {
        if child.try_wait()?.is_some() {
            return child
                .wait_with_output()
                .with_context(|| description.to_string());
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let captured = child.wait_with_output().ok();
            let detail = captured
                .as_ref()
                .map(summarize_process_output)
                .unwrap_or_else(|| "no stdout/stderr captured".to_string());
            anyhow::bail!("{description} timed out after {:?}. {detail}", timeout);
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn summarize_process_output(output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => "no stdout/stderr captured".to_string(),
        (false, true) => format!("stdout: {stdout}"),
        (true, false) => format!("stderr: {stderr}"),
        (false, false) => format!("stdout: {stdout}; stderr: {stderr}"),
    }
}

fn remember_runtime_error(guard: &mut DesktopState, message: String) {
    match guard.status.last_error.as_mut() {
        Some(existing) => {
            if !existing.contains(&message) {
                existing.push('\n');
                existing.push_str(&message);
            }
        }
        None => guard.status.last_error = Some(message),
    }
}

fn prepare_runtime_assets(
    asset_root: &Path,
    manifest_path: Option<&Path>,
) -> anyhow::Result<PreparedRuntime> {
    let report = bootstrap_assets(asset_root, manifest_path)?;
    let embeddings_model = report.ready.get("embeddings_model").cloned();
    let transcribe = match (
        report.ready.get("transcribe_exe"),
        report.ready.get("transcribe_model"),
    ) {
        (Some(exe_path), Some(model_path)) => Some(TranscribeAssets {
            exe_path: exe_path.clone(),
            model_path: model_path.clone(),
            model_id: model_id_from_path(model_path),
        }),
        _ => None,
    };
    let tts = match (
        report.ready.get("tts_exe"),
        report.ready.get("tts_model"),
        report.ready.get("tts_voice"),
        report.ready.get("tts_tokens"),
        report.ready.get("tts_data"),
    ) {
        (Some(exe_path), Some(model_path), Some(voices_path), Some(tokens_path), Some(data_dir)) => Some(TtsAssets {
            exe_path: exe_path.clone(),
            model_path: model_path.clone(),
            voices_path: voices_path.clone(),
            tokens_path: tokens_path.clone(),
            data_dir: data_dir.clone(),
        }),
        _ => None,
    };
    Ok(PreparedRuntime {
        report,
        embeddings_model,
        transcribe,
        tts,
    })
}

fn start_runtime_locked(
    guard: &mut DesktopState,
    launch_id: &str,
    request: &PackagedLaunchCmd,
    asset_root: &Path,
    manifest_path: Option<&Path>,
) -> Result<(), String> {
    stop_runtime_locked(guard);

    let prepared = prepare_runtime_assets(asset_root, manifest_path)
        .map_err(|error| format!("Bootstrap failed: {error}"))?;
    let mut whisper = match prepared.transcribe.as_ref() {
        Some(config) => match start_go_whisper_server(config) {
            Ok(service) => Some(service),
            Err(error) => {
                remember_runtime_error(guard, format!("STT helper launch failed: {error}"));
                None
            }
        },
        None => None,
    };
    let whisper_ready = whisper.is_some();
    let whisper_service = whisper.as_ref().map(|value| value.service.clone());
    let validated_tts = match prepared.tts.as_ref() {
        Some(config) => match probe_tts_assets(config) {
            Ok(()) => Some(config.clone()),
            Err(error) => {
                remember_runtime_error(guard, format!("TTS helper probe failed: {error}"));
                None
            }
        },
        None => None,
    };
    let mut bridge = match start_audio_bridge(whisper_service, validated_tts.clone()) {
        Ok(bridge) => bridge,
        Err(error) => {
            if let Some(whisper) = whisper.as_mut() {
                stop_child(&mut whisper.child);
            }
            return Err(format!("Audio bridge launch failed: {error}"));
        }
    };
    let bridge_ready = bridge.is_some();
    let tts_ready = validated_tts.is_some() && bridge_ready;
    let audio_ready = whisper_ready || tts_ready;
    let child = match spawn_hypura_koboldcpp(
        request,
        asset_root,
        prepared.embeddings_model.as_deref(),
        bridge
            .as_ref()
            .filter(|_| whisper_ready)
            .map(|handle| handle.base_url.as_str()),
        bridge
            .as_ref()
            .filter(|_| prepared.tts.is_some())
            .map(|handle| handle.base_url.as_str()),
    ) {
        Ok(child) => child,
        Err(error) => {
            if let Some(mut handle) = bridge.take() {
                if let Some(shutdown_tx) = handle.shutdown_tx.take() {
                    let _ = shutdown_tx.send(());
                }
                if let Some(thread) = handle.thread.take() {
                    let _ = thread.join();
                }
            }
            if let Some(whisper) = whisper.as_mut() {
                stop_child(&mut whisper.child);
            }
            return Err(error);
        }
    };

    guard.runtime = Some(ManagedRuntime {
        launch_id: launch_id.to_string(),
        request: request.clone(),
        child,
        whisper_child: whisper.map(|value| value.child),
        bridge,
    });

    guard.status = RuntimeStatus {
        launch_id: Some(launch_id.to_string()),
        running: true,
        base_url: Some(format!("http://{}:{}", request.host, request.port)),
        lite_url: Some(format!("http://{}:{}/kobold-lite", request.host, request.port)),
        model_path: Some(request.model_path.clone()),
        asset_root: Some(asset_root.to_string_lossy().to_string()),
        embeddings_ready: prepared.embeddings_model.is_some(),
        transcribe_ready: whisper_ready,
        tts_ready,
        audio_ready,
        pending_assets: prepared.report.pending.iter().map(|entry| entry.id.clone()).collect(),
        downloaded_assets: prepared.report.downloaded.clone(),
        last_error: guard.status.last_error.clone(),
        status_text: if prepared.report.pending.is_empty() && guard.status.last_error.is_none() {
            "Packaged koboldcpp runtime started.".to_string()
        } else if prepared.report.pending.is_empty() {
            "Packaged koboldcpp runtime started with some optional packaged features disabled.".to_string()
        } else {
            "Packaged koboldcpp runtime started in text-only mode while optional assets download in the background.".to_string()
        },
    };
    Ok(())
}

fn spawn_hypura_koboldcpp(
    request: &PackagedLaunchCmd,
    asset_root: &Path,
    embeddings_model: Option<&Path>,
    transcribe_url: Option<&str>,
    tts_url: Option<&str>,
) -> Result<Child, String> {
    let exe = resolve_hypura_exe()?;
    let mut command = Command::new(&exe);
    command
        .arg("koboldcpp")
        .arg(&request.model_path)
        .arg("--host")
        .arg(&request.host)
        .arg("--port")
        .arg(request.port.to_string())
        .arg("--asset-root")
        .arg(asset_root.to_string_lossy().to_string())
        .arg("--no-show-gui")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(path) = embeddings_model {
        command.arg("--embeddings-model").arg(path);
    }
    if let Some(url) = transcribe_url {
        command.env("HYPURA_KCPP_TRANSCRIBE_URL", url);
    }
    if let Some(url) = tts_url {
        command.env("HYPURA_KCPP_TTS_URL", url);
    }
    command
        .spawn()
        .map_err(|error| format!("spawn packaged hypura koboldcpp: {error}"))
}

fn wav_response(bytes: Vec<u8>) -> Response {
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    response
}

fn bad_request_json(message: String) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "error": message }))
}

fn bad_gateway_json(message: String) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "error": message }))
}

fn unavailable_json(feature: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": format!("{feature} is not available in the packaged desktop runtime"),
            "feature": feature,
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_voice_aliases_map_to_known_kitten_sids() {
        assert_eq!(resolve_tts_voice_id(Some("alloy")), 0);
        assert_eq!(resolve_tts_voice_id(Some("nova")), 1);
        assert_eq!(resolve_tts_voice_id(Some("onyx")), 6);
        assert_eq!(resolve_tts_voice_id(Some("expr-voice-5-f")), 7);
    }

    #[test]
    fn model_id_strips_common_local_suffixes() {
        assert_eq!(
            model_id_from_path(Path::new("ggml-base.en.bin")),
            "ggml-base.en"
        );
        assert_eq!(
            model_id_from_path(Path::new("embeddings.gguf")),
            "embeddings"
        );
    }

    #[test]
    fn go_whisper_help_requires_a_server_command() {
        let help = r#"
Usage: gowhisper-windows-amd64.exe <command> [flags]

MODEL
  models            List models.
  model             Get model.

TRANSCRIBE & TRANSLATE
  transcribe    Transcribe audio file.
  translate     Translate audio file to English.
"#;
        assert!(!help_lists_go_whisper_command(help, "run"));
        assert!(!help_lists_go_whisper_command(help, "server"));
    }

    #[test]
    fn wav_magic_probe_accepts_stdout_fallback() {
        assert!(looks_like_wav_bytes(b"RIFFxxxxWAVEfmt "));
        assert!(!looks_like_wav_bytes(b"not-a-wave"));
    }
}
