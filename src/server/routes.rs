use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;
use std::{fs, path::PathBuf};

use axum::body::{Body, Bytes};
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, Method, Request, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::Html;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::{mpsc, oneshot};

use crate::compute::inference::{
    CompatRuntimeSession, ContextStateSnapshot, GenerateFromLoadedParams, GenerationResult,
    LlamaTurboquantCliBridge, LoadedModel,
};
use crate::council::store::{
    CouncilInputKind, CouncilRequestRecord, CouncilStore, PersistedCouncilRecord,
    StoredCouncilRecord,
};
use crate::council::{
    AhaThresholds, AnswerCouncilConfig, CouncilRuntimeConfig, CouncilUrtDescriptor,
    EmbeddedKaController, KaController, KaGateConfig, NoSafetyPenalty,
    prepare_embedded_ka_controller,
};
use crate::model::file_identity::{GuardedModelFile, open_read_guard};
use crate::model::gguf::GgufFile;
use crate::model::turboquant_sidecar::{
    GgufTurboQuantConfig, GgufUrtConfig, ResolvedTurboQuantConfig, TurboQuantMode,
};
use crate::scheduler::types::{CouncilExecutionMode, CouncilParallelism, ResidencyPolicyConfig};
use crate::server::chat::format_chat_prompt;
use crate::server::compat::{self, CompatFeatureFlags, CompatPerfState};
use crate::server::compat_storage::{CompatStorage, LauncherProfile, RuntimeStateSlotMetadata};
use crate::server::ollama_types::*;
use crate::server::streaming;
use crate::server::supervisor::{CompatControlPlaneClient, CompatSupervisorCommand};
use crate::telemetry::metrics::{TelemetryEmitter, TelemetryEvent};
use crate::urt::{
    RepresentationId, RepresentationKind, UrtAssessment, UrtObservation, UrtRegistry,
};

const VENDORED_KOBOLD_LITE_INDEX: &str = include_str!("../../vendor/kobold-lite/index.html");
const VENDORED_KOBOLDCPP_API_HTML: &str =
    include_str!("../../vendor/kobold-lite/koboldcpp_api.html");
const VENDORED_KOBOLDCPP_API_JSON: &str =
    include_str!("../../vendor/kobold-lite/koboldcpp_api.json");
const VENDORED_KOBOLD_LITE_MANIFEST: &str = include_str!("../../vendor/kobold-lite/manifest.json");
const VENDORED_KOBOLD_LITE_SW: &str = include_str!("../../vendor/kobold-lite/sw.js");
const VENDORED_KOBOLD_LITE_NIKO_PNG: &[u8] = include_bytes!("../../vendor/kobold-lite/niko.png");

pub struct AppState {
    pub loaded_model: Arc<std::sync::Mutex<LoadedModel>>,
    pub model_name: Arc<Mutex<String>>,
    pub model_name_is_explicit_alias: bool,
    pub model_path: Arc<Mutex<PathBuf>>,
    pub model_sha256: Arc<std::sync::Mutex<Option<String>>>,
    pub gguf_info: Arc<Mutex<GgufInfo>>,
    pub model_dir: PathBuf,
    pub default_context: Arc<AtomicU32>,
    pub load_duration_ns: u64,
    pub telemetry: Arc<TelemetryEmitter>,
    pub council_store: Arc<CouncilStore>,
    pub urt_registry: Arc<Mutex<UrtRegistry>>,
    pub turboquant: ResolvedTurboQuantConfig,
    /// CLI `hypura serve` TurboQuant mode — reused by hot model switch for parity.
    pub serve_turboquant_mode: TurboQuantMode,
    pub serve_turboquant_config_path: Option<PathBuf>,
    pub serve_llama_bridge: LlamaTurboquantCliBridge,
    pub serve_residency_policy: ResidencyPolicyConfig,
    pub serve_tq_allow_exact_fallback: bool,
    pub active_cancel: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    pub generation_in_progress: Arc<AtomicBool>,
    pub gui_presets: Arc<Mutex<HashMap<String, GuiPresetItem>>>,
    pub gui_history: Arc<Mutex<VecDeque<GuiHistoryItem>>>,
    pub gui_events: Arc<Mutex<VecDeque<GuiEventItem>>>,
    pub ui_theme: Arc<Mutex<String>>,
    pub compat_storage: Option<Arc<CompatStorage>>,
    pub compat_started_at: Instant,
    pub compat_default_max_length: Arc<AtomicU32>,
    pub compat_features: Arc<Mutex<CompatFeatureFlags>>,
    pub compat_perf: Arc<Mutex<CompatPerfState>>,
    pub compat_control_client: Option<CompatControlPlaneClient>,
    pub compat_session: Option<Arc<Mutex<CompatRuntimeSession>>>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(health_handler))
        .route("/api", get(kobold_api_docs_handler))
        .route("/koboldcpp_api.html", get(kobold_api_docs_handler))
        .route("/koboldcpp_api.json", get(kobold_api_spec_handler))
        .route("/manifest.json", get(kobold_lite_manifest_handler))
        .route("/sw.js", get(kobold_lite_service_worker_handler))
        .route("/niko.png", get(kobold_lite_niko_handler))
        .route("/api/version", get(version_handler))
        .route("/api/extra/version", get(kobold_extra_version_handler))
        .route("/api/extra/perf", get(kobold_perf_handler))
        .route("/api/extra/tokencount", post(kobold_token_count_handler))
        .route("/api/extra/tokenize", post(kobold_token_count_handler))
        .route("/api/extra/preloadstory", get(kobold_preload_story_handler))
        .route("/api/extra/data/list", post(kobold_savedata_list_handler))
        .route("/api/extra/data/save", post(kobold_savedata_save_handler))
        .route("/api/extra/data/load", post(kobold_savedata_load_handler))
        .route("/api/extra/websearch", post(kobold_websearch_handler))
        .route("/api/extra/embeddings", post(openai_embeddings_handler))
        .route(
            "/api/admin/list_options",
            get(kobold_admin_list_options_handler),
        )
        .route(
            "/api/admin/reload_config",
            post(kobold_admin_reload_config_handler),
        )
        .route(
            "/api/admin/save_state",
            post(kobold_admin_save_state_handler),
        )
        .route(
            "/api/admin/load_state",
            post(kobold_admin_load_state_handler),
        )
        .route("/api/v1/info/version", get(kobold_api_info_version_handler))
        .route("/api/v1/config/max_length", get(kobold_max_length_handler))
        .route(
            "/api/v1/config/max_context_length",
            get(kobold_max_context_length_handler),
        )
        .route("/api/tags", get(tags_handler))
        .route("/api/show", post(show_handler))
        .route("/kobold-lite", get(kobold_lite_gui_handler))
        .route("/api/extra/models", get(models_handler))
        .route("/api/extra/model/switch", post(model_switch_handler))
        .route("/api/extra/presets/list", get(gui_presets_list_handler))
        .route("/api/extra/presets/save", post(gui_presets_save_handler))
        .route(
            "/api/extra/presets/delete",
            post(gui_presets_delete_handler),
        )
        .route("/api/extra/history", get(gui_history_handler))
        .route("/api/extra/events", get(gui_events_handler))
        .route("/api/extra/ui-theme", get(ui_theme_get_handler))
        .route("/api/extra/ui-theme", post(ui_theme_set_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/chat", post(chat_handler))
        .route(
            "/api/extra/triality/council",
            post(triality_council_handler),
        )
        .route("/v1/triality/council", post(triality_council_handler))
        .route(
            "/api/extra/triality/council/:id",
            get(triality_council_get_handler),
        )
        .route("/api/extra/triality/events", get(triality_events_handler))
        .route("/api/v1/model", get(kobold_model_handler))
        .route("/api/v1/generate", post(kobold_generate_handler))
        .route("/v1/completions", post(openai_completions_handler))
        .route(
            "/v1/chat/completions",
            post(openai_chat_completions_handler),
        )
        .route(
            "/lcpp/v1/chat/completions",
            post(openai_chat_completions_handler),
        )
        .route("/v1/embeddings", post(openai_embeddings_handler))
        .route("/sdapi/v1/txt2img", post(txt2img_proxy_handler))
        .route("/sdapi/v1/img2img", post(img2img_proxy_handler))
        .route("/sdapi/v1/interrogate", post(interrogate_proxy_handler))
        .route("/sdapi/v1/upscale", post(upscale_proxy_handler))
        .route("/sdapi/v1/options", get(sd_options_proxy_handler))
        .route("/sdapi/v1/sd-models", get(sd_models_proxy_handler))
        .route("/api/extra/transcribe", post(transcribe_proxy_handler))
        .route(
            "/v1/audio/transcriptions",
            post(audio_transcriptions_proxy_handler),
        )
        .route("/api/extra/tts", post(tts_proxy_handler))
        .route("/v1/audio/speech", post(audio_speech_proxy_handler))
        .route("/speakers_list", get(speakers_list_proxy_handler))
        .route(
            "/api/extra/generate/stream",
            post(kobold_generate_stream_handler),
        )
        .route(
            "/api/extra/generate/check",
            get(kobold_generate_check_handler),
        )
        .route("/api/extra/abort", post(kobold_abort_handler))
        .route(
            "/api/extra/true_max_context_length",
            get(kobold_true_max_context_length_handler),
        )
        .layer(middleware::from_fn(enforce_optional_api_key))
        .with_state(state)
}

async fn enforce_optional_api_key(request: Request<Body>, next: Next) -> Response {
    let path = request.uri().path();
    if matches!(
        path,
        "/" | "/kobold-lite"
            | "/api"
            | "/koboldcpp_api.html"
            | "/koboldcpp_api.json"
            | "/manifest.json"
            | "/sw.js"
            | "/niko.png"
    ) {
        return next.run(request).await;
    }
    let Ok(expected) = std::env::var("HYPURA_API_KEY") else {
        return next.run(request).await;
    };
    let expected = expected.trim();
    if expected.is_empty() {
        return next.run(request).await;
    }
    let bearer_ok = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer ").map(str::trim))
        .map(|t| t == expected)
        .unwrap_or(false);
    let xkey_ok = request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|t| t.trim() == expected)
        .unwrap_or(false);
    if bearer_ok || xkey_ok {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "invalid or missing API key" })),
        )
            .into_response()
    }
}

#[derive(Debug)]
struct CouncilApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl CouncilApiError {
    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }

    fn unsupported(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            code,
            message: message.into(),
        }
    }

    fn conflict(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code,
            message: message.into(),
        }
    }

    fn internal(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code,
            message: message.into(),
        }
    }
}

impl IntoResponse for CouncilApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": {
                    "code": self.code,
                    "message": self.message,
                }
            })),
        )
            .into_response()
    }
}

struct ValidatedCouncilRequest {
    request_id: String,
    requested_model: Option<String>,
    prompt: String,
    input_kind: CouncilInputKind,
    message_count: Option<usize>,
    sampling: crate::compute::ffi::SamplingParams,
    parallelism: CouncilParallelism,
    aha: bool,
    trace: bool,
    stream: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveModelIdentity {
    public_alias: String,
    physical_name: String,
    path: PathBuf,
    sha256: Option<String>,
}

struct LockedModelIdentity<'a, T> {
    loaded: MutexGuard<'a, T>,
    identity: ActiveModelIdentity,
}

fn lock_model_identity<'a, T>(
    loaded: &'a Mutex<T>,
    name: &Mutex<String>,
    path: &Mutex<PathBuf>,
    sha256: &Mutex<Option<String>>,
    physical_name: impl FnOnce(&T) -> String,
) -> Result<LockedModelIdentity<'a, T>, &'static str> {
    lock_model_identity_after_loaded(loaded, name, path, sha256, physical_name, || {})
}

fn lock_model_identity_after_loaded<'a, T>(
    loaded: &'a Mutex<T>,
    name: &Mutex<String>,
    path: &Mutex<PathBuf>,
    sha256: &Mutex<Option<String>>,
    physical_name: impl FnOnce(&T) -> String,
    after_loaded: impl FnOnce(),
) -> Result<LockedModelIdentity<'a, T>, &'static str> {
    let loaded = loaded.lock().map_err(|_| "loaded model lock poisoned")?;
    let physical_name = physical_name(&loaded);
    after_loaded();
    let public_alias = name.lock().map_err(|_| "model name lock poisoned")?.clone();
    let path = path.lock().map_err(|_| "model path lock poisoned")?.clone();
    let sha256 = sha256
        .lock()
        .map_err(|_| "model hash lock poisoned")?
        .clone();
    Ok(LockedModelIdentity {
        loaded,
        identity: ActiveModelIdentity {
            public_alias,
            physical_name,
            path,
            sha256,
        },
    })
}

async fn triality_council_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TrialityCouncilRequest>,
) -> Response {
    let validated = match validate_council_request(&state, request) {
        Ok(value) => value,
        Err(error) => return error.into_response(),
    };
    let stream = validated.stream;
    match execute_triality_council(state, validated).await {
        Ok(response) => triality_council_response(response, stream),
        Err(error) => error.into_response(),
    }
}

async fn triality_council_get_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(request_id): AxumPath<String>,
) -> Response {
    triality_council_get_from_store(state.council_store.clone(), request_id).await
}

async fn triality_council_get_from_store(store: Arc<CouncilStore>, request_id: String) -> Response {
    let stored = match tokio::task::spawn_blocking(move || store.read(&request_id)).await {
        Ok(Ok(Some(record))) => record,
        Ok(Ok(None)) => {
            return CouncilApiError {
                status: StatusCode::NOT_FOUND,
                code: "council_result_not_found",
                message: "Council result was not found".to_string(),
            }
            .into_response();
        }
        Ok(Err(error)) => {
            return CouncilApiError::bad_request("invalid_council_result_id", error.to_string())
                .into_response();
        }
        Err(error) => {
            return CouncilApiError::internal(
                "council_store_join_failed",
                format!("Council store read task failed: {error}"),
            )
            .into_response();
        }
    };
    Json(stored_council_response(stored)).into_response()
}

async fn triality_events_handler(State(state): State<Arc<AppState>>) -> Response {
    triality_events_from_emitter(state.telemetry.clone()).await
}

async fn triality_events_from_emitter(telemetry: Arc<TelemetryEmitter>) -> Response {
    let mut events = telemetry.subscribe();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);
    tokio::spawn(async move {
        loop {
            let event = match events.recv().await {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
            };
            if !is_triality_event(&event) {
                continue;
            }
            let Ok(payload) = serde_json::to_string(&event) else {
                continue;
            };
            if tx.send(Ok(format!("data: {payload}\n\n"))).await.is_err() {
                return;
            }
        }
    });
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)),
    )
        .into_response()
}

fn is_triality_event(event: &TelemetryEvent) -> bool {
    matches!(
        event,
        TelemetryEvent::TrialityBranchCompleted { .. }
            | TelemetryEvent::TrialityConsensusCompleted { .. }
            | TelemetryEvent::TrialityUrtChecked { .. }
            | TelemetryEvent::TrialityAha { .. }
    )
}

fn validate_council_request(
    state: &Arc<AppState>,
    request: TrialityCouncilRequest,
) -> Result<ValidatedCouncilRequest, CouncilApiError> {
    validate_council_feature_flags(&request)?;
    let (prompt, input_kind, message_count) =
        validate_council_input(request.prompt, request.messages)?;

    let requested_model = request.model.map(|model| model.trim().to_string());
    if let Some(model) = requested_model.as_deref() {
        if model.trim().is_empty() {
            return Err(CouncilApiError::bad_request(
                "invalid_model",
                "model must not be empty",
            ));
        }
    }

    let max_tokens = request
        .max_tokens
        .unwrap_or_else(|| state.compat_default_max_length.load(Ordering::Relaxed))
        .max(1);
    if let Some(temperature) = request.temperature {
        if !temperature.is_finite() || !(0.0..=10.0).contains(&temperature) {
            return Err(CouncilApiError::bad_request(
                "invalid_temperature",
                "temperature must be finite and between 0 and 10",
            ));
        }
    }

    let mut sampling = crate::compute::ffi::SamplingParams::default();
    sampling.max_tokens = max_tokens;
    if let Some(temperature) = request.temperature {
        sampling.temperature = temperature;
    }
    if let Some(seed) = request.seed {
        sampling.seed = seed;
    }

    Ok(ValidatedCouncilRequest {
        request_id: format!("tc-{}", uuid::Uuid::new_v4().simple()),
        requested_model,
        prompt,
        input_kind,
        message_count,
        sampling,
        parallelism: request
            .parallelism
            .unwrap_or(CouncilParallelism::Sequential),
        aha: request.aha,
        trace: request.trace,
        stream: request.stream,
    })
}

fn validate_council_feature_flags(request: &TrialityCouncilRequest) -> Result<(), CouncilApiError> {
    if request.attention_consensus {
        return Err(CouncilApiError::unsupported(
            "attention_consensus_unsupported",
            "Dedicated Answer Council does not execute attention-logit consensus",
        ));
    }
    if request.synthesis {
        return Err(CouncilApiError::unsupported(
            "unsupported_until_enabled",
            "Council synthesis is not enabled in this release",
        ));
    }
    if !request.cross_score {
        return Err(CouncilApiError::unsupported(
            "cross_score_required",
            "The v1 Answer Council requires deterministic 3x3 cross-scoring",
        ));
    }
    Ok(())
}

fn validate_council_input(
    prompt: Option<String>,
    messages: Option<Vec<OpenAiChatMessage>>,
) -> Result<(String, CouncilInputKind, Option<usize>), CouncilApiError> {
    let (prompt, input_kind, message_count) = match (prompt, messages) {
        (Some(prompt), None) => {
            if prompt.trim().is_empty() {
                return Err(CouncilApiError::bad_request(
                    "invalid_prompt",
                    "prompt must not be empty",
                ));
            }
            (prompt, CouncilInputKind::Prompt, None)
        }
        (None, Some(messages)) => {
            if messages.is_empty() || messages.len() > 1024 {
                return Err(CouncilApiError::bad_request(
                    "invalid_messages",
                    "messages must contain between 1 and 1024 entries",
                ));
            }
            let mut chat_messages = Vec::with_capacity(messages.len());
            for message in &messages {
                if !matches!(
                    message.role.as_str(),
                    "system" | "developer" | "user" | "assistant" | "tool"
                ) {
                    return Err(CouncilApiError::bad_request(
                        "invalid_message_role",
                        format!("Unsupported message role `{}`", message.role),
                    ));
                }
                let chat = message.to_chat_message();
                if chat.content.trim().is_empty() {
                    return Err(CouncilApiError::bad_request(
                        "invalid_message_content",
                        "Each message must contain non-empty text content",
                    ));
                }
                chat_messages.push(chat);
            }
            let count = chat_messages.len();
            (
                format_chat_prompt(&chat_messages),
                CouncilInputKind::Messages,
                Some(count),
            )
        }
        (Some(_), Some(_)) => {
            return Err(CouncilApiError::bad_request(
                "ambiguous_input",
                "Exactly one of prompt or messages must be supplied",
            ));
        }
        (None, None) => {
            return Err(CouncilApiError::bad_request(
                "missing_input",
                "Exactly one of prompt or messages must be supplied",
            ));
        }
    };
    if prompt.len() > 1_048_576 {
        return Err(CouncilApiError::bad_request(
            "input_too_large",
            "Council input exceeds the 1 MiB request limit",
        ));
    }
    Ok((prompt, input_kind, message_count))
}

fn prepare_route_embedded_ka_controller(
    model_path: &Path,
    gguf: &GgufFile,
    config: &GgufTurboQuantConfig,
    gate_config: &mut KaGateConfig,
) -> Result<Option<EmbeddedKaController>, CouncilApiError> {
    prepare_embedded_ka_controller(model_path, gguf, config, gate_config).map_err(|error| {
        CouncilApiError::unsupported("ncka_controller_unavailable", error.to_string())
    })
}

fn council_trace_storage(
    identity: &ActiveModelIdentity,
    persisted: Option<&PersistedCouncilRecord>,
) -> serde_json::Value {
    serde_json::json!({
        "model_alias": identity.public_alias,
        "model_sha256": identity.sha256,
        "result_persisted": persisted.is_some(),
        "branch_content_persisted": persisted
            .map(|value| value.branch_content_persisted)
            .unwrap_or(false),
        "final_answer_persisted": persisted
            .map(|value| value.final_answer_persisted)
            .unwrap_or(false),
    })
}

fn validate_requested_model_alias(
    requested_model: Option<&str>,
    identity: &ActiveModelIdentity,
) -> Result<(), CouncilApiError> {
    if let Some(requested_model) = requested_model {
        if requested_model != identity.public_alias {
            return Err(CouncilApiError::conflict(
                "model_not_loaded",
                format!("Requested model `{requested_model}` is not the published model alias"),
            ));
        }
    }
    Ok(())
}

fn switched_public_model_alias(
    current_alias: &str,
    next_physical_name: &str,
    explicit_alias: bool,
) -> String {
    if explicit_alias {
        current_alias.to_string()
    } else {
        next_physical_name.to_string()
    }
}

async fn execute_triality_council(
    state: Arc<AppState>,
    request: ValidatedCouncilRequest,
) -> Result<TrialityCouncilResponse, CouncilApiError> {
    let request_id = request.request_id.clone();
    let request_id_for_error = request_id.clone();
    tokio::task::spawn_blocking(move || {
        let LockedModelIdentity {
            loaded: model,
            identity: active_identity,
        } = lock_model_identity(
            &state.loaded_model,
            &state.model_name,
            &state.model_path,
            &state.model_sha256,
            |model| model.model_name.clone(),
        )
        .map_err(|error| {
            CouncilApiError::internal(
                "model_state_poisoned",
                format!("Loaded model identity is unavailable: {error}"),
            )
        })?;
        if model.model_name != active_identity.physical_name {
            return Err(CouncilApiError::internal(
                "model_identity_inconsistent",
                "Loaded model and captured physical identity differ",
            ));
        }
        validate_requested_model_alias(request.requested_model.as_deref(), &active_identity)?;
        let _capabilities = model.model.triality_capabilities().map_err(|error| {
            CouncilApiError::unsupported(
                "triality_capabilities_unavailable",
                format!("Triality model capabilities are unavailable: {error}"),
            )
        })?;
        let triality = model.config.triality.clone().ok_or_else(|| {
            CouncilApiError::unsupported(
                "triality_schema_unavailable",
                "Embedded schema-v2 Triality context policy is unavailable",
            )
        })?;
        let prompt_tokens = model.model.tokenize(&request.prompt, true, true).len();
        let required_tokens = prompt_tokens.saturating_add(request.sampling.max_tokens as usize);
        if required_tokens > model.config.n_ctx as usize {
            return Err(CouncilApiError::bad_request(
                "council_context_exceeded",
                format!(
                    "Prompt tokens ({prompt_tokens}) plus max_tokens ({}) exceed context {}",
                    request.sampling.max_tokens, model.config.n_ctx
                ),
            ));
        }

        let metadata = model.turboquant.gguf_metadata.as_ref().ok_or_else(|| {
            CouncilApiError::unsupported(
                "triality_schema_unavailable",
                "Embedded schema-v2 Triality metadata is required",
            )
        })?;
        if triality.ncka_required && triality.ncka.is_none() {
            return Err(CouncilApiError::unsupported(
                "ncka_descriptor_unavailable",
                "Required typed NC-KA policy is unavailable",
            ));
        }
        let mut ka_gate = triality
            .ncka
            .as_ref()
            .map(|ncka| KaGateConfig {
                enabled: ncka.enabled,
                required: triality.ncka_required,
                controller_s3_equivariant: ncka.s3_equivariant,
                static_fallback_weights: ncka.fallback_weights,
                ..KaGateConfig::default()
            })
            .unwrap_or_default();
        ka_gate.required = triality.ncka_required;
        let ka_controller = if ka_gate.enabled {
            let _model_file_guard = open_read_guard(&active_identity.path).map_err(|error| {
                CouncilApiError::unsupported(
                    "ncka_controller_unavailable",
                    format!("Loaded GGUF cannot be guarded for NC-KA: {error}"),
                )
            })?;
            let gguf = GgufFile::open(&active_identity.path).map_err(|error| {
                CouncilApiError::unsupported(
                    "ncka_controller_unavailable",
                    format!("Loaded GGUF cannot be parsed for NC-KA: {error}"),
                )
            })?;
            prepare_route_embedded_ka_controller(
                &active_identity.path,
                &gguf,
                metadata,
                &mut ka_gate,
            )?
        } else {
            None
        };
        let urt = if triality.urt_enabled {
            let urt = triality.urt.as_ref().ok_or_else(|| {
                CouncilApiError::unsupported(
                    "urt_descriptor_unavailable",
                    "Required typed URT policy is unavailable",
                )
            })?;
            let model_sha256 = active_identity.sha256.clone().ok_or_else(|| {
                CouncilApiError::unsupported(
                    "urt_model_hash_unavailable",
                    "The loaded GGUF content hash is unavailable",
                )
            })?;
            Some(council_urt_descriptor(metadata, urt, &model_sha256)?)
        } else {
            None
        };
        let moment_degree = triality
            .urt
            .as_ref()
            .filter(|urt| urt.enabled)
            .map(|urt| urt.moment_degree)
            .unwrap_or(3);
        let admission =
            model.admit_council_memory(CouncilExecutionMode::Answer, request.parallelism);
        if !admission.budget.admitted {
            let refusal = admission.refusal.as_ref().map_or_else(
                || admission.budget.reason.clone(),
                |refusal| serde_json::to_string(refusal).unwrap_or_else(|_| refusal.reason.clone()),
            );
            return Err(CouncilApiError::unsupported(
                "council_memory_not_admitted",
                refusal,
            ));
        }
        let memory_ratio = model
            .council_memory_peak_utilization_ratio(&admission)
            .map_err(|refusal| {
                CouncilApiError::unsupported(
                    "council_memory_not_admitted",
                    serde_json::to_string(&refusal).unwrap_or_else(|_| refusal.reason.clone()),
                )
            })?;
        let memory_budget = admission.budget;

        let runtime = model
            .council_runtime(CouncilRuntimeConfig {
                inference: model.config.clone(),
                triality: triality.context,
                memory_budget: memory_budget.clone(),
                answer: AnswerCouncilConfig::default(),
                ka_gate,
                moment_degree,
                memory_ratio,
                attention_consensus_requested: false,
                attention_consensus_required: false,
                aha_enabled: request.aha,
                aha_thresholds: AhaThresholds::default(),
                urt,
            })
            .map_err(|error| {
                CouncilApiError::unsupported(
                    "council_runtime_unavailable",
                    format!("Council runtime initialization failed: {error}"),
                )
            })?;
        let execution = runtime
            .execute(
                &request.request_id,
                &request.prompt,
                &request.sampling,
                &[],
                &NoSafetyPenalty,
                ka_controller
                    .as_ref()
                    .map(|controller| controller as &dyn KaController),
                None,
            )
            .map_err(|error| {
                CouncilApiError::unsupported(
                    "council_execution_failed",
                    format!("Council execution failed: {error}"),
                )
            })?;
        drop(model);

        let request_record = CouncilRequestRecord {
            request_id: request.request_id.clone(),
            created_at: chrono::Utc::now(),
            model: Some(active_identity.public_alias.clone()),
            input_kind: request.input_kind,
            message_count: request.message_count,
            max_tokens: Some(request.sampling.max_tokens),
            temperature: Some(request.sampling.temperature),
            seed: Some(request.sampling.seed),
            parallelism: memory_budget.parallelism,
            attention_consensus: false,
            cross_score: true,
            synthesis: false,
            aha: request.aha,
            trace: request.trace,
        };
        let persisted = state
            .council_store
            .persist(&request_record, &execution.answer)
            .map_err(|error| {
                CouncilApiError::internal(
                    "council_persistence_failed",
                    format!("Council result persistence failed: {error}"),
                )
            })?;
        let urt_trace = record_council_urt_observation(
            &state.urt_registry,
            execution.urt_observation.as_ref(),
        )?;
        emit_council_events(
            &state.telemetry,
            &execution,
            request.trace,
            persisted.as_ref(),
            urt_trace.as_ref(),
        );

        let trace = if request.trace {
            Some(TrialityCouncilTrace {
                parallelism: parallelism_label(memory_budget.parallelism).to_string(),
                context_count: memory_budget.context_count,
                estimated_kv_bytes: Some(memory_budget.estimated_kv_bytes),
                branch_generated_tokens: std::array::from_fn(|index| {
                    execution.answer.candidates[index].generated_tokens
                }),
                branch_runtime_ms: std::array::from_fn(|index| {
                    execution.answer.candidates[index].runtime_ms
                }),
                cross_scores: execution.answer.cross_scores.scores,
                attention: serde_json::to_value(&execution.attention).map_err(|error| {
                    CouncilApiError::internal(
                        "council_trace_serialization_failed",
                        error.to_string(),
                    )
                })?,
                ncka: serde_json::to_value(&execution.ka_gate).map_err(|error| {
                    CouncilApiError::internal(
                        "council_trace_serialization_failed",
                        error.to_string(),
                    )
                })?,
                aha: serde_json::json!({
                    "requested": request.aha,
                    "evaluation": execution.aha,
                }),
                urt: urt_trace
                    .as_ref()
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|error| {
                        CouncilApiError::internal(
                            "council_trace_serialization_failed",
                            error.to_string(),
                        )
                    })?,
                capabilities: serde_json::to_value(&execution.capabilities).map_err(|error| {
                    CouncilApiError::internal(
                        "council_trace_serialization_failed",
                        error.to_string(),
                    )
                })?,
                storage: Some(council_trace_storage(&active_identity, persisted.as_ref())),
            })
        } else {
            None
        };

        Ok(TrialityCouncilResponse {
            id: execution.answer.request_id.clone(),
            object: "triality.council".to_string(),
            model: active_identity.public_alias,
            selected_text: execution.answer.selected_text.clone(),
            selected_view: execution.answer.selected_view,
            candidate_scores: execution.answer.candidate_scores,
            winner_margin: execution.answer.winner_margin,
            agreement: execution.answer.agreement,
            aha: execution.answer.aha.clone(),
            trace,
        })
    })
    .await
    .map_err(|error| {
        CouncilApiError::internal(
            "council_task_join_failed",
            format!("Council task {request_id_for_error} failed to join: {error}"),
        )
    })?
}

fn emit_council_events(
    telemetry: &TelemetryEmitter,
    execution: &crate::council::CouncilExecutionResult,
    trace_enabled: bool,
    persisted: Option<&crate::council::store::PersistedCouncilRecord>,
    urt_trace: Option<&UrtAssessment>,
) {
    let content_persisted = persisted
        .map(|value| value.branch_content_persisted)
        .unwrap_or(false);
    for candidate in &execution.answer.candidates {
        telemetry.emit(TelemetryEvent::TrialityBranchCompleted {
            request_id: execution.answer.request_id.clone(),
            view: candidate.view,
            prompt_tokens: candidate.prompt_tokens,
            generated_tokens: candidate.generated_tokens,
            runtime_ms: candidate.runtime_ms,
            tok_per_sec: candidate.tok_per_sec,
            trace_enabled,
            content_persisted,
        });
    }
    telemetry.emit(TelemetryEvent::TrialityConsensusCompleted {
        request_id: execution.answer.request_id.clone(),
        selected_view: execution.answer.selected_view,
        candidate_scores: execution.answer.candidate_scores,
        winner_margin: execution.answer.winner_margin,
        agreement: execution.answer.agreement,
        result_persisted: persisted.is_some(),
    });
    if let Some(urt_trace) = urt_trace {
        emit_council_urt_event(telemetry, execution.answer.request_id.clone(), urt_trace);
    }
    telemetry.emit(TelemetryEvent::TrialityAha {
        request_id: execution.answer.request_id.clone(),
        emitted: execution.answer.aha.is_some(),
        mode: execution.answer.aha.as_ref().map(|event| event.mode),
        reason_code: execution.answer.aha.as_ref().map(|event| event.reason_code),
    });
}

fn record_council_urt_observation(
    registry: &Mutex<UrtRegistry>,
    observation: Option<&UrtObservation>,
) -> Result<Option<UrtAssessment>, CouncilApiError> {
    let Some(observation) = observation else {
        return Ok(None);
    };
    let mut registry = registry.lock().map_err(|_| {
        CouncilApiError::internal(
            "urt_registry_state_poisoned",
            "The URT registry state is unavailable",
        )
    })?;
    let assessment = registry
        .record_and_assess(observation.clone())
        .map_err(|error| {
            CouncilApiError::internal(
                "urt_observation_persistence_failed",
                format!("URT observation persistence or assessment failed: {error}"),
            )
        })?;
    Ok(Some(assessment))
}

fn emit_council_urt_event(
    telemetry: &TelemetryEmitter,
    request_id: String,
    urt_trace: &UrtAssessment,
) {
    telemetry.emit(TelemetryEvent::TrialityUrtChecked {
        request_id,
        comparison_count: urt_trace
            .report
            .as_ref()
            .map(|report| report.comparisons.len() as u32)
            .unwrap_or(0),
        consistent: urt_trace.report.as_ref().map(|report| report.consistent),
        max_absolute_error: urt_trace
            .report
            .as_ref()
            .map(|report| report.max_absolute_error),
    });
}

fn triality_council_response(response: TrialityCouncilResponse, stream: bool) -> Response {
    if !stream {
        return Json(response).into_response();
    }
    match serde_json::to_string(&response) {
        Ok(payload) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            format!("data: {payload}\n\ndata: [DONE]\n\n"),
        )
            .into_response(),
        Err(error) => {
            CouncilApiError::internal("council_response_serialization_failed", error.to_string())
                .into_response()
        }
    }
}

fn openai_council_chat_response(response: TrialityCouncilResponse, stream: bool) -> Response {
    let created = chrono::Utc::now().timestamp();
    if !stream {
        return Json(compat::openai_chat_response(
            &format!("chatcmpl-{}", response.id),
            created,
            &response.model,
            &response.selected_text,
            0,
            0,
        ))
        .into_response();
    }

    let request_id = format!("chatcmpl-{}", response.id);
    let content_chunk = serde_json::json!({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": response.model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": response.selected_text,
            },
            "finish_reason": serde_json::Value::Null,
        }],
    });
    let final_chunk = serde_json::json!({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": response.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "usage": compat::openai_usage(0, 0),
    });
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        format!("data: {content_chunk}\n\ndata: {final_chunk}\n\ndata: [DONE]\n\n"),
    )
        .into_response()
}

fn ollama_council_chat_response(response: TrialityCouncilResponse, stream: bool) -> Response {
    let payload = ChatResponseChunk {
        model: response.model,
        created_at: chrono::Utc::now().to_rfc3339(),
        message: ChatMessage {
            role: "assistant".to_string(),
            content: response.selected_text,
        },
        done: true,
        done_reason: Some("stop".to_string()),
        total_duration: None,
        load_duration: None,
        prompt_eval_count: None,
        prompt_eval_duration: None,
        eval_count: None,
        eval_duration: None,
    };
    if !stream {
        return Json(payload).into_response();
    }
    match serde_json::to_string(&payload) {
        Ok(json) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            format!("{json}\n"),
        )
            .into_response(),
        Err(error) => {
            CouncilApiError::internal("council_response_serialization_failed", error.to_string())
                .into_response()
        }
    }
}

fn stored_council_response(stored: StoredCouncilRecord) -> TrialityCouncilResponse {
    let context_count = if stored.request.parallelism == CouncilParallelism::Parallel {
        3
    } else {
        1
    };
    let trace = stored.request.trace.then(|| TrialityCouncilTrace {
        parallelism: parallelism_label(stored.request.parallelism).to_string(),
        context_count,
        estimated_kv_bytes: None,
        branch_generated_tokens: std::array::from_fn(|index| {
            stored.branch_candidates[index].generated_tokens
        }),
        branch_runtime_ms: std::array::from_fn(|index| stored.branch_candidates[index].runtime_ms),
        cross_scores: stored.cross_scores.scores,
        attention: serde_json::json!({ "status": "not_requested" }),
        ncka: serde_json::json!({ "status": "not_persisted" }),
        aha: serde_json::json!({
            "requested": stored.request.aha,
            "status": "not_persisted",
        }),
        urt: None,
        capabilities: serde_json::json!({ "status": "not_persisted" }),
        storage: Some(serde_json::json!({
            "branch_content_redacted": stored.branch_content_redacted,
            "final_answer_redacted": stored.final_answer_redacted,
        })),
    });
    TrialityCouncilResponse {
        id: stored.consensus.request_id,
        object: "triality.council".to_string(),
        model: stored
            .request
            .model
            .unwrap_or_else(|| "unknown".to_string()),
        selected_text: stored.final_answer,
        selected_view: stored.consensus.selected_view,
        candidate_scores: stored.consensus.candidate_scores,
        winner_margin: stored.consensus.winner_margin,
        agreement: stored.consensus.agreement,
        aha: stored.aha,
        trace,
    }
}

fn council_urt_descriptor(
    metadata: &GgufTurboQuantConfig,
    urt: &GgufUrtConfig,
    model_sha256: &str,
) -> Result<CouncilUrtDescriptor, CouncilApiError> {
    if model_sha256.len() != 64 || !model_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CouncilApiError::unsupported(
            "urt_model_hash_invalid",
            "The loaded GGUF content hash is invalid",
        ));
    }
    if !urt
        .supported_representations
        .iter()
        .any(|value| value == RepresentationKind::HypuraNative.as_str())
    {
        return Err(CouncilApiError::unsupported(
            "urt_representation_unsupported",
            "Embedded URT policy does not support the Hypura native representation",
        ));
    }
    if urt.operator_word_sha256.trim().is_empty()
        || !urt.consistency_tolerance.is_finite()
        || urt.consistency_tolerance <= 0.0
    {
        return Err(CouncilApiError::unsupported(
            "urt_descriptor_invalid",
            "Embedded URT descriptor is incomplete",
        ));
    }
    let manifest: serde_json::Value =
        serde_json::from_str(&urt.operator_word_manifest).map_err(|_| {
            CouncilApiError::unsupported(
                "urt_descriptor_invalid",
                "Embedded URT operator-word manifest is invalid",
            )
        })?;
    let operator_word = manifest
        .get("words")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            CouncilApiError::unsupported(
                "urt_descriptor_invalid",
                "Embedded URT operator-word manifest has no words",
            )
        })?
        .iter()
        .map(|value| value.as_str().map(str::trim))
        .collect::<Option<Vec<_>>>()
        .filter(|words| !words.is_empty() && words.iter().all(|word| !word.is_empty()))
        .ok_or_else(|| {
            CouncilApiError::unsupported(
                "urt_descriptor_invalid",
                "Embedded URT operator words are invalid",
            )
        })?
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();

    Ok(CouncilUrtDescriptor {
        representation: RepresentationId {
            kind: RepresentationKind::HypuraNative,
            model_hash: model_sha256.to_ascii_lowercase(),
            artefact_hash: Some(model_sha256.to_ascii_lowercase()),
            backend: RepresentationKind::HypuraNative.as_str().to_string(),
            precision: format!("k{:.3}_v{:.3}", metadata.k_bits, metadata.v_bits),
            view: None,
        },
        operator_word,
        operator_word_sha256: urt.operator_word_sha256.clone(),
        tolerance: f64::from(urt.consistency_tolerance),
    })
}

const fn parallelism_label(value: CouncilParallelism) -> &'static str {
    match value {
        CouncilParallelism::Sequential => "sequential",
        CouncilParallelism::Parallel => "parallel",
        CouncilParallelism::Auto => "auto",
    }
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn version_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"version": env!("CARGO_PKG_VERSION")}))
}

fn compat_ui_enabled() -> bool {
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

async fn kobold_api_docs_handler() -> Html<&'static str> {
    Html(VENDORED_KOBOLDCPP_API_HTML)
}

async fn kobold_api_spec_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/json; charset=utf-8")],
        VENDORED_KOBOLDCPP_API_JSON,
    )
}

async fn kobold_lite_manifest_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/manifest+json; charset=utf-8",
        )],
        VENDORED_KOBOLD_LITE_MANIFEST,
    )
}

async fn kobold_lite_service_worker_handler() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        VENDORED_KOBOLD_LITE_SW,
    )
}

async fn kobold_lite_niko_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "image/png")],
        VENDORED_KOBOLD_LITE_NIKO_PNG,
    )
}

async fn kobold_extra_version_handler(
    State(state): State<Arc<AppState>>,
) -> Json<KcppVersionResponse> {
    let protected = std::env::var("HYPURA_API_KEY")
        .ok()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    Json(compat::build_version_response(
        protected,
        compat_features_snapshot(&state),
    ))
}

async fn kobold_perf_handler(State(state): State<Arc<AppState>>) -> Json<KcppPerfResponse> {
    let snapshot = state.compat_perf.lock().unwrap().clone();
    Json(compat::build_perf_response(
        &snapshot,
        state.compat_started_at,
        state.generation_in_progress.load(Ordering::Relaxed),
        false,
    ))
}

async fn kobold_token_count_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenCountRequest>,
) -> Json<TokenCountResponse> {
    let model = state.loaded_model.lock().unwrap();
    let ids = model.model.tokenize(&req.prompt, true, true);
    Json(TokenCountResponse {
        value: ids.len(),
        ids,
    })
}

async fn kobold_api_info_version_handler() -> Json<KoboldInfoVersionResponse> {
    Json(KoboldInfoVersionResponse {
        result: compat::KOBOLDAI_API_VERSION.into(),
    })
}

async fn kobold_max_length_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ScalarValueResponse> {
    Json(ScalarValueResponse {
        value: state.compat_default_max_length.load(Ordering::Relaxed),
    })
}

async fn kobold_max_context_length_handler(
    State(state): State<Arc<AppState>>,
) -> Json<ScalarValueResponse> {
    Json(ScalarValueResponse {
        value: state.default_context.load(Ordering::Relaxed),
    })
}

async fn kobold_lite_gui_handler() -> Html<&'static str> {
    if compat_ui_enabled() {
        return Html(VENDORED_KOBOLD_LITE_INDEX);
    }
    Html(
        r#"<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Hypura Kobold GUI</title>
  <style>
    :root {
      --bg: #f1f2f5;
      --panel: #ffffff;
      --panel-soft: #fafafa;
      --text: #1c1f23;
      --muted: #5f6670;
      --border: #d4d9e1;
      --accent: #3254ff;
      --ok: #1b8f3b;
      --err: #b3261e;
      --kbd: #eceff7;
    }
    body[data-theme="dark"] {
      --bg: #101319;
      --panel: #181d26;
      --panel-soft: #202736;
      --text: #e8edf6;
      --muted: #a9b4c6;
      --border: #2d3648;
      --accent: #7ea2ff;
      --ok: #61d889;
      --err: #ff8f87;
      --kbd: #252d3d;
    }
    body[data-theme="classic"] {
      --bg: #efe7d2;
      --panel: #f6f0e1;
      --panel-soft: #ece2c8;
      --text: #3f2f1f;
      --muted: #685741;
      --border: #b7a37f;
      --accent: #7e4f2b;
      --ok: #3d7a42;
      --err: #9d3a2d;
      --kbd: #e6d7b5;
    }
    body {
      font-family: "Segoe UI", "Noto Sans JP", sans-serif;
      max-width: 1380px;
      margin: 1rem auto;
      padding: 0 1rem;
      color: var(--text);
      background: var(--bg);
    }
    textarea, input, select {
      width: 100%;
      box-sizing: border-box;
      margin-top: .3rem;
      margin-bottom: .7rem;
      background: var(--panel-soft);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: .45rem .55rem;
    }
    .row { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap: .7rem; }
    .toolbar { display:flex; gap:.6rem; margin:.8rem 0; flex-wrap: wrap; }
    .panel { border:1px solid var(--border); border-radius:8px; padding:.8rem; margin:.8rem 0; background:var(--panel); }
    .layout { display:grid; grid-template-columns: 1.1fr 0.9fr; gap: .9rem; }
    .metrics-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.5rem; }
    .metric { border:1px solid var(--border); border-radius:8px; padding:.6rem; background:var(--panel-soft); }
    .metric .k { font-size:.78rem; color:var(--muted); }
    .metric .v { font-size:1.05rem; font-weight:600; }
    .status { border-left:4px solid var(--muted); padding:.5rem .7rem; background:var(--panel-soft); margin:.6rem 0; }
    .status.ok { border-left-color:var(--ok); }
    .status.err { border-left-color:var(--err); }
    button {
      padding:.55rem .95rem;
      cursor:pointer;
      border-radius:8px;
      border:1px solid var(--border);
      background: var(--panel-soft);
      color: var(--text);
    }
    button:hover { border-color: var(--accent); }
    button[disabled] { opacity:.55; cursor:not-allowed; }
    pre { white-space: pre-wrap; border:1px solid var(--border); padding:1rem; border-radius:6px; min-height:220px; background:var(--panel); }
  </style>
</head>
<body data-theme="classic">
  <h2>Hypura Kobold GUI (Parity++)</h2>
  <div class="panel">
    <div class="toolbar">
      <label style="min-width:220px;">Theme
        <select id="themeSelect" onchange="changeThemeFromSelect()">
          <option value="light">light</option>
          <option value="dark">dark</option>
          <option value="classic">classic</option>
        </select>
      </label>
      <span id="themeStatus">Theme source: initializing</span>
    </div>
  </div>
  <div id="connStatus" class="status">Connection: checking...</div>
  <div class="panel">
    <strong>GGUF Model Switcher</strong>
    <div class="toolbar">
      <select id="modelList"></select>
      <button onclick="refreshModelList()">Refresh Models</button>
      <button onclick="switchModel()">Switch Model</button>
    </div>
    <div id="modelStatus">Model: -</div>
  </div>
  <div class="toolbar">
    <button onclick="applyStage('short')">Stage: Short (4096/64)</button>
    <button onclick="applyStage('medium')">Stage: Medium (4096/256)</button>
    <button onclick="applyStage('long')">Stage: Long (8192/512)</button>
    <button onclick="checkConnection()">Recheck Connection</button>
  </div>
  <div class="panel">
    <strong>Runtime Metrics</strong>
    <div class="metrics-grid">
      <div class="metric"><div class="k">tok/s</div><div class="v" id="mTok">-</div></div>
      <div class="metric"><div class="k">prompt ms</div><div class="v" id="mPrompt">-</div></div>
      <div class="metric"><div class="k">eval count</div><div class="v" id="mEval">-</div></div>
      <div class="metric"><div class="k">total ms</div><div class="v" id="mTotal">-</div></div>
    </div>
    <div class="toolbar">
      <span id="busyState">State: idle</span>
      <span id="lastSuccess">Last success: -</span>
      <span id="tokenProgress">Progress: 0 token / 0 ms</span>
    </div>
  </div>
  <div class="layout">
  <div>
  <div class="panel">
    <label>Preset Name</label>
    <input id="presetName" type="text" placeholder="my-preset">
    <div class="toolbar">
      <button onclick="savePreset()">Preset Save</button>
      <button onclick="loadPreset()">Preset Load</button>
      <button onclick="deletePreset()">Preset Delete</button>
      <button onclick="diffPreset()">Preset Diff</button>
      <button onclick="exportPresetCli()">Export CLI</button>
      <button onclick="importPresetCli()">Import CLI</button>
      <select id="presetList"></select>
    </div>
    <label>CLI Bridge (paste / generated)</label>
    <textarea id="cliBridge" rows="2" placeholder="hypura serve --model ./model.gguf --temperature 0.8 ..."></textarea>
  </div>
  <div class="panel">
    <label>Prompt</label>
    <textarea id="prompt" rows="7"></textarea>
    <div class="row">
      <label>max_length<input id="maxlen" type="number" value="256"></label>
      <label>temperature<input id="temp" type="number" step="0.01" value="0.8"></label>
      <label>top_k<input id="topk" type="number" value="40"></label>
      <label>top_p<input id="topp" type="number" step="0.01" value="0.9"></label>
    </div>
    <div class="row">
      <label>min_p<input id="minp" type="number" step="0.01" value="0.05"></label>
      <label>rep_pen<input id="repen" type="number" step="0.01" value="1.0"></label>
      <label>rep_pen_range<input id="repenRange" type="number" value="64"></label>
      <label>seed<input id="seed" type="number" value="42"></label>
    </div>
    <div class="row">
      <label>top_a<input id="topa" type="number" step="0.01" value="0"></label>
      <label>tfs<input id="tfs" type="number" step="0.01" value="1"></label>
      <label>typical<input id="typical" type="number" step="0.01" value="1"></label>
      <label>sampler_order<input id="samplerOrder" type="text" value="6,0,1,3,4,2,5"></label>
    </div>
    <div class="row">
      <label>mirostat<input id="mirostat" type="number" value="0"></label>
      <label>mirostat_tau<input id="mirostatTau" type="number" step="0.01" value="5"></label>
      <label>mirostat_eta<input id="mirostatEta" type="number" step="0.01" value="0.1"></label>
      <label>dynatemp_range<input id="dynatempRange" type="number" step="0.01" value="0"></label>
    </div>
    <div class="row">
      <label>dynatemp_exponent<input id="dynatempExponent" type="number" step="0.01" value="1"></label>
      <label>smoothing_factor<input id="smoothingFactor" type="number" step="0.01" value="0"></label>
      <label>presence_penalty<input id="presencePenalty" type="number" step="0.01" value="0"></label>
      <label>frequency_penalty<input id="frequencyPenalty" type="number" step="0.01" value="0"></label>
    </div>
    <div class="row">
      <label>tq_so8_off<select id="tqSo8Off"><option value="">(default)</option><option value="1">true</option><option value="0">false</option></select></label>
      <label>tq_so8_learned<select id="tqSo8Learned"><option value="">(default)</option><option value="1">true</option><option value="0">false</option></select></label>
      <label>tq_triality_off<select id="tqTrialityOff"><option value="">(default)</option><option value="1">true</option><option value="0">false</option></select></label>
      <label>tq_triality_mix<input id="tqTrialityMix" type="number" step="0.01" placeholder="(default)"></label>
    </div>
    <div class="row">
      <label>tq_rotation_seed<input id="tqRotationSeed" type="number" placeholder="(default)"></label>
      <label>tq_artifact<input id="tqArtifact" type="text" placeholder="C:\path\to\artifact.json"></label>
      <label></label>
      <label></label>
    </div>
    <label>stop_sequence (1行1個)</label>
    <textarea id="stop" rows="3"></textarea>
  </div>
  <div class="toolbar">
    <button onclick="generateOnce()">Generate</button>
    <button onclick="generateStream()">Generate Stream</button>
    <button onclick="retryLast()">Retry Last</button>
    <button onclick="abortGen()">Abort</button>
    <button onclick="checkStatus()">Check</button>
  </div>
  <h3>Output</h3>
  <pre id="out"></pre>
  </div>
  <div>
  <div class="panel">
  <strong>UI Mode</strong>
  <div class="toolbar">
    <button onclick="switchMode('chat')">Chat</button>
    <button onclick="switchMode('instruct')">Instruct</button>
    <button onclick="switchMode('story')">Storywriter</button>
    <button onclick="switchMode('adventure')">Adventure</button>
  </div>
  <div id="uiMode">Mode: chat</div>
  </div>
  <h3>Preset Diff</h3>
  <pre id="presetDiff"></pre>
  <h3>Server Presets</h3>
  <pre id="serverPresets"></pre>
  <h3>Generation History</h3>
  <pre id="genHistory"></pre>
  <h3>Event Log</h3>
  <pre id="eventLog"></pre>
  </div>
  </div>
  <script>
    const KEY = 'hypura_kobold_presets_v2';
    const MODE_KEY = 'hypura_kobold_ui_mode';
    const THEME_KEY = 'hypura_kobold_ui_theme';
    let lastRequest = null;
    let generationBusy = false;
    let latestError = '';

    function parseSamplerOrder(v) {
      return v.split(',').map(x => Number(x.trim())).filter(x => Number.isInteger(x));
    }

    function readForm() {
      const stop = document.getElementById('stop').value.split('\n').map(s => s.trim()).filter(Boolean);
      const tqSo8Off = document.getElementById('tqSo8Off').value;
      const tqSo8Learned = document.getElementById('tqSo8Learned').value;
      const tqTrialityOff = document.getElementById('tqTrialityOff').value;
      const tqTrialityMixRaw = document.getElementById('tqTrialityMix').value.trim();
      const tqRotationSeedRaw = document.getElementById('tqRotationSeed').value.trim();
      const tqArtifact = document.getElementById('tqArtifact').value.trim();
      const body = {
        prompt: document.getElementById('prompt').value,
        max_length: Number(document.getElementById('maxlen').value),
        temperature: Number(document.getElementById('temp').value),
        top_k: Number(document.getElementById('topk').value),
        top_p: Number(document.getElementById('topp').value),
        min_p: Number(document.getElementById('minp').value),
        rep_pen: Number(document.getElementById('repen').value),
        rep_pen_range: Number(document.getElementById('repenRange').value),
        seed: Number(document.getElementById('seed').value),
        top_a: Number(document.getElementById('topa').value),
        tfs: Number(document.getElementById('tfs').value),
        typical: Number(document.getElementById('typical').value),
        mirostat: Number(document.getElementById('mirostat').value),
        mirostat_tau: Number(document.getElementById('mirostatTau').value),
        mirostat_eta: Number(document.getElementById('mirostatEta').value),
        dynatemp_range: Number(document.getElementById('dynatempRange').value),
        dynatemp_exponent: Number(document.getElementById('dynatempExponent').value),
        smoothing_factor: Number(document.getElementById('smoothingFactor').value),
        presence_penalty: Number(document.getElementById('presencePenalty').value),
        frequency_penalty: Number(document.getElementById('frequencyPenalty').value),
        sampler_order: parseSamplerOrder(document.getElementById('samplerOrder').value),
        stop_sequence: stop
      };
      if (tqSo8Off) body.tq_so8_off = tqSo8Off === '1';
      if (tqSo8Learned) body.tq_so8_learned = tqSo8Learned === '1';
      if (tqTrialityOff) body.tq_triality_off = tqTrialityOff === '1';
      if (tqTrialityMixRaw) body.tq_triality_mix = Number(tqTrialityMixRaw);
      if (tqRotationSeedRaw) body.tq_rotation_seed = Number(tqRotationSeedRaw);
      if (tqArtifact) body.tq_artifact = tqArtifact;
      return body;
    }

    function writeForm(v) {
      document.getElementById('prompt').value = v.prompt ?? '';
      document.getElementById('maxlen').value = v.max_length ?? 256;
      document.getElementById('temp').value = v.temperature ?? 0.8;
      document.getElementById('topk').value = v.top_k ?? 40;
      document.getElementById('topp').value = v.top_p ?? 0.9;
      document.getElementById('minp').value = v.min_p ?? 0.05;
      document.getElementById('repen').value = v.rep_pen ?? 1.0;
      document.getElementById('repenRange').value = v.rep_pen_range ?? 64;
      document.getElementById('seed').value = v.seed ?? 42;
      document.getElementById('topa').value = v.top_a ?? 0;
      document.getElementById('tfs').value = v.tfs ?? 1;
      document.getElementById('typical').value = v.typical ?? 1;
      document.getElementById('mirostat').value = v.mirostat ?? 0;
      document.getElementById('mirostatTau').value = v.mirostat_tau ?? 5;
      document.getElementById('mirostatEta').value = v.mirostat_eta ?? 0.1;
      document.getElementById('dynatempRange').value = v.dynatemp_range ?? 0;
      document.getElementById('dynatempExponent').value = v.dynatemp_exponent ?? 1;
      document.getElementById('smoothingFactor').value = v.smoothing_factor ?? 0;
      document.getElementById('presencePenalty').value = v.presence_penalty ?? 0;
      document.getElementById('frequencyPenalty').value = v.frequency_penalty ?? 0;
      document.getElementById('samplerOrder').value = (v.sampler_order ?? [6,0,1,3,4,2,5]).join(',');
      document.getElementById('stop').value = (v.stop_sequence ?? []).join('\n');
      document.getElementById('tqSo8Off').value = v.tq_so8_off === undefined ? '' : (v.tq_so8_off ? '1' : '0');
      document.getElementById('tqSo8Learned').value = v.tq_so8_learned === undefined ? '' : (v.tq_so8_learned ? '1' : '0');
      document.getElementById('tqTrialityOff').value = v.tq_triality_off === undefined ? '' : (v.tq_triality_off ? '1' : '0');
      document.getElementById('tqTrialityMix').value = v.tq_triality_mix ?? '';
      document.getElementById('tqRotationSeed').value = v.tq_rotation_seed ?? '';
      document.getElementById('tqArtifact').value = v.tq_artifact ?? '';
    }

    function applyStage(stage) {
      const stageMap = {
        short: { max_length: 64, prompt: '短く挨拶して。' },
        medium: { max_length: 256, prompt: '数段落で説明して。' },
        long: { max_length: 512, prompt: '長めに説明して。' }
      };
      const cur = readForm();
      const patch = stageMap[stage] || stageMap.short;
      const contextHint = stage === 'long' ? '8192' : '4096';
      writeForm({ ...cur, ...patch });
      setStatus(`Preset applied: ${stage} (recommended context ${contextHint})`, false);
    }

    function setStatus(message, isError = false) {
      const el = document.getElementById('connStatus');
      el.textContent = message;
      el.className = isError ? 'status err' : 'status ok';
      if (isError) latestError = message;
    }

    function setBusy(busy) {
      generationBusy = busy;
      document.getElementById('busyState').textContent = `State: ${busy ? 'generating' : 'idle'}`;
      for (const b of document.querySelectorAll('button')) {
        if (['Abort', 'Recheck Connection', 'Refresh Models', 'Switch Model'].includes(b.textContent.trim())) continue;
        b.disabled = busy;
      }
      document.querySelector('button[onclick="abortGen()"]').disabled = !busy;
    }

    function switchMode(mode) {
      localStorage.setItem(MODE_KEY, mode);
      document.getElementById('uiMode').textContent = `Mode: ${mode}`;
      const presets = {
        chat: { temperature: 0.8, top_p: 0.92, max_length: 256 },
        instruct: { temperature: 0.6, top_p: 0.9, max_length: 320 },
        story: { temperature: 0.95, top_p: 0.96, max_length: 512 },
        adventure: { temperature: 0.85, top_p: 0.94, max_length: 384 }
      };
      writeForm({ ...readForm(), ...(presets[mode] || presets.chat) });
    }

    function applyTheme(theme, source = 'local') {
      const t = ['light', 'dark', 'classic'].includes(theme) ? theme : 'classic';
      document.body.setAttribute('data-theme', t);
      document.getElementById('themeSelect').value = t;
      document.getElementById('themeStatus').textContent = `Theme source: ${source} (${t})`;
      localStorage.setItem(THEME_KEY, t);
    }

    async function syncThemeFromServer() {
      try {
        const r = await fetch('/api/extra/ui-theme');
        const j = await r.json();
        const localTheme = localStorage.getItem(THEME_KEY);
        if (localTheme) {
          applyTheme(localTheme, 'local');
        } else {
          applyTheme(j.theme || 'classic', 'server');
        }
      } catch {
        applyTheme(localStorage.getItem(THEME_KEY) || 'classic', 'fallback');
      }
    }

    async function setThemeServer(theme) {
      try {
        await fetch('/api/extra/ui-theme', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ theme })
        });
      } catch {}
    }

    function changeThemeFromSelect() {
      const theme = document.getElementById('themeSelect').value;
      applyTheme(theme, 'ui');
      setThemeServer(theme);
    }

    async function checkConnection() {
      try {
        const r = await fetch('/api/v1/model');
        const j = await r.json();
        setStatus(`Connection OK: ${j.result || 'unknown model'}`, false);
        document.getElementById('modelStatus').textContent = `Model: ${j.result || '-'}`;
      } catch (e) {
        setStatus(`Connection ERROR: ${String(e)}`, true);
      }
    }

    async function refreshModelList() {
      try {
        const r = await fetch('/api/extra/models');
        const j = await r.json();
        const list = document.getElementById('modelList');
        list.innerHTML = '';
        for (const m of (j.models || [])) {
          const opt = document.createElement('option');
          opt.value = m.path;
          opt.textContent = `${m.name}${m.selected ? ' (active)' : ''}`;
          if (m.selected) opt.selected = true;
          list.appendChild(opt);
        }
      } catch (e) {
        setStatus(`Model list ERROR: ${String(e)}`, true);
      }
    }

    async function switchModel() {
      if (generationBusy) return;
      const path = document.getElementById('modelList').value;
      if (!path) return;
      setBusy(true);
      setStatus('Switching model...', false);
      try {
        const r = await fetch('/api/extra/model/switch', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ path })
        });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
        setStatus(`Model switched: ${j.model} (context=${j.context})`, false);
        document.getElementById('modelStatus').textContent = `Model: ${j.model}`;
        await refreshModelList();
      } catch (e) {
        setStatus(`Switch ERROR: ${String(e)}`, true);
      } finally {
        setBusy(false);
      }
    }

    function toCliArgs(v) {
      const args = [
        '--temperature', String(v.temperature),
        '--top-k', String(v.top_k),
        '--top-p', String(v.top_p),
        '--min-p', String(v.min_p),
        '--repeat-penalty', String(v.rep_pen),
        '--repeat-last-n', String(v.rep_pen_range),
        '--seed', String(v.seed),
      ];
      if (v.top_a !== undefined) args.push('--top-a', String(v.top_a));
      if (v.tfs !== undefined) args.push('--tfs', String(v.tfs));
      if (v.typical !== undefined) args.push('--typical', String(v.typical));
      if (v.mirostat !== undefined) args.push('--mirostat', String(v.mirostat));
      if (v.mirostat_tau !== undefined) args.push('--mirostat-tau', String(v.mirostat_tau));
      if (v.mirostat_eta !== undefined) args.push('--mirostat-eta', String(v.mirostat_eta));
      if (v.dynatemp_range !== undefined) args.push('--dynatemp-range', String(v.dynatemp_range));
      if (v.dynatemp_exponent !== undefined) args.push('--dynatemp-exp', String(v.dynatemp_exponent));
      if (v.smoothing_factor !== undefined) args.push('--smoothing-factor', String(v.smoothing_factor));
      if (v.presence_penalty !== undefined) args.push('--presence-penalty', String(v.presence_penalty));
      if (v.frequency_penalty !== undefined) args.push('--frequency-penalty', String(v.frequency_penalty));
      if (Array.isArray(v.sampler_order) && v.sampler_order.length > 0) args.push('--sampler-order', `"${v.sampler_order.join(',')}"`);
      if (Array.isArray(v.stop_sequence)) for (const s of v.stop_sequence) args.push('--stop', `"${s}"`);
      if (typeof v.tq_so8_off === 'boolean') args.push('--tq-so8-off', v.tq_so8_off ? 'true' : 'false');
      if (typeof v.tq_so8_learned === 'boolean') args.push('--tq-so8-learned', v.tq_so8_learned ? 'true' : 'false');
      if (typeof v.tq_triality_off === 'boolean') args.push('--tq-triality-off', v.tq_triality_off ? 'true' : 'false');
      if (v.tq_triality_mix !== undefined) args.push('--tq-triality-mix', String(v.tq_triality_mix));
      if (v.tq_rotation_seed !== undefined) args.push('--tq-rotation-seed', String(v.tq_rotation_seed));
      if (v.tq_artifact) args.push('--tq-artifact', `"${v.tq_artifact}"`);
      return `hypura serve --model ./model.gguf ${args.join(' ')}`;
    }

    function parseCliBridge(input) {
      const out = {};
      const read = (flag) => {
        const i = input.indexOf(flag);
        if (i < 0) return null;
        const rest = input.slice(i + flag.length).trim();
        const m = rest.match(/^"([^"]+)"|^(\S+)/);
        return m ? (m[1] ?? m[2]) : null;
      };
      const asNum = (v) => (v == null ? undefined : Number(v));
      out.temperature = asNum(read('--temperature'));
      out.top_k = asNum(read('--top-k'));
      out.top_p = asNum(read('--top-p'));
      out.min_p = asNum(read('--min-p'));
      out.rep_pen = asNum(read('--repeat-penalty'));
      out.rep_pen_range = asNum(read('--repeat-last-n'));
      out.seed = asNum(read('--seed'));
      out.top_a = asNum(read('--top-a'));
      out.tfs = asNum(read('--tfs'));
      out.typical = asNum(read('--typical'));
      out.mirostat = asNum(read('--mirostat'));
      out.mirostat_tau = asNum(read('--mirostat-tau'));
      out.mirostat_eta = asNum(read('--mirostat-eta'));
      out.dynatemp_range = asNum(read('--dynatemp-range'));
      out.dynatemp_exponent = asNum(read('--dynatemp-exp'));
      out.smoothing_factor = asNum(read('--smoothing-factor'));
      out.presence_penalty = asNum(read('--presence-penalty'));
      out.frequency_penalty = asNum(read('--frequency-penalty'));
      const order = read('--sampler-order');
      if (order) out.sampler_order = parseSamplerOrder(order);
      const tqSo8Off = read('--tq-so8-off');
      if (tqSo8Off) out.tq_so8_off = tqSo8Off === 'true';
      const tqSo8Learned = read('--tq-so8-learned');
      if (tqSo8Learned) out.tq_so8_learned = tqSo8Learned === 'true';
      const tqTrialityOff = read('--tq-triality-off');
      if (tqTrialityOff) out.tq_triality_off = tqTrialityOff === 'true';
      out.tq_triality_mix = asNum(read('--tq-triality-mix'));
      out.tq_rotation_seed = asNum(read('--tq-rotation-seed'));
      const tqArtifact = read('--tq-artifact');
      if (tqArtifact) out.tq_artifact = tqArtifact;
      return out;
    }

    function exportPresetCli() {
      const cmd = toCliArgs(readForm());
      document.getElementById('cliBridge').value = cmd;
    }

    function importPresetCli() {
      const input = document.getElementById('cliBridge').value;
      const parsed = parseCliBridge(input);
      writeForm({ ...readForm(), ...parsed });
    }

    function presets() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; } }
    function persist(p) { localStorage.setItem(KEY, JSON.stringify(p)); refreshPresetList(); }
    function refreshPresetList() {
      const list = document.getElementById('presetList');
      const p = presets();
      list.innerHTML = '';
      Object.keys(p).sort().forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        list.appendChild(opt);
      });
    }

    function savePreset() {
      const name = (document.getElementById('presetName').value || '').trim();
      if (!name) return;
      const p = presets();
      p[name] = { ...readForm(), _meta: { version: 2, saved_at: new Date().toISOString() } };
      persist(p);
      savePresetServer(name, p[name]);
    }

    function loadPreset() {
      const name = document.getElementById('presetList').value;
      const p = presets();
      if (name && p[name]) writeForm(p[name]);
    }

    function deletePreset() {
      const name = document.getElementById('presetList').value;
      const p = presets();
      if (name && p[name]) {
        delete p[name];
        persist(p);
        deletePresetServer(name);
      }
    }

    async function savePresetServer(name, payload) {
      try {
        await fetch('/api/extra/presets/save', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ name, payload })
        });
        await refreshServerPanels();
      } catch {}
    }

    async function deletePresetServer(name) {
      try {
        await fetch('/api/extra/presets/delete', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ name })
        });
        await refreshServerPanels();
      } catch {}
    }

    async function refreshServerPanels() {
      try {
        const [presetRes, historyRes, eventRes] = await Promise.all([
          fetch('/api/extra/presets/list'),
          fetch('/api/extra/history'),
          fetch('/api/extra/events')
        ]);
        const presetJson = await presetRes.json();
        const historyJson = await historyRes.json();
        const eventJson = await eventRes.json();
        document.getElementById('serverPresets').textContent = JSON.stringify(presetJson, null, 2);
        document.getElementById('genHistory').textContent = JSON.stringify(historyJson, null, 2);
        document.getElementById('eventLog').textContent = JSON.stringify(eventJson, null, 2);
      } catch (e) {
        document.getElementById('eventLog').textContent = `[panel refresh error] ${String(e)}`;
      }
    }

    function diffPreset() {
      const name = document.getElementById('presetList').value;
      const p = presets();
      if (!name || !p[name]) return;
      const current = readForm();
      const saved = p[name];
      const keys = Array.from(new Set([...Object.keys(current), ...Object.keys(saved)])).sort();
      const diffs = [];
      for (const k of keys) {
        if (k === '_meta') continue;
        const a = JSON.stringify(current[k]);
        const b = JSON.stringify(saved[k]);
        if (a !== b) diffs.push(`${k}: current=${a} | preset=${b}`);
      }
      document.getElementById('presetDiff').textContent = diffs.length > 0 ? diffs.join('\n') : 'no diff';
    }

    async function generateOnce() {
      if (generationBusy) return;
      const body = readForm();
      lastRequest = body;
      document.getElementById('out').textContent = 'Generating...';
      setBusy(true);
      const startedAt = performance.now();
      try {
        const r = await fetch('/api/v1/generate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
        const j = await r.json();
        const text = (j.results && j.results[0] ? j.results[0].text : JSON.stringify(j, null, 2));
        document.getElementById('out').textContent = text;
        document.getElementById('lastSuccess').textContent = `Last success: ${new Date().toLocaleTimeString()}`;
        document.getElementById('mEval').textContent = String(text.length);
        document.getElementById('mTotal').textContent = (performance.now() - startedAt).toFixed(1);
        document.getElementById('tokenProgress').textContent = `Progress: ${text.length} chars / ${(performance.now() - startedAt).toFixed(0)} ms`;
      } catch (e) {
        document.getElementById('out').textContent = `[error] ${String(e)}`;
        setStatus(`Generate ERROR: ${String(e)}`, true);
      } finally {
        await refreshServerPanels();
        setBusy(false);
      }
    }

    async function generateStream() {
      if (generationBusy) return;
      const body = { ...readForm(), stream: true };
      lastRequest = body;
      document.getElementById('out').textContent = '';
      const startedAt = performance.now();
      setBusy(true);
      try {
        const r = await fetch('/api/extra/generate/stream', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        let tokenCount = 0;
        let latestTokPerSec = null;
        while (true) {
          const {value, done} = await reader.read();
          if (done) break;
          buf += dec.decode(value, {stream: true});
          const lines = buf.split('\n');
          buf = lines.pop();
          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const obj = JSON.parse(line);
              if (obj.token) {
                tokenCount += 1;
                latestTokPerSec = obj.tok_per_sec ?? latestTokPerSec;
                document.getElementById('out').textContent += obj.token;
                document.getElementById('tokenProgress').textContent = `Progress: ${tokenCount} token / ${(performance.now() - startedAt).toFixed(0)} ms`;
              }
              if (obj.done) {
                const elapsedMs = Math.max(1, performance.now() - startedAt);
                const tps = latestTokPerSec ?? ((tokenCount * 1000.0) / elapsedMs);
                document.getElementById('mTok').textContent = tps.toFixed(2);
                document.getElementById('mEval').textContent = String(obj.eval_count ?? '-');
                document.getElementById('mTotal').textContent = (obj.total_duration ? obj.total_duration / 1e6 : elapsedMs).toFixed(1);
                document.getElementById('mPrompt').textContent = String(obj.prompt_eval_ms ?? '-');
                document.getElementById('lastSuccess').textContent = `Last success: ${new Date().toLocaleTimeString()}`;
                document.getElementById('out').textContent += '\n\n[done]';
              }
            } catch {}
          }
        }
      } catch (e) {
        document.getElementById('out').textContent += `\n\n[error] ${String(e)}`;
        setStatus(`Stream ERROR: ${String(e)}`, true);
      } finally {
        await refreshServerPanels();
        setBusy(false);
      }
    }

    async function retryLast() {
      if (!lastRequest) return;
      document.getElementById('out').textContent = '';
      writeForm(lastRequest);
      await generateStream();
    }

    async function abortGen() {
      await fetch('/api/extra/abort', { method: 'POST' });
      setBusy(false);
    }

    async function checkStatus() {
      const r = await fetch('/api/extra/generate/check');
      const j = await r.json();
      const msg = (j.results && j.results[0] ? j.results[0].text : JSON.stringify(j, null, 2));
      document.getElementById('out').textContent += '\n\n[check] ' + msg;
    }

    refreshPresetList();
    setBusy(false);
    syncThemeFromServer();
    switchMode(localStorage.getItem(MODE_KEY) || 'chat');
    refreshModelList();
    checkConnection();
    refreshServerPanels();
    setInterval(refreshServerPanels, 5000);
  </script>
</body>
</html>"#,
    )
}

async fn tags_handler(State(state): State<Arc<AppState>>) -> Json<TagsResponse> {
    let info = state.gguf_info.lock().unwrap().clone();
    let model_name = state.model_name.lock().unwrap().clone();
    Json(TagsResponse {
        models: vec![ModelTag {
            name: model_name.clone(),
            model: model_name,
            size: info.file_size,
            details: ModelDetails {
                format: "gguf".into(),
                family: info.architecture,
                parameter_size: format_parameter_size(info.parameter_count),
                quantization_level: info.quantization,
            },
        }],
    })
}

async fn show_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShowRequest>,
) -> Json<ShowResponse> {
    let info = state.gguf_info.lock().unwrap().clone();
    let model_name = state.model_name.lock().unwrap().clone();
    let requested_model = req
        .name
        .as_deref()
        .or(req.model.as_deref())
        .unwrap_or(model_name.as_str())
        .to_string();
    let mut model_info = serde_json::json!({
        "general.name": requested_model,
        "general.architecture": info.architecture.clone(),
        "general.context_length": info.context_length,
        "general.parameter_count": info.parameter_count,
        "hypura.turboquant.mode": state.turboquant.mode.as_str(),
        "hypura.turboquant.schema": state.turboquant.schema_label(),
        "hypura.turboquant.config_path": state.turboquant.source_label(),
        "hypura.turboquant.runtime_status": turboquant_runtime_status(
            state.turboquant.mode,
            state.turboquant.config.is_some(),
        ),
    });
    if let (Some(elt_loop), Some(object)) = (&info.elt_loop, model_info.as_object_mut()) {
        object.insert(
            "elt.loop.enabled".into(),
            serde_json::json!(elt_loop.enabled),
        );
        object.insert(
            "elt.loop.required".into(),
            serde_json::json!(elt_loop.required),
        );
        object.insert("elt.loop.L_min".into(), serde_json::json!(elt_loop.l_min));
        object.insert(
            "elt.loop.L_default".into(),
            serde_json::json!(elt_loop.l_default),
        );
        object.insert("elt.loop.L_max".into(), serde_json::json!(elt_loop.l_max));
        object.insert(
            "elt.loop_unit".into(),
            serde_json::json!(elt_loop.loop_unit_label()),
        );
        object.insert(
            "elt.loop.model_family".into(),
            serde_json::json!(elt_loop.family_label()),
        );
        object.insert(
            "elt.gguf.runtime_status".into(),
            serde_json::json!(elt_loop.gguf_runtime_status.as_deref().unwrap_or("unknown")),
        );
        object.insert(
            "hypura.elt_loop.runtime_gate".into(),
            serde_json::json!(
                elt_loop.runtime_gate_label(
                    crate::model::elt_loop::elt_loop_runtime_supported_from_env()
                )
            ),
        );
    }
    Json(ShowResponse {
        details: ModelDetails {
            format: "gguf".into(),
            family: info.architecture.clone(),
            parameter_size: format_parameter_size(info.parameter_count),
            quantization_level: info.quantization.clone(),
        },
        model_info,
    })
}

async fn kobold_model_handler(State(state): State<Arc<AppState>>) -> Json<KoboldModelResponse> {
    let model_name = state.model_name.lock().unwrap().clone();
    Json(KoboldModelResponse { result: model_name })
}

fn collect_gguf_models(
    model_dir: &PathBuf,
    active_path: &str,
) -> anyhow::Result<Vec<AvailableModelItem>> {
    let mut models = Vec::new();
    for entry in fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_gguf = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if !is_gguf {
            continue;
        }
        let full = path.to_string_lossy().to_string();
        let name = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| full.clone());
        models.push(AvailableModelItem {
            name,
            selected: full.eq_ignore_ascii_case(active_path),
            path: full,
        });
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

#[allow(dead_code)]
fn parse_u32_value(value: Option<&serde_json::Value>) -> Option<u32> {
    value
        .and_then(|value| value.as_u64().and_then(|raw| u32::try_from(raw).ok()))
        .or_else(|| {
            value.and_then(|value| {
                value
                    .as_str()
                    .and_then(|text| text.trim().parse::<u32>().ok())
            })
        })
}

#[allow(dead_code)]
fn launcher_profile_context(profile: &LauncherProfile, fallback: u32) -> u32 {
    parse_u32_value(profile.raw_config.get("contextsize"))
        .or_else(|| parse_u32_value(profile.raw_config.get("max_context_length")))
        .or_else(|| parse_u32_value(profile.raw_config.get("context")))
        .or_else(|| parse_u32_value(profile.raw_config.get("n_ctx")))
        .unwrap_or(fallback)
        .max(256)
}

#[allow(dead_code)]
fn launcher_profile_max_length(profile: &LauncherProfile, fallback: u32) -> u32 {
    profile
        .gendefaults
        .as_ref()
        .and_then(|defaults| parse_u32_value(defaults.get("max_length")))
        .unwrap_or(fallback)
        .max(1)
}

#[allow(dead_code)]
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

fn runtime_state_metadata_from_snapshot(
    state: &Arc<AppState>,
    slot: i64,
    snapshot: &ContextStateSnapshot,
) -> RuntimeStateSlotMetadata {
    let model_path = state
        .model_path
        .lock()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let model_identity = state.model_name.lock().unwrap().clone();
    let architecture = state.gguf_info.lock().unwrap().architecture.clone();
    let now = now_rfc3339();
    RuntimeStateSlotMetadata {
        slot,
        model_path,
        model_identity,
        architecture,
        context_size: state.default_context.load(Ordering::Relaxed),
        token_count: snapshot.token_count,
        byte_size: snapshot.state_bytes.len() as u64,
        created_at: now.clone(),
        updated_at: now,
    }
}

fn validate_runtime_state_compatibility(
    state: &Arc<AppState>,
    metadata: &RuntimeStateSlotMetadata,
) -> anyhow::Result<()> {
    let current_model_path = state
        .model_path
        .lock()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let current_architecture = state.gguf_info.lock().unwrap().architecture.clone();
    let current_context = state.default_context.load(Ordering::Relaxed);

    anyhow::ensure!(
        metadata
            .model_path
            .eq_ignore_ascii_case(&current_model_path),
        "runtime state slot model path does not match the active model"
    );
    anyhow::ensure!(
        metadata
            .architecture
            .eq_ignore_ascii_case(&current_architecture),
        "runtime state slot architecture does not match the active model"
    );
    anyhow::ensure!(
        metadata.context_size == current_context,
        "runtime state slot context size does not match the active model context"
    );
    Ok(())
}

async fn switch_loaded_model_runtime(
    state: Arc<AppState>,
    next_model_path: PathBuf,
    context: u32,
) -> anyhow::Result<ModelSwitchResponse> {
    anyhow::ensure!(
        !state.generation_in_progress.load(Ordering::Relaxed),
        "generation in progress; abort first"
    );
    anyhow::ensure!(next_model_path.exists(), "model path does not exist");
    let guarded_path = next_model_path.clone();
    let model_file_guard =
        tokio::task::spawn_blocking(move || GuardedModelFile::acquire(&guarded_path)).await??;
    let canonical_model_path = model_file_guard.canonical_path().to_path_buf();
    let initial_file_snapshot = model_file_guard.initial_snapshot().clone();

    let context = context.max(256);
    let path_for_setup = canonical_model_path.clone();
    let tq_mode = state.serve_turboquant_mode;
    let tq_config = state.serve_turboquant_config_path.clone();
    let bridge = state.serve_llama_bridge.clone();
    let residency_policy = state.serve_residency_policy;
    let tq_allow_exact_fallback = state.serve_tq_allow_exact_fallback;
    let mut setup = tokio::task::spawn_blocking(move || {
        crate::compute::inference::resolve_runtime_setup(
            &path_for_setup,
            context,
            tq_mode,
            tq_config.as_deref(),
            bridge,
            residency_policy,
            tq_allow_exact_fallback,
        )
    })
    .await??;
    if setup.triality.is_some() {
        setup.plan = crate::compute::inference::council_compatible_resident_plan(&setup.plan);
    }
    let urt_enabled = setup
        .triality
        .as_ref()
        .is_some_and(|triality| triality.urt_enabled);
    let next_model_sha256 = urt_enabled.then(|| initial_file_snapshot.sha256().to_string());

    let gguf_info = GgufInfo {
        file_size: initial_file_snapshot.size(),
        architecture: setup.metadata.architecture.clone(),
        parameter_count: setup.metadata.parameter_count,
        quantization: setup
            .metadata
            .quantization
            .clone()
            .unwrap_or_else(|| "unknown".into()),
        context_length: setup.metadata.context_length,
        elt_loop: setup.elt_loop.clone(),
    };

    let config = crate::compute::inference::InferenceConfig {
        n_ctx: context,
        triality: setup.triality.clone(),
        ..crate::compute::inference::InferenceConfig::default()
    };
    let n_gpu_layers = setup.n_gpu_layers;
    let plan = setup.plan.clone();
    let gguf = setup.gguf.clone();
    let turboquant = setup.turboquant.clone();
    let path_for_load = canonical_model_path.clone();

    let loaded_next = tokio::task::spawn_blocking(move || {
        crate::compute::inference::load_model(
            &path_for_load,
            &config,
            n_gpu_layers,
            &plan,
            &gguf,
            &turboquant,
        )
    })
    .await??;
    let model_file_guard = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        model_file_guard.verify_unchanged()?;
        Ok(model_file_guard)
    })
    .await??;

    let next_physical_name = loaded_next.model_name.clone();
    let mut loaded_guard = state.loaded_model.lock().unwrap();
    let mut name_guard = state.model_name.lock().unwrap();
    let mut path_guard = state.model_path.lock().unwrap();
    let mut hash_guard = state.model_sha256.lock().unwrap();
    let mut gguf_guard = state.gguf_info.lock().unwrap();
    *loaded_guard = loaded_next;
    *name_guard = switched_public_model_alias(
        &name_guard,
        &next_physical_name,
        state.model_name_is_explicit_alias,
    );
    let public_alias = name_guard.clone();
    *path_guard = canonical_model_path;
    *hash_guard = next_model_sha256;
    *gguf_guard = gguf_info;
    state.default_context.store(context, Ordering::Relaxed);
    drop(gguf_guard);
    drop(hash_guard);
    drop(path_guard);
    drop(name_guard);
    drop(loaded_guard);
    drop(model_file_guard);
    if let Some(session) = state.compat_session.as_ref() {
        session.lock().unwrap().reset_context()?;
    }

    push_gui_event(
        &state,
        "info",
        format!("model switched to {public_alias} (ctx {context})"),
    );

    Ok(ModelSwitchResponse {
        success: true,
        model: public_alias,
        context,
    })
}

async fn models_handler(State(state): State<Arc<AppState>>) -> Response {
    let active_path = state
        .model_path
        .lock()
        .unwrap()
        .to_string_lossy()
        .to_string();
    match collect_gguf_models(&state.model_dir, &active_path) {
        Ok(models) => Json(AvailableModelsResponse {
            models,
            active_model_path: active_path,
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("failed to scan models: {e}") })),
        )
            .into_response(),
    }
}

async fn model_switch_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelSwitchRequest>,
) -> Response {
    let next_model_path = PathBuf::from(req.path.trim());
    let context = req
        .context
        .unwrap_or_else(|| state.default_context.load(Ordering::Relaxed));
    match switch_loaded_model_runtime(state.clone(), next_model_path, context).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => {
            let message = error.to_string();
            let status = if message.contains("generation in progress") {
                StatusCode::CONFLICT
            } else {
                StatusCode::BAD_REQUEST
            };
            push_gui_event(&state, "error", format!("model switch failed: {message}"));
            (status, Json(serde_json::json!({ "error": message }))).into_response()
        }
    }
}

fn push_gui_event(state: &Arc<AppState>, level: &str, message: impl Into<String>) {
    let item = GuiEventItem {
        ts: now_rfc3339(),
        level: level.to_string(),
        message: message.into(),
    };
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Err(error) = storage.push_gui_event(&item) {
            tracing::warn!("failed to persist compat GUI event: {error}");
        }
    }
    let mut events = state.gui_events.lock().unwrap();
    events.push_front(item);
    while events.len() > 200 {
        events.pop_back();
    }
}

fn push_gui_history(
    state: &Arc<AppState>,
    mode: &str,
    model: String,
    prompt_chars: usize,
    output_chars: usize,
    tok_per_sec_avg: Option<f64>,
    total_ms: u64,
) {
    let item = GuiHistoryItem {
        ts: now_rfc3339(),
        mode: mode.to_string(),
        model,
        prompt_chars,
        output_chars,
        tok_per_sec_avg,
        total_ms,
    };
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Err(error) = storage.push_gui_history(&item) {
            tracing::warn!("failed to persist compat GUI history: {error}");
        }
    }
    let mut history = state.gui_history.lock().unwrap();
    history.push_front(item);
    while history.len() > 200 {
        history.pop_back();
    }
}

fn parse_slot_value(value: Option<&serde_json::Value>) -> Option<i64> {
    value.and_then(|slot| {
        slot.as_i64().or_else(|| {
            slot.as_str()
                .and_then(|text| text.trim().parse::<i64>().ok())
        })
    })
}

async fn kobold_preload_story_handler(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let payload = state
        .compat_storage
        .as_ref()
        .and_then(|storage| storage.get_preload_story_json().ok())
        .unwrap_or_else(|| serde_json::json!({}));
    Json(payload)
}

async fn kobold_savedata_list_handler(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let slots = state
        .compat_storage
        .as_ref()
        .and_then(|storage| storage.list_save_slot_titles().ok())
        .unwrap_or_default();
    Json(serde_json::json!(slots))
}

async fn kobold_savedata_save_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    let Some(storage) = state.compat_storage.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "error": "SaveDataFile not enabled!" })),
        )
            .into_response();
    };
    if !storage.savedata_enabled() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "error": "SaveDataFile not enabled!" })),
        )
            .into_response();
    }

    let slot = match parse_slot_value(req.get("slot")) {
        Some(slot) => slot,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "success": false, "error": "No story submitted or invalid slot!" })),
            )
                .into_response();
        }
    };
    let format = req
        .get("format")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let title = req
        .get("title")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let data = req
        .get("data")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    match storage.save_slot(slot, format, title, data) {
        Ok(()) => Json(serde_json::json!({ "success": true, "error": "" })).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "error": error.to_string() })),
        )
            .into_response(),
    }
}

async fn kobold_savedata_load_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    let Some(storage) = state.compat_storage.as_ref() else {
        return Json(serde_json::json!({ "success": false, "data": serde_json::Value::Null }))
            .into_response();
    };
    let Some(slot) = parse_slot_value(req.get("slot")) else {
        return Json(serde_json::json!({ "success": false, "data": serde_json::Value::Null }))
            .into_response();
    };

    match storage.load_save_slot(slot) {
        Ok(Some(record)) => Json(serde_json::json!({
            "success": true,
            "data": {
                "title": record.title,
                "format": record.format,
                "data": record.data,
            }
        }))
        .into_response(),
        Ok(None) => Json(serde_json::json!({ "success": false, "data": serde_json::Value::Null }))
            .into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "error": error.to_string(), "data": serde_json::Value::Null })),
        )
            .into_response(),
    }
}

async fn kobold_admin_list_options_handler(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let items = state
        .compat_storage
        .as_ref()
        .and_then(|storage| storage.list_admin_options().ok())
        .unwrap_or_default();
    Json(serde_json::json!(items))
}

async fn kobold_admin_reload_config_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    if compat_features_snapshot(&state).admin == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "error": "admin mode is not enabled" })),
        )
            .into_response();
    }
    let Some(_storage) = state.compat_storage.as_ref() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "error": "admindir is not configured" })),
        )
            .into_response();
    };
    if state.generation_in_progress.load(Ordering::Relaxed) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "success": false, "error": "generation in progress; abort first" })),
        )
            .into_response();
    }
    let filename = req
        .get("filename")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let Some(filename) = filename else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "success": false, "error": "filename is required" })),
        )
            .into_response();
    };
    let baseconfig = req
        .get("baseconfig")
        .or_else(|| req.get("overrideconfig"))
        .and_then(serde_json::Value::as_str);
    let Some(control_client) = state.compat_control_client.as_ref() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "success": false, "error": "compat control channel is not available" })),
        )
            .into_response();
    };
    match control_client
        .send_command(CompatSupervisorCommand::ReloadConfig {
            filename: filename.to_string(),
            baseconfig: baseconfig.map(str::to_string),
        })
        .await
    {
        Ok(()) => Json(serde_json::json!({ "success": true })).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "success": false, "error": format!("failed to dispatch reload: {error}") })),
        )
            .into_response(),
    }
}

async fn kobold_admin_save_state_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    if compat_features_snapshot(&state).admin == 0 {
        return Json(serde_json::json!({
            "success": false,
            "new_state_size": 0,
            "new_tokens": 0,
            "error": "admin mode is not enabled"
        }))
        .into_response();
    }
    if state.generation_in_progress.load(Ordering::Relaxed) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "success": false,
                "new_state_size": 0,
                "new_tokens": 0,
                "error": "generation in progress; abort first"
            })),
        )
            .into_response();
    }
    let Some(storage) = state.compat_storage.as_ref() else {
        return Json(serde_json::json!({
            "success": false,
            "new_state_size": 0,
            "new_tokens": 0,
            "error": "compat storage is not configured"
        }))
        .into_response();
    };
    let slot = parse_slot_value(req.get("slot")).unwrap_or(0);
    let Some(session) = state.compat_session.as_ref() else {
        return Json(serde_json::json!({
            "success": false,
            "new_state_size": 0,
            "new_tokens": 0,
            "error": "compat runtime session is not configured"
        }))
        .into_response();
    };
    let snapshot = match session.lock().unwrap().current_runtime_metadata() {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => {
            return Json(serde_json::json!({
                "success": false,
                "new_state_size": 0,
                "new_tokens": 0,
                "error": "no runtime state is available yet; generate or load a state first"
            }))
            .into_response();
        }
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "success": false,
                    "new_state_size": 0,
                    "new_tokens": 0,
                    "error": error.to_string()
                })),
            )
                .into_response();
        }
    };
    let metadata = runtime_state_metadata_from_snapshot(&state, slot, &snapshot);
    match storage.save_runtime_state_slot(
        slot,
        &metadata,
        &snapshot.token_ids,
        &snapshot.state_bytes,
    ) {
        Ok(()) => {
            push_gui_event(
                &state,
                "info",
                format!("runtime state saved to slot {slot}"),
            );
            Json(serde_json::json!({
                "success": true,
                "new_state_size": snapshot.state_bytes.len(),
                "new_tokens": snapshot.token_count,
            }))
            .into_response()
        }
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "success": false,
                "new_state_size": 0,
                "new_tokens": 0,
                "error": error.to_string()
            })),
        )
            .into_response(),
    }
}

async fn kobold_admin_load_state_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    if compat_features_snapshot(&state).admin == 0 {
        return Json(serde_json::json!({
            "success": false,
            "new_tokens": 0,
            "error": "admin mode is not enabled"
        }))
        .into_response();
    }
    if state.generation_in_progress.load(Ordering::Relaxed) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "success": false,
                "new_tokens": 0,
                "error": "generation in progress; abort first"
            })),
        )
            .into_response();
    }
    let Some(storage) = state.compat_storage.as_ref() else {
        return Json(serde_json::json!({
            "success": false,
            "new_tokens": 0,
            "error": "compat storage is not configured"
        }))
        .into_response();
    };
    let slot = parse_slot_value(req.get("slot")).unwrap_or(0);
    let record = match storage.load_runtime_state_slot(slot) {
        Ok(Some(record)) => record,
        Ok(None) => {
            return Json(serde_json::json!({
                "success": false,
                "new_tokens": 0,
                "error": "runtime state slot is empty"
            }))
            .into_response();
        }
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "success": false,
                    "new_tokens": 0,
                    "error": error.to_string()
                })),
            )
                .into_response();
        }
    };
    if let Err(error) = validate_runtime_state_compatibility(&state, &record.metadata) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "success": false,
                "new_tokens": 0,
                "error": error.to_string()
            })),
        )
            .into_response();
    }

    let Some(session) = state.compat_session.as_ref() else {
        return Json(serde_json::json!({
            "success": false,
            "new_tokens": 0,
            "error": "compat runtime session is not configured"
        }))
        .into_response();
    };
    if let Err(error) = session
        .lock()
        .unwrap()
        .load_state_snapshot(ContextStateSnapshot {
            token_ids: record.token_ids,
            token_count: record.metadata.token_count,
            state_bytes: record.state_bytes,
        })
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "success": false,
                "new_tokens": 0,
                "error": error.to_string()
            })),
        )
            .into_response();
    }
    push_gui_event(
        &state,
        "info",
        format!("runtime state loaded from slot {slot}"),
    );
    Json(serde_json::json!({
        "success": true,
        "new_tokens": record.metadata.token_count,
    }))
    .into_response()
}

async fn gui_presets_list_handler(
    State(state): State<Arc<AppState>>,
) -> Json<GuiPresetListResponse> {
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Ok(presets) = storage.list_gui_presets() {
            return Json(GuiPresetListResponse { presets });
        }
    }
    let mut presets: Vec<GuiPresetItem> = state
        .gui_presets
        .lock()
        .unwrap()
        .values()
        .cloned()
        .collect();
    presets.sort_by(|a, b| a.name.cmp(&b.name));
    Json(GuiPresetListResponse { presets })
}

async fn gui_presets_save_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GuiPresetSaveRequest>,
) -> Response {
    let name = req.name.trim();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name is required" })),
        )
            .into_response();
    }
    let preset = GuiPresetItem {
        name: name.to_string(),
        payload: req.payload,
        updated_at: now_rfc3339(),
    };
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Err(error) = storage.save_gui_preset(&preset) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error.to_string() })),
            )
                .into_response();
        }
    }
    state
        .gui_presets
        .lock()
        .unwrap()
        .insert(name.to_string(), preset);
    push_gui_event(&state, "info", format!("preset saved: {name}"));
    Json(serde_json::json!({ "success": true })).into_response()
}

async fn gui_presets_delete_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GuiPresetDeleteRequest>,
) -> Response {
    let name = req.name.trim();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name is required" })),
        )
            .into_response();
    }
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Err(error) = storage.delete_gui_preset(name) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error.to_string() })),
            )
                .into_response();
        }
    }
    state.gui_presets.lock().unwrap().remove(name);
    push_gui_event(&state, "info", format!("preset deleted: {name}"));
    Json(serde_json::json!({ "success": true })).into_response()
}

async fn gui_history_handler(State(state): State<Arc<AppState>>) -> Json<GuiHistoryResponse> {
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Ok(items) = storage.list_gui_history(200) {
            return Json(GuiHistoryResponse { items });
        }
    }
    let items = state.gui_history.lock().unwrap().iter().cloned().collect();
    Json(GuiHistoryResponse { items })
}

async fn gui_events_handler(State(state): State<Arc<AppState>>) -> Json<GuiEventsResponse> {
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Ok(items) = storage.list_gui_events(200) {
            return Json(GuiEventsResponse { items });
        }
    }
    let items = state.gui_events.lock().unwrap().iter().cloned().collect();
    Json(GuiEventsResponse { items })
}

async fn ui_theme_get_handler(State(state): State<Arc<AppState>>) -> Json<UiThemeResponse> {
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Ok(theme) = storage.get_ui_theme() {
            return Json(UiThemeResponse { theme });
        }
    }
    Json(UiThemeResponse {
        theme: state.ui_theme.lock().unwrap().clone(),
    })
}

async fn ui_theme_set_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UiThemeUpdateRequest>,
) -> Response {
    let t = req.theme.trim().to_ascii_lowercase();
    let next = match t.as_str() {
        "light" | "dark" | "classic" => t,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "theme must be light|dark|classic" })),
            )
                .into_response();
        }
    };
    if let Some(storage) = state.compat_storage.as_ref() {
        if let Err(error) = storage.set_ui_theme(&next) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": error.to_string() })),
            )
                .into_response();
        }
    }
    *state.ui_theme.lock().unwrap() = next.clone();
    set_process_env_var("HYPURA_UI_THEME", &next);
    push_gui_event(&state, "info", format!("ui theme changed: {next}"));
    Json(serde_json::json!({ "success": true, "theme": next })).into_response()
}

fn begin_generation(state: &Arc<AppState>) -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut slot = state.active_cancel.lock().unwrap();
        *slot = Some(cancel.clone());
    }
    state.generation_in_progress.store(true, Ordering::Relaxed);
    cancel
}

fn end_generation(state: &Arc<AppState>) {
    state.generation_in_progress.store(false, Ordering::Relaxed);
    let mut slot = state.active_cancel.lock().unwrap();
    *slot = None;
}

fn compat_features_snapshot(state: &Arc<AppState>) -> CompatFeatureFlags {
    *state.compat_features.lock().unwrap()
}

fn spawn_generation_task(
    state: Arc<AppState>,
    history_label: &'static str,
    prompt: String,
    sampling: crate::compute::ffi::SamplingParams,
    sampler_order: Option<Vec<i32>>,
    stop_sequences: Vec<String>,
) -> (
    mpsc::UnboundedReceiver<crate::compute::inference::GeneratedToken>,
    oneshot::Receiver<GenerationResult>,
) {
    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();
    let loaded = state.loaded_model.clone();
    let compat_session = state.compat_session.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();

    tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = if let Some(session) = compat_session {
            let mut session = session.lock().unwrap();
            session.generate(params)
        } else {
            let mut model = loaded.lock().unwrap();
            crate::compute::inference::generate_from_loaded(&mut model, params)
        };
        end_generation(&state_for_task);
        match result {
            Ok(gen_result) => {
                let model_name = state_for_task.model_name.lock().unwrap().clone();
                push_gui_history(
                    &state_for_task,
                    history_label,
                    model_name,
                    prompt.chars().count(),
                    gen_result.text.chars().count(),
                    Some(gen_result.tok_per_sec_avg),
                    started.elapsed().as_millis() as u64,
                );
                record_compat_text_generation(&state_for_task, &sampling, &gen_result);
                let _ = result_tx.send(gen_result);
            }
            Err(error) => {
                tracing::error!("generation error: {error}");
                push_gui_event(
                    &state_for_task,
                    "error",
                    format!("generation failed: {error}"),
                );
            }
        }
    });

    (token_rx, result_rx)
}

async fn proxy_multimodal_request(
    state: Arc<AppState>,
    feature_name: &'static str,
    internal_path: &'static str,
    upstream_path: &'static str,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(client) = state.compat_control_client.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(compat::feature_unavailable_error(feature_name)),
        )
            .into_response();
    };
    let mut headers = headers;
    if let Ok(value) = header::HeaderValue::from_str(upstream_path) {
        headers.insert("x-hypura-upstream-path", value);
    }
    match client
        .proxy_request(internal_path, method, &headers, body)
        .await
    {
        Ok(response) => response,
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({ "error": error.to_string() })),
        )
            .into_response(),
    }
}

async fn proxy_embeddings_request(
    state: Arc<AppState>,
    internal_path: &'static str,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let features = compat_features_snapshot(&state);
    if !features.embeddings {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(compat::openai_error(
                "embeddings model is not available in the current compatibility runtime",
                "unsupported_feature",
                "embeddings_unavailable",
            )),
        )
            .into_response();
    }
    let Some(client) = state.compat_control_client.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(compat::openai_error(
                "embeddings control plane is not available in the current compatibility runtime",
                "server_error",
                "embeddings_control_plane_unavailable",
            )),
        )
            .into_response();
    };
    match client
        .proxy_request(internal_path, method, &headers, body)
        .await
    {
        Ok(response) => response,
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(compat::openai_error(
                &format!("failed to proxy embeddings request: {error}"),
                "server_error",
                "embeddings_proxy_failed",
            )),
        )
            .into_response(),
    }
}

async fn generate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Response {
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    apply_turboquant_runtime_overrides(&req.options);
    let sampling = build_sampling(&req.options);
    let stop_sequences = req.options.stop.clone().unwrap_or_default();
    let prompt = req.prompt;
    let model_name = state.model_name.lock().unwrap().clone();
    let sampler_order = req.options.sampler_order.clone();
    let (token_rx, result_rx) = spawn_generation_task(
        state.clone(),
        "generate",
        prompt.clone(),
        sampling,
        sampler_order,
        stop_sequences,
    );

    if req.stream {
        let body = streaming::ndjson_generate_stream(
            model_name,
            token_rx,
            result_rx,
            request_start,
            load_duration_ns,
        );
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            body,
        )
            .into_response()
    } else {
        // Non-streaming: collect all tokens, return single JSON
        let result = collect_generate(
            model_name,
            token_rx,
            result_rx,
            request_start,
            load_duration_ns,
        )
        .await;
        Json(result).into_response()
    }
}

async fn kobold_generate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<KoboldGenerateRequest>,
) -> Json<KoboldGenerateResponse> {
    apply_turboquant_runtime_overrides_kobold(&req);
    let mut sampling = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = req.temperature {
        sampling.temperature = t;
    }
    if let Some(k) = req.top_k {
        sampling.top_k = k;
    }
    if let Some(a) = req.top_a {
        sampling.top_a = a;
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p;
    }
    if let Some(tfs) = req.tfs {
        sampling.tfs = tfs;
    }
    if let Some(typ) = req.typical {
        sampling.typical = typ;
    }
    if let Some(mp) = req.min_p {
        sampling.min_p = mp;
    }
    if let Some(n) = req.max_length {
        sampling.max_tokens = n;
    }
    if let Some(rep_pen) = req.rep_pen {
        sampling.repeat_penalty = rep_pen;
    }
    if let Some(rep_range) = req.rep_pen_range {
        sampling.repeat_last_n = rep_range;
    }
    let sampler_order = req.sampler_order.clone();
    let stop_sequences = req.stop_sequence.unwrap_or_default();

    let prompt = req.prompt;
    let (mut token_rx, _result_rx) = spawn_generation_task(
        state.clone(),
        "kobold-generate",
        prompt,
        sampling,
        sampler_order,
        stop_sequences,
    );

    let mut full_response = String::new();
    while let Some(token) = token_rx.recv().await {
        full_response.push_str(&token.text);
    }

    Json(KoboldGenerateResponse {
        results: vec![KoboldGenerateResult {
            text: full_response,
        }],
    })
}

async fn kobold_generate_stream_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<KoboldGenerateRequest>,
) -> Response {
    apply_turboquant_runtime_overrides_kobold(&req);
    let mut sampling = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = req.temperature {
        sampling.temperature = t;
    }
    if let Some(k) = req.top_k {
        sampling.top_k = k;
    }
    if let Some(a) = req.top_a {
        sampling.top_a = a;
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p;
    }
    if let Some(tfs) = req.tfs {
        sampling.tfs = tfs;
    }
    if let Some(typ) = req.typical {
        sampling.typical = typ;
    }
    if let Some(mp) = req.min_p {
        sampling.min_p = mp;
    }
    if let Some(n) = req.max_length {
        sampling.max_tokens = n;
    }
    if let Some(rep_pen) = req.rep_pen {
        sampling.repeat_penalty = rep_pen;
    }
    if let Some(rep_range) = req.rep_pen_range {
        sampling.repeat_last_n = rep_range;
    }
    let sampler_order = req.sampler_order.clone();
    let stop_sequences = req.stop_sequence.unwrap_or_default();
    let prompt = req.prompt;

    let (mut token_rx, result_rx) = spawn_generation_task(
        state.clone(),
        "kobold-stream",
        prompt,
        sampling,
        sampler_order,
        stop_sequences,
    );
    let model_name = state.model_name.lock().unwrap().clone();
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);
    tokio::spawn(async move {
        while let Some(token) = token_rx.recv().await {
            let line = serde_json::json!({
                "token": token.text,
                "done": false,
                "model": model_name.clone(),
            })
            .to_string()
                + "\n";
            if tx.send(Ok(line)).await.is_err() {
                return;
            }
        }
        let total_ns = request_start.elapsed().as_nanos() as u64;
        let result = result_rx.await.ok();
        let line = serde_json::json!({
            "token": "",
            "done": true,
            "model": model_name.clone(),
            "total_duration": total_ns,
            "load_duration": load_duration_ns,
            "eval_count": result.as_ref().map(|r| r.tokens_generated),
            "tok_per_sec_avg": result.as_ref().map(|r| r.tok_per_sec_avg),
            "prompt_eval_ms": result.as_ref().map(|r| r.prompt_eval_ms),
        })
        .to_string()
            + "\n";
        let _ = tx.send(Ok(line)).await;
    });

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)),
    )
        .into_response()
}

async fn kobold_generate_check_handler(
    State(state): State<Arc<AppState>>,
) -> Json<KoboldGenerateResponse> {
    let status = if state.generation_in_progress.load(Ordering::Relaxed) {
        "busy"
    } else {
        ""
    };
    Json(KoboldGenerateResponse {
        results: vec![KoboldGenerateResult {
            text: status.to_string(),
        }],
    })
}

async fn kobold_abort_handler(State(state): State<Arc<AppState>>) -> Json<KoboldAbortResponse> {
    let mut success = false;
    if let Some(flag) = state.active_cancel.lock().unwrap().as_ref() {
        flag.store(true, Ordering::Relaxed);
        success = true;
    }
    Json(KoboldAbortResponse { success })
}

async fn kobold_true_max_context_length_handler(
    State(state): State<Arc<AppState>>,
) -> Json<KoboldTrueMaxContextLengthResponse> {
    Json(KoboldTrueMaxContextLengthResponse {
        value: state.default_context.load(Ordering::Relaxed),
    })
}

async fn kobold_websearch_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !compat_features_snapshot(&state).websearch {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(compat::feature_unavailable_error("websearch")),
        )
            .into_response();
    }
    proxy_multimodal_request(
        state,
        "websearch",
        "/builtin/api/extra/websearch",
        "/api/extra/websearch",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn txt2img_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_multimodal_request(
        state,
        "txt2img",
        "/multimodal/sdapi/v1/txt2img",
        "/sdapi/v1/txt2img",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn img2img_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_multimodal_request(
        state,
        "txt2img",
        "/multimodal/sdapi/v1/img2img",
        "/sdapi/v1/img2img",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn interrogate_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_multimodal_request(
        state,
        "txt2img",
        "/multimodal/sdapi/v1/interrogate",
        "/sdapi/v1/interrogate",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn upscale_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_multimodal_request(
        state,
        "txt2img",
        "/multimodal/sdapi/v1/upscale",
        "/sdapi/v1/upscale",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn sd_options_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    proxy_multimodal_request(
        state,
        "txt2img",
        "/multimodal/sdapi/v1/options",
        "/sdapi/v1/options",
        Method::GET,
        headers,
        Bytes::new(),
    )
    .await
}

async fn sd_models_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    proxy_multimodal_request(
        state,
        "txt2img",
        "/multimodal/sdapi/v1/sd-models",
        "/sdapi/v1/sd-models",
        Method::GET,
        headers,
        Bytes::new(),
    )
    .await
}

async fn transcribe_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_multimodal_request(
        state,
        "transcribe",
        "/multimodal/api/extra/transcribe",
        "/api/extra/transcribe",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn audio_transcriptions_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_multimodal_request(
        state,
        "transcribe",
        "/multimodal/v1/audio/transcriptions",
        "/v1/audio/transcriptions",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn tts_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_multimodal_request(
        state,
        "tts",
        "/multimodal/api/extra/tts",
        "/api/extra/tts",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn audio_speech_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_multimodal_request(
        state,
        "tts",
        "/multimodal/v1/audio/speech",
        "/v1/audio/speech",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn speakers_list_proxy_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    proxy_multimodal_request(
        state,
        "speakers_list",
        "/multimodal/speakers_list",
        "/speakers_list",
        Method::GET,
        headers,
        Bytes::new(),
    )
    .await
}

async fn openai_embeddings_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_embeddings_request(
        state,
        "/embeddings/v1/embeddings",
        Method::POST,
        headers,
        body,
    )
    .await
}

async fn openai_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAiCompletionRequest>,
) -> Response {
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;
    let prompt = req.prompt.to_prompt_text();
    let stop_sequences = req
        .stop
        .as_ref()
        .map(OpenAiStopInput::to_stop_sequences)
        .unwrap_or_default();
    let sampling = build_sampling_from_openai_completion(
        &req,
        state.compat_default_max_length.load(Ordering::Relaxed),
    );
    let sampler_order = req.sampler_order.clone();
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_name.lock().unwrap().clone());
    let created = chrono::Utc::now().timestamp();
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());

    let (mut token_rx, result_rx) = spawn_generation_task(
        state.clone(),
        "openai-completion",
        prompt.clone(),
        sampling,
        sampler_order,
        stop_sequences,
    );

    if req.stream {
        let request_id_for_stream = request_id.clone();
        let model_name_for_stream = model_name.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);
        tokio::spawn(async move {
            while let Some(token) = token_rx.recv().await {
                let chunk = serde_json::json!({
                    "id": request_id_for_stream,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name_for_stream,
                    "choices": [{
                        "text": token.text,
                        "index": 0,
                        "finish_reason": serde_json::Value::Null,
                    }]
                });
                if tx.send(Ok(format!("data: {}\n\n", chunk))).await.is_err() {
                    return;
                }
            }
            let result = result_rx.await.ok();
            let final_chunk = serde_json::json!({
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "finish_reason": "stop",
                }],
                "usage": result.as_ref().map(|r| compat::openai_usage(r.prompt_tokens, r.tokens_generated)),
                "total_duration": request_start.elapsed().as_nanos() as u64,
                "load_duration": load_duration_ns,
            });
            let _ = tx.send(Ok(format!("data: {}\n\n", final_chunk))).await;
            let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;
        });
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/event-stream")],
            axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)),
        )
            .into_response()
    } else {
        let mut full_response = String::new();
        while let Some(token) = token_rx.recv().await {
            full_response.push_str(&token.text);
        }
        let result = result_rx.await.ok();
        let body = compat::openai_completion_response(
            &request_id,
            created,
            &model_name,
            &full_response,
            result.as_ref().map(|r| r.prompt_tokens).unwrap_or(0),
            result.as_ref().map(|r| r.tokens_generated).unwrap_or(0),
        );
        Json(body).into_response()
    }
}

async fn openai_chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAiChatCompletionRequest>,
) -> Response {
    if let Some(extension) = council_extension(req.hypura.as_ref()) {
        let council_request = TrialityCouncilRequest {
            model: req.model.clone(),
            prompt: None,
            messages: Some(req.messages.clone()),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            seed: req.seed,
            parallelism: None,
            attention_consensus: false,
            cross_score: true,
            synthesis: false,
            aha: true,
            trace: extension.trace,
            stream: req.stream,
        };
        let validated = match validate_council_request(&state, council_request) {
            Ok(value) => value,
            Err(error) => return error.into_response(),
        };
        return match execute_triality_council(state, validated).await {
            Ok(response) => openai_council_chat_response(response, req.stream),
            Err(error) => error.into_response(),
        };
    }

    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(OpenAiChatMessage::to_chat_message)
        .collect();
    let prompt = format_chat_prompt(&messages);
    let stop_sequences = req
        .stop
        .as_ref()
        .map(OpenAiStopInput::to_stop_sequences)
        .unwrap_or_default();
    let sampling = build_sampling_from_openai_chat(
        &req,
        state.compat_default_max_length.load(Ordering::Relaxed),
    );
    let sampler_order = req.sampler_order.clone();
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_name.lock().unwrap().clone());
    let created = chrono::Utc::now().timestamp();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());

    let (mut token_rx, result_rx) = spawn_generation_task(
        state.clone(),
        "openai-chat",
        prompt.clone(),
        sampling,
        sampler_order,
        stop_sequences,
    );

    if req.stream {
        let request_id_for_stream = request_id.clone();
        let model_name_for_stream = model_name.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);
        tokio::spawn(async move {
            let first_chunk = serde_json::json!({
                "id": request_id_for_stream,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name_for_stream,
                "choices": [{
                    "index": 0,
                    "delta": { "role": "assistant", "content": "" },
                    "finish_reason": serde_json::Value::Null,
                }]
            });
            if tx
                .send(Ok(format!("data: {}\n\n", first_chunk)))
                .await
                .is_err()
            {
                return;
            }
            while let Some(token) = token_rx.recv().await {
                let chunk = serde_json::json!({
                    "id": request_id_for_stream,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name_for_stream,
                    "choices": [{
                        "index": 0,
                        "delta": { "content": token.text },
                        "finish_reason": serde_json::Value::Null,
                    }]
                });
                if tx.send(Ok(format!("data: {}\n\n", chunk))).await.is_err() {
                    return;
                }
            }
            let result = result_rx.await.ok();
            let final_chunk = serde_json::json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": result.as_ref().map(|r| compat::openai_usage(r.prompt_tokens, r.tokens_generated)),
                "total_duration": request_start.elapsed().as_nanos() as u64,
                "load_duration": load_duration_ns,
            });
            let _ = tx.send(Ok(format!("data: {}\n\n", final_chunk))).await;
            let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;
        });
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/event-stream")],
            axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)),
        )
            .into_response()
    } else {
        let mut full_response = String::new();
        while let Some(token) = token_rx.recv().await {
            full_response.push_str(&token.text);
        }
        let result = result_rx.await.ok();
        let body = compat::openai_chat_response(
            &request_id,
            created,
            &model_name,
            &full_response,
            result.as_ref().map(|r| r.prompt_tokens).unwrap_or(0),
            result.as_ref().map(|r| r.tokens_generated).unwrap_or(0),
        );
        Json(body).into_response()
    }
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    if let Some(extension) = council_extension(req.hypura.as_ref()) {
        let messages = req
            .messages
            .iter()
            .map(|message| OpenAiChatMessage {
                role: message.role.clone(),
                content: Some(OpenAiMessageContent::Text(message.content.clone())),
            })
            .collect();
        let council_request = TrialityCouncilRequest {
            model: Some(req.model.clone()),
            prompt: None,
            messages: Some(messages),
            max_tokens: req.options.num_predict,
            temperature: req.options.temperature,
            seed: req.options.seed,
            parallelism: None,
            attention_consensus: false,
            cross_score: true,
            synthesis: false,
            aha: true,
            trace: extension.trace,
            stream: req.stream,
        };
        let validated = match validate_council_request(&state, council_request) {
            Ok(value) => value,
            Err(error) => return error.into_response(),
        };
        return match execute_triality_council(state, validated).await {
            Ok(response) => ollama_council_chat_response(response, req.stream),
            Err(error) => error.into_response(),
        };
    }

    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    apply_turboquant_runtime_overrides(&req.options);
    let sampling = build_sampling(&req.options);
    let stop_sequences = req.options.stop.clone().unwrap_or_default();
    let prompt = format_chat_prompt(&req.messages);
    let model_name = state.model_name.lock().unwrap().clone();
    let sampler_order = req.options.sampler_order.clone();
    let (token_rx, result_rx) = spawn_generation_task(
        state.clone(),
        "chat",
        prompt,
        sampling,
        sampler_order,
        stop_sequences,
    );

    if req.stream {
        let body = streaming::ndjson_chat_stream(
            model_name,
            token_rx,
            result_rx,
            request_start,
            load_duration_ns,
        );
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson")],
            body,
        )
            .into_response()
    } else {
        let result = collect_chat(
            model_name,
            token_rx,
            result_rx,
            request_start,
            load_duration_ns,
        )
        .await;
        Json(result).into_response()
    }
}

fn council_extension(
    extension: Option<&HypuraRequestExtension>,
) -> Option<&HypuraRequestExtension> {
    extension.filter(|value| value.triality_council)
}

// ── Helpers ──

fn build_sampling(opts: &GenerateOptions) -> crate::compute::ffi::SamplingParams {
    let mut s = crate::compute::ffi::SamplingParams::default();
    if let Some(t) = opts.temperature {
        s.temperature = t;
    }
    if let Some(k) = opts.top_k {
        s.top_k = k;
    }
    if let Some(a) = opts.top_a {
        s.top_a = a;
    }
    if let Some(p) = opts.top_p {
        s.top_p = p;
    }
    if let Some(tfs) = opts.tfs {
        s.tfs = tfs;
    }
    if let Some(typ) = opts.typical {
        s.typical = typ;
    }
    if let Some(mp) = opts.min_p {
        s.min_p = mp;
    }
    if let Some(rp) = opts.repeat_penalty {
        s.repeat_penalty = rp;
    }
    if let Some(rn) = opts.repeat_last_n {
        s.repeat_last_n = rn;
    }
    if let Some(n) = opts.num_predict {
        s.max_tokens = n;
    }
    if let Some(seed) = opts.seed {
        s.seed = seed;
    }
    s
}

fn build_sampling_from_openai_completion(
    req: &OpenAiCompletionRequest,
    default_max_length: u32,
) -> crate::compute::ffi::SamplingParams {
    let mut s = crate::compute::ffi::SamplingParams::default();
    s.max_tokens = req.max_tokens.unwrap_or(default_max_length);
    if let Some(t) = req.temperature {
        s.temperature = t;
    }
    if let Some(p) = req.top_p {
        s.top_p = p;
    }
    if let Some(k) = req.top_k {
        s.top_k = k;
    }
    if let Some(a) = req.top_a {
        s.top_a = a;
    }
    if let Some(tfs) = req.tfs {
        s.tfs = tfs;
    }
    if let Some(typ) = req.typical {
        s.typical = typ;
    }
    if let Some(mp) = req.min_p {
        s.min_p = mp;
    }
    if let Some(seed) = req.seed {
        s.seed = seed;
    }
    if let Some(v) = req.presence_penalty {
        s.repeat_penalty = (1.0 + v).max(0.1);
    }
    if let Some(v) = req.frequency_penalty {
        s.repeat_penalty = s.repeat_penalty.max((1.0 + v).max(0.1));
    }
    s
}

fn build_sampling_from_openai_chat(
    req: &OpenAiChatCompletionRequest,
    default_max_length: u32,
) -> crate::compute::ffi::SamplingParams {
    let mut s = crate::compute::ffi::SamplingParams::default();
    s.max_tokens = req.max_tokens.unwrap_or(default_max_length);
    if let Some(t) = req.temperature {
        s.temperature = t;
    }
    if let Some(p) = req.top_p {
        s.top_p = p;
    }
    if let Some(k) = req.top_k {
        s.top_k = k;
    }
    if let Some(a) = req.top_a {
        s.top_a = a;
    }
    if let Some(tfs) = req.tfs {
        s.tfs = tfs;
    }
    if let Some(typ) = req.typical {
        s.typical = typ;
    }
    if let Some(mp) = req.min_p {
        s.min_p = mp;
    }
    if let Some(seed) = req.seed {
        s.seed = seed;
    }
    if let Some(v) = req.presence_penalty {
        s.repeat_penalty = (1.0 + v).max(0.1);
    }
    if let Some(v) = req.frequency_penalty {
        s.repeat_penalty = s.repeat_penalty.max((1.0 + v).max(0.1));
    }
    s
}

fn record_compat_text_generation(
    state: &Arc<AppState>,
    sampling: &crate::compute::ffi::SamplingParams,
    result: &GenerationResult,
) {
    let mut compat_perf = state.compat_perf.lock().unwrap();
    compat::record_text_generation(&mut compat_perf, sampling, result);
}

fn set_process_env_var<K: AsRef<std::ffi::OsStr>, V: AsRef<std::ffi::OsStr>>(key: K, value: V) {
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_process_env_var<K: AsRef<std::ffi::OsStr>>(key: K) {
    unsafe {
        std::env::remove_var(key);
    }
}

fn apply_turboquant_runtime_overrides(opts: &GenerateOptions) {
    if let Some(v) = opts.tq_so8_off {
        set_process_env_var("LLAMA_TURBOQUANT_SO8", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_so8_learned {
        set_process_env_var("LLAMA_TURBOQUANT_SO8_LEARNED", if v { "1" } else { "0" });
    }
    if let Some(v) = opts.tq_triality_off {
        set_process_env_var("LLAMA_TURBOQUANT_TRIALITY", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_triality_mix {
        set_process_env_var("LLAMA_TURBOQUANT_TRIALITY_MIX", format!("{v:.3}"));
    }
    if let Some(v) = opts.tq_rotation_seed {
        set_process_env_var("LLAMA_TURBOQUANT_ROTATION_SEED", v.to_string());
    }
    if let Some(path) = &opts.tq_artifact {
        if path.trim().is_empty() {
            remove_process_env_var("LLAMA_TURBOQUANT_ARTIFACT");
        } else {
            set_process_env_var("LLAMA_TURBOQUANT_ARTIFACT", path);
        }
    }
}

fn apply_turboquant_runtime_overrides_kobold(opts: &KoboldGenerateRequest) {
    if let Some(v) = opts.tq_so8_off {
        set_process_env_var("LLAMA_TURBOQUANT_SO8", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_so8_learned {
        set_process_env_var("LLAMA_TURBOQUANT_SO8_LEARNED", if v { "1" } else { "0" });
    }
    if let Some(v) = opts.tq_triality_off {
        set_process_env_var("LLAMA_TURBOQUANT_TRIALITY", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_triality_mix {
        set_process_env_var("LLAMA_TURBOQUANT_TRIALITY_MIX", format!("{v:.3}"));
    }
    if let Some(v) = opts.tq_rotation_seed {
        set_process_env_var("LLAMA_TURBOQUANT_ROTATION_SEED", v.to_string());
    }
    if let Some(path) = &opts.tq_artifact {
        if path.trim().is_empty() {
            remove_process_env_var("LLAMA_TURBOQUANT_ARTIFACT");
        } else {
            set_process_env_var("LLAMA_TURBOQUANT_ARTIFACT", path);
        }
    }
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

async fn collect_generate(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<crate::compute::inference::GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> GenerateResponseChunk {
    let mut full_response = String::new();
    while let Some(token) = token_rx.recv().await {
        full_response.push_str(&token.text);
    }
    let total_ns = request_start.elapsed().as_nanos() as u64;
    let result = result_rx.await.ok();

    GenerateResponseChunk {
        model: model_name,
        created_at: now_rfc3339(),
        response: full_response,
        done: true,
        done_reason: Some("stop".into()),
        total_duration: Some(total_ns),
        load_duration: Some(load_duration_ns),
        prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
        prompt_eval_duration: result
            .as_ref()
            .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
        eval_count: result.as_ref().map(|r| r.tokens_generated),
        eval_duration: result.as_ref().map(|r| {
            if r.tok_per_sec_avg > 0.0 {
                (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
            } else {
                0
            }
        }),
    }
}

async fn collect_chat(
    model_name: String,
    mut token_rx: mpsc::UnboundedReceiver<crate::compute::inference::GeneratedToken>,
    result_rx: oneshot::Receiver<GenerationResult>,
    request_start: Instant,
    load_duration_ns: u64,
) -> ChatResponseChunk {
    let mut full_response = String::new();
    while let Some(token) = token_rx.recv().await {
        full_response.push_str(&token.text);
    }
    let total_ns = request_start.elapsed().as_nanos() as u64;
    let result = result_rx.await.ok();

    ChatResponseChunk {
        model: model_name,
        created_at: now_rfc3339(),
        message: ChatMessage {
            role: "assistant".into(),
            content: full_response,
        },
        done: true,
        done_reason: Some("stop".into()),
        total_duration: Some(total_ns),
        load_duration: Some(load_duration_ns),
        prompt_eval_count: result.as_ref().map(|r| r.prompt_tokens),
        prompt_eval_duration: result
            .as_ref()
            .map(|r| (r.prompt_eval_ms * 1_000_000.0) as u64),
        eval_count: result.as_ref().map(|r| r.tokens_generated),
        eval_duration: result.as_ref().map(|r| {
            if r.tok_per_sec_avg > 0.0 {
                (r.tokens_generated as f64 / r.tok_per_sec_avg * 1e9) as u64
            } else {
                0
            }
        }),
    }
}

fn format_parameter_size(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.0}M", params as f64 / 1e6)
    } else {
        format!("{params}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn council_response(selected_text: &str) -> TrialityCouncilResponse {
        TrialityCouncilResponse {
            id: "tc-test".to_string(),
            object: "triality.council".to_string(),
            model: "test.gguf".to_string(),
            selected_text: selected_text.to_string(),
            selected_view: crate::council::CouncilView::Vector,
            candidate_scores: [1.0, 0.5, 0.25],
            winner_margin: 0.5,
            agreement: 0.75,
            aha: None,
            trace: None,
        }
    }

    async fn council_preflight_handler(Json(request): Json<TrialityCouncilRequest>) -> Response {
        if let Err(error) = validate_council_feature_flags(&request) {
            return error.into_response();
        }
        match validate_council_input(request.prompt, request.messages) {
            Ok(_) => StatusCode::NO_CONTENT.into_response(),
            Err(error) => error.into_response(),
        }
    }

    async fn openai_extension_probe(
        Json(request): Json<OpenAiChatCompletionRequest>,
    ) -> &'static str {
        if council_extension(request.hypura.as_ref()).is_some() {
            "council"
        } else {
            "legacy"
        }
    }

    async fn ollama_extension_probe(Json(request): Json<ChatRequest>) -> &'static str {
        if council_extension(request.hypura.as_ref()).is_some() {
            "council"
        } else {
            "legacy"
        }
    }

    async fn council_get_probe(
        State(store): State<Arc<CouncilStore>>,
        AxumPath(request_id): AxumPath<String>,
    ) -> Response {
        triality_council_get_from_store(store, request_id).await
    }

    async fn council_events_probe(State(telemetry): State<Arc<TelemetryEmitter>>) -> Response {
        triality_events_from_emitter(telemetry).await
    }

    async fn post_preflight(value: serde_json::Value) -> reqwest::Response {
        let app = Router::new().route("/v1/triality/council", post(council_preflight_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        reqwest::Client::new()
            .post(format!("http://{address}/v1/triality/council"))
            .json(&value)
            .send()
            .await
            .unwrap()
    }

    #[test]
    fn launcher_profile_context_reads_known_kcpps_fields() {
        let profile = LauncherProfile::from_kcpps_value(
            "demo.kcpps",
            json!({
                "contextsize": 8192,
                "gendefaults": {
                    "max_length": 640
                }
            }),
        )
        .unwrap();

        assert_eq!(launcher_profile_context(&profile, 4096), 8192);
        assert_eq!(launcher_profile_max_length(&profile, 256), 640);
    }

    #[test]
    fn launcher_profile_limits_fall_back_to_current_runtime_values() {
        let profile = LauncherProfile::from_kcpps_value("demo.kcpps", json!({})).unwrap();

        assert_eq!(launcher_profile_context(&profile, 4096), 4096);
        assert_eq!(launcher_profile_max_length(&profile, 256), 256);
    }

    #[tokio::test]
    async fn council_preflight_rejects_unsupported_modes_over_http() {
        for (field, value, expected_code) in [
            (
                "attention_consensus",
                true,
                "attention_consensus_unsupported",
            ),
            ("synthesis", true, "unsupported_until_enabled"),
            ("cross_score", false, "cross_score_required"),
        ] {
            let mut request = json!({"prompt": "hello", "cross_score": true});
            request[field] = json!(value);
            let response = post_preflight(request).await;
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
            let body: serde_json::Value = response.json().await.unwrap();
            assert_eq!(body["error"]["code"], expected_code);
        }
    }

    #[tokio::test]
    async fn council_preflight_requires_exactly_one_input_over_http() {
        for request in [
            json!({"cross_score": true}),
            json!({
                "prompt": "hello",
                "messages": [{"role": "user", "content": "hello"}],
                "cross_score": true
            }),
        ] {
            let response = post_preflight(request).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn council_streams_emit_only_the_selected_final_answer() {
        let selected = "selected-final-answer";
        let native = triality_council_response(council_response(selected), true);
        let native_body = axum::body::to_bytes(native.into_body(), usize::MAX)
            .await
            .unwrap();
        let native_text = String::from_utf8(native_body.to_vec()).unwrap();
        assert_eq!(native_text.matches(selected).count(), 1);
        assert!(native_text.ends_with("data: [DONE]\n\n"));

        let openai = openai_council_chat_response(council_response(selected), true);
        let openai_body = axum::body::to_bytes(openai.into_body(), usize::MAX)
            .await
            .unwrap();
        let openai_text = String::from_utf8(openai_body.to_vec()).unwrap();
        assert_eq!(openai_text.matches(selected).count(), 1);
        assert!(openai_text.ends_with("data: [DONE]\n\n"));

        let ollama = ollama_council_chat_response(council_response(selected), true);
        let ollama_body = axum::body::to_bytes(ollama.into_body(), usize::MAX)
            .await
            .unwrap();
        let ollama_text = String::from_utf8(ollama_body.to_vec()).unwrap();
        assert_eq!(ollama_text.matches(selected).count(), 1);
        assert_eq!(ollama_text.lines().count(), 1);
    }

    #[tokio::test]
    async fn absent_extensions_retain_legacy_dispatch_over_http() {
        let app = Router::new()
            .route("/v1/chat/completions", post(openai_extension_probe))
            .route("/api/chat", post(ollama_extension_probe));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = reqwest::Client::new();
        let openai = client
            .post(format!("http://{address}/v1/chat/completions"))
            .json(&json!({
                "model": "test.gguf",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(openai.text().await.unwrap(), "legacy");
        let ollama = client
            .post(format!("http://{address}/api/chat"))
            .json(&json!({
                "model": "test.gguf",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(ollama.text().await.unwrap(), "legacy");
    }

    #[tokio::test]
    async fn council_get_rejects_oversized_persisted_input_over_http() {
        let data_root = std::env::temp_dir().join(format!(
            "hypura-council-get-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let store = Arc::new(
            CouncilStore::open(crate::council::CouncilStoreConfig::for_data_root(
                data_root.clone(),
            ))
            .unwrap(),
        );
        let request_id = "tc-oversized";
        let record_dir = store.record_directory(request_id).unwrap();
        std::fs::create_dir_all(&record_dir).unwrap();
        let request_file = std::fs::File::create(record_dir.join("request.json")).unwrap();
        request_file.set_len(64 * 1024 * 1024 + 1).unwrap();
        drop(request_file);

        let app = Router::new()
            .route("/api/extra/triality/council/:id", get(council_get_probe))
            .with_state(store);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let response = reqwest::get(format!(
            "http://{address}/api/extra/triality/council/{request_id}"
        ))
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["error"]["code"], "invalid_council_result_id");
        std::fs::remove_dir_all(data_root).unwrap();
    }

    #[tokio::test]
    async fn council_events_endpoint_never_broadcasts_prompt_or_candidate_secrets() {
        let secret = "secret-prompt-and-candidate-content";
        let prompt = secret.to_string();
        let candidates = [secret.to_string(), secret.to_string(), secret.to_string()];
        let telemetry = Arc::new(TelemetryEmitter::new(8));
        let app = Router::new()
            .route("/api/extra/triality/events", get(council_events_probe))
            .with_state(telemetry.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut response = reqwest::get(format!("http://{address}/api/extra/triality/events"))
            .await
            .unwrap();
        telemetry.emit(TelemetryEvent::TokenGenerated {
            tok_per_sec: 1.0,
            token: secret.to_string(),
        });
        telemetry.emit(TelemetryEvent::TrialityConsensusCompleted {
            request_id: "tc-safe".to_string(),
            selected_view: crate::council::CouncilView::Vector,
            candidate_scores: [1.0, 0.5, 0.25],
            winner_margin: 0.5,
            agreement: 0.75,
            result_persisted: false,
        });
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), response.chunk())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let payload = String::from_utf8(chunk.to_vec()).unwrap();
        assert!(!payload.contains(&prompt));
        assert!(
            candidates
                .iter()
                .all(|candidate| !payload.contains(candidate))
        );
    }

    #[test]
    fn council_identity_snapshot_serializes_hot_switch_interleaving() {
        let loaded = Arc::new(Mutex::new(0_u64));
        let name = Arc::new(Mutex::new("model-a".to_string()));
        let path = Arc::new(Mutex::new(PathBuf::from("model-a.gguf")));
        let sha256 = Arc::new(Mutex::new(Some("hash-a".to_string())));
        let (begin_switch_tx, begin_switch_rx) = std::sync::mpsc::channel();
        let (attempting_tx, attempting_rx) = std::sync::mpsc::channel();
        let (completed_tx, completed_rx) = std::sync::mpsc::channel();

        let switch_loaded = loaded.clone();
        let switch_name = name.clone();
        let switch_path = path.clone();
        let switch_sha256 = sha256.clone();
        let switch = std::thread::spawn(move || {
            begin_switch_rx.recv().unwrap();
            attempting_tx.send(()).unwrap();
            let mut loaded = switch_loaded.lock().unwrap();
            let mut name = switch_name.lock().unwrap();
            let mut path = switch_path.lock().unwrap();
            let mut sha256 = switch_sha256.lock().unwrap();
            *loaded = 1;
            *name = "model-b".to_string();
            *path = PathBuf::from("model-b.gguf");
            *sha256 = Some("hash-b".to_string());
            completed_tx.send(()).unwrap();
        });

        let first = lock_model_identity_after_loaded(
            &loaded,
            &name,
            &path,
            &sha256,
            |generation| format!("physical-{generation}"),
            || {
                begin_switch_tx.send(()).unwrap();
                attempting_rx.recv().unwrap();
            },
        )
        .unwrap();
        assert_eq!(*first.loaded, 0);
        assert_eq!(first.identity.public_alias, "model-a");
        assert_eq!(first.identity.physical_name, "physical-0");
        assert_eq!(first.identity.path, PathBuf::from("model-a.gguf"));
        assert_eq!(first.identity.sha256.as_deref(), Some("hash-a"));
        assert!(completed_rx.try_recv().is_err());
        drop(first);

        completed_rx.recv().unwrap();
        switch.join().unwrap();
        let second = lock_model_identity(&loaded, &name, &path, &sha256, |generation| {
            format!("physical-{generation}")
        })
        .unwrap();
        assert_eq!(*second.loaded, 1);
        assert_eq!(second.identity.public_alias, "model-b");
        assert_eq!(second.identity.physical_name, "physical-1");
        assert_eq!(second.identity.path, PathBuf::from("model-b.gguf"));
        assert_eq!(second.identity.sha256.as_deref(), Some("hash-b"));
    }

    #[test]
    fn council_request_uses_public_alias_without_conflating_physical_name() {
        let identity = ActiveModelIdentity {
            public_alias: "public-council".to_string(),
            physical_name: "weights-q4.gguf".to_string(),
            path: PathBuf::from("C:/models/weights-q4.gguf"),
            sha256: Some("a".repeat(64)),
        };

        assert!(validate_requested_model_alias(Some("public-council"), &identity).is_ok());
        let error = validate_requested_model_alias(Some("weights-q4.gguf"), &identity).unwrap_err();
        assert_eq!(error.status, StatusCode::CONFLICT);
        assert_eq!(error.code, "model_not_loaded");
        assert_eq!(
            switched_public_model_alias("public-council", "next.gguf", true),
            "public-council"
        );
        assert_eq!(
            switched_public_model_alias("weights-q4.gguf", "next.gguf", false),
            "next.gguf"
        );
    }

    #[test]
    fn council_trace_storage_exposes_alias_and_hash_without_server_path() {
        let identity = ActiveModelIdentity {
            public_alias: "public-council".to_string(),
            physical_name: "weights-q4.gguf".to_string(),
            path: PathBuf::from("C:/private/models/weights-q4.gguf"),
            sha256: Some("b".repeat(64)),
        };

        let storage = council_trace_storage(&identity, None);
        let serialized = serde_json::to_string(&storage).unwrap();
        assert_eq!(storage["model_alias"], "public-council");
        assert_eq!(storage["model_sha256"], "b".repeat(64));
        assert!(storage.get("model_path").is_none());
        assert!(!serialized.contains("C:/private"));
        assert!(!serialized.contains("weights-q4.gguf"));
    }

    #[test]
    fn model_file_guard_rejects_same_size_content_replacement() {
        use crate::model::file_identity::{ensure_unchanged, snapshot_path};

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"abc").unwrap();
        let before = snapshot_path(&path).unwrap();
        assert!(ensure_unchanged(&before, &before).is_ok());

        std::fs::write(&path, b"xyz").unwrap();
        let after = snapshot_path(&path).unwrap();
        assert_eq!(before.size(), after.size());
        assert_ne!(before.sha256(), after.sha256());
        assert!(ensure_unchanged(&before, &after).is_err());
    }

    fn route_ncka_config(required: bool, fallback_weights: [f32; 3]) -> GgufTurboQuantConfig {
        GgufTurboQuantConfig {
            enabled: true,
            schema_version: 2,
            mode: TurboQuantMode::PaperFullKv,
            public_mode_label: "paper-full-kv".to_string(),
            runtime_mode: "paper-full-kv".to_string(),
            rotation_policy: None,
            triality_view: None,
            triality_mode: None,
            triality_mix: None,
            paper_fidelity: true,
            k_bits: 4.0,
            v_bits: 4.0,
            payload_format: None,
            payload_bytes: 0,
            payload_json: None,
            rotation_seed: 0,
            artifact_path: None,
            head_dim: 128,
            num_layers: 1,
            num_kv_heads: 1,
            layers: Vec::new(),
            weight: None,
            consensus: None,
            ncka: Some(crate::model::turboquant_sidecar::GgufNcKaConfig {
                enabled: true,
                required,
                schema_version: u32::MAX,
                controller_type: "unsupported".to_string(),
                coordinate_names: Vec::new(),
                outer_count: 0,
                knot_count: 0,
                s3_equivariant: true,
                controller_sha256: "0".repeat(64),
                normalisation_sha256: "0".repeat(64),
                static_fallback_selected: !required,
                fallback_weights,
            }),
            urt: None,
        }
    }

    fn empty_route_gguf() -> GgufFile {
        GgufFile {
            version: 3,
            metadata: Default::default(),
            tensors: Vec::new(),
            data_offset: 0,
        }
    }

    #[test]
    fn route_ncka_optional_failure_uses_declared_static_fallback() {
        let fallback_weights = [0.2, 0.3, 0.5];
        let config = route_ncka_config(false, fallback_weights);
        let mut gate = KaGateConfig {
            enabled: true,
            static_fallback_weights: [1.0 / 3.0; 3],
            ..KaGateConfig::default()
        };

        let controller = prepare_route_embedded_ka_controller(
            Path::new("unused.gguf"),
            &empty_route_gguf(),
            &config,
            &mut gate,
        )
        .unwrap();

        assert!(controller.is_none());
        assert_eq!(gate.static_fallback_weights, fallback_weights);
    }

    #[test]
    fn route_ncka_required_and_invalid_fallback_fail_closed() {
        for config in [
            route_ncka_config(true, [0.2, 0.3, 0.5]),
            route_ncka_config(false, [0.6, 0.6, -0.2]),
        ] {
            let mut gate = KaGateConfig {
                enabled: true,
                ..KaGateConfig::default()
            };
            let error = prepare_route_embedded_ka_controller(
                Path::new("unused.gguf"),
                &empty_route_gguf(),
                &config,
                &mut gate,
            )
            .unwrap_err();

            assert_eq!(error.status, StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(error.code, "ncka_controller_unavailable");
        }
    }

    #[test]
    fn single_urt_observation_is_persisted_and_emitted_unassessed() {
        let directory = tempfile::tempdir().unwrap();
        let registry = Mutex::new(
            UrtRegistry::open(crate::urt::UrtRegistryConfig::persistent(directory.path())).unwrap(),
        );
        let observation = UrtObservation {
            request_id: "tc-urt-single".to_string(),
            representation: RepresentationId {
                kind: RepresentationKind::HypuraNative,
                model_hash: "model-sha256".to_string(),
                artefact_hash: Some("artefact-sha256".to_string()),
                backend: "hypura_native".to_string(),
                precision: "k4.000_v4.000".to_string(),
                view: Some("vector".to_string()),
            },
            state_id: "state-single".to_string(),
            layer: None,
            operator_word: vec!["Q".to_string(), "U".to_string()],
            operator_word_sha256: "operator-sha256".to_string(),
            observable: "selected_candidate_mean_log_likelihood".to_string(),
            value_real: -1.25,
            value_imag: 0.0,
            tolerance: 0.001,
        };

        let trace = record_council_urt_observation(&registry, Some(&observation))
            .unwrap()
            .unwrap();
        assert_eq!(trace.status, crate::urt::UrtAssessmentStatus::Unassessed);
        assert!(trace.report.is_none());
        let trace_json = serde_json::to_value(&trace).unwrap();
        assert_eq!(trace_json["status"], "unassessed");
        assert!(trace_json.get("report").is_none());

        let urt_directory = directory.path().join("artifacts").join("urt");
        assert!(urt_directory.join("representations.json").is_file());
        assert!(urt_directory.join("observations.jsonl").is_file());
        assert!(!urt_directory.join("consistency_summary.csv").exists());
        assert!(!urt_directory.join("consistency_failures.jsonl").exists());

        let telemetry = TelemetryEmitter::new(4);
        let mut receiver = telemetry.subscribe();
        emit_council_urt_event(&telemetry, observation.request_id.clone(), &trace);
        match receiver.try_recv().unwrap() {
            TelemetryEvent::TrialityUrtChecked {
                request_id,
                comparison_count,
                consistent,
                max_absolute_error,
            } => {
                assert_eq!(request_id, observation.request_id);
                assert_eq!(comparison_count, 0);
                assert_eq!(consistent, None);
                assert_eq!(max_absolute_error, None);
            }
            other => panic!("unexpected telemetry event: {other:?}"),
        }
    }

    #[test]
    fn hot_switch_model_hash_uses_guarded_snapshot_only_for_urt() {
        use crate::model::file_identity::snapshot_path;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"abc").unwrap();
        let mut active_hash = Some("old-model-hash".to_string());
        assert_eq!(active_hash.as_deref(), Some("old-model-hash"));

        let snapshot = snapshot_path(&path).unwrap();
        active_hash = true.then(|| snapshot.sha256().to_string());
        assert_eq!(
            active_hash.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );

        active_hash = false.then(|| snapshot.sha256().to_string());
        assert!(active_hash.is_none());
    }
}
