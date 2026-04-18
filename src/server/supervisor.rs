use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;

use crate::model::turboquant_sidecar::{RotationPolicy, TurboQuantMode};
use crate::scheduler::types::{HostPinnedPolicy, ResidencyProfile};

use super::compat::{self, CompatFeatureFlags};
use super::embeddings::EmbeddingsRuntime;
use super::ollama_types::OpenAiEmbeddingsRequest;
use super::websearch::{WebSearchRequest, WebSearchService};

pub const COMPAT_CONTROL_URL_ENV: &str = "HYPURA_COMPAT_CONTROL_URL";
pub const COMPAT_CONTROL_TOKEN_ENV: &str = "HYPURA_COMPAT_CONTROL_TOKEN";
pub const COMPAT_FEATURES_JSON_ENV: &str = "HYPURA_COMPAT_FEATURES_JSON";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompatControlPlaneClientInfo {
    pub base_url: String,
    pub bearer_token: String,
}

#[derive(Clone)]
pub struct CompatControlPlaneClient {
    inner: CompatControlPlaneClientInfo,
    client: reqwest::Client,
}

impl CompatControlPlaneClient {
    pub fn new(inner: CompatControlPlaneClientInfo) -> Self {
        Self {
            inner,
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var(COMPAT_CONTROL_URL_ENV).ok()?;
        let bearer_token = std::env::var(COMPAT_CONTROL_TOKEN_ENV).ok()?;
        Some(Self::new(CompatControlPlaneClientInfo {
            base_url,
            bearer_token,
        }))
    }

    pub async fn send_command(&self, command: CompatSupervisorCommand) -> anyhow::Result<()> {
        let response = self
            .client
            .post(format!("{}/control/command", self.inner.base_url))
            .bearer_auth(&self.inner.bearer_token)
            .json(&command)
            .send()
            .await?;
        anyhow::ensure!(
            response.status().is_success(),
            "compat control plane rejected command with status {}",
            response.status()
        );
        Ok(())
    }

    pub async fn proxy_request(
        &self,
        path: &str,
        method: Method,
        headers: &HeaderMap,
        body: Bytes,
    ) -> anyhow::Result<Response> {
        let mut request = self
            .client
            .request(
                reqwest::Method::from_bytes(method.as_str().as_bytes())?,
                format!("{}{}", self.inner.base_url, path),
            )
            .bearer_auth(&self.inner.bearer_token);
        if let Some(content_type) = headers.get(header::CONTENT_TYPE) {
            request = request.header(header::CONTENT_TYPE, content_type);
        }
        if let Some(accept) = headers.get(header::ACCEPT) {
            request = request.header(header::ACCEPT, accept);
        }
        let upstream = request.body(body.to_vec()).send().await?;
        let status = StatusCode::from_u16(upstream.status().as_u16())
            .unwrap_or(StatusCode::BAD_GATEWAY);
        let content_type = upstream.headers().get(header::CONTENT_TYPE).cloned();
        let bytes = upstream.bytes().await?;
        let mut response = Response::new(Body::from(bytes));
        *response.status_mut() = status;
        if let Some(content_type) = content_type {
            response
                .headers_mut()
                .insert(header::CONTENT_TYPE, content_type);
        }
        Ok(response)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatWorkerBootstrap {
    pub public_host: String,
    pub public_port: u16,
    pub model_path: String,
    pub context: u32,
    pub default_max_length: u32,
    pub turboquant_mode: TurboQuantMode,
    pub turboquant_config: Option<String>,
    pub rotation_policy: RotationPolicy,
    pub rotation_seed: u32,
    pub tq_so8_off: bool,
    pub tq_so8_learned: bool,
    pub tq_triality_off: bool,
    pub tq_triality_mix: f32,
    pub tq_rotation_seed: u32,
    pub tq_artifact: Option<String>,
    pub model_dir: Option<String>,
    pub ui_theme: String,
    pub savedatafile: Option<String>,
    pub preloadstory: Option<String>,
    pub admindir: Option<String>,
    pub config: Option<String>,
    pub migration_dir: Option<String>,
    pub residency_profile: ResidencyProfile,
    pub host_pinned: HostPinnedPolicy,
    pub control_plane: CompatControlPlaneClientInfo,
    pub feature_state: CompatFeatureFlags,
}

impl CompatWorkerBootstrap {
    pub fn read_from_path(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn write_to_path(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum CompatSupervisorCommand {
    ReloadConfig {
        filename: String,
        baseconfig: Option<String>,
    },
    ReprobeBundles,
    ShutdownWorker,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultimodalBackendConfig {
    pub transcribe_base_url: Option<String>,
    pub tts_base_url: Option<String>,
    pub sd_base_url: Option<String>,
}

impl MultimodalBackendConfig {
    pub fn from_env() -> Self {
        Self {
            transcribe_base_url: env_url("HYPURA_KCPP_TRANSCRIBE_URL"),
            tts_base_url: env_url("HYPURA_KCPP_TTS_URL"),
            sd_base_url: env_url("HYPURA_KCPP_SD_URL"),
        }
    }

    pub fn apply_to_features(&self, mut base: CompatFeatureFlags) -> CompatFeatureFlags {
        let transcribe = self
            .transcribe_base_url
            .as_deref()
            .map(|value| !value.is_empty())
            .unwrap_or(false);
        let tts = self
            .tts_base_url
            .as_deref()
            .map(|value| !value.is_empty())
            .unwrap_or(false);
        let txt2img = self
            .sd_base_url
            .as_deref()
            .map(|value| !value.is_empty())
            .unwrap_or(false);
        base.transcribe = transcribe;
        base.tts = tts;
        base.audio = transcribe || tts;
        base.txt2img = txt2img;
        base.vision = false;
        base
    }
}

fn env_url(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn compat_feature_flags_from_env() -> Option<CompatFeatureFlags> {
    let raw = std::env::var(COMPAT_FEATURES_JSON_ENV).ok()?;
    serde_json::from_str(&raw).ok()
}

#[derive(Clone)]
struct ControlPlaneState {
    bearer_token: String,
    command_tx: mpsc::UnboundedSender<CompatSupervisorCommand>,
    multimodal: Arc<Mutex<MultimodalBackendConfig>>,
    embeddings: Arc<Mutex<Option<EmbeddingsRuntime>>>,
    websearch: WebSearchService,
    proxy_client: reqwest::Client,
}

pub struct SupervisorControlPlane {
    pub client_info: CompatControlPlaneClientInfo,
    pub command_rx: mpsc::UnboundedReceiver<CompatSupervisorCommand>,
    pub multimodal: Arc<Mutex<MultimodalBackendConfig>>,
    pub embeddings: Arc<Mutex<Option<EmbeddingsRuntime>>>,
}

pub async fn spawn_supervisor_control_plane(
    multimodal: MultimodalBackendConfig,
    embeddings: Arc<Mutex<Option<EmbeddingsRuntime>>>,
) -> anyhow::Result<SupervisorControlPlane> {
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let bearer_token = uuid::Uuid::new_v4().simple().to_string();
    let multimodal = Arc::new(Mutex::new(multimodal));
    let websearch = WebSearchService::new()?;
    let state = ControlPlaneState {
        bearer_token: bearer_token.clone(),
        command_tx,
        multimodal: multimodal.clone(),
        embeddings: embeddings.clone(),
        websearch,
        proxy_client: reqwest::Client::new(),
    };
    let app = Router::new()
        .route("/control/command", post(control_command_handler))
        .route(
            "/embeddings/api/extra/embeddings",
            post(embeddings_handler),
        )
        .route("/embeddings/v1/embeddings", post(embeddings_handler))
        .route("/builtin/api/extra/websearch", post(websearch_handler))
        .route(
            "/multimodal/api/extra/transcribe",
            post(multimodal_transcribe_handler),
        )
        .route(
            "/multimodal/v1/audio/transcriptions",
            post(multimodal_transcribe_handler),
        )
        .route("/multimodal/api/extra/tts", post(multimodal_tts_handler))
        .route("/multimodal/v1/audio/speech", post(multimodal_tts_handler))
        .route("/multimodal/speakers_list", get(multimodal_speakers_handler))
        .route("/multimodal/sdapi/v1/txt2img", post(multimodal_sd_post_handler))
        .route("/multimodal/sdapi/v1/img2img", post(multimodal_sd_post_handler))
        .route(
            "/multimodal/sdapi/v1/interrogate",
            post(multimodal_sd_post_handler),
        )
        .route("/multimodal/sdapi/v1/upscale", post(multimodal_sd_post_handler))
        .route("/multimodal/sdapi/v1/options", get(multimodal_sd_get_handler))
        .route("/multimodal/sdapi/v1/sd-models", get(multimodal_sd_get_handler))
        .with_state(state);
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::error!("compat supervisor control plane stopped: {error}");
        }
    });

    Ok(SupervisorControlPlane {
        client_info: CompatControlPlaneClientInfo {
            base_url: format!("http://{addr}"),
            bearer_token,
        },
        command_rx,
        multimodal,
        embeddings,
    })
}

fn authorize(headers: &HeaderMap, state: &ControlPlaneState) -> anyhow::Result<()> {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        anyhow::bail!("missing authorization header");
    };
    let expected = format!("Bearer {}", state.bearer_token);
    anyhow::ensure!(
        value.to_str().ok() == Some(expected.as_str()),
        "invalid bearer token"
    );
    Ok(())
}

async fn control_command_handler(
    State(state): State<ControlPlaneState>,
    headers: HeaderMap,
    Json(command): Json<CompatSupervisorCommand>,
) -> Response {
    if let Err(error) = authorize(&headers, &state) {
        return unauthorized_response(error);
    }
    match state.command_tx.send(command) {
        Ok(()) => Json(json!({ "success": true })).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "success": false, "error": error.to_string() })),
        )
            .into_response(),
    }
}

async fn embeddings_handler(
    State(state): State<ControlPlaneState>,
    headers: HeaderMap,
    Json(request): Json<OpenAiEmbeddingsRequest>,
) -> Response {
    if let Err(error) = authorize(&headers, &state) {
        return unauthorized_response(error);
    }
    if state.embeddings.lock().unwrap().is_none() {
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
    let embeddings = state.embeddings.clone();
    match tokio::task::spawn_blocking(move || {
        let mut guard = embeddings.lock().unwrap();
        let runtime = guard.as_mut().ok_or_else(|| {
            anyhow::anyhow!("embeddings model is not available in the current compatibility runtime")
        })?;
        runtime.embed_request(&request)
    })
    .await
    {
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err(error)) => {
            let message = error.to_string();
            let status = if message.contains("not available in the current compatibility runtime")
            {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_REQUEST
            };
            (
                status,
                Json(compat::openai_error(
                    &message,
                    if status == StatusCode::BAD_REQUEST {
                        "invalid_request_error"
                    } else {
                        "unsupported_feature"
                    },
                    if status == StatusCode::BAD_REQUEST {
                        "embeddings_failed"
                    } else {
                        "embeddings_unavailable"
                    },
                )),
            )
                .into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(compat::openai_error(
                &format!("embeddings runtime task failed: {error}"),
                "server_error",
                "embeddings_runtime_failed",
            )),
        )
            .into_response(),
    }
}

async fn websearch_handler(
    State(state): State<ControlPlaneState>,
    headers: HeaderMap,
    Json(request): Json<WebSearchRequest>,
) -> Response {
    if let Err(error) = authorize(&headers, &state) {
        return unauthorized_response(error);
    }
    match state.websearch.search(&request.q).await {
        Ok(results) => Json(results).into_response(),
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "error": format!("websearch failed: {error}"),
                "feature": "websearch",
            })),
        )
            .into_response(),
    }
}

async fn multimodal_transcribe_handler(
    State(state): State<ControlPlaneState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(error) = authorize(&headers, &state) {
        return unauthorized_response(error);
    }
    let multimodal = state.multimodal.lock().unwrap().clone();
    let Some(base_url) = multimodal.transcribe_base_url.as_deref() else {
        return multimodal_openai_unavailable_response(
            "transcribe is not available in the current bundled multimodal runtime",
            "transcribe_unavailable",
        );
    };
    let path = headers
        .get("x-hypura-upstream-path")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| "/api/extra/transcribe".to_string());
    proxy_upstream(&state, headers, body, Method::POST, base_url, &path).await
}

async fn multimodal_tts_handler(
    State(state): State<ControlPlaneState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(error) = authorize(&headers, &state) {
        return unauthorized_response(error);
    }
    let multimodal = state.multimodal.lock().unwrap().clone();
    let Some(base_url) = multimodal.tts_base_url.as_deref() else {
        return multimodal_openai_unavailable_response(
            "tts is not available in the current bundled multimodal runtime",
            "tts_unavailable",
        );
    };
    let path = headers
        .get("x-hypura-upstream-path")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| "/api/extra/tts".to_string());
    proxy_upstream(&state, headers, body, Method::POST, base_url, &path).await
}

async fn multimodal_speakers_handler(
    State(state): State<ControlPlaneState>,
    headers: HeaderMap,
) -> Response {
    if let Err(error) = authorize(&headers, &state) {
        return unauthorized_response(error);
    }
    let multimodal = state.multimodal.lock().unwrap().clone();
    let Some(base_url) = multimodal.tts_base_url.as_deref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(compat::feature_unavailable_error("speakers_list")),
        )
            .into_response();
    };
    proxy_upstream(
        &state,
        headers,
        Bytes::new(),
        Method::GET,
        base_url,
        "/speakers_list",
    )
    .await
}

async fn multimodal_sd_post_handler(
    State(state): State<ControlPlaneState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Err(error) = authorize(&headers, &state) {
        return unauthorized_response(error);
    }
    let multimodal = state.multimodal.lock().unwrap().clone();
    let Some(base_url) = multimodal.sd_base_url.as_deref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(compat::feature_unavailable_error("txt2img")),
        )
            .into_response();
    };
    let path = headers
        .get("x-hypura-upstream-path")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| "/sdapi/v1/txt2img".to_string());
    proxy_upstream(&state, headers, body, Method::POST, base_url, &path).await
}

async fn multimodal_sd_get_handler(
    State(state): State<ControlPlaneState>,
    headers: HeaderMap,
) -> Response {
    if let Err(error) = authorize(&headers, &state) {
        return unauthorized_response(error);
    }
    let multimodal = state.multimodal.lock().unwrap().clone();
    let Some(base_url) = multimodal.sd_base_url.as_deref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(compat::feature_unavailable_error("txt2img")),
        )
            .into_response();
    };
    let path = headers
        .get("x-hypura-upstream-path")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| "/sdapi/v1/options".to_string());
    proxy_upstream(&state, headers, Bytes::new(), Method::GET, base_url, &path).await
}

async fn proxy_upstream(
    state: &ControlPlaneState,
    headers: HeaderMap,
    body: Bytes,
    method: Method,
    base_url: &str,
    path: &str,
) -> Response {
    let mut request = state
        .proxy_client
        .request(
            reqwest::Method::from_bytes(method.as_str().as_bytes())
                .unwrap_or(reqwest::Method::GET),
            format!("{base_url}{path}"),
        );
    if let Some(content_type) = headers.get(header::CONTENT_TYPE) {
        request = request.header(header::CONTENT_TYPE, content_type);
    }
    if let Some(accept) = headers.get(header::ACCEPT) {
        request = request.header(header::ACCEPT, accept);
    }
    match request.body(body.to_vec()).send().await {
        Ok(response) => {
            let status = StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::BAD_GATEWAY);
            let content_type = response.headers().get(header::CONTENT_TYPE).cloned();
            match response.bytes().await {
                Ok(bytes) => {
                    let mut proxied = Response::new(Body::from(bytes));
                    *proxied.status_mut() = status;
                    if let Some(content_type) = content_type {
                        proxied
                            .headers_mut()
                            .insert(header::CONTENT_TYPE, content_type);
                    }
                    proxied
                }
                Err(error) => (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("failed to read multimodal upstream response: {error}") })),
                )
                    .into_response(),
            }
        }
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("failed to proxy multimodal request: {error}") })),
        )
            .into_response(),
    }
}

fn unauthorized_response(error: anyhow::Error) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "success": false, "error": error.to_string() })),
    )
        .into_response()
}

fn multimodal_openai_unavailable_response(message: &str, code: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(compat::openai_error(message, "unsupported_feature", code)),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_bootstrap_roundtrips_json() {
        let bootstrap = CompatWorkerBootstrap {
            public_host: "127.0.0.1".to_string(),
            public_port: 5001,
            model_path: "demo.gguf".to_string(),
            context: 4096,
            default_max_length: 256,
            turboquant_mode: TurboQuantMode::ResearchKvSplit,
            turboquant_config: None,
            rotation_policy: RotationPolicy::TrialityVector,
            rotation_seed: 0,
            tq_so8_off: false,
            tq_so8_learned: false,
            tq_triality_off: false,
            tq_triality_mix: 0.5,
            tq_rotation_seed: 0,
            tq_artifact: None,
            model_dir: None,
            ui_theme: "classic".to_string(),
            savedatafile: None,
            preloadstory: None,
            admindir: None,
            config: None,
            migration_dir: None,
            residency_profile: ResidencyProfile::FourTier,
            host_pinned: HostPinnedPolicy::Auto,
            control_plane: CompatControlPlaneClientInfo {
                base_url: "http://127.0.0.1:12345".to_string(),
                bearer_token: "token".to_string(),
            },
            feature_state: CompatFeatureFlags {
                savedata: true,
                admin: 1,
                ..CompatFeatureFlags::default()
            },
        };

        let encoded = serde_json::to_string(&bootstrap).unwrap();
        let decoded: CompatWorkerBootstrap = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.public_port, 5001);
        assert!(decoded.feature_state.savedata);
        assert_eq!(decoded.feature_state.admin, 1);
    }

    #[test]
    fn multimodal_backend_probe_sets_expected_feature_flags() {
        let mut base = CompatFeatureFlags {
            savedata: true,
            admin: 2,
            ..CompatFeatureFlags::default()
        };
        let backend = MultimodalBackendConfig {
            transcribe_base_url: Some("http://127.0.0.1:1".to_string()),
            tts_base_url: None,
            sd_base_url: Some("http://127.0.0.1:2".to_string()),
        };

        base = backend.apply_to_features(base);

        assert!(base.savedata);
        assert_eq!(base.admin, 2);
        assert!(base.transcribe);
        assert!(base.audio);
        assert!(base.txt2img);
        assert!(!base.tts);
        assert!(!base.vision);
    }
}
