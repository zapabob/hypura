use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use hypura::compute::inference;
use hypura::model::turboquant_sidecar::TurboQuantMode;
use hypura::server::ollama_types::GgufInfo;
use hypura::server::routes::{self, AppState};
use hypura::telemetry::metrics::TelemetryEmitter;

pub fn run(
    model_path: &str,
    host: &str,
    port: u16,
    context: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
    _rotation_policy: Option<&str>,
    _rotation_seed: u32,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_async(
        model_path,
        host,
        port,
        context,
        turboquant_mode,
        turboquant_config,
    ))
}

async fn run_async(
    model_path: &str,
    host: &str,
    port: u16,
    context: u32,
    turboquant_mode: TurboQuantMode,
    turboquant_config: Option<&str>,
) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    let runtime = inference::resolve_runtime_setup(
        path,
        context,
        turboquant_mode,
        turboquant_config.map(Path::new),
    )?;
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

    // Load model on a blocking thread
    println!("Loading model...");
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

    let load_duration_ns = load_start.elapsed().as_nanos() as u64;
    // Allow explicit model tag override for Ollama-compatible `/api/tags` naming.
    // This is used by local operators who need a stable tag suffix (e.g. `-Q4_K_M`)
    // even when GGUF metadata `general.name` omits quantization details.
    let model_name = std::env::var("HYPURA_MODEL_NAME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| loaded.model_name.clone());

    let telemetry = Arc::new(TelemetryEmitter::new(256));

    let state = Arc::new(AppState {
        loaded_model: Arc::new(std::sync::Mutex::new(loaded)),
        model_name: model_name.clone(),
        gguf_info,
        load_duration_ns,
        telemetry,
        turboquant: runtime.turboquant,
    });

    let app = routes::router(state.clone());
    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    println!();
    println!("Hypura serving {model_name}");
    println!("  Endpoint: http://{bind_addr}");
    println!("  Ollama-compatible API: /api/generate, /api/chat, /api/tags");
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
    println!();

    axum::serve(listener, app).await?;
    Ok(())
}
