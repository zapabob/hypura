use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum::response::Html;
use tokio::sync::{mpsc, oneshot};

use crate::compute::inference::{GenerateFromLoadedParams, GenerationResult, LoadedModel};
use crate::model::turboquant_sidecar::{ResolvedTurboQuantConfig, TurboQuantMode};
use crate::server::chat::format_chat_prompt;
use crate::server::ollama_types::*;
use crate::server::streaming;
use crate::telemetry::metrics::TelemetryEmitter;

pub struct AppState {
    pub loaded_model: Arc<std::sync::Mutex<LoadedModel>>,
    pub model_name: String,
    pub gguf_info: GgufInfo,
    pub load_duration_ns: u64,
    pub telemetry: Arc<TelemetryEmitter>,
    pub turboquant: ResolvedTurboQuantConfig,
    pub active_cancel: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    pub generation_in_progress: Arc<AtomicBool>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(health_handler))
        .route("/api/version", get(version_handler))
        .route("/api/tags", get(tags_handler))
        .route("/api/show", post(show_handler))
        .route("/kobold-lite", get(kobold_lite_gui_handler))
        .route("/api/generate", post(generate_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/v1/model", get(kobold_model_handler))
        .route("/api/v1/generate", post(kobold_generate_handler))
        .route("/api/extra/generate/stream", post(kobold_generate_stream_handler))
        .route("/api/extra/generate/check", get(kobold_generate_check_handler))
        .route("/api/extra/abort", post(kobold_abort_handler))
        .route(
            "/api/extra/true_max_context_length",
            get(kobold_true_max_context_length_handler),
        )
        .with_state(state)
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn version_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({"version": env!("CARGO_PKG_VERSION")}))
}

async fn kobold_lite_gui_handler() -> Html<&'static str> {
    Html(
        r#"<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Hypura Kobold GUI</title>
  <style>
    body { font-family: sans-serif; max-width: 1080px; margin: 1rem auto; padding: 0 1rem; }
    textarea, input, select { width: 100%; box-sizing: border-box; margin-top: .3rem; margin-bottom: .7rem; }
    .row { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap: .7rem; }
    .toolbar { display:flex; gap:.6rem; margin:.8rem 0; flex-wrap: wrap; }
    .panel { border:1px solid #ddd; border-radius:8px; padding:.8rem; margin:.8rem 0; }
    button { padding:.55rem .95rem; cursor:pointer; }
    pre { white-space: pre-wrap; border:1px solid #ddd; padding:1rem; border-radius:6px; min-height:220px; }
  </style>
</head>
<body>
  <h2>Hypura Kobold GUI (Parity+)</h2>
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
  <div class="panel">
    <strong>Runtime Metrics</strong>
    <div id="metrics">tok/s: -, eval_count: -, total_ms: -, prompt_ms: -</div>
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
  <h3>Preset Diff</h3>
  <pre id="presetDiff"></pre>
  <script>
    const KEY = 'hypura_kobold_presets_v2';
    let lastRequest = null;

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
      document.getElementById('samplerOrder').value = (v.sampler_order ?? [6,0,1,3,4,2,5]).join(',');
      document.getElementById('stop').value = (v.stop_sequence ?? []).join('\n');
      document.getElementById('tqSo8Off').value = v.tq_so8_off === undefined ? '' : (v.tq_so8_off ? '1' : '0');
      document.getElementById('tqSo8Learned').value = v.tq_so8_learned === undefined ? '' : (v.tq_so8_learned ? '1' : '0');
      document.getElementById('tqTrialityOff').value = v.tq_triality_off === undefined ? '' : (v.tq_triality_off ? '1' : '0');
      document.getElementById('tqTrialityMix').value = v.tq_triality_mix ?? '';
      document.getElementById('tqRotationSeed').value = v.tq_rotation_seed ?? '';
      document.getElementById('tqArtifact').value = v.tq_artifact ?? '';
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
      const body = readForm();
      lastRequest = body;
      document.getElementById('out').textContent = 'Generating...';
      const r = await fetch('/api/v1/generate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const j = await r.json();
      document.getElementById('out').textContent = (j.results && j.results[0] ? j.results[0].text : JSON.stringify(j, null, 2));
    }

    async function generateStream() {
      const body = { ...readForm(), stream: true };
      lastRequest = body;
      document.getElementById('out').textContent = '';
      const startedAt = performance.now();
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
            }
            if (obj.done) {
              const elapsedMs = Math.max(1, performance.now() - startedAt);
              const tps = latestTokPerSec ?? ((tokenCount * 1000.0) / elapsedMs);
              document.getElementById('metrics').textContent = `tok/s: ${tps.toFixed(2)}, eval_count: ${obj.eval_count ?? '-'}, total_ms: ${(obj.total_duration ? obj.total_duration / 1e6 : elapsedMs).toFixed(1)}, prompt_ms: ${obj.prompt_eval_ms ?? '-'}`;
              document.getElementById('out').textContent += '\n\n[done]';
            }
          } catch {}
        }
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
    }

    async function checkStatus() {
      const r = await fetch('/api/extra/generate/check');
      const j = await r.json();
      const msg = (j.results && j.results[0] ? j.results[0].text : JSON.stringify(j, null, 2));
      document.getElementById('out').textContent += '\n\n[check] ' + msg;
    }

    refreshPresetList();
  </script>
</body>
</html>"#,
    )
}

async fn tags_handler(State(state): State<Arc<AppState>>) -> Json<TagsResponse> {
    let info = &state.gguf_info;
    Json(TagsResponse {
        models: vec![ModelTag {
            name: state.model_name.clone(),
            model: state.model_name.clone(),
            size: info.file_size,
            details: ModelDetails {
                format: "gguf".into(),
                family: info.architecture.clone(),
                parameter_size: format_parameter_size(info.parameter_count),
                quantization_level: info.quantization.clone(),
            },
        }],
    })
}

async fn show_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ShowRequest>,
) -> Json<ShowResponse> {
    let info = &state.gguf_info;
    let requested_model = req
        .name
        .as_deref()
        .or(req.model.as_deref())
        .unwrap_or(state.model_name.as_str())
        .to_string();
    Json(ShowResponse {
        details: ModelDetails {
            format: "gguf".into(),
            family: info.architecture.clone(),
            parameter_size: format_parameter_size(info.parameter_count),
            quantization_level: info.quantization.clone(),
        },
        model_info: serde_json::json!({
            "general.name": requested_model,
            "general.architecture": info.architecture,
            "general.context_length": info.context_length,
            "general.parameter_count": info.parameter_count,
            "hypura.turboquant.mode": state.turboquant.mode.as_str(),
            "hypura.turboquant.schema": state.turboquant.schema_label(),
            "hypura.turboquant.config_path": state.turboquant.source_label(),
            "hypura.turboquant.runtime_status": turboquant_runtime_status(
                state.turboquant.mode,
                state.turboquant.config.is_some(),
            ),
        }),
    })
}

async fn kobold_model_handler(State(state): State<Arc<AppState>>) -> Json<KoboldModelResponse> {
    Json(KoboldModelResponse {
        result: state.model_name.clone(),
    })
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
    let model_name = state.model_name.clone();

    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();

    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();
    let sampler_order = req.options.sampler_order.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        end_generation(&state_for_task);
        match result {
            Ok(gen_result) => {
                let _ = result_tx.send(gen_result);
            }
            Err(e) => {
                tracing::error!("Generation error: {e}");
            }
        }
    });

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
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    let (result_tx, _result_rx) = oneshot::channel::<GenerationResult>();
    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        end_generation(&state_for_task);
        if let Ok(gen_result) = result {
            let _ = result_tx.send(gen_result);
        }
    });

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

    let (token_tx, mut token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();
    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();
    let model_name = state.model_name.clone();
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        end_generation(&state_for_task);
        if let Ok(gen_result) = result {
            let _ = result_tx.send(gen_result);
        }
    });

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

async fn kobold_generate_check_handler(State(state): State<Arc<AppState>>) -> Json<KoboldGenerateResponse> {
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
        value: state.gguf_info.context_length,
    })
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let request_start = Instant::now();
    let load_duration_ns = state.load_duration_ns;

    apply_turboquant_runtime_overrides(&req.options);
    let sampling = build_sampling(&req.options);
    let stop_sequences = req.options.stop.clone().unwrap_or_default();
    let prompt = format_chat_prompt(&req.messages);
    let model_name = state.model_name.clone();

    let (token_tx, token_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = oneshot::channel::<GenerationResult>();

    let loaded = state.loaded_model.clone();
    let telemetry = state.telemetry.clone();
    let cancel_flag = begin_generation(&state);
    let state_for_task = state.clone();
    let sampler_order = req.options.sampler_order.clone();

    tokio::task::spawn_blocking(move || {
        let mut model = loaded.lock().unwrap();
        let params = GenerateFromLoadedParams {
            prompt: &prompt,
            sampling: &sampling,
            sampler_order: sampler_order.as_deref(),
            stop_sequences: &stop_sequences,
            cancel_flag: Some(cancel_flag),
            token_tx,
            telemetry,
        };
        let result = crate::compute::inference::generate_from_loaded(&mut model, params);
        end_generation(&state_for_task);
        match result {
            Ok(gen_result) => {
                let _ = result_tx.send(gen_result);
            }
            Err(e) => {
                tracing::error!("Chat generation error: {e}");
            }
        }
    });

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

fn apply_turboquant_runtime_overrides(opts: &GenerateOptions) {
    if let Some(v) = opts.tq_so8_off {
        std::env::set_var("LLAMA_TURBOQUANT_SO8", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_so8_learned {
        std::env::set_var("LLAMA_TURBOQUANT_SO8_LEARNED", if v { "1" } else { "0" });
    }
    if let Some(v) = opts.tq_triality_off {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_triality_mix {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY_MIX", format!("{v:.3}"));
    }
    if let Some(v) = opts.tq_rotation_seed {
        std::env::set_var("LLAMA_TURBOQUANT_ROTATION_SEED", v.to_string());
    }
    if let Some(path) = &opts.tq_artifact {
        if path.trim().is_empty() {
            std::env::remove_var("LLAMA_TURBOQUANT_ARTIFACT");
        } else {
            std::env::set_var("LLAMA_TURBOQUANT_ARTIFACT", path);
        }
    }
}

fn apply_turboquant_runtime_overrides_kobold(opts: &KoboldGenerateRequest) {
    if let Some(v) = opts.tq_so8_off {
        std::env::set_var("LLAMA_TURBOQUANT_SO8", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_so8_learned {
        std::env::set_var("LLAMA_TURBOQUANT_SO8_LEARNED", if v { "1" } else { "0" });
    }
    if let Some(v) = opts.tq_triality_off {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY", if v { "0" } else { "1" });
    }
    if let Some(v) = opts.tq_triality_mix {
        std::env::set_var("LLAMA_TURBOQUANT_TRIALITY_MIX", format!("{v:.3}"));
    }
    if let Some(v) = opts.tq_rotation_seed {
        std::env::set_var("LLAMA_TURBOQUANT_ROTATION_SEED", v.to_string());
    }
    if let Some(path) = &opts.tq_artifact {
        if path.trim().is_empty() {
            std::env::remove_var("LLAMA_TURBOQUANT_ARTIFACT");
        } else {
            std::env::set_var("LLAMA_TURBOQUANT_ARTIFACT", path);
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
