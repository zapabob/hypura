use std::time::Instant;

use serde_json::{json, Value};

use crate::compute::ffi::SamplingParams;
use crate::compute::inference::GenerationResult;

use super::ollama_types::{ChatMessage, KcppPerfResponse, KcppVersionResponse};

pub const KOBOLDCPP_COMPAT_RELEASE_TAG: &str = "v1.111.2";
pub const KOBOLDCPP_API_SCHEMA_VERSION: &str = "2025.06.03";
pub const KOBOLDAI_API_VERSION: &str = "1.2.5";

#[derive(Debug, Clone, Copy, Default)]
pub struct CompatFeatureFlags {
    pub txt2img: bool,
    pub vision: bool,
    pub transcribe: bool,
    pub multiplayer: bool,
    pub websearch: bool,
    pub tts: bool,
    pub embeddings: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CompatPerfState {
    pub last_process_secs: f64,
    pub last_eval_secs: f64,
    pub last_token_count: u32,
    pub last_input_count: u32,
    pub last_seed: u32,
    pub last_stop_reason: i32,
    pub total_gens: u64,
    pub total_img_gens: u64,
    pub total_tts_gens: u64,
    pub total_transcribe_gens: u64,
    pub last_generation_finished_at: Option<Instant>,
}

pub fn build_version_response(
    protected: bool,
    features: CompatFeatureFlags,
) -> KcppVersionResponse {
    KcppVersionResponse {
        result: "KoboldCpp".into(),
        version: KOBOLDCPP_API_SCHEMA_VERSION.into(),
        protected,
        txt2img: features.txt2img,
        vision: features.vision,
        transcribe: features.transcribe,
        multiplayer: features.multiplayer,
        websearch: features.websearch,
        tts: features.tts,
        embeddings: features.embeddings,
    }
}

pub fn build_perf_response(
    snapshot: &CompatPerfState,
    started_at: Instant,
    generation_in_progress: bool,
    quiet: bool,
) -> KcppPerfResponse {
    let uptime = started_at.elapsed().as_secs();
    let idletime = if generation_in_progress {
        0.0
    } else {
        snapshot
            .last_generation_finished_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or_else(|| started_at.elapsed().as_secs_f64())
    };

    KcppPerfResponse {
        last_process: snapshot.last_process_secs,
        last_eval: snapshot.last_eval_secs,
        last_token_count: snapshot.last_token_count,
        last_seed: snapshot.last_seed,
        last_draft_success: 0,
        last_draft_failed: 0,
        total_gens: snapshot.total_gens,
        total_img_gens: snapshot.total_img_gens,
        total_tts_gens: snapshot.total_tts_gens,
        total_transcribe_gens: snapshot.total_transcribe_gens,
        stop_reason: snapshot.last_stop_reason,
        queue: if generation_in_progress { 1 } else { 0 },
        idle: if generation_in_progress { 0 } else { 1 },
        hordeexitcounter: -1,
        uptime,
        idletime,
        quiet,
        last_input_count: snapshot.last_input_count,
    }
}

pub fn record_text_generation(
    snapshot: &mut CompatPerfState,
    sampling: &SamplingParams,
    result: &GenerationResult,
) {
    snapshot.last_process_secs = result.prompt_eval_ms / 1_000.0;
    snapshot.last_eval_secs = if result.perf.t_eval_ms > 0.0 {
        result.perf.t_eval_ms / 1_000.0
    } else if result.tok_per_sec_avg > 0.0 {
        result.tokens_generated as f64 / result.tok_per_sec_avg
    } else {
        0.0
    };
    snapshot.last_token_count = result.tokens_generated;
    snapshot.last_input_count = result.prompt_tokens;
    snapshot.last_seed = sampling.seed;
    snapshot.last_stop_reason = 0;
    snapshot.total_gens += 1;
    snapshot.last_generation_finished_at = Some(Instant::now());
}

pub fn openai_usage(prompt_tokens: u32, completion_tokens: u32) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

pub fn openai_completion_response(
    id: &str,
    created: i64,
    model: &str,
    text: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": "stop",
        }],
        "usage": openai_usage(prompt_tokens, completion_tokens),
    })
}

pub fn openai_chat_response(
    id: &str,
    created: i64,
    model: &str,
    text: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "finish_reason": "stop",
        }],
        "usage": openai_usage(prompt_tokens, completion_tokens),
    })
}

pub fn openai_error(message: &str, error_type: &str, code: &str) -> Value {
    json!({
        "error": {
            "message": message,
            "type": error_type,
            "param": Value::Null,
            "code": code,
        }
    })
}

pub fn feature_unavailable_error(feature: &str) -> Value {
    json!({
        "error": format!("{feature} is not bundled in this Hypura compatibility slice yet"),
        "feature": feature,
        "release": KOBOLDCPP_COMPAT_RELEASE_TAG,
    })
}

pub fn flatten_chat_messages(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::ffi::PerfData;

    #[test]
    fn version_response_uses_pinned_schema_version() {
        let response = build_version_response(false, CompatFeatureFlags::default());
        assert_eq!(response.result, "KoboldCpp");
        assert_eq!(response.version, KOBOLDCPP_API_SCHEMA_VERSION);
        assert!(!response.embeddings);
    }

    #[test]
    fn perf_response_marks_idle_when_no_generation_is_running() {
        let mut snapshot = CompatPerfState::default();
        snapshot.last_generation_finished_at = Some(Instant::now());
        let response = build_perf_response(&snapshot, Instant::now(), false, false);
        assert_eq!(response.idle, 1);
        assert_eq!(response.queue, 0);
    }

    #[test]
    fn record_text_generation_updates_last_values() {
        let mut snapshot = CompatPerfState::default();
        let sampling = SamplingParams {
            seed: 777,
            ..SamplingParams::default()
        };
        let result = GenerationResult {
            text: "hello".into(),
            tokens_generated: 12,
            prompt_tokens: 34,
            tok_per_sec_avg: 6.0,
            prompt_eval_ms: 250.0,
            perf: PerfData {
                t_eval_ms: 2_000.0,
                ..PerfData::default()
            },
        };

        record_text_generation(&mut snapshot, &sampling, &result);

        assert_eq!(snapshot.last_seed, 777);
        assert_eq!(snapshot.last_token_count, 12);
        assert_eq!(snapshot.last_input_count, 34);
        assert_eq!(snapshot.total_gens, 1);
        assert!(snapshot.last_generation_finished_at.is_some());
    }
}
