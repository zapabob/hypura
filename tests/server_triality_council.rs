use hypura::council::CouncilView;
use hypura::server::ollama_types::{
    ChatRequest, OpenAiChatCompletionRequest, TrialityCouncilRequest, TrialityCouncilResponse,
};
use hypura::telemetry::metrics::{TelemetryEmitter, TelemetryEvent};

#[test]
fn dedicated_request_accepts_all_v1_fields() {
    let request: TrialityCouncilRequest = serde_json::from_value(serde_json::json!({
        "model": "model.gguf",
        "prompt": "answer without private reasoning",
        "max_tokens": 128,
        "temperature": 0.25,
        "seed": 7,
        "parallelism": "sequential",
        "attention_consensus": false,
        "cross_score": true,
        "synthesis": false,
        "aha": true,
        "trace": true,
        "stream": true
    }))
    .unwrap();
    assert_eq!(request.model.as_deref(), Some("model.gguf"));
    assert!(request.prompt.is_some());
    assert!(request.stream);
}

#[test]
fn request_rejects_unknown_fields_and_parallelism_values() {
    assert!(
        serde_json::from_value::<TrialityCouncilRequest>(serde_json::json!({
            "prompt": "hello",
            "unknown": true
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<TrialityCouncilRequest>(serde_json::json!({
            "prompt": "hello",
            "parallelism": "sometimes"
        }))
        .is_err()
    );
}

#[test]
fn existing_chat_requests_remain_compatible_without_extension() {
    let openai: OpenAiChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "model.gguf",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    }))
    .unwrap();
    assert!(openai.hypura.is_none());

    let ollama: ChatRequest = serde_json::from_value(serde_json::json!({
        "model": "model.gguf",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false
    }))
    .unwrap();
    assert!(ollama.hypura.is_none());
}

#[test]
fn response_core_fields_are_stable_without_trace() {
    let response = TrialityCouncilResponse {
        id: "tc-1".to_string(),
        object: "triality.council".to_string(),
        model: "model.gguf".to_string(),
        selected_text: "final".to_string(),
        selected_view: CouncilView::Vector,
        candidate_scores: [1.0, 0.5, 0.25],
        winner_margin: 0.5,
        agreement: 0.75,
        aha: None,
        trace: None,
    };
    let value = serde_json::to_value(response).unwrap();
    for field in [
        "id",
        "object",
        "model",
        "selected_text",
        "selected_view",
        "candidate_scores",
        "winner_margin",
        "agreement",
    ] {
        assert!(value.get(field).is_some(), "missing {field}");
    }
    assert!(value.get("trace").is_none());
}

#[tokio::test]
async fn broadcast_event_payload_never_contains_prompt_or_candidate_secrets() {
    let secret = "secret-prompt-and-branch-content";
    let prompt = secret.to_string();
    let candidates = [secret.to_string(), secret.to_string(), secret.to_string()];
    let telemetry = TelemetryEmitter::new(8);
    let mut events = telemetry.subscribe();
    telemetry.emit(TelemetryEvent::TrialityConsensusCompleted {
        request_id: "tc-safe".to_string(),
        selected_view: CouncilView::Vector,
        candidate_scores: [1.0, 0.5, 0.25],
        winner_margin: 0.5,
        agreement: 0.75,
        result_persisted: false,
    });
    let payload = serde_json::to_string(&events.recv().await.unwrap()).unwrap();
    assert!(!payload.contains(&prompt));
    assert!(
        candidates
            .iter()
            .all(|candidate| !payload.contains(candidate))
    );
}
