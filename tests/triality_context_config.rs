use hypura::compute::ffi::{
    SamplingParams, TrialityBranchConfig, TrialityContextConfig, TrialityExecution,
    TrialityLayerConfig, TrialityView,
};
use hypura::compute::inference::{InferenceConfig, TrialityRuntimePolicy};

fn branch(view: TrialityView, weight: f32) -> TrialityBranchConfig {
    TrialityBranchConfig {
        view,
        weight,
        bias: 0.0,
        scale: 1.0,
        temperature: 1.0,
        expected_error: 0.0,
        bits_per_channel: 4.0,
    }
}

fn policy() -> TrialityRuntimePolicy {
    TrialityRuntimePolicy {
        context: TrialityContextConfig {
            schema_version: 2,
            execution: TrialityExecution::BestPerLayer,
            layers: vec![TrialityLayerConfig {
                branches: [
                    branch(TrialityView::Vector, 0.5),
                    branch(TrialityView::SpinorPlusProxy, 0.25),
                    branch(TrialityView::SpinorMinusProxy, 0.25),
                ],
                active_branch_mask: 0b111,
            }],
            required: true,
            trace_enabled: true,
            js_fallback_threshold: 0.2,
            allow_identity_view_fallback: false,
        },
        ncka_required: true,
        urt_enabled: true,
        embedded_override_allowed: false,
        ncka: None,
        urt: None,
    }
}

#[test]
fn legacy_inference_config_without_triality_remains_compatible() {
    let config: InferenceConfig = serde_json::from_value(serde_json::json!({
        "n_ctx": 4096,
        "n_batch": 512,
        "n_threads": 4,
        "sampling": SamplingParams::default()
    }))
    .unwrap();
    assert!(config.triality.is_none());
}

#[test]
fn explicit_runtime_policy_roundtrips_without_process_state() {
    let config = InferenceConfig {
        triality: Some(policy()),
        ..InferenceConfig::default()
    };
    let value = serde_json::to_value(&config).unwrap();
    let decoded: InferenceConfig = serde_json::from_value(value).unwrap();
    let decoded_policy = decoded.triality.unwrap();
    assert_eq!(decoded_policy, policy());
    assert_eq!(
        decoded_policy.context.execution,
        TrialityExecution::BestPerLayer
    );
}

#[test]
fn execution_capability_bits_are_distinct() {
    let values = [
        TrialityExecution::SingleView,
        TrialityExecution::BestPerLayer,
        TrialityExecution::AttentionLogitConsensus,
        TrialityExecution::ResidualParity,
    ];
    let mut aggregate = 0_u32;
    for value in values {
        let bit = value.capability_bit();
        assert_eq!(aggregate & bit, 0);
        aggregate |= bit;
    }
    assert_eq!(aggregate.count_ones(), 4);
}
