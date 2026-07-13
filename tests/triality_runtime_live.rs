use std::path::Path;
use std::time::Instant;

use hypura::compute::ffi::{
    LlamaContext, LlamaModel, LlamaSampler, SamplingParams, TrialityContextConfig,
    TrialityExecution,
};
use hypura::compute::inference::{
    InferenceConfig, LlamaTurboquantCliBridge, load_model, resolve_runtime_setup,
};
use hypura::council::{
    AhaThresholds, AnswerCouncilConfig, CouncilRuntimeConfig, CouncilView, KaGateConfig,
    NoSafetyPenalty, context_config_for_view,
};
use hypura::model::turboquant_sidecar::TurboQuantMode;
use hypura::scheduler::types::{
    ComputeTarget, CouncilExecutionMode, CouncilMemoryBudget, CouncilParallelism,
    ResidencyPolicyConfig, TensorResidence,
};

fn admitted_budget(parallelism: CouncilParallelism) -> CouncilMemoryBudget {
    CouncilMemoryBudget {
        execution: CouncilExecutionMode::Answer,
        parallelism,
        context_count: if parallelism == CouncilParallelism::Parallel {
            3
        } else {
            1
        },
        estimated_kv_bytes: 0,
        estimated_controller_bytes: 0,
        admitted: true,
        reason: "live integration test admission".to_string(),
    }
}

fn generate_mode_once(
    model: &LlamaModel,
    inference: &InferenceConfig,
    triality: &TrialityContextConfig,
    prompt: &str,
) -> anyhow::Result<(Vec<i32>, u128)> {
    let started = Instant::now();
    let mut context = LlamaContext::new_with_triality(
        model,
        inference.n_ctx,
        inference.n_batch,
        inference.n_threads,
        triality,
    )?;
    let prompt_tokens = model.tokenize(prompt, true, true);
    anyhow::ensure!(
        !prompt_tokens.is_empty(),
        "live prompt tokenized to zero tokens"
    );
    for chunk in prompt_tokens.chunks(inference.n_batch as usize) {
        context.decode(chunk)?;
    }
    let mut sampler = LlamaSampler::new(&inference.sampling);
    let mut token_ids = Vec::new();
    for _ in 0..inference.sampling.max_tokens {
        let token_id = sampler.sample(&mut context, -1);
        if model.is_eog(token_id) {
            break;
        }
        token_ids.push(token_id);
        context.decode(&[token_id])?;
    }
    Ok((token_ids, started.elapsed().as_millis()))
}

fn run_mode_twice(
    mode: &str,
    model: &LlamaModel,
    inference: &InferenceConfig,
    triality: &TrialityContextConfig,
    prompt: &str,
) -> anyhow::Result<serde_json::Value> {
    let (first_tokens, first_latency_ms) = generate_mode_once(model, inference, triality, prompt)?;
    let (second_tokens, second_latency_ms) =
        generate_mode_once(model, inference, triality, prompt)?;
    let deterministic = first_tokens == second_tokens;
    anyhow::ensure!(deterministic, "{mode} generation was not deterministic");
    Ok(serde_json::json!({
        "executed": true,
        "deterministic": deterministic,
        "first_token_ids": first_tokens,
        "second_token_ids": second_tokens,
        "latency_ms": [first_latency_ms, second_latency_ms],
    }))
}

#[test]
#[ignore = "requires HYPURA_TEST_TRIALITY_MODEL pointing to an inference-capable schema-v2 GGUF"]
fn live_contexts_are_independent_and_council_peak_matches_admission() -> anyhow::Result<()> {
    let model_path = std::env::var_os("HYPURA_TEST_TRIALITY_MODEL")
        .ok_or_else(|| anyhow::anyhow!("HYPURA_TEST_TRIALITY_MODEL must be set"))?;
    let model_path = Path::new(&model_path);
    let bridge = LlamaTurboquantCliBridge {
        tq_developer_override: true,
        tq_allow_identity_view_fallback: true,
        ..LlamaTurboquantCliBridge::default()
    };
    let runtime = resolve_runtime_setup(
        model_path,
        128,
        TurboQuantMode::ResearchKvSplit,
        None,
        bridge,
        ResidencyPolicyConfig::default(),
        false,
    )?;
    let triality = runtime
        .triality
        .clone()
        .ok_or_else(|| anyhow::anyhow!("schema-v2 fixture did not resolve Triality policy"))?;
    let config = InferenceConfig {
        n_ctx: 128,
        n_batch: 32,
        triality: Some(triality.clone()),
        sampling: SamplingParams {
            max_tokens: 2,
            seed: 17,
            ..SamplingParams::default()
        },
        ..InferenceConfig::default()
    };
    let mut live_plan = runtime.plan.clone();
    for placement in live_plan.tensor_placements.values_mut() {
        placement.residence = TensorResidence::HostPageable;
        placement.compute_target = ComputeTarget::CpuFallback;
    }
    let loaded = load_model(
        model_path,
        &config,
        runtime.n_gpu_layers,
        &live_plan,
        &runtime.gguf,
        &runtime.turboquant,
    )?;

    let vector_config = context_config_for_view(&triality.context, CouncilView::Vector)?;
    let spinor_config = context_config_for_view(&triality.context, CouncilView::SpinorPlusProxy)?;
    let vector_context = LlamaContext::new_with_triality(
        &loaded.model,
        config.n_ctx,
        config.n_batch,
        config.n_threads,
        &vector_config,
    )?;
    let spinor_context = LlamaContext::new_with_triality(
        &loaded.model,
        config.n_ctx,
        config.n_batch,
        config.n_threads,
        &spinor_config,
    )?;
    let vector_roundtrip = vector_context.triality_config()?;
    let spinor_roundtrip = spinor_context.triality_config()?;
    assert_ne!(
        vector_roundtrip.layers[0].active_branch_mask,
        spinor_roundtrip.layers[0].active_branch_mask
    );
    drop((vector_context, spinor_context));

    let prompt =
        std::env::var("HYPURA_TEST_TRIALITY_PROMPT").unwrap_or_else(|_| "Hello".to_string());
    let vector_mode = run_mode_twice("vector", &loaded.model, &config, &vector_config, &prompt)?;
    let mut best_per_layer_config = triality.context.clone();
    best_per_layer_config.execution = TrialityExecution::BestPerLayer;
    for layer in &mut best_per_layer_config.layers {
        layer.active_branch_mask = 0b111;
    }
    let best_per_layer_mode = run_mode_twice(
        "best_per_layer",
        &loaded.model,
        &config,
        &best_per_layer_config,
        &prompt,
    )?;
    let mut attention_config = triality.context.clone();
    attention_config.execution = TrialityExecution::AttentionLogitConsensus;
    for layer in &mut attention_config.layers {
        layer.active_branch_mask = 0b111;
    }
    let attention_mode = run_mode_twice(
        "attention_logit_consensus",
        &loaded.model,
        &config,
        &attention_config,
        &prompt,
    )?;
    let run = |parallelism| -> anyhow::Result<_> {
        loaded
            .council_runtime(CouncilRuntimeConfig {
                inference: config.clone(),
                triality: triality.context.clone(),
                memory_budget: admitted_budget(parallelism),
                answer: AnswerCouncilConfig::default(),
                ka_gate: KaGateConfig::default(),
                moment_degree: 3,
                memory_ratio: 1.0,
                attention_consensus_requested: false,
                attention_consensus_required: false,
                aha_enabled: false,
                aha_thresholds: AhaThresholds::default(),
                urt: None,
            })?
            .execute(
                "triality-runtime-live",
                &prompt,
                &config.sampling,
                &[],
                &NoSafetyPenalty,
                None,
                None,
            )
    };

    let sequential_started = Instant::now();
    let sequential = run(CouncilParallelism::Sequential)?;
    let sequential_latency_ms = sequential_started.elapsed().as_millis();
    let parallel_started = Instant::now();
    let parallel = run(CouncilParallelism::Parallel)?;
    let parallel_latency_ms = parallel_started.elapsed().as_millis();
    assert_eq!(sequential.peak_live_contexts, 1);
    assert_eq!(parallel.peak_live_contexts, 3);
    assert_eq!(sequential.detailed_cross_scores.len(), 9);
    assert_eq!(parallel.detailed_cross_scores.len(), 9);
    assert!(
        sequential
            .answer
            .cross_scores
            .scores
            .iter()
            .flatten()
            .all(|score| score.is_finite())
    );
    assert_eq!(
        sequential.answer.cross_scores.scores,
        parallel.answer.cross_scores.scores
    );
    assert_eq!(
        sequential
            .answer
            .candidates
            .each_ref()
            .map(|candidate| candidate.token_ids.as_slice()),
        parallel
            .answer
            .candidates
            .each_ref()
            .map(|candidate| candidate.token_ids.as_slice())
    );

    let ncka_rank = sequential.moments.as_ref().map(|moments| {
        serde_json::json!({
            "numerical_rank": moments.numerical_rank,
            "effective_rank": moments.effective_rank,
        })
    });
    let ncka_measured = ncka_rank.is_some();
    let report = serde_json::json!({
        "schema": "hypura.triality_live_report.v1",
        "fixture": model_path.display().to_string(),
        "fixture_rotation": "identity_dev",
        "quality_claim": false,
        "prompt": prompt,
        "seed": config.sampling.seed,
        "max_tokens": config.sampling.max_tokens,
        "modes": {
            "vector": vector_mode,
            "best_per_layer": best_per_layer_mode,
            "attention_logit_consensus": attention_mode,
            "answer_council": {
                "executed": true,
                "deterministic": sequential.answer.cross_scores.scores
                    == parallel.answer.cross_scores.scores,
                "sequential_latency_ms": sequential_latency_ms,
                "parallel_latency_ms": parallel_latency_ms,
                "sequential_peak_live_contexts": sequential.peak_live_contexts,
                "parallel_peak_live_contexts": parallel.peak_live_contexts,
            },
        },
        "cross_score_matrix": sequential.answer.cross_scores.scores,
        "winner_margin": sequential.answer.winner_margin,
        "selected_view": sequential.answer.selected_view.as_str(),
        "candidate_latency_ms": sequential
            .answer
            .candidates
            .each_ref()
            .map(|candidate| candidate.runtime_ms),
        "ncka_rank": ncka_rank,
        "ncka_measurement": if ncka_measured {
            "measured"
        } else {
            "unavailable_without_low_level_metrics"
        },
        "urt_error": serde_json::Value::Null,
        "urt_measurement": "unavailable_without_cross_representation_observation",
        "aha": {
            "enabled": false,
            "reason": "no_labeled_evidence",
        },
    });
    println!("TRIALITY_LIVE_REPORT={report}");
    Ok(())
}
