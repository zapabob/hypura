use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::compute::ffi::{
    LlamaContext, LlamaModel, LlamaSampler, SamplingParams, TrialityContextConfig,
    TrialityExecution, TrialityModelCapabilities, TrialityView,
};
use crate::compute::inference::InferenceConfig;
use crate::scheduler::types::{CouncilExecutionMode, CouncilMemoryBudget, CouncilParallelism};
use crate::urt::{RepresentationId, UrtObservation};

use super::aha::{
    AhaCalibrationEvidence, AhaEvaluation, AhaEvidence, AhaInput, AhaSafetyEvidence, AhaThresholds,
    classify_aha_with_status,
};
use super::cross_score::{
    CandidateViewScore, CrossScoreMatrix, TeacherForcedScoreInput, token_log_probability,
};
use super::ka_gate::{
    GateSource, KaController, KaGateConfig, KaGateEvaluation, KaGateOutput, evaluate_ka_gate,
};
use super::moments::{BranchMomentObservables, CouncilMomentInput, CouncilMomentVector};
use super::scoring::{AnswerCouncilConfig, AnswerCouncilResult, SafetyPenalty, select_answer};
use super::types::{CouncilCandidate, CouncilView};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum AttentionCapabilityDecision {
    NotRequested,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouncilAhaRuntimeEvidence {
    pub safety: Option<AhaSafetyEvidence>,
    pub calibration: Option<AhaCalibrationEvidence>,
    pub evidence: Option<AhaEvidence>,
    pub urt_errors: Option<(f64, f64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouncilUrtDescriptor {
    pub representation: RepresentationId,
    pub operator_word: Vec<String>,
    pub operator_word_sha256: String,
    pub tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilRuntimeConfig {
    pub inference: InferenceConfig,
    pub triality: TrialityContextConfig,
    pub memory_budget: CouncilMemoryBudget,
    pub answer: AnswerCouncilConfig,
    pub ka_gate: KaGateConfig,
    pub moment_degree: u32,
    pub memory_ratio: f32,
    pub attention_consensus_requested: bool,
    pub attention_consensus_required: bool,
    pub aha_enabled: bool,
    pub aha_thresholds: AhaThresholds,
    pub urt: Option<CouncilUrtDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouncilExecutionResult {
    pub answer: AnswerCouncilResult,
    pub detailed_cross_scores: Vec<CandidateViewScore>,
    pub moments: Option<CouncilMomentVector>,
    pub ka_gate: KaGateEvaluation,
    pub aha: Option<AhaEvaluation>,
    pub urt_observation: Option<UrtObservation>,
    pub attention: AttentionCapabilityDecision,
    pub capabilities: TrialityModelCapabilities,
    pub peak_live_contexts: u32,
}

#[derive(Default)]
struct ContextPeakTracker {
    live: AtomicU32,
    peak: AtomicU32,
}

impl ContextPeakTracker {
    fn track(&self, context: LlamaContext) -> TrackedContext<'_> {
        let live = self.live.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak.fetch_max(live, Ordering::Relaxed);
        TrackedContext {
            context: Some(context),
            tracker: self,
        }
    }

    fn peak(&self) -> u32 {
        self.peak.load(Ordering::Relaxed)
    }
}

struct TrackedContext<'a> {
    context: Option<LlamaContext>,
    tracker: &'a ContextPeakTracker,
}

impl Deref for TrackedContext<'_> {
    type Target = LlamaContext;

    fn deref(&self) -> &Self::Target {
        self.context.as_ref().expect("tracked context must be live")
    }
}

impl DerefMut for TrackedContext<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.context.as_mut().expect("tracked context must be live")
    }
}

impl Drop for TrackedContext<'_> {
    fn drop(&mut self) {
        drop(self.context.take());
        self.tracker.live.fetch_sub(1, Ordering::Relaxed);
    }
}

pub struct CouncilRuntime<'a> {
    model: &'a LlamaModel,
    config: CouncilRuntimeConfig,
    capabilities: TrialityModelCapabilities,
    attention: AttentionCapabilityDecision,
}

impl<'a> CouncilRuntime<'a> {
    pub fn new(model: &'a LlamaModel, config: CouncilRuntimeConfig) -> anyhow::Result<Self> {
        validate_runtime_config(&config)?;
        if let Some(policy) = config.inference.triality.as_ref() {
            anyhow::ensure!(
                policy.context == config.triality,
                "Council Triality context conflicts with the inference policy"
            );
        }
        let capabilities = model.triality_capabilities()?;
        anyhow::ensure!(
            capabilities.metadata_present,
            "Answer Council requires embedded Triality metadata"
        );
        anyhow::ensure!(
            capabilities.three_view_bundle,
            "Answer Council requires a complete three-view bundle"
        );
        anyhow::ensure!(
            capabilities.supports(TrialityExecution::SingleView),
            "llama.cpp model does not support per-context single-view execution"
        );
        anyhow::ensure!(
            capabilities.n_layers as usize == config.triality.layers.len(),
            "Triality layer count does not match llama.cpp model capabilities"
        );
        anyhow::ensure!(
            !config.ka_gate.required
                || (capabilities.ncka_available && !capabilities.ncka_static_fallback_selected),
            "required NC-KA controller is unavailable or selected a static fallback"
        );
        anyhow::ensure!(
            config.urt.is_none() || capabilities.urt_available,
            "Council URT descriptor requires embedded URT metadata"
        );

        anyhow::ensure!(
            !config.attention_consensus_requested && !config.attention_consensus_required,
            "dedicated Answer Council does not execute native attention-logit consensus; use the normal context runtime for that mode"
        );
        let attention = AttentionCapabilityDecision::NotRequested;

        Ok(Self {
            model,
            config,
            capabilities,
            attention,
        })
    }

    pub fn capabilities(&self) -> &TrialityModelCapabilities {
        &self.capabilities
    }

    pub fn attention_decision(&self) -> &AttentionCapabilityDecision {
        &self.attention
    }

    pub fn execute(
        &self,
        request_id: &str,
        prompt: &str,
        sampling: &SamplingParams,
        stop_sequences: &[String],
        safety: &dyn SafetyPenalty,
        ka_controller: Option<&dyn KaController>,
        aha_evidence: Option<&CouncilAhaRuntimeEvidence>,
    ) -> anyhow::Result<CouncilExecutionResult> {
        anyhow::ensure!(
            !request_id.trim().is_empty(),
            "Council request id must not be empty"
        );
        anyhow::ensure!(!prompt.is_empty(), "Council prompt must not be empty");
        anyhow::ensure!(
            sampling.max_tokens > 0,
            "Council max_tokens must be positive"
        );
        anyhow::ensure!(
            !self.config.attention_consensus_requested && !self.config.attention_consensus_required,
            "dedicated Answer Council does not execute native attention-logit consensus; use the normal context runtime for that mode"
        );

        let context_tracker = ContextPeakTracker::default();
        let candidates = self.generate_candidates_with_tracker(
            prompt,
            sampling,
            stop_sequences,
            &context_tracker,
        )?;
        let (cross_scores, detailed_cross_scores) =
            self.cross_score_with_tracker(prompt, &candidates, &context_tracker)?;
        let moments = build_moments(
            &candidates,
            &cross_scores,
            self.config.moment_degree,
            self.config.memory_ratio,
        )?;
        let ka_gate = resolve_ka_gate(moments.as_ref(), &self.config.ka_gate, ka_controller)?;
        let mut answer_config = self.config.answer.clone();
        if self.config.ka_gate.enabled {
            answer_config.evaluator_weights = ka_gate.output.evaluator_weights.map(f64::from);
        }
        let mut answer = select_answer(
            request_id,
            candidates,
            &cross_scores,
            &answer_config,
            safety,
        )?;

        let aha = if self.config.aha_enabled {
            let evidence = aha_evidence.cloned().unwrap_or(CouncilAhaRuntimeEvidence {
                safety: None,
                calibration: None,
                evidence: None,
                urt_errors: None,
            });
            let vector_score = answer.candidate_scores[CouncilView::Vector.index()];
            let selected_score = answer.candidate_scores[answer.selected_view.index()];
            let input = AhaInput {
                selected_view: answer.selected_view,
                baseline_view: CouncilView::Vector,
                pre_consensus_js: candidates_max_js(&answer.candidates),
                council_score_gain: selected_score - vector_score,
                winner_margin: answer.winner_margin,
                safety: evidence.safety,
                calibration: evidence.calibration,
                urt_errors: evidence.urt_errors,
                moment_effective_rank: moments
                    .as_ref()
                    .map(|value| f64::from(value.effective_rank)),
                evidence: evidence.evidence,
            };
            let evaluation = classify_aha_with_status(&input, &self.config.aha_thresholds)?;
            answer.aha = evaluation.event.clone();
            Some(evaluation)
        } else {
            None
        };

        let urt_observation = self.config.urt.as_ref().map(|descriptor| {
            let mut representation = descriptor.representation.clone();
            representation.view = Some(answer.selected_view.as_str().to_string());
            let row = cross_scores.scores[answer.selected_view.index()];
            UrtObservation {
                request_id: request_id.to_string(),
                representation,
                state_id: request_id.to_string(),
                layer: None,
                operator_word: descriptor.operator_word.clone(),
                operator_word_sha256: descriptor.operator_word_sha256.clone(),
                observable: "selected_candidate_mean_log_likelihood".to_string(),
                value_real: row.iter().sum::<f64>() / 3.0,
                value_imag: 0.0,
                tolerance: descriptor.tolerance,
            }
        });
        let peak_live_contexts = context_tracker.peak();
        anyhow::ensure!(
            peak_live_contexts > 0 && peak_live_contexts <= self.config.memory_budget.context_count,
            "observed Council context peak exceeds memory admission"
        );

        Ok(CouncilExecutionResult {
            answer,
            detailed_cross_scores,
            moments,
            ka_gate,
            aha,
            urt_observation,
            attention: self.attention.clone(),
            capabilities: self.capabilities.clone(),
            peak_live_contexts,
        })
    }

    pub fn generate_candidates(
        &self,
        prompt: &str,
        sampling: &SamplingParams,
        stop_sequences: &[String],
    ) -> anyhow::Result<[CouncilCandidate; 3]> {
        self.generate_candidates_with_tracker(
            prompt,
            sampling,
            stop_sequences,
            &ContextPeakTracker::default(),
        )
    }

    fn generate_candidates_with_tracker(
        &self,
        prompt: &str,
        sampling: &SamplingParams,
        stop_sequences: &[String],
        context_tracker: &ContextPeakTracker,
    ) -> anyhow::Result<[CouncilCandidate; 3]> {
        match self.config.memory_budget.parallelism {
            CouncilParallelism::Sequential => {
                let values = CouncilView::ALL.map(|view| {
                    self.generate_candidate(view, prompt, sampling, stop_sequences, context_tracker)
                });
                collect_three(values)
            }
            CouncilParallelism::Parallel => std::thread::scope(|scope| {
                let handles = CouncilView::ALL.map(|view| {
                    scope.spawn(move || {
                        self.generate_candidate(
                            view,
                            prompt,
                            sampling,
                            stop_sequences,
                            context_tracker,
                        )
                    })
                });
                collect_joined(handles)
            }),
            CouncilParallelism::Auto => anyhow::bail!(
                "Council Auto parallelism must be resolved by CouncilMemoryBudget before runtime"
            ),
        }
    }

    pub fn cross_score(
        &self,
        prompt: &str,
        candidates: &[CouncilCandidate; 3],
    ) -> anyhow::Result<(CrossScoreMatrix, Vec<CandidateViewScore>)> {
        self.cross_score_with_tracker(prompt, candidates, &ContextPeakTracker::default())
    }

    fn cross_score_with_tracker(
        &self,
        prompt: &str,
        candidates: &[CouncilCandidate; 3],
        context_tracker: &ContextPeakTracker,
    ) -> anyhow::Result<(CrossScoreMatrix, Vec<CandidateViewScore>)> {
        let per_evaluator = match self.config.memory_budget.parallelism {
            CouncilParallelism::Sequential => {
                let mut values = Vec::with_capacity(3);
                for evaluator in CouncilView::ALL {
                    values.push(self.score_evaluator(
                        evaluator,
                        prompt,
                        candidates,
                        context_tracker,
                    )?);
                }
                values
            }
            CouncilParallelism::Parallel => std::thread::scope(|scope| {
                let handles = CouncilView::ALL.map(|evaluator| {
                    scope.spawn(move || {
                        self.score_evaluator(evaluator, prompt, candidates, context_tracker)
                    })
                });
                handles
                    .into_iter()
                    .map(|handle| {
                        handle
                            .join()
                            .map_err(|_| anyhow::anyhow!("Council evaluator thread panicked"))?
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })?,
            CouncilParallelism::Auto => anyhow::bail!(
                "Council Auto parallelism must be resolved by CouncilMemoryBudget before runtime"
            ),
        };
        let inputs = per_evaluator.into_iter().flatten().collect::<Vec<_>>();
        Ok(CrossScoreMatrix::from_teacher_forced(&inputs)?)
    }

    fn generate_candidate(
        &self,
        view: CouncilView,
        prompt: &str,
        sampling: &SamplingParams,
        stop_sequences: &[String],
        context_tracker: &ContextPeakTracker,
    ) -> anyhow::Result<CouncilCandidate> {
        let triality = context_config_for_view(&self.config.triality, view)?;
        let mut context = context_tracker.track(LlamaContext::new_with_triality(
            self.model,
            self.config.inference.n_ctx,
            self.config.inference.n_batch,
            self.config.inference.n_threads,
            &triality,
        )?);
        let prompt_tokens = self.model.tokenize(prompt, true, true);
        anyhow::ensure!(
            !prompt_tokens.is_empty(),
            "Council prompt tokenized to zero tokens"
        );
        decode_prompt(&mut context, &prompt_tokens, self.config.inference.n_batch)?;

        let mut sampler = LlamaSampler::new(sampling);
        let mut text = String::new();
        let mut token_ids = Vec::new();
        let started = Instant::now();
        for _ in 0..sampling.max_tokens {
            let token_id = sampler.sample(&mut context, -1);
            let is_eog = self.model.is_eog(token_id);
            if is_eog {
                break;
            }
            token_ids.push(token_id);
            text.push_str(&self.model.token_to_piece(token_id));
            if let Some(stop) = stop_sequences
                .iter()
                .find(|stop| !stop.is_empty() && text.ends_with(stop.as_str()))
            {
                text.truncate(text.len().saturating_sub(stop.len()));
                token_ids = self.model.tokenize(&text, false, true);
                break;
            }
            context.decode(&[token_id])?;
        }
        anyhow::ensure!(
            !token_ids.is_empty(),
            "Council candidate generated no tokens"
        );
        let elapsed = started.elapsed();
        let seconds = elapsed.as_secs_f64();
        let metrics = if triality.trace_enabled {
            match context.triality_metrics_optional()? {
                Some(metrics) => Some(metrics),
                None if self.config.ka_gate.required => {
                    anyhow::bail!("required NC-KA evaluation cannot run without low-level metrics")
                }
                None => None,
            }
        } else {
            None
        };
        let generated_tokens = u32::try_from(token_ids.len())?;
        Ok(CouncilCandidate {
            view,
            text,
            token_ids,
            prompt_tokens: u32::try_from(prompt_tokens.len())?,
            generated_tokens,
            tok_per_sec: if seconds > 0.0 {
                f64::from(generated_tokens) / seconds
            } else {
                0.0
            },
            runtime_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
            low_level_metrics: metrics,
        })
    }

    fn score_evaluator(
        &self,
        evaluator: CouncilView,
        prompt: &str,
        candidates: &[CouncilCandidate; 3],
        context_tracker: &ContextPeakTracker,
    ) -> anyhow::Result<Vec<TeacherForcedScoreInput>> {
        let triality = context_config_for_view(&self.config.triality, evaluator)?;
        let mut context = context_tracker.track(LlamaContext::new_with_triality(
            self.model,
            self.config.inference.n_ctx,
            self.config.inference.n_batch,
            self.config.inference.n_threads,
            &triality,
        )?);
        let prompt_tokens = self.model.tokenize(prompt, true, true);
        anyhow::ensure!(
            !prompt_tokens.is_empty(),
            "Council prompt tokenized to zero tokens"
        );
        let (last_prompt_token, prompt_prefix) = prompt_tokens
            .split_last()
            .ok_or_else(|| anyhow::anyhow!("Council prompt tokenized to zero tokens"))?;
        decode_prompt(&mut context, prompt_prefix, self.config.inference.n_batch)?;
        let candidate_position = i32::try_from(prompt_prefix.len())?;
        let vocab_size = self.model.vocab_size()?;
        let mut outputs = Vec::with_capacity(3);
        for candidate in candidates {
            context.decode(&[*last_prompt_token])?;
            let mut token_log_probabilities = Vec::with_capacity(candidate.token_ids.len());
            for token_id in &candidate.token_ids {
                token_log_probabilities.push(token_log_probability(
                    context.logits(vocab_size)?,
                    *token_id,
                )?);
                context.decode(&[*token_id])?;
            }
            outputs.push(TeacherForcedScoreInput {
                candidate: candidate.view,
                evaluator,
                token_log_probabilities,
            });
            context.remove_sequence_tokens(-1, candidate_position, -1)?;
        }
        Ok(outputs)
    }
}

pub fn context_config_for_view(
    base: &TrialityContextConfig,
    view: CouncilView,
) -> anyhow::Result<TrialityContextConfig> {
    let target = triality_view(view);
    let mut config = base.clone();
    config.execution = TrialityExecution::SingleView;
    for layer in &mut config.layers {
        let mut selected = None;
        for (index, branch) in layer.branches.iter_mut().enumerate() {
            let active = branch.view == target;
            branch.weight = if active { 1.0 } else { 0.0 };
            if active {
                anyhow::ensure!(
                    selected.replace(index).is_none(),
                    "duplicate Triality view in layer"
                );
            }
        }
        let selected =
            selected.ok_or_else(|| anyhow::anyhow!("Triality layer is missing {view:?}"))?;
        layer.active_branch_mask = 1_u32 << selected;
    }
    Ok(config)
}

fn triality_view(view: CouncilView) -> TrialityView {
    match view {
        CouncilView::Vector => TrialityView::Vector,
        CouncilView::SpinorPlusProxy => TrialityView::SpinorPlusProxy,
        CouncilView::SpinorMinusProxy => TrialityView::SpinorMinusProxy,
    }
}

fn validate_runtime_config(config: &CouncilRuntimeConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        !config.attention_consensus_requested && !config.attention_consensus_required,
        "dedicated Answer Council does not execute native attention-logit consensus; use the normal context runtime for that mode"
    );
    anyhow::ensure!(
        config.memory_budget.admitted,
        "{}",
        config.memory_budget.reason
    );
    anyhow::ensure!(
        config.triality.schema_version == 2 && !config.triality.layers.is_empty(),
        "Council runtime requires a non-empty schema-v2 Triality context"
    );
    anyhow::ensure!(
        !config.ka_gate.required || config.triality.trace_enabled,
        "required NC-KA evaluation requires Triality telemetry"
    );
    anyhow::ensure!(
        matches!(
            config.memory_budget.execution,
            CouncilExecutionMode::Answer | CouncilExecutionMode::Hybrid
        ),
        "CouncilRuntime requires Answer or Hybrid memory admission"
    );
    let expected_contexts = match config.memory_budget.parallelism {
        CouncilParallelism::Sequential => 1,
        CouncilParallelism::Parallel => 3,
        CouncilParallelism::Auto => anyhow::bail!(
            "Council Auto parallelism must be resolved by CouncilMemoryBudget before runtime"
        ),
    };
    anyhow::ensure!(
        config.memory_budget.context_count == expected_contexts,
        "Council memory admission context count does not match resolved parallelism"
    );
    anyhow::ensure!(
        config.memory_ratio.is_finite() && config.memory_ratio >= 0.0,
        "Council memory ratio must be finite and non-negative"
    );
    if let Some(urt) = &config.urt {
        anyhow::ensure!(
            !urt.operator_word.is_empty()
                && urt
                    .operator_word
                    .iter()
                    .all(|operator| !operator.trim().is_empty())
                && !urt.operator_word_sha256.trim().is_empty()
                && !urt.representation.model_hash.trim().is_empty()
                && !urt.representation.backend.trim().is_empty()
                && !urt.representation.precision.trim().is_empty()
                && urt.tolerance.is_finite()
                && urt.tolerance >= 0.0,
            "Council URT descriptor is incomplete or invalid"
        );
    }
    Ok(())
}

fn decode_prompt(
    context: &mut LlamaContext,
    prompt_tokens: &[i32],
    batch_size: u32,
) -> anyhow::Result<()> {
    anyhow::ensure!(batch_size > 0, "Council batch size must be positive");
    for chunk in prompt_tokens.chunks(batch_size as usize) {
        context.decode(chunk)?;
    }
    Ok(())
}

fn collect_three<T>(values: [anyhow::Result<T>; 3]) -> anyhow::Result<[T; 3]> {
    let mut output = Vec::with_capacity(3);
    for value in values {
        output.push(value?);
    }
    output
        .try_into()
        .map_err(|_| anyhow::anyhow!("Council requires exactly three views"))
}

fn collect_joined<T>(
    handles: [std::thread::ScopedJoinHandle<'_, anyhow::Result<T>>; 3],
) -> anyhow::Result<[T; 3]> {
    let mut output = Vec::with_capacity(3);
    for handle in handles {
        output.push(
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("Council generation thread panicked"))??,
        );
    }
    output
        .try_into()
        .map_err(|_| anyhow::anyhow!("Council requires exactly three views"))
}

fn build_moments(
    candidates: &[CouncilCandidate; 3],
    cross_scores: &CrossScoreMatrix,
    degree: u32,
    memory_ratio: f32,
) -> anyhow::Result<Option<CouncilMomentVector>> {
    let Some(metrics) = candidates
        .iter()
        .map(|candidate| candidate.low_level_metrics.as_ref())
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let mut entropy = [0.0_f32; 3];
    let mut orthogonality = [0.0_f32; 3];
    let mut determinant = [0.0_f32; 3];
    let mut expected_error = [0.0_f32; 3];
    let mut pairwise_js = [0.0_f32; 3];
    for candidate in candidates {
        let index = candidate.view.index();
        let metric = candidate.low_level_metrics.as_ref().unwrap();
        let branch = &metric.branches[index];
        entropy[index] = branch.probability_entropy as f32;
        orthogonality[index] = branch.orthogonality_error as f32;
        determinant[index] = branch.determinant_error as f32;
        expected_error[index] = branch.expected_quantisation_error as f32;
    }
    for index in 0..3 {
        pairwise_js[index] = (metrics
            .iter()
            .map(|metric| metric.pairwise_js[index])
            .sum::<f64>()
            / 3.0) as f32;
    }
    let means = cross_scores.scores.map(|row| row.iter().sum::<f64>() / 3.0);
    let mut ranked = means;
    ranked.sort_by(f64::total_cmp);
    let winner_margin = (ranked[2] - ranked[1]) as f32;
    let fastest = candidates
        .iter()
        .map(|candidate| candidate.runtime_ms.max(1))
        .min()
        .unwrap_or(1);
    let slowest = candidates
        .iter()
        .map(|candidate| candidate.runtime_ms.max(1))
        .max()
        .unwrap_or(1);
    Ok(Some(CouncilMomentVector::from_input(
        &CouncilMomentInput {
            branches: BranchMomentObservables {
                branch_entropy: entropy,
                orthogonality_error: orthogonality,
                determinant_error: determinant,
                expected_quantisation_error: expected_error,
            },
            pairwise_js,
            cross_scores: cross_scores.scores,
            winner_margin,
            latency_multiplier: slowest as f32 / fastest as f32,
            memory_ratio,
        },
        degree,
    )?))
}

fn resolve_ka_gate(
    moments: Option<&CouncilMomentVector>,
    config: &KaGateConfig,
    controller: Option<&dyn KaController>,
) -> anyhow::Result<KaGateEvaluation> {
    if let Some(moments) = moments {
        return Ok(evaluate_ka_gate(moments, config, controller)?);
    }
    anyhow::ensure!(
        !config.required,
        "required NC-KA evaluation cannot run without low-level metrics"
    );
    let weights = normalise_weights(config.static_fallback_weights)?;
    Ok(KaGateEvaluation {
        source: if config.enabled {
            GateSource::StaticFallback
        } else {
            GateSource::Disabled
        },
        output: KaGateOutput {
            branch_weights: weights,
            evaluator_weights: weights,
        },
        numerical_rank: 0.0,
        effective_rank: 0.0,
        fallback_used: config.enabled,
        reason: Some(
            if config.enabled {
                "low_level_metrics_unavailable"
            } else {
                "ncka_disabled"
            }
            .to_string(),
        ),
    })
}

fn normalise_weights(mut weights: [f32; 3]) -> anyhow::Result<[f32; 3]> {
    anyhow::ensure!(
        weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0),
        "NC-KA fallback weights must be finite and non-negative"
    );
    let sum = weights.iter().sum::<f32>();
    anyhow::ensure!(
        sum.is_finite() && sum > f32::EPSILON,
        "NC-KA fallback weights must have positive sum"
    );
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok(weights)
}

fn candidates_max_js(candidates: &[CouncilCandidate; 3]) -> f64 {
    candidates
        .iter()
        .filter_map(|candidate| candidate.low_level_metrics.as_ref())
        .map(|metrics| metrics.max_pairwise_js)
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::ffi::{TrialityBranchConfig, TrialityLayerConfig};

    fn base_config() -> TrialityContextConfig {
        TrialityContextConfig {
            schema_version: 2,
            execution: TrialityExecution::AttentionLogitConsensus,
            layers: vec![TrialityLayerConfig {
                branches: [
                    branch(TrialityView::SpinorMinusProxy),
                    branch(TrialityView::Vector),
                    branch(TrialityView::SpinorPlusProxy),
                ],
                active_branch_mask: 0b111,
            }],
            required: false,
            trace_enabled: true,
            js_fallback_threshold: 0.1,
            allow_identity_view_fallback: false,
        }
    }

    fn branch(view: TrialityView) -> TrialityBranchConfig {
        TrialityBranchConfig {
            view,
            weight: 1.0 / 3.0,
            bias: 0.0,
            scale: 1.0,
            temperature: 1.0,
            expected_error: 0.0,
            bits_per_channel: 16.0,
        }
    }

    #[test]
    fn view_context_selects_by_identity_not_array_position() {
        let config = context_config_for_view(&base_config(), CouncilView::Vector).unwrap();
        assert_eq!(config.execution, TrialityExecution::SingleView);
        assert_eq!(config.layers[0].active_branch_mask, 0b010);
        assert_eq!(config.layers[0].branches[1].weight, 1.0);
        assert_eq!(config.layers[0].branches[0].weight, 0.0);
        assert_eq!(config.layers[0].branches[2].weight, 0.0);
    }

    #[test]
    fn missing_metrics_take_optional_static_fallback_and_fail_when_required() {
        let optional = KaGateConfig {
            enabled: true,
            static_fallback_weights: [2.0, 1.0, 1.0],
            ..KaGateConfig::default()
        };
        let evaluation = resolve_ka_gate(None, &optional, None).unwrap();
        assert_eq!(evaluation.source, GateSource::StaticFallback);
        assert_eq!(evaluation.output.evaluator_weights, [0.5, 0.25, 0.25]);
        let required = KaGateConfig {
            required: true,
            ..optional
        };
        assert!(resolve_ka_gate(None, &required, None).is_err());
    }
}
