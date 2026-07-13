use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::{Args, ValueEnum};
use hypura::compute::ffi::{SamplingParams, TrialityExecution};
use hypura::compute::inference::{
    self, InferenceConfig, LlamaTurboquantCliBridge, TrialityRuntimePolicy,
};
use hypura::council::{
    AhaThresholds, AnswerCouncilConfig, CouncilArtifactPolicy, CouncilInputKind,
    CouncilRequestRecord, CouncilRuntimeConfig, CouncilStore, CouncilStoreConfig,
    CouncilUrtDescriptor, EmbeddedKaController, KaController, KaGateConfig, NoSafetyPenalty,
    prepare_embedded_ka_controller,
};
use hypura::model::file_identity::GuardedModelFile;
use hypura::model::gguf::GgufFile;
use hypura::model::turboquant_sidecar::{
    GgufTurboQuantConfig, GgufUrtConfig, RotationPolicy, TurboQuantMode,
};
use hypura::scheduler::placement::conservative_council_headroom;
use hypura::scheduler::types::{
    CouncilExecutionMode, CouncilMemoryAdmission, CouncilParallelism, HostPinnedPolicy,
    ResidencyPolicyConfig, ResidencyProfile,
};
use hypura::urt::{
    RepresentationId, RepresentationKind, UrtAssessment, UrtRegistry, UrtRegistryConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TrialityExecutionArg {
    #[value(name = "single-view")]
    SingleView,
    #[value(name = "best-per-layer")]
    BestPerLayer,
    #[value(name = "attention-logit-consensus")]
    AttentionLogitConsensus,
    #[value(name = "residual-parity")]
    ResidualParity,
}

impl From<TrialityExecutionArg> for TrialityExecution {
    fn from(value: TrialityExecutionArg) -> Self {
        match value {
            TrialityExecutionArg::SingleView => Self::SingleView,
            TrialityExecutionArg::BestPerLayer => Self::BestPerLayer,
            TrialityExecutionArg::AttentionLogitConsensus => Self::AttentionLogitConsensus,
            TrialityExecutionArg::ResidualParity => Self::ResidualParity,
        }
    }
}

impl From<TrialityExecution> for TrialityExecutionArg {
    fn from(value: TrialityExecution) -> Self {
        match value {
            TrialityExecution::SingleView => Self::SingleView,
            TrialityExecution::BestPerLayer => Self::BestPerLayer,
            TrialityExecution::AttentionLogitConsensus => Self::AttentionLogitConsensus,
            TrialityExecution::ResidualParity => Self::ResidualParity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrialityWeights(pub [f32; 3]);

impl FromStr for TrialityWeights {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_triality_weights(value).map(Self)
    }
}

pub fn parse_triality_weights(value: &str) -> Result<[f32; 3], String> {
    let parsed = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            if part.is_empty() {
                return Err("Triality weights must contain exactly three numbers".to_string());
            }
            part.parse::<f32>()
                .map_err(|_| format!("invalid Triality weight `{part}`"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights: [f32; 3] = parsed
        .try_into()
        .map_err(|_| "Triality weights must contain exactly three numbers".to_string())?;
    if !weights
        .iter()
        .all(|weight| weight.is_finite() && *weight >= 0.0)
    {
        return Err("Triality weights must be finite and non-negative".to_string());
    }
    let sum = weights.iter().copied().sum::<f32>();
    if (sum - 1.0).abs() > 1.0e-6 {
        return Err(format!(
            "Triality weights must sum to one within 1e-6; received {sum}"
        ));
    }
    Ok(weights)
}

#[derive(Debug, Clone, Default, Args)]
pub struct TrialityCliOverrides {
    #[arg(long = "tq-triality-execution", value_enum)]
    pub execution: Option<TrialityExecutionArg>,
    #[arg(long = "tq-triality-weights", value_name = "W0,W1,W2")]
    pub weights: Option<TrialityWeights>,
    #[arg(long = "tq-triality-trace")]
    pub trace_enabled: bool,
    #[arg(long = "tq-ncka-required")]
    pub ncka_required: bool,
    #[arg(long = "tq-urt")]
    pub urt_enabled: bool,
    #[arg(long = "tq-developer-override")]
    pub developer_override: bool,
    #[arg(long = "tq-allow-identity-view-fallback")]
    pub allow_identity_view_fallback: bool,
}

impl TrialityCliOverrides {
    pub fn apply_to_bridge(
        &self,
        mut bridge: LlamaTurboquantCliBridge,
    ) -> LlamaTurboquantCliBridge {
        bridge.execution = self.execution.map(Into::into);
        bridge.weights = self.weights.map(|weights| weights.0);
        bridge.trace_enabled = self.trace_enabled;
        bridge.ncka_required = self.ncka_required;
        bridge.urt_enabled = self.urt_enabled;
        bridge.tq_developer_override = self.developer_override;
        bridge.tq_allow_identity_view_fallback = self.allow_identity_view_fallback;
        bridge
    }
}

#[derive(Debug, Args)]
pub struct CouncilCommandOptions {
    pub model: String,
    #[arg(long)]
    pub prompt: String,
    #[arg(long, default_value = "256")]
    pub max_tokens: u32,
    #[arg(long, value_enum, default_value_t = CouncilParallelism::Sequential)]
    pub parallelism: CouncilParallelism,
    #[arg(long)]
    pub cross_score: bool,
    #[arg(long)]
    pub aha: bool,
    #[arg(long, default_value = "artifacts/triality_council")]
    pub output_dir: PathBuf,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long, default_value = "4096")]
    pub context: u32,
    #[arg(long, value_enum, default_value_t = TurboQuantMode::ResearchKvSplit)]
    pub turboquant_mode: TurboQuantMode,
    #[arg(long)]
    pub turboquant_config: Option<String>,
    #[arg(long, value_enum, default_value_t = RotationPolicy::TrialityVector)]
    pub rotation_policy: RotationPolicy,
    #[arg(long, default_value = "0")]
    pub rotation_seed: u32,
    #[arg(long, value_enum, default_value_t = ResidencyProfile::FourTier)]
    pub residency_profile: ResidencyProfile,
    #[arg(long, value_enum, default_value_t = HostPinnedPolicy::Auto)]
    pub host_pinned: HostPinnedPolicy,
    #[arg(long)]
    pub tq_allow_exact_fallback: bool,
    #[command(flatten)]
    pub triality: TrialityCliOverrides,
}

pub fn run(options: CouncilCommandOptions) -> anyhow::Result<()> {
    anyhow::ensure!(options.cross_score, "Council CLI requires --cross-score");
    anyhow::ensure!(
        !options.prompt.trim().is_empty(),
        "Council prompt must not be empty"
    );
    anyhow::ensure!(
        options.max_tokens > 0,
        "max_tokens must be greater than zero"
    );
    anyhow::ensure!(
        options.max_tokens <= options.context,
        "max_tokens must not exceed context"
    );

    let requested_path = Path::new(&options.model);
    anyhow::ensure!(
        requested_path.exists(),
        "Model file not found: {}",
        requested_path.display()
    );
    let model_guard = GuardedModelFile::acquire(requested_path).map_err(|error| {
        anyhow::anyhow!(
            "Model file identity could not be guarded for {}: {error}",
            requested_path.display()
        )
    })?;
    let path = model_guard.canonical_path();
    let bridge = options.triality.apply_to_bridge(LlamaTurboquantCliBridge {
        rotation_policy: options.rotation_policy,
        llama_rotation_seed: options.rotation_seed,
        ..LlamaTurboquantCliBridge::default()
    });
    let runtime = inference::resolve_runtime_setup(
        path,
        options.context,
        options.turboquant_mode,
        options.turboquant_config.as_deref().map(Path::new),
        bridge,
        ResidencyPolicyConfig::new(options.residency_profile, options.host_pinned),
        options.tq_allow_exact_fallback,
    )?;
    let triality = runtime
        .triality
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Council requires embedded schema-v2 Triality metadata"))?;
    let dry_memory_admission =
        council_memory_admission(&runtime, options.context, options.parallelism)?;
    print_dry_run(&runtime, &triality, &dry_memory_admission, &options);
    if let Some(refusal) = dry_memory_admission.refusal.as_ref() {
        anyhow::bail!(
            "Council memory admission refused: {}",
            serde_json::to_string(refusal)?
        );
    }
    if options.dry_run {
        model_guard.verify_unchanged()?;
        return Ok(());
    }

    let mut config = InferenceConfig {
        n_ctx: options.context,
        triality: Some(triality.clone()),
        ..InferenceConfig::default()
    };
    config.sampling = SamplingParams {
        max_tokens: options.max_tokens,
        seed: 42,
        ..SamplingParams::default()
    };
    let loaded = inference::load_model(
        path,
        &config,
        runtime.n_gpu_layers,
        &runtime.plan,
        &runtime.gguf,
        &runtime.turboquant,
    )?;
    model_guard.verify_unchanged()?;
    let capabilities = loaded.model.triality_capabilities()?;
    let memory_admission =
        loaded.admit_council_memory(CouncilExecutionMode::Answer, options.parallelism);
    if let Some(refusal) = memory_admission.refusal.as_ref() {
        anyhow::bail!(
            "Council memory admission refused after model load: {}",
            serde_json::to_string(refusal)?
        );
    }
    let memory_ratio = loaded
        .council_memory_peak_utilization_ratio(&memory_admission)
        .map_err(|refusal| {
            anyhow::anyhow!(
                "Council memory utilization is unavailable: {}",
                refusal.reason
            )
        })?;
    let memory_budget = memory_admission.budget.clone();
    print_runtime_capabilities(&capabilities);
    let metadata = runtime
        .turboquant
        .gguf_metadata
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Council requires typed schema-v2 metadata"))?;
    anyhow::ensure!(
        !triality.ncka_required || triality.ncka.is_some(),
        "required typed NC-KA policy is unavailable"
    );
    let mut ka_gate = triality
        .ncka
        .as_ref()
        .map(|ncka| KaGateConfig {
            enabled: ncka.enabled && capabilities.ncka_available,
            required: triality.ncka_required,
            controller_s3_equivariant: ncka.s3_equivariant,
            static_fallback_weights: ncka.fallback_weights,
            ..KaGateConfig::default()
        })
        .unwrap_or_default();
    ka_gate.required = triality.ncka_required;
    let ka_controller = if ka_gate.enabled {
        prepare_cli_embedded_ka_controller(path, &runtime.gguf, metadata, &mut ka_gate)?
    } else {
        None
    };
    let gguf_sha256 = if triality.urt_enabled {
        Some(model_guard.initial_snapshot().sha256().to_string())
    } else {
        None
    };
    let urt = if triality.urt_enabled {
        let urt = triality
            .urt
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("required typed URT policy is unavailable"))?;
        let gguf_sha256 = gguf_sha256
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("GGUF content hash is unavailable"))?;
        Some(council_urt_descriptor(metadata, urt, gguf_sha256)?)
    } else {
        None
    };
    let moment_degree = triality
        .urt
        .as_ref()
        .filter(|urt| urt.enabled)
        .map(|urt| urt.moment_degree)
        .unwrap_or(3);
    let council = loaded.council_runtime(CouncilRuntimeConfig {
        inference: config.clone(),
        triality: triality.context,
        memory_budget: memory_budget.clone(),
        answer: AnswerCouncilConfig::default(),
        ka_gate,
        moment_degree,
        memory_ratio,
        attention_consensus_requested: false,
        attention_consensus_required: false,
        aha_enabled: options.aha,
        aha_thresholds: AhaThresholds::default(),
        urt,
    })?;
    let request_id = format!("tc-{}", uuid::Uuid::new_v4().simple());
    let result = council.execute(
        &request_id,
        &options.prompt,
        &config.sampling,
        &[],
        &NoSafetyPenalty,
        ka_controller
            .as_ref()
            .map(|controller| controller as &dyn KaController),
        None,
    )?;

    let data_root = data_root_for_output_dir(&options.output_dir)?;
    let urt_assessment = record_cli_urt_observation(&data_root, result.urt_observation.as_ref())?;
    let mut store_config = CouncilStoreConfig::for_data_root(data_root);
    if options.triality.trace_enabled {
        store_config.policy = CouncilArtifactPolicy::trace_content();
    }
    let store = CouncilStore::open(store_config)?;
    let persisted = store.persist(
        &CouncilRequestRecord {
            request_id: request_id.clone(),
            created_at: chrono::Utc::now(),
            model: Some(loaded.model_name.clone()),
            input_kind: CouncilInputKind::Prompt,
            message_count: None,
            max_tokens: Some(options.max_tokens),
            temperature: Some(config.sampling.temperature),
            seed: Some(config.sampling.seed),
            parallelism: memory_budget.parallelism,
            attention_consensus: false,
            cross_score: true,
            synthesis: false,
            aha: options.aha,
            trace: options.triality.trace_enabled,
        },
        &result.answer,
    )?;
    model_guard.verify_unchanged()?;

    println!();
    println!("Selected view: {:?}", result.answer.selected_view);
    println!("Winner margin: {:.6}", result.answer.winner_margin);
    println!("Agreement: {:.6}", result.answer.agreement);
    print_urt_assessment(urt_assessment.as_ref());
    println!("Result ID: {request_id}");
    if let Some(record) = persisted {
        println!("Artifacts: {}", record.directory.display());
    }
    println!();
    println!("{}", result.answer.selected_text);
    Ok(())
}

fn prepare_cli_embedded_ka_controller(
    model_path: &Path,
    gguf: &GgufFile,
    config: &GgufTurboQuantConfig,
    gate_config: &mut KaGateConfig,
) -> anyhow::Result<Option<EmbeddedKaController>> {
    prepare_embedded_ka_controller(model_path, gguf, config, gate_config)
        .map_err(|error| anyhow::anyhow!("CLI NC-KA controller is unavailable: {error}"))
}

fn council_urt_descriptor(
    metadata: &GgufTurboQuantConfig,
    urt: &GgufUrtConfig,
    model_sha256: &str,
) -> anyhow::Result<CouncilUrtDescriptor> {
    anyhow::ensure!(
        model_sha256.len() == 64 && model_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "GGUF content hash is invalid"
    );
    anyhow::ensure!(
        urt.supported_representations
            .iter()
            .any(|value| value == RepresentationKind::HypuraNative.as_str()),
        "embedded URT policy does not support the Hypura native representation"
    );
    anyhow::ensure!(
        !urt.operator_word_sha256.trim().is_empty()
            && urt.consistency_tolerance.is_finite()
            && urt.consistency_tolerance > 0.0,
        "embedded URT descriptor is incomplete"
    );
    let manifest: serde_json::Value = serde_json::from_str(&urt.operator_word_manifest)?;
    let operator_word = manifest
        .get("words")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("embedded URT operator-word manifest has no words"))?
        .iter()
        .map(|value| value.as_str().map(str::trim))
        .collect::<Option<Vec<_>>>()
        .filter(|words| !words.is_empty() && words.iter().all(|word| !word.is_empty()))
        .ok_or_else(|| anyhow::anyhow!("embedded URT operator words are invalid"))?
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

fn council_memory_admission(
    runtime: &inference::RuntimeSetup,
    context: u32,
    parallelism: CouncilParallelism,
) -> anyhow::Result<CouncilMemoryAdmission> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    runtime
        .admit_council_memory(
            CouncilExecutionMode::Answer,
            parallelism,
            context,
            system.available_memory().max(system.free_memory()),
            conservative_council_headroom(&runtime.hardware),
        )
        .map_err(|refusal| {
            anyhow::anyhow!(
                "Council memory request is invalid: {}",
                serde_json::to_string(&refusal).unwrap_or_else(|_| refusal.reason.clone())
            )
        })
}

fn record_cli_urt_observation(
    data_root: &Path,
    observation: Option<&hypura::urt::UrtObservation>,
) -> anyhow::Result<Option<UrtAssessment>> {
    let Some(observation) = observation else {
        return Ok(None);
    };
    let mut registry = UrtRegistry::open(UrtRegistryConfig::persistent(data_root))?;
    Ok(Some(registry.record_and_assess(observation.clone())?))
}

fn print_urt_assessment(assessment: Option<&UrtAssessment>) {
    let Some(assessment) = assessment else {
        println!("URT status: disabled");
        return;
    };
    if let Some(report) = assessment.report.as_ref() {
        println!(
            "URT status: assessed comparison_count={} consistent={} max_absolute_error={}",
            report.comparisons.len(),
            report.consistent,
            report.max_absolute_error
        );
    } else {
        println!("URT status: unassessed comparison_count=0");
    }
}

fn print_runtime_capabilities(capabilities: &hypura::compute::ffi::TrialityModelCapabilities) {
    println!("Runtime Triality capability probe:");
    println!("  runtime-probed: true");
    println!("  metadata-present: {}", capabilities.metadata_present);
    println!("  schema-version: {}", capabilities.schema_version);
    println!("  profile-id: {}", capabilities.profile_id);
    println!("  layers: {}", capabilities.n_layers);
    println!("  three-view-bundle: {}", capabilities.three_view_bundle);
    println!(
        "  supported-execution-mask: {:#x}",
        capabilities.supported_execution_mask
    );
    println!(
        "  selected-execution: {:?}",
        capabilities.selected_execution
    );
    println!("  ncka-available: {}", capabilities.ncka_available);
    println!(
        "  ncka-static-fallback-selected: {}",
        capabilities.ncka_static_fallback_selected
    );
    println!("  urt-available: {}", capabilities.urt_available);
}

fn declared_capability_lines(
    runtime: &inference::RuntimeSetup,
    triality: &TrialityRuntimePolicy,
) -> Vec<String> {
    let metadata = runtime.turboquant.gguf_metadata.as_ref();
    let consensus = metadata.and_then(|value| value.consensus.as_ref());
    let ncka = triality.ncka.as_ref();
    let urt = triality.urt.as_ref();
    vec![
        "runtime-probed: false".to_string(),
        format!(
            "declared-schema-version: {}",
            metadata.map_or(0, |value| value.schema_version)
        ),
        format!(
            "declared-profile-id: {}",
            consensus.map_or("unavailable", |value| value.profile_id.as_str())
        ),
        format!(
            "declared-execution: {}",
            consensus.map_or("unavailable", |value| value.execution.as_str())
        ),
        format!(
            "execution-required: {}",
            consensus.is_some_and(|value| value.required)
        ),
        format!(
            "declared-layer-count: {}",
            consensus.map_or(0, |value| value.branches_by_layer.len())
        ),
        format!("ncka-declared: {}", ncka.is_some_and(|value| value.enabled)),
        format!("ncka-required: {}", triality.ncka_required),
        format!(
            "ncka-static-fallback-selected: {}",
            ncka.is_some_and(|value| value.static_fallback_selected)
        ),
        format!("urt-declared: {}", urt.is_some_and(|value| value.enabled)),
        format!("urt-required: {}", triality.urt_enabled),
        format!(
            "residency-profile: {}",
            runtime.plan.residency_policy.residency_profile.label()
        ),
        format!(
            "host-pinned-policy: {:?}",
            runtime.plan.residency_policy.host_pinned_policy
        ),
    ]
}

fn print_dry_run(
    runtime: &inference::RuntimeSetup,
    triality: &TrialityRuntimePolicy,
    admission: &CouncilMemoryAdmission,
    options: &CouncilCommandOptions,
) {
    let memory = &admission.budget;
    println!("Hypura Triality Council plan");
    println!("  GGUF schema: {}", runtime.turboquant.schema_label());
    println!("  Declared Triality capabilities:");
    for line in declared_capability_lines(runtime, triality) {
        println!("    {line}");
    }
    println!("  Council execution: answer");
    println!("  Context count: {}", memory.context_count);
    println!("  Estimated KV memory: {} bytes", memory.estimated_kv_bytes);
    println!(
        "  Residency profile: {}",
        runtime.plan.residency_policy.residency_profile.label()
    );
    println!(
        "  NC-KA controller: {}",
        if triality.ncka_required {
            "required"
        } else {
            "optional"
        }
    );
    println!(
        "  URT: {}",
        if triality.urt_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "  Aha mode: {}",
        if options.aha {
            "objective-evidence-only"
        } else {
            "disabled"
        }
    );
    println!("  Memory admission: {}", memory.reason);
    println!(
        "  Memory projection: gpu={} host_pageable={} host_pinned={} unified={} controller={}",
        admission.projection.gpu_bytes,
        admission.projection.host_pageable_bytes,
        admission.projection.host_pinned_bytes,
        admission.projection.unified_bytes,
        admission.projection.controller_bytes,
    );
    println!(
        "  Memory refusal: {}",
        admission
            .refusal
            .as_ref()
            .and_then(|refusal| serde_json::to_string(refusal).ok())
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "  Parallel refusal: {}",
        admission
            .parallel_refusal
            .as_ref()
            .and_then(|refusal| serde_json::to_string(refusal).ok())
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "  Output directory: {}",
        absolute_path(&options.output_dir).display()
    );
}

fn data_root_for_output_dir(output_dir: &Path) -> anyhow::Result<PathBuf> {
    let absolute = absolute_path(output_dir);
    anyhow::ensure!(
        absolute.file_name().and_then(|value| value.to_str()) == Some("triality_council"),
        "output directory must end with artifacts/triality_council"
    );
    let artifacts = absolute
        .parent()
        .ok_or_else(|| anyhow::anyhow!("output directory has no artifacts parent"))?;
    anyhow::ensure!(
        artifacts.file_name().and_then(|value| value.to_str()) == Some("artifacts"),
        "output directory must end with artifacts/triality_council"
    );
    artifacts
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| anyhow::anyhow!("output directory has no data root"))
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Write;

    use hypura::model::gguf::{GgmlType, TensorInfo};
    use hypura::model::turboquant_sidecar::{GgufNcKaConfig, GgufTrialityConsensusConfig};
    use serde_json::{Map, Value, json};
    use sha2::{Digest, Sha256};

    struct CliNckaFixture {
        file: tempfile::NamedTempFile,
        gguf: GgufFile,
        config: GgufTurboQuantConfig,
    }

    fn test_sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn test_json_sha256(value: &Value) -> String {
        test_sha256(&serde_json::to_vec(value).unwrap())
    }

    fn cli_ncka_fixture() -> CliNckaFixture {
        const COORDINATES: usize = 24;
        const OUTER: usize = 2;
        const KNOTS: usize = 3;
        let axis = [0.0_f32, 0.5, 1.0];
        let inner_knots = axis.repeat(3 * OUTER * COORDINATES);
        let mut inner_values = Vec::with_capacity(inner_knots.len());
        for branch in 0..3 {
            for outer in 0..OUTER {
                for coordinate in 0..COORDINATES {
                    for knot in axis {
                        inner_values.push(if outer == 0 && coordinate == branch {
                            knot
                        } else {
                            0.0
                        });
                    }
                }
            }
        }
        let outer_knots = [0.0_f32, 0.08, 0.16].repeat(3 * OUTER);
        let mut outer_values = Vec::with_capacity(3 * OUTER * KNOTS);
        for _branch in 0..3 {
            outer_values.extend([0.0_f32, 0.02, 0.20]);
            outer_values.extend([0.0_f32; KNOTS]);
        }
        let tensors = BTreeMap::from([
            (
                "coordinate_min",
                (vec![COORDINATES as u64], vec![0.0; COORDINATES]),
            ),
            (
                "coordinate_max",
                (vec![COORDINATES as u64], vec![1.0; COORDINATES]),
            ),
            (
                "inner_knots",
                (
                    vec![KNOTS as u64, COORDINATES as u64, OUTER as u64, 3],
                    inner_knots,
                ),
            ),
            (
                "inner_values",
                (
                    vec![KNOTS as u64, COORDINATES as u64, OUTER as u64, 3],
                    inner_values,
                ),
            ),
            (
                "outer_knots",
                (vec![KNOTS as u64, OUTER as u64, 3], outer_knots),
            ),
            (
                "outer_values",
                (vec![KNOTS as u64, OUTER as u64, 3], outer_values),
            ),
            ("fallback_weights", (vec![3], vec![0.2, 0.3, 0.5])),
        ]);

        let mut bytes = vec![0_u8; 32];
        let mut tensor_infos = Vec::new();
        let mut manifest = Map::new();
        let mut relative_offset = 0_u64;
        for (field, (shape, values)) in tensors {
            let name = format!("turboquant.profile.v2.ncka.{field}");
            let tensor_bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            manifest.insert(
                name.clone(),
                json!({
                    "dtype": "f32",
                    "shape": shape,
                    "sha256": test_sha256(&tensor_bytes),
                }),
            );
            tensor_infos.push(TensorInfo {
                name,
                dimensions: shape,
                dtype: GgmlType::F32,
                offset: relative_offset,
                size_bytes: tensor_bytes.len() as u64,
                layer_index: None,
            });
            relative_offset += tensor_bytes.len() as u64;
            bytes.extend_from_slice(&tensor_bytes);
        }

        let manifest_value = Value::Object(manifest);
        let controller_sha256 = test_json_sha256(&manifest_value);
        let payload_json = serde_json::to_string(&json!({
            "tensor_manifest": manifest_value,
        }))
        .unwrap();
        let normalisation_sha256 = test_json_sha256(&json!({
            "coordinate_names": hypura::council::NCKA_COORDINATE_NAMES,
            "range": [0.0, 1.0],
            "clamp": true,
        }));
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        CliNckaFixture {
            file,
            gguf: GgufFile {
                version: 3,
                metadata: BTreeMap::new(),
                tensors: tensor_infos,
                data_offset: 32,
            },
            config: GgufTurboQuantConfig {
                enabled: true,
                schema_version: 2,
                mode: TurboQuantMode::TrialityConsensus,
                public_mode_label: "triality_consensus".into(),
                runtime_mode: "key_only_block_so8_triality_consensus".into(),
                rotation_policy: Some(RotationPolicy::BlockSo8Learned),
                triality_view: Some("vector".into()),
                triality_mode: Some("triality_proxy".into()),
                triality_mix: Some(1.0),
                paper_fidelity: false,
                k_bits: 3.0,
                v_bits: 16.0,
                payload_format: Some("json-inline-v2".into()),
                payload_bytes: payload_json.len() as u64,
                payload_json: Some(payload_json),
                rotation_seed: 7,
                artifact_path: None,
                head_dim: 8,
                num_layers: 1,
                num_kv_heads: 1,
                layers: Vec::new(),
                weight: None,
                consensus: Some(GgufTrialityConsensusConfig {
                    profile_id: "v2".into(),
                    execution: "attention_logit_consensus".into(),
                    branches_by_layer: Vec::new(),
                    js_fallback_threshold: 0.2,
                    required: true,
                    override_allowed: false,
                }),
                ncka: Some(GgufNcKaConfig {
                    enabled: true,
                    required: false,
                    schema_version: 1,
                    controller_type: "finite_moment_ka_v1".into(),
                    coordinate_names: hypura::council::NCKA_COORDINATE_NAMES
                        .map(str::to_string)
                        .to_vec(),
                    outer_count: OUTER as u32,
                    knot_count: KNOTS as u32,
                    s3_equivariant: true,
                    controller_sha256,
                    normalisation_sha256,
                    static_fallback_selected: false,
                    fallback_weights: [0.2, 0.3, 0.5],
                }),
                urt: None,
            },
        }
    }

    #[test]
    fn triality_weights_require_three_finite_nonnegative_values_summing_to_one() {
        assert_eq!(
            parse_triality_weights("0.2,0.3,0.5").unwrap(),
            [0.2, 0.3, 0.5]
        );
        assert!(parse_triality_weights("0.5,0.5").is_err());
        assert!(parse_triality_weights("-0.1,0.6,0.5").is_err());
        assert!(parse_triality_weights("NaN,0.5,0.5").is_err());
        assert!(parse_triality_weights("0.2,0.2,0.2").is_err());
    }

    #[test]
    fn guarded_model_identity_supplies_the_urt_content_hash() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"abc").unwrap();
        let guard = GuardedModelFile::acquire(&path).unwrap();
        assert_eq!(
            guard.initial_snapshot().sha256(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        guard.verify_unchanged().unwrap();
    }

    #[test]
    fn cli_ncka_happy_path_loads_and_evaluates_the_embedded_controller() {
        let fixture = cli_ncka_fixture();
        let mut gate = KaGateConfig {
            enabled: true,
            controller_s3_equivariant: true,
            static_fallback_weights: [0.2, 0.3, 0.5],
            ..KaGateConfig::default()
        };
        let controller = prepare_cli_embedded_ka_controller(
            fixture.file.path(),
            &fixture.gguf,
            &fixture.config,
            &mut gate,
        )
        .unwrap()
        .unwrap();
        let weights = controller.evaluate(&[0.25; 24]).unwrap();
        assert!((weights.iter().sum::<f32>() - 1.0).abs() <= 1.0e-5);
    }

    #[test]
    fn cli_ncka_optional_failure_uses_only_the_declared_static_fallback() {
        let fixture = cli_ncka_fixture();
        let mut config = fixture.config.clone();
        let ncka = config.ncka.as_mut().unwrap();
        ncka.schema_version = u32::MAX;
        ncka.fallback_weights = [0.1, 0.2, 0.7];
        let mut gate = KaGateConfig {
            enabled: true,
            static_fallback_weights: [1.0, 0.0, 0.0],
            ..KaGateConfig::default()
        };
        let controller = prepare_cli_embedded_ka_controller(
            fixture.file.path(),
            &fixture.gguf,
            &config,
            &mut gate,
        )
        .unwrap();
        assert!(controller.is_none());
        assert_eq!(gate.static_fallback_weights, [0.1, 0.2, 0.7]);
    }

    #[test]
    fn cli_ncka_required_failure_remains_fatal() {
        let fixture = cli_ncka_fixture();
        let mut config = fixture.config.clone();
        let ncka = config.ncka.as_mut().unwrap();
        ncka.required = true;
        ncka.schema_version = u32::MAX;
        let mut gate = KaGateConfig {
            enabled: true,
            required: true,
            static_fallback_weights: [0.2, 0.3, 0.5],
            ..KaGateConfig::default()
        };
        let error = prepare_cli_embedded_ka_controller(
            fixture.file.path(),
            &fixture.gguf,
            &config,
            &mut gate,
        )
        .unwrap_err();
        assert!(error.to_string().contains("required embedded NC-KA"));
        assert_eq!(gate.static_fallback_weights, [0.2, 0.3, 0.5]);
    }

    #[test]
    fn cli_ncka_invalid_fallback_is_fatal_and_does_not_mutate_the_gate() {
        let fixture = cli_ncka_fixture();
        let mut config = fixture.config.clone();
        let ncka = config.ncka.as_mut().unwrap();
        ncka.schema_version = u32::MAX;
        ncka.fallback_weights = [0.6, 0.6, -0.2];
        let mut gate = KaGateConfig {
            enabled: true,
            static_fallback_weights: [0.2, 0.3, 0.5],
            ..KaGateConfig::default()
        };
        let error = prepare_cli_embedded_ka_controller(
            fixture.file.path(),
            &fixture.gguf,
            &config,
            &mut gate,
        )
        .unwrap_err();
        assert!(error.to_string().contains("no valid fallback"));
        assert_eq!(gate.static_fallback_weights, [0.2, 0.3, 0.5]);
    }
}
