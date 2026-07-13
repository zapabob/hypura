use std::collections::BTreeSet;
use std::fmt;
use std::path::{Component, Path, PathBuf};

use anyhow::Context;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::compute::ffi::{
    TrialityBranchConfig, TrialityContextConfig, TrialityExecution, TrialityLayerConfig,
    TrialityView,
};
use crate::model::gguf::{GgmlType, GgufFile, GgufValue};
use crate::model::metadata::ModelMetadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum TurboQuantMode {
    Exact,
    PaperKeyOnly,
    PaperFullKv,
    ResearchKvSplit,
    TrialityConsensus,
    TrialityResidualParity,
}

impl TurboQuantMode {
    pub fn requires_sidecar(self) -> bool {
        !matches!(self, Self::Exact)
    }

    pub fn expected_schema_kind(self) -> Option<TurboQuantSchemaKind> {
        match self {
            Self::Exact => None,
            Self::PaperKeyOnly | Self::PaperFullKv => Some(TurboQuantSchemaKind::Paper),
            Self::ResearchKvSplit | Self::TrialityConsensus | Self::TrialityResidualParity => {
                Some(TurboQuantSchemaKind::Research)
            }
        }
    }

    pub fn config_suffix(self) -> Option<&'static str> {
        match self.expected_schema_kind()? {
            TurboQuantSchemaKind::Paper => Some("turboquant_config.paper.json"),
            TurboQuantSchemaKind::Research => Some("turboquant_config.research.json"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::PaperKeyOnly => "paper-key-only",
            Self::PaperFullKv => "paper-full-kv",
            Self::ResearchKvSplit => "research-kv-split",
            Self::TrialityConsensus => "triality-consensus",
            Self::TrialityResidualParity => "triality-residual-parity",
        }
    }
}

impl fmt::Display for TurboQuantMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TurboQuantSchemaKind {
    Paper,
    Research,
}

impl TurboQuantSchemaKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Paper => "paper",
            Self::Research => "research",
        }
    }
}

impl fmt::Display for TurboQuantSchemaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Rotation policies for TurboQuant KV compression
/// Based on: random_haar, block_so8_static, block_so8_learned (zapabob/Turboquant-CUDA)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum RotationPolicy {
    /// Random Haar orthogonal rotation (paper baseline)
    RandomHaar,
    /// Block-diagonal SO(8) static rotation
    BlockSo8Static,
    /// Block-diagonal SO(8) with learned rotations
    BlockSo8Learned,
    /// Triality vector view (standard basis)
    TrialityVector,
    /// Triality spinor+ view
    TrialitySpinorPlus,
    /// Triality spinor- view
    TrialitySpinorMinus,
    #[value(skip)]
    IdentityDev,
}

impl RotationPolicy {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "random_haar" => Some(Self::RandomHaar),
            "block_so8_static" => Some(Self::BlockSo8Static),
            "block_so8_learned" => Some(Self::BlockSo8Learned),
            "triality_vector" => Some(Self::TrialityVector),
            "triality_spinor_plus" => Some(Self::TrialitySpinorPlus),
            "triality_spinor_minus" => Some(Self::TrialitySpinorMinus),
            "identity_dev" => Some(Self::IdentityDev),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RandomHaar => "random_haar",
            Self::BlockSo8Static => "block_so8_static",
            Self::BlockSo8Learned => "block_so8_learned",
            Self::TrialityVector => "triality_vector",
            Self::TrialitySpinorPlus => "triality_spinor_plus",
            Self::TrialitySpinorMinus => "triality_spinor_minus",
            Self::IdentityDev => "identity_dev",
        }
    }

    pub fn is_triality(&self) -> bool {
        matches!(
            self,
            Self::TrialityVector | Self::TrialitySpinorPlus | Self::TrialitySpinorMinus
        )
    }

    pub fn triality_view(&self) -> Option<&'static str> {
        match self {
            Self::TrialityVector => Some("vector"),
            Self::TrialitySpinorPlus => Some("spinor_plus_proxy"),
            Self::TrialitySpinorMinus => Some("spinor_minus_proxy"),
            Self::IdentityDev => None,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTurboQuantLayerConfig {
    pub total_bits: f32,
    pub runtime_bits_per_channel: f32,
    pub stage1_effective_bits: f32,
    pub qjl_bits: u32,
    pub qjl_dim: u32,
    pub rotation_policy: RotationPolicy,
    pub rotation_seed: u32,
    pub qjl_seed: u32,
    pub triality_mode: String,
    pub triality_view: String,
    pub stage1_allocation_scheme: String,
    pub stage1_bitwidth_payload_dtype: String,
    pub norm_dtype: String,
    pub sign_pack_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationArtifact {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    pub matrix: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizerArtifact {
    pub centroids: Vec<f32>,
    #[serde(default)]
    pub decision_boundaries: Vec<f32>,
    #[serde(default)]
    pub protected_centroids: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualQjlArtifact {
    pub sketch_dim: u32,
    pub scale: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub projections: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperTurboQuantConfig {
    pub schema_kind: TurboQuantSchemaKind,
    pub codec: String,
    pub num_layers: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub key_bits: f32,
    #[serde(default)]
    pub value_bits: Option<f32>,
    #[serde(default)]
    pub mixed_bits: Option<f32>,
    #[serde(default)]
    pub value_exact: bool,
    pub rotation: RotationArtifact,
    pub scalar_quantizer: ScalarQuantizerArtifact,
    #[serde(default)]
    pub residual_qjl: Option<ResidualQjlArtifact>,
    #[serde(default)]
    pub protected_channels: Vec<u32>,
    #[serde(default)]
    pub outlier_channels: Vec<u32>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResearchTurboQuantConfig {
    #[serde(default)]
    pub schema_kind: Option<TurboQuantSchemaKind>,
    #[serde(default)]
    pub codec: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurboQuantSidecarConfig {
    Paper(PaperTurboQuantConfig),
    Research(ResearchTurboQuantConfig),
}

impl TurboQuantSidecarConfig {
    pub fn schema_kind(&self) -> TurboQuantSchemaKind {
        match self {
            Self::Paper(_) => TurboQuantSchemaKind::Paper,
            Self::Research(_) => TurboQuantSchemaKind::Research,
        }
    }

    pub fn paper(&self) -> Option<&PaperTurboQuantConfig> {
        match self {
            Self::Paper(cfg) => Some(cfg),
            Self::Research(_) => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTurboQuantConfig {
    pub mode: TurboQuantMode,
    pub schema_kind: Option<TurboQuantSchemaKind>,
    pub source_path: Option<PathBuf>,
    pub config: Option<TurboQuantSidecarConfig>,
    pub gguf_metadata: Option<GgufTurboQuantConfig>,
}

impl ResolvedTurboQuantConfig {
    pub fn exact() -> Self {
        Self {
            mode: TurboQuantMode::Exact,
            schema_kind: None,
            source_path: None,
            config: None,
            gguf_metadata: None,
        }
    }

    pub fn schema_label(&self) -> &'static str {
        if let Some(schema_kind) = self.schema_kind {
            return schema_kind.as_str();
        }
        match self
            .gguf_metadata
            .as_ref()
            .map(|config| config.schema_version)
        {
            Some(2) => "schema-v2",
            Some(1) => "schema-v1",
            Some(_) => "gguf",
            None => "none",
        }
    }

    pub fn source_label(&self) -> String {
        self.source_path
            .as_ref()
            .map(|p| p.display().to_string())
            .or_else(|| {
                self.gguf_metadata
                    .as_ref()
                    .map(|cfg| cfg.source_label().to_string())
            })
            .unwrap_or_else(|| "none".to_string())
    }

    pub fn paper_config(&self) -> Option<&PaperTurboQuantConfig> {
        self.config.as_ref()?.paper()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTurboQuantConfig {
    pub enabled: bool,
    pub schema_version: u32,
    pub mode: TurboQuantMode,
    pub public_mode_label: String,
    pub runtime_mode: String,
    pub rotation_policy: Option<RotationPolicy>,
    pub triality_view: Option<String>,
    pub triality_mode: Option<String>,
    pub triality_mix: Option<f32>,
    pub paper_fidelity: bool,
    pub k_bits: f32,
    pub v_bits: f32,
    pub payload_format: Option<String>,
    pub payload_bytes: u64,
    pub payload_json: Option<String>,
    pub rotation_seed: u32,
    pub artifact_path: Option<String>,
    pub head_dim: u32,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub layers: Vec<GgufTurboQuantLayerConfig>,
    pub weight: Option<GgufTurboQuantWeightConfig>,
    pub consensus: Option<GgufTrialityConsensusConfig>,
    pub ncka: Option<GgufNcKaConfig>,
    pub urt: Option<GgufUrtConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufTrialityBranchConfig {
    pub view: String,
    pub weight: f32,
    pub bias: f32,
    pub scale: f32,
    pub temperature: f32,
    pub expected_error: f32,
    pub bits_per_channel: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufTrialityConsensusConfig {
    pub profile_id: String,
    pub execution: String,
    pub branches_by_layer: Vec<[GgufTrialityBranchConfig; 3]>,
    pub js_fallback_threshold: f32,
    pub required: bool,
    pub override_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufNcKaConfig {
    pub enabled: bool,
    pub required: bool,
    pub schema_version: u32,
    pub controller_type: String,
    pub coordinate_names: Vec<String>,
    pub outer_count: u32,
    pub knot_count: u32,
    pub s3_equivariant: bool,
    pub controller_sha256: String,
    pub normalisation_sha256: String,
    pub static_fallback_selected: bool,
    pub fallback_weights: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufUrtConfig {
    pub enabled: bool,
    pub schema_version: u32,
    pub abstract_algebra_id: String,
    pub operator_word_manifest: String,
    pub operator_word_sha256: String,
    pub reference_representation: String,
    pub supported_representations: Vec<String>,
    pub consistency_tolerance: f32,
    pub moment_degree: u32,
    pub moment_manifest_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufTurboQuantWeightConfig {
    pub enabled: bool,
    pub codec: Option<String>,
    pub source_ftype: Option<String>,
    pub policy: Option<String>,
    pub protected_roles_json: Option<String>,
    pub protected_layers_json: Option<String>,
    pub modality_scope: Option<String>,
    pub payload_format: Option<String>,
    pub payload_bytes: u64,
    pub payload_json: Option<String>,
    pub payload_schema: Option<String>,
    pub payload_valid: bool,
    pub tensor_plan_entries: u32,
}

impl GgufTurboQuantWeightConfig {
    pub fn runtime_status(&self) -> &'static str {
        if !self.enabled {
            "disabled"
        } else if !self.payload_valid {
            "invalid"
        } else {
            match self.codec.as_deref() {
                Some("tq4_1s") => "contract-only",
                Some(_) => "unsupported-codec",
                None => "missing-codec",
            }
        }
    }

    pub fn runtime_ready(&self) -> bool {
        false
    }
}

impl GgufTurboQuantConfig {
    pub fn source_label(&self) -> &'static str {
        "gguf-embedded"
    }

    pub fn llama_runtime_mode(&self) -> &'static str {
        match self.triality_view.as_deref() {
            Some("vector") => "triality_vector",
            Some("spinor_plus_proxy") => "triality_spinor_plus",
            Some("spinor_minus_proxy") => "triality_spinor_minus",
            _ => match self.mode {
                TurboQuantMode::PaperFullKv => "asym_q8_turbo4",
                TurboQuantMode::PaperKeyOnly
                | TurboQuantMode::ResearchKvSplit
                | TurboQuantMode::TrialityConsensus
                | TurboQuantMode::TrialityResidualParity => "triality_vector",
                TurboQuantMode::Exact => "exact",
            },
        }
    }
}

fn parse_embedded_mode(raw_mode: &str) -> Option<(TurboQuantMode, String, String)> {
    match raw_mode {
        "paper-faithful" => Some((
            TurboQuantMode::PaperKeyOnly,
            "paper-faithful".to_string(),
            "paper-key-only".to_string(),
        )),
        "triality-proxy-so8-pareto" | "triality-so8-pareto" => Some((
            TurboQuantMode::ResearchKvSplit,
            "triality-proxy-so8-pareto".to_string(),
            "research-kv-split".to_string(),
        )),
        "paper-key-only" => Some((
            TurboQuantMode::PaperKeyOnly,
            "paper-faithful".to_string(),
            "paper-key-only".to_string(),
        )),
        "paper-full-kv" => Some((
            TurboQuantMode::PaperFullKv,
            "paper-faithful".to_string(),
            "paper-full-kv".to_string(),
        )),
        "research-kv-split" => Some((
            TurboQuantMode::ResearchKvSplit,
            "triality-proxy-so8-pareto".to_string(),
            "research-kv-split".to_string(),
        )),
        _ => None,
    }
}

pub fn resolve_turboquant_config(
    model_path: &Path,
    metadata: &ModelMetadata,
    gguf: &GgufFile,
    mode: TurboQuantMode,
    explicit_config: Option<&Path>,
    allow_exact_fallback: bool,
) -> anyhow::Result<ResolvedTurboQuantConfig> {
    let gguf_metadata = read_gguf_turboquant_config(gguf, metadata)?;
    if let Some(gguf_metadata) = gguf_metadata {
        if let Some(weight) = gguf_metadata.weight.as_ref() {
            if weight.enabled && !weight.runtime_ready() {
                if allow_exact_fallback {
                    tracing::warn!(
                        "Weight TurboQuant contract present for {} but runtime support is incomplete (codec={}, status={}). Falling back to exact runtime because --tq-allow-exact-fallback was set.",
                        model_path.display(),
                        weight.codec.as_deref().unwrap_or("none"),
                        weight.runtime_status(),
                    );
                    return Ok(ResolvedTurboQuantConfig::exact());
                }
                return Err(anyhow::anyhow!(
                    "Weight TurboQuant contract present for {} but runtime support is incomplete (codec={}, status={}). Pass --tq-allow-exact-fallback for a developer exact-runtime escape hatch.",
                    model_path.display(),
                    weight.codec.as_deref().unwrap_or("none"),
                    weight.runtime_status(),
                ));
            }
        }
        return Ok(ResolvedTurboQuantConfig {
            mode: gguf_metadata.mode,
            schema_kind: None,
            source_path: None,
            config: None,
            gguf_metadata: Some(gguf_metadata),
        });
    }

    if mode == TurboQuantMode::Exact {
        return Ok(ResolvedTurboQuantConfig::exact());
    }

    let config_path = match explicit_config {
        Some(path) => path.to_path_buf(),
        None => match auto_discover_sidecar_path(model_path, mode) {
            Some(path) => path,
            None => {
                if mode == TurboQuantMode::ResearchKvSplit {
                    if allow_exact_fallback {
                        tracing::warn!(
                            "No TurboQuant research sidecar or GGUF metadata found next to {}. Falling back to exact runtime because --tq-allow-exact-fallback was set.",
                            model_path.display()
                        );
                        return Ok(ResolvedTurboQuantConfig::exact());
                    }
                    return Err(anyhow::anyhow!(
                        "No TurboQuant research sidecar or GGUF metadata found next to {}. Fail-closed is active; provide embedded metadata or pass --turboquant-config. For developer exact fallback only, pass --tq-allow-exact-fallback.",
                        model_path.display()
                    ));
                }
                return Err(anyhow::anyhow!(
                    "No TurboQuant sidecar found for mode `{mode}` next to {}. \
                     Pass `--turboquant-config <path>` or place a matching sidecar beside the model.",
                    model_path.display()
                ));
            }
        },
    };

    anyhow::ensure!(
        config_path.exists(),
        "TurboQuant sidecar for mode `{mode}` not found: {}",
        config_path.display()
    );

    let json = std::fs::read_to_string(&config_path).with_context(|| {
        format!(
            "Failed to read TurboQuant sidecar {} for mode `{mode}`",
            config_path.display()
        )
    })?;
    let config = parse_sidecar_config(&config_path, &json, mode).with_context(|| {
        format!(
            "Invalid TurboQuant sidecar {} for mode `{mode}`",
            config_path.display()
        )
    })?;
    let schema_kind = config.schema_kind();

    validate_mode_schema(mode, schema_kind, &config_path)?;
    validate_config_against_metadata(&config, metadata, &config_path)?;

    Ok(ResolvedTurboQuantConfig {
        mode,
        schema_kind: Some(schema_kind),
        source_path: Some(config_path),
        config: Some(config),
        gguf_metadata: None,
    })
}

fn parse_turboquant_mode(raw: &str) -> anyhow::Result<TurboQuantMode> {
    match raw {
        "exact" => Ok(TurboQuantMode::Exact),
        "paper-faithful" | "paper-key-only" => Ok(TurboQuantMode::PaperKeyOnly),
        "paper-full-kv" => Ok(TurboQuantMode::PaperFullKv),
        "triality-proxy-so8-pareto" | "triality-so8-pareto" | "research-kv-split" => {
            Ok(TurboQuantMode::ResearchKvSplit)
        }
        "triality-consensus" => Ok(TurboQuantMode::TrialityConsensus),
        "triality-residual-parity" => Ok(TurboQuantMode::TrialityResidualParity),
        other => Err(anyhow::anyhow!(
            "Unsupported GGUF TurboQuant mode `{other}`"
        )),
    }
}

fn head_dim_from_metadata(metadata: &ModelMetadata) -> u32 {
    if metadata.num_heads == 0 {
        0
    } else {
        metadata.embedding_dim / metadata.num_heads
    }
}

fn parse_gguf_weight_config(gguf: &GgufFile) -> anyhow::Result<Option<GgufTurboQuantWeightConfig>> {
    let enabled = gguf
        .get_bool("hypura.turboquant.weight.enabled")
        .unwrap_or(false);
    let codec = gguf
        .get_string("hypura.turboquant.weight.codec")
        .map(ToOwned::to_owned);
    let source_ftype = gguf
        .get_string("hypura.turboquant.weight.source_ftype")
        .map(ToOwned::to_owned);
    let policy = gguf
        .get_string("hypura.turboquant.weight.policy")
        .map(ToOwned::to_owned);
    let protected_roles_json = gguf
        .get_string("hypura.turboquant.weight.protected_roles")
        .map(ToOwned::to_owned);
    let protected_layers_json = gguf
        .get_string("hypura.turboquant.weight.protected_layers")
        .map(ToOwned::to_owned);
    let modality_scope = gguf
        .get_string("hypura.turboquant.weight.modality_scope")
        .map(ToOwned::to_owned);
    let payload_format = gguf
        .get_string("hypura.turboquant.weight.payload_format")
        .map(ToOwned::to_owned);
    let payload_bytes = gguf
        .get_u64("hypura.turboquant.weight.payload_bytes")
        .unwrap_or(0);
    let payload_json = gguf
        .get_string("hypura.turboquant.weight.payload_json")
        .map(ToOwned::to_owned);

    let any_present = enabled
        || codec.is_some()
        || source_ftype.is_some()
        || policy.is_some()
        || protected_roles_json.is_some()
        || protected_layers_json.is_some()
        || modality_scope.is_some()
        || payload_format.is_some()
        || payload_json.is_some()
        || payload_bytes > 0;
    if !any_present {
        return Ok(None);
    }

    let mut payload_schema = None;
    let mut payload_valid = false;
    let mut tensor_plan_entries = 0;

    if enabled {
        let payload_format_value = payload_format.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF weight metadata is missing `hypura.turboquant.weight.payload_format`"
            )
        })?;
        anyhow::ensure!(
            payload_format_value == "json-inline-v1",
            "Unsupported GGUF weight payload format `{payload_format_value}`; expected `json-inline-v1`"
        );
        let payload_json_value = payload_json.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF weight metadata is missing `hypura.turboquant.weight.payload_json`"
            )
        })?;
        anyhow::ensure!(
            payload_bytes == payload_json_value.len() as u64,
            "GGUF weight payload bytes do not match payload_json length"
        );
        let payload: Value = serde_json::from_str(payload_json_value)
            .context("GGUF weight payload must contain valid JSON")?;
        let obj = payload
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("GGUF weight payload must decode to an object"))?;
        let schema = obj
            .get("schema")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("GGUF weight payload is missing string key `schema`"))?;
        anyhow::ensure!(
            schema == "hypura.turboquant.weight.v1",
            "Unsupported GGUF weight payload schema `{schema}`; expected `hypura.turboquant.weight.v1`"
        );
        payload_schema = Some(schema.to_string());

        let payload_codec = obj
            .get("codec")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("GGUF weight payload is missing string key `codec`"))?;
        anyhow::ensure!(
            codec.as_deref() == Some(payload_codec),
            "GGUF weight payload codec does not match `hypura.turboquant.weight.codec`"
        );
        anyhow::ensure!(
            source_ftype.as_deref() == obj.get("source_ftype").and_then(Value::as_str),
            "GGUF weight payload source_ftype does not match `hypura.turboquant.weight.source_ftype`"
        );
        anyhow::ensure!(
            policy.as_deref() == obj.get("policy").and_then(Value::as_str),
            "GGUF weight payload policy does not match `hypura.turboquant.weight.policy`"
        );
        anyhow::ensure!(
            modality_scope.as_deref() == obj.get("modality_scope").and_then(Value::as_str),
            "GGUF weight payload modality_scope does not match `hypura.turboquant.weight.modality_scope`"
        );
        anyhow::ensure!(
            obj.get("protected_roles")
                .and_then(Value::as_array)
                .is_some(),
            "GGUF weight payload is missing array key `protected_roles`"
        );
        anyhow::ensure!(
            obj.get("protected_layers")
                .and_then(Value::as_array)
                .is_some(),
            "GGUF weight payload is missing array key `protected_layers`"
        );
        let tensor_plan = obj
            .get("tensor_plan")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                anyhow::anyhow!("GGUF weight payload is missing object key `tensor_plan`")
            })?;
        anyhow::ensure!(
            !tensor_plan.is_empty(),
            "GGUF weight payload tensor_plan must not be empty"
        );
        tensor_plan_entries = tensor_plan.len() as u32;
        payload_valid = true;
    }

    Ok(Some(GgufTurboQuantWeightConfig {
        enabled,
        codec,
        source_ftype,
        policy,
        protected_roles_json,
        protected_layers_json,
        modality_scope,
        payload_format,
        payload_bytes,
        payload_json,
        payload_schema,
        payload_valid,
        tensor_plan_entries,
    }))
}

fn require_f32_array(gguf: &GgufFile, key: &str, expected_len: usize) -> anyhow::Result<Vec<f32>> {
    let values = gguf
        .get_f32_array(key)
        .ok_or_else(|| anyhow::anyhow!("GGUF TurboQuant metadata is missing `{key}`"))?;
    anyhow::ensure!(
        values.len() == expected_len,
        "GGUF TurboQuant metadata `{key}` must have length {expected_len}, got {}",
        values.len()
    );
    Ok(values)
}

fn require_u32_array(gguf: &GgufFile, key: &str, expected_len: usize) -> anyhow::Result<Vec<u32>> {
    let values = gguf
        .get_u32_array(key)
        .ok_or_else(|| anyhow::anyhow!("GGUF TurboQuant metadata is missing `{key}`"))?;
    anyhow::ensure!(
        values.len() == expected_len,
        "GGUF TurboQuant metadata `{key}` must have length {expected_len}, got {}",
        values.len()
    );
    Ok(values)
}

fn require_string_array(
    gguf: &GgufFile,
    key: &str,
    expected_len: usize,
) -> anyhow::Result<Vec<String>> {
    let values = gguf
        .get_string_array(key)
        .ok_or_else(|| anyhow::anyhow!("GGUF TurboQuant metadata is missing `{key}`"))?;
    anyhow::ensure!(
        values.len() == expected_len,
        "GGUF TurboQuant metadata `{key}` must have length {expected_len}, got {}",
        values.len()
    );
    Ok(values)
}

fn infer_triality_view_from_legacy_policy(rotation_policy: RotationPolicy) -> Option<String> {
    rotation_policy.triality_view().map(str::to_string)
}

fn parse_legacy_gguf_turboquant_config(
    gguf: &GgufFile,
    metadata: &ModelMetadata,
) -> anyhow::Result<Option<GgufTurboQuantConfig>> {
    let Some(enabled) = gguf.get_bool("hypura.turboquant.enabled") else {
        return Ok(None);
    };
    if !enabled {
        return Ok(None);
    }

    let raw_mode = gguf.get_string("hypura.turboquant.mode").ok_or_else(|| {
        anyhow::anyhow!("Legacy GGUF TurboQuant metadata is missing `hypura.turboquant.mode`")
    })?;
    let mode = parse_turboquant_mode(raw_mode)?;
    let (public_mode_label, default_runtime_mode) = parse_embedded_mode(raw_mode)
        .map(|(_, public, runtime)| (public, runtime))
        .unwrap_or_else(|| (mode.as_str().to_string(), mode.as_str().to_string()));

    let rotation_policy = gguf
        .get_string("hypura.turboquant.rotation_policy")
        .and_then(RotationPolicy::from_str);
    let triality_view = gguf
        .get_string("hypura.turboquant.triality_view")
        .map(ToOwned::to_owned)
        .or_else(|| rotation_policy.and_then(infer_triality_view_from_legacy_policy));
    let triality_mode = gguf
        .get_string("hypura.turboquant.triality_mode")
        .map(ToOwned::to_owned)
        .or_else(|| {
            rotation_policy
                .filter(|policy| policy.is_triality())
                .map(|_| "triality_proxy".to_string())
        });

    Ok(Some(GgufTurboQuantConfig {
        enabled,
        schema_version: gguf
            .get_u32("hypura.turboquant.schema_version")
            .unwrap_or(0),
        mode,
        public_mode_label,
        runtime_mode: gguf
            .get_string("hypura.turboquant.runtime_mode")
            .unwrap_or(&default_runtime_mode)
            .to_string(),
        rotation_policy,
        triality_view,
        triality_mode,
        triality_mix: gguf.get_f32("hypura.turboquant.triality_mix"),
        paper_fidelity: gguf
            .get_bool("hypura.turboquant.paper_fidelity")
            .unwrap_or(matches!(
                mode,
                TurboQuantMode::PaperKeyOnly | TurboQuantMode::PaperFullKv
            )),
        k_bits: gguf.get_f32("hypura.turboquant.k_bits").unwrap_or(0.0),
        v_bits: gguf.get_f32("hypura.turboquant.v_bits").unwrap_or(0.0),
        payload_format: gguf
            .get_string("hypura.turboquant.payload_format")
            .map(ToOwned::to_owned),
        payload_bytes: gguf.get_u64("hypura.turboquant.payload_bytes").unwrap_or(0),
        payload_json: gguf
            .get_string("hypura.turboquant.payload_json")
            .map(ToOwned::to_owned),
        rotation_seed: gguf.get_u32("hypura.turboquant.rotation_seed").unwrap_or(0),
        artifact_path: gguf
            .get_string("hypura.turboquant.artifact")
            .map(ToOwned::to_owned),
        head_dim: head_dim_from_metadata(metadata),
        num_layers: metadata.num_layers,
        num_kv_heads: metadata.num_kv_heads,
        layers: Vec::new(),
        weight: parse_gguf_weight_config(gguf)?,
        consensus: None,
        ncka: None,
        urt: None,
    }))
}

fn parse_strict_gguf_turboquant_config(
    gguf: &GgufFile,
    metadata: &ModelMetadata,
    schema_version: u32,
) -> anyhow::Result<GgufTurboQuantConfig> {
    let expected_len = gguf
        .get_f32_array("tq_total_bits")
        .map(|values| values.len())
        .unwrap_or(metadata.num_layers as usize);
    anyhow::ensure!(
        expected_len > 0,
        "GGUF TurboQuant metadata requires a non-zero layer count"
    );

    let total_bits = require_f32_array(gguf, "tq_total_bits", expected_len)?;
    let runtime_bits = require_f32_array(gguf, "tq_runtime_bits_per_channel", expected_len)?;
    let stage1_effective_bits = require_f32_array(gguf, "tq_stage1_effective_bits", expected_len)?;
    let qjl_bits = require_u32_array(gguf, "tq_qjl_bits", expected_len)?;
    let qjl_dim = require_u32_array(gguf, "tq_qjl_dim", expected_len)?;
    let rotation_policy = require_string_array(gguf, "tq_rotation_policy", expected_len)?;
    let rotation_seed = require_u32_array(gguf, "tq_rotation_seed", expected_len)?;
    let qjl_seed = require_u32_array(gguf, "tq_qjl_seed", expected_len)?;
    let triality_mode = require_string_array(gguf, "tq_triality_mode", expected_len)?;
    let triality_view = require_string_array(gguf, "tq_triality_view", expected_len)?;
    let stage1_allocation_scheme =
        require_string_array(gguf, "tq_stage1_allocation_scheme", expected_len)?;
    let stage1_bitwidth_payload_dtype =
        require_string_array(gguf, "tq_stage1_bitwidth_payload_dtype", expected_len)?;
    let norm_dtype = require_string_array(gguf, "tq_norm_dtype", expected_len)?;
    let sign_pack_format = require_string_array(gguf, "tq_sign_pack_format", expected_len)?;

    let mut layers = Vec::with_capacity(expected_len);
    for i in 0..expected_len {
        anyhow::ensure!(
            (stage1_effective_bits[i] + qjl_bits[i] as f32 - runtime_bits[i]).abs() <= 1e-6,
            "GGUF TurboQuant metadata layer {i} is inconsistent: tq_stage1_effective_bits + tq_qjl_bits != tq_runtime_bits_per_channel"
        );
        let parsed_rotation_policy =
            RotationPolicy::from_str(&rotation_policy[i]).ok_or_else(|| {
                anyhow::anyhow!(
                    "GGUF TurboQuant metadata layer {i} has unsupported tq_rotation_policy `{}`",
                    rotation_policy[i]
                )
            })?;
        layers.push(GgufTurboQuantLayerConfig {
            total_bits: total_bits[i],
            runtime_bits_per_channel: runtime_bits[i],
            stage1_effective_bits: stage1_effective_bits[i],
            qjl_bits: qjl_bits[i],
            qjl_dim: qjl_dim[i],
            rotation_policy: parsed_rotation_policy,
            rotation_seed: rotation_seed[i],
            qjl_seed: qjl_seed[i],
            triality_mode: triality_mode[i].clone(),
            triality_view: triality_view[i].clone(),
            stage1_allocation_scheme: stage1_allocation_scheme[i].clone(),
            stage1_bitwidth_payload_dtype: stage1_bitwidth_payload_dtype[i].clone(),
            norm_dtype: norm_dtype[i].clone(),
            sign_pack_format: sign_pack_format[i].clone(),
        });
    }

    let first_layer = layers
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("GGUF TurboQuant metadata did not contain any layers"))?;
    let raw_mode = gguf.get_string("hypura.turboquant.mode");
    let mode = raw_mode
        .map(parse_turboquant_mode)
        .transpose()?
        .unwrap_or(TurboQuantMode::ResearchKvSplit);
    let (public_mode_label, default_runtime_mode) = raw_mode
        .and_then(parse_embedded_mode)
        .map(|(_, public, runtime)| (public, runtime))
        .unwrap_or_else(|| (mode.as_str().to_string(), mode.as_str().to_string()));
    let legacy_rotation_policy = gguf
        .get_string("hypura.turboquant.rotation_policy")
        .and_then(RotationPolicy::from_str)
        .or_else(|| match first_layer.triality_view.as_str() {
            "vector" => Some(RotationPolicy::TrialityVector),
            "spinor_plus_proxy" => Some(RotationPolicy::TrialitySpinorPlus),
            "spinor_minus_proxy" => Some(RotationPolicy::TrialitySpinorMinus),
            _ => Some(first_layer.rotation_policy),
        });

    Ok(GgufTurboQuantConfig {
        enabled: gguf.get_bool("hypura.turboquant.enabled").unwrap_or(true),
        schema_version,
        mode,
        public_mode_label,
        runtime_mode: gguf
            .get_string("hypura.turboquant.runtime_mode")
            .unwrap_or(&default_runtime_mode)
            .to_string(),
        rotation_policy: legacy_rotation_policy,
        triality_view: Some(first_layer.triality_view.clone()),
        triality_mode: Some(first_layer.triality_mode.clone()),
        triality_mix: gguf.get_f32("hypura.turboquant.triality_mix"),
        paper_fidelity: gguf
            .get_bool("hypura.turboquant.paper_fidelity")
            .unwrap_or(matches!(
                mode,
                TurboQuantMode::PaperKeyOnly | TurboQuantMode::PaperFullKv
            )),
        k_bits: gguf
            .get_f32("hypura.turboquant.k_bits")
            .unwrap_or(first_layer.total_bits),
        v_bits: gguf.get_f32("hypura.turboquant.v_bits").unwrap_or(0.0),
        payload_format: gguf
            .get_string("hypura.turboquant.payload_format")
            .map(ToOwned::to_owned),
        payload_bytes: gguf.get_u64("hypura.turboquant.payload_bytes").unwrap_or(0),
        payload_json: gguf
            .get_string("hypura.turboquant.payload_json")
            .map(ToOwned::to_owned),
        rotation_seed: first_layer.rotation_seed,
        artifact_path: gguf
            .get_string("hypura.turboquant.artifact")
            .map(ToOwned::to_owned),
        head_dim: head_dim_from_metadata(metadata),
        num_layers: metadata.num_layers,
        num_kv_heads: metadata.num_kv_heads,
        layers,
        weight: parse_gguf_weight_config(gguf)?,
        consensus: None,
        ncka: None,
        urt: None,
    })
}

const TRIALITY_VIEWS: [&str; 3] = ["vector", "spinor_plus_proxy", "spinor_minus_proxy"];
const NCKA_COORDINATES: [&str; 24] = [
    "branch_entropy.vector",
    "branch_entropy.spinor_plus_proxy",
    "branch_entropy.spinor_minus_proxy",
    "orthogonality_error.vector",
    "orthogonality_error.spinor_plus_proxy",
    "orthogonality_error.spinor_minus_proxy",
    "determinant_error.vector",
    "determinant_error.spinor_plus_proxy",
    "determinant_error.spinor_minus_proxy",
    "expected_quant_error.vector",
    "expected_quant_error.spinor_plus_proxy",
    "expected_quant_error.spinor_minus_proxy",
    "pairwise_js.vector_plus",
    "pairwise_js.vector_minus",
    "pairwise_js.plus_minus",
    "candidate_cross_score_mean.vector",
    "candidate_cross_score_mean.spinor_plus_proxy",
    "candidate_cross_score_mean.spinor_minus_proxy",
    "candidate_cross_score_variance.vector",
    "candidate_cross_score_variance.spinor_plus_proxy",
    "candidate_cross_score_variance.spinor_minus_proxy",
    "winner_margin",
    "latency_multiplier",
    "memory_ratio",
];

fn exact_metadata<'a>(gguf: &'a GgufFile, key: &str) -> anyhow::Result<&'a GgufValue> {
    gguf.metadata
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("GGUF Triality schema-v2 metadata is missing `{key}`"))
}

fn exact_bool(gguf: &GgufFile, key: &str) -> anyhow::Result<bool> {
    match exact_metadata(gguf, key)? {
        GgufValue::Bool(value) => Ok(*value),
        _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` must be BOOL"),
    }
}

fn exact_u32(gguf: &GgufFile, key: &str) -> anyhow::Result<u32> {
    match exact_metadata(gguf, key)? {
        GgufValue::Uint32(value) => Ok(*value),
        _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` must be UINT32"),
    }
}

fn exact_u64(gguf: &GgufFile, key: &str) -> anyhow::Result<u64> {
    match exact_metadata(gguf, key)? {
        GgufValue::Uint64(value) => Ok(*value),
        _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` must be UINT64"),
    }
}

fn exact_f32(gguf: &GgufFile, key: &str) -> anyhow::Result<f32> {
    match exact_metadata(gguf, key)? {
        GgufValue::Float32(value) => Ok(*value),
        _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` must be FLOAT32"),
    }
}

fn exact_string(gguf: &GgufFile, key: &str) -> anyhow::Result<String> {
    match exact_metadata(gguf, key)? {
        GgufValue::String(value) => Ok(value.clone()),
        _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` must be STRING"),
    }
}

fn exact_array<'a>(gguf: &'a GgufFile, key: &str) -> anyhow::Result<&'a [GgufValue]> {
    match exact_metadata(gguf, key)? {
        GgufValue::Array(values) => Ok(values),
        _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` must be ARRAY"),
    }
}

fn exact_f32_array(gguf: &GgufFile, key: &str, len: usize) -> anyhow::Result<Vec<f32>> {
    let values = exact_array(gguf, key)?;
    anyhow::ensure!(
        values.len() == len,
        "GGUF Triality schema-v2 metadata `{key}` must have length {len}, got {}",
        values.len()
    );
    values
        .iter()
        .map(|value| match value {
            GgufValue::Float32(value) => Ok(*value),
            _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` elements must be FLOAT32"),
        })
        .collect()
}

fn exact_u32_array(gguf: &GgufFile, key: &str, len: usize) -> anyhow::Result<Vec<u32>> {
    let values = exact_array(gguf, key)?;
    anyhow::ensure!(
        values.len() == len,
        "GGUF Triality schema-v2 metadata `{key}` must have length {len}, got {}",
        values.len()
    );
    values
        .iter()
        .map(|value| match value {
            GgufValue::Uint32(value) => Ok(*value),
            _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` elements must be UINT32"),
        })
        .collect()
}

fn exact_string_array(gguf: &GgufFile, key: &str, len: usize) -> anyhow::Result<Vec<String>> {
    let values = exact_array(gguf, key)?;
    anyhow::ensure!(
        values.len() == len,
        "GGUF Triality schema-v2 metadata `{key}` must have length {len}, got {}",
        values.len()
    );
    values
        .iter()
        .map(|value| match value {
            GgufValue::String(value) => Ok(value.clone()),
            _ => anyhow::bail!("GGUF Triality schema-v2 metadata `{key}` elements must be STRING"),
        })
        .collect()
}

fn json_object<'a>(value: &'a Value, name: &str) -> anyhow::Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Triality payload `{name}` must be an object"))
}

fn json_field<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    name: &str,
) -> anyhow::Result<&'a Value> {
    object
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("Triality payload `{name}` is missing `{key}`"))
}

fn json_string(object: &Map<String, Value>, key: &str, name: &str) -> anyhow::Result<String> {
    json_field(object, key, name)?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("Triality payload `{name}.{key}` must be a string"))
}

fn json_bool(object: &Map<String, Value>, key: &str, name: &str) -> anyhow::Result<bool> {
    json_field(object, key, name)?
        .as_bool()
        .ok_or_else(|| anyhow::anyhow!("Triality payload `{name}.{key}` must be a boolean"))
}

fn json_u32(object: &Map<String, Value>, key: &str, name: &str) -> anyhow::Result<u32> {
    let value = json_field(object, key, name)?.as_u64().ok_or_else(|| {
        anyhow::anyhow!("Triality payload `{name}.{key}` must be an unsigned integer")
    })?;
    u32::try_from(value)
        .map_err(|_| anyhow::anyhow!("Triality payload `{name}.{key}` exceeds UINT32"))
}

fn json_f32(object: &Map<String, Value>, key: &str, name: &str) -> anyhow::Result<f32> {
    let value = json_field(object, key, name)?
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("Triality payload `{name}.{key}` must be numeric"))?;
    anyhow::ensure!(
        value.is_finite() && value >= f32::MIN as f64 && value <= f32::MAX as f64,
        "Triality payload `{name}.{key}` is outside finite FLOAT32 range"
    );
    Ok(value as f32)
}

fn json_string_array(
    object: &Map<String, Value>,
    key: &str,
    name: &str,
) -> anyhow::Result<Vec<String>> {
    let values = json_field(object, key, name)?
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Triality payload `{name}.{key}` must be an array"))?;
    values
        .iter()
        .map(|value| {
            value.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                anyhow::anyhow!("Triality payload `{name}.{key}` elements must be strings")
            })
        })
        .collect()
}

fn json_f32_vector(value: &Value, name: &str, len: usize) -> anyhow::Result<Vec<f32>> {
    let values = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Triality payload `{name}` must be an array"))?;
    anyhow::ensure!(
        values.len() == len,
        "Triality payload `{name}` must have length {len}"
    );
    values
        .iter()
        .map(|value| {
            let value = value.as_f64().ok_or_else(|| {
                anyhow::anyhow!("Triality payload `{name}` elements must be numeric")
            })?;
            anyhow::ensure!(
                value.is_finite() && value >= f32::MIN as f64 && value <= f32::MAX as f64,
                "Triality payload `{name}` contains a non-finite or out-of-range value"
            );
            Ok(value as f32)
        })
        .collect()
}

fn exact_json_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    name: &str,
) -> anyhow::Result<()> {
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let extra = actual.difference(&expected).copied().collect::<Vec<_>>();
    anyhow::ensure!(
        missing.is_empty() && extra.is_empty(),
        "Triality payload `{name}` keys are invalid: missing={missing:?}, unexpected={extra:?}"
    );
    Ok(())
}

fn valid_profile_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 64
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .skip(1)
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_sha256(value: &Value) -> anyhow::Result<String> {
    let encoded = serde_json::to_vec(value).context("Failed to canonicalise Triality JSON")?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn same_f32(left: f32, right: f32) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= 1.0e-6
}

fn same_f32_slice(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| same_f32(*left, *right))
}

fn validate_probability(values: &[f32], name: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        values.len() == 3
            && values
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && (values.iter().sum::<f32>() - 1.0).abs() <= 1.0e-6,
        "{name} must contain three finite non-negative values summing to one"
    );
    Ok(())
}

fn validate_v2_namespace_keys(gguf: &GgufFile) -> anyhow::Result<()> {
    const ALLOWED: &[&str] = &[
        "hypura.turboquant.schema_version",
        "hypura.turboquant.enabled",
        "hypura.turboquant.mode",
        "hypura.turboquant.codec",
        "hypura.turboquant.rotation_policy",
        "hypura.turboquant.rotation_block_size",
        "hypura.turboquant.rotation_seed",
        "hypura.turboquant.triality_view",
        "hypura.turboquant.triality_mix",
        "hypura.turboquant.cache_type_k",
        "hypura.turboquant.cache_type_v",
        "hypura.turboquant.view_bundle_complete",
        "hypura.turboquant.orthogonality_error",
        "hypura.turboquant.determinant_error_max",
        "hypura.turboquant.paper_fidelity",
        "hypura.turboquant.k_bits",
        "hypura.turboquant.v_bits",
        "hypura.turboquant.payload_format",
        "hypura.turboquant.payload_bytes",
        "hypura.turboquant.payload_json",
        "hypura.turboquant.runtime_mode",
        "hypura.turboquant.source_profile",
        "hypura.turboquant.artifact",
        "hypura.turboquant.weight.enabled",
        "hypura.turboquant.weight.codec",
        "hypura.turboquant.weight.source_ftype",
        "hypura.turboquant.weight.policy",
        "hypura.turboquant.weight.protected_roles",
        "hypura.turboquant.weight.protected_layers",
        "hypura.turboquant.weight.modality_scope",
        "hypura.turboquant.weight.payload_format",
        "hypura.turboquant.weight.payload_bytes",
        "hypura.turboquant.weight.payload_json",
        "hypura.turboquant.weight.generated_at_utc",
        "hypura.turboquant.triality.profile_id",
        "hypura.turboquant.triality.execution",
        "hypura.turboquant.triality.override_allowed",
        "hypura.turboquant.triality.view_count",
        "hypura.turboquant.triality.views",
        "hypura.turboquant.triality.weights",
        "hypura.turboquant.triality.bias",
        "hypura.turboquant.triality.scale",
        "hypura.turboquant.triality.temperature",
        "hypura.turboquant.triality.js_fallback_threshold",
        "hypura.turboquant.ncka.enabled",
        "hypura.turboquant.ncka.required",
        "hypura.turboquant.ncka.schema_version",
        "hypura.turboquant.ncka.controller_type",
        "hypura.turboquant.ncka.coordinate_names",
        "hypura.turboquant.ncka.outer_count",
        "hypura.turboquant.ncka.knot_count",
        "hypura.turboquant.ncka.s3_equivariant",
        "hypura.turboquant.ncka.controller_sha256",
        "hypura.turboquant.ncka.normalisation_sha256",
        "hypura.turboquant.urt.enabled",
        "hypura.turboquant.urt.schema_version",
        "hypura.turboquant.urt.abstract_algebra_id",
        "hypura.turboquant.urt.operator_word_manifest",
        "hypura.turboquant.urt.operator_word_sha256",
        "hypura.turboquant.urt.reference_representation",
        "hypura.turboquant.urt.supported_representations",
        "hypura.turboquant.urt.consistency_tolerance",
        "hypura.turboquant.urt.moment_degree",
        "hypura.turboquant.urt.moment_manifest_sha256",
    ];
    for key in gguf.metadata.keys() {
        let allowed = !key.starts_with("hypura.turboquant.") || ALLOWED.contains(&key.as_str());
        anyhow::ensure!(allowed, "Unknown Triality schema-v2 metadata key `{key}`");
    }
    Ok(())
}

fn validate_v2_base_types(gguf: &GgufFile, metadata: &ModelMetadata) -> anyhow::Result<()> {
    anyhow::ensure!(
        exact_u32(gguf, "tq_schema_version")? == 1,
        "Triality schema-v2 requires canonical TurboQuant artifact schema 1"
    );
    anyhow::ensure!(
        exact_bool(gguf, "hypura.turboquant.enabled")?,
        "Triality schema-v2 cannot be disabled"
    );
    exact_string(gguf, "hypura.turboquant.mode")?;
    exact_string(gguf, "hypura.turboquant.runtime_mode")?;
    anyhow::ensure!(
        exact_string(gguf, "hypura.turboquant.payload_format")? == "json-inline-v2",
        "Triality schema-v2 payload format must be `json-inline-v2`"
    );
    exact_u64(gguf, "hypura.turboquant.payload_bytes")?;
    exact_string(gguf, "hypura.turboquant.payload_json")?;

    let layer_count = metadata.num_layers as usize;
    anyhow::ensure!(
        layer_count > 0,
        "Triality schema-v2 requires at least one layer"
    );
    for key in [
        "tq_total_bits",
        "tq_runtime_bits_per_channel",
        "tq_stage1_effective_bits",
    ] {
        let values = exact_f32_array(gguf, key, layer_count)?;
        anyhow::ensure!(
            values
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "Triality schema-v2 metadata `{key}` must contain finite non-negative values"
        );
    }
    for key in [
        "tq_qjl_bits",
        "tq_qjl_dim",
        "tq_rotation_seed",
        "tq_qjl_seed",
    ] {
        exact_u32_array(gguf, key, layer_count)?;
    }
    for key in [
        "tq_rotation_policy",
        "tq_triality_mode",
        "tq_triality_view",
        "tq_stage1_allocation_scheme",
        "tq_stage1_bitwidth_payload_dtype",
        "tq_norm_dtype",
        "tq_sign_pack_format",
    ] {
        exact_string_array(gguf, key, layer_count)?;
    }
    Ok(())
}

fn validate_artifact_path(gguf: &GgufFile) -> anyhow::Result<()> {
    let Some(value) = gguf.metadata.get("hypura.turboquant.artifact") else {
        return Ok(());
    };
    let GgufValue::String(value) = value else {
        anyhow::bail!("Triality schema-v2 artifact path must be STRING");
    };
    anyhow::ensure!(
        !value.is_empty(),
        "Triality schema-v2 artifact path must not be empty"
    );
    let path = Path::new(value);
    anyhow::ensure!(
        !path.is_absolute(),
        "Triality schema-v2 artifact path must be relative"
    );
    anyhow::ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "Triality schema-v2 artifact path must not contain root, prefix, parent, or current-directory components"
    );
    Ok(())
}

fn validate_payload_header(
    payload: &Map<String, Value>,
    gguf: &GgufFile,
    metadata: &ModelMetadata,
) -> anyhow::Result<(u32, u32)> {
    let paper_fidelity = json_bool(payload, "paper_fidelity", "payload")?;
    let mut expected_keys = vec![
        "schema_kind",
        "schema_version",
        "codec",
        "mode",
        "model_family",
        "runtime_mode",
        "head_dim",
        "num_layers",
        "num_kv_heads",
        "rotation_policy",
        "rotation_block_size",
        "rotation_seed",
        "triality_view",
        "triality_mix",
        "cache_type_k",
        "cache_type_v",
        "view_bundle_complete",
        "orthogonality_error",
        "determinant_error_max",
        "paper_fidelity",
        "k_bits",
        "v_bits",
        "offline_metrics",
        "weight_plan",
        "profile_id",
        "consensus",
        "ncka",
        "urt",
        "tensor_manifest",
    ];
    let conditional_key = if paper_fidelity {
        "paper_config"
    } else {
        "pareto_profile"
    };
    expected_keys.push(conditional_key);
    if payload.contains_key("source_manifest") {
        expected_keys.push("source_manifest");
    }
    exact_json_keys(payload, &expected_keys, "payload")?;
    json_object(
        json_field(payload, conditional_key, "payload")?,
        conditional_key,
    )?;
    if let Some(source_manifest) = payload.get("source_manifest") {
        json_object(source_manifest, "source_manifest")?;
    }
    anyhow::ensure!(
        json_string(payload, "schema_kind", "payload")? == "triality_gguf_payload",
        "Unsupported Triality payload schema_kind"
    );
    anyhow::ensure!(
        json_u32(payload, "schema_version", "payload")? == 2,
        "Unsupported Triality payload schema_version"
    );
    anyhow::ensure!(
        json_string(payload, "codec", "payload")? == "tq4_1s",
        "Unsupported Triality payload codec"
    );
    anyhow::ensure!(
        json_string(payload, "model_family", "payload")? == metadata.architecture,
        "Triality payload model_family does not match GGUF architecture"
    );
    anyhow::ensure!(
        json_string(payload, "mode", "payload")? == exact_string(gguf, "hypura.turboquant.mode")?,
        "Triality payload mode contradicts flattened metadata"
    );
    anyhow::ensure!(
        json_string(payload, "runtime_mode", "payload")?
            == exact_string(gguf, "hypura.turboquant.runtime_mode")?,
        "Triality payload runtime_mode contradicts flattened metadata"
    );
    anyhow::ensure!(
        json_bool(payload, "view_bundle_complete", "payload")?,
        "Triality schema-v2 requires a complete three-view bundle"
    );
    let num_layers = json_u32(payload, "num_layers", "payload")?;
    let head_dim = json_u32(payload, "head_dim", "payload")?;
    anyhow::ensure!(
        num_layers == metadata.num_layers,
        "Triality payload layer count does not match GGUF model metadata"
    );
    anyhow::ensure!(
        json_u32(payload, "num_kv_heads", "payload")? == metadata.num_kv_heads,
        "Triality payload KV-head count does not match GGUF model metadata"
    );
    anyhow::ensure!(
        head_dim > 0 && head_dim % 8 == 0 && head_dim == head_dim_from_metadata(metadata),
        "Triality payload head_dim must match the model and be a positive multiple of 8"
    );
    let qjl_dim = exact_u32_array(gguf, "tq_qjl_dim", num_layers as usize)?;
    anyhow::ensure!(
        qjl_dim.iter().all(|value| *value == head_dim),
        "Triality schema-v2 requires every tq_qjl_dim to equal head_dim"
    );
    Ok((num_layers, head_dim))
}

fn parse_v2_consensus(
    gguf: &GgufFile,
    payload: &Map<String, Value>,
    layers: &[GgufTurboQuantLayerConfig],
) -> anyhow::Result<GgufTrialityConsensusConfig> {
    let profile_id = exact_string(gguf, "hypura.turboquant.triality.profile_id")?;
    anyhow::ensure!(
        valid_profile_id(&profile_id),
        "Triality profile_id must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}"
    );
    anyhow::ensure!(
        json_string(payload, "profile_id", "payload")? == profile_id,
        "Triality profile_id contradicts payload"
    );
    let execution = exact_string(gguf, "hypura.turboquant.triality.execution")?;
    anyhow::ensure!(
        matches!(
            execution.as_str(),
            "single_view" | "best_per_layer" | "attention_logit_consensus" | "residual_parity"
        ),
        "Unsupported Triality execution `{execution}`"
    );
    anyhow::ensure!(
        exact_u32(gguf, "hypura.turboquant.triality.view_count")? == 3,
        "Triality view_count must be 3"
    );
    let views = exact_string_array(gguf, "hypura.turboquant.triality.views", 3)?;
    anyhow::ensure!(
        views.iter().map(String::as_str).eq(TRIALITY_VIEWS),
        "Triality views must use canonical branch names and order"
    );
    let flattened_len = layers.len() * 3;
    let weights = exact_f32_array(gguf, "hypura.turboquant.triality.weights", flattened_len)?;
    let bias = exact_f32_array(gguf, "hypura.turboquant.triality.bias", flattened_len)?;
    let scale = exact_f32_array(gguf, "hypura.turboquant.triality.scale", flattened_len)?;
    let temperature = exact_f32_array(
        gguf,
        "hypura.turboquant.triality.temperature",
        flattened_len,
    )?;
    let js_fallback_threshold =
        exact_f32(gguf, "hypura.turboquant.triality.js_fallback_threshold")?;
    anyhow::ensure!(
        js_fallback_threshold.is_finite() && js_fallback_threshold >= 0.0,
        "Triality JS fallback threshold must be finite and non-negative"
    );
    let override_allowed = match gguf
        .metadata
        .get("hypura.turboquant.triality.override_allowed")
    {
        None => false,
        Some(GgufValue::Bool(value)) => *value,
        Some(_) => anyhow::bail!(
            "GGUF Triality schema-v2 metadata `hypura.turboquant.triality.override_allowed` must be BOOL"
        ),
    };

    let consensus = json_object(json_field(payload, "consensus", "payload")?, "consensus")?;
    exact_json_keys(
        consensus,
        &[
            "schema_version",
            "execution",
            "view_count",
            "views",
            "rows",
            "js_fallback_threshold",
            "fallback_policy",
            "fallback_weights",
        ],
        "consensus",
    )?;
    anyhow::ensure!(
        json_u32(consensus, "schema_version", "consensus")? == 1,
        "Unsupported Triality consensus schema_version"
    );
    anyhow::ensure!(
        json_string(consensus, "execution", "consensus")? == execution,
        "Triality consensus execution contradicts flattened metadata"
    );
    anyhow::ensure!(
        json_u32(consensus, "view_count", "consensus")? == 3,
        "Triality payload consensus view_count must be 3"
    );
    anyhow::ensure!(
        json_string_array(consensus, "views", "consensus")?
            .iter()
            .map(String::as_str)
            .eq(TRIALITY_VIEWS),
        "Triality payload consensus views must use canonical order"
    );
    anyhow::ensure!(
        same_f32(
            json_f32(consensus, "js_fallback_threshold", "consensus")?,
            js_fallback_threshold
        ),
        "Triality JS fallback threshold contradicts flattened metadata"
    );
    anyhow::ensure!(
        json_string(consensus, "fallback_policy", "consensus")? == "static",
        "Triality consensus fallback_policy must be `static`"
    );
    let fallback_weights = json_f32_vector(
        json_field(consensus, "fallback_weights", "consensus")?,
        "consensus.fallback_weights",
        3,
    )?;
    validate_probability(&fallback_weights, "Triality consensus fallback_weights")?;

    let rows = json_field(consensus, "rows", "consensus")?
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Triality payload `consensus.rows` must be an array"))?;
    anyhow::ensure!(
        rows.len() == layers.len(),
        "Triality consensus rows must contain exactly one row per layer"
    );
    let mut branches_by_layer = Vec::with_capacity(layers.len());
    for (layer_index, (row, layer)) in rows.iter().zip(layers).enumerate() {
        let row_name = format!("consensus.rows[{layer_index}]");
        let row = json_object(row, &row_name)?;
        exact_json_keys(
            row,
            &["layer", "weights", "bias", "scale", "temperature"],
            &row_name,
        )?;
        anyhow::ensure!(
            json_u32(row, "layer", &row_name)? == layer_index as u32,
            "Triality consensus row indices must be contiguous"
        );
        let row_weights = json_f32_vector(
            json_field(row, "weights", &row_name)?,
            &format!("{row_name}.weights"),
            3,
        )?;
        let row_bias = json_f32_vector(
            json_field(row, "bias", &row_name)?,
            &format!("{row_name}.bias"),
            3,
        )?;
        let row_scale = json_f32_vector(
            json_field(row, "scale", &row_name)?,
            &format!("{row_name}.scale"),
            3,
        )?;
        let row_temperature = json_f32_vector(
            json_field(row, "temperature", &row_name)?,
            &format!("{row_name}.temperature"),
            3,
        )?;
        validate_probability(
            &row_weights,
            &format!("Triality layer {layer_index} weights"),
        )?;
        anyhow::ensure!(
            row_bias.iter().all(|value| value.is_finite()),
            "Triality layer {layer_index} bias must be finite"
        );
        anyhow::ensure!(
            row_scale
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "Triality layer {layer_index} scale must be finite and positive"
        );
        anyhow::ensure!(
            row_temperature
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "Triality layer {layer_index} temperature must be finite and positive"
        );
        let offset = layer_index * 3;
        anyhow::ensure!(
            same_f32_slice(&weights[offset..offset + 3], &row_weights)
                && same_f32_slice(&bias[offset..offset + 3], &row_bias)
                && same_f32_slice(&scale[offset..offset + 3], &row_scale)
                && same_f32_slice(&temperature[offset..offset + 3], &row_temperature),
            "Triality consensus layer {layer_index} contradicts flattened metadata"
        );
        if execution == "single_view" {
            anyhow::ensure!(
                row_weights
                    .iter()
                    .filter(|weight| **weight > 1.0e-6)
                    .count()
                    == 1,
                "single_view Triality execution requires one active branch per layer"
            );
        }
        let branches = (0..3)
            .map(|branch| GgufTrialityBranchConfig {
                view: TRIALITY_VIEWS[branch].to_string(),
                weight: row_weights[branch],
                bias: row_bias[branch],
                scale: row_scale[branch],
                temperature: row_temperature[branch],
                expected_error: 0.0,
                bits_per_channel: layer.runtime_bits_per_channel,
            })
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| anyhow::anyhow!("Triality layer must contain exactly three branches"))?;
        branches_by_layer.push(branches);
    }

    Ok(GgufTrialityConsensusConfig {
        profile_id,
        execution,
        branches_by_layer,
        js_fallback_threshold,
        required: true,
        override_allowed,
    })
}

fn parse_v2_ncka(gguf: &GgufFile, payload: &Map<String, Value>) -> anyhow::Result<GgufNcKaConfig> {
    let enabled = exact_bool(gguf, "hypura.turboquant.ncka.enabled")?;
    let required = exact_bool(gguf, "hypura.turboquant.ncka.required")?;
    let schema_version = exact_u32(gguf, "hypura.turboquant.ncka.schema_version")?;
    let controller_type = exact_string(gguf, "hypura.turboquant.ncka.controller_type")?;
    let coordinate_names = exact_string_array(
        gguf,
        "hypura.turboquant.ncka.coordinate_names",
        if enabled { NCKA_COORDINATES.len() } else { 0 },
    )?;
    let outer_count = exact_u32(gguf, "hypura.turboquant.ncka.outer_count")?;
    let knot_count = exact_u32(gguf, "hypura.turboquant.ncka.knot_count")?;
    let s3_equivariant = exact_bool(gguf, "hypura.turboquant.ncka.s3_equivariant")?;
    let controller_sha256 = exact_string(gguf, "hypura.turboquant.ncka.controller_sha256")?;
    let normalisation_sha256 = exact_string(gguf, "hypura.turboquant.ncka.normalisation_sha256")?;

    let ncka = json_object(json_field(payload, "ncka", "payload")?, "ncka")?;
    exact_json_keys(
        ncka,
        &[
            "enabled",
            "required",
            "schema_version",
            "controller_type",
            "coordinate_names",
            "outer_count",
            "knot_count",
            "s3_equivariant",
            "fallback_policy",
            "fallback_weights",
            "normalisation_sha256",
            "controller_sha256",
        ],
        "ncka",
    )?;
    anyhow::ensure!(
        json_bool(ncka, "enabled", "ncka")? == enabled
            && json_bool(ncka, "required", "ncka")? == required
            && json_u32(ncka, "schema_version", "ncka")? == schema_version
            && json_string(ncka, "controller_type", "ncka")? == controller_type
            && json_string_array(ncka, "coordinate_names", "ncka")? == coordinate_names
            && json_u32(ncka, "outer_count", "ncka")? == outer_count
            && json_u32(ncka, "knot_count", "ncka")? == knot_count
            && json_bool(ncka, "s3_equivariant", "ncka")? == s3_equivariant
            && json_string(ncka, "controller_sha256", "ncka")? == controller_sha256
            && json_string(ncka, "normalisation_sha256", "ncka")? == normalisation_sha256,
        "NC-KA payload contradicts flattened metadata"
    );
    anyhow::ensure!(
        json_string(ncka, "fallback_policy", "ncka")? == "static",
        "NC-KA fallback_policy must be `static`"
    );
    let fallback = json_f32_vector(
        json_field(ncka, "fallback_weights", "ncka")?,
        "ncka.fallback_weights",
        3,
    )?;
    validate_probability(&fallback, "NC-KA fallback_weights")?;
    let fallback_weights: [f32; 3] = fallback
        .try_into()
        .map_err(|_| anyhow::anyhow!("NC-KA fallback_weights must contain three values"))?;
    let supported = schema_version == 1 && controller_type == "finite_moment_ka_v1";

    if enabled {
        anyhow::ensure!(schema_version == 1, "Unsupported NC-KA schema_version");
        anyhow::ensure!(
            !required || supported,
            "Required NC-KA controller `{controller_type}` is unsupported"
        );
        anyhow::ensure!(
            coordinate_names
                .iter()
                .map(String::as_str)
                .eq(NCKA_COORDINATES),
            "Enabled NC-KA requires canonical coordinate_names"
        );
        anyhow::ensure!(
            s3_equivariant && outer_count > 0 && knot_count >= 2,
            "Enabled NC-KA requires S3 equivariance, positive outer_count, and at least two knots"
        );
        anyhow::ensure!(
            valid_sha256(&controller_sha256) && valid_sha256(&normalisation_sha256),
            "Enabled NC-KA requires lowercase SHA256 hashes"
        );
        let normalisation = serde_json::json!({
            "coordinate_names": NCKA_COORDINATES,
            "range": [0.0, 1.0],
            "clamp": true,
        });
        anyhow::ensure!(
            canonical_sha256(&normalisation)? == normalisation_sha256,
            "NC-KA normalisation hash mismatch"
        );
    } else {
        anyhow::ensure!(
            !required
                && schema_version == 0
                && controller_type.is_empty()
                && coordinate_names.is_empty()
                && outer_count == 0
                && knot_count == 0
                && !s3_equivariant
                && controller_sha256.is_empty()
                && normalisation_sha256.is_empty(),
            "Disabled NC-KA must use the canonical empty contract"
        );
    }

    Ok(GgufNcKaConfig {
        enabled,
        required,
        schema_version,
        controller_type,
        coordinate_names,
        outer_count,
        knot_count,
        s3_equivariant,
        controller_sha256,
        normalisation_sha256,
        static_fallback_selected: enabled && !supported,
        fallback_weights,
    })
}

fn parse_v2_urt(gguf: &GgufFile, payload: &Map<String, Value>) -> anyhow::Result<GgufUrtConfig> {
    let enabled = exact_bool(gguf, "hypura.turboquant.urt.enabled")?;
    let schema_version = exact_u32(gguf, "hypura.turboquant.urt.schema_version")?;
    let abstract_algebra_id = exact_string(gguf, "hypura.turboquant.urt.abstract_algebra_id")?;
    let operator_word_manifest =
        exact_string(gguf, "hypura.turboquant.urt.operator_word_manifest")?;
    let operator_word_sha256 = exact_string(gguf, "hypura.turboquant.urt.operator_word_sha256")?;
    let reference_representation =
        exact_string(gguf, "hypura.turboquant.urt.reference_representation")?;
    let supported_representations = exact_string_array(
        gguf,
        "hypura.turboquant.urt.supported_representations",
        exact_array(gguf, "hypura.turboquant.urt.supported_representations")?.len(),
    )?;
    let consistency_tolerance = exact_f32(gguf, "hypura.turboquant.urt.consistency_tolerance")?;
    let moment_degree = exact_u32(gguf, "hypura.turboquant.urt.moment_degree")?;
    let moment_manifest_sha256 =
        exact_string(gguf, "hypura.turboquant.urt.moment_manifest_sha256")?;

    let urt = json_object(json_field(payload, "urt", "payload")?, "urt")?;
    exact_json_keys(
        urt,
        &[
            "enabled",
            "schema_version",
            "abstract_algebra_id",
            "operator_word_manifest",
            "operator_word_sha256",
            "reference_representation",
            "supported_representations",
            "consistency_tolerance",
            "moment_degree",
            "moment_manifest_sha256",
        ],
        "urt",
    )?;
    let manifest_value = json_field(urt, "operator_word_manifest", "urt")?;
    let canonical_manifest = serde_json::to_string(manifest_value)
        .context("Failed to canonicalise URT operator manifest")?;
    anyhow::ensure!(
        json_bool(urt, "enabled", "urt")? == enabled
            && json_u32(urt, "schema_version", "urt")? == schema_version
            && json_string(urt, "abstract_algebra_id", "urt")? == abstract_algebra_id
            && canonical_manifest == operator_word_manifest
            && json_string(urt, "operator_word_sha256", "urt")? == operator_word_sha256
            && json_string(urt, "reference_representation", "urt")? == reference_representation
            && json_string_array(urt, "supported_representations", "urt")?
                == supported_representations
            && same_f32(
                json_f32(urt, "consistency_tolerance", "urt")?,
                consistency_tolerance
            )
            && json_u32(urt, "moment_degree", "urt")? == moment_degree
            && json_string(urt, "moment_manifest_sha256", "urt")? == moment_manifest_sha256,
        "URT payload contradicts flattened metadata"
    );

    if enabled {
        anyhow::ensure!(schema_version == 1, "Unsupported URT schema_version");
        anyhow::ensure!(
            abstract_algebra_id == "octonion_triality_proxy_v1",
            "Unsupported URT abstract_algebra_id"
        );
        anyhow::ensure!(
            valid_sha256(&operator_word_sha256)
                && canonical_sha256(manifest_value)? == operator_word_sha256,
            "URT operator word hash mismatch"
        );
        anyhow::ensure!(
            !supported_representations.is_empty()
                && supported_representations.contains(&reference_representation),
            "URT reference representation must be listed as supported"
        );
        anyhow::ensure!(
            consistency_tolerance.is_finite() && consistency_tolerance > 0.0,
            "URT consistency_tolerance must be finite and positive"
        );
        anyhow::ensure!(moment_degree == 4, "URT moment_degree must be 4");
        let moment_manifest = serde_json::json!({
            "degree": 4,
            "moments": ["mean", "variance", "skewness", "kurtosis"],
        });
        anyhow::ensure!(
            valid_sha256(&moment_manifest_sha256)
                && canonical_sha256(&moment_manifest)? == moment_manifest_sha256,
            "URT moment manifest hash mismatch"
        );
    } else {
        anyhow::ensure!(
            schema_version == 0
                && abstract_algebra_id.is_empty()
                && manifest_value.as_object().is_some_and(Map::is_empty)
                && operator_word_manifest == "{}"
                && operator_word_sha256.is_empty()
                && reference_representation.is_empty()
                && supported_representations.is_empty()
                && consistency_tolerance == 0.0
                && moment_degree == 0
                && moment_manifest_sha256.is_empty(),
            "Disabled URT must use the canonical empty contract"
        );
    }

    Ok(GgufUrtConfig {
        enabled,
        schema_version,
        abstract_algebra_id,
        operator_word_manifest,
        operator_word_sha256,
        reference_representation,
        supported_representations,
        consistency_tolerance,
        moment_degree,
        moment_manifest_sha256,
    })
}

fn manifest_shape(entry: &Map<String, Value>, name: &str) -> anyhow::Result<Vec<u64>> {
    json_field(entry, "shape", name)?
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Triality tensor manifest `{name}.shape` must be an array"))?
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                anyhow::anyhow!(
                    "Triality tensor manifest `{name}.shape` must contain UINT64 values"
                )
            })
        })
        .collect()
}

fn validate_v2_tensor_manifest(
    gguf: &GgufFile,
    payload: &Map<String, Value>,
    consensus: &GgufTrialityConsensusConfig,
    ncka: &GgufNcKaConfig,
    head_dim: u32,
) -> anyhow::Result<()> {
    let manifest = json_object(
        json_field(payload, "tensor_manifest", "payload")?,
        "tensor_manifest",
    )?;
    let profile = &consensus.profile_id;
    let layers = consensus.branches_by_layer.len() as u64;
    let mut expected = Vec::<(String, Vec<u64>)>::new();
    for layer in 0..layers {
        for view in TRIALITY_VIEWS {
            expected.push((
                format!("turboquant.profile.{profile}.layer.{layer}.rotation.{view}"),
                vec![u64::from(head_dim), u64::from(head_dim)],
            ));
        }
    }
    for field in ["weights", "bias", "scale", "temperature"] {
        expected.push((
            format!("turboquant.profile.{profile}.consensus.{field}"),
            vec![3, layers],
        ));
    }
    if ncka.enabled {
        let coordinate_count = ncka.coordinate_names.len() as u64;
        let outer_count = u64::from(ncka.outer_count);
        let knot_count = u64::from(ncka.knot_count);
        for (field, shape) in [
            ("coordinate_min", vec![coordinate_count]),
            ("coordinate_max", vec![coordinate_count]),
            (
                "inner_knots",
                vec![knot_count, coordinate_count, outer_count, 3],
            ),
            (
                "inner_values",
                vec![knot_count, coordinate_count, outer_count, 3],
            ),
            ("outer_knots", vec![knot_count, outer_count, 3]),
            ("outer_values", vec![knot_count, outer_count, 3]),
            ("fallback_weights", vec![3]),
        ] {
            expected.push((format!("turboquant.profile.{profile}.ncka.{field}"), shape));
        }
    }
    let expected_names = expected
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    let manifest_names = manifest.keys().map(String::as_str).collect::<BTreeSet<_>>();
    anyhow::ensure!(
        expected_names == manifest_names,
        "Triality tensor manifest key set does not match the required schema-v2 tensors"
    );
    let tensor_names = gguf
        .tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("turboquant."))
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        tensor_names == expected_names,
        "Triality GGUF tensor headers do not match the schema-v2 manifest"
    );

    for (name, shape) in &expected {
        let entry = json_object(
            manifest
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Triality tensor manifest is missing `{name}`"))?,
            name,
        )?;
        exact_json_keys(entry, &["dtype", "shape", "sha256"], name)?;
        anyhow::ensure!(
            json_string(entry, "dtype", name)? == "f32",
            "Triality tensor manifest `{name}` must declare f32"
        );
        anyhow::ensure!(
            manifest_shape(entry, name)? == *shape,
            "Triality tensor manifest `{name}` has the wrong shape"
        );
        let hash = json_string(entry, "sha256", name)?;
        anyhow::ensure!(
            valid_sha256(&hash),
            "Triality tensor manifest `{name}` requires a lowercase SHA256"
        );
        let tensor = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == *name)
            .ok_or_else(|| anyhow::anyhow!("Triality tensor `{name}` is absent from GGUF"))?;
        anyhow::ensure!(
            tensor.dtype == GgmlType::F32 && tensor.dimensions == *shape,
            "Triality tensor `{name}` must be F32 with the manifest shape"
        );
    }

    if ncka.enabled {
        let marker = format!(".profile.{profile}.ncka.");
        let controller_manifest = manifest
            .iter()
            .filter(|(name, _)| name.contains(&marker))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<Map<_, _>>();
        anyhow::ensure!(
            canonical_sha256(&Value::Object(controller_manifest))? == ncka.controller_sha256,
            "NC-KA controller manifest hash mismatch"
        );
    }
    Ok(())
}

fn parse_strict_gguf_turboquant_config_v2(
    gguf: &GgufFile,
    metadata: &ModelMetadata,
) -> anyhow::Result<GgufTurboQuantConfig> {
    validate_v2_namespace_keys(gguf)?;
    validate_v2_base_types(gguf, metadata)?;
    validate_artifact_path(gguf)?;
    let payload_json = exact_string(gguf, "hypura.turboquant.payload_json")?;
    anyhow::ensure!(
        exact_u64(gguf, "hypura.turboquant.payload_bytes")? == payload_json.len() as u64,
        "Triality payload_bytes does not match payload_json length"
    );
    let payload_value: Value = serde_json::from_str(&payload_json)
        .context("Triality schema-v2 payload_json must contain valid JSON")?;
    let payload = json_object(&payload_value, "payload")?;
    let (num_layers, head_dim) = validate_payload_header(payload, gguf, metadata)?;
    let mut config = parse_strict_gguf_turboquant_config(gguf, metadata, 1)?;
    anyhow::ensure!(
        config.layers.len() == num_layers as usize,
        "Triality schema-v2 layer count contradicts TurboQuant artifact metadata"
    );
    let rotation_policy_name = json_string(payload, "rotation_policy", "payload")?;
    let rotation_policy = RotationPolicy::from_str(&rotation_policy_name).ok_or_else(|| {
        anyhow::anyhow!("Unsupported Triality schema-v2 rotation_policy `{rotation_policy_name}`")
    })?;
    anyhow::ensure!(
        matches!(
            rotation_policy,
            RotationPolicy::RandomHaar
                | RotationPolicy::BlockSo8Learned
                | RotationPolicy::IdentityDev
        ),
        "Unsupported Triality schema-v2 rotation_policy `{rotation_policy_name}`"
    );
    let rotation_seed = json_u32(payload, "rotation_seed", "payload")?;
    anyhow::ensure!(
        config.layers.iter().all(|layer| {
            layer.rotation_policy == rotation_policy && layer.rotation_seed == rotation_seed
        }),
        "Triality payload rotation policy or seed contradicts layer metadata"
    );
    anyhow::ensure!(
        rotation_policy != RotationPolicy::IdentityDev || rotation_seed == 0,
        "identity_dev Triality rotations require rotation_seed=0"
    );
    let consensus = parse_v2_consensus(gguf, payload, &config.layers)?;
    let ncka = parse_v2_ncka(gguf, payload)?;
    let urt = parse_v2_urt(gguf, payload)?;
    validate_v2_tensor_manifest(gguf, payload, &consensus, &ncka, head_dim)?;
    config.schema_version = 2;
    config.mode = if consensus.execution == "residual_parity" {
        TurboQuantMode::TrialityResidualParity
    } else {
        TurboQuantMode::TrialityConsensus
    };
    config.public_mode_label = config.mode.as_str().to_string();
    config.rotation_policy = Some(rotation_policy);
    config.head_dim = head_dim;
    config.consensus = Some(consensus);
    config.ncka = Some(ncka);
    config.urt = Some(urt);
    Ok(config)
}

fn ffi_triality_view(view: &str) -> anyhow::Result<TrialityView> {
    match view {
        "vector" => Ok(TrialityView::Vector),
        "spinor_plus_proxy" => Ok(TrialityView::SpinorPlusProxy),
        "spinor_minus_proxy" => Ok(TrialityView::SpinorMinusProxy),
        other => anyhow::bail!("Unsupported Triality branch view `{other}`"),
    }
}

impl TryFrom<&GgufTurboQuantConfig> for TrialityContextConfig {
    type Error = anyhow::Error;

    fn try_from(config: &GgufTurboQuantConfig) -> Result<Self, Self::Error> {
        anyhow::ensure!(
            config.schema_version == 2,
            "TrialityContextConfig requires GGUF Triality schema version 2"
        );
        let consensus = config
            .consensus
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GGUF Triality schema-v2 consensus is absent"))?;
        let execution = match consensus.execution.as_str() {
            "single_view" => TrialityExecution::SingleView,
            "best_per_layer" => TrialityExecution::BestPerLayer,
            "attention_logit_consensus" => TrialityExecution::AttentionLogitConsensus,
            "residual_parity" => TrialityExecution::ResidualParity,
            other => anyhow::bail!("Unsupported Triality execution `{other}`"),
        };
        let layers = consensus
            .branches_by_layer
            .iter()
            .map(|branches| {
                let active_branch_mask = if execution == TrialityExecution::SingleView {
                    let index = branches
                        .iter()
                        .position(|branch| branch.weight > 1.0e-6)
                        .ok_or_else(|| anyhow::anyhow!("single_view layer has no active branch"))?;
                    1_u32 << index
                } else {
                    0b111
                };
                let branches = branches.clone().map(|branch| {
                    Ok(TrialityBranchConfig {
                        view: ffi_triality_view(&branch.view)?,
                        weight: branch.weight,
                        bias: branch.bias,
                        scale: branch.scale,
                        temperature: branch.temperature,
                        expected_error: branch.expected_error,
                        bits_per_channel: branch.bits_per_channel,
                    })
                });
                Ok(TrialityLayerConfig {
                    branches: branches
                        .into_iter()
                        .collect::<anyhow::Result<Vec<_>>>()?
                        .try_into()
                        .map_err(|_| {
                            anyhow::anyhow!("Triality layer must contain three branches")
                        })?,
                    active_branch_mask,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(TrialityContextConfig {
            schema_version: 2,
            execution,
            layers,
            required: consensus.required,
            trace_enabled: false,
            js_fallback_threshold: consensus.js_fallback_threshold,
            allow_identity_view_fallback: false,
        })
    }
}

pub fn read_gguf_turboquant_config(
    gguf: &GgufFile,
    metadata: &ModelMetadata,
) -> anyhow::Result<Option<GgufTurboQuantConfig>> {
    match gguf.metadata.get("hypura.turboquant.schema_version") {
        Some(GgufValue::Uint32(2)) => {
            return parse_strict_gguf_turboquant_config_v2(gguf, metadata).map(Some);
        }
        Some(GgufValue::Uint32(1)) | None => {}
        Some(GgufValue::Uint32(version)) => {
            anyhow::bail!("Unsupported public GGUF TurboQuant schema version {version}")
        }
        Some(_) => anyhow::bail!(
            "Public GGUF TurboQuant schema_version must be UINT32 and is invalid for strict parsing"
        ),
    }
    if let Some(schema_version) = gguf.get_u32("tq_schema_version") {
        return parse_strict_gguf_turboquant_config(gguf, metadata, schema_version).map(Some);
    }
    parse_legacy_gguf_turboquant_config(gguf, metadata)
}

pub fn auto_discover_sidecar_path(model_path: &Path, mode: TurboQuantMode) -> Option<PathBuf> {
    let suffix = mode.config_suffix()?;
    let candidate = model_path.with_extension(suffix);
    candidate.exists().then_some(candidate)
}

pub fn parse_sidecar_config(
    config_path: &Path,
    json: &str,
    _mode: TurboQuantMode,
) -> anyhow::Result<TurboQuantSidecarConfig> {
    let value: Value = serde_json::from_str(json).with_context(|| {
        format!(
            "TurboQuant sidecar {} must contain valid JSON",
            config_path.display()
        )
    })?;
    let schema_kind = detect_schema_kind(config_path, &value)?;

    Ok(match schema_kind {
        TurboQuantSchemaKind::Paper => {
            let config: PaperTurboQuantConfig =
                serde_json::from_value(value).with_context(|| {
                    format!(
                        "TurboQuant sidecar {} could not be parsed as a paper config",
                        config_path.display()
                    )
                })?;
            validate_paper_artifacts(&config, config_path)?;
            TurboQuantSidecarConfig::Paper(config)
        }
        TurboQuantSchemaKind::Research => {
            let mut config: ResearchTurboQuantConfig =
                serde_json::from_value(value).with_context(|| {
                    format!(
                        "TurboQuant sidecar {} could not be parsed as a research config",
                        config_path.display()
                    )
                })?;
            config.schema_kind = Some(TurboQuantSchemaKind::Research);
            TurboQuantSidecarConfig::Research(config)
        }
    })
}

fn detect_schema_kind(
    config_path: &Path,
    value: &serde_json::Value,
) -> anyhow::Result<TurboQuantSchemaKind> {
    let object = value.as_object().ok_or_else(|| {
        anyhow::anyhow!(
            "TurboQuant sidecar {} must be a JSON object",
            config_path.display()
        )
    })?;

    let (schema_key, raw_kind) = ["schema_kind", "kind", "schema"]
        .into_iter()
        .find_map(|key| object.get(key).map(|value| (key, value)))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "TurboQuant sidecar {} is missing `schema_kind` (expected `paper` or `research`)",
                config_path.display()
            )
        })?;

    let raw_kind = raw_kind.as_str().ok_or_else(|| {
        anyhow::anyhow!(
            "TurboQuant sidecar {} field `{schema_key}` must be a string",
            config_path.display()
        )
    })?;

    match raw_kind {
        "paper" => Ok(TurboQuantSchemaKind::Paper),
        "research" => Ok(TurboQuantSchemaKind::Research),
        other => Err(anyhow::anyhow!(
            "TurboQuant sidecar {} has unsupported schema kind `{other}` (expected `paper` or `research`)",
            config_path.display()
        )),
    }
}

fn validate_paper_artifacts(cfg: &PaperTurboQuantConfig, config_path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cfg.codec.trim().is_empty(),
        "TurboQuant sidecar {} field `codec` must not be empty",
        config_path.display()
    );
    anyhow::ensure!(
        cfg.num_layers > 0 && cfg.num_kv_heads > 0 && cfg.head_dim > 0,
        "TurboQuant sidecar {} layer/head dimensions must be greater than zero",
        config_path.display()
    );
    anyhow::ensure!(
        cfg.key_bits > 0.0,
        "TurboQuant sidecar {} field `key_bits` must be greater than zero",
        config_path.display()
    );
    anyhow::ensure!(
        !cfg.rotation.matrix.is_empty(),
        "TurboQuant sidecar {} must include a non-empty `rotation.matrix` artifact",
        config_path.display()
    );
    anyhow::ensure!(
        cfg.rotation.matrix.len() == (cfg.head_dim * cfg.head_dim) as usize,
        "TurboQuant sidecar {} rotation.matrix length={} does not match head_dim^2={}",
        config_path.display(),
        cfg.rotation.matrix.len(),
        cfg.head_dim * cfg.head_dim
    );
    anyhow::ensure!(
        !cfg.scalar_quantizer.centroids.is_empty(),
        "TurboQuant sidecar {} must include scalar quantizer centroids",
        config_path.display()
    );
    if let Some(residual_qjl) = &cfg.residual_qjl {
        anyhow::ensure!(
            residual_qjl.sketch_dim > 0,
            "TurboQuant sidecar {} residual_qjl.sketch_dim must be greater than zero",
            config_path.display()
        );
        if !residual_qjl.projections.is_empty() {
            anyhow::ensure!(
                residual_qjl.projections.len()
                    == residual_qjl.sketch_dim as usize * cfg.head_dim as usize,
                "TurboQuant sidecar {} residual_qjl.projections length={} does not match sketch_dim*head_dim={}",
                config_path.display(),
                residual_qjl.projections.len(),
                residual_qjl.sketch_dim as usize * cfg.head_dim as usize
            );
        }
    }

    Ok(())
}

fn validate_mode_schema(
    mode: TurboQuantMode,
    schema_kind: TurboQuantSchemaKind,
    config_path: &Path,
) -> anyhow::Result<()> {
    let expected = mode.expected_schema_kind().ok_or_else(|| {
        anyhow::anyhow!("TurboQuant validation requested for exact mode unexpectedly")
    })?;

    anyhow::ensure!(
        expected == schema_kind,
        "TurboQuant sidecar {} declares `{schema_kind}` schema, but mode `{mode}` requires `{expected}`",
        config_path.display()
    );

    Ok(())
}

fn validate_config_against_metadata(
    config: &TurboQuantSidecarConfig,
    metadata: &ModelMetadata,
    config_path: &Path,
) -> anyhow::Result<()> {
    match config {
        TurboQuantSidecarConfig::Paper(cfg) => {
            anyhow::ensure!(
                cfg.num_layers == metadata.num_layers as u64,
                "TurboQuant sidecar {} num_layers={} does not match model num_layers={}",
                config_path.display(),
                cfg.num_layers,
                metadata.num_layers
            );

            anyhow::ensure!(
                cfg.num_kv_heads == metadata.num_kv_heads as u64,
                "TurboQuant sidecar {} num_kv_heads={} does not match model num_kv_heads={}",
                config_path.display(),
                cfg.num_kv_heads,
                metadata.num_kv_heads
            );

            let model_head_dim = if metadata.num_heads > 0 {
                metadata.embedding_dim as u64 / metadata.num_heads as u64
            } else {
                0
            };
            anyhow::ensure!(
                cfg.head_dim == model_head_dim,
                "TurboQuant sidecar {} head_dim={} does not match model head_dim={}",
                config_path.display(),
                cfg.head_dim,
                model_head_dim
            );
        }
        TurboQuantSidecarConfig::Research(cfg) => {
            if let Some(schema_kind) = cfg.schema_kind {
                anyhow::ensure!(
                    schema_kind == TurboQuantSchemaKind::Research,
                    "TurboQuant sidecar {} research config must declare `research` schema",
                    config_path.display()
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::model::gguf::GgufValue;

    fn sample_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: "qwen3".into(),
            parameter_count: 9_000_000_000,
            context_length: 4096,
            quantization: Some("q4".into()),
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            embedding_dim: 16,
            vocab_size: 32_000,
            is_moe: false,
            num_experts: None,
            num_experts_used: None,
        }
    }

    fn sample_paper_json() -> String {
        serde_json::json!({
            "schema_kind": "paper",
            "codec": "turboquant-prod",
            "num_layers": 2,
            "num_kv_heads": 2,
            "head_dim": 4,
            "key_bits": 3.5,
            "value_bits": 3.5,
            "mixed_bits": 3.5,
            "value_exact": false,
            "rotation": {
                "kind": "matrix",
                "seed": 7,
                "matrix": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0
                ]
            },
            "scalar_quantizer": {
                "centroids": [-1.0, -0.25, 0.25, 1.0],
                "decision_boundaries": [-0.5, 0.0, 0.5]
            },
            "residual_qjl": {
                "sketch_dim": 2,
                "scale": 1.0,
                "seed": 123,
                "projections": [
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0
                ]
            },
            "protected_channels": [1],
            "outlier_channels": [1]
        })
        .to_string()
    }

    fn sample_gguf() -> GgufFile {
        GgufFile {
            version: 3,
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
            data_offset: 0,
        }
    }

    fn sample_weight_payload_json() -> String {
        serde_json::json!({
            "schema": "hypura.turboquant.weight.v1",
            "codec": "tq4_1s",
            "policy": "qwen35-config-i",
            "source_ftype": "q8_0",
            "protected_roles": [
                "embedding",
                "norm",
                "output_head",
                "recurrent_state"
            ],
            "protected_layers": [0, 1],
            "modality_scope": "text",
            "tensor_plan": {
                "blk.*.attn_q.weight": "tq4_1s",
                "blk.*.attn_k.weight": "tq4_1s",
                "blk.*.attn_v.weight": "tq4_1s",
                "blk.*.attn_output.weight": "tq4_1s",
                "blk.*.ffn_gate.weight": "tq4_1s",
                "blk.*.ffn_up.weight": "tq4_1s",
                "blk.*.ffn_down.weight": "q4_k"
            }
        })
        .to_string()
    }

    fn sample_gguf_with_weight_contract() -> GgufFile {
        let mut gguf = sample_gguf();
        let payload_json = sample_weight_payload_json();
        gguf.metadata
            .insert("hypura.turboquant.enabled".into(), GgufValue::Bool(true));
        gguf.metadata.insert(
            "hypura.turboquant.mode".into(),
            GgufValue::String("triality-proxy-so8-pareto".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.schema_version".into(),
            GgufValue::Uint32(1),
        );
        gguf.metadata.insert(
            "hypura.turboquant.runtime_mode".into(),
            GgufValue::String("research-kv-split".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.rotation_policy".into(),
            GgufValue::String("triality_vector".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.enabled".into(),
            GgufValue::Bool(true),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.codec".into(),
            GgufValue::String("tq4_1s".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.source_ftype".into(),
            GgufValue::String("q8_0".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.policy".into(),
            GgufValue::String("qwen35-config-i".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.protected_roles".into(),
            GgufValue::String(r#"["embedding","norm","output_head","recurrent_state"]"#.into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.protected_layers".into(),
            GgufValue::String("[0,1]".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.modality_scope".into(),
            GgufValue::String("text".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.payload_format".into(),
            GgufValue::String("json-inline-v1".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.payload_bytes".into(),
            GgufValue::Uint64(payload_json.len() as u64),
        );
        gguf.metadata.insert(
            "hypura.turboquant.weight.payload_json".into(),
            GgufValue::String(payload_json),
        );
        gguf
    }

    fn array(values: Vec<GgufValue>) -> GgufValue {
        GgufValue::Array(values)
    }

    #[test]
    fn paper_config_parse_and_validate() {
        let path = Path::new("model.turboquant_config.paper.json");
        let parsed =
            parse_sidecar_config(path, &sample_paper_json(), TurboQuantMode::PaperKeyOnly).unwrap();
        let metadata = sample_metadata();
        validate_config_against_metadata(&parsed, &metadata, path).unwrap();
        let paper = parsed.paper().unwrap();
        assert_eq!(paper.codec, "turboquant-prod");
        assert_eq!(paper.rotation.matrix.len(), 16);
        assert_eq!(paper.scalar_quantizer.centroids.len(), 4);
    }

    #[test]
    fn schema_mode_mismatch_fails() {
        let path = Path::new("model.turboquant_config.paper.json");
        let parsed =
            parse_sidecar_config(path, &sample_paper_json(), TurboQuantMode::PaperKeyOnly).unwrap();
        let err = validate_mode_schema(TurboQuantMode::ResearchKvSplit, parsed.schema_kind(), path)
            .unwrap_err();
        assert!(err.to_string().contains("requires `research`"));
    }

    #[test]
    fn exact_mode_ignores_sidecar() {
        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &sample_gguf(),
            TurboQuantMode::Exact,
            None,
            false,
        )
        .unwrap();
        assert_eq!(resolved.mode, TurboQuantMode::Exact);
        assert!(resolved.config.is_none());
    }

    #[test]
    fn gguf_triality_metadata_resolves_without_sidecar() {
        let mut gguf = sample_gguf();
        let payload_json = serde_json::json!({
            "schema_kind": "triality_gguf_payload",
            "schema_version": 1,
            "mode": "triality-so8-pareto"
        })
        .to_string();
        gguf.metadata
            .insert("hypura.turboquant.enabled".into(), GgufValue::Bool(true));
        gguf.metadata.insert(
            "hypura.turboquant.mode".into(),
            GgufValue::String("triality-so8-pareto".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.schema_version".into(),
            GgufValue::Uint32(1),
        );
        gguf.metadata.insert(
            "hypura.turboquant.rotation_policy".into(),
            GgufValue::String("triality_spinor_plus".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.rotation_seed".into(),
            GgufValue::Uint32(17),
        );
        gguf.metadata.insert(
            "hypura.turboquant.triality_mix".into(),
            GgufValue::Float32(0.75),
        );
        gguf.metadata
            .insert("hypura.turboquant.k_bits".into(), GgufValue::Float32(3.5));
        gguf.metadata
            .insert("hypura.turboquant.v_bits".into(), GgufValue::Float32(8.0));
        gguf.metadata.insert(
            "hypura.turboquant.paper_fidelity".into(),
            GgufValue::Bool(false),
        );
        gguf.metadata.insert(
            "hypura.turboquant.payload_format".into(),
            GgufValue::String("json-inline-v1".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.payload_bytes".into(),
            GgufValue::Uint64(payload_json.len() as u64),
        );
        gguf.metadata.insert(
            "hypura.turboquant.payload_json".into(),
            GgufValue::String(payload_json),
        );

        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &gguf,
            TurboQuantMode::ResearchKvSplit,
            None,
            false,
        )
        .unwrap();

        let gguf_cfg = resolved
            .gguf_metadata
            .expect("gguf metadata should be attached");
        assert_eq!(gguf_cfg.mode, TurboQuantMode::ResearchKvSplit);
        assert_eq!(gguf_cfg.public_mode_label, "triality-proxy-so8-pareto");
        assert_eq!(gguf_cfg.runtime_mode, "research-kv-split");
        assert_eq!(gguf_cfg.schema_version, 1);
        assert_eq!(
            gguf_cfg.rotation_policy,
            Some(RotationPolicy::TrialitySpinorPlus)
        );
        assert_eq!(gguf_cfg.triality_view.as_deref(), Some("spinor_plus_proxy"));
        assert_eq!(gguf_cfg.rotation_seed, 17);
        assert_eq!(gguf_cfg.payload_format.as_deref(), Some("json-inline-v1"));
        assert_eq!(
            gguf_cfg.payload_bytes,
            gguf_cfg.payload_json.as_ref().unwrap().len() as u64
        );
    }

    #[test]
    fn gguf_triality_metadata_overrides_cli_mode_defaults() {
        let mut gguf = sample_gguf();
        gguf.metadata
            .insert("hypura.turboquant.enabled".into(), GgufValue::Bool(true));
        gguf.metadata.insert(
            "hypura.turboquant.mode".into(),
            GgufValue::String("paper-faithful".into()),
        );
        gguf.metadata.insert(
            "hypura.turboquant.schema_version".into(),
            GgufValue::Uint32(1),
        );

        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &gguf,
            TurboQuantMode::ResearchKvSplit,
            None,
            false,
        )
        .unwrap();

        assert_eq!(resolved.mode, TurboQuantMode::PaperKeyOnly);
        assert!(resolved.config.is_none());
        assert!(resolved.source_path.is_none());
        assert_eq!(
            resolved
                .gguf_metadata
                .as_ref()
                .expect("gguf metadata should win")
                .public_mode_label,
            "paper-faithful"
        );
    }

    #[test]
    fn strict_gguf_turboquant_metadata_resolves_with_layer_arrays() {
        let mut gguf = sample_gguf();
        gguf.metadata
            .insert("tq_schema_version".into(), GgufValue::Uint32(1));
        gguf.metadata.insert(
            "tq_total_bits".into(),
            array(vec![GgufValue::Float32(3.5), GgufValue::Float32(3.5)]),
        );
        gguf.metadata.insert(
            "tq_runtime_bits_per_channel".into(),
            array(vec![GgufValue::Float32(3.25), GgufValue::Float32(3.25)]),
        );
        gguf.metadata.insert(
            "tq_stage1_effective_bits".into(),
            array(vec![GgufValue::Float32(2.25), GgufValue::Float32(2.25)]),
        );
        gguf.metadata.insert(
            "tq_qjl_bits".into(),
            array(vec![GgufValue::Uint32(1), GgufValue::Uint32(1)]),
        );
        gguf.metadata.insert(
            "tq_qjl_dim".into(),
            array(vec![GgufValue::Uint32(128), GgufValue::Uint32(128)]),
        );
        gguf.metadata.insert(
            "tq_rotation_policy".into(),
            array(vec![
                GgufValue::String("block_so8_learned".into()),
                GgufValue::String("block_so8_learned".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_rotation_seed".into(),
            array(vec![GgufValue::Uint32(17), GgufValue::Uint32(17)]),
        );
        gguf.metadata.insert(
            "tq_qjl_seed".into(),
            array(vec![GgufValue::Uint32(1), GgufValue::Uint32(1)]),
        );
        gguf.metadata.insert(
            "tq_triality_mode".into(),
            array(vec![
                GgufValue::String("triality_proxy".into()),
                GgufValue::String("triality_proxy".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_triality_view".into(),
            array(vec![
                GgufValue::String("vector".into()),
                GgufValue::String("vector".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_stage1_allocation_scheme".into(),
            array(vec![
                GgufValue::String("magnitude-topk".into()),
                GgufValue::String("magnitude-topk".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_stage1_bitwidth_payload_dtype".into(),
            array(vec![
                GgufValue::String("uint8".into()),
                GgufValue::String("uint8".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_norm_dtype".into(),
            array(vec![
                GgufValue::String("float32".into()),
                GgufValue::String("float32".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_sign_pack_format".into(),
            array(vec![
                GgufValue::String("int8_unpacked_binary".into()),
                GgufValue::String("int8_unpacked_binary".into()),
            ]),
        );

        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &gguf,
            TurboQuantMode::ResearchKvSplit,
            None,
            false,
        )
        .unwrap();

        let gguf_cfg = resolved
            .gguf_metadata
            .expect("strict gguf metadata should be attached");
        assert_eq!(gguf_cfg.schema_version, 1);
        assert_eq!(gguf_cfg.mode, TurboQuantMode::ResearchKvSplit);
        assert_eq!(
            gguf_cfg.rotation_policy,
            Some(RotationPolicy::TrialityVector)
        );
        assert_eq!(gguf_cfg.triality_mode.as_deref(), Some("triality_proxy"));
        assert_eq!(gguf_cfg.triality_view.as_deref(), Some("vector"));
        assert_eq!(gguf_cfg.layers.len(), 2);
        assert_eq!(gguf_cfg.layers[0].qjl_dim, 128);
        assert_eq!(gguf_cfg.layers[1].sign_pack_format, "int8_unpacked_binary");
    }

    #[test]
    fn resolves_sparse_strict_gguf_turboquant_metadata() {
        let mut gguf = GgufFile {
            version: 3,
            metadata: Default::default(),
            tensors: Vec::new(),
            data_offset: 0,
        };
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufValue::String("qwen35".into()),
        );
        gguf.metadata
            .insert("tq_schema_version".into(), GgufValue::Uint32(1));
        gguf.metadata.insert(
            "tq_total_bits".into(),
            array(vec![GgufValue::Float32(3.5), GgufValue::Float32(3.5)]),
        );
        gguf.metadata.insert(
            "tq_runtime_bits_per_channel".into(),
            array(vec![GgufValue::Float32(3.0), GgufValue::Float32(3.0)]),
        );
        gguf.metadata.insert(
            "tq_stage1_effective_bits".into(),
            array(vec![GgufValue::Float32(2.0), GgufValue::Float32(2.0)]),
        );
        gguf.metadata.insert(
            "tq_qjl_bits".into(),
            array(vec![GgufValue::Uint32(1), GgufValue::Uint32(1)]),
        );
        gguf.metadata.insert(
            "tq_qjl_dim".into(),
            array(vec![GgufValue::Uint32(128), GgufValue::Uint32(128)]),
        );
        gguf.metadata.insert(
            "tq_rotation_policy".into(),
            array(vec![
                GgufValue::String("block_so8_learned".into()),
                GgufValue::String("block_so8_learned".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_rotation_seed".into(),
            array(vec![GgufValue::Uint32(7), GgufValue::Uint32(9)]),
        );
        gguf.metadata.insert(
            "tq_qjl_seed".into(),
            array(vec![GgufValue::Uint32(1), GgufValue::Uint32(1)]),
        );
        gguf.metadata.insert(
            "tq_triality_mode".into(),
            array(vec![
                GgufValue::String("triality_proxy".into()),
                GgufValue::String("triality_proxy".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_triality_view".into(),
            array(vec![
                GgufValue::String("vector".into()),
                GgufValue::String("vector".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_stage1_allocation_scheme".into(),
            array(vec![
                GgufValue::String("magnitude-topk".into()),
                GgufValue::String("magnitude-topk".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_stage1_bitwidth_payload_dtype".into(),
            array(vec![
                GgufValue::String("uint8".into()),
                GgufValue::String("uint8".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_norm_dtype".into(),
            array(vec![
                GgufValue::String("float32".into()),
                GgufValue::String("float32".into()),
            ]),
        );
        gguf.metadata.insert(
            "tq_sign_pack_format".into(),
            array(vec![
                GgufValue::String("int8_unpacked_binary".into()),
                GgufValue::String("int8_unpacked_binary".into()),
            ]),
        );

        let mut metadata = sample_metadata();
        metadata.num_layers = 4;

        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &metadata,
            &gguf,
            TurboQuantMode::ResearchKvSplit,
            None,
            false,
        )
        .unwrap();

        let gguf_cfg = resolved
            .gguf_metadata
            .expect("sparse strict gguf metadata should be attached");
        assert_eq!(gguf_cfg.num_layers, 4);
        assert_eq!(gguf_cfg.layers.len(), 2);
        assert_eq!(gguf_cfg.layers[0].rotation_seed, 7);
        assert_eq!(gguf_cfg.layers[1].rotation_seed, 9);
    }

    #[test]
    fn gguf_weight_contract_parses_and_reports_contract_only_status() {
        let gguf = sample_gguf_with_weight_contract();
        let parsed = read_gguf_turboquant_config(&gguf, &sample_metadata())
            .unwrap()
            .expect("gguf metadata should parse");
        let weight = parsed.weight.expect("weight contract should parse");
        assert_eq!(weight.codec.as_deref(), Some("tq4_1s"));
        assert_eq!(weight.policy.as_deref(), Some("qwen35-config-i"));
        assert_eq!(
            weight.payload_schema.as_deref(),
            Some("hypura.turboquant.weight.v1")
        );
        assert!(weight.payload_valid);
        assert_eq!(weight.tensor_plan_entries, 7);
        assert_eq!(weight.runtime_status(), "contract-only");
        assert!(!weight.runtime_ready());
    }

    #[test]
    fn gguf_weight_contract_fails_closed_without_exact_fallback_flag() {
        let gguf = sample_gguf_with_weight_contract();
        let err = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &gguf,
            TurboQuantMode::ResearchKvSplit,
            None,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("runtime support is incomplete"));
        assert!(err.to_string().contains("--tq-allow-exact-fallback"));
    }

    #[test]
    fn gguf_weight_contract_can_exact_fallback_with_escape_hatch() {
        let gguf = sample_gguf_with_weight_contract();
        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &gguf,
            TurboQuantMode::ResearchKvSplit,
            None,
            true,
        )
        .unwrap();
        assert_eq!(resolved.mode, TurboQuantMode::Exact);
        assert!(resolved.gguf_metadata.is_none());
    }

    #[test]
    fn research_mode_missing_artifacts_fails_closed_by_default() {
        let err = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &sample_gguf(),
            TurboQuantMode::ResearchKvSplit,
            None,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("Fail-closed is active"));
    }

    #[test]
    fn research_mode_missing_artifacts_can_exact_fallback_with_escape_hatch() {
        let resolved = resolve_turboquant_config(
            Path::new("model.gguf"),
            &sample_metadata(),
            &sample_gguf(),
            TurboQuantMode::ResearchKvSplit,
            None,
            true,
        )
        .unwrap();
        assert_eq!(resolved.mode, TurboQuantMode::Exact);
        assert!(resolved.config.is_none());
        assert!(resolved.gguf_metadata.is_none());
    }
}
