use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::model::gguf::GgufFile;
use crate::model::metadata::ModelMetadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum TurboQuantMode {
    Exact,
    PaperKeyOnly,
    PaperFullKv,
    ResearchKvSplit,
}

impl TurboQuantMode {
    pub fn requires_sidecar(self) -> bool {
        !matches!(self, Self::Exact)
    }

    pub fn expected_schema_kind(self) -> Option<TurboQuantSchemaKind> {
        match self {
            Self::Exact => None,
            Self::PaperKeyOnly | Self::PaperFullKv => Some(TurboQuantSchemaKind::Paper),
            Self::ResearchKvSplit => Some(TurboQuantSchemaKind::Research),
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
        self.schema_kind
            .map(TurboQuantSchemaKind::as_str)
            .unwrap_or("none")
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
                TurboQuantMode::PaperKeyOnly | TurboQuantMode::ResearchKvSplit => "triality_vector",
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
) -> anyhow::Result<ResolvedTurboQuantConfig> {
    let gguf_metadata = read_gguf_turboquant_config(gguf, metadata)?;
    if let Some(gguf_metadata) = gguf_metadata {
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
                // NOTE: Exact fallback preserves safety when research artifacts are missing.
                // Broader "Triality-by-default without sidecar" alignment is tracked separately.
                if mode == TurboQuantMode::ResearchKvSplit {
                    tracing::warn!(
                        "No TurboQuant research sidecar or GGUF metadata found next to {}. Falling back to exact runtime.",
                        model_path.display()
                    );
                    return Ok(ResolvedTurboQuantConfig::exact());
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
    })
}

pub fn read_gguf_turboquant_config(
    gguf: &GgufFile,
    metadata: &ModelMetadata,
) -> anyhow::Result<Option<GgufTurboQuantConfig>> {
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
        )
        .unwrap();

        let gguf_cfg = resolved
            .gguf_metadata
            .expect("gguf metadata should be attached");
        assert_eq!(gguf_cfg.mode, TurboQuantMode::ResearchKvSplit);
        assert_eq!(gguf_cfg.public_mode_label, "triality-so8-pareto");
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
}
