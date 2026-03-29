use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
}

impl ResolvedTurboQuantConfig {
    pub fn exact() -> Self {
        Self {
            mode: TurboQuantMode::Exact,
            schema_kind: None,
            source_path: None,
            config: None,
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
            .unwrap_or_else(|| "none".to_string())
    }

    pub fn paper_config(&self) -> Option<&PaperTurboQuantConfig> {
        self.config.as_ref()?.paper()
    }
}

pub fn resolve_turboquant_config(
    model_path: &Path,
    metadata: &ModelMetadata,
    mode: TurboQuantMode,
    explicit_config: Option<&Path>,
) -> anyhow::Result<ResolvedTurboQuantConfig> {
    if mode == TurboQuantMode::Exact {
        return Ok(ResolvedTurboQuantConfig::exact());
    }

    let config_path = match explicit_config {
        Some(path) => path.to_path_buf(),
        None => auto_discover_sidecar_path(model_path, mode).ok_or_else(|| {
            anyhow::anyhow!(
                "No TurboQuant sidecar found for mode `{mode}` next to {}. \
                 Pass `--turboquant-config <path>` or place a matching sidecar beside the model.",
                model_path.display()
            )
        })?,
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
    })
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
    use super::*;

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
            TurboQuantMode::Exact,
            None,
        )
        .unwrap();
        assert_eq!(resolved.mode, TurboQuantMode::Exact);
        assert!(resolved.config.is_none());
    }
}
