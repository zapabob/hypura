use serde::{Deserialize, Serialize};

use crate::model::gguf::GgufFile;

pub const HYPURA_ELT_LOOP_RUNTIME_SUPPORTED_ENV: &str = "HYPURA_ELT_LOOP_RUNTIME_SUPPORTED";
pub const LLAMA_ELT_LOOP_RUNTIME_SUPPORTED_ENV: &str = "LLAMA_ELT_LOOP_RUNTIME_SUPPORTED";

/// ELT loop metadata carried in GGUF files produced by the zapabob conversion
/// path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EltLoopMetadata {
    pub enabled: bool,
    pub required: bool,
    pub l_min: u32,
    pub l_max: u32,
    pub l_default: u32,
    pub loop_unit: Option<String>,
    pub gguf_architecture: Option<String>,
    pub gguf_runtime_status: Option<String>,
    pub model_family: Option<String>,
    pub loop_model_family: Option<String>,
}

impl EltLoopMetadata {
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        if !gguf.metadata.keys().any(|key| key.starts_with("elt.")) {
            return None;
        }

        let required = get_bool_any(gguf, &["elt.loop.required"]).unwrap_or(false);
        let l_min = get_u32_any(gguf, &["elt.loop.L_min", "elt.loop.min"])
            .filter(|value| *value > 0)
            .unwrap_or(if required { 2 } else { 1 });
        let l_default = get_u32_any(gguf, &["elt.loop.L_default", "elt.loop.default"])
            .filter(|value| *value > 0)
            .unwrap_or(l_min)
            .max(1);
        let l_max = get_u32_any(gguf, &["elt.loop.L_max", "elt.loop.max"])
            .filter(|value| *value > 0)
            .unwrap_or(l_default.max(l_min))
            .max(l_default)
            .max(l_min);
        let model_family = get_string_any(gguf, &["elt.model_family"]).map(str::to_string);
        let loop_model_family =
            get_string_any(gguf, &["elt.loop.model_family"]).map(str::to_string);
        let family_marks_looped = is_looped_qwen35_family(model_family.as_deref())
            || is_looped_qwen35_family(loop_model_family.as_deref());
        let enabled = get_bool_any(gguf, &["elt.loop.enabled"])
            .unwrap_or(required || l_min > 1 || l_default > 1 || l_max > 1 || family_marks_looped);

        Some(Self {
            enabled,
            required,
            l_min,
            l_max,
            l_default,
            loop_unit: get_string_any(gguf, &["elt.loop_unit"]).map(str::to_string),
            gguf_architecture: get_string_any(gguf, &["elt.gguf.architecture"]).map(str::to_string),
            gguf_runtime_status: get_string_any(gguf, &["elt.gguf.runtime_status"])
                .map(str::to_string),
            model_family,
            loop_model_family,
        })
    }

    pub fn requires_looped_runtime(&self) -> bool {
        if !self.enabled {
            return false;
        }
        self.required
            || self.l_min > 1
            || self.l_default > 1
            || is_looped_qwen35_family(self.model_family.as_deref())
            || is_looped_qwen35_family(self.loop_model_family.as_deref())
            || self
                .gguf_runtime_status
                .as_deref()
                .map(|status| status.eq_ignore_ascii_case("requires_looped_qwen35_runtime"))
                .unwrap_or(false)
    }

    pub fn runtime_gate_label(&self, runtime_supported: bool) -> &'static str {
        if !self.enabled {
            "inactive"
        } else if self.requires_looped_runtime() && runtime_supported {
            "loop-aware-runtime-declared"
        } else if self.requires_looped_runtime() {
            "blocked-missing-loop-aware-runtime"
        } else {
            "metadata-only"
        }
    }

    pub fn family_label(&self) -> &str {
        self.loop_model_family
            .as_deref()
            .or(self.model_family.as_deref())
            .unwrap_or("unknown")
    }

    pub fn loop_unit_label(&self) -> &str {
        self.loop_unit.as_deref().unwrap_or("decoder_layer")
    }

    pub fn ensure_runtime_supported(&self, runtime_supported: bool) -> anyhow::Result<()> {
        if !self.requires_looped_runtime() || runtime_supported {
            return Ok(());
        }

        anyhow::bail!(
            "GGUF requires loop-aware ELT runtime (elt.loop.required={}, L_min={}, L_default={}, L_max={}, unit={}, family={}, runtime_status={}). Hypura will not run it as an L=1-compatible model. Use a verified zapabob/llama.cpp build with loop-aware decode/graph support, then set {}=1 for that verified runtime.",
            self.required,
            self.l_min,
            self.l_default,
            self.l_max,
            self.loop_unit_label(),
            self.family_label(),
            self.gguf_runtime_status.as_deref().unwrap_or("unknown"),
            HYPURA_ELT_LOOP_RUNTIME_SUPPORTED_ENV,
        )
    }

    pub fn llama_env_pairs(&self) -> Vec<(&'static str, String)> {
        let mut pairs = vec![
            (
                "LLAMA_ELT_LOOP_ENABLED",
                if self.enabled { "1" } else { "0" }.to_string(),
            ),
            (
                "LLAMA_ELT_LOOP_REQUIRED",
                if self.required { "1" } else { "0" }.to_string(),
            ),
            ("LLAMA_ELT_LOOP_L_MIN", self.l_min.to_string()),
            ("LLAMA_ELT_LOOP_L_MAX", self.l_max.to_string()),
            ("LLAMA_ELT_LOOP_L_DEFAULT", self.l_default.to_string()),
            ("LLAMA_ELT_LOOP_UNIT", self.loop_unit_label().to_string()),
        ];

        if let Some(family) = self
            .loop_model_family
            .as_deref()
            .or(self.model_family.as_deref())
            .filter(|value| !value.trim().is_empty())
        {
            pairs.push(("LLAMA_ELT_LOOP_MODEL_FAMILY", family.trim().to_string()));
        }
        if let Some(status) = self
            .gguf_runtime_status
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            pairs.push(("LLAMA_ELT_GGUF_RUNTIME_STATUS", status.trim().to_string()));
        }
        pairs
    }
}

pub fn elt_loop_runtime_supported_from_env() -> bool {
    env_flag(HYPURA_ELT_LOOP_RUNTIME_SUPPORTED_ENV)
        || env_flag(LLAMA_ELT_LOOP_RUNTIME_SUPPORTED_ENV)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn get_bool_any(gguf: &GgufFile, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| gguf.get_metadata(key).and_then(|value| value.as_bool()))
}

fn get_u32_any(gguf: &GgufFile, keys: &[&str]) -> Option<u32> {
    keys.iter()
        .find_map(|key| gguf.get_metadata(key).and_then(|value| value.as_u32()))
}

fn get_string_any<'a>(gguf: &'a GgufFile, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| gguf.get_metadata(key).and_then(|value| value.as_str()))
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn is_looped_qwen35_family(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };
    let normalized = value.trim().to_ascii_lowercase().replace('\\', "/");
    normalized == "elt/qwen3.5-looped"
        || (normalized.contains("elastic-looped-transformer") && normalized.contains("qwen3.5"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::model::gguf::{GgufFile, GgufValue};

    use super::*;

    fn gguf(metadata: BTreeMap<String, GgufValue>) -> GgufFile {
        GgufFile {
            version: 3,
            metadata,
            tensors: vec![],
            data_offset: 0,
        }
    }

    #[test]
    fn missing_elt_metadata_returns_none() {
        assert!(EltLoopMetadata::from_gguf(&gguf(BTreeMap::new())).is_none());
    }

    #[test]
    fn parses_loop_required_metadata_and_blocks_without_runtime_support() {
        let mut metadata = BTreeMap::new();
        metadata.insert("elt.loop.enabled".into(), GgufValue::Bool(true));
        metadata.insert("elt.loop.required".into(), GgufValue::Bool(true));
        metadata.insert("elt.loop.L_min".into(), GgufValue::Uint32(2));
        metadata.insert("elt.loop.L_max".into(), GgufValue::Uint32(6));
        metadata.insert("elt.loop.L_default".into(), GgufValue::Uint32(4));
        metadata.insert(
            "elt.loop_unit".into(),
            GgufValue::String("decoder_layer".into()),
        );
        metadata.insert(
            "elt.gguf.runtime_status".into(),
            GgufValue::String("requires_looped_qwen35_runtime".into()),
        );
        metadata.insert(
            "elt.loop.model_family".into(),
            GgufValue::String("ELT/Qwen3.5-looped".into()),
        );

        let parsed = EltLoopMetadata::from_gguf(&gguf(metadata)).expect("elt metadata");

        assert!(parsed.requires_looped_runtime());
        assert_eq!(parsed.l_min, 2);
        assert_eq!(parsed.l_max, 6);
        assert_eq!(parsed.l_default, 4);
        assert_eq!(parsed.family_label(), "ELT/Qwen3.5-looped");
        assert_eq!(
            parsed.runtime_gate_label(false),
            "blocked-missing-loop-aware-runtime"
        );
        assert!(parsed.ensure_runtime_supported(false).is_err());
        assert!(parsed.ensure_runtime_supported(true).is_ok());
    }

    #[test]
    fn optional_loop_metadata_can_describe_plain_l1_runtime() {
        let mut metadata = BTreeMap::new();
        metadata.insert("elt.loop.enabled".into(), GgufValue::Bool(true));
        metadata.insert("elt.loop.required".into(), GgufValue::Bool(false));
        metadata.insert("elt.loop.L_min".into(), GgufValue::Uint32(1));
        metadata.insert("elt.loop.L_max".into(), GgufValue::Uint32(4));
        metadata.insert("elt.loop.L_default".into(), GgufValue::Uint32(1));

        let parsed = EltLoopMetadata::from_gguf(&gguf(metadata)).expect("elt metadata");

        assert!(!parsed.requires_looped_runtime());
        assert_eq!(parsed.runtime_gate_label(false), "metadata-only");
        assert!(parsed.ensure_runtime_supported(false).is_ok());
    }

    #[test]
    fn emits_llama_runtime_env_pairs() {
        let mut metadata = BTreeMap::new();
        metadata.insert("elt.loop.required".into(), GgufValue::Bool(true));
        metadata.insert("elt.loop.default".into(), GgufValue::Uint32(3));
        metadata.insert(
            "elt.model_family".into(),
            GgufValue::String("elastic-looped-transformer/qwen3.5-4b".into()),
        );

        let parsed = EltLoopMetadata::from_gguf(&gguf(metadata)).expect("elt metadata");
        let pairs = parsed.llama_env_pairs();

        assert!(pairs.contains(&("LLAMA_ELT_LOOP_REQUIRED", "1".to_string())));
        assert!(pairs.contains(&("LLAMA_ELT_LOOP_L_DEFAULT", "3".to_string())));
        assert!(pairs.contains(&(
            "LLAMA_ELT_LOOP_MODEL_FAMILY",
            "elastic-looped-transformer/qwen3.5-4b".to_string()
        )));
    }
}
