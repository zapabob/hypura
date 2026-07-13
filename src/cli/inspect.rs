use std::path::Path;

use hypura::model::{
    elt_loop::EltLoopMetadata,
    gguf::GgufFile,
    metadata::ModelMetadata,
    tensor_role::TensorRole,
    turboquant_sidecar::{
        GgufTurboQuantConfig, TurboQuantMode, read_gguf_turboquant_config,
        resolve_turboquant_config,
    },
};
use sha2::{Digest, Sha256};

use super::fmt_util::{format_bytes, format_params, print_elt_loop_status};

pub fn run(
    model_path: &str,
    show_tensors: bool,
    turboquant_mode: Option<TurboQuantMode>,
    turboquant_config: Option<&str>,
    tq_allow_exact_fallback: bool,
) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    anyhow::ensure!(path.exists(), "Model file not found: {model_path}");

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "gguf" => inspect_gguf(
            path,
            show_tensors,
            turboquant_mode,
            turboquant_config,
            tq_allow_exact_fallback,
        ),
        "safetensors" => anyhow::bail!("Safetensors inspect not yet implemented"),
        _ => anyhow::bail!("Unsupported model format: .{ext}"),
    }
}

fn inspect_gguf(
    path: &Path,
    show_tensors: bool,
    turboquant_mode: Option<TurboQuantMode>,
    turboquant_config: Option<&str>,
    tq_allow_exact_fallback: bool,
) -> anyhow::Result<()> {
    let gguf = GgufFile::open(path)?;
    let metadata = ModelMetadata::from_gguf(&gguf)?;
    let elt_loop = EltLoopMetadata::from_gguf(&gguf);

    println!("Model: {}", path.display());
    println!("  Format: GGUF v{}", gguf.version);
    println!("  Architecture: {}", metadata.architecture);
    println!("  Parameters: {}", format_params(metadata.parameter_count));
    println!("  Layers: {}", metadata.num_layers);
    println!("  Embedding dim: {}", metadata.embedding_dim);
    println!("  Attention heads: {}", metadata.num_heads);
    println!("  KV heads: {}", metadata.num_kv_heads);
    println!("  Vocab size: {}", metadata.vocab_size);
    println!("  Context length: {}", metadata.context_length);
    if let Some(ref q) = metadata.quantization {
        println!("  Quantization: {q}");
    }
    if metadata.is_moe {
        println!(
            "  MoE: {} experts, {} active per token",
            metadata.num_experts.unwrap_or(0),
            metadata.num_experts_used.unwrap_or(0)
        );
    }
    println!(
        "  Total size: {:.1} GB",
        gguf.total_tensor_bytes() as f64 / (1u64 << 30) as f64
    );
    println!("  Tensors: {}", gguf.tensors.len());
    print_elt_loop_status(elt_loop.as_ref(), "  ");
    print_consensus_metadata(&gguf);

    if let Some(tq) = read_gguf_turboquant_config(&gguf, &metadata)? {
        for line in schema_v2_audit_lines(&tq)? {
            println!("    {line}");
        }
        println!("  Triality/TurboQuant:");
        println!("    Source: {}", tq.source_label());
        println!("    Public mode: {}", tq.public_mode_label);
        println!("    Runtime mode: {}", tq.runtime_mode);
        println!("    Schema version: {}", tq.schema_version);
        println!(
            "    Rotation: {} (seed={})",
            tq.rotation_policy
                .map(|policy| policy.as_str().to_string())
                .unwrap_or_else(|| "none".to_string()),
            tq.rotation_seed
        );
        println!(
            "    Triality view: {} (mix={:.3})",
            tq.triality_view.as_deref().unwrap_or("none"),
            tq.triality_mix.unwrap_or(0.0)
        );
        println!(
            "    Fidelity: paper_fidelity={} k_bits={:.3} v_bits={:.3}",
            tq.paper_fidelity, tq.k_bits, tq.v_bits
        );
        println!(
            "    Payload: format={} bytes={} inline_json={}",
            tq.payload_format.as_deref().unwrap_or("none"),
            tq.payload_bytes,
            if tq.payload_json.is_some() {
                "yes"
            } else {
                "no"
            }
        );
        if let Some(weight) = tq.weight.as_ref() {
            println!(
                "    Weight: codec={} policy={} status={} payload_valid={} tensor_plan_entries={}",
                weight.codec.as_deref().unwrap_or("none"),
                weight.policy.as_deref().unwrap_or("none"),
                weight.runtime_status(),
                weight.payload_valid,
                weight.tensor_plan_entries,
            );
        }
    }

    if let Some(mode) = turboquant_mode {
        let resolved = resolve_turboquant_config(
            path,
            &metadata,
            &gguf,
            mode,
            turboquant_config.map(Path::new),
            tq_allow_exact_fallback,
        )?;
        println!("  TurboQuant validation:");
        println!("    Requested mode: {}", mode);
        println!("    Resolved mode: {}", resolved.mode);
        println!("    Schema: {}", resolved.schema_label());
        println!("    Source: {}", resolved.source_label());
    }

    if show_tensors {
        println!();
        println!(
            "  {:50} {:>12} {:>8} {:>10}",
            "Name", "Size", "Type", "Role"
        );
        println!("  {}", "-".repeat(84));
        for t in &gguf.tensors {
            let role = TensorRole::from_name(&t.name);
            println!(
                "  {:50} {:>12} {:>8} {:>10}",
                t.name,
                format_bytes(t.size_bytes),
                format!("{:?}", t.dtype),
                format!("{role:?}"),
            );
        }
    }

    Ok(())
}

fn print_consensus_metadata(gguf: &GgufFile) {
    let u32_keys = [
        "hypura.turboquant.schema_version",
        "hypura.turboquant.triality.view_count",
        "hypura.turboquant.ncka.schema_version",
        "hypura.turboquant.ncka.outer_count",
        "hypura.turboquant.ncka.knot_count",
        "hypura.turboquant.urt.schema_version",
        "hypura.turboquant.urt.moment_degree",
    ];
    let bool_keys = [
        "hypura.turboquant.triality.required",
        "hypura.turboquant.triality.override_allowed",
        "hypura.turboquant.ncka.enabled",
        "hypura.turboquant.ncka.required",
        "hypura.turboquant.ncka.s3_equivariant",
        "hypura.turboquant.urt.enabled",
    ];
    let f32_keys = [
        "hypura.turboquant.triality.js_fallback_threshold",
        "hypura.turboquant.urt.consistency_tolerance",
    ];
    let string_keys = [
        "hypura.turboquant.triality.profile_id",
        "hypura.turboquant.triality.execution",
        "hypura.turboquant.ncka.controller_type",
        "hypura.turboquant.ncka.controller_sha256",
        "hypura.turboquant.ncka.normalisation_sha256",
        "hypura.turboquant.urt.abstract_algebra_id",
        "hypura.turboquant.urt.operator_word_sha256",
        "hypura.turboquant.urt.reference_representation",
        "hypura.turboquant.urt.moment_manifest_sha256",
    ];
    let string_array_keys = [
        "hypura.turboquant.triality.views",
        "hypura.turboquant.ncka.coordinate_names",
        "hypura.turboquant.urt.supported_representations",
    ];
    let f32_array_keys = [
        "hypura.turboquant.triality.weights",
        "hypura.turboquant.triality.bias",
        "hypura.turboquant.triality.scale",
        "hypura.turboquant.triality.temperature",
    ];

    let present = u32_keys.iter().any(|key| gguf.get_u32(key).is_some())
        || bool_keys.iter().any(|key| gguf.get_bool(key).is_some())
        || f32_keys.iter().any(|key| gguf.get_f32(key).is_some())
        || string_keys.iter().any(|key| gguf.get_string(key).is_some())
        || string_array_keys
            .iter()
            .any(|key| gguf.get_string_array(key).is_some())
        || f32_array_keys
            .iter()
            .any(|key| gguf.get_f32_array(key).is_some());
    if !present {
        return;
    }

    println!("  Consensus metadata:");
    for key in u32_keys {
        if let Some(value) = gguf.get_u32(key) {
            println!("    {key}: {value}");
        }
    }
    for key in bool_keys {
        if let Some(value) = gguf.get_bool(key) {
            println!("    {key}: {value}");
        }
    }
    for key in f32_keys {
        if let Some(value) = gguf.get_f32(key) {
            println!("    {key}: {value}");
        }
    }
    for key in string_keys {
        if let Some(value) = gguf.get_string(key) {
            println!("    {key}: {value}");
        }
    }
    for key in string_array_keys {
        if let Some(value) = gguf.get_string_array(key) {
            println!("    {key}: {value:?}");
        }
    }
    for key in f32_array_keys {
        if let Some(value) = gguf.get_f32_array(key) {
            println!("    {key}: {value:?}");
        }
    }
}

fn schema_v2_audit_lines(config: &GgufTurboQuantConfig) -> anyhow::Result<Vec<String>> {
    if config.schema_version != 2 {
        return Ok(Vec::new());
    }
    let consensus = config
        .consensus
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("schema-v2 consensus metadata is unavailable"))?;
    let mut lines = vec![
        "Schema-v2 consensus audit:".to_string(),
        format!("profile_id: {}", consensus.profile_id),
        format!("execution: {}", consensus.execution),
        format!("required: {}", consensus.required),
        format!("override_allowed: {}", consensus.override_allowed),
        format!("js_fallback_threshold: {}", consensus.js_fallback_threshold),
        format!("layer_count: {}", consensus.branches_by_layer.len()),
    ];
    for (layer, branches) in consensus.branches_by_layer.iter().enumerate() {
        lines.push(format!(
            "layer[{layer}] views={:?} weights={:?} bias={:?} scale={:?} temperature={:?} expected_error={:?} bits_per_channel={:?}",
            branches.iter().map(|branch| branch.view.as_str()).collect::<Vec<_>>(),
            branches.iter().map(|branch| branch.weight).collect::<Vec<_>>(),
            branches.iter().map(|branch| branch.bias).collect::<Vec<_>>(),
            branches.iter().map(|branch| branch.scale).collect::<Vec<_>>(),
            branches
                .iter()
                .map(|branch| branch.temperature)
                .collect::<Vec<_>>(),
            branches
                .iter()
                .map(|branch| branch.expected_error)
                .collect::<Vec<_>>(),
            branches
                .iter()
                .map(|branch| branch.bits_per_channel)
                .collect::<Vec<_>>(),
        ));
    }

    if let Some(ncka) = config.ncka.as_ref() {
        lines.extend([
            format!("ncka.enabled: {}", ncka.enabled),
            format!("ncka.required: {}", ncka.required),
            format!("ncka.schema_version: {}", ncka.schema_version),
            format!("ncka.controller_type: {}", ncka.controller_type),
            format!("ncka.coordinate_count: {}", ncka.coordinate_names.len()),
            format!("ncka.coordinate_names: {:?}", ncka.coordinate_names),
            format!("ncka.outer_count: {}", ncka.outer_count),
            format!("ncka.knot_count: {}", ncka.knot_count),
            format!("ncka.s3_equivariant: {}", ncka.s3_equivariant),
            format!("ncka.controller_sha256: {}", ncka.controller_sha256),
            format!("ncka.normalisation_sha256: {}", ncka.normalisation_sha256),
            format!(
                "ncka.static_fallback_selected: {}",
                ncka.static_fallback_selected
            ),
            format!("ncka.fallback_weights: {:?}", ncka.fallback_weights),
        ]);
    }

    if let Some(urt) = config.urt.as_ref() {
        let operator_word_count =
            serde_json::from_str::<serde_json::Value>(&urt.operator_word_manifest)
                .ok()
                .and_then(|manifest| {
                    manifest
                        .get("words")
                        .and_then(|words| words.as_array())
                        .cloned()
                })
                .map(|words| words.len())
                .unwrap_or(0);
        lines.extend([
            format!("urt.enabled: {}", urt.enabled),
            format!("urt.schema_version: {}", urt.schema_version),
            format!("urt.abstract_algebra_id: {}", urt.abstract_algebra_id),
            format!("urt.operator_word_count: {operator_word_count}"),
            format!("urt.operator_word_sha256: {}", urt.operator_word_sha256),
            format!(
                "urt.reference_representation: {}",
                urt.reference_representation
            ),
            format!(
                "urt.supported_representations: {:?}",
                urt.supported_representations
            ),
            format!("urt.consistency_tolerance: {}", urt.consistency_tolerance),
            format!("urt.moment_degree: {}", urt.moment_degree),
            format!("urt.moment_manifest_sha256: {}", urt.moment_manifest_sha256),
        ]);
    }

    let payload_json = config
        .payload_json
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("schema-v2 payload JSON is unavailable"))?;
    lines.push(format!(
        "payload_sha256: {:x}",
        Sha256::digest(payload_json.as_bytes())
    ));
    let payload: serde_json::Value = serde_json::from_str(payload_json)?;
    let tensor_manifest = payload
        .get("tensor_manifest")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("schema-v2 tensor manifest is unavailable"))?;
    let manifest_value = serde_json::Value::Object(tensor_manifest.clone());
    lines.push(format!(
        "tensor_manifest_sha256: {:x}",
        Sha256::digest(serde_json::to_vec(&manifest_value)?)
    ));
    lines.push(format!(
        "tensor_manifest_entries: {}",
        tensor_manifest.len()
    ));
    for (name, entry) in tensor_manifest {
        let sha256 = entry
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("tensor manifest entry `{name}` has no SHA256"))?;
        lines.push(format!("tensor_sha256[{name}]: {sha256}"));
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hypura::model::turboquant_sidecar::{
        GgufNcKaConfig, GgufTrialityBranchConfig, GgufTrialityConsensusConfig, GgufUrtConfig,
    };
    use serde_json::json;

    #[test]
    fn schema_v2_audit_exposes_profile_contract_and_every_manifest_hash() {
        let tensor_a_hash = "a".repeat(64);
        let tensor_b_hash = "b".repeat(64);
        let payload = json!({
            "tensor_manifest": {
                "tensor.a": {"sha256": tensor_a_hash.clone()},
                "tensor.b": {"sha256": tensor_b_hash.clone()},
            }
        });
        let payload_json = serde_json::to_string(&payload).unwrap();
        let branch = GgufTrialityBranchConfig {
            view: "vector".to_string(),
            weight: 1.0,
            bias: 0.0,
            scale: 1.0,
            temperature: 1.0,
            expected_error: 0.125,
            bits_per_channel: 4.5,
        };
        let config = GgufTurboQuantConfig {
            enabled: true,
            schema_version: 2,
            mode: TurboQuantMode::TrialityConsensus,
            public_mode_label: "triality-consensus".to_string(),
            runtime_mode: "key_only".to_string(),
            rotation_policy: None,
            triality_view: Some("vector".to_string()),
            triality_mode: Some("triality".to_string()),
            triality_mix: Some(1.0),
            paper_fidelity: false,
            k_bits: 4.0,
            v_bits: 8.0,
            payload_format: Some("json-inline-v2".to_string()),
            payload_bytes: payload_json.len() as u64,
            payload_json: Some(payload_json),
            rotation_seed: 0,
            artifact_path: None,
            head_dim: 128,
            num_layers: 1,
            num_kv_heads: 8,
            layers: Vec::new(),
            weight: None,
            consensus: Some(GgufTrialityConsensusConfig {
                profile_id: "audit-profile".to_string(),
                execution: "residual_parity".to_string(),
                branches_by_layer: vec![[branch.clone(), branch.clone(), branch]],
                js_fallback_threshold: 0.2,
                required: true,
                override_allowed: false,
            }),
            ncka: Some(GgufNcKaConfig {
                enabled: true,
                required: false,
                schema_version: 1,
                controller_type: "finite_moment_ka_v1".to_string(),
                coordinate_names: vec!["entropy".to_string()],
                outer_count: 2,
                knot_count: 3,
                s3_equivariant: true,
                controller_sha256: "c".repeat(64),
                normalisation_sha256: "d".repeat(64),
                static_fallback_selected: false,
                fallback_weights: [1.0, 0.0, 0.0],
            }),
            urt: Some(GgufUrtConfig {
                enabled: true,
                schema_version: 1,
                abstract_algebra_id: "octonion_triality_proxy_v1".to_string(),
                operator_word_manifest: json!({"words": ["Q", "U"]}).to_string(),
                operator_word_sha256: "e".repeat(64),
                reference_representation: "hypura_native".to_string(),
                supported_representations: vec!["hypura_native".to_string()],
                consistency_tolerance: 1.0e-5,
                moment_degree: 4,
                moment_manifest_sha256: "f".repeat(64),
            }),
        };

        let output = schema_v2_audit_lines(&config).unwrap().join("\n");
        assert!(output.contains("profile_id: audit-profile"));
        assert!(output.contains("execution: residual_parity"));
        assert!(output.contains("expected_error=[0.125, 0.125, 0.125]"));
        assert!(output.contains("ncka.controller_sha256:"));
        assert!(output.contains("urt.operator_word_count: 2"));
        assert!(output.contains("payload_sha256:"));
        assert!(output.contains("tensor_manifest_sha256:"));
        assert!(output.contains(&format!("tensor_sha256[tensor.a]: {tensor_a_hash}")));
        assert!(output.contains(&format!("tensor_sha256[tensor.b]: {tensor_b_hash}")));
    }
}
