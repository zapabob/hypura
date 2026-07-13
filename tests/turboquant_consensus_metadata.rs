use std::collections::BTreeMap;

use hypura::compute::ffi::{TrialityContextConfig, TrialityExecution};
use hypura::model::gguf::{GgmlType, GgufFile, GgufValue, TensorInfo};
use hypura::model::metadata::ModelMetadata;
use hypura::model::turboquant_sidecar::{
    ResolvedTurboQuantConfig, TurboQuantMode, read_gguf_turboquant_config,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const VIEWS: [&str; 3] = ["vector", "spinor_plus_proxy", "spinor_minus_proxy"];
const COORDINATES: [&str; 24] = [
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

fn sha256_json(value: &Value) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn metadata() -> ModelMetadata {
    ModelMetadata {
        architecture: "llama".to_string(),
        parameter_count: 0,
        context_length: 128,
        embedding_dim: 8,
        num_layers: 2,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 32,
        quantization: Some("Q4_0".to_string()),
        is_moe: false,
        num_experts: None,
        num_experts_used: None,
    }
}

fn f32_array(values: impl IntoIterator<Item = f32>) -> GgufValue {
    GgufValue::Array(values.into_iter().map(GgufValue::Float32).collect())
}

fn u32_array(values: impl IntoIterator<Item = u32>) -> GgufValue {
    GgufValue::Array(values.into_iter().map(GgufValue::Uint32).collect())
}

fn string_array(values: impl IntoIterator<Item = impl Into<String>>) -> GgufValue {
    GgufValue::Array(
        values
            .into_iter()
            .map(|value| GgufValue::String(value.into()))
            .collect(),
    )
}

fn tensor(name: String, dimensions: Vec<u64>) -> TensorInfo {
    TensorInfo {
        name,
        dimensions,
        dtype: GgmlType::F32,
        offset: 0,
        size_bytes: 0,
        layer_index: None,
    }
}

fn manifest_entry(shape: &[u64], fill: char) -> Value {
    json!({
        "dtype": "f32",
        "shape": shape,
        "sha256": fill.to_string().repeat(64),
    })
}

fn strict_base_metadata() -> BTreeMap<String, GgufValue> {
    BTreeMap::from([
        ("tq_schema_version".to_string(), GgufValue::Uint32(1)),
        ("tq_total_bits".to_string(), f32_array([3.5, 3.5])),
        (
            "tq_runtime_bits_per_channel".to_string(),
            f32_array([3.5, 3.5]),
        ),
        (
            "tq_stage1_effective_bits".to_string(),
            f32_array([2.5, 2.5]),
        ),
        ("tq_qjl_bits".to_string(), u32_array([1, 1])),
        ("tq_qjl_dim".to_string(), u32_array([8, 8])),
        (
            "tq_rotation_policy".to_string(),
            string_array(["block_so8_learned", "block_so8_learned"]),
        ),
        ("tq_rotation_seed".to_string(), u32_array([0, 0])),
        ("tq_qjl_seed".to_string(), u32_array([1, 1])),
        (
            "tq_triality_mode".to_string(),
            string_array([
                "key_only_block_so8_triality_vector",
                "key_only_block_so8_triality_vector",
            ]),
        ),
        (
            "tq_triality_view".to_string(),
            string_array(["vector", "vector"]),
        ),
        (
            "tq_stage1_allocation_scheme".to_string(),
            string_array(["magnitude-topk", "magnitude-topk"]),
        ),
        (
            "tq_stage1_bitwidth_payload_dtype".to_string(),
            string_array(["uint8", "uint8"]),
        ),
        (
            "tq_norm_dtype".to_string(),
            string_array(["float32", "float32"]),
        ),
        (
            "tq_sign_pack_format".to_string(),
            string_array(["int8_unpacked_binary", "int8_unpacked_binary"]),
        ),
    ])
}

fn schema_v2_gguf(enable_ncka: bool, enable_urt: bool) -> GgufFile {
    let profile = "v2";
    let mut tensors = Vec::new();
    let mut manifest = Map::new();
    for layer in 0..2 {
        for (branch, view) in VIEWS.into_iter().enumerate() {
            let name = format!("turboquant.profile.{profile}.layer.{layer}.rotation.{view}");
            tensors.push(tensor(name.clone(), vec![8, 8]));
            manifest.insert(
                name,
                manifest_entry(&[8, 8], char::from(b'a' + branch as u8)),
            );
        }
    }
    for (field, fill) in [
        ("weights", 'd'),
        ("bias", 'e'),
        ("scale", 'f'),
        ("temperature", '0'),
    ] {
        let name = format!("turboquant.profile.{profile}.consensus.{field}");
        tensors.push(tensor(name.clone(), vec![3, 2]));
        manifest.insert(name, manifest_entry(&[3, 2], fill));
    }

    let coordinate_names = COORDINATES.map(str::to_string).to_vec();
    let normalisation = json!({
        "coordinate_names": coordinate_names,
        "range": [0.0, 1.0],
        "clamp": true,
    });
    let mut ncka = json!({
        "enabled": false,
        "required": false,
        "schema_version": 0,
        "controller_type": "",
        "coordinate_names": [],
        "outer_count": 0,
        "knot_count": 0,
        "s3_equivariant": false,
        "fallback_policy": "static",
        "fallback_weights": [1.0, 0.0, 0.0],
        "normalisation_sha256": "",
        "controller_sha256": "",
    });
    if enable_ncka {
        for (field, shape, fill) in [
            ("coordinate_min", vec![24], '1'),
            ("coordinate_max", vec![24], '2'),
            ("inner_knots", vec![3, 24, 2, 3], '3'),
            ("inner_values", vec![3, 24, 2, 3], '4'),
            ("outer_knots", vec![3, 2, 3], '5'),
            ("outer_values", vec![3, 2, 3], '6'),
            ("fallback_weights", vec![3], '7'),
        ] {
            let name = format!("turboquant.profile.{profile}.ncka.{field}");
            tensors.push(tensor(name.clone(), shape.clone()));
            manifest.insert(name, manifest_entry(&shape, fill));
        }
        let ncka_manifest = manifest
            .iter()
            .filter(|(name, _)| name.contains(".ncka."))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<Map<_, _>>();
        ncka = json!({
            "enabled": true,
            "required": false,
            "schema_version": 1,
            "controller_type": "finite_moment_ka_v1",
            "coordinate_names": COORDINATES,
            "outer_count": 2,
            "knot_count": 3,
            "s3_equivariant": true,
            "fallback_policy": "static",
            "fallback_weights": [1.0, 0.0, 0.0],
            "normalisation_sha256": sha256_json(&normalisation),
            "controller_sha256": sha256_json(&Value::Object(ncka_manifest)),
        });
    }

    let operator_manifest = json!({
        "algebra_id": "octonion_triality_proxy_v1",
        "generators": ["e1", "e2", "e3"],
        "words": ["e1", "e2", "e3", "e1*e2", "e2*e3", "e3*e1"],
        "multiplication": "left_associative",
    });
    let moment_manifest = json!({
        "degree": 4,
        "moments": ["mean", "variance", "skewness", "kurtosis"],
    });
    let urt = if enable_urt {
        json!({
            "enabled": true,
            "schema_version": 1,
            "abstract_algebra_id": "octonion_triality_proxy_v1",
            "operator_word_manifest": operator_manifest,
            "operator_word_sha256": sha256_json(&operator_manifest),
            "reference_representation": "python_quantised_reference",
            "supported_representations": [
                "python_quantised_reference",
                "llama_cpu_gguf",
                "llama_cuda_gguf",
                "hypura_native",
                "hypura_kobold_worker"
            ],
            "consistency_tolerance": 1.0e-5,
            "moment_degree": 4,
            "moment_manifest_sha256": sha256_json(&moment_manifest),
        })
    } else {
        json!({
            "enabled": false,
            "schema_version": 0,
            "abstract_algebra_id": "",
            "operator_word_manifest": {},
            "operator_word_sha256": "",
            "reference_representation": "",
            "supported_representations": [],
            "consistency_tolerance": 0.0,
            "moment_degree": 0,
            "moment_manifest_sha256": "",
        })
    };
    let consensus = json!({
        "schema_version": 1,
        "execution": "attention_logit_consensus",
        "view_count": 3,
        "views": VIEWS,
        "rows": [
            {"layer": 0, "weights": [0.5, 0.3, 0.2], "bias": [0.0, 0.1, -0.1], "scale": [1.0, 1.1, 0.9], "temperature": [1.0, 0.95, 1.05]},
            {"layer": 1, "weights": [0.4, 0.4, 0.2], "bias": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "temperature": [1.0, 1.0, 1.0]}
        ],
        "js_fallback_threshold": 0.2,
        "fallback_policy": "static",
        "fallback_weights": [1.0, 0.0, 0.0],
    });
    let payload = json!({
        "schema_kind": "triality_gguf_payload",
        "schema_version": 2,
        "codec": "tq4_1s",
        "mode": "triality-proxy-so8-pareto",
        "model_family": "llama",
        "runtime_mode": "key_only_block_so8_triality_vector",
        "head_dim": 8,
        "num_layers": 2,
        "num_kv_heads": 1,
        "rotation_policy": "block_so8_learned",
        "rotation_block_size": 8,
        "rotation_seed": 0,
        "triality_view": "vector",
        "triality_mix": 0.5,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "view_bundle_complete": true,
        "orthogonality_error": 0.0,
        "determinant_error_max": 0.0,
        "paper_fidelity": false,
        "k_bits": 3.5,
        "v_bits": 8.0,
        "offline_metrics": {},
        "weight_plan": {},
        "pareto_profile": {},
        "profile_id": profile,
        "consensus": consensus,
        "ncka": ncka,
        "urt": urt,
        "tensor_manifest": manifest,
    });
    let payload_json = serde_json::to_string(&payload).unwrap();
    let mut metadata = strict_base_metadata();
    metadata.extend([
        (
            "hypura.turboquant.schema_version".to_string(),
            GgufValue::Uint32(2),
        ),
        (
            "hypura.turboquant.enabled".to_string(),
            GgufValue::Bool(true),
        ),
        (
            "hypura.turboquant.mode".to_string(),
            GgufValue::String("triality-proxy-so8-pareto".to_string()),
        ),
        (
            "hypura.turboquant.runtime_mode".to_string(),
            GgufValue::String("key_only_block_so8_triality_vector".to_string()),
        ),
        (
            "hypura.turboquant.payload_format".to_string(),
            GgufValue::String("json-inline-v2".to_string()),
        ),
        (
            "hypura.turboquant.payload_bytes".to_string(),
            GgufValue::Uint64(payload_json.len() as u64),
        ),
        (
            "hypura.turboquant.payload_json".to_string(),
            GgufValue::String(payload_json),
        ),
        (
            "hypura.turboquant.triality.profile_id".to_string(),
            GgufValue::String(profile.to_string()),
        ),
        (
            "hypura.turboquant.triality.execution".to_string(),
            GgufValue::String("attention_logit_consensus".to_string()),
        ),
        (
            "hypura.turboquant.triality.override_allowed".to_string(),
            GgufValue::Bool(false),
        ),
        (
            "hypura.turboquant.triality.view_count".to_string(),
            GgufValue::Uint32(3),
        ),
        (
            "hypura.turboquant.triality.views".to_string(),
            string_array(VIEWS),
        ),
        (
            "hypura.turboquant.triality.weights".to_string(),
            f32_array([0.5, 0.3, 0.2, 0.4, 0.4, 0.2]),
        ),
        (
            "hypura.turboquant.triality.bias".to_string(),
            f32_array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0]),
        ),
        (
            "hypura.turboquant.triality.scale".to_string(),
            f32_array([1.0, 1.1, 0.9, 1.0, 1.0, 1.0]),
        ),
        (
            "hypura.turboquant.triality.temperature".to_string(),
            f32_array([1.0, 0.95, 1.05, 1.0, 1.0, 1.0]),
        ),
        (
            "hypura.turboquant.triality.js_fallback_threshold".to_string(),
            GgufValue::Float32(0.2),
        ),
    ]);

    let ncka = payload["ncka"].as_object().unwrap();
    metadata.extend([
        (
            "hypura.turboquant.ncka.enabled".to_string(),
            GgufValue::Bool(ncka["enabled"].as_bool().unwrap()),
        ),
        (
            "hypura.turboquant.ncka.required".to_string(),
            GgufValue::Bool(ncka["required"].as_bool().unwrap()),
        ),
        (
            "hypura.turboquant.ncka.schema_version".to_string(),
            GgufValue::Uint32(ncka["schema_version"].as_u64().unwrap() as u32),
        ),
        (
            "hypura.turboquant.ncka.controller_type".to_string(),
            GgufValue::String(ncka["controller_type"].as_str().unwrap().to_string()),
        ),
        (
            "hypura.turboquant.ncka.coordinate_names".to_string(),
            string_array(
                ncka["coordinate_names"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_str().unwrap()),
            ),
        ),
        (
            "hypura.turboquant.ncka.outer_count".to_string(),
            GgufValue::Uint32(ncka["outer_count"].as_u64().unwrap() as u32),
        ),
        (
            "hypura.turboquant.ncka.knot_count".to_string(),
            GgufValue::Uint32(ncka["knot_count"].as_u64().unwrap() as u32),
        ),
        (
            "hypura.turboquant.ncka.s3_equivariant".to_string(),
            GgufValue::Bool(ncka["s3_equivariant"].as_bool().unwrap()),
        ),
        (
            "hypura.turboquant.ncka.controller_sha256".to_string(),
            GgufValue::String(ncka["controller_sha256"].as_str().unwrap().to_string()),
        ),
        (
            "hypura.turboquant.ncka.normalisation_sha256".to_string(),
            GgufValue::String(ncka["normalisation_sha256"].as_str().unwrap().to_string()),
        ),
    ]);

    let urt = payload["urt"].as_object().unwrap();
    metadata.extend([
        (
            "hypura.turboquant.urt.enabled".to_string(),
            GgufValue::Bool(urt["enabled"].as_bool().unwrap()),
        ),
        (
            "hypura.turboquant.urt.schema_version".to_string(),
            GgufValue::Uint32(urt["schema_version"].as_u64().unwrap() as u32),
        ),
        (
            "hypura.turboquant.urt.abstract_algebra_id".to_string(),
            GgufValue::String(urt["abstract_algebra_id"].as_str().unwrap().to_string()),
        ),
        (
            "hypura.turboquant.urt.operator_word_manifest".to_string(),
            GgufValue::String(serde_json::to_string(&urt["operator_word_manifest"]).unwrap()),
        ),
        (
            "hypura.turboquant.urt.operator_word_sha256".to_string(),
            GgufValue::String(urt["operator_word_sha256"].as_str().unwrap().to_string()),
        ),
        (
            "hypura.turboquant.urt.reference_representation".to_string(),
            GgufValue::String(
                urt["reference_representation"]
                    .as_str()
                    .unwrap()
                    .to_string(),
            ),
        ),
        (
            "hypura.turboquant.urt.supported_representations".to_string(),
            string_array(
                urt["supported_representations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_str().unwrap()),
            ),
        ),
        (
            "hypura.turboquant.urt.consistency_tolerance".to_string(),
            GgufValue::Float32(urt["consistency_tolerance"].as_f64().unwrap() as f32),
        ),
        (
            "hypura.turboquant.urt.moment_degree".to_string(),
            GgufValue::Uint32(urt["moment_degree"].as_u64().unwrap() as u32),
        ),
        (
            "hypura.turboquant.urt.moment_manifest_sha256".to_string(),
            GgufValue::String(urt["moment_manifest_sha256"].as_str().unwrap().to_string()),
        ),
    ]);
    GgufFile {
        version: 3,
        metadata,
        tensors,
        data_offset: 0,
    }
}

fn update_payload(gguf: &mut GgufFile, update: impl FnOnce(&mut Value)) {
    let mut payload: Value = serde_json::from_str(
        gguf.metadata["hypura.turboquant.payload_json"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    update(&mut payload);
    let encoded = serde_json::to_string(&payload).unwrap();
    gguf.metadata.insert(
        "hypura.turboquant.payload_bytes".to_string(),
        GgufValue::Uint64(encoded.len() as u64),
    );
    gguf.metadata.insert(
        "hypura.turboquant.payload_json".to_string(),
        GgufValue::String(encoded),
    );
}

#[test]
fn parses_complete_schema_v2_and_converts_to_context_config() {
    let gguf = schema_v2_gguf(true, true);
    let parsed = read_gguf_turboquant_config(&gguf, &metadata())
        .unwrap()
        .unwrap();
    assert_eq!(parsed.mode, TurboQuantMode::TrialityConsensus);
    let consensus = parsed.consensus.as_ref().unwrap();
    assert_eq!(consensus.branches_by_layer.len(), 2);
    assert!(!consensus.override_allowed);
    assert_eq!(
        parsed.ncka.as_ref().unwrap().controller_type,
        "finite_moment_ka_v1"
    );
    assert!(parsed.urt.as_ref().unwrap().enabled);
    let context = TrialityContextConfig::try_from(&parsed).unwrap();
    assert_eq!(
        context.execution,
        TrialityExecution::AttentionLogitConsensus
    );
    assert_eq!(context.layers.len(), 2);
    assert_eq!(context.layers[0].active_branch_mask, 0b111);
    assert_eq!(context.layers[0].branches[0].bits_per_channel, 3.5);
    let resolved = ResolvedTurboQuantConfig {
        mode: parsed.mode,
        schema_kind: None,
        source_path: None,
        config: None,
        gguf_metadata: Some(parsed),
    };
    assert_eq!(resolved.schema_label(), "schema-v2");
}

#[test]
fn accepts_writer_optional_metadata_and_source_manifest() {
    let mut gguf = schema_v2_gguf(false, false);
    gguf.metadata.insert(
        "hypura.turboquant.source_profile".to_string(),
        GgufValue::String("live-schema-v2".to_string()),
    );
    gguf.metadata.insert(
        "hypura.turboquant.artifact".to_string(),
        GgufValue::String("profiles/v2.json".to_string()),
    );
    gguf.metadata.insert(
        "hypura.turboquant.weight.generated_at_utc".to_string(),
        GgufValue::String("2026-07-13T00:00:00+00:00".to_string()),
    );
    update_payload(&mut gguf, |payload| {
        payload["source_manifest"] = json!({
            "generator": "turboquant.triality_live_gguf",
            "sha256": "0".repeat(64),
        });
    });
    assert!(read_gguf_turboquant_config(&gguf, &metadata()).is_ok());
}

#[test]
fn omitted_override_key_defaults_false() {
    let mut gguf = schema_v2_gguf(false, false);
    gguf.metadata
        .remove("hypura.turboquant.triality.override_allowed");
    let parsed = read_gguf_turboquant_config(&gguf, &metadata())
        .unwrap()
        .unwrap();
    assert!(!parsed.consensus.unwrap().override_allowed);
}

#[test]
fn maps_residual_parity_execution_to_typed_mode() {
    let mut gguf = schema_v2_gguf(false, false);
    gguf.metadata.insert(
        "hypura.turboquant.triality.execution".to_string(),
        GgufValue::String("residual_parity".to_string()),
    );
    update_payload(&mut gguf, |payload| {
        payload["consensus"]["execution"] = json!("residual_parity");
    });
    let parsed = read_gguf_turboquant_config(&gguf, &metadata())
        .unwrap()
        .unwrap();
    assert_eq!(parsed.mode, TurboQuantMode::TrialityResidualParity);
    let context = TrialityContextConfig::try_from(&parsed).unwrap();
    assert_eq!(context.execution, TrialityExecution::ResidualParity);
}

#[test]
fn preserves_identity_dev_rotation_policy_for_guarded_runtime_admission() {
    let mut gguf = schema_v2_gguf(false, false);
    gguf.metadata.insert(
        "tq_rotation_policy".to_string(),
        string_array(["identity_dev", "identity_dev"]),
    );
    update_payload(&mut gguf, |payload| {
        payload["rotation_policy"] = json!("identity_dev");
    });
    let parsed = read_gguf_turboquant_config(&gguf, &metadata())
        .unwrap()
        .unwrap();
    assert_eq!(parsed.rotation_policy.unwrap().as_str(), "identity_dev");
    assert!(
        parsed
            .layers
            .iter()
            .all(|layer| layer.rotation_policy.as_str() == "identity_dev")
    );
}

#[test]
fn preserves_strict_schema_v1_without_v2_extensions() {
    let mut metadata_map = strict_base_metadata();
    metadata_map.insert(
        "hypura.turboquant.future_v1_extension".to_string(),
        GgufValue::String("preserved".to_string()),
    );
    let gguf = GgufFile {
        version: 3,
        metadata: metadata_map,
        tensors: Vec::new(),
        data_offset: 0,
    };
    let parsed = read_gguf_turboquant_config(&gguf, &metadata())
        .unwrap()
        .unwrap();
    assert_eq!(parsed.schema_version, 1);
    assert_eq!(parsed.mode, TurboQuantMode::ResearchKvSplit);
    assert!(parsed.consensus.is_none());
    assert!(parsed.ncka.is_none());
    assert!(parsed.urt.is_none());
}

#[test]
fn rejects_incomplete_unknown_and_wrongly_typed_v2_metadata() {
    let mut missing = schema_v2_gguf(false, false);
    missing.metadata.remove("hypura.turboquant.triality.views");
    assert!(read_gguf_turboquant_config(&missing, &metadata()).is_err());

    let mut unknown = schema_v2_gguf(false, false);
    unknown.metadata.insert(
        "hypura.turboquant.triality.alias_views".to_string(),
        string_array(VIEWS),
    );
    assert!(read_gguf_turboquant_config(&unknown, &metadata()).is_err());

    let mut unknown_root = schema_v2_gguf(false, false);
    unknown_root.metadata.insert(
        "hypura.turboquant.future_extension".to_string(),
        GgufValue::String("unrecognised".to_string()),
    );
    let error = read_gguf_turboquant_config(&unknown_root, &metadata())
        .unwrap_err()
        .to_string();
    assert!(error.contains("hypura.turboquant.future_extension"));

    let mut wrong_type = schema_v2_gguf(false, false);
    wrong_type.metadata.insert(
        "hypura.turboquant.triality.view_count".to_string(),
        GgufValue::Int32(3),
    );
    assert!(read_gguf_turboquant_config(&wrong_type, &metadata()).is_err());

    let mut public_schema_type = schema_v2_gguf(false, false);
    public_schema_type.metadata.insert(
        "hypura.turboquant.schema_version".to_string(),
        GgufValue::Int32(2),
    );
    assert!(read_gguf_turboquant_config(&public_schema_type, &metadata()).is_err());

    let mut public_schema_zero = schema_v2_gguf(false, false);
    public_schema_zero.metadata.insert(
        "hypura.turboquant.schema_version".to_string(),
        GgufValue::Uint32(0),
    );
    assert!(read_gguf_turboquant_config(&public_schema_zero, &metadata()).is_err());
}

#[test]
fn rejects_unknown_and_wrong_conditional_payload_root_keys() {
    let mut unknown = schema_v2_gguf(false, false);
    update_payload(&mut unknown, |payload| {
        payload["future_extension"] = json!({});
    });
    let error = read_gguf_turboquant_config(&unknown, &metadata())
        .unwrap_err()
        .to_string();
    assert!(error.contains("future_extension"));

    let mut conflicting_conditional = schema_v2_gguf(false, false);
    update_payload(&mut conflicting_conditional, |payload| {
        payload["paper_config"] = json!({});
    });
    let error = read_gguf_turboquant_config(&conflicting_conditional, &metadata())
        .unwrap_err()
        .to_string();
    assert!(error.contains("paper_config"));
}

#[test]
fn rejects_contradictory_payload_and_invalid_branch_rows() {
    let mut contradiction = schema_v2_gguf(false, false);
    contradiction.metadata.insert(
        "hypura.turboquant.triality.execution".to_string(),
        GgufValue::String("residual_parity".to_string()),
    );
    assert!(read_gguf_turboquant_config(&contradiction, &metadata()).is_err());

    let mut negative = schema_v2_gguf(false, false);
    negative.metadata.insert(
        "hypura.turboquant.triality.weights".to_string(),
        f32_array([1.1, -0.1, 0.0, 0.4, 0.4, 0.2]),
    );
    update_payload(&mut negative, |payload| {
        payload["consensus"]["rows"][0]["weights"] = json!([1.1, -0.1, 0.0]);
    });
    assert!(read_gguf_turboquant_config(&negative, &metadata()).is_err());

    let mut zero_scale = schema_v2_gguf(false, false);
    zero_scale.metadata.insert(
        "hypura.turboquant.triality.scale".to_string(),
        f32_array([0.0, 1.1, 0.9, 1.0, 1.0, 1.0]),
    );
    update_payload(&mut zero_scale, |payload| {
        payload["consensus"]["rows"][0]["scale"] = json!([0.0, 1.1, 0.9]);
    });
    assert!(read_gguf_turboquant_config(&zero_scale, &metadata()).is_err());
}

#[test]
fn rejects_profile_traversal_required_unknown_controller_and_hash_mismatch() {
    let mut traversal = schema_v2_gguf(false, false);
    traversal.metadata.insert(
        "hypura.turboquant.triality.profile_id".to_string(),
        GgufValue::String("../escape".to_string()),
    );
    update_payload(&mut traversal, |payload| {
        payload["profile_id"] = json!("../escape");
    });
    assert!(read_gguf_turboquant_config(&traversal, &metadata()).is_err());

    let mut controller = schema_v2_gguf(true, false);
    controller.metadata.insert(
        "hypura.turboquant.ncka.required".to_string(),
        GgufValue::Bool(true),
    );
    controller.metadata.insert(
        "hypura.turboquant.ncka.controller_type".to_string(),
        GgufValue::String("unknown_controller".to_string()),
    );
    update_payload(&mut controller, |payload| {
        payload["ncka"]["required"] = json!(true);
        payload["ncka"]["controller_type"] = json!("unknown_controller");
    });
    assert!(read_gguf_turboquant_config(&controller, &metadata()).is_err());

    let mut hash = schema_v2_gguf(false, true);
    hash.metadata.insert(
        "hypura.turboquant.urt.operator_word_sha256".to_string(),
        GgufValue::String("0".repeat(64)),
    );
    assert!(read_gguf_turboquant_config(&hash, &metadata()).is_err());
}

#[test]
fn optional_unknown_controller_selects_static_fallback() {
    let mut gguf = schema_v2_gguf(true, false);
    gguf.metadata.insert(
        "hypura.turboquant.ncka.controller_type".to_string(),
        GgufValue::String("future_controller".to_string()),
    );
    update_payload(&mut gguf, |payload| {
        payload["ncka"]["controller_type"] = json!("future_controller");
    });
    let parsed = read_gguf_turboquant_config(&gguf, &metadata())
        .unwrap()
        .unwrap();
    assert!(parsed.ncka.unwrap().static_fallback_selected);
}
