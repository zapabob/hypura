use std::collections::BTreeMap;
use std::io::{Seek, SeekFrom, Write};

use hypura::council::{
    BranchMomentObservables, CouncilMomentInput, CouncilMomentVector, EmbeddedKaController,
    EmbeddedKaControllerError, GateSource, KaController, KaGateConfig, NCKA_CONTROLLER_TYPE,
    NCKA_COORDINATE_NAMES, NCKA_PROTOCOL_VERSION, evaluate_ka_gate, prepare_embedded_ka_controller,
};
use hypura::model::gguf::{GgmlType, GgufFile, TensorInfo};
use hypura::model::turboquant_sidecar::{
    GgufNcKaConfig, GgufTrialityConsensusConfig, GgufTurboQuantConfig, RotationPolicy,
    TurboQuantMode,
};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

const PROFILE: &str = "s3oracle";
const COORDINATE_COUNT: usize = 24;
const OUTER_COUNT: usize = 2;
const KNOT_COUNT: usize = 3;

fn writer_inner_slope(output_branch: usize, outer: usize, coordinate: usize) -> f32 {
    if outer != 0 {
        return 0.0;
    }
    for (start, slope) in [
        (0, 0.024),
        (3, 0.012),
        (6, 0.012),
        (9, 0.024),
        (15, 0.030),
        (18, 0.018),
    ] {
        if (start..start + 3).contains(&coordinate) {
            return if coordinate - start == output_branch {
                slope
            } else {
                0.0
            };
        }
    }
    if (12..15).contains(&coordinate) {
        let edge = [(0, 1), (0, 2), (1, 2)][coordinate - 12];
        return if output_branch == edge.0 || output_branch == edge.1 {
            0.012
        } else {
            0.004
        };
    }
    [0.004, 0.003, 0.002][coordinate - 21]
}

fn writer_outer_value(value: f32) -> f32 {
    if value <= 0.08 {
        value * 0.25
    } else {
        0.02 + (value - 0.08) * 2.25
    }
}

struct Fixture {
    file: NamedTempFile,
    gguf: GgufFile,
    config: GgufTurboQuantConfig,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json(value: &Value) -> String {
    sha256(&serde_json::to_vec(value).unwrap())
}

fn canonical_normalisation_hash() -> String {
    sha256_json(&json!({
        "coordinate_names": NCKA_COORDINATE_NAMES,
        "range": [0.0, 1.0],
        "clamp": true,
    }))
}

#[test]
fn canonical_coordinate_contract_matches_the_turboquant_writer_hash() {
    assert_eq!(NCKA_PROTOCOL_VERSION, 1);
    assert_eq!(NCKA_CONTROLLER_TYPE, "finite_moment_ka_v1");
    assert_eq!(
        canonical_normalisation_hash(),
        "2790a91b0883cc6db9a97c28364ec967f82c4d9da720cbd4f44b2ab33fe9083d"
    );
}

fn tensor_bundle(
    malformed_knots: bool,
    nonfinite: bool,
    writer_values: bool,
) -> BTreeMap<&'static str, (Vec<u64>, Vec<f32>)> {
    let axis = [0.0_f32, 0.5, 1.0];
    let inner_functions = 3 * OUTER_COUNT * COORDINATE_COUNT;
    let outer_functions = 3 * OUTER_COUNT;
    let mut inner_knots = axis.repeat(inner_functions);
    if malformed_knots {
        inner_knots[1] = 0.0;
    }
    let mut inner_values = Vec::with_capacity(inner_functions * KNOT_COUNT);
    for branch in 0..3 {
        for outer in 0..OUTER_COUNT {
            for coordinate in 0..COORDINATE_COUNT {
                for knot in axis {
                    inner_values.push(if writer_values {
                        writer_inner_slope(branch, outer, coordinate) * knot
                    } else if outer == 0 && coordinate == branch {
                        knot
                    } else {
                        0.0
                    });
                }
            }
        }
    }
    if nonfinite {
        inner_values[0] = f32::NAN;
    }
    let outer_axis = if writer_values {
        [0.0_f32, 0.08, 0.16]
    } else {
        axis
    };
    let outer_knots = outer_axis.repeat(outer_functions);
    let mut outer_values = Vec::with_capacity(outer_functions * KNOT_COUNT);
    for _branch in 0..3 {
        for outer in 0..OUTER_COUNT {
            for (knot_index, knot) in outer_axis.into_iter().enumerate() {
                outer_values.push(if writer_values {
                    if outer == 0 {
                        [0.0_f32, 0.02, 0.20][knot_index]
                    } else {
                        0.0
                    }
                } else if outer == 0 {
                    knot
                } else {
                    0.0
                });
            }
        }
    }
    BTreeMap::from([
        (
            "coordinate_min",
            (vec![COORDINATE_COUNT as u64], vec![0.0; COORDINATE_COUNT]),
        ),
        (
            "coordinate_max",
            (vec![COORDINATE_COUNT as u64], vec![1.0; COORDINATE_COUNT]),
        ),
        (
            "inner_knots",
            (
                vec![
                    KNOT_COUNT as u64,
                    COORDINATE_COUNT as u64,
                    OUTER_COUNT as u64,
                    3,
                ],
                inner_knots,
            ),
        ),
        (
            "inner_values",
            (
                vec![
                    KNOT_COUNT as u64,
                    COORDINATE_COUNT as u64,
                    OUTER_COUNT as u64,
                    3,
                ],
                inner_values,
            ),
        ),
        (
            "outer_knots",
            (vec![KNOT_COUNT as u64, OUTER_COUNT as u64, 3], outer_knots),
        ),
        (
            "outer_values",
            (vec![KNOT_COUNT as u64, OUTER_COUNT as u64, 3], outer_values),
        ),
        ("fallback_weights", (vec![3], vec![1.0, 0.0, 0.0])),
    ])
}

fn fixture(malformed_knots: bool, nonfinite: bool, writer_values: bool) -> Fixture {
    let tensors = tensor_bundle(malformed_knots, nonfinite, writer_values);
    let mut bytes = vec![0_u8; 32];
    let mut tensor_infos = Vec::new();
    let mut manifest = Map::new();
    let mut relative_offset = 0_u64;
    for (field, (shape, values)) in tensors {
        let name = format!("turboquant.profile.{PROFILE}.ncka.{field}");
        let tensor_bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        manifest.insert(
            name.clone(),
            json!({
                "dtype": "f32",
                "shape": shape,
                "sha256": sha256(&tensor_bytes),
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
    let controller_sha256 = sha256_json(&manifest_value);
    let payload_json = serde_json::to_string(&json!({
        "tensor_manifest": manifest_value,
    }))
    .unwrap();
    let ncka = GgufNcKaConfig {
        enabled: true,
        required: false,
        schema_version: 1,
        controller_type: "finite_moment_ka_v1".into(),
        coordinate_names: NCKA_COORDINATE_NAMES.map(str::to_string).to_vec(),
        outer_count: OUTER_COUNT as u32,
        knot_count: KNOT_COUNT as u32,
        s3_equivariant: true,
        controller_sha256,
        normalisation_sha256: canonical_normalisation_hash(),
        static_fallback_selected: false,
        fallback_weights: [1.0, 0.0, 0.0],
    };
    let consensus = GgufTrialityConsensusConfig {
        profile_id: PROFILE.into(),
        execution: "attention_logit_consensus".into(),
        branches_by_layer: Vec::new(),
        js_fallback_threshold: 0.2,
        required: true,
        override_allowed: false,
    };
    let config = GgufTurboQuantConfig {
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
        consensus: Some(consensus),
        ncka: Some(ncka),
        urt: None,
    };
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();
    Fixture {
        file,
        gguf: GgufFile {
            version: 3,
            metadata: BTreeMap::new(),
            tensors: tensor_infos,
            data_offset: 32,
        },
        config,
    }
}

fn load(fixture: &Fixture) -> Result<EmbeddedKaController, EmbeddedKaControllerError> {
    EmbeddedKaController::load(fixture.file.path(), &fixture.gguf, &fixture.config)
        .map(|controller| controller.expect("enabled controller must be present"))
}

fn linear_softmax_baseline(values: &[f32]) -> [f32; 3] {
    let maximum = values[..3]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights = [
        (values[0] - maximum).exp(),
        (values[1] - maximum).exp(),
        (values[2] - maximum).exp(),
    ];
    let sum = weights.iter().sum::<f32>();
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

fn pair_index(left: usize, right: usize) -> usize {
    match (left.min(right), left.max(right)) {
        (0, 1) => 12,
        (0, 2) => 13,
        (1, 2) => 14,
        _ => unreachable!(),
    }
}

fn permute_coordinates(values: &[f32], permutation: [usize; 3]) -> Vec<f32> {
    let mut output = values.to_vec();
    for start in [0, 3, 6, 9, 15, 18] {
        for (new_branch, old_branch) in permutation.into_iter().enumerate() {
            output[start + new_branch] = values[start + old_branch];
        }
    }
    for (new_pair, (left, right)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
        output[12 + new_pair] = values[pair_index(permutation[left], permutation[right])];
    }
    output
}

#[test]
fn embedded_controller_matches_linear_softmax_and_not_static_baseline() {
    let controller = load(&fixture(false, false, false)).unwrap();
    assert_eq!(
        controller.coordinate_names(),
        NCKA_COORDINATE_NAMES.map(str::to_string)
    );
    let mut coordinates = vec![0.0_f32; COORDINATE_COUNT];
    coordinates[..3].copy_from_slice(&[0.15, 0.55, 0.9]);
    let actual = controller.evaluate(&coordinates).unwrap();
    let linear = linear_softmax_baseline(&coordinates);
    for branch in 0..3 {
        assert!((actual[branch] - linear[branch]).abs() < 1.0e-6);
    }
    assert_ne!(actual, controller.fallback_weights());
}

#[test]
fn exported_writer_layout_matches_the_cpp_finite_moment_equations() {
    let controller = load(&fixture(false, false, true)).unwrap();
    let coordinates = (0..COORDINATE_COUNT)
        .map(|index| (index as f32 + 1.0) / 50.0)
        .collect::<Vec<_>>();
    let mut logits = [0.0_f32; 3];
    for branch in 0..3 {
        let inner = coordinates
            .iter()
            .enumerate()
            .map(|(coordinate, value)| writer_inner_slope(branch, 0, coordinate) * value)
            .sum::<f32>();
        logits[branch] = writer_outer_value(inner);
    }
    let expected = linear_softmax_baseline(&logits);
    let actual = controller.evaluate(&coordinates).unwrap();
    for branch in 0..3 {
        assert!((actual[branch] - expected[branch]).abs() < 1.0e-6);
    }
}

#[test]
fn latest_turboquant_writer_hash_and_oracle_weights_are_stable() {
    let fixture = fixture(false, false, true);
    assert_eq!(
        fixture.config.ncka.as_ref().unwrap().controller_sha256,
        "7a1bdd43cdc7e105076d8171b1e29436dbb5367b7616330feb4fa9581bbffab9"
    );
    let controller = load(&fixture).unwrap();
    let coordinates = (0..COORDINATE_COUNT)
        .map(|index| ((index * 37 + 11) % 101) as f32 / 100.0)
        .collect::<Vec<_>>();
    let actual = controller.evaluate(&coordinates).unwrap();
    let expected = [
        0.32970839831029586_f32,
        0.34043480116596675_f32,
        0.3298568005237375_f32,
    ];
    for branch in 0..3 {
        assert!((actual[branch] - expected[branch]).abs() <= 2.0e-7);
    }
}

#[test]
fn memory_pressure_changes_writer_weights_and_preserves_all_s3_permutations() {
    let controller = load(&fixture(false, false, true)).unwrap();
    let mut low = (0..COORDINATE_COUNT)
        .map(|index| ((index * 37 + 11) % 101) as f32 / 100.0)
        .collect::<Vec<_>>();
    low[23] = 0.0;
    let mut high = low.clone();
    high[23] = 1.0;
    let low_weights = controller.evaluate(&low).unwrap();
    let high_weights = controller.evaluate(&high).unwrap();
    let max_delta = low_weights
        .into_iter()
        .zip(high_weights)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_delta > 5.0e-4, "memory weight delta was {max_delta}");

    for coordinates in [&low, &high] {
        let before = controller.evaluate(coordinates).unwrap();
        for permutation in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let after = controller
                .evaluate(&permute_coordinates(coordinates, permutation))
                .unwrap();
            for (new_branch, old_branch) in permutation.into_iter().enumerate() {
                assert!((after[new_branch] - before[old_branch]).abs() <= 2.0e-7);
            }
        }
    }
}

#[test]
fn raw_production_observables_remain_interior_and_change_controller_weights() {
    let controller = load(&fixture(false, false, true)).unwrap();
    let first = CouncilMomentInput {
        branches: BranchMomentObservables {
            branch_entropy: [0.2, 2.0, 8.0],
            orthogonality_error: [0.01, 0.02, 0.03],
            determinant_error: [0.001, 0.002, 0.003],
            expected_quantisation_error: [0.1, 0.2, 0.4],
        },
        pairwise_js: [0.1, 0.2, 0.3],
        cross_scores: [[-0.1, -0.2, -0.3], [-2.0, -2.2, -2.4], [-8.0, -8.1, -8.2]],
        winner_margin: 0.25,
        latency_multiplier: 3.0,
        memory_ratio: 0.4,
    };
    let mut second = first.clone();
    second.branches.branch_entropy[0] = 5.0;
    second.cross_scores[0] = [-5.0, -5.1, -5.2];
    let first = CouncilMomentVector::from_input(&first, 3).unwrap();
    let second = CouncilMomentVector::from_input(&second, 3).unwrap();
    assert!(
        first
            .values
            .iter()
            .all(|value| *value > 0.0 && *value < 1.0)
    );
    let gate = KaGateConfig {
        enabled: true,
        required: true,
        controller_s3_equivariant: true,
        minimum_numerical_rank: 0.0,
        minimum_effective_rank: 0.0,
        static_fallback_weights: [0.2, 0.3, 0.5],
    };
    let first = evaluate_ka_gate(&first, &gate, Some(&controller)).unwrap();
    let second = evaluate_ka_gate(&second, &gate, Some(&controller)).unwrap();
    assert_eq!(first.source, GateSource::Controller);
    assert_eq!(second.source, GateSource::Controller);
    assert!(
        first
            .output
            .branch_weights
            .iter()
            .zip(second.output.branch_weights)
            .any(|(left, right)| (left - right).abs() > 1.0e-5)
    );
}

fn inverse_nonnegative_ratio(value: f32) -> f32 {
    value / (1.0 - value)
}

fn cross_score_row(normalised_mean: f64, normalised_variance: f64) -> [f64; 3] {
    let mean = normalised_mean.ln();
    let variance = normalised_variance / (1.0 - normalised_variance);
    let offset = (3.0 * variance / 2.0).sqrt();
    [mean - offset, mean, mean + offset]
}

#[test]
fn admitted_memory_pressure_changes_weights_from_raw_production_observables() {
    let controller = load(&fixture(false, false, true)).unwrap();
    let branch_targets = [0.8527778_f32, 0.45, 0.9430556];
    let input = CouncilMomentInput {
        branches: BranchMomentObservables {
            branch_entropy: branch_targets.map(inverse_nonnegative_ratio),
            orthogonality_error: branch_targets.map(inverse_nonnegative_ratio),
            determinant_error: branch_targets.map(inverse_nonnegative_ratio),
            expected_quantisation_error: branch_targets.map(inverse_nonnegative_ratio),
        },
        pairwise_js: [0.2 * std::f32::consts::LN_2; 3],
        cross_scores: [
            cross_score_row(0.2, 0.2),
            cross_score_row(0.2, 0.2),
            cross_score_row(0.1, 0.75),
        ],
        winner_margin: 0.0,
        latency_multiplier: 1.0,
        memory_ratio: 0.0,
    };
    let mut high_pressure = input.clone();
    high_pressure.memory_ratio = 1.0;
    let low = CouncilMomentVector::from_input(&input, 3).unwrap();
    let high = CouncilMomentVector::from_input(&high_pressure, 3).unwrap();
    assert_eq!(low.values[23], 0.0);
    assert_eq!(high.values[23], 1.0);
    let low = controller.evaluate(&low.values).unwrap();
    let high = controller.evaluate(&high.values).unwrap();
    let max_delta = low
        .into_iter()
        .zip(high)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_delta > 5.0e-4,
        "raw memory pressure weight delta was {max_delta}"
    );
}

#[test]
fn controller_is_continuous_at_every_exported_knot() {
    let controller = load(&fixture(false, false, false)).unwrap();
    for branch in 0..3 {
        for knot in [0.0_f32, 0.5, 1.0] {
            let mut left = vec![0.25_f32; COORDINATE_COUNT];
            let mut right = left.clone();
            left[branch] = knot - 1.0e-6;
            right[branch] = knot + 1.0e-6;
            let left = controller.evaluate(&left).unwrap();
            let right = controller.evaluate(&right).unwrap();
            for index in 0..3 {
                assert!((left[index] - right[index]).abs() < 2.0e-6);
            }
        }
    }
}

#[test]
fn coordinate_normalisation_clamps_to_the_exported_bounds() {
    let controller = load(&fixture(false, false, false)).unwrap();
    let mut outside = vec![0.0_f32; COORDINATE_COUNT];
    outside[..3].copy_from_slice(&[-10.0, 0.25, 20.0]);
    let mut clamped = outside.clone();
    clamped[..3].copy_from_slice(&[0.0, 0.25, 1.0]);
    assert_eq!(
        controller.evaluate(&outside).unwrap(),
        controller.evaluate(&clamped).unwrap()
    );
}

#[test]
fn simultaneous_s3_permutation_permutes_controller_weights() {
    let controller = load(&fixture(false, false, true)).unwrap();
    let coordinates = (0..COORDINATE_COUNT)
        .map(|index| (index as f32 + 1.0) / (COORDINATE_COUNT as f32 + 1.0))
        .collect::<Vec<_>>();
    let before = controller.evaluate(&coordinates).unwrap();
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let after = controller
            .evaluate(&permute_coordinates(&coordinates, permutation))
            .unwrap();
        for (new_branch, old_branch) in permutation.into_iter().enumerate() {
            assert!((after[new_branch] - before[old_branch]).abs() <= 2.0e-7);
        }
    }
}

#[test]
fn malformed_knots_are_an_optional_static_fallback_error() {
    let error = load(&fixture(true, false, false)).unwrap_err();
    assert_eq!(error.fallback_weights(), Some([1.0, 0.0, 0.0]));
    assert!(error.to_string().contains("strictly increasing"));
}

#[test]
fn malformed_tensor_shape_is_rejected_before_reading() {
    let mut fixture = fixture(false, false, false);
    let tensor = fixture
        .gguf
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name.ends_with(".inner_knots"))
        .unwrap();
    tensor.dimensions[0] = 2;
    let error = load(&fixture).unwrap_err();
    assert!(error.to_string().contains("canonical shape"));
}

#[test]
fn nonfinite_tensor_is_rejected_before_evaluation() {
    let error = load(&fixture(false, true, false)).unwrap_err();
    assert!(error.to_string().contains("non-finite"));
}

#[test]
fn coordinate_bound_tensor_must_match_the_hashed_zero_to_one_contract() {
    let mut fixture = fixture(false, false, false);
    let tensor = fixture
        .gguf
        .tensors
        .iter()
        .find(|tensor| tensor.name.ends_with(".coordinate_min"))
        .unwrap()
        .clone();
    let mut tensor_bytes = vec![0_u8; tensor.size_bytes as usize];
    tensor_bytes[..4].copy_from_slice(&0.25_f32.to_le_bytes());
    fixture
        .file
        .as_file_mut()
        .seek(SeekFrom::Start(fixture.gguf.data_offset + tensor.offset))
        .unwrap();
    fixture.file.as_file_mut().write_all(&tensor_bytes).unwrap();
    fixture.file.as_file_mut().flush().unwrap();

    let mut payload: Value =
        serde_json::from_str(fixture.config.payload_json.as_ref().unwrap()).unwrap();
    let manifest = payload["tensor_manifest"].as_object_mut().unwrap();
    manifest.get_mut(&tensor.name).unwrap()["sha256"] = Value::String(sha256(&tensor_bytes));
    fixture.config.ncka.as_mut().unwrap().controller_sha256 =
        sha256_json(&Value::Object(manifest.clone()));
    fixture.config.payload_json = Some(serde_json::to_string(&payload).unwrap());

    let error = load(&fixture).unwrap_err();
    assert!(error.to_string().contains("[0,1] contract"));
}

#[test]
fn truncated_tensor_bytes_fail_closed() {
    let mut fixture = fixture(false, false, false);
    let length = fixture.file.as_file().metadata().unwrap().len();
    fixture.file.as_file_mut().set_len(length - 1).unwrap();
    let error = load(&fixture).unwrap_err();
    assert!(error.to_string().contains("truncated"));
}

#[test]
fn tensor_hash_mismatch_uses_only_the_declared_optional_fallback() {
    let mut fixture = fixture(false, false, false);
    let mut payload: Value =
        serde_json::from_str(fixture.config.payload_json.as_ref().unwrap()).unwrap();
    let manifest = payload["tensor_manifest"].as_object_mut().unwrap();
    let tensor = manifest
        .get_mut(&format!("turboquant.profile.{PROFILE}.ncka.inner_values"))
        .unwrap();
    tensor["sha256"] = Value::String("0".repeat(64));
    fixture.config.ncka.as_mut().unwrap().controller_sha256 =
        sha256_json(&Value::Object(manifest.clone()));
    fixture.config.payload_json = Some(serde_json::to_string(&payload).unwrap());
    let error = load(&fixture).unwrap_err();
    assert!(error.to_string().contains("tensor SHA256"));
    assert_eq!(error.fallback_weights(), Some([1.0, 0.0, 0.0]));
}

#[test]
fn optional_load_error_can_only_enter_the_declared_static_gate_path() {
    let fixture = fixture(true, false, false);
    let mut gate_config = KaGateConfig {
        enabled: true,
        required: false,
        controller_s3_equivariant: true,
        minimum_numerical_rank: 0.0,
        minimum_effective_rank: 0.0,
        static_fallback_weights: [0.0, 1.0, 0.0],
    };
    let controller = prepare_embedded_ka_controller(
        fixture.file.path(),
        &fixture.gguf,
        &fixture.config,
        &mut gate_config,
    )
    .unwrap();
    assert!(controller.is_none());
    let fallback = [1.0, 0.0, 0.0];
    assert_eq!(gate_config.static_fallback_weights, fallback);
    let moments = CouncilMomentVector {
        names: NCKA_COORDINATE_NAMES.map(str::to_string).to_vec(),
        values: vec![0.25; COORDINATE_COUNT],
        numerical_rank: 3.0,
        effective_rank: 3.0,
    };
    let result = evaluate_ka_gate(&moments, &gate_config, None).unwrap();
    assert_eq!(result.source, GateSource::StaticFallback);
    assert_eq!(result.output.branch_weights, fallback);
}

#[test]
fn prepared_controller_runs_through_the_public_gate_path() {
    let fixture = fixture(false, false, false);
    let mut gate_config = KaGateConfig {
        enabled: true,
        required: true,
        controller_s3_equivariant: true,
        minimum_numerical_rank: 0.0,
        minimum_effective_rank: 0.0,
        static_fallback_weights: [1.0, 0.0, 0.0],
    };
    let controller = prepare_embedded_ka_controller(
        fixture.file.path(),
        &fixture.gguf,
        &fixture.config,
        &mut gate_config,
    )
    .unwrap()
    .unwrap();
    let moments = CouncilMomentVector {
        names: NCKA_COORDINATE_NAMES.map(str::to_string).to_vec(),
        values: vec![0.25; COORDINATE_COUNT],
        numerical_rank: 3.0,
        effective_rank: 3.0,
    };
    let expected = controller.evaluate(&moments.values).unwrap();
    let result = evaluate_ka_gate(&moments, &gate_config, Some(&controller)).unwrap();
    assert_eq!(result.source, GateSource::Controller);
    assert_eq!(result.output.branch_weights, expected);
    assert!(!result.fallback_used);
}

#[test]
fn required_controller_converts_tensor_failure_to_fail_closed_error() {
    let mut fixture = fixture(true, false, false);
    fixture.config.ncka.as_mut().unwrap().required = true;
    let mut gate_config = KaGateConfig::default();
    let error = prepare_embedded_ka_controller(
        fixture.file.path(),
        &fixture.gguf,
        &fixture.config,
        &mut gate_config,
    )
    .unwrap_err();
    assert!(matches!(error, EmbeddedKaControllerError::Required { .. }));
    assert_eq!(error.fallback_weights(), None);
}

#[test]
fn invalid_optional_fallback_aborts_without_mutating_the_gate() {
    let mut fixture = fixture(true, false, false);
    fixture.config.ncka.as_mut().unwrap().fallback_weights = [2.0, 3.0, 5.0];
    let mut gate_config = KaGateConfig {
        static_fallback_weights: [0.0, 1.0, 0.0],
        ..KaGateConfig::default()
    };
    let error = prepare_embedded_ka_controller(
        fixture.file.path(),
        &fixture.gguf,
        &fixture.config,
        &mut gate_config,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        EmbeddedKaControllerError::InvalidFallback { .. }
    ));
    assert_eq!(gate_config.static_fallback_weights, [0.0, 1.0, 0.0]);
}

#[test]
fn unsupported_optional_and_required_controller_semantics_are_distinct() {
    let mut optional = fixture(false, false, false);
    optional.config.ncka.as_mut().unwrap().schema_version = 7;
    assert!(matches!(
        load(&optional),
        Err(EmbeddedKaControllerError::OptionalFallback { .. })
    ));

    let mut required = fixture(false, false, false);
    let ncka = required.config.ncka.as_mut().unwrap();
    ncka.required = true;
    ncka.schema_version = 7;
    assert!(matches!(
        load(&required),
        Err(EmbeddedKaControllerError::Required { .. })
    ));
}

#[test]
fn disabled_required_controller_metadata_fails_closed() {
    let mut fixture = fixture(false, false, false);
    let ncka = fixture.config.ncka.as_mut().unwrap();
    ncka.enabled = false;
    ncka.required = true;
    assert!(matches!(
        load(&fixture),
        Err(EmbeddedKaControllerError::Required { .. })
    ));
}
