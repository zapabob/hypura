use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::Context;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::model::gguf::{GgmlType, GgufFile, TensorInfo};
use crate::model::turboquant_sidecar::{GgufNcKaConfig, GgufTurboQuantConfig};

use super::ka_gate::{KaController, KaGateConfig};
use super::moments::{NCKA_CONTROLLER_TYPE, NCKA_COORDINATE_NAMES, NCKA_PROTOCOL_VERSION};

const BRANCH_COUNT: usize = 3;

#[derive(Debug, Clone)]
pub struct EmbeddedKaController {
    coordinate_names: Vec<String>,
    coordinate_min: Vec<f32>,
    coordinate_max: Vec<f32>,
    outer_count: usize,
    knot_count: usize,
    inner_knots: Vec<f32>,
    inner_values: Vec<f32>,
    outer_knots: Vec<f32>,
    outer_values: Vec<f32>,
    fallback_weights: [f32; BRANCH_COUNT],
}

#[derive(Debug, Error, PartialEq)]
pub enum EmbeddedKaControllerError {
    #[error("required embedded NC-KA controller failed closed: {reason}")]
    Required { reason: String },
    #[error("optional embedded NC-KA controller requires static fallback: {reason}")]
    OptionalFallback {
        reason: String,
        fallback_weights: [f32; BRANCH_COUNT],
    },
    #[error("embedded NC-KA controller has no valid fallback: {reason}")]
    InvalidFallback { reason: String },
}

impl EmbeddedKaControllerError {
    pub fn fallback_weights(&self) -> Option<[f32; BRANCH_COUNT]> {
        match self {
            Self::OptionalFallback {
                fallback_weights, ..
            } => Some(*fallback_weights),
            Self::Required { .. } | Self::InvalidFallback { .. } => None,
        }
    }
}

impl EmbeddedKaController {
    pub fn load(
        model_path: &Path,
        gguf: &GgufFile,
        config: &GgufTurboQuantConfig,
    ) -> Result<Option<Self>, EmbeddedKaControllerError> {
        let Some(ncka) = config.ncka.as_ref() else {
            return Ok(None);
        };
        if !ncka.enabled {
            if ncka.required {
                return Err(EmbeddedKaControllerError::Required {
                    reason: "disabled NC-KA metadata cannot require a controller".into(),
                });
            }
            return Ok(None);
        }
        let controller = load_enabled_controller(model_path, gguf, config, ncka)
            .map_err(|reason| controller_failure(ncka, reason))?;
        Ok(Some(controller))
    }

    pub fn coordinate_names(&self) -> &[String] {
        &self.coordinate_names
    }

    pub fn fallback_weights(&self) -> [f32; BRANCH_COUNT] {
        self.fallback_weights
    }

    fn evaluate_weights(&self, coordinates: &[f32]) -> anyhow::Result<[f32; BRANCH_COUNT]> {
        anyhow::ensure!(
            coordinates.len() == self.coordinate_names.len(),
            "NC-KA coordinate count does not match the embedded controller"
        );
        let mut normalized = Vec::with_capacity(coordinates.len());
        for ((value, minimum), maximum) in coordinates
            .iter()
            .zip(&self.coordinate_min)
            .zip(&self.coordinate_max)
        {
            anyhow::ensure!(
                value.is_finite()
                    && minimum.is_finite()
                    && maximum.is_finite()
                    && maximum >= minimum,
                "NC-KA controller received non-finite coordinates or bounds"
            );
            let denominator = (maximum - minimum).max(1.0e-6);
            normalized.push(((value - minimum) / denominator).clamp(0.0, 1.0));
        }

        let coordinate_count = normalized.len();
        let mut outer_inputs = vec![0.0_f32; BRANCH_COUNT * self.outer_count];
        for branch in 0..BRANCH_COUNT {
            for outer in 0..self.outer_count {
                let mut sum = 0.0_f64;
                for (coordinate, value) in normalized.iter().enumerate() {
                    let function =
                        (branch * self.outer_count + outer) * coordinate_count + coordinate;
                    let range = function * self.knot_count..(function + 1) * self.knot_count;
                    sum += f64::from(interp_linear(
                        &self.inner_knots[range.clone()],
                        &self.inner_values[range],
                        *value,
                    )?);
                }
                anyhow::ensure!(
                    sum.is_finite(),
                    "NC-KA inner bank produced a non-finite value"
                );
                outer_inputs[branch * self.outer_count + outer] = sum as f32;
            }
        }

        let mut logits = [0.0_f32; BRANCH_COUNT];
        for (branch, logit) in logits.iter_mut().enumerate() {
            let mut sum = 0.0_f64;
            for outer in 0..self.outer_count {
                let function = branch * self.outer_count + outer;
                let range = function * self.knot_count..(function + 1) * self.knot_count;
                sum += f64::from(interp_linear(
                    &self.outer_knots[range.clone()],
                    &self.outer_values[range],
                    outer_inputs[function],
                )?);
            }
            anyhow::ensure!(
                sum.is_finite(),
                "NC-KA outer bank produced a non-finite value"
            );
            *logit = sum as f32;
        }

        softmax(logits).context("NC-KA softmax normalization failed")
    }
}

pub fn prepare_embedded_ka_controller(
    model_path: &Path,
    gguf: &GgufFile,
    config: &GgufTurboQuantConfig,
    gate_config: &mut KaGateConfig,
) -> Result<Option<EmbeddedKaController>, EmbeddedKaControllerError> {
    match EmbeddedKaController::load(model_path, gguf, config) {
        Ok(controller) => Ok(controller),
        Err(EmbeddedKaControllerError::OptionalFallback {
            fallback_weights, ..
        }) => {
            gate_config.static_fallback_weights = fallback_weights;
            Ok(None)
        }
        Err(error @ EmbeddedKaControllerError::Required { .. })
        | Err(error @ EmbeddedKaControllerError::InvalidFallback { .. }) => Err(error),
    }
}

impl KaController for EmbeddedKaController {
    fn evaluate(&self, finite_moments: &[f32]) -> anyhow::Result<[f32; BRANCH_COUNT]> {
        self.evaluate_weights(finite_moments)
    }
}

fn load_enabled_controller(
    model_path: &Path,
    gguf: &GgufFile,
    config: &GgufTurboQuantConfig,
    ncka: &GgufNcKaConfig,
) -> Result<EmbeddedKaController, String> {
    if ncka.schema_version != NCKA_PROTOCOL_VERSION || ncka.controller_type != NCKA_CONTROLLER_TYPE
    {
        return Err(format!(
            "unsupported schema version {} or controller type `{}`",
            ncka.schema_version, ncka.controller_type
        ));
    }
    if ncka.static_fallback_selected {
        return Err(
            "metadata selected the static fallback instead of the embedded controller".into(),
        );
    }
    if !ncka.s3_equivariant {
        return Err("embedded controller is not declared S3-equivariant".into());
    }
    if ncka.outer_count != 2 || ncka.knot_count != 3 {
        return Err("finite_moment_ka_v1 requires exactly two outer banks and three knots".into());
    }
    if !weights_valid(ncka.fallback_weights) {
        return Err("embedded controller metadata fallback weights are invalid".into());
    }
    if ncka
        .coordinate_names
        .iter()
        .map(String::as_str)
        .ne(NCKA_COORDINATE_NAMES)
    {
        return Err("embedded controller coordinate order is not canonical".into());
    }
    if canonical_normalisation_sha256() != ncka.normalisation_sha256 {
        return Err("embedded controller normalisation hash mismatch".into());
    }

    let consensus = config
        .consensus
        .as_ref()
        .ok_or_else(|| "enabled NC-KA requires Triality consensus metadata".to_string())?;
    let payload = config
        .payload_json
        .as_ref()
        .ok_or_else(|| "enabled NC-KA requires the schema-v2 payload".to_string())?;
    let payload: Value = serde_json::from_str(payload)
        .map_err(|error| format!("embedded controller payload is invalid JSON: {error}"))?;
    let manifest = payload
        .get("tensor_manifest")
        .and_then(Value::as_object)
        .ok_or_else(|| "embedded controller tensor manifest is missing".to_string())?;
    let prefix = format!("turboquant.profile.{}.ncka.", consensus.profile_id);
    let controller_manifest = manifest
        .iter()
        .filter(|(name, _)| name.starts_with(&prefix))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<Map<_, _>>();
    if sha256_json(&Value::Object(controller_manifest.clone())) != ncka.controller_sha256 {
        return Err("embedded controller manifest hash mismatch".into());
    }

    let coordinate_count = ncka.coordinate_names.len();
    let outer_count = ncka.outer_count as usize;
    let knot_count = ncka.knot_count as usize;
    let shapes = BTreeMap::from([
        ("coordinate_min", vec![coordinate_count as u64]),
        ("coordinate_max", vec![coordinate_count as u64]),
        (
            "inner_knots",
            vec![
                knot_count as u64,
                coordinate_count as u64,
                outer_count as u64,
                BRANCH_COUNT as u64,
            ],
        ),
        (
            "inner_values",
            vec![
                knot_count as u64,
                coordinate_count as u64,
                outer_count as u64,
                BRANCH_COUNT as u64,
            ],
        ),
        (
            "outer_knots",
            vec![knot_count as u64, outer_count as u64, BRANCH_COUNT as u64],
        ),
        (
            "outer_values",
            vec![knot_count as u64, outer_count as u64, BRANCH_COUNT as u64],
        ),
        ("fallback_weights", vec![BRANCH_COUNT as u64]),
    ]);
    let expected_names = shapes
        .keys()
        .map(|field| format!("{prefix}{field}"))
        .collect::<BTreeSet<_>>();
    let manifest_names = controller_manifest.keys().cloned().collect::<BTreeSet<_>>();
    if manifest_names != expected_names {
        return Err(
            "embedded controller manifest tensor set is incomplete or contains extras".into(),
        );
    }
    let tensor_names = gguf
        .tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with(&prefix))
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    if tensor_names != expected_names {
        return Err("embedded controller GGUF tensor set is incomplete or contains extras".into());
    }

    let mut file = File::open(model_path)
        .map_err(|error| format!("failed to open model for NC-KA tensors: {error}"))?;
    let file_length = file
        .metadata()
        .map_err(|error| format!("failed to stat model for NC-KA tensors: {error}"))?
        .len();
    let mut tensors = BTreeMap::<&str, Vec<f32>>::new();
    for (&field, shape) in &shapes {
        let name = format!("{prefix}{field}");
        let entry = controller_manifest
            .get(&name)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("embedded controller manifest entry `{name}` is invalid"))?;
        tensors.insert(
            field,
            read_f32_tensor(&mut file, file_length, gguf, &name, shape, entry)
                .map_err(|error| format!("{name}: {error}"))?,
        );
    }

    let controller = EmbeddedKaController {
        coordinate_names: ncka.coordinate_names.clone(),
        coordinate_min: tensors.remove("coordinate_min").unwrap_or_default(),
        coordinate_max: tensors.remove("coordinate_max").unwrap_or_default(),
        outer_count,
        knot_count,
        inner_knots: tensors.remove("inner_knots").unwrap_or_default(),
        inner_values: tensors.remove("inner_values").unwrap_or_default(),
        outer_knots: tensors.remove("outer_knots").unwrap_or_default(),
        outer_values: tensors.remove("outer_values").unwrap_or_default(),
        fallback_weights: tensors
            .remove("fallback_weights")
            .unwrap_or_default()
            .try_into()
            .map_err(|_| "embedded controller fallback tensor must contain three values")?,
    };
    validate_controller(&controller)?;
    if controller
        .fallback_weights
        .iter()
        .zip(ncka.fallback_weights)
        .any(|(tensor, metadata)| (*tensor - metadata).abs() > 1.0e-6)
    {
        return Err("embedded controller fallback tensor contradicts metadata".into());
    }
    Ok(controller)
}

fn read_f32_tensor(
    file: &mut File,
    file_length: u64,
    gguf: &GgufFile,
    name: &str,
    expected_shape: &[u64],
    manifest: &Map<String, Value>,
) -> Result<Vec<f32>, String> {
    let tensor = unique_tensor(gguf, name)?;
    if tensor.dtype != GgmlType::F32 || tensor.dimensions != expected_shape {
        return Err("tensor header must be F32 with the canonical shape".into());
    }
    let element_count = expected_shape
        .iter()
        .try_fold(1_u64, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| "tensor shape overflows the address space".to_string())?;
    let byte_count = element_count
        .checked_mul(4)
        .ok_or_else(|| "tensor byte length overflows the address space".to_string())?;
    if tensor.size_bytes != byte_count {
        return Err("tensor header byte length contradicts its shape".into());
    }
    let manifest_dtype = manifest.get("dtype").and_then(Value::as_str);
    let manifest_shape = manifest
        .get("shape")
        .and_then(Value::as_array)
        .and_then(|values| values.iter().map(Value::as_u64).collect::<Option<Vec<_>>>());
    let manifest_hash = manifest.get("sha256").and_then(Value::as_str);
    if manifest.len() != 3
        || manifest_dtype != Some("f32")
        || manifest_shape.as_deref() != Some(expected_shape)
        || !manifest_hash.is_some_and(is_lower_sha256)
    {
        return Err("tensor manifest record is not the exact F32 shape/hash contract".into());
    }
    let absolute_offset = gguf
        .data_offset
        .checked_add(tensor.offset)
        .ok_or_else(|| "tensor offset overflows the model file".to_string())?;
    let end = absolute_offset
        .checked_add(byte_count)
        .ok_or_else(|| "tensor end offset overflows the model file".to_string())?;
    if end > file_length {
        return Err("tensor bytes are truncated".into());
    }
    file.seek(SeekFrom::Start(absolute_offset))
        .map_err(|error| format!("failed to seek tensor bytes: {error}"))?;
    let byte_count = usize::try_from(byte_count)
        .map_err(|_| "tensor byte length exceeds the process address space".to_string())?;
    let mut bytes = vec![0_u8; byte_count];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("failed to read tensor bytes: {error}"))?;
    if sha256_bytes(&bytes) != manifest_hash.unwrap_or_default() {
        return Err("tensor SHA256 does not match the payload manifest".into());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|bytes| {
            let mut value = [0_u8; 4];
            value.copy_from_slice(bytes);
            f32::from_le_bytes(value)
        })
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("tensor contains a non-finite F32 value".into());
    }
    Ok(values)
}

fn unique_tensor<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a TensorInfo, String> {
    let mut matches = gguf.tensors.iter().filter(|tensor| tensor.name == name);
    let tensor = matches
        .next()
        .ok_or_else(|| "tensor header is missing".to_string())?;
    if matches.next().is_some() {
        return Err("tensor header is duplicated".into());
    }
    Ok(tensor)
}

fn validate_controller(controller: &EmbeddedKaController) -> Result<(), String> {
    let coordinate_count = controller.coordinate_names.len();
    let expected_inner = BRANCH_COUNT
        .checked_mul(controller.outer_count)
        .and_then(|value| value.checked_mul(coordinate_count))
        .and_then(|value| value.checked_mul(controller.knot_count))
        .ok_or_else(|| "embedded controller inner tensor size overflows".to_string())?;
    let expected_outer = BRANCH_COUNT
        .checked_mul(controller.outer_count)
        .and_then(|value| value.checked_mul(controller.knot_count))
        .ok_or_else(|| "embedded controller outer tensor size overflows".to_string())?;
    if controller.coordinate_min.len() != coordinate_count
        || controller.coordinate_max.len() != coordinate_count
        || controller.inner_knots.len() != expected_inner
        || controller.inner_values.len() != expected_inner
        || controller.outer_knots.len() != expected_outer
        || controller.outer_values.len() != expected_outer
    {
        return Err("embedded controller tensors have inconsistent lengths".into());
    }
    if controller
        .coordinate_min
        .iter()
        .zip(&controller.coordinate_max)
        .any(|(minimum, maximum)| !minimum.is_finite() || !maximum.is_finite() || maximum < minimum)
    {
        return Err("embedded controller coordinate bounds are invalid".into());
    }
    if controller
        .coordinate_min
        .iter()
        .any(|minimum| minimum.to_bits() != 0.0_f32.to_bits())
        || controller
            .coordinate_max
            .iter()
            .any(|maximum| maximum.to_bits() != 1.0_f32.to_bits())
    {
        return Err("embedded controller coordinate bounds contradict the [0,1] contract".into());
    }
    validate_knots(&controller.inner_knots, controller.knot_count)?;
    validate_knots(&controller.outer_knots, controller.knot_count)?;
    if controller
        .inner_values
        .iter()
        .any(|value| !value.is_finite())
        || controller
            .outer_values
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("embedded controller values must be finite".into());
    }
    if !weights_valid(controller.fallback_weights) {
        return Err("embedded controller fallback weights are invalid".into());
    }
    Ok(())
}

fn validate_knots(knots: &[f32], knot_count: usize) -> Result<(), String> {
    if knot_count < 2 || !knots.len().is_multiple_of(knot_count) {
        return Err("embedded controller knot bank has an invalid shape".into());
    }
    for function in knots.chunks_exact(knot_count) {
        if function.iter().any(|knot| !knot.is_finite())
            || function.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err("embedded controller knots must be finite and strictly increasing".into());
        }
    }
    Ok(())
}

fn interp_linear(knots: &[f32], values: &[f32], x: f32) -> anyhow::Result<f32> {
    anyhow::ensure!(
        !knots.is_empty() && knots.len() == values.len() && x.is_finite(),
        "invalid NC-KA interpolation input"
    );
    if knots.len() == 1 || x <= knots[0] {
        return Ok(values[0]);
    }
    if x >= knots[knots.len() - 1] {
        return Ok(values[values.len() - 1]);
    }
    let right = knots.partition_point(|knot| *knot <= x);
    let left = right - 1;
    let width = knots[right] - knots[left];
    anyhow::ensure!(
        width > 0.0 && width.is_finite(),
        "invalid NC-KA knot interval"
    );
    let ratio = (x - knots[left]) / width;
    Ok(values[left] + ratio * (values[right] - values[left]))
}

fn softmax(logits: [f32; BRANCH_COUNT]) -> anyhow::Result<[f32; BRANCH_COUNT]> {
    anyhow::ensure!(
        logits.iter().all(|logit| logit.is_finite()),
        "NC-KA logits must be finite"
    );
    let maximum = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = logits.map(|logit| (logit - maximum).exp());
    let denominator = weights.iter().map(|weight| f64::from(*weight)).sum::<f64>();
    anyhow::ensure!(
        denominator > 0.0 && denominator.is_finite(),
        "NC-KA softmax denominator is invalid"
    );
    for weight in &mut weights {
        *weight = (f64::from(*weight) / denominator) as f32;
    }
    anyhow::ensure!(weights_valid(weights), "NC-KA weights are invalid");
    Ok(weights)
}

fn controller_failure(ncka: &GgufNcKaConfig, reason: String) -> EmbeddedKaControllerError {
    if ncka.required {
        return EmbeddedKaControllerError::Required { reason };
    }
    if weights_valid(ncka.fallback_weights) {
        EmbeddedKaControllerError::OptionalFallback {
            reason,
            fallback_weights: ncka.fallback_weights,
        }
    } else {
        EmbeddedKaControllerError::InvalidFallback { reason }
    }
}

fn weights_valid(weights: [f32; BRANCH_COUNT]) -> bool {
    let sum = weights.iter().sum::<f32>();
    weights
        .iter()
        .all(|weight| weight.is_finite() && *weight >= 0.0)
        && sum.is_finite()
        && (sum - 1.0).abs() <= 1.0e-5
}

fn canonical_normalisation_sha256() -> String {
    sha256_json(&serde_json::json!({
        "coordinate_names": NCKA_COORDINATE_NAMES,
        "range": [0.0, 1.0],
        "clamp": true,
    }))
}

fn sha256_json(value: &Value) -> String {
    sha256_bytes(&serde_json::to_vec(value).unwrap_or_default())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
