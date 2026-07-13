use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::{RepresentationId, UrtObservation};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UrtComparisonKey {
    pub model_hash: String,
    pub state_id: String,
    pub layer: Option<u32>,
    pub operator_word_sha256: String,
    pub observable: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UrtPairComparison {
    pub left: RepresentationId,
    pub right: RepresentationId,
    pub absolute_error: f64,
    pub tolerance: f64,
    pub consistent: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UrtConsistencyReport {
    pub schema_version: u32,
    pub key: UrtComparisonKey,
    pub comparisons: Vec<UrtPairComparison>,
    pub max_absolute_error: f64,
    pub consistent: bool,
}

#[derive(Debug, Error, PartialEq)]
pub enum UrtComparisonError {
    #[error("at least two distinct concrete representations are required")]
    InsufficientRepresentations,
    #[error("model hashes differ; cross-model URT comparison is refused")]
    ModelHashMismatch,
    #[error("operator-word hashes differ; cross-manifest URT comparison is refused")]
    OperatorWordHashMismatch,
    #[error("state, layer, or observable differs; comparison is refused")]
    ObservationKeyMismatch,
    #[error("URT observation contains non-finite values or a negative tolerance")]
    InvalidNumericValue,
}

pub fn build_consistency_report(
    observations: &[UrtObservation],
) -> Result<UrtConsistencyReport, UrtComparisonError> {
    let Some(first) = observations.first() else {
        return Err(UrtComparisonError::InsufficientRepresentations);
    };
    if observations.iter().any(|observation| {
        !observation.value_real.is_finite()
            || !observation.value_imag.is_finite()
            || !observation.tolerance.is_finite()
            || observation.tolerance < 0.0
    }) {
        return Err(UrtComparisonError::InvalidNumericValue);
    }
    if observations
        .iter()
        .any(|observation| observation.representation.model_hash != first.representation.model_hash)
    {
        return Err(UrtComparisonError::ModelHashMismatch);
    }
    if observations
        .iter()
        .any(|observation| observation.operator_word_sha256 != first.operator_word_sha256)
    {
        return Err(UrtComparisonError::OperatorWordHashMismatch);
    }
    if observations.iter().any(|observation| {
        observation.state_id != first.state_id
            || observation.layer != first.layer
            || observation.observable != first.observable
    }) {
        return Err(UrtComparisonError::ObservationKeyMismatch);
    }

    let mut comparisons = Vec::new();
    for left_index in 0..observations.len() {
        for right_index in (left_index + 1)..observations.len() {
            let left = &observations[left_index];
            let right = &observations[right_index];
            if left.representation == right.representation {
                continue;
            }
            let absolute_error =
                (left.value_real - right.value_real).hypot(left.value_imag - right.value_imag);
            let tolerance = left.tolerance.max(right.tolerance);
            comparisons.push(UrtPairComparison {
                left: left.representation.clone(),
                right: right.representation.clone(),
                absolute_error,
                tolerance,
                consistent: absolute_error <= tolerance,
            });
        }
    }
    if comparisons.is_empty() {
        return Err(UrtComparisonError::InsufficientRepresentations);
    }
    comparisons.sort_by(|left, right| (&left.left, &left.right).cmp(&(&right.left, &right.right)));
    let max_absolute_error = comparisons
        .iter()
        .map(|comparison| comparison.absolute_error)
        .fold(0.0_f64, f64::max);
    let consistent = comparisons.iter().all(|comparison| comparison.consistent);
    Ok(UrtConsistencyReport {
        schema_version: 1,
        key: UrtComparisonKey {
            model_hash: first.representation.model_hash.clone(),
            state_id: first.state_id.clone(),
            layer: first.layer,
            operator_word_sha256: first.operator_word_sha256.clone(),
            observable: first.observable.clone(),
        },
        comparisons,
        max_absolute_error,
        consistent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::urt::{RepresentationId, RepresentationKind};

    fn observation(
        kind: RepresentationKind,
        model_hash: &str,
        operator_hash: &str,
    ) -> UrtObservation {
        UrtObservation {
            request_id: "request".to_string(),
            representation: RepresentationId {
                kind,
                model_hash: model_hash.to_string(),
                artefact_hash: None,
                backend: kind.as_str().to_string(),
                precision: "f32".to_string(),
                view: Some("vector".to_string()),
            },
            state_id: "state".to_string(),
            layer: Some(1),
            operator_word: vec!["Q".to_string(), "U".to_string()],
            operator_word_sha256: operator_hash.to_string(),
            observable: "norm".to_string(),
            value_real: 1.0,
            value_imag: 0.0,
            tolerance: 0.01,
        }
    }

    #[test]
    fn mismatched_model_and_operator_hashes_are_refused() {
        let left = observation(RepresentationKind::LlamaCpuGguf, "a", "word-a");
        let mut right = observation(RepresentationKind::LlamaCudaGguf, "b", "word-a");
        assert_eq!(
            build_consistency_report(&[left.clone(), right.clone()]),
            Err(UrtComparisonError::ModelHashMismatch)
        );
        right.representation.model_hash = "a".to_string();
        right.operator_word_sha256 = "word-b".to_string();
        assert_eq!(
            build_consistency_report(&[left, right]),
            Err(UrtComparisonError::OperatorWordHashMismatch)
        );
    }
}
