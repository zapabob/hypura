use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::moments::{CouncilMomentVector, NCKA_COORDINATE_NAMES};

pub trait KaController: Send + Sync {
    fn evaluate(&self, finite_moments: &[f32]) -> anyhow::Result<[f32; 3]>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KaGateConfig {
    pub enabled: bool,
    pub required: bool,
    pub controller_s3_equivariant: bool,
    pub minimum_numerical_rank: f32,
    pub minimum_effective_rank: f32,
    pub static_fallback_weights: [f32; 3],
}

impl Default for KaGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            required: false,
            controller_s3_equivariant: true,
            minimum_numerical_rank: 2.0,
            minimum_effective_rank: 1.5,
            static_fallback_weights: [1.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateSource {
    Disabled,
    Controller,
    StaticFallback,
    RankFallback,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KaGateOutput {
    pub branch_weights: [f32; 3],
    pub evaluator_weights: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KaGateEvaluation {
    pub source: GateSource,
    pub output: KaGateOutput,
    pub numerical_rank: f32,
    pub effective_rank: f32,
    pub fallback_used: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Error, PartialEq)]
pub enum KaGateError {
    #[error("KA gate thresholds, enablement, or static probability weights are invalid")]
    InvalidConfig,
    #[error("KA gate moments must use the canonical finite coordinate names, order, and values")]
    InvalidMoments,
    #[error("required KA controller is unavailable")]
    RequiredControllerUnavailable,
    #[error("required KA controller is not declared S3-equivariant")]
    RequiredControllerNotEquivariant,
    #[error(
        "required KA controller cannot run because the finite-moment rank is below the configured gate"
    )]
    RequiredMomentRankInsufficient,
    #[error("required KA controller failed: {0}")]
    RequiredControllerFailed(String),
    #[error("required KA controller returned invalid weights")]
    RequiredControllerInvalidWeights,
}

pub fn evaluate_ka_gate(
    moments: &CouncilMomentVector,
    config: &KaGateConfig,
    controller: Option<&dyn KaController>,
) -> Result<KaGateEvaluation, KaGateError> {
    validate_config(config)?;
    validate_moments(moments)?;
    let fallback =
        validate_probability(config.static_fallback_weights).ok_or(KaGateError::InvalidConfig)?;
    if !config.enabled {
        return Ok(evaluation(
            GateSource::Disabled,
            fallback,
            moments,
            false,
            Some("ncka_disabled"),
        ));
    }
    if moments.numerical_rank < config.minimum_numerical_rank
        || moments.effective_rank < config.minimum_effective_rank
    {
        if config.required {
            return Err(KaGateError::RequiredMomentRankInsufficient);
        }
        return Ok(evaluation(
            GateSource::RankFallback,
            fallback,
            moments,
            true,
            Some("moment_rank_below_gate"),
        ));
    }
    if !config.controller_s3_equivariant {
        if config.required {
            return Err(KaGateError::RequiredControllerNotEquivariant);
        }
        return Ok(evaluation(
            GateSource::StaticFallback,
            fallback,
            moments,
            true,
            Some("controller_not_s3_equivariant"),
        ));
    }
    let Some(controller) = controller else {
        if config.required {
            return Err(KaGateError::RequiredControllerUnavailable);
        }
        return Ok(evaluation(
            GateSource::StaticFallback,
            fallback,
            moments,
            true,
            Some("controller_unavailable"),
        ));
    };
    let controller_weights = match controller.evaluate(&moments.values) {
        Ok(weights) => weights,
        Err(error) if config.required => {
            return Err(KaGateError::RequiredControllerFailed(error.to_string()));
        }
        Err(_) => {
            return Ok(evaluation(
                GateSource::StaticFallback,
                fallback,
                moments,
                true,
                Some("controller_evaluation_failed"),
            ));
        }
    };
    let Some(weights) = validate_probability(controller_weights) else {
        if config.required {
            return Err(KaGateError::RequiredControllerInvalidWeights);
        }
        return Ok(evaluation(
            GateSource::StaticFallback,
            fallback,
            moments,
            true,
            Some("controller_invalid_weights"),
        ));
    };
    Ok(evaluation(
        GateSource::Controller,
        weights,
        moments,
        false,
        None,
    ))
}

fn evaluation(
    source: GateSource,
    weights: [f32; 3],
    moments: &CouncilMomentVector,
    fallback_used: bool,
    reason: Option<&str>,
) -> KaGateEvaluation {
    KaGateEvaluation {
        source,
        output: KaGateOutput {
            branch_weights: weights,
            evaluator_weights: weights,
        },
        numerical_rank: moments.numerical_rank,
        effective_rank: moments.effective_rank,
        fallback_used,
        reason: reason.map(str::to_string),
    }
}

fn validate_config(config: &KaGateConfig) -> Result<(), KaGateError> {
    if !config.minimum_numerical_rank.is_finite()
        || !config.minimum_effective_rank.is_finite()
        || config.minimum_numerical_rank < 0.0
        || config.minimum_effective_rank < 0.0
        || config.minimum_numerical_rank > 3.0
        || config.minimum_effective_rank > 3.0
        || (config.required && !config.enabled)
        || validate_probability(config.static_fallback_weights).is_none()
    {
        return Err(KaGateError::InvalidConfig);
    }
    Ok(())
}

fn validate_moments(moments: &CouncilMomentVector) -> Result<(), KaGateError> {
    if moments.names.is_empty()
        || moments.names.len() != moments.values.len()
        || moments
            .names
            .iter()
            .map(String::as_str)
            .ne(NCKA_COORDINATE_NAMES)
        || moments.values.iter().any(|value| !value.is_finite())
        || !moments.numerical_rank.is_finite()
        || !moments.effective_rank.is_finite()
        || moments.numerical_rank < 0.0
        || moments.effective_rank < 0.0
        || moments.numerical_rank > 3.0
        || moments.effective_rank > 3.0
    {
        return Err(KaGateError::InvalidMoments);
    }
    Ok(())
}

fn validate_probability(mut weights: [f32; 3]) -> Option<[f32; 3]> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return None;
    }
    let sum = weights.iter().sum::<f32>();
    if !sum.is_finite() || (sum - 1.0).abs() > 1.0e-5 {
        return None;
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    Some(weights)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingController<'a> {
        calls: &'a AtomicUsize,
        weights: [f32; 3],
    }

    impl KaController for CountingController<'_> {
        fn evaluate(&self, finite_moments: &[f32]) -> anyhow::Result<[f32; 3]> {
            assert!(finite_moments.iter().all(|value| value.is_finite()));
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.weights)
        }
    }

    fn moments(rank: f32) -> CouncilMomentVector {
        CouncilMomentVector {
            names: NCKA_COORDINATE_NAMES.map(str::to_string).to_vec(),
            values: vec![0.5; NCKA_COORDINATE_NAMES.len()],
            numerical_rank: rank,
            effective_rank: rank,
        }
    }

    #[test]
    fn rank_gate_falls_back_without_evaluating_controller() {
        let calls = AtomicUsize::new(0);
        let controller = CountingController {
            calls: &calls,
            weights: [0.2, 0.3, 0.5],
        };
        let config = KaGateConfig {
            enabled: true,
            ..KaGateConfig::default()
        };
        let result = evaluate_ka_gate(&moments(1.0), &config, Some(&controller)).unwrap();
        assert_eq!(result.source, GateSource::RankFallback);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn near_probability_controller_weights_are_normalised() {
        let calls = AtomicUsize::new(0);
        let controller = CountingController {
            calls: &calls,
            weights: [0.2, 0.3, 0.500_001],
        };
        let config = KaGateConfig {
            enabled: true,
            ..KaGateConfig::default()
        };
        let result = evaluate_ka_gate(&moments(3.0), &config, Some(&controller)).unwrap();
        assert_eq!(result.source, GateSource::Controller);
        assert!(
            result
                .output
                .branch_weights
                .iter()
                .zip([0.2, 0.3, 0.5])
                .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-6)
        );
        assert_eq!(
            result.output.branch_weights,
            result.output.evaluator_weights
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn arbitrary_nonnegative_controller_scores_are_not_treated_as_weights() {
        let calls = AtomicUsize::new(0);
        let controller = CountingController {
            calls: &calls,
            weights: [2.0, 3.0, 5.0],
        };
        let config = KaGateConfig {
            enabled: true,
            ..KaGateConfig::default()
        };
        let result = evaluate_ka_gate(&moments(3.0), &config, Some(&controller)).unwrap();
        assert_eq!(result.source, GateSource::StaticFallback);
        assert_eq!(result.output.branch_weights, [1.0, 0.0, 0.0]);
        assert!(result.fallback_used);
    }

    #[test]
    fn required_controller_fails_closed_on_non_probability_output() {
        let calls = AtomicUsize::new(0);
        let controller = CountingController {
            calls: &calls,
            weights: [2.0, 3.0, 5.0],
        };
        let config = KaGateConfig {
            enabled: true,
            required: true,
            ..KaGateConfig::default()
        };
        assert_eq!(
            evaluate_ka_gate(&moments(3.0), &config, Some(&controller)),
            Err(KaGateError::RequiredControllerInvalidWeights)
        );
    }

    #[test]
    fn required_controller_fails_closed_when_missing() {
        let config = KaGateConfig {
            enabled: true,
            required: true,
            ..KaGateConfig::default()
        };
        assert_eq!(
            evaluate_ka_gate(&moments(3.0), &config, None),
            Err(KaGateError::RequiredControllerUnavailable)
        );
    }

    #[test]
    fn required_controller_fails_closed_when_rank_is_insufficient() {
        let config = KaGateConfig {
            enabled: true,
            required: true,
            ..KaGateConfig::default()
        };
        assert_eq!(
            evaluate_ka_gate(&moments(1.0), &config, None),
            Err(KaGateError::RequiredMomentRankInsufficient)
        );
    }

    #[test]
    fn impossible_rank_statistics_are_rejected() {
        assert_eq!(
            evaluate_ka_gate(
                &moments(3.1),
                &KaGateConfig {
                    enabled: true,
                    ..KaGateConfig::default()
                },
                None,
            ),
            Err(KaGateError::InvalidMoments)
        );
    }

    #[test]
    fn noncanonical_coordinate_order_is_rejected() {
        let mut input = moments(3.0);
        input.names.swap(0, 1);
        assert_eq!(
            evaluate_ka_gate(&input, &KaGateConfig::default(), None),
            Err(KaGateError::InvalidMoments)
        );
    }
}
