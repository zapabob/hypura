use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::CouncilView;

pub const AHA_TRUTH_DISCLAIMER: &str = "Aha indicates resolved cross-view disagreement and improved observable support; it does not guarantee factual truth.";
pub const AHA_MAX_FALSE_POSITIVE_RATE: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AhaMode {
    OfflineGrounded,
    OnlineObservable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AhaReasonCode {
    OfflineReferenceGain,
    OnlineObservableResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AhaDisabledReason {
    MissingSafetyHookEvidence,
    MissingCalibrationEvidence,
    CalibrationGateNotMet,
    MissingModeEvidence,
    ObservableThresholdNotMet,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AhaThresholds {
    pub disagreement: f64,
    pub score_gain: f64,
    pub winner_margin: f64,
    pub urt_consistency_gain: f64,
}

impl Default for AhaThresholds {
    fn default() -> Self {
        Self {
            disagreement: 0.08,
            score_gain: 0.015,
            winner_margin: 0.010,
            urt_consistency_gain: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum AhaEvidence {
    OfflineGrounded { reference_score_gain: f64 },
    OnlineObservable { post_consensus_js: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AhaSafetyEvidence {
    pub hook_id: String,
    pub pre_penalty: f64,
    pub post_penalty: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AhaCalibrationEvidence {
    pub calibration_id: String,
    pub labeled_samples: u32,
    pub false_positive_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AhaInput {
    pub selected_view: CouncilView,
    pub baseline_view: CouncilView,
    pub pre_consensus_js: f64,
    pub council_score_gain: f64,
    pub winner_margin: f64,
    pub safety: Option<AhaSafetyEvidence>,
    pub calibration: Option<AhaCalibrationEvidence>,
    pub urt_errors: Option<(f64, f64)>,
    pub moment_effective_rank: Option<f64>,
    pub evidence: Option<AhaEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AhaEvent {
    pub schema_version: u32,
    pub mode: AhaMode,
    pub reason_code: AhaReasonCode,
    pub selected_view: CouncilView,
    pub baseline_view: CouncilView,
    pub pre_consensus_js: f64,
    pub post_consensus_js: Option<f64>,
    pub score_gain: f64,
    pub winner_margin: f64,
    pub urt_pre_error: Option<f64>,
    pub urt_post_error: Option<f64>,
    pub moment_effective_rank: Option<f64>,
    pub message: String,
    pub truth_disclaimer: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AhaEvaluation {
    pub event: Option<AhaEvent>,
    pub disabled_reason: Option<AhaDisabledReason>,
}

#[derive(Debug, Error, PartialEq)]
pub enum AhaError {
    #[error("Aha thresholds must be finite and non-negative")]
    InvalidThresholds,
    #[error("Aha evidence must contain finite, non-negative observable values")]
    InvalidEvidence,
}

pub fn classify_aha(
    input: &AhaInput,
    thresholds: &AhaThresholds,
) -> Result<Option<AhaEvent>, AhaError> {
    Ok(classify_aha_with_status(input, thresholds)?.event)
}

pub fn classify_aha_with_status(
    input: &AhaInput,
    thresholds: &AhaThresholds,
) -> Result<AhaEvaluation, AhaError> {
    validate_thresholds(thresholds)?;
    validate_input(input)?;
    let Some(safety) = input.safety.as_ref() else {
        return Ok(disabled(AhaDisabledReason::MissingSafetyHookEvidence));
    };
    let Some(calibration) = input.calibration.as_ref() else {
        return Ok(disabled(AhaDisabledReason::MissingCalibrationEvidence));
    };
    if calibration.false_positive_rate > AHA_MAX_FALSE_POSITIVE_RATE {
        return Ok(disabled(AhaDisabledReason::CalibrationGateNotMet));
    }
    let Some(evidence) = input.evidence.as_ref() else {
        return Ok(disabled(AhaDisabledReason::MissingModeEvidence));
    };

    let urt_passes = match input.urt_errors {
        Some((before, after)) => before - after >= thresholds.urt_consistency_gain,
        None => true,
    };
    let common_passes = input.selected_view != input.baseline_view
        && input.pre_consensus_js >= thresholds.disagreement
        && input.council_score_gain >= thresholds.score_gain
        && input.winner_margin >= thresholds.winner_margin
        && safety.post_penalty <= safety.pre_penalty
        && urt_passes;
    if !common_passes {
        return Ok(disabled(AhaDisabledReason::ObservableThresholdNotMet));
    }

    let (mode, reason_code, post_consensus_js, mode_passes, message) = match evidence {
        AhaEvidence::OfflineGrounded {
            reference_score_gain,
        } => (
            AhaMode::OfflineGrounded,
            AhaReasonCode::OfflineReferenceGain,
            None,
            *reference_score_gain >= thresholds.score_gain,
            "Reference-scored support improved after cross-view disagreement.".to_string(),
        ),
        AhaEvidence::OnlineObservable { post_consensus_js } => (
            AhaMode::OnlineObservable,
            AhaReasonCode::OnlineObservableResolution,
            Some(*post_consensus_js),
            *post_consensus_js < input.pre_consensus_js,
            "Cross-view disagreement decreased while observable winner support improved."
                .to_string(),
        ),
    };
    if !mode_passes {
        return Ok(disabled(AhaDisabledReason::ObservableThresholdNotMet));
    }
    let (urt_pre_error, urt_post_error) = input
        .urt_errors
        .map(|(before, after)| (Some(before), Some(after)))
        .unwrap_or((None, None));

    Ok(AhaEvaluation {
        event: Some(AhaEvent {
            schema_version: 1,
            mode,
            reason_code,
            selected_view: input.selected_view,
            baseline_view: input.baseline_view,
            pre_consensus_js: input.pre_consensus_js,
            post_consensus_js,
            score_gain: input.council_score_gain,
            winner_margin: input.winner_margin,
            urt_pre_error,
            urt_post_error,
            moment_effective_rank: input.moment_effective_rank,
            message,
            truth_disclaimer: AHA_TRUTH_DISCLAIMER.to_string(),
        }),
        disabled_reason: None,
    })
}

fn disabled(reason: AhaDisabledReason) -> AhaEvaluation {
    AhaEvaluation {
        event: None,
        disabled_reason: Some(reason),
    }
}

fn validate_thresholds(thresholds: &AhaThresholds) -> Result<(), AhaError> {
    if [
        thresholds.disagreement,
        thresholds.score_gain,
        thresholds.winner_margin,
        thresholds.urt_consistency_gain,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(AhaError::InvalidThresholds);
    }
    Ok(())
}

fn validate_input(input: &AhaInput) -> Result<(), AhaError> {
    let mut values = vec![
        input.pre_consensus_js,
        input.council_score_gain,
        input.winner_margin,
    ];
    if let Some(safety) = &input.safety {
        if safety.hook_id.trim().is_empty() {
            return Err(AhaError::InvalidEvidence);
        }
        values.extend([safety.pre_penalty, safety.post_penalty]);
    }
    if let Some(calibration) = &input.calibration {
        if calibration.calibration_id.trim().is_empty() || calibration.labeled_samples == 0 {
            return Err(AhaError::InvalidEvidence);
        }
        values.push(calibration.false_positive_rate);
        if calibration.false_positive_rate > 1.0 {
            return Err(AhaError::InvalidEvidence);
        }
    }
    if let Some(evidence) = &input.evidence {
        match evidence {
            AhaEvidence::OfflineGrounded {
                reference_score_gain,
            } => values.push(*reference_score_gain),
            AhaEvidence::OnlineObservable { post_consensus_js } => values.push(*post_consensus_js),
        }
    }
    if let Some((before, after)) = input.urt_errors {
        values.extend([before, after]);
    }
    if let Some(rank) = input.moment_effective_rank {
        values.push(rank);
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(AhaError::InvalidEvidence);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn online_input() -> AhaInput {
        AhaInput {
            selected_view: CouncilView::SpinorPlusProxy,
            baseline_view: CouncilView::Vector,
            pre_consensus_js: 0.12,
            council_score_gain: 0.02,
            winner_margin: 0.02,
            safety: Some(AhaSafetyEvidence {
                hook_id: "test-safety-v1".to_string(),
                pre_penalty: 0.1,
                post_penalty: 0.1,
            }),
            calibration: Some(AhaCalibrationEvidence {
                calibration_id: "labeled-validation-v1".to_string(),
                labeled_samples: 100,
                false_positive_rate: AHA_MAX_FALSE_POSITIVE_RATE,
            }),
            urt_errors: Some((0.02, 0.01)),
            moment_effective_rank: Some(2.5),
            evidence: Some(AhaEvidence::OnlineObservable {
                post_consensus_js: 0.04,
            }),
        }
    }

    #[test]
    fn online_event_requires_resolved_disagreement_and_safe_observables() {
        let event = classify_aha(&online_input(), &AhaThresholds::default())
            .unwrap()
            .unwrap();
        assert_eq!(event.truth_disclaimer, AHA_TRUTH_DISCLAIMER);

        let mut unresolved = online_input();
        unresolved.evidence = Some(AhaEvidence::OnlineObservable {
            post_consensus_js: 0.13,
        });
        assert!(
            classify_aha(&unresolved, &AhaThresholds::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn offline_event_requires_reference_scored_gain() {
        let mut grounded = online_input();
        grounded.evidence = Some(AhaEvidence::OfflineGrounded {
            reference_score_gain: 0.02,
        });
        let event = classify_aha(&grounded, &AhaThresholds::default())
            .unwrap()
            .unwrap();
        assert_eq!(event.mode, AhaMode::OfflineGrounded);

        grounded.evidence = Some(AhaEvidence::OfflineGrounded {
            reference_score_gain: 0.01,
        });
        assert!(
            classify_aha(&grounded, &AhaThresholds::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn safety_or_urt_regression_disables_online_event() {
        let mut unsafe_input = online_input();
        unsafe_input.safety.as_mut().unwrap().post_penalty = 0.2;
        assert!(
            classify_aha(&unsafe_input, &AhaThresholds::default())
                .unwrap()
                .is_none()
        );

        let mut inconsistent = online_input();
        inconsistent.urt_errors = Some((0.01, 0.02));
        assert!(
            classify_aha(&inconsistent, &AhaThresholds::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn aha_is_disabled_without_real_safety_calibration_or_mode_evidence() {
        let mut missing_safety = online_input();
        missing_safety.safety = None;
        let evaluation =
            classify_aha_with_status(&missing_safety, &AhaThresholds::default()).unwrap();
        assert_eq!(
            evaluation.disabled_reason,
            Some(AhaDisabledReason::MissingSafetyHookEvidence)
        );

        let mut missing_calibration = online_input();
        missing_calibration.calibration = None;
        let evaluation =
            classify_aha_with_status(&missing_calibration, &AhaThresholds::default()).unwrap();
        assert_eq!(
            evaluation.disabled_reason,
            Some(AhaDisabledReason::MissingCalibrationEvidence)
        );

        let mut failed_calibration = online_input();
        failed_calibration
            .calibration
            .as_mut()
            .unwrap()
            .false_positive_rate = AHA_MAX_FALSE_POSITIVE_RATE + 0.001;
        let evaluation =
            classify_aha_with_status(&failed_calibration, &AhaThresholds::default()).unwrap();
        assert_eq!(
            evaluation.disabled_reason,
            Some(AhaDisabledReason::CalibrationGateNotMet)
        );

        let mut missing_mode = online_input();
        missing_mode.evidence = None;
        let evaluation =
            classify_aha_with_status(&missing_mode, &AhaThresholds::default()).unwrap();
        assert_eq!(
            evaluation.disabled_reason,
            Some(AhaDisabledReason::MissingModeEvidence)
        );
    }

    #[test]
    fn serialized_event_has_no_private_reasoning_fields() {
        let event = classify_aha(&online_input(), &AhaThresholds::default())
            .unwrap()
            .unwrap();
        let value = serde_json::to_value(event).unwrap();
        let keys = value
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        for forbidden in [
            "reasoning",
            "rationale",
            "analysis",
            "thoughts",
            "chain_of_thought",
        ] {
            assert!(!keys.iter().any(|key| key == forbidden));
        }
    }
}
