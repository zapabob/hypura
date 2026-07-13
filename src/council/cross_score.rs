use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::CouncilView;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateViewScore {
    pub candidate: CouncilView,
    pub evaluator: CouncilView,
    pub token_count: u32,
    pub log_likelihood: f64,
    pub mean_log_likelihood: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TeacherForcedScoreInput {
    pub candidate: CouncilView,
    pub evaluator: CouncilView,
    pub token_log_probabilities: Vec<f64>,
}

/// Mean log-likelihoods indexed as `scores[candidate][evaluator]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossScoreMatrix {
    pub scores: [[f64; 3]; 3],
    pub token_counts: [u32; 3],
}

#[derive(Debug, Error, PartialEq)]
pub enum CrossScoreError {
    #[error("logits must not be empty")]
    EmptyLogits,
    #[error("token id {token_id} is outside logits range 0..{vocab_size}")]
    TokenOutOfRange { token_id: i32, vocab_size: usize },
    #[error("logits must all be finite")]
    NonFiniteLogits,
    #[error("teacher-forced score for {candidate:?}/{evaluator:?} has no candidate tokens")]
    EmptyCandidate {
        candidate: CouncilView,
        evaluator: CouncilView,
    },
    #[error("teacher-forced token log probabilities must all be finite")]
    NonFiniteTokenScore,
    #[error("candidate token count exceeds u32")]
    TokenCountOverflow,
    #[error("duplicate score for {candidate:?}/{evaluator:?}")]
    DuplicateCell {
        candidate: CouncilView,
        evaluator: CouncilView,
    },
    #[error("missing score for {candidate:?}/{evaluator:?}")]
    MissingCell {
        candidate: CouncilView,
        evaluator: CouncilView,
    },
    #[error("candidate {candidate:?} has inconsistent token counts")]
    InconsistentTokenCount { candidate: CouncilView },
    #[error("cross-score matrix values must all be finite")]
    NonFiniteMatrix,
    #[error("cross-score token counts must all be positive")]
    ZeroTokenCount,
}

pub fn token_log_probability(logits: &[f32], token_id: i32) -> Result<f64, CrossScoreError> {
    if logits.is_empty() {
        return Err(CrossScoreError::EmptyLogits);
    }
    let token_index = usize::try_from(token_id)
        .ok()
        .filter(|&id| id < logits.len());
    let Some(token_index) = token_index else {
        return Err(CrossScoreError::TokenOutOfRange {
            token_id,
            vocab_size: logits.len(),
        });
    };
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(CrossScoreError::NonFiniteLogits);
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exp_sum = logits
        .iter()
        .map(|&value| ((value as f64) - max).exp())
        .sum::<f64>();
    let result = logits[token_index] as f64 - (max + exp_sum.ln());
    if !result.is_finite() {
        return Err(CrossScoreError::NonFiniteTokenScore);
    }
    Ok(result)
}

impl CrossScoreMatrix {
    pub fn try_new(scores: [[f64; 3]; 3], token_counts: [u32; 3]) -> Result<Self, CrossScoreError> {
        if scores.iter().flatten().any(|score| !score.is_finite()) {
            return Err(CrossScoreError::NonFiniteMatrix);
        }
        if token_counts.contains(&0) {
            return Err(CrossScoreError::ZeroTokenCount);
        }
        Ok(Self {
            scores,
            token_counts,
        })
    }

    pub fn from_teacher_forced(
        inputs: &[TeacherForcedScoreInput],
    ) -> Result<(Self, Vec<CandidateViewScore>), CrossScoreError> {
        let mut matrix = [[0.0; 3]; 3];
        let mut counts = [0_u32; 3];
        let mut present = [[false; 3]; 3];
        let mut detailed = Vec::with_capacity(9);

        for input in inputs {
            let candidate = input.candidate.index();
            let evaluator = input.evaluator.index();
            if present[candidate][evaluator] {
                return Err(CrossScoreError::DuplicateCell {
                    candidate: input.candidate,
                    evaluator: input.evaluator,
                });
            }
            if input.token_log_probabilities.is_empty() {
                return Err(CrossScoreError::EmptyCandidate {
                    candidate: input.candidate,
                    evaluator: input.evaluator,
                });
            }
            if input
                .token_log_probabilities
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(CrossScoreError::NonFiniteTokenScore);
            }
            let token_count = u32::try_from(input.token_log_probabilities.len())
                .map_err(|_| CrossScoreError::TokenCountOverflow)?;
            if counts[candidate] != 0 && counts[candidate] != token_count {
                return Err(CrossScoreError::InconsistentTokenCount {
                    candidate: input.candidate,
                });
            }
            counts[candidate] = token_count;
            let total = input.token_log_probabilities.iter().sum::<f64>();
            let mean = total / f64::from(token_count);
            matrix[candidate][evaluator] = mean;
            present[candidate][evaluator] = true;
            detailed.push(CandidateViewScore {
                candidate: input.candidate,
                evaluator: input.evaluator,
                token_count,
                log_likelihood: total,
                mean_log_likelihood: mean,
            });
        }

        for candidate in CouncilView::ALL {
            for evaluator in CouncilView::ALL {
                if !present[candidate.index()][evaluator.index()] {
                    return Err(CrossScoreError::MissingCell {
                        candidate,
                        evaluator,
                    });
                }
            }
        }
        detailed.sort_by_key(|score| (score.candidate.index(), score.evaluator.index()));
        Ok((Self::try_new(matrix, counts)?, detailed))
    }

    pub fn score(&self, candidate: CouncilView, evaluator: CouncilView) -> f64 {
        self.scores[candidate.index()][evaluator.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_is_stable_and_rejects_bad_boundaries() {
        let probability = token_log_probability(&[10_000.0, 10_001.0, 10_002.0], 2).unwrap();
        let expected = -(1.0_f64 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln();
        assert!((probability - expected).abs() < 1.0e-12);
        assert!(matches!(
            token_log_probability(&[0.0], -1),
            Err(CrossScoreError::TokenOutOfRange { .. })
        ));
        assert!(matches!(
            token_log_probability(&[f32::NAN], 0),
            Err(CrossScoreError::NonFiniteLogits)
        ));
    }

    #[test]
    fn teacher_forced_matrix_has_candidate_rows_and_evaluator_columns() {
        let mut inputs = Vec::new();
        for candidate in CouncilView::ALL {
            for evaluator in CouncilView::ALL {
                let marker = (candidate.index() * 10 + evaluator.index()) as f64;
                inputs.push(TeacherForcedScoreInput {
                    candidate,
                    evaluator,
                    token_log_probabilities: vec![-marker - 1.0, -marker - 3.0],
                });
            }
        }

        let (first, detailed) = CrossScoreMatrix::from_teacher_forced(&inputs).unwrap();
        let (second, _) = CrossScoreMatrix::from_teacher_forced(&inputs).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.token_counts, [2, 2, 2]);
        assert_eq!(
            first.score(CouncilView::SpinorPlusProxy, CouncilView::Vector),
            -12.0
        );
        assert_eq!(detailed.len(), 9);
    }
}
