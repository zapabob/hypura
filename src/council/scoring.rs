use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::aha::AhaEvent;
use super::cross_score::CrossScoreMatrix;
use super::types::{CouncilCandidate, CouncilView};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerCouncilConfig {
    pub evaluator_weights: [f64; 3],
    pub variance_penalty: f64,
    pub agreement_weight: f64,
    pub repetition_penalty: f64,
    pub safety_penalty: f64,
    pub synthesis_enabled: bool,
    pub synthesis_min_gain: f64,
}

impl Default for AnswerCouncilConfig {
    fn default() -> Self {
        Self {
            evaluator_weights: [1.0 / 3.0; 3],
            variance_penalty: 0.1,
            agreement_weight: 0.1,
            repetition_penalty: 0.1,
            safety_penalty: 1.0,
            synthesis_enabled: false,
            synthesis_min_gain: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerCouncilResult {
    pub request_id: String,
    pub candidates: [CouncilCandidate; 3],
    pub cross_scores: CrossScoreMatrix,
    pub candidate_scores: [f64; 3],
    pub selected_view: CouncilView,
    pub selected_text: String,
    pub winner_margin: f64,
    pub agreement: f64,
    pub aha: Option<AhaEvent>,
}

pub trait SafetyPenalty {
    fn penalty(&self, candidate: &CouncilCandidate) -> anyhow::Result<f64>;
}

#[derive(Debug, Default)]
pub struct NoSafetyPenalty;

impl SafetyPenalty for NoSafetyPenalty {
    fn penalty(&self, _candidate: &CouncilCandidate) -> anyhow::Result<f64> {
        Ok(0.0)
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum CouncilSelectionError {
    #[error("unsupported_until_enabled")]
    UnsupportedSynthesis,
    #[error("council penalties and gains must be finite and non-negative")]
    InvalidConfig,
    #[error("evaluator weights must be finite, non-negative, and sum to one")]
    InvalidEvaluatorWeights,
    #[error("candidate views must contain each canonical view exactly once")]
    InvalidCandidateViews,
    #[error("cross-score matrix contains a non-finite value")]
    NonFiniteCrossScore,
    #[error("safety hook failed for {view:?}: {message}")]
    SafetyHook { view: CouncilView, message: String },
    #[error("safety penalty for {view:?} must be finite and non-negative")]
    InvalidSafetyPenalty { view: CouncilView },
    #[error("no candidate received a finite score")]
    NoFiniteCandidate,
}

pub fn select_answer(
    request_id: impl Into<String>,
    mut candidates: [CouncilCandidate; 3],
    cross_scores: &CrossScoreMatrix,
    config: &AnswerCouncilConfig,
    safety: &dyn SafetyPenalty,
) -> Result<AnswerCouncilResult, CouncilSelectionError> {
    if config.synthesis_enabled {
        return Err(CouncilSelectionError::UnsupportedSynthesis);
    }
    validate_config(config)?;
    if cross_scores
        .scores
        .iter()
        .flatten()
        .any(|score| !score.is_finite())
    {
        return Err(CouncilSelectionError::NonFiniteCrossScore);
    }

    candidates.sort_by_key(|candidate| candidate.view.index());
    if candidates
        .iter()
        .zip(CouncilView::ALL)
        .any(|(candidate, expected)| candidate.view != expected)
    {
        return Err(CouncilSelectionError::InvalidCandidateViews);
    }

    let texts = [
        candidates[0].text.as_str(),
        candidates[1].text.as_str(),
        candidates[2].text.as_str(),
    ];
    let agreement_by_candidate = agreement_scores(texts);
    let mut candidate_scores = [f64::NEG_INFINITY; 3];

    for candidate_index in 0..3 {
        let weighted_mean = (0..3)
            .map(|evaluator| {
                config.evaluator_weights[evaluator]
                    * cross_scores.scores[candidate_index][evaluator]
            })
            .sum::<f64>();
        let weighted_variance = (0..3)
            .map(|evaluator| {
                let delta = cross_scores.scores[candidate_index][evaluator] - weighted_mean;
                config.evaluator_weights[evaluator] * delta * delta
            })
            .sum::<f64>();
        let safety_value = safety
            .penalty(&candidates[candidate_index])
            .map_err(|error| CouncilSelectionError::SafetyHook {
                view: candidates[candidate_index].view,
                message: error.to_string(),
            })?;
        if !safety_value.is_finite() || safety_value < 0.0 {
            return Err(CouncilSelectionError::InvalidSafetyPenalty {
                view: candidates[candidate_index].view,
            });
        }
        let repetition = repeated_fourgram_ratio(&candidates[candidate_index].text);
        let score = weighted_mean - config.variance_penalty * weighted_variance
            + config.agreement_weight * agreement_by_candidate[candidate_index]
            - config.repetition_penalty * repetition
            - config.safety_penalty * safety_value;
        if score.is_finite() {
            candidate_scores[candidate_index] = score;
        }
    }

    let winner_index = (0..3)
        .filter(|&index| candidate_scores[index].is_finite())
        .max_by(|&left, &right| {
            candidate_scores[left]
                .total_cmp(&candidate_scores[right])
                .then_with(|| right.cmp(&left))
        })
        .ok_or(CouncilSelectionError::NoFiniteCandidate)?;
    let runner_up = (0..3)
        .filter(|&index| index != winner_index && candidate_scores[index].is_finite())
        .map(|index| candidate_scores[index])
        .max_by(f64::total_cmp);
    let winner_margin = runner_up
        .map(|score| candidate_scores[winner_index] - score)
        .unwrap_or(0.0);
    let agreement = (pair_agreement(texts[0], texts[1])
        + pair_agreement(texts[0], texts[2])
        + pair_agreement(texts[1], texts[2]))
        / 3.0;

    Ok(AnswerCouncilResult {
        request_id: request_id.into(),
        selected_view: candidates[winner_index].view,
        selected_text: candidates[winner_index].text.clone(),
        candidates,
        cross_scores: cross_scores.clone(),
        candidate_scores,
        winner_margin,
        agreement,
        aha: None,
    })
}

fn validate_config(config: &AnswerCouncilConfig) -> Result<(), CouncilSelectionError> {
    let penalties = [
        config.variance_penalty,
        config.agreement_weight,
        config.repetition_penalty,
        config.safety_penalty,
        config.synthesis_min_gain,
    ];
    if penalties
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(CouncilSelectionError::InvalidConfig);
    }
    if config
        .evaluator_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(CouncilSelectionError::InvalidEvaluatorWeights);
    }
    let sum = config.evaluator_weights.iter().sum::<f64>();
    if (sum - 1.0).abs() > 1.0e-9 {
        return Err(CouncilSelectionError::InvalidEvaluatorWeights);
    }
    Ok(())
}

pub fn agreement_scores(texts: [&str; 3]) -> [f64; 3] {
    let pair_01 = pair_agreement(texts[0], texts[1]);
    let pair_02 = pair_agreement(texts[0], texts[2]);
    let pair_12 = pair_agreement(texts[1], texts[2]);
    [
        (pair_01 + pair_02) / 2.0,
        (pair_01 + pair_12) / 2.0,
        (pair_02 + pair_12) / 2.0,
    ]
}

fn pair_agreement(left: &str, right: &str) -> f64 {
    let left_words = normalized_words(left);
    let right_words = normalized_words(right);
    let left_word_set = left_words.iter().cloned().collect::<BTreeSet<_>>();
    let right_word_set = right_words.iter().cloned().collect::<BTreeSet<_>>();
    let word_jaccard = jaccard(&left_word_set, &right_word_set);

    let left_normalized = left_words.join(" ");
    let right_normalized = right_words.join(" ");
    let left_trigrams = character_ngrams(&left_normalized, 3);
    let right_trigrams = character_ngrams(&right_normalized, 3);
    let trigram_jaccard = jaccard(&left_trigrams, &right_trigrams);

    let left_length = left_normalized.chars().count();
    let right_length = right_normalized.chars().count();
    let length_ratio = match left_length.max(right_length) {
        0 => 1.0,
        maximum => left_length.min(right_length) as f64 / maximum as f64,
    };
    (word_jaccard + trigram_jaccard + length_ratio) / 3.0
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|word| {
            let normalized = word
                .chars()
                .flat_map(char::to_lowercase)
                .filter(|character| character.is_alphanumeric())
                .collect::<String>();
            (!normalized.is_empty()).then_some(normalized)
        })
        .collect()
}

fn character_ngrams(text: &str, width: usize) -> BTreeSet<String> {
    let characters = text.chars().collect::<Vec<_>>();
    if characters.is_empty() {
        return BTreeSet::new();
    }
    if characters.len() < width {
        return BTreeSet::from([characters.iter().collect()]);
    }
    characters
        .windows(width)
        .map(|window| window.iter().collect())
        .collect()
}

fn jaccard<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> f64 {
    let union = left.union(right).count();
    if union == 0 {
        1.0
    } else {
        left.intersection(right).count() as f64 / union as f64
    }
}

pub fn repeated_fourgram_ratio(text: &str) -> f64 {
    let words = normalized_words(text);
    if words.len() < 4 {
        return 0.0;
    }
    let mut counts = BTreeMap::<Vec<String>, usize>::new();
    for window in words.windows(4) {
        *counts.entry(window.to_vec()).or_default() += 1;
    }
    let total = words.len() - 3;
    let repeated = counts
        .values()
        .map(|count| count.saturating_sub(1))
        .sum::<usize>();
    repeated as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> [CouncilCandidate; 3] {
        [
            CouncilCandidate::new(CouncilView::Vector, "alpha beta gamma delta", vec![1]),
            CouncilCandidate::new(
                CouncilView::SpinorPlusProxy,
                "alpha beta gamma delta",
                vec![2],
            ),
            CouncilCandidate::new(
                CouncilView::SpinorMinusProxy,
                "different final answer",
                vec![3],
            ),
        ]
    }

    #[test]
    fn selects_highest_score_and_uses_canonical_tie_break() {
        let config = AnswerCouncilConfig {
            agreement_weight: 0.0,
            variance_penalty: 0.0,
            ..AnswerCouncilConfig::default()
        };
        let matrix = CrossScoreMatrix::try_new(
            [[-2.0, -2.0, -2.0], [-1.0, -1.0, -1.0], [-3.0, -3.0, -3.0]],
            [1, 1, 1],
        )
        .unwrap();
        let result =
            select_answer("request", candidates(), &matrix, &config, &NoSafetyPenalty).unwrap();
        assert_eq!(result.selected_view, CouncilView::SpinorPlusProxy);

        let tied = CrossScoreMatrix::try_new([[-1.0; 3]; 3], [1, 1, 1]).unwrap();
        let result =
            select_answer("request", candidates(), &tied, &config, &NoSafetyPenalty).unwrap();
        assert_eq!(result.selected_view, CouncilView::Vector);
    }

    #[test]
    fn scoring_is_independent_of_input_candidate_order() {
        let matrix =
            CrossScoreMatrix::try_new([[-2.0; 3], [-1.0; 3], [-3.0; 3]], [1, 1, 1]).unwrap();
        let mut shuffled = candidates();
        shuffled.swap(0, 2);
        let result = select_answer(
            "request",
            shuffled,
            &matrix,
            &AnswerCouncilConfig::default(),
            &NoSafetyPenalty,
        )
        .unwrap();
        assert_eq!(result.selected_view, CouncilView::SpinorPlusProxy);
    }

    #[test]
    fn repetition_and_synthesis_boundaries_are_explicit() {
        assert_eq!(repeated_fourgram_ratio("a b c d a b c d"), 1.0 / 5.0);
        let config = AnswerCouncilConfig {
            synthesis_enabled: true,
            ..AnswerCouncilConfig::default()
        };
        let matrix = CrossScoreMatrix::try_new([[-1.0; 3]; 3], [1, 1, 1]).unwrap();
        assert_eq!(
            select_answer("request", candidates(), &matrix, &config, &NoSafetyPenalty),
            Err(CouncilSelectionError::UnsupportedSynthesis)
        );
    }
}
