use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouncilView {
    Vector,
    SpinorPlusProxy,
    SpinorMinusProxy,
}

impl CouncilView {
    pub const ALL: [Self; 3] = [Self::Vector, Self::SpinorPlusProxy, Self::SpinorMinusProxy];

    pub const fn index(self) -> usize {
        match self {
            Self::Vector => 0,
            Self::SpinorPlusProxy => 1,
            Self::SpinorMinusProxy => 2,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::SpinorPlusProxy => "spinor_plus_proxy",
            Self::SpinorMinusProxy => "spinor_minus_proxy",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrialityBranchMetrics {
    pub logit_mean: f64,
    pub logit_variance: f64,
    pub logit_l2: f64,
    pub probability_entropy: f64,
    pub top1_probability: f64,
    pub orthogonality_error: f64,
    pub determinant_error: f64,
    pub expected_quantisation_error: f64,
    pub bytes_read: u64,
    pub duration_us: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialityConsensusMetrics {
    pub branches: [TrialityBranchMetrics; 3],
    pub pairwise_js: [f64; 3],
    pub mean_pairwise_js: f64,
    pub max_pairwise_js: f64,
    pub numerical_rank: f64,
    pub effective_rank: f64,
    pub ka_fallback_used: bool,
    pub operator_word_hash_128: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouncilCandidate {
    pub view: CouncilView,
    pub text: String,
    pub token_ids: Vec<i32>,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub tok_per_sec: f64,
    pub runtime_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_level_metrics: Option<TrialityConsensusMetrics>,
}

impl CouncilCandidate {
    pub fn new(view: CouncilView, text: impl Into<String>, token_ids: Vec<i32>) -> Self {
        let generated_tokens = u32::try_from(token_ids.len()).unwrap_or(u32::MAX);
        Self {
            view,
            text: text.into(),
            token_ids,
            prompt_tokens: 0,
            generated_tokens,
            tok_per_sec: 0.0,
            runtime_ms: 0,
            low_level_metrics: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_view_order_is_stable() {
        for (index, view) in CouncilView::ALL.into_iter().enumerate() {
            assert_eq!(view.index(), index);
        }
    }
}
