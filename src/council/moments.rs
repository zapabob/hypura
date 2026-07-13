use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::CouncilView;

const FEATURE_COUNT: usize = 10;
const MAX_MOMENT_DEGREE: u32 = 16;
const DOMAIN_EPSILON: f64 = 1.0e-6;

pub const NCKA_PROTOCOL_VERSION: u32 = 1;
pub const NCKA_CONTROLLER_TYPE: &str = "finite_moment_ka_v1";

pub const NCKA_COORDINATE_NAMES: [&str; 24] = [
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NckaNormalisation {
    NonnegativeRatio,
    JensenShannonLn2,
    LogLikelihoodExp,
    LatencyMultiplier,
    UnitInterval,
}

pub const NCKA_NORMALISATION_CONTRACT: [NckaNormalisation; 24] = [
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::JensenShannonLn2,
    NckaNormalisation::JensenShannonLn2,
    NckaNormalisation::JensenShannonLn2,
    NckaNormalisation::LogLikelihoodExp,
    NckaNormalisation::LogLikelihoodExp,
    NckaNormalisation::LogLikelihoodExp,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::NonnegativeRatio,
    NckaNormalisation::LatencyMultiplier,
    NckaNormalisation::UnitInterval,
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchMomentObservables {
    pub branch_entropy: [f32; 3],
    pub orthogonality_error: [f32; 3],
    pub determinant_error: [f32; 3],
    pub expected_quantisation_error: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouncilMomentInput {
    pub branches: BranchMomentObservables,
    pub pairwise_js: [f32; 3],
    pub cross_scores: [[f64; 3]; 3],
    pub winner_margin: f32,
    pub latency_multiplier: f32,
    pub memory_ratio: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouncilMomentVector {
    pub names: Vec<String>,
    pub values: Vec<f32>,
    pub numerical_rank: f32,
    pub effective_rank: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankStatistics {
    pub numerical_rank: f32,
    pub effective_rank: f32,
    pub eigenvalues: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruncatedMomentGram {
    pub degree: u32,
    pub values: [[f64; 3]; 3],
    pub rank: RankStatistics,
}

#[derive(Debug, Error, PartialEq)]
pub enum MomentError {
    #[error("moment degree {degree} exceeds maximum {maximum}")]
    DegreeTooLarge { degree: u32, maximum: u32 },
    #[error("moment observables must all be finite")]
    NonFiniteObservable,
    #[error("moment observable `{coordinate}` is outside its canonical domain")]
    InvalidObservableRange { coordinate: String },
}

impl CouncilMomentVector {
    pub fn from_input(input: &CouncilMomentInput, degree: u32) -> Result<Self, MomentError> {
        validate_input(input)?;
        let gram = truncated_moment_gram(input, degree)?;
        let names = NCKA_COORDINATE_NAMES.map(str::to_string).to_vec();
        let mut raw_values = Vec::with_capacity(24);

        for coordinates in [
            input.branches.branch_entropy,
            input.branches.orthogonality_error,
            input.branches.determinant_error,
            input.branches.expected_quantisation_error,
        ] {
            for view in CouncilView::ALL {
                raw_values.push(f64::from(coordinates[view.index()]));
            }
        }

        for value in input.pairwise_js {
            raw_values.push(f64::from(value));
        }

        for view in CouncilView::ALL {
            let (mean, _) = cross_score_statistics(input.cross_scores[view.index()])?;
            raw_values.push(mean);
        }
        for view in CouncilView::ALL {
            let (_, variance) = cross_score_statistics(input.cross_scores[view.index()])?;
            raw_values.push(variance);
        }

        raw_values.extend([
            f64::from(input.winner_margin),
            f64::from(input.latency_multiplier),
            f64::from(input.memory_ratio),
        ]);
        let values = raw_values
            .into_iter()
            .zip(NCKA_NORMALISATION_CONTRACT)
            .zip(NCKA_COORDINATE_NAMES)
            .map(|((value, normalisation), coordinate)| {
                normalise_ncka_coordinate(value, normalisation, coordinate)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            names,
            values,
            numerical_rank: gram.rank.numerical_rank,
            effective_rank: gram.rank.effective_rank,
        })
    }
}

pub fn branch_feature_rows(
    input: &CouncilMomentInput,
) -> Result<[[f64; FEATURE_COUNT]; 3], MomentError> {
    validate_input(input)?;
    let mut rows = [[0.0; FEATURE_COUNT]; 3];
    for view in CouncilView::ALL {
        let index = view.index();
        let (cross_mean, cross_variance) = cross_score_statistics(input.cross_scores[index])?;
        let incident_js = match index {
            0 => (f64::from(input.pairwise_js[0]) + f64::from(input.pairwise_js[1])) / 2.0,
            1 => (f64::from(input.pairwise_js[0]) + f64::from(input.pairwise_js[2])) / 2.0,
            _ => (f64::from(input.pairwise_js[1]) + f64::from(input.pairwise_js[2])) / 2.0,
        };
        rows[index] = [
            f64::from(input.branches.branch_entropy[index]),
            f64::from(input.branches.orthogonality_error[index]),
            f64::from(input.branches.determinant_error[index]),
            f64::from(input.branches.expected_quantisation_error[index]),
            cross_mean,
            cross_variance,
            incident_js,
            f64::from(input.winner_margin),
            f64::from(input.latency_multiplier),
            f64::from(input.memory_ratio),
        ];
    }
    Ok(rows)
}

pub fn truncated_moment_gram(
    input: &CouncilMomentInput,
    degree: u32,
) -> Result<TruncatedMomentGram, MomentError> {
    if degree > MAX_MOMENT_DEGREE {
        return Err(MomentError::DegreeTooLarge {
            degree,
            maximum: MAX_MOMENT_DEGREE,
        });
    }
    let rows = branch_feature_rows(input)?;
    let mut scalar_products = [[0.0_f64; 3]; 3];
    for left in 0..3 {
        for right in 0..3 {
            scalar_products[left][right] = rows[left]
                .iter()
                .zip(rows[right])
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f64>();
        }
    }
    let scale = scalar_products
        .iter()
        .enumerate()
        .map(|(index, row)| row[index].abs())
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);
    let mut values = [[0.0_f64; 3]; 3];
    for left in 0..3 {
        for right in 0..3 {
            let normalised = scalar_products[left][right] / scale;
            values[left][right] = (0..=degree)
                .map(|power| normalised.powi(power as i32))
                .sum::<f64>();
        }
    }
    let rank = rank_statistics(values);
    Ok(TruncatedMomentGram {
        degree,
        values,
        rank,
    })
}

fn validate_input(input: &CouncilMomentInput) -> Result<(), MomentError> {
    let finite = input
        .branches
        .branch_entropy
        .iter()
        .chain(&input.branches.orthogonality_error)
        .chain(&input.branches.determinant_error)
        .chain(&input.branches.expected_quantisation_error)
        .chain(&input.pairwise_js)
        .copied()
        .chain([
            input.winner_margin,
            input.latency_multiplier,
            input.memory_ratio,
        ])
        .all(f32::is_finite)
        && input
            .cross_scores
            .iter()
            .flatten()
            .all(|value| value.is_finite());
    if finite {
        Ok(())
    } else {
        Err(MomentError::NonFiniteObservable)
    }
}

fn cross_score_statistics(scores: [f64; 3]) -> Result<(f64, f64), MomentError> {
    let mean = scores.iter().sum::<f64>() / 3.0;
    let variance = scores
        .iter()
        .map(|score| {
            let delta = score - mean;
            delta * delta
        })
        .sum::<f64>()
        / 3.0;
    if mean.is_finite() && variance.is_finite() {
        Ok((mean, variance))
    } else {
        Err(MomentError::NonFiniteObservable)
    }
}

fn normalise_ncka_coordinate(
    value: f64,
    normalisation: NckaNormalisation,
    coordinate: &str,
) -> Result<f32, MomentError> {
    if !value.is_finite() {
        return Err(MomentError::NonFiniteObservable);
    }
    let invalid_range = || MomentError::InvalidObservableRange {
        coordinate: coordinate.to_string(),
    };
    let normalised = match normalisation {
        NckaNormalisation::NonnegativeRatio => {
            if value < -DOMAIN_EPSILON {
                return Err(invalid_range());
            }
            let nonnegative = value.max(0.0);
            nonnegative / (1.0 + nonnegative)
        }
        NckaNormalisation::JensenShannonLn2 => {
            if !(-DOMAIN_EPSILON..=std::f64::consts::LN_2 + DOMAIN_EPSILON).contains(&value) {
                return Err(invalid_range());
            }
            value.clamp(0.0, std::f64::consts::LN_2) / std::f64::consts::LN_2
        }
        NckaNormalisation::LogLikelihoodExp => {
            if value > DOMAIN_EPSILON {
                return Err(invalid_range());
            }
            value.min(0.0).exp()
        }
        NckaNormalisation::LatencyMultiplier => {
            if value < 1.0 - DOMAIN_EPSILON {
                return Err(invalid_range());
            }
            let ratio = value.max(1.0);
            (ratio - 1.0) / ratio
        }
        NckaNormalisation::UnitInterval => {
            if !(-DOMAIN_EPSILON..=1.0 + DOMAIN_EPSILON).contains(&value) {
                return Err(invalid_range());
            }
            value.clamp(0.0, 1.0)
        }
    };
    if normalised.is_finite() && (0.0..=1.0).contains(&normalised) {
        Ok(normalised as f32)
    } else {
        Err(MomentError::NonFiniteObservable)
    }
}

fn rank_statistics(mut matrix: [[f64; 3]; 3]) -> RankStatistics {
    for _ in 0..32 {
        let (row, column, maximum) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .map(|(row, column)| (row, column, matrix[row][column].abs()))
            .max_by(|left, right| left.2.total_cmp(&right.2))
            .unwrap();
        if maximum <= 1.0e-12 {
            break;
        }
        let angle =
            0.5 * (2.0 * matrix[row][column]).atan2(matrix[column][column] - matrix[row][row]);
        let cosine = angle.cos();
        let sine = angle.sin();
        let old = matrix;
        for index in 0..3 {
            if index != row && index != column {
                matrix[index][row] = cosine * old[index][row] - sine * old[index][column];
                matrix[row][index] = matrix[index][row];
                matrix[index][column] = sine * old[index][row] + cosine * old[index][column];
                matrix[column][index] = matrix[index][column];
            }
        }
        matrix[row][row] = cosine * cosine * old[row][row] - 2.0 * sine * cosine * old[row][column]
            + sine * sine * old[column][column];
        matrix[column][column] = sine * sine * old[row][row]
            + 2.0 * sine * cosine * old[row][column]
            + cosine * cosine * old[column][column];
        matrix[row][column] = 0.0;
        matrix[column][row] = 0.0;
    }

    let mut eigenvalues = [matrix[0][0], matrix[1][1], matrix[2][2]];
    eigenvalues.sort_by(|left, right| right.total_cmp(left));
    for eigenvalue in &mut eigenvalues {
        if *eigenvalue < 0.0 && eigenvalue.abs() <= 1.0e-9 {
            *eigenvalue = 0.0;
        }
    }
    let maximum = eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let numerical_rank = if maximum <= f64::EPSILON {
        0.0
    } else {
        eigenvalues
            .iter()
            .filter(|value| **value > maximum * 1.0e-6)
            .count() as f32
    };
    let positive_sum = eigenvalues
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .sum::<f64>();
    let effective_rank = if positive_sum <= f64::EPSILON {
        0.0
    } else {
        let entropy = eigenvalues
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .map(|value| {
                let probability = value / positive_sum;
                -probability * probability.ln()
            })
            .sum::<f64>();
        entropy.exp() as f32
    };
    RankStatistics {
        numerical_rank,
        effective_rank,
        eigenvalues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> CouncilMomentInput {
        CouncilMomentInput {
            branches: BranchMomentObservables {
                branch_entropy: [0.1, 0.2, 0.3],
                orthogonality_error: [0.01, 0.02, 0.03],
                determinant_error: [0.001, 0.002, 0.003],
                expected_quantisation_error: [0.4, 0.5, 0.6],
            },
            pairwise_js: [0.07, 0.08, 0.09],
            cross_scores: [[-1.0, -1.1, -1.2], [-2.0, -2.2, -2.4], [-3.0, -3.3, -3.6]],
            winner_margin: 0.02,
            latency_multiplier: 2.5,
            memory_ratio: 0.75,
        }
    }

    fn pair_index(left: usize, right: usize) -> usize {
        match (left.min(right), left.max(right)) {
            (0, 1) => 0,
            (0, 2) => 1,
            (1, 2) => 2,
            _ => unreachable!(),
        }
    }

    #[test]
    fn moment_vector_contains_all_required_coordinates() {
        let vector = CouncilMomentVector::from_input(&input(), 2).unwrap();
        assert_eq!(vector.names.len(), 24);
        assert_eq!(vector.names.len(), vector.values.len());
        assert_eq!(
            vector.names.iter().map(String::as_str).collect::<Vec<_>>(),
            NCKA_COORDINATE_NAMES
        );
        assert!(
            vector
                .values
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
        let expected = [
            0.1 / 1.1,
            0.2 / 1.2,
            0.3 / 1.3,
            0.01 / 1.01,
            0.02 / 1.02,
            0.03 / 1.03,
            0.001 / 1.001,
            0.002 / 1.002,
            0.003 / 1.003,
            0.4 / 1.4,
            0.5 / 1.5,
            0.6 / 1.6,
            0.07 / std::f32::consts::LN_2,
            0.08 / std::f32::consts::LN_2,
            0.09 / std::f32::consts::LN_2,
            (-1.1_f32).exp(),
            (-2.2_f32).exp(),
            (-3.3_f32).exp(),
            (0.02 / 3.0) / (1.0 + 0.02 / 3.0),
            (0.08 / 3.0) / (1.0 + 0.08 / 3.0),
            0.06 / 1.06,
            0.02 / 1.02,
            0.6,
            0.75,
        ];
        for (actual, expected) in vector.values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert!((0.0..=3.0).contains(&vector.effective_rank));
    }

    #[test]
    fn controller_protocol_and_normalisation_kinds_are_fixed() {
        assert_eq!(NCKA_PROTOCOL_VERSION, 1);
        assert_eq!(NCKA_CONTROLLER_TYPE, "finite_moment_ka_v1");
        assert_eq!(
            NCKA_NORMALISATION_CONTRACT.len(),
            NCKA_COORDINATE_NAMES.len()
        );
        assert_eq!(
            NCKA_NORMALISATION_CONTRACT[12],
            NckaNormalisation::JensenShannonLn2
        );
        assert_eq!(
            NCKA_NORMALISATION_CONTRACT[15],
            NckaNormalisation::LogLikelihoodExp
        );
        assert_eq!(
            NCKA_NORMALISATION_CONTRACT[22],
            NckaNormalisation::LatencyMultiplier
        );
        assert_eq!(
            NCKA_NORMALISATION_CONTRACT[23],
            NckaNormalisation::UnitInterval
        );
    }

    #[test]
    fn invalid_raw_observable_domains_fail_closed() {
        let mut cases = Vec::new();
        let mut negative_entropy = input();
        negative_entropy.branches.branch_entropy[0] = -0.1;
        cases.push(negative_entropy);
        let mut oversized_js = input();
        oversized_js.pairwise_js[0] = std::f32::consts::LN_2 + 0.01;
        cases.push(oversized_js);
        let mut positive_log_likelihood = input();
        positive_log_likelihood.cross_scores[0][0] = 4.0;
        positive_log_likelihood.cross_scores[0][1] = 4.0;
        positive_log_likelihood.cross_scores[0][2] = 4.0;
        cases.push(positive_log_likelihood);
        let mut invalid_latency = input();
        invalid_latency.latency_multiplier = 0.5;
        cases.push(invalid_latency);
        let mut invalid_memory = input();
        invalid_memory.memory_ratio = 1.5;
        cases.push(invalid_memory);

        for case in cases {
            assert!(matches!(
                CouncilMomentVector::from_input(&case, 2),
                Err(MomentError::InvalidObservableRange { .. })
            ));
        }
    }

    #[test]
    fn truncated_gram_is_simultaneously_s3_equivariant() {
        let original = input();
        let permutation = [2, 0, 1];
        let mut permuted = original.clone();
        for (new_index, old_index) in permutation.into_iter().enumerate() {
            permuted.branches.branch_entropy[new_index] =
                original.branches.branch_entropy[old_index];
            permuted.branches.orthogonality_error[new_index] =
                original.branches.orthogonality_error[old_index];
            permuted.branches.determinant_error[new_index] =
                original.branches.determinant_error[old_index];
            permuted.branches.expected_quantisation_error[new_index] =
                original.branches.expected_quantisation_error[old_index];
            for (new_evaluator, old_evaluator) in permutation.into_iter().enumerate() {
                permuted.cross_scores[new_index][new_evaluator] =
                    original.cross_scores[old_index][old_evaluator];
            }
        }
        for (new_pair, (left, right)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            permuted.pairwise_js[new_pair] =
                original.pairwise_js[pair_index(permutation[left], permutation[right])];
        }

        let before = truncated_moment_gram(&original, 3).unwrap();
        let after = truncated_moment_gram(&permuted, 3).unwrap();
        for left in 0..3 {
            for right in 0..3 {
                assert!(
                    (after.values[left][right]
                        - before.values[permutation[left]][permutation[right]])
                        .abs()
                        < 1.0e-10
                );
            }
        }
        assert_eq!(before.rank.numerical_rank, after.rank.numerical_rank);
        assert!((before.rank.effective_rank - after.rank.effective_rank).abs() < 1.0e-5);

        let before = CouncilMomentVector::from_input(&original, 3).unwrap();
        let after = CouncilMomentVector::from_input(&permuted, 3).unwrap();
        for start in [0, 3, 6, 9, 15, 18] {
            for (new_index, old_index) in permutation.into_iter().enumerate() {
                assert_eq!(
                    after.values[start + new_index],
                    before.values[start + old_index]
                );
            }
        }
        for (new_pair, (left, right)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            assert_eq!(
                after.values[12 + new_pair],
                before.values[12 + pair_index(permutation[left], permutation[right])]
            );
        }
        assert_eq!(&after.values[21..], &before.values[21..]);
    }
}
