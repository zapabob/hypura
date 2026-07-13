pub mod aha;
pub mod cross_score;
pub mod embedded_ka;
pub mod ka_gate;
pub mod moments;
pub mod runtime;
pub mod scoring;
pub mod store;
pub mod types;

pub use aha::{
    AHA_MAX_FALSE_POSITIVE_RATE, AHA_TRUTH_DISCLAIMER, AhaCalibrationEvidence, AhaDisabledReason,
    AhaEvaluation, AhaEvent, AhaEvidence, AhaInput, AhaMode, AhaReasonCode, AhaSafetyEvidence,
    AhaThresholds, classify_aha, classify_aha_with_status,
};
pub use cross_score::{
    CandidateViewScore, CrossScoreMatrix, TeacherForcedScoreInput, token_log_probability,
};
pub use embedded_ka::{
    EmbeddedKaController, EmbeddedKaControllerError, prepare_embedded_ka_controller,
};
pub use ka_gate::{
    GateSource, KaController, KaGateConfig, KaGateError, KaGateEvaluation, KaGateOutput,
    evaluate_ka_gate,
};
pub use moments::{
    BranchMomentObservables, CouncilMomentInput, CouncilMomentVector, MomentError,
    NCKA_CONTROLLER_TYPE, NCKA_COORDINATE_NAMES, NCKA_NORMALISATION_CONTRACT,
    NCKA_PROTOCOL_VERSION, NckaNormalisation, RankStatistics, TruncatedMomentGram,
    branch_feature_rows, truncated_moment_gram,
};
pub use runtime::{
    AttentionCapabilityDecision, CouncilAhaRuntimeEvidence, CouncilExecutionResult, CouncilRuntime,
    CouncilRuntimeConfig, CouncilUrtDescriptor, context_config_for_view,
};
pub use scoring::{
    AnswerCouncilConfig, AnswerCouncilResult, NoSafetyPenalty, SafetyPenalty, agreement_scores,
    repeated_fourgram_ratio, select_answer,
};
pub use store::{
    CouncilArtifactPolicy, CouncilInputKind, CouncilRequestRecord, CouncilStore,
    CouncilStoreConfig, CouncilStoreError, DEFAULT_COUNCIL_RETENTION_RECORDS,
    PersistedCouncilRecord, REDACTED_FINAL_ANSWER, StoredConsensusResult, StoredCouncilCandidate,
    StoredCouncilRecord,
};
pub use types::{CouncilCandidate, CouncilView, TrialityBranchMetrics, TrialityConsensusMetrics};
