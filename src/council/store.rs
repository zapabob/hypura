use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use super::{AnswerCouncilResult, CouncilView};
use crate::scheduler::types::CouncilParallelism;

pub const DEFAULT_COUNCIL_RETENTION_RECORDS: usize = 100;
pub const REDACTED_FINAL_ANSWER: &str = "[redacted by council storage policy]\n";

const COUNCIL_ARTIFACT_FILES: [&str; 6] = [
    "request.json",
    "branch_candidates.json",
    "cross_scores.csv",
    "consensus_result.json",
    "aha_event.json",
    "final_answer.txt",
];
const MAX_COUNCIL_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouncilInputKind {
    Prompt,
    Messages,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CouncilRequestRecord {
    pub request_id: String,
    pub created_at: DateTime<Utc>,
    pub model: Option<String>,
    pub input_kind: CouncilInputKind,
    pub message_count: Option<usize>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub seed: Option<u32>,
    pub parallelism: CouncilParallelism,
    pub attention_consensus: bool,
    pub cross_score: bool,
    pub synthesis: bool,
    pub aha: bool,
    pub trace: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CouncilArtifactPolicy {
    pub persist_metadata: bool,
    pub persist_branch_content: bool,
    pub persist_final_answer: bool,
}

impl Default for CouncilArtifactPolicy {
    fn default() -> Self {
        Self {
            persist_metadata: true,
            persist_branch_content: false,
            persist_final_answer: false,
        }
    }
}

impl CouncilArtifactPolicy {
    pub const fn disabled() -> Self {
        Self {
            persist_metadata: false,
            persist_branch_content: false,
            persist_final_answer: false,
        }
    }

    pub const fn trace_content() -> Self {
        Self {
            persist_metadata: true,
            persist_branch_content: true,
            persist_final_answer: true,
        }
    }

    pub const fn branch_content_permitted(self, trace_enabled: bool) -> bool {
        self.persist_metadata && self.persist_branch_content && trace_enabled
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CouncilStoreConfig {
    pub data_root: PathBuf,
    pub retention_records: usize,
    pub policy: CouncilArtifactPolicy,
}

impl CouncilStoreConfig {
    pub fn app_data_default() -> Result<Self, CouncilStoreError> {
        Ok(Self {
            data_root: app_data_root()?,
            retention_records: DEFAULT_COUNCIL_RETENTION_RECORDS,
            policy: CouncilArtifactPolicy::default(),
        })
    }

    pub fn for_data_root(data_root: impl Into<PathBuf>) -> Self {
        Self {
            data_root: data_root.into(),
            retention_records: DEFAULT_COUNCIL_RETENTION_RECORDS,
            policy: CouncilArtifactPolicy::default(),
        }
    }

    pub fn artifact_root(&self) -> PathBuf {
        self.data_root.join("artifacts").join("triality_council")
    }
}

#[derive(Debug)]
pub struct CouncilStore {
    config: CouncilStoreConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedCouncilRecord {
    pub directory: PathBuf,
    pub branch_content_persisted: bool,
    pub final_answer_persisted: bool,
}

#[derive(Debug, Error)]
pub enum CouncilStoreError {
    #[error("Hypura application data root is unavailable")]
    AppDataRootUnavailable,
    #[error("council data root must be an absolute non-root path")]
    InvalidDataRoot,
    #[error("council retention must keep at least one request record")]
    InvalidRetention,
    #[error("council content policy requires metadata persistence")]
    InvalidPolicy,
    #[error("request ID is not a portable path-safe identifier")]
    InvalidRequestId,
    #[error("request ID differs between request metadata and council result")]
    RequestIdMismatch,
    #[error("council record already exists for request ID {0}")]
    RecordAlreadyExists(String),
    #[error("retention target failed the council artifact boundary check")]
    UnsafeRetentionTarget,
    #[error("council record contains a non-finite numeric value")]
    InvalidNumericValue,
    #[error("persisted council record failed schema or consistency validation")]
    InvalidPersistedSchema,
    #[error("council publication failure injected for regression testing")]
    PublicationFailureInjected,
    #[error(
        "council operation failed ({operation}) and retention rollback also failed: {rollback}"
    )]
    RetentionRollbackFailed { operation: String, rollback: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl CouncilStore {
    pub fn open(config: CouncilStoreConfig) -> Result<Self, CouncilStoreError> {
        if !valid_data_root(&config.data_root) {
            return Err(CouncilStoreError::InvalidDataRoot);
        }
        if config.retention_records == 0 {
            return Err(CouncilStoreError::InvalidRetention);
        }
        if !config.policy.persist_metadata
            && (config.policy.persist_branch_content || config.policy.persist_final_answer)
        {
            return Err(CouncilStoreError::InvalidPolicy);
        }
        let store = Self { config };
        let _recovery_lock = acquire_recovered_root_lock(&store.config)?;
        Ok(store)
    }

    pub fn config(&self) -> &CouncilStoreConfig {
        &self.config
    }

    pub fn record_directory(&self, request_id: &str) -> Result<PathBuf, CouncilStoreError> {
        validate_request_id(request_id)?;
        Ok(self.config.artifact_root().join(request_id))
    }

    pub fn persist(
        &self,
        request: &CouncilRequestRecord,
        result: &AnswerCouncilResult,
    ) -> Result<Option<PersistedCouncilRecord>, CouncilStoreError> {
        self.persist_inner(request, result, false)
    }

    #[cfg(debug_assertions)]
    #[doc(hidden)]
    pub fn persist_with_injected_publication_failure(
        &self,
        request: &CouncilRequestRecord,
        result: &AnswerCouncilResult,
    ) -> Result<Option<PersistedCouncilRecord>, CouncilStoreError> {
        self.persist_inner(request, result, true)
    }

    fn persist_inner(
        &self,
        request: &CouncilRequestRecord,
        result: &AnswerCouncilResult,
        inject_publication_failure: bool,
    ) -> Result<Option<PersistedCouncilRecord>, CouncilStoreError> {
        if !self.config.policy.persist_metadata {
            return Ok(None);
        }
        validate_request_id(&request.request_id)?;
        if request.request_id != result.request_id {
            return Err(CouncilStoreError::RequestIdMismatch);
        }
        validate_numeric_values(request, result)?;

        let root = self.config.artifact_root();
        create_private_dir_all(&root)?;
        validate_store_root(&self.config.data_root, &root)?;
        let _lock = StoreRootLock::acquire(&root)?;
        validate_store_root(&self.config.data_root, &root)?;
        cleanup_internal_directories(&root)?;
        let destination = root.join(&request.request_id);
        if path_entry_exists(&destination)? {
            return Err(CouncilStoreError::RecordAlreadyExists(
                request.request_id.clone(),
            ));
        }

        let staging = root.join(format!(".pending-{}", Uuid::new_v4()));
        create_private_dir(&staging)?;
        let branch_content_persisted = self.config.policy.branch_content_permitted(request.trace);
        let final_answer_persisted = self.config.policy.persist_final_answer;
        let write_result = self.write_record(
            &staging,
            request,
            result,
            branch_content_persisted,
            final_answer_persisted,
        );
        if let Err(error) = write_result {
            let _ = cleanup_failed_staging(&root, &staging);
            return Err(error);
        }
        let mut quarantined = match self.prepare_retention(&root) {
            Ok(quarantined) => quarantined,
            Err(error) => {
                let _ = cleanup_failed_staging(&root, &staging);
                return Err(error);
            }
        };
        let publication = publish_staging(
            &staging,
            &destination,
            &request.request_id,
            inject_publication_failure,
        );
        if let Err(error) = publication {
            let rollback = rollback_quarantined(&root, &mut quarantined);
            let _ = cleanup_failed_staging(&root, &staging);
            if let Err(rollback) = rollback {
                return Err(CouncilStoreError::RetentionRollbackFailed {
                    operation: error.to_string(),
                    rollback: rollback.to_string(),
                });
            }
            return Err(error);
        }

        commit_quarantined(&root, &mut quarantined);

        Ok(Some(PersistedCouncilRecord {
            directory: destination,
            branch_content_persisted,
            final_answer_persisted,
        }))
    }

    pub fn read(&self, request_id: &str) -> Result<Option<StoredCouncilRecord>, CouncilStoreError> {
        validate_request_id(request_id)?;
        if !self.config.policy.persist_metadata {
            return Ok(None);
        }

        let root = self.config.artifact_root();
        let Some(_lock) = acquire_recovered_root_lock(&self.config)? else {
            return Ok(None);
        };

        let directory = root.join(request_id);
        match fs::symlink_metadata(&directory) {
            Ok(_) => validate_read_record_directory(&root, &directory)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        }

        let request_bytes = read_plain_artifact(&directory, "request.json")?;
        let candidates_bytes = read_plain_artifact(&directory, "branch_candidates.json")?;
        let cross_scores_bytes = read_plain_artifact(&directory, "cross_scores.csv")?;
        let consensus_bytes = read_plain_artifact(&directory, "consensus_result.json")?;
        let aha_bytes = read_plain_artifact(&directory, "aha_event.json")?;
        let final_answer_bytes = read_plain_artifact(&directory, "final_answer.txt")?;

        let request: CouncilRequestRecord = parse_persisted_json(&request_bytes)?;
        let candidate_value = parse_persisted_json_value(&candidates_bytes)?;
        validate_branch_candidates_json_schema(&candidate_value)?;
        let mut branch_candidates: [StoredCouncilCandidate; 3] =
            serde_json::from_value(candidate_value)
                .map_err(|_| CouncilStoreError::InvalidPersistedSchema)?;
        let cross_scores = parse_cross_scores_csv(&cross_scores_bytes)?;
        let consensus: StoredConsensusResult = parse_persisted_json(&consensus_bytes)?;
        let aha_value = parse_persisted_json_value(&aha_bytes)?;
        validate_aha_json_schema(&aha_value)?;
        let aha: Option<super::AhaEvent> = serde_json::from_value(aha_value)
            .map_err(|_| CouncilStoreError::InvalidPersistedSchema)?;
        let mut final_answer = String::from_utf8(final_answer_bytes)
            .map_err(|_| CouncilStoreError::InvalidPersistedSchema)?;

        validate_persisted_record(
            request_id,
            &request,
            &branch_candidates,
            &cross_scores,
            &consensus,
            aha.as_ref(),
        )?;

        let branch_content_redacted = !self.config.policy.branch_content_permitted(request.trace)
            || branch_candidates
                .iter()
                .any(|candidate| candidate.text.is_none() || candidate.token_ids.is_none());
        if branch_content_redacted {
            for candidate in &mut branch_candidates {
                candidate.text = None;
                candidate.token_ids = None;
            }
        }

        let final_answer_redacted =
            !self.config.policy.persist_final_answer || final_answer == REDACTED_FINAL_ANSWER;
        if final_answer_redacted {
            final_answer.clear();
            final_answer.push_str(REDACTED_FINAL_ANSWER);
        }

        Ok(Some(StoredCouncilRecord {
            request,
            branch_candidates,
            cross_scores,
            consensus,
            aha,
            final_answer,
            branch_content_redacted,
            final_answer_redacted,
        }))
    }

    fn write_record(
        &self,
        staging: &Path,
        request: &CouncilRequestRecord,
        result: &AnswerCouncilResult,
        persist_branch_content: bool,
        persist_final_answer: bool,
    ) -> Result<(), CouncilStoreError> {
        write_json(staging.join("request.json"), request)?;

        let candidates = result
            .candidates
            .iter()
            .map(|candidate| StoredCouncilCandidate {
                view: candidate.view,
                prompt_tokens: candidate.prompt_tokens,
                generated_tokens: candidate.generated_tokens,
                tok_per_sec: candidate.tok_per_sec,
                runtime_ms: candidate.runtime_ms,
                low_level_metrics: candidate.low_level_metrics.clone(),
                text: persist_branch_content.then(|| candidate.text.clone()),
                token_ids: persist_branch_content.then(|| candidate.token_ids.clone()),
            })
            .collect::<Vec<_>>();
        write_json(staging.join("branch_candidates.json"), &candidates)?;
        write_new_file(
            &staging.join("cross_scores.csv"),
            cross_scores_csv(result).as_bytes(),
        )?;
        write_json(
            staging.join("consensus_result.json"),
            &StoredConsensusResult {
                request_id: result.request_id.clone(),
                selected_view: result.selected_view,
                candidate_scores: result.candidate_scores,
                winner_margin: result.winner_margin,
                agreement: result.agreement,
            },
        )?;
        write_json(staging.join("aha_event.json"), &result.aha)?;
        let final_answer = if persist_final_answer {
            result.selected_text.as_bytes()
        } else {
            REDACTED_FINAL_ANSWER.as_bytes()
        };
        write_new_file(&staging.join("final_answer.txt"), final_answer)?;
        Ok(())
    }

    fn prepare_retention(
        &self,
        root: &Path,
    ) -> Result<Vec<QuarantinedCouncilRecord>, CouncilStoreError> {
        let mut records = list_owned_records(root)?;
        records.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let remove_count = records
            .len()
            .saturating_add(1)
            .saturating_sub(self.config.retention_records);
        let mut quarantined = Vec::with_capacity(remove_count);
        for (_, name, path) in records.into_iter().take(remove_count) {
            match quarantine_record(root, &path, &name) {
                Ok(record) => quarantined.push(record),
                Err(error) => {
                    if let Err(rollback) = rollback_quarantined(root, &mut quarantined) {
                        return Err(CouncilStoreError::RetentionRollbackFailed {
                            operation: error.to_string(),
                            rollback: rollback.to_string(),
                        });
                    }
                    return Err(error);
                }
            }
        }
        Ok(quarantined)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredCouncilCandidate {
    pub view: CouncilView,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub tok_per_sec: f64,
    pub runtime_ms: u64,
    pub low_level_metrics: Option<super::TrialityConsensusMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<i32>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredConsensusResult {
    pub request_id: String,
    pub selected_view: CouncilView,
    pub candidate_scores: [f64; 3],
    pub winner_margin: f64,
    pub agreement: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredCouncilRecord {
    pub request: CouncilRequestRecord,
    pub branch_candidates: [StoredCouncilCandidate; 3],
    pub cross_scores: super::CrossScoreMatrix,
    pub consensus: StoredConsensusResult,
    pub aha: Option<super::AhaEvent>,
    pub final_answer: String,
    pub branch_content_redacted: bool,
    pub final_answer_redacted: bool,
}

fn valid_data_root(path: &Path) -> bool {
    path.is_absolute()
        && !path.as_os_str().is_empty()
        && path.parent().is_some_and(|parent| parent != path)
}

fn validate_request_id(request_id: &str) -> Result<(), CouncilStoreError> {
    let bytes = request_id.as_bytes();
    let valid = (1..=128).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
    if valid && !is_windows_reserved_name(request_id) {
        Ok(())
    } else {
        Err(CouncilStoreError::InvalidRequestId)
    }
}

fn validate_numeric_values(
    request: &CouncilRequestRecord,
    result: &AnswerCouncilResult,
) -> Result<(), CouncilStoreError> {
    let request_is_finite = request.temperature.is_none_or(f32::is_finite);
    let result_is_finite = result
        .candidate_scores
        .iter()
        .all(|value| value.is_finite())
        && result.winner_margin.is_finite()
        && result.agreement.is_finite()
        && result
            .cross_scores
            .scores
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        && result
            .candidates
            .iter()
            .all(candidate_numeric_values_are_finite)
        && result
            .aha
            .as_ref()
            .is_none_or(aha_numeric_values_are_finite);
    if request_is_finite && result_is_finite {
        Ok(())
    } else {
        Err(CouncilStoreError::InvalidNumericValue)
    }
}

fn cross_scores_csv(result: &AnswerCouncilResult) -> String {
    let mut csv =
        String::from("candidate_view,vector,spinor_plus_proxy,spinor_minus_proxy,token_count\n");
    for view in CouncilView::ALL {
        let row = result.cross_scores.scores[view.index()];
        let count = result.cross_scores.token_counts[view.index()];
        csv.push_str(&format!(
            "{},{},{},{},{}\n",
            view.as_str(),
            row[0],
            row[1],
            row[2],
            count
        ));
    }
    csv
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), CouncilStoreError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new_file(&path, &bytes)?;
    Ok(())
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn parse_persisted_json<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CouncilStoreError> {
    serde_json::from_slice(bytes).map_err(|_| CouncilStoreError::InvalidPersistedSchema)
}

fn parse_persisted_json_value(bytes: &[u8]) -> Result<serde_json::Value, CouncilStoreError> {
    parse_persisted_json(bytes)
}

fn validate_branch_candidates_json_schema(
    value: &serde_json::Value,
) -> Result<(), CouncilStoreError> {
    let candidates = value
        .as_array()
        .filter(|candidates| candidates.len() == CouncilView::ALL.len())
        .ok_or(CouncilStoreError::InvalidPersistedSchema)?;
    for candidate in candidates {
        let candidate = candidate
            .as_object()
            .ok_or(CouncilStoreError::InvalidPersistedSchema)?;
        ensure_exact_json_keys(
            candidate,
            &[
                "view",
                "prompt_tokens",
                "generated_tokens",
                "tok_per_sec",
                "runtime_ms",
                "low_level_metrics",
            ],
            &["text", "token_ids"],
        )?;
        let Some(metrics) = candidate.get("low_level_metrics") else {
            return Err(CouncilStoreError::InvalidPersistedSchema);
        };
        if metrics.is_null() {
            continue;
        }
        let metrics = metrics
            .as_object()
            .ok_or(CouncilStoreError::InvalidPersistedSchema)?;
        ensure_exact_json_keys(
            metrics,
            &[
                "branches",
                "pairwise_js",
                "mean_pairwise_js",
                "max_pairwise_js",
                "numerical_rank",
                "effective_rank",
                "ka_fallback_used",
                "operator_word_hash_128",
            ],
            &[],
        )?;
        let branches = metrics
            .get("branches")
            .and_then(serde_json::Value::as_array)
            .filter(|branches| branches.len() == CouncilView::ALL.len())
            .ok_or(CouncilStoreError::InvalidPersistedSchema)?;
        for branch in branches {
            let branch = branch
                .as_object()
                .ok_or(CouncilStoreError::InvalidPersistedSchema)?;
            ensure_exact_json_keys(
                branch,
                &[
                    "logit_mean",
                    "logit_variance",
                    "logit_l2",
                    "probability_entropy",
                    "top1_probability",
                    "orthogonality_error",
                    "determinant_error",
                    "expected_quantisation_error",
                    "bytes_read",
                    "duration_us",
                ],
                &[],
            )?;
        }
    }
    Ok(())
}

fn validate_aha_json_schema(value: &serde_json::Value) -> Result<(), CouncilStoreError> {
    if value.is_null() {
        return Ok(());
    }
    let event = value
        .as_object()
        .ok_or(CouncilStoreError::InvalidPersistedSchema)?;
    ensure_exact_json_keys(
        event,
        &[
            "schema_version",
            "mode",
            "reason_code",
            "selected_view",
            "baseline_view",
            "pre_consensus_js",
            "post_consensus_js",
            "score_gain",
            "winner_margin",
            "urt_pre_error",
            "urt_post_error",
            "moment_effective_rank",
            "message",
            "truth_disclaimer",
        ],
        &[],
    )
}

fn ensure_exact_json_keys(
    object: &serde_json::Map<String, serde_json::Value>,
    required: &[&str],
    optional: &[&str],
) -> Result<(), CouncilStoreError> {
    let expected_len = required.len()
        + optional
            .iter()
            .filter(|key| object.contains_key::<str>(*key))
            .count();
    if object.len() != expected_len
        || required.iter().any(|key| !object.contains_key(*key))
        || object
            .keys()
            .any(|key| !required.contains(&key.as_str()) && !optional.contains(&key.as_str()))
    {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    Ok(())
}

fn parse_cross_scores_csv(bytes: &[u8]) -> Result<super::CrossScoreMatrix, CouncilStoreError> {
    let csv = std::str::from_utf8(bytes).map_err(|_| CouncilStoreError::InvalidPersistedSchema)?;
    let lines = csv.lines().collect::<Vec<_>>();
    if lines.len() != 4
        || lines[0] != "candidate_view,vector,spinor_plus_proxy,spinor_minus_proxy,token_count"
    {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    let mut scores = [[0.0; 3]; 3];
    let mut token_counts = [0; 3];
    for (index, view) in CouncilView::ALL.into_iter().enumerate() {
        let fields = lines[index + 1].split(',').collect::<Vec<_>>();
        if fields.len() != 5 || fields[0] != view.as_str() {
            return Err(CouncilStoreError::InvalidPersistedSchema);
        }
        for evaluator in 0..3 {
            scores[index][evaluator] = fields[evaluator + 1]
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .ok_or(CouncilStoreError::InvalidPersistedSchema)?;
        }
        token_counts[index] = fields[4]
            .parse::<u32>()
            .ok()
            .filter(|count| *count > 0)
            .ok_or(CouncilStoreError::InvalidPersistedSchema)?;
    }
    super::CrossScoreMatrix::try_new(scores, token_counts)
        .map_err(|_| CouncilStoreError::InvalidPersistedSchema)
}

fn validate_persisted_record(
    expected_request_id: &str,
    request: &CouncilRequestRecord,
    candidates: &[StoredCouncilCandidate; 3],
    cross_scores: &super::CrossScoreMatrix,
    consensus: &StoredConsensusResult,
    aha: Option<&super::AhaEvent>,
) -> Result<(), CouncilStoreError> {
    if request.request_id != expected_request_id || consensus.request_id != expected_request_id {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    let content_present = candidates[0].text.is_some() && candidates[0].token_ids.is_some();
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.view != CouncilView::ALL[index]
            || candidate.text.is_some() != candidate.token_ids.is_some()
            || candidate.text.is_some() != content_present
            || (!request.trace && candidate.text.is_some())
            || candidate.token_ids.as_ref().is_some_and(|tokens| {
                usize::try_from(candidate.generated_tokens).ok() != Some(tokens.len())
            })
            || cross_scores.token_counts[index] != candidate.generated_tokens
            || !stored_candidate_numeric_values_are_finite(candidate)
        {
            return Err(CouncilStoreError::InvalidPersistedSchema);
        }
    }
    if request.temperature.is_some_and(|value| !value.is_finite())
        || consensus
            .candidate_scores
            .iter()
            .any(|value| !value.is_finite())
        || !consensus.winner_margin.is_finite()
        || !consensus.agreement.is_finite()
        || aha.is_some_and(|event| {
            event.schema_version != 1
                || event.selected_view != consensus.selected_view
                || !aha_numeric_values_are_finite(event)
        })
    {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    Ok(())
}

fn validate_store_root(data_root: &Path, root: &Path) -> Result<(), CouncilStoreError> {
    let artifacts = data_root.join("artifacts");
    for directory in [data_root, artifacts.as_path(), root] {
        let metadata = fs::symlink_metadata(directory)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
            return Err(CouncilStoreError::InvalidPersistedSchema);
        }
    }
    let data_root = fs::canonicalize(data_root)?;
    let artifacts = fs::canonicalize(artifacts)?;
    let root = fs::canonicalize(root)?;
    if artifacts.parent() != Some(data_root.as_path()) || root.parent() != Some(artifacts.as_path())
    {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    Ok(())
}

fn validate_read_record_directory(root: &Path, directory: &Path) -> Result<(), CouncilStoreError> {
    validate_plain_directory_under_root(root, directory)
        .map_err(|_| CouncilStoreError::InvalidPersistedSchema)?;
    let mut found = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| CouncilStoreError::InvalidPersistedSchema)?;
        if !COUNCIL_ARTIFACT_FILES.contains(&name.as_str()) {
            return Err(CouncilStoreError::InvalidPersistedSchema);
        }
        found.push(name);
    }
    found.sort_unstable();
    let mut expected = COUNCIL_ARTIFACT_FILES.to_vec();
    expected.sort_unstable();
    if found != expected {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    Ok(())
}

fn read_plain_artifact(directory: &Path, name: &str) -> Result<Vec<u8>, CouncilStoreError> {
    if !COUNCIL_ARTIFACT_FILES.contains(&name) {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    let path = directory.join(name);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    let directory = fs::canonicalize(directory)?;
    let canonical = fs::canonicalize(&path)?;
    if canonical.parent() != Some(directory.as_path()) {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;

        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(&path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || is_reparse_point(&metadata)
        || metadata.len() > MAX_COUNCIL_ARTIFACT_BYTES
    {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
    file.take(MAX_COUNCIL_ARTIFACT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_COUNCIL_ARTIFACT_BYTES {
        return Err(CouncilStoreError::InvalidPersistedSchema);
    }
    Ok(bytes)
}

fn path_entry_exists(path: &Path) -> Result<bool, std::io::Error> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn list_owned_records(root: &Path) -> Result<Vec<(i128, String, PathBuf)>, CouncilStoreError> {
    let mut records = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if !file_type.is_dir() || file_type.is_symlink() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if validate_request_id(&name).is_err() {
            continue;
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if is_reparse_point(&metadata) {
            return Err(CouncilStoreError::UnsafeRetentionTarget);
        }
        if let Some(ordering_time) = record_ordering_time(&path, &name)? {
            records.push((ordering_time, name, path));
        }
    }
    Ok(records)
}

struct QuarantinedCouncilRecord {
    request_id: String,
    original: PathBuf,
    quarantine: PathBuf,
}

fn publish_staging(
    staging: &Path,
    destination: &Path,
    request_id: &str,
    inject_failure: bool,
) -> Result<(), CouncilStoreError> {
    if inject_failure {
        return Err(CouncilStoreError::PublicationFailureInjected);
    }
    if path_entry_exists(destination)? {
        return Err(CouncilStoreError::RecordAlreadyExists(
            request_id.to_string(),
        ));
    }
    if let Err(error) = rename_noreplace(staging, destination) {
        if path_entry_exists(destination)? {
            return Err(CouncilStoreError::RecordAlreadyExists(
                request_id.to_string(),
            ));
        }
        return Err(error.into());
    }
    Ok(())
}

fn quarantine_record(
    root: &Path,
    path: &Path,
    request_id: &str,
) -> Result<QuarantinedCouncilRecord, CouncilStoreError> {
    validate_retention_target(root, path, request_id)?;
    let quarantine = root.join(format!(".rollback-{}", Uuid::new_v4()));
    rename_noreplace(path, &quarantine)?;
    let record = QuarantinedCouncilRecord {
        request_id: request_id.to_string(),
        original: path.to_path_buf(),
        quarantine,
    };
    if let Err(error) = validate_retention_target(root, &record.quarantine, request_id) {
        if let Err(rollback) = restore_quarantined_record(root, &record) {
            return Err(CouncilStoreError::RetentionRollbackFailed {
                operation: error.to_string(),
                rollback: rollback.to_string(),
            });
        }
        return Err(error);
    }
    Ok(record)
}

fn rollback_quarantined(
    root: &Path,
    quarantined: &mut Vec<QuarantinedCouncilRecord>,
) -> Result<(), CouncilStoreError> {
    let mut failed = Vec::new();
    let mut first_error = None;
    while let Some(record) = quarantined.pop() {
        if let Err(error) = restore_quarantined_record(root, &record) {
            if first_error.is_none() {
                first_error = Some(error);
            }
            failed.push(record);
        }
    }
    failed.reverse();
    quarantined.extend(failed);
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn restore_quarantined_record(
    root: &Path,
    record: &QuarantinedCouncilRecord,
) -> Result<(), CouncilStoreError> {
    validate_retention_target(root, &record.quarantine, &record.request_id)?;
    if path_entry_exists(&record.original)? {
        return Err(CouncilStoreError::UnsafeRetentionTarget);
    }
    rename_noreplace(&record.quarantine, &record.original)?;
    validate_retention_target(root, &record.original, &record.request_id)
}

fn commit_quarantined(root: &Path, quarantined: &mut Vec<QuarantinedCouncilRecord>) {
    while let Some(record) = quarantined.pop() {
        if validate_retention_target(root, &record.quarantine, &record.request_id).is_err() {
            continue;
        }
        let purge = root.join(format!(".purge-{}", Uuid::new_v4()));
        if rename_noreplace(&record.quarantine, &purge).is_err() {
            continue;
        }
        if validate_retention_target(root, &purge, &record.request_id).is_err() {
            continue;
        }
        let _ = fs::remove_dir_all(purge);
    }
}

fn cleanup_failed_staging(root: &Path, staging: &Path) -> Result<(), CouncilStoreError> {
    if !path_entry_exists(staging)? {
        return Ok(());
    }
    validate_plain_directory_under_root(root, staging)?;
    fs::remove_dir_all(staging)?;
    Ok(())
}

fn cleanup_internal_directories(root: &Path) -> Result<(), CouncilStoreError> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if internal_uuid_suffix(&name, ".rollback-").is_some() {
            restore_stale_rollback(root, &entry.path())?;
        } else if internal_uuid_suffix(&name, ".pending-").is_some()
            || internal_uuid_suffix(&name, ".purge-").is_some()
        {
            validate_plain_directory_under_root(root, &entry.path())?;
            fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

fn acquire_recovered_root_lock(
    config: &CouncilStoreConfig,
) -> Result<Option<StoreRootLock>, CouncilStoreError> {
    let root = config.artifact_root();
    match fs::symlink_metadata(&root) {
        Ok(_) => validate_store_root(&config.data_root, &root)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    let lock = StoreRootLock::acquire(&root)?;
    validate_store_root(&config.data_root, &root)?;
    cleanup_internal_directories(&root)?;
    Ok(Some(lock))
}

fn internal_uuid_suffix<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    let suffix = name.strip_prefix(prefix)?;
    Uuid::parse_str(suffix).ok().map(|_| suffix)
}

fn restore_stale_rollback(root: &Path, quarantine: &Path) -> Result<(), CouncilStoreError> {
    validate_plain_directory_under_root(root, quarantine)?;
    let request: CouncilRequestRecord =
        parse_persisted_json(&read_plain_artifact(quarantine, "request.json")?)?;
    validate_request_id(&request.request_id)?;
    let record = QuarantinedCouncilRecord {
        request_id: request.request_id.clone(),
        original: root.join(&request.request_id),
        quarantine: quarantine.to_path_buf(),
    };
    restore_quarantined_record(root, &record)
}

fn candidate_numeric_values_are_finite(candidate: &super::CouncilCandidate) -> bool {
    candidate.tok_per_sec.is_finite()
        && candidate
            .low_level_metrics
            .as_ref()
            .is_none_or(triality_metrics_are_finite)
}

fn stored_candidate_numeric_values_are_finite(candidate: &StoredCouncilCandidate) -> bool {
    candidate.tok_per_sec.is_finite()
        && candidate
            .low_level_metrics
            .as_ref()
            .is_none_or(triality_metrics_are_finite)
}

fn triality_metrics_are_finite(metrics: &super::TrialityConsensusMetrics) -> bool {
    metrics.branches.iter().all(|branch| {
        branch.logit_mean.is_finite()
            && branch.logit_variance.is_finite()
            && branch.logit_l2.is_finite()
            && branch.probability_entropy.is_finite()
            && branch.top1_probability.is_finite()
            && branch.orthogonality_error.is_finite()
            && branch.determinant_error.is_finite()
            && branch.expected_quantisation_error.is_finite()
    }) && metrics.pairwise_js.iter().all(|value| value.is_finite())
        && metrics.mean_pairwise_js.is_finite()
        && metrics.max_pairwise_js.is_finite()
        && metrics.numerical_rank.is_finite()
        && metrics.effective_rank.is_finite()
}

fn aha_numeric_values_are_finite(event: &super::AhaEvent) -> bool {
    event.pre_consensus_js.is_finite()
        && event.post_consensus_js.is_none_or(f64::is_finite)
        && event.score_gain.is_finite()
        && event.winner_margin.is_finite()
        && event.urt_pre_error.is_none_or(f64::is_finite)
        && event.urt_post_error.is_none_or(f64::is_finite)
        && event.moment_effective_rank.is_none_or(f64::is_finite)
}

fn is_windows_reserved_name(request_id: &str) -> bool {
    let name = request_id.to_ascii_uppercase();
    matches!(name.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || name
            .strip_prefix("COM")
            .or_else(|| name.strip_prefix("LPT"))
            .is_some_and(|suffix| {
                matches!(suffix, "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9")
            })
}

fn create_private_dir_all(path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(path)?;
    set_private_directory_permissions(path)
}

fn create_private_dir(path: &Path) -> Result<(), std::io::Error> {
    fs::create_dir(path)?;
    set_private_directory_permissions(path)
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &Path) -> Result<(), std::io::Error> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &Path) -> Result<(), std::io::Error> {
    Ok(())
}

struct StoreRootLock {
    file: File,
}

impl StoreRootLock {
    fn acquire(root: &Path) -> Result<Self, std::io::Error> {
        let path = root.join(".store.lock");
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;

            options.mode(0o600);
            options.custom_flags(libc::O_NOFOLLOW);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;

            options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
        }
        let file = options.open(&path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || is_reparse_point(&metadata) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "council store lock is not a plain file",
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        lock_store_file(&file)?;
        Ok(Self { file })
    }
}

impl Drop for StoreRootLock {
    fn drop(&mut self) {
        unlock_store_file(&self.file);
    }
}

#[cfg(unix)]
fn lock_store_file(file: &File) -> Result<(), std::io::Error> {
    use std::os::fd::AsRawFd;

    // SAFETY: flock receives the live descriptor owned by file and no borrowed memory.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn unlock_store_file(file: &File) {
    use std::os::fd::AsRawFd;

    // SAFETY: flock receives the still-live descriptor owned by the guard.
    let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
}

#[cfg(windows)]
fn lock_store_file(file: &File) -> Result<(), std::io::Error> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{LOCKFILE_EXCLUSIVE_LOCK, LockFileEx};
    use windows_sys::Win32::System::IO::OVERLAPPED;

    // SAFETY: zero is a valid OVERLAPPED value for a synchronous whole-file lock.
    let mut overlapped = unsafe { std::mem::zeroed::<OVERLAPPED>() };
    // SAFETY: the handle and OVERLAPPED pointer remain valid for the duration of the call.
    let result = unsafe {
        LockFileEx(
            file.as_raw_handle(),
            LOCKFILE_EXCLUSIVE_LOCK,
            0,
            u32::MAX,
            u32::MAX,
            &mut overlapped,
        )
    };
    if result != 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(windows)]
fn unlock_store_file(file: &File) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::UnlockFileEx;
    use windows_sys::Win32::System::IO::OVERLAPPED;

    // SAFETY: zero is a valid OVERLAPPED value for the same whole-file region.
    let mut overlapped = unsafe { std::mem::zeroed::<OVERLAPPED>() };
    // SAFETY: the handle and OVERLAPPED pointer remain valid for the duration of the call.
    let _ = unsafe { UnlockFileEx(file.as_raw_handle(), 0, u32::MAX, u32::MAX, &mut overlapped) };
}

#[cfg(not(any(unix, windows)))]
fn lock_store_file(_file: &File) -> Result<(), std::io::Error> {
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn unlock_store_file(_file: &File) {}

fn record_ordering_time(path: &Path, request_id: &str) -> Result<Option<i128>, CouncilStoreError> {
    if let Ok(bytes) = read_plain_artifact(path, "request.json") {
        if let Ok(record) = serde_json::from_slice::<CouncilRequestRecord>(&bytes) {
            if record.request_id == request_id {
                return Ok(Some(i128::from(record.created_at.timestamp_millis())));
            }
        }
    }
    Ok(None)
}

fn validate_retention_target(
    root: &Path,
    path: &Path,
    expected_request_id: &str,
) -> Result<(), CouncilStoreError> {
    validate_plain_directory_under_root(root, path)?;
    let request = fs::read(path.join("request.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<CouncilRequestRecord>(&bytes).ok());
    if request.as_ref().map(|record| record.request_id.as_str()) != Some(expected_request_id) {
        return Err(CouncilStoreError::UnsafeRetentionTarget);
    }
    Ok(())
}

fn validate_plain_directory_under_root(root: &Path, path: &Path) -> Result<(), CouncilStoreError> {
    let root = fs::canonicalize(root)?;
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
        return Err(CouncilStoreError::UnsafeRetentionTarget);
    }
    let canonical = fs::canonicalize(path)?;
    if canonical.parent() != Some(root.as_path()) {
        return Err(CouncilStoreError::UnsafeRetentionTarget);
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if !metadata.is_file() || metadata.file_type().is_symlink() || is_reparse_point(&metadata) {
            return Err(CouncilStoreError::UnsafeRetentionTarget);
        }
    }
    Ok(())
}

#[cfg(windows)]
fn is_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;

    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_reparse_point(_metadata: &fs::Metadata) -> bool {
    false
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn rename_noreplace(from: &Path, to: &Path) -> Result<(), std::io::Error> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let from = CString::new(from.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let to = CString::new(to.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    // SAFETY: both pointers reference live NUL-terminated path buffers for this call.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(target_vendor = "apple")]
fn rename_noreplace(from: &Path, to: &Path) -> Result<(), std::io::Error> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let from = CString::new(from.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let to = CString::new(to.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    // SAFETY: both pointers reference live NUL-terminated path buffers for this call.
    let result = unsafe { libc::renamex_np(from.as_ptr(), to.as_ptr(), libc::RENAME_EXCL) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(windows)]
fn rename_noreplace(from: &Path, to: &Path) -> Result<(), std::io::Error> {
    if to.exists() {
        return Err(std::io::ErrorKind::AlreadyExists.into());
    }
    fs::rename(from, to)
}

#[cfg(not(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple"
)))]
fn rename_noreplace(from: &Path, to: &Path) -> Result<(), std::io::Error> {
    if to.exists() {
        return Err(std::io::ErrorKind::AlreadyExists.into());
    }
    fs::rename(from, to)
}

fn app_data_root() -> Result<PathBuf, CouncilStoreError> {
    #[cfg(target_os = "windows")]
    {
        for (variable, suffix) in [
            ("LOCALAPPDATA", vec!["Hypura"]),
            ("USERPROFILE", vec!["AppData", "Local", "Hypura"]),
        ] {
            if let Some(root) = non_empty_env_path(variable) {
                let candidate = suffix
                    .into_iter()
                    .fold(root, |path, component| path.join(component));
                if valid_data_root(&candidate) {
                    return Ok(candidate);
                }
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Some(root) = non_empty_env_path("XDG_DATA_HOME") {
            let candidate = root.join("hypura");
            if valid_data_root(&candidate) {
                return Ok(candidate);
            }
        }
        if let Some(root) = non_empty_env_path("HOME") {
            let candidate = root.join(".local").join("share").join("hypura");
            if valid_data_root(&candidate) {
                return Ok(candidate);
            }
        }
    }

    Err(CouncilStoreError::AppDataRootUnavailable)
}

fn non_empty_env_path(variable: &str) -> Option<PathBuf> {
    let value = std::env::var_os(variable)?;
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value))
    }
}
