use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::thread;
#[cfg(windows)]
use std::time::Duration;

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;

#[cfg(windows)]
use windows_sys::Win32::Storage::FileSystem::{
    MOVE_FILE_FLAGS, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

use super::report::{UrtComparisonError, UrtConsistencyReport, build_consistency_report};
use super::types::{RepresentationId, UrtObservation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UrtPersistence {
    Disabled,
    HashedRequestIds,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrtRegistryConfig {
    pub data_root: PathBuf,
    pub persistence: UrtPersistence,
}

impl UrtRegistryConfig {
    pub fn app_data_persistent() -> Result<Self, UrtRegistryError> {
        Ok(Self::persistent(app_data_root()?))
    }

    pub fn app_data_memory_only() -> Result<Self, UrtRegistryError> {
        Ok(Self::memory_only(app_data_root()?))
    }

    pub fn persistent(data_root: impl Into<PathBuf>) -> Self {
        Self {
            data_root: data_root.into(),
            persistence: UrtPersistence::HashedRequestIds,
        }
    }

    pub fn memory_only(data_root: impl Into<PathBuf>) -> Self {
        Self {
            data_root: data_root.into(),
            persistence: UrtPersistence::Disabled,
        }
    }

    pub fn urt_directory(&self) -> PathBuf {
        self.data_root.join("artifacts").join("urt")
    }
}

#[derive(Debug, Error)]
pub enum UrtRegistryError {
    #[error("Hypura application data root is unavailable")]
    AppDataRootUnavailable,
    #[error("URT registry data root must not be empty")]
    InvalidDataRoot,
    #[error("URT observation is missing a required identifier or operator word")]
    InvalidObservation,
    #[error("URT observation contains non-finite values or a negative tolerance")]
    InvalidNumericValue,
    #[error("URT registry persistence is incomplete; both registry artifacts are required")]
    IncompletePersistentState,
    #[error("URT observation references an unregistered representation")]
    UnregisteredRepresentation,
    #[error("URT persistent report or transaction state is malformed")]
    InvalidPersistentState,
    #[error("URT registry commit succeeded but transaction cleanup is incomplete: {0}")]
    CommittedCleanup(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Comparison(#[from] UrtComparisonError),
}

fn app_data_root() -> Result<PathBuf, UrtRegistryError> {
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
                if candidate.is_absolute() {
                    return Ok(candidate);
                }
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Some(root) = non_empty_env_path("XDG_DATA_HOME") {
            let candidate = root.join("hypura");
            if candidate.is_absolute() {
                return Ok(candidate);
            }
        }
        if let Some(root) = non_empty_env_path("HOME") {
            let candidate = root.join(".local").join("share").join("hypura");
            if candidate.is_absolute() {
                return Ok(candidate);
            }
        }
    }

    Err(UrtRegistryError::AppDataRootUnavailable)
}

fn non_empty_env_path(variable: &str) -> Option<PathBuf> {
    let value = std::env::var_os(variable)?;
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value))
    }
}

#[derive(Debug)]
pub struct UrtRegistry {
    config: UrtRegistryConfig,
    representations: BTreeSet<RepresentationId>,
    observations: Vec<UrtObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegistryPublishFault {
    None,
    #[cfg(test)]
    AfterRepresentations,
    #[cfg(test)]
    CrashAfterRepresentations,
    #[cfg(test)]
    BeforeCommitMarker,
    #[cfg(test)]
    AfterCommitBeforeCleanup,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct RegistryTransactionJournal {
    schema_version: u32,
    transaction_id: String,
    representations_had_previous: bool,
    observations_had_previous: bool,
    representations_previous_sha256: Option<String>,
    observations_previous_sha256: Option<String>,
    representations_sha256: String,
    observations_sha256: String,
}

#[derive(Debug)]
struct PersistentRegistryState {
    representations: BTreeSet<RepresentationId>,
    observations: Vec<UrtObservation>,
    requires_rewrite: bool,
}

struct RegistryDirectoryLock {
    _file: File,
}

const SUMMARY_HEADER: &str = "model_hash,state_id,layer,operator_word_sha256,observable,left_kind,right_kind,absolute_error,tolerance,consistent\n";
const REGISTRY_JOURNAL_FILE: &str = ".registry-transaction.json";
const REGISTRY_COMMIT_FILE: &str = ".registry-transaction.commit";
const REPORT_JOURNAL_FILE: &str = ".report-transaction.json";
const REPORT_COMMIT_FILE: &str = ".report-transaction.commit";

impl RegistryDirectoryLock {
    fn acquire(directory: &Path) -> Result<Self, std::io::Error> {
        fs::create_dir_all(directory)?;
        let path = directory.join(".registry.lock");

        #[cfg(windows)]
        {
            let mut last_error = None;
            for _ in 0..500 {
                match OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .share_mode(0)
                    .open(&path)
                {
                    Ok(file) => return Ok(Self { _file: file }),
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::WouldBlock
                        ) =>
                    {
                        last_error = Some(error);
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => return Err(error),
                }
            }
            Err(last_error.unwrap_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "timed out acquiring URT registry directory lock",
                )
            }))
        }

        #[cfg(unix)]
        {
            let file = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(path)?;
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if result != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self { _file: file })
        }

        #[cfg(not(any(windows, unix)))]
        {
            compile_error!("URT registry locking is not implemented for this platform");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UrtAssessmentStatus {
    Unassessed,
    Assessed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UrtAssessment {
    pub observation: UrtObservation,
    pub status: UrtAssessmentStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<UrtConsistencyReport>,
}

impl UrtRegistry {
    pub fn open(config: UrtRegistryConfig) -> Result<Self, UrtRegistryError> {
        if config.data_root.as_os_str().is_empty() {
            return Err(UrtRegistryError::InvalidDataRoot);
        }
        let mut registry = Self {
            config,
            representations: BTreeSet::new(),
            observations: Vec::new(),
        };
        if registry.config.persistence == UrtPersistence::HashedRequestIds {
            registry.load_persisted()?;
        }
        Ok(registry)
    }

    pub fn representations(&self) -> &BTreeSet<RepresentationId> {
        &self.representations
    }

    pub fn observations(&self) -> &[UrtObservation] {
        &self.observations
    }

    pub fn register_representation(
        &mut self,
        representation: RepresentationId,
    ) -> Result<bool, UrtRegistryError> {
        validate_representation(&representation)?;
        if self.config.persistence == UrtPersistence::Disabled {
            return Ok(self.representations.insert(representation));
        }

        let directory = self.config.urt_directory();
        let _lock = RegistryDirectoryLock::acquire(&directory)?;
        prepare_persistent_directory(&directory)?;
        let mut state = read_registry_state(&directory)?;
        let inserted = state.representations.insert(representation);
        if inserted || state.requires_rewrite {
            if let Err(error) = self.persist_registry_state(
                &state.representations,
                &state.observations,
                RegistryPublishFault::None,
            ) {
                if matches!(error, UrtRegistryError::CommittedCleanup(_)) {
                    self.representations = state.representations;
                    self.observations = state.observations;
                }
                return Err(error);
            }
        }
        self.representations = state.representations;
        self.observations = state.observations;
        Ok(inserted)
    }

    pub fn record_observation(
        &mut self,
        observation: UrtObservation,
    ) -> Result<(), UrtRegistryError> {
        self.record_observation_with_fault(observation, RegistryPublishFault::None)
    }

    pub fn record_and_assess(
        &mut self,
        observation: UrtObservation,
    ) -> Result<UrtAssessment, UrtRegistryError> {
        validate_observation(&observation)?;
        let assessment_observation = observation.clone();
        let stored_observation = self.observation_for_storage(observation);
        self.record_observation(stored_observation.clone())?;
        let representations = self
            .observations
            .iter()
            .filter(|candidate| same_comparison_key(candidate, &stored_observation))
            .map(|candidate| &candidate.representation)
            .collect::<BTreeSet<_>>();
        let report = if representations.len() >= 2 {
            Some(self.consistency_report(
                &stored_observation.state_id,
                stored_observation.layer,
                &stored_observation.operator_word_sha256,
                &stored_observation.observable,
            )?)
        } else {
            None
        };
        Ok(UrtAssessment {
            observation: assessment_observation,
            status: if report.is_some() {
                UrtAssessmentStatus::Assessed
            } else {
                UrtAssessmentStatus::Unassessed
            },
            report,
        })
    }

    pub fn consistency_report(
        &self,
        state_id: &str,
        layer: Option<u32>,
        operator_word_sha256: &str,
        observable: &str,
    ) -> Result<UrtConsistencyReport, UrtRegistryError> {
        let state_id = if self.config.persistence == UrtPersistence::HashedRequestIds {
            hashed_identifier(state_id, b"hypura.urt.state-id.v1\0", "sha256:state:")
        } else {
            state_id.to_string()
        };
        let observations = if self.config.persistence == UrtPersistence::HashedRequestIds {
            let directory = self.config.urt_directory();
            let _lock = RegistryDirectoryLock::acquire(&directory)?;
            prepare_persistent_directory(&directory)?;
            read_registry_state(&directory)?.observations
        } else {
            self.observations.clone()
        };
        let matching = observations
            .iter()
            .filter(|observation| {
                observation.state_id == state_id
                    && observation.layer == layer
                    && observation.operator_word_sha256 == operator_word_sha256
                    && observation.observable == observable
            })
            .cloned()
            .collect::<Vec<_>>();
        let report = build_consistency_report(&matching)?;
        self.persist_report(&report)?;
        Ok(report)
    }

    pub fn compare(
        left: &UrtObservation,
        right: &UrtObservation,
    ) -> Result<UrtConsistencyReport, UrtComparisonError> {
        build_consistency_report(&[left.clone(), right.clone()])
    }

    fn load_persisted(&mut self) -> Result<(), UrtRegistryError> {
        let directory = self.config.urt_directory();
        let _lock = RegistryDirectoryLock::acquire(&directory)?;
        prepare_persistent_directory(&directory)?;
        let state = read_registry_state(&directory)?;
        if state.requires_rewrite {
            self.persist_registry_state(
                &state.representations,
                &state.observations,
                RegistryPublishFault::None,
            )?;
        }
        self.representations = state.representations;
        self.observations = state.observations;
        Ok(())
    }

    fn observation_for_storage(&self, observation: UrtObservation) -> UrtObservation {
        if self.config.persistence == UrtPersistence::HashedRequestIds {
            persisted_observation(&observation)
        } else {
            observation
        }
    }

    fn record_observation_with_fault(
        &mut self,
        observation: UrtObservation,
        fault: RegistryPublishFault,
    ) -> Result<(), UrtRegistryError> {
        validate_observation(&observation)?;
        let observation = self.observation_for_storage(observation);
        if self.config.persistence == UrtPersistence::Disabled {
            self.representations
                .insert(observation.representation.clone());
            self.observations.push(observation);
            return Ok(());
        }

        let directory = self.config.urt_directory();
        let _lock = RegistryDirectoryLock::acquire(&directory)?;
        prepare_persistent_directory(&directory)?;
        let mut state = read_registry_state(&directory)?;
        let representation_inserted = state
            .representations
            .insert(observation.representation.clone());
        let observation_inserted = if state.observations.contains(&observation) {
            false
        } else {
            state.observations.push(observation);
            true
        };
        if representation_inserted || observation_inserted || state.requires_rewrite {
            if let Err(error) =
                self.persist_registry_state(&state.representations, &state.observations, fault)
            {
                if matches!(error, UrtRegistryError::CommittedCleanup(_)) {
                    self.representations = state.representations;
                    self.observations = state.observations;
                }
                return Err(error);
            }
        }
        self.representations = state.representations;
        self.observations = state.observations;
        Ok(())
    }

    #[cfg(test)]
    fn record_observation_with_injected_publication_failure(
        &mut self,
        observation: UrtObservation,
    ) -> Result<(), UrtRegistryError> {
        self.record_observation_with_fault(observation, RegistryPublishFault::AfterRepresentations)
    }

    #[cfg(test)]
    fn record_observation_with_injected_crash(
        &mut self,
        observation: UrtObservation,
    ) -> Result<(), UrtRegistryError> {
        self.record_observation_with_fault(
            observation,
            RegistryPublishFault::CrashAfterRepresentations,
        )
    }

    #[cfg(test)]
    fn record_observation_with_injected_commit_marker_failure(
        &mut self,
        observation: UrtObservation,
    ) -> Result<(), UrtRegistryError> {
        self.record_observation_with_fault(observation, RegistryPublishFault::BeforeCommitMarker)
    }

    #[cfg(test)]
    fn record_observation_with_injected_commit_cleanup_failure(
        &mut self,
        observation: UrtObservation,
    ) -> Result<(), UrtRegistryError> {
        self.record_observation_with_fault(
            observation,
            RegistryPublishFault::AfterCommitBeforeCleanup,
        )
    }

    fn persist_registry_state(
        &self,
        representations: &BTreeSet<RepresentationId>,
        observations: &[UrtObservation],
        fault: RegistryPublishFault,
    ) -> Result<(), UrtRegistryError> {
        if self.config.persistence == UrtPersistence::Disabled {
            return Ok(());
        }
        let representation_bytes = serde_json::to_vec_pretty(representations)?;
        let mut observation_bytes = Vec::new();
        for observation in observations {
            serde_json::to_writer(
                &mut observation_bytes,
                &self.observation_for_storage(observation.clone()),
            )?;
            observation_bytes.push(b'\n');
        }
        transactional_write_registry_pair(
            &self.config.urt_directory(),
            &representation_bytes,
            &observation_bytes,
            fault,
        )?;
        Ok(())
    }

    fn persist_report(&self, report: &UrtConsistencyReport) -> Result<(), UrtRegistryError> {
        self.persist_report_with_fault(report, RegistryPublishFault::None)
    }

    fn persist_report_with_fault(
        &self,
        report: &UrtConsistencyReport,
        fault: RegistryPublishFault,
    ) -> Result<(), UrtRegistryError> {
        if self.config.persistence == UrtPersistence::Disabled {
            return Ok(());
        }
        let report = persisted_report(report);
        let directory = self.config.urt_directory();
        let _lock = RegistryDirectoryLock::acquire(&directory)?;
        prepare_persistent_directory(&directory)?;
        let summary_path = directory.join("consistency_summary.csv");
        let failures_path = directory.join("consistency_failures.jsonl");
        let needs_pair_creation = !summary_path.exists() || !failures_path.exists();
        let mut summary = if summary_path.exists() {
            fs::read_to_string(&summary_path)?
        } else {
            SUMMARY_HEADER.to_string()
        };
        let mut summary_records = parse_csv_records(&summary)?;
        let mut changed = false;
        for comparison in &report.comparisons {
            let fields = vec![
                report.key.model_hash.clone(),
                report.key.state_id.clone(),
                report
                    .key
                    .layer
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                report.key.operator_word_sha256.clone(),
                report.key.observable.clone(),
                comparison.left.kind.as_str().to_string(),
                comparison.right.kind.as_str().to_string(),
                comparison.absolute_error.to_string(),
                comparison.tolerance.to_string(),
                comparison.consistent.to_string(),
            ];
            if !summary_records.contains(&fields) {
                summary.push_str(
                    &fields
                        .iter()
                        .map(|field| csv_field(field))
                        .collect::<Vec<_>>()
                        .join(","),
                );
                summary.push('\n');
                summary_records.push(fields);
                changed = true;
            }
        }

        let mut failures = if failures_path.exists() {
            fs::read(&failures_path)?
        } else {
            Vec::new()
        };
        if !report.consistent {
            let serialized = serde_json::to_vec(&report)?;
            if !failures
                .split(|byte| *byte == b'\n')
                .any(|line| line == serialized)
            {
                failures.extend_from_slice(&serialized);
                failures.push(b'\n');
                changed = true;
            }
        }
        if changed || needs_pair_creation {
            transactional_write_report_pair(&directory, summary.as_bytes(), &failures, fault)?;
        }
        Ok(())
    }
}

fn prepare_persistent_directory(directory: &Path) -> Result<(), UrtRegistryError> {
    recover_registry_transaction(directory)?;
    recover_report_transaction(directory)?;
    cleanup_orphan_transaction_files(directory)?;
    migrate_persistent_reports(directory)?;
    Ok(())
}

fn read_registry_state(directory: &Path) -> Result<PersistentRegistryState, UrtRegistryError> {
    let representations_path = directory.join("representations.json");
    let observations_path = directory.join("observations.jsonl");
    match (representations_path.exists(), observations_path.exists()) {
        (false, false) => {
            return Ok(PersistentRegistryState {
                representations: BTreeSet::new(),
                observations: Vec::new(),
                requires_rewrite: false,
            });
        }
        (true, true) => {}
        _ => return Err(UrtRegistryError::IncompletePersistentState),
    }

    let representations: BTreeSet<RepresentationId> =
        serde_json::from_slice(&fs::read(&representations_path)?)?;
    for representation in &representations {
        validate_representation(representation)?;
    }

    let contents = fs::read_to_string(&observations_path)?;
    let mut observations = Vec::new();
    let mut requires_rewrite = false;
    for line in contents.lines().filter(|line| !line.trim().is_empty()) {
        let observation: UrtObservation = serde_json::from_str(line)?;
        validate_observation(&observation)?;
        if !representations.contains(&observation.representation) {
            return Err(UrtRegistryError::UnregisteredRepresentation);
        }
        let persisted = persisted_observation(&observation);
        requires_rewrite |= persisted != observation;
        observations.push(persisted);
    }
    Ok(PersistentRegistryState {
        representations,
        observations,
        requires_rewrite,
    })
}

fn migrate_persistent_reports(directory: &Path) -> Result<(), UrtRegistryError> {
    let summary_path = directory.join("consistency_summary.csv");
    let failures_path = directory.join("consistency_failures.jsonl");
    let summary_existed = summary_path.exists();
    let failures_existed = failures_path.exists();
    if !summary_existed && !failures_existed {
        return Ok(());
    }

    let original_summary = if summary_existed {
        fs::read_to_string(&summary_path)?
    } else {
        SUMMARY_HEADER.to_string()
    };
    let records = parse_csv_records(&original_summary)?;
    let expected_header = parse_csv_records(SUMMARY_HEADER)?
        .into_iter()
        .next()
        .ok_or(UrtRegistryError::InvalidPersistentState)?;
    if records.first() != Some(&expected_header) {
        return Err(UrtRegistryError::InvalidPersistentState);
    }
    let mut migrated_summary = SUMMARY_HEADER.to_string();
    for mut fields in records.into_iter().skip(1) {
        if fields.iter().all(String::is_empty) {
            continue;
        }
        if fields.len() != 10 {
            return Err(UrtRegistryError::InvalidPersistentState);
        }
        fields[1] = hashed_identifier(&fields[1], b"hypura.urt.state-id.v1\0", "sha256:state:");
        migrated_summary.push_str(
            &fields
                .iter()
                .map(|field| csv_field(field))
                .collect::<Vec<_>>()
                .join(","),
        );
        migrated_summary.push('\n');
    }

    let original_failures = if failures_existed {
        fs::read_to_string(&failures_path)?
    } else {
        String::new()
    };
    let mut migrated_failures = Vec::new();
    for line in original_failures
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let report: UrtConsistencyReport = serde_json::from_str(line)?;
        serde_json::to_writer(&mut migrated_failures, &persisted_report(&report))?;
        migrated_failures.push(b'\n');
    }
    if !summary_existed
        || !failures_existed
        || migrated_summary != original_summary
        || migrated_failures != original_failures.as_bytes()
    {
        transactional_write_report_pair(
            directory,
            migrated_summary.as_bytes(),
            &migrated_failures,
            RegistryPublishFault::None,
        )?;
    }
    Ok(())
}

fn parse_csv_records(input: &str) -> Result<Vec<Vec<String>>, UrtRegistryError> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut chars = input.chars().peekable();
    let mut quoted = false;
    let mut field_was_quoted = false;

    while let Some(character) = chars.next() {
        if quoted {
            if character == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(character);
            }
            continue;
        }

        match character {
            '"' if field.is_empty() && !field_was_quoted => {
                quoted = true;
                field_was_quoted = true;
            }
            '"' => return Err(UrtRegistryError::InvalidPersistentState),
            ',' => {
                record.push(std::mem::take(&mut field));
                field_was_quoted = false;
            }
            '\n' => {
                record.push(std::mem::take(&mut field));
                field_was_quoted = false;
                records.push(std::mem::take(&mut record));
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            character => field.push(character),
        }
    }
    if quoted {
        return Err(UrtRegistryError::InvalidPersistentState);
    }
    if !field.is_empty() || !record.is_empty() || field_was_quoted {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

fn same_comparison_key(left: &UrtObservation, right: &UrtObservation) -> bool {
    left.representation.model_hash == right.representation.model_hash
        && left.state_id == right.state_id
        && left.layer == right.layer
        && left.operator_word_sha256 == right.operator_word_sha256
        && left.observable == right.observable
}

fn validate_representation(representation: &RepresentationId) -> Result<(), UrtRegistryError> {
    if representation.model_hash.trim().is_empty()
        || representation.backend.trim().is_empty()
        || representation.precision.trim().is_empty()
        || representation
            .artefact_hash
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        || representation
            .view
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
    {
        return Err(UrtRegistryError::InvalidObservation);
    }
    Ok(())
}

fn validate_observation(observation: &UrtObservation) -> Result<(), UrtRegistryError> {
    validate_representation(&observation.representation)?;
    if observation.request_id.trim().is_empty()
        || observation.state_id.trim().is_empty()
        || observation.operator_word.is_empty()
        || observation
            .operator_word
            .iter()
            .any(|word| word.trim().is_empty())
        || observation.operator_word_sha256.trim().is_empty()
        || observation.observable.trim().is_empty()
    {
        return Err(UrtRegistryError::InvalidObservation);
    }
    if !observation.value_real.is_finite()
        || !observation.value_imag.is_finite()
        || !observation.tolerance.is_finite()
        || observation.tolerance < 0.0
    {
        return Err(UrtRegistryError::InvalidNumericValue);
    }
    Ok(())
}

fn persisted_observation(observation: &UrtObservation) -> UrtObservation {
    let mut persisted = observation.clone();
    persisted.request_id = hashed_identifier(
        &persisted.request_id,
        b"hypura.urt.request-id.v1\0",
        "sha256:request:",
    );
    persisted.state_id = hashed_identifier(
        &persisted.state_id,
        b"hypura.urt.state-id.v1\0",
        "sha256:state:",
    );
    persisted
}

fn persisted_report(report: &UrtConsistencyReport) -> UrtConsistencyReport {
    let mut persisted = report.clone();
    persisted.key.state_id = hashed_identifier(
        &persisted.key.state_id,
        b"hypura.urt.state-id.v1\0",
        "sha256:state:",
    );
    persisted
}

fn hashed_identifier(value: &str, domain: &[u8], prefix: &str) -> String {
    if value.strip_prefix(prefix).is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    }) {
        return value.to_string();
    }
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(value.as_bytes());
    let hex = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{prefix}{hex}")
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[derive(Debug)]
struct PublishedFile {
    target: PathBuf,
    backup: Option<PathBuf>,
}

fn transactional_write_registry_pair(
    directory: &Path,
    representation_bytes: &[u8],
    observation_bytes: &[u8],
    fault: RegistryPublishFault,
) -> Result<(), UrtRegistryError> {
    transactional_write_pair(
        directory,
        &directory.join("representations.json"),
        &directory.join("observations.jsonl"),
        REGISTRY_JOURNAL_FILE,
        REGISTRY_COMMIT_FILE,
        representation_bytes,
        observation_bytes,
        fault,
    )
}

fn transactional_write_report_pair(
    directory: &Path,
    summary_bytes: &[u8],
    failure_bytes: &[u8],
    fault: RegistryPublishFault,
) -> Result<(), UrtRegistryError> {
    transactional_write_pair(
        directory,
        &directory.join("consistency_summary.csv"),
        &directory.join("consistency_failures.jsonl"),
        REPORT_JOURNAL_FILE,
        REPORT_COMMIT_FILE,
        summary_bytes,
        failure_bytes,
        fault,
    )
}

#[allow(clippy::too_many_arguments)]
fn transactional_write_pair(
    directory: &Path,
    first: &Path,
    second: &Path,
    journal_file: &str,
    commit_file: &str,
    first_bytes: &[u8],
    second_bytes: &[u8],
    fault: RegistryPublishFault,
) -> Result<(), UrtRegistryError> {
    fs::create_dir_all(directory)?;
    let transaction = Uuid::new_v4();
    let staged_first = transaction_path(first, transaction, "staged");
    let staged_second = transaction_path(second, transaction, "staged");
    write_new_synced(&staged_first, first_bytes)?;
    if let Err(error) = write_new_synced(&staged_second, second_bytes) {
        remove_file_if_exists(&staged_first)?;
        return Err(error.into());
    }

    let first_previous_sha256 = first.exists().then(|| sha256_file(first)).transpose()?;
    let second_previous_sha256 = second.exists().then(|| sha256_file(second)).transpose()?;
    let journal = RegistryTransactionJournal {
        schema_version: 1,
        transaction_id: transaction.to_string(),
        representations_had_previous: first_previous_sha256.is_some(),
        observations_had_previous: second_previous_sha256.is_some(),
        representations_previous_sha256: first_previous_sha256,
        observations_previous_sha256: second_previous_sha256,
        representations_sha256: sha256_bytes(first_bytes),
        observations_sha256: sha256_bytes(second_bytes),
    };
    write_transaction_record(&directory.join(journal_file), transaction, &journal)?;

    let published_first = match publish_staged_file(&staged_first, first, transaction) {
        Ok(published) => published,
        Err(error) => {
            recover_pair_transaction(directory, first, second, journal_file, commit_file)?;
            return Err(error.into());
        }
    };

    #[cfg(test)]
    if fault == RegistryPublishFault::AfterRepresentations {
        let injected = std::io::Error::other("injected pair failure after first publication");
        recover_pair_transaction(directory, first, second, journal_file, commit_file)?;
        return Err(injected.into());
    }
    #[cfg(test)]
    if fault == RegistryPublishFault::CrashAfterRepresentations {
        return Err(std::io::Error::other("injected pair crash after first publication").into());
    }
    #[cfg(not(test))]
    let _ = fault;

    let published_second = match publish_staged_file(&staged_second, second, transaction) {
        Ok(published) => published,
        Err(error) => {
            recover_pair_transaction(directory, first, second, journal_file, commit_file)?;
            return Err(error.into());
        }
    };

    debug_assert_eq!(published_first.target, first);
    debug_assert_eq!(published_second.target, second);
    debug_assert_eq!(
        published_first.backup.is_some(),
        journal.representations_had_previous
    );
    debug_assert_eq!(
        published_second.backup.is_some(),
        journal.observations_had_previous
    );
    #[cfg(test)]
    if fault == RegistryPublishFault::BeforeCommitMarker {
        let injected = std::io::Error::other("injected pair commit marker write failure");
        recover_pair_transaction(directory, first, second, journal_file, commit_file)?;
        return Err(injected.into());
    }
    if let Err(error) =
        write_transaction_record(&directory.join(commit_file), transaction, &journal)
    {
        let commit_was_published = directory.join(commit_file).exists();
        return match recover_pair_transaction(directory, first, second, journal_file, commit_file) {
            Ok(()) if commit_was_published => {
                Err(UrtRegistryError::CommittedCleanup(error.to_string()))
            }
            Ok(()) => Err(error),
            Err(recovery_error) => Err(recovery_error),
        };
    }
    let committed_bytes = verify_committed_pair(first, second, &journal)?;
    #[cfg(test)]
    if fault == RegistryPublishFault::AfterCommitBeforeCleanup {
        return Err(UrtRegistryError::CommittedCleanup(
            "injected cleanup failure after URT registry commit".to_string(),
        ));
    }
    if let Err(error) = cleanup_committed_pair_transaction(
        directory,
        first,
        second,
        journal_file,
        commit_file,
        &journal,
        &committed_bytes.0,
        &committed_bytes.1,
    ) {
        return Err(UrtRegistryError::CommittedCleanup(error.to_string()));
    }
    Ok(())
}

fn write_transaction_record(
    path: &Path,
    transaction: Uuid,
    journal: &RegistryTransactionJournal,
) -> Result<(), UrtRegistryError> {
    let staged = transaction_path(path, transaction, "staged");
    write_new_synced(&staged, &serde_json::to_vec(journal)?)?;
    replace_staged_file(&staged, path)?;
    Ok(())
}

fn recover_registry_transaction(directory: &Path) -> Result<(), UrtRegistryError> {
    recover_pair_transaction(
        directory,
        &directory.join("representations.json"),
        &directory.join("observations.jsonl"),
        REGISTRY_JOURNAL_FILE,
        REGISTRY_COMMIT_FILE,
    )
}

fn recover_report_transaction(directory: &Path) -> Result<(), UrtRegistryError> {
    recover_pair_transaction(
        directory,
        &directory.join("consistency_summary.csv"),
        &directory.join("consistency_failures.jsonl"),
        REPORT_JOURNAL_FILE,
        REPORT_COMMIT_FILE,
    )
}

fn recover_pair_transaction(
    directory: &Path,
    first: &Path,
    second: &Path,
    journal_file: &str,
    commit_file: &str,
) -> Result<(), UrtRegistryError> {
    let journal_path = directory.join(journal_file);
    let commit_path = directory.join(commit_file);
    if !journal_path.exists() && !commit_path.exists() {
        return Ok(());
    }
    let record_path = if journal_path.exists() {
        &journal_path
    } else {
        &commit_path
    };
    let journal: RegistryTransactionJournal = serde_json::from_slice(&fs::read(record_path)?)?;
    validate_transaction_journal(&journal)?;
    if commit_path.exists() {
        let commit: RegistryTransactionJournal = serde_json::from_slice(&fs::read(&commit_path)?)?;
        validate_transaction_journal(&commit)?;
        if commit != journal {
            return Err(UrtRegistryError::InvalidPersistentState);
        }
        return finish_committed_pair_transaction(
            directory,
            first,
            second,
            journal_file,
            commit_file,
            &journal,
        );
    }
    rollback_pair_transaction(
        directory,
        first,
        second,
        journal_file,
        commit_file,
        &journal,
    )
}

fn validate_transaction_journal(
    journal: &RegistryTransactionJournal,
) -> Result<(), UrtRegistryError> {
    if journal.schema_version != 1
        || Uuid::parse_str(&journal.transaction_id).is_err()
        || journal.representations_had_previous != journal.representations_previous_sha256.is_some()
        || journal.observations_had_previous != journal.observations_previous_sha256.is_some()
        || journal
            .representations_previous_sha256
            .as_ref()
            .is_some_and(|hash| !valid_sha256_hex(hash))
        || journal
            .observations_previous_sha256
            .as_ref()
            .is_some_and(|hash| !valid_sha256_hex(hash))
        || !valid_sha256_hex(&journal.representations_sha256)
        || !valid_sha256_hex(&journal.observations_sha256)
    {
        return Err(UrtRegistryError::InvalidPersistentState);
    }
    Ok(())
}

fn finish_committed_pair_transaction(
    directory: &Path,
    first: &Path,
    second: &Path,
    journal_file: &str,
    commit_file: &str,
    journal: &RegistryTransactionJournal,
) -> Result<(), UrtRegistryError> {
    let (first_bytes, second_bytes) = verify_committed_pair(first, second, journal)?;
    cleanup_committed_pair_transaction(
        directory,
        first,
        second,
        journal_file,
        commit_file,
        journal,
        &first_bytes,
        &second_bytes,
    )?;
    Ok(())
}

fn verify_committed_pair(
    first: &Path,
    second: &Path,
    journal: &RegistryTransactionJournal,
) -> Result<(Vec<u8>, Vec<u8>), UrtRegistryError> {
    let first_bytes = fs::read(first)?;
    let second_bytes = fs::read(second)?;
    if sha256_bytes(&first_bytes) != journal.representations_sha256
        || sha256_bytes(&second_bytes) != journal.observations_sha256
    {
        return Err(UrtRegistryError::InvalidPersistentState);
    }
    Ok((first_bytes, second_bytes))
}

#[allow(clippy::too_many_arguments)]
fn cleanup_committed_pair_transaction(
    directory: &Path,
    first: &Path,
    second: &Path,
    journal_file: &str,
    commit_file: &str,
    journal: &RegistryTransactionJournal,
    first_bytes: &[u8],
    second_bytes: &[u8],
) -> Result<(), UrtRegistryError> {
    let transaction = Uuid::parse_str(&journal.transaction_id)
        .map_err(|_| UrtRegistryError::InvalidPersistentState)?;
    for (target, bytes) in [(first, first_bytes), (second, second_bytes)] {
        let backup = transaction_path(target, transaction, "backup");
        sanitize_and_remove_backup(&backup, bytes)?;
        remove_file_if_exists(&transaction_path(target, transaction, "staged"))?;
    }
    remove_file_if_exists(&directory.join(journal_file))?;
    remove_file_if_exists(&directory.join(commit_file))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rollback_pair_transaction(
    directory: &Path,
    first: &Path,
    second: &Path,
    journal_file: &str,
    commit_file: &str,
    journal: &RegistryTransactionJournal,
) -> Result<(), UrtRegistryError> {
    let transaction = Uuid::parse_str(&journal.transaction_id)
        .map_err(|_| UrtRegistryError::InvalidPersistentState)?;
    for (target, previous_hash, new_hash) in [
        (
            first.to_path_buf(),
            journal.representations_previous_sha256.as_deref(),
            journal.representations_sha256.as_str(),
        ),
        (
            second.to_path_buf(),
            journal.observations_previous_sha256.as_deref(),
            journal.observations_sha256.as_str(),
        ),
    ] {
        let backup = transaction_path(&target, transaction, "backup");
        if backup.exists() {
            let Some(previous_hash) = previous_hash else {
                return Err(UrtRegistryError::InvalidPersistentState);
            };
            if sha256_file(&backup)? != previous_hash {
                return Err(UrtRegistryError::InvalidPersistentState);
            }
            if target.exists() && sha256_file(&target)? != new_hash {
                return Err(UrtRegistryError::InvalidPersistentState);
            }
            remove_file_if_exists(&target)?;
            fs::rename(&backup, &target)?;
            if sha256_file(&target)? != previous_hash {
                return Err(UrtRegistryError::InvalidPersistentState);
            }
        } else if let Some(previous_hash) = previous_hash {
            if !target.exists() || sha256_file(&target)? != previous_hash {
                return Err(UrtRegistryError::InvalidPersistentState);
            }
        } else if target.exists() {
            if sha256_file(&target)? != new_hash {
                return Err(UrtRegistryError::InvalidPersistentState);
            }
            remove_file_if_exists(&target)?;
        }
        remove_file_if_exists(&transaction_path(&target, transaction, "staged"))?;
    }
    remove_file_if_exists(&transaction_path(
        &directory.join(commit_file),
        transaction,
        "staged",
    ))?;
    remove_file_if_exists(&directory.join(commit_file))?;
    remove_file_if_exists(&directory.join(journal_file))?;
    Ok(())
}

fn cleanup_orphan_transaction_files(directory: &Path) -> Result<(), std::io::Error> {
    if !directory.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".staged") || name.ends_with(".backup") {
            remove_file_if_exists(&entry.path())?;
        }
    }
    Ok(())
}

fn sanitize_and_remove_backup(path: &Path, replacement: &[u8]) -> Result<(), std::io::Error> {
    if !path.exists() {
        return Ok(());
    }
    let mut file = OpenOptions::new().write(true).truncate(true).open(path)?;
    file.write_all(replacement)?;
    file.sync_all()?;
    drop(file);
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            if fs::read(path)? == replacement {
                Ok(())
            } else {
                Err(error)
            }
        }
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    Ok(sha256_bytes(&fs::read(path)?))
}

fn remove_file_if_exists(path: &Path) -> Result<(), std::io::Error> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn replace_staged_file(staged: &Path, target: &Path) -> Result<(), std::io::Error> {
    #[cfg(windows)]
    {
        let staged = staged
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect::<Vec<_>>();
        let target = target
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect::<Vec<_>>();
        let flags: MOVE_FILE_FLAGS = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
        let result = unsafe { MoveFileExW(staged.as_ptr(), target.as_ptr(), flags) };
        if result == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }

    #[cfg(not(windows))]
    {
        fs::rename(staged, target)
    }
}

fn transaction_path(path: &Path, transaction: Uuid, suffix: &str) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("urt-output");
    parent.join(format!(".{file_name}.{transaction}.{suffix}"))
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        return match remove_file_if_exists(path) {
            Ok(()) => Err(error),
            Err(cleanup_error) => Err(cleanup_error),
        };
    }
    Ok(())
}

fn publish_staged_file(
    staged: &Path,
    target: &Path,
    transaction: Uuid,
) -> Result<PublishedFile, std::io::Error> {
    let backup = if target.exists() {
        let backup = transaction_path(target, transaction, "backup");
        fs::rename(target, &backup)?;
        Some(backup)
    } else {
        None
    };
    if let Err(error) = fs::rename(staged, target) {
        if let Some(backup) = backup.as_ref() {
            fs::rename(backup, target)?;
        }
        return Err(error);
    }
    Ok(PublishedFile {
        target: target.to_path_buf(),
        backup,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::urt::RepresentationKind;

    fn observation(
        kind: RepresentationKind,
        backend: &str,
        request_id: &str,
        value_real: f64,
    ) -> UrtObservation {
        UrtObservation {
            request_id: request_id.to_string(),
            representation: RepresentationId {
                kind,
                model_hash: "model-sha256".to_string(),
                artefact_hash: Some("artefact-sha256".to_string()),
                backend: backend.to_string(),
                precision: "f32".to_string(),
                view: Some("vector".to_string()),
            },
            state_id: "state-1".to_string(),
            layer: Some(7),
            operator_word: vec!["Q".to_string(), "U".to_string(), "TQIP".to_string()],
            operator_word_sha256: "operator-sha256".to_string(),
            observable: "output_norm".to_string(),
            value_real,
            value_imag: 0.0,
            tolerance: 0.01,
        }
    }

    fn report(state_id: &str, left_value: f64, right_value: f64) -> UrtConsistencyReport {
        let mut left = observation(
            RepresentationKind::LlamaCpuGguf,
            "cpu",
            "report-request-left",
            left_value,
        );
        left.state_id = state_id.to_string();
        let mut right = observation(
            RepresentationKind::LlamaCudaGguf,
            "cuda",
            "report-request-right",
            right_value,
        );
        right.state_id = state_id.to_string();
        UrtRegistry::compare(&left, &right).unwrap()
    }

    #[test]
    fn registry_distinguishes_concrete_representations_and_hashes_request_ids() {
        let temporary = tempfile::tempdir().unwrap();
        let raw_request_id = "private-request-correlation-value";
        let mut registry =
            UrtRegistry::open(UrtRegistryConfig::persistent(temporary.path())).unwrap();
        let mut cpu_observation =
            observation(RepresentationKind::LlamaCpuGguf, "cpu", raw_request_id, 1.0);
        cpu_observation.state_id = raw_request_id.to_string();
        let assessment = registry.record_and_assess(cpu_observation).unwrap();
        assert_eq!(assessment.observation.request_id, raw_request_id);
        assert_eq!(assessment.observation.state_id, raw_request_id);
        assert_eq!(assessment.status, UrtAssessmentStatus::Unassessed);
        let mut cuda_observation = observation(
            RepresentationKind::LlamaCudaGguf,
            "cuda",
            raw_request_id,
            1.02,
        );
        cuda_observation.state_id = raw_request_id.to_string();
        registry.record_observation(cuda_observation).unwrap();
        assert_eq!(registry.representations().len(), 2);
        let persisted_request_id = registry.observations()[0].request_id.clone();
        let persisted_state_id = registry.observations()[0].state_id.clone();
        assert!(persisted_request_id.starts_with("sha256:request:"));
        assert!(persisted_state_id.starts_with("sha256:state:"));
        assert_ne!(persisted_request_id, persisted_state_id);

        let report = registry
            .consistency_report(raw_request_id, Some(7), "operator-sha256", "output_norm")
            .unwrap();
        assert!(!report.consistent);
        let directory = temporary.path().join("artifacts").join("urt");
        let persisted = fs::read_to_string(directory.join("observations.jsonl")).unwrap();
        assert!(!persisted.contains(raw_request_id));
        assert!(persisted.contains(&persisted_request_id));
        assert!(persisted.contains(&persisted_state_id));
        let summary = fs::read_to_string(directory.join("consistency_summary.csv")).unwrap();
        assert!(!summary.contains(raw_request_id));
        assert!(summary.contains(&persisted_state_id));
        let failures = fs::read_to_string(directory.join("consistency_failures.jsonl")).unwrap();
        assert!(!failures.contains(raw_request_id));
        assert!(failures.contains(&persisted_state_id));
        let reopened = UrtRegistry::open(UrtRegistryConfig::persistent(temporary.path())).unwrap();
        assert_eq!(reopened.observations()[0].request_id, persisted_request_id);
        assert_eq!(reopened.observations()[0].state_id, persisted_state_id);
        assert!(directory.join("representations.json").exists());
        assert!(directory.join("consistency_summary.csv").exists());
        assert!(directory.join("consistency_failures.jsonl").exists());
        assert!(!fs::read_dir(directory).unwrap().flatten().any(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".staged") || name.ends_with(".backup")
        }));
    }

    #[test]
    fn failed_second_artifact_publication_rolls_back_disk_and_memory() {
        let temporary = tempfile::tempdir().unwrap();
        let mut registry =
            UrtRegistry::open(UrtRegistryConfig::persistent(temporary.path())).unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request-old",
                1.0,
            ))
            .unwrap();
        let directory = temporary.path().join("artifacts").join("urt");
        let representations_before = fs::read(directory.join("representations.json")).unwrap();
        let observations_before = fs::read(directory.join("observations.jsonl")).unwrap();
        let memory_representations_before = registry.representations().clone();
        let memory_observations_before = registry.observations().to_vec();

        let error = registry
            .record_observation_with_injected_publication_failure(observation(
                RepresentationKind::LlamaCudaGguf,
                "cuda",
                "request-new",
                2.0,
            ))
            .unwrap_err();
        assert!(error.to_string().contains("injected pair failure"));
        assert_eq!(registry.representations(), &memory_representations_before);
        assert_eq!(registry.observations(), memory_observations_before);
        assert_eq!(
            fs::read(directory.join("representations.json")).unwrap(),
            representations_before
        );
        assert_eq!(
            fs::read(directory.join("observations.jsonl")).unwrap(),
            observations_before
        );
        let reopened = UrtRegistry::open(UrtRegistryConfig::persistent(temporary.path())).unwrap();
        assert_eq!(reopened.representations(), registry.representations());
        assert_eq!(reopened.observations(), registry.observations());
    }

    #[test]
    fn commit_marker_failure_restores_byte_exact_registry_and_memory() {
        let temporary = tempfile::tempdir().unwrap();
        let config = UrtRegistryConfig::persistent(temporary.path());
        let mut registry = UrtRegistry::open(config.clone()).unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request-old",
                1.0,
            ))
            .unwrap();
        let directory = config.urt_directory();
        let representations_before = fs::read(directory.join("representations.json")).unwrap();
        let observations_before = fs::read(directory.join("observations.jsonl")).unwrap();
        let memory_representations_before = registry.representations().clone();
        let memory_observations_before = registry.observations().to_vec();

        let error = registry
            .record_observation_with_injected_commit_marker_failure(observation(
                RepresentationKind::LlamaCudaGguf,
                "cuda",
                "request-new",
                2.0,
            ))
            .unwrap_err();
        assert!(error.to_string().contains("commit marker write failure"));
        assert_eq!(registry.representations(), &memory_representations_before);
        assert_eq!(registry.observations(), memory_observations_before);
        assert_eq!(
            fs::read(directory.join("representations.json")).unwrap(),
            representations_before
        );
        assert_eq!(
            fs::read(directory.join("observations.jsonl")).unwrap(),
            observations_before
        );
        assert_no_transaction_residue(&directory);

        let reopened = UrtRegistry::open(config).unwrap();
        assert_eq!(reopened.representations(), registry.representations());
        assert_eq!(reopened.observations(), registry.observations());
    }

    #[test]
    fn interrupted_pair_publication_is_recovered_on_reopen() {
        let temporary = tempfile::tempdir().unwrap();
        let config = UrtRegistryConfig::persistent(temporary.path());
        let mut registry = UrtRegistry::open(config.clone()).unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request-old",
                1.0,
            ))
            .unwrap();
        let directory = config.urt_directory();
        let representations_before = fs::read(directory.join("representations.json")).unwrap();
        let observations_before = fs::read(directory.join("observations.jsonl")).unwrap();

        let error = registry
            .record_observation_with_injected_crash(observation(
                RepresentationKind::LlamaCudaGguf,
                "cuda",
                "request-new",
                2.0,
            ))
            .unwrap_err();
        assert!(error.to_string().contains("injected pair crash"));
        assert!(directory.join(REGISTRY_JOURNAL_FILE).exists());

        let reopened = UrtRegistry::open(config).unwrap();
        assert_eq!(reopened.observations().len(), 1);
        assert_eq!(
            fs::read(directory.join("representations.json")).unwrap(),
            representations_before
        );
        assert_eq!(
            fs::read(directory.join("observations.jsonl")).unwrap(),
            observations_before
        );
        assert_no_transaction_residue(&directory);
    }

    #[test]
    fn rollback_rejects_tampered_or_unexpected_backups() {
        let tampered_root = tempfile::tempdir().unwrap();
        let tampered_config = UrtRegistryConfig::persistent(tampered_root.path());
        let mut registry = UrtRegistry::open(tampered_config.clone()).unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request-old",
                1.0,
            ))
            .unwrap();
        registry
            .record_observation_with_injected_crash(observation(
                RepresentationKind::LlamaCudaGguf,
                "cuda",
                "request-new",
                2.0,
            ))
            .unwrap_err();
        let directory = tampered_config.urt_directory();
        let journal: RegistryTransactionJournal =
            serde_json::from_slice(&fs::read(directory.join(REGISTRY_JOURNAL_FILE)).unwrap())
                .unwrap();
        let transaction = Uuid::parse_str(&journal.transaction_id).unwrap();
        let backup = transaction_path(
            &directory.join("representations.json"),
            transaction,
            "backup",
        );
        assert!(backup.exists());
        fs::write(&backup, b"tampered backup").unwrap();
        assert!(matches!(
            UrtRegistry::open(tampered_config),
            Err(UrtRegistryError::InvalidPersistentState)
        ));

        let unexpected_root = tempfile::tempdir().unwrap();
        let unexpected_config = UrtRegistryConfig::persistent(unexpected_root.path());
        let mut registry = UrtRegistry::open(unexpected_config.clone()).unwrap();
        registry
            .record_observation_with_injected_crash(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request-first",
                1.0,
            ))
            .unwrap_err();
        let directory = unexpected_config.urt_directory();
        let journal: RegistryTransactionJournal =
            serde_json::from_slice(&fs::read(directory.join(REGISTRY_JOURNAL_FILE)).unwrap())
                .unwrap();
        assert!(!journal.representations_had_previous);
        let transaction = Uuid::parse_str(&journal.transaction_id).unwrap();
        let unexpected_backup = transaction_path(
            &directory.join("representations.json"),
            transaction,
            "backup",
        );
        fs::write(unexpected_backup, b"unexpected backup").unwrap();
        assert!(matches!(
            UrtRegistry::open(unexpected_config),
            Err(UrtRegistryError::InvalidPersistentState)
        ));
    }

    #[test]
    fn committed_cleanup_failure_synchronizes_memory_and_retry_is_idempotent() {
        let temporary = tempfile::tempdir().unwrap();
        let config = UrtRegistryConfig::persistent(temporary.path());
        let mut registry = UrtRegistry::open(config.clone()).unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request-old",
                1.0,
            ))
            .unwrap();
        let retry = observation(
            RepresentationKind::LlamaCudaGguf,
            "cuda",
            "request-committed",
            2.0,
        );

        let error = registry
            .record_observation_with_injected_commit_cleanup_failure(retry.clone())
            .unwrap_err();
        assert!(matches!(error, UrtRegistryError::CommittedCleanup(_)));
        assert_eq!(registry.observations().len(), 2);
        let directory = config.urt_directory();
        assert!(directory.join(REGISTRY_JOURNAL_FILE).exists());
        assert!(directory.join(REGISTRY_COMMIT_FILE).exists());

        registry.record_observation(retry).unwrap();
        assert_eq!(registry.observations().len(), 2);
        let reopened = UrtRegistry::open(config).unwrap();
        assert_eq!(reopened.observations().len(), 2);
        assert_no_transaction_residue(&directory);
    }

    #[test]
    fn report_pair_failure_retry_and_restart_preserve_one_audit_generation() {
        let temporary = tempfile::tempdir().unwrap();
        let config = UrtRegistryConfig::persistent(temporary.path());
        let registry = UrtRegistry::open(config.clone()).unwrap();
        registry
            .persist_report(&report("report-state-old", 1.0, 2.0))
            .unwrap();
        let directory = config.urt_directory();
        let summary_path = directory.join("consistency_summary.csv");
        let failures_path = directory.join("consistency_failures.jsonl");
        let summary_before = fs::read(&summary_path).unwrap();
        let failures_before = fs::read(&failures_path).unwrap();
        let retry = report("report-state-retry", 3.0, 4.0);

        let error = registry
            .persist_report_with_fault(&retry, RegistryPublishFault::AfterRepresentations)
            .unwrap_err();
        assert!(error.to_string().contains("injected pair failure"));
        assert_eq!(fs::read(&summary_path).unwrap(), summary_before);
        assert_eq!(fs::read(&failures_path).unwrap(), failures_before);
        assert_no_transaction_residue(&directory);

        registry.persist_report(&retry).unwrap();
        let summary_after_retry = fs::read(&summary_path).unwrap();
        let failures_after_retry = fs::read(&failures_path).unwrap();
        registry.persist_report(&retry).unwrap();
        assert_eq!(fs::read(&summary_path).unwrap(), summary_after_retry);
        assert_eq!(fs::read(&failures_path).unwrap(), failures_after_retry);

        let interrupted = report("report-state-interrupted", 5.0, 6.0);
        let error = registry
            .persist_report_with_fault(
                &interrupted,
                RegistryPublishFault::CrashAfterRepresentations,
            )
            .unwrap_err();
        assert!(error.to_string().contains("injected pair crash"));
        assert!(directory.join(REPORT_JOURNAL_FILE).exists());

        UrtRegistry::open(config).unwrap();
        assert_eq!(fs::read(&summary_path).unwrap(), summary_after_retry);
        assert_eq!(fs::read(&failures_path).unwrap(), failures_after_retry);
        assert_no_transaction_residue(&directory);
    }

    #[test]
    fn multiple_registry_instances_merge_updates_under_the_directory_lock() {
        let temporary = tempfile::tempdir().unwrap();
        let config = UrtRegistryConfig::persistent(temporary.path());
        let mut first = UrtRegistry::open(config.clone()).unwrap();
        let mut second = UrtRegistry::open(config.clone()).unwrap();
        first
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request-first",
                1.0,
            ))
            .unwrap();
        second
            .record_observation(observation(
                RepresentationKind::LlamaCudaGguf,
                "cuda",
                "request-second",
                1.0,
            ))
            .unwrap();

        let fresh_report = first
            .consistency_report("state-1", Some(7), "operator-sha256", "output_norm")
            .unwrap();
        assert!(fresh_report.consistent);
        assert!(!fresh_report.comparisons.is_empty());

        let reopened = UrtRegistry::open(config).unwrap();
        assert_eq!(reopened.representations().len(), 2);
        assert_eq!(reopened.observations().len(), 2);
        assert!(reopened.observations().iter().any(|candidate| {
            candidate.request_id
                == hashed_identifier(
                    "request-first",
                    b"hypura.urt.request-id.v1\0",
                    "sha256:request:",
                )
        }));
        assert!(reopened.observations().iter().any(|candidate| {
            candidate.request_id
                == hashed_identifier(
                    "request-second",
                    b"hypura.urt.request-id.v1\0",
                    "sha256:request:",
                )
        }));
    }

    #[test]
    fn reopening_migrates_legacy_plaintext_registry_and_report_identifiers() {
        let temporary = tempfile::tempdir().unwrap();
        let config = UrtRegistryConfig::persistent(temporary.path());
        let mut registry = UrtRegistry::open(config.clone()).unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "legacy-private-request",
                1.0,
            ))
            .unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCudaGguf,
                "cuda",
                "legacy-private-request",
                2.0,
            ))
            .unwrap();
        registry
            .consistency_report("state-1", Some(7), "operator-sha256", "output_norm")
            .unwrap();

        let directory = config.urt_directory();
        let request_hash = hashed_identifier(
            "legacy-private-request",
            b"hypura.urt.request-id.v1\0",
            "sha256:request:",
        );
        let state_hash = hashed_identifier("state-1", b"hypura.urt.state-id.v1\0", "sha256:state:");
        for (name, substitutions) in [
            (
                "observations.jsonl",
                vec![
                    (request_hash.as_str(), "legacy-private-request"),
                    (state_hash.as_str(), "state-1"),
                ],
            ),
            (
                "consistency_summary.csv",
                vec![(state_hash.as_str(), "state-1")],
            ),
            (
                "consistency_failures.jsonl",
                vec![(state_hash.as_str(), "state-1")],
            ),
        ] {
            let path = directory.join(name);
            let mut legacy = fs::read_to_string(&path).unwrap();
            for (hashed, raw) in substitutions {
                legacy = legacy.replace(hashed, raw);
            }
            fs::write(path, legacy).unwrap();
        }

        let reopened = UrtRegistry::open(config).unwrap();
        assert_eq!(reopened.observations().len(), 2);
        for name in [
            "observations.jsonl",
            "consistency_summary.csv",
            "consistency_failures.jsonl",
        ] {
            let persisted = fs::read_to_string(directory.join(name)).unwrap();
            assert!(!persisted.contains("legacy-private-request"));
            assert!(!persisted.contains("state-1"));
        }
        assert_no_transaction_residue(&directory);
    }

    #[test]
    fn persisted_registry_tampering_fails_closed() {
        let invalid_representation_root = tempfile::tempdir().unwrap();
        let mut registry = UrtRegistry::open(UrtRegistryConfig::persistent(
            invalid_representation_root.path(),
        ))
        .unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request",
                1.0,
            ))
            .unwrap();
        let directory = invalid_representation_root
            .path()
            .join("artifacts")
            .join("urt");
        let mut representations: BTreeSet<RepresentationId> =
            serde_json::from_slice(&fs::read(directory.join("representations.json")).unwrap())
                .unwrap();
        let mut representation = representations.pop_first().unwrap();
        representation.backend.clear();
        representations.insert(representation);
        fs::write(
            directory.join("representations.json"),
            serde_json::to_vec_pretty(&representations).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            UrtRegistry::open(UrtRegistryConfig::persistent(
                invalid_representation_root.path()
            )),
            Err(UrtRegistryError::InvalidObservation)
        ));

        let invalid_observation_root = tempfile::tempdir().unwrap();
        let mut registry = UrtRegistry::open(UrtRegistryConfig::persistent(
            invalid_observation_root.path(),
        ))
        .unwrap();
        registry
            .record_observation(observation(
                RepresentationKind::LlamaCpuGguf,
                "cpu",
                "request",
                1.0,
            ))
            .unwrap();
        let directory = invalid_observation_root
            .path()
            .join("artifacts")
            .join("urt");
        let persisted = fs::read_to_string(directory.join("observations.jsonl")).unwrap();
        let mut persisted_observation: UrtObservation =
            serde_json::from_str(persisted.trim()).unwrap();
        persisted_observation.representation.backend = "tampered-backend".to_string();
        fs::write(
            directory.join("observations.jsonl"),
            format!(
                "{}\n",
                serde_json::to_string(&persisted_observation).unwrap()
            ),
        )
        .unwrap();
        assert!(matches!(
            UrtRegistry::open(UrtRegistryConfig::persistent(
                invalid_observation_root.path()
            )),
            Err(UrtRegistryError::UnregisteredRepresentation)
        ));

        let incomplete_root = tempfile::tempdir().unwrap();
        let directory = incomplete_root.path().join("artifacts").join("urt");
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("representations.json"), b"[]").unwrap();
        assert!(matches!(
            UrtRegistry::open(UrtRegistryConfig::persistent(incomplete_root.path())),
            Err(UrtRegistryError::IncompletePersistentState)
        ));
    }

    #[test]
    fn application_data_configuration_never_falls_back_to_the_working_directory() {
        let persistent = UrtRegistryConfig::app_data_persistent().unwrap();
        let memory_only = UrtRegistryConfig::app_data_memory_only().unwrap();
        assert!(persistent.data_root.is_absolute());
        assert_eq!(persistent.data_root, memory_only.data_root);
        assert_eq!(persistent.persistence, UrtPersistence::HashedRequestIds);
        assert_eq!(memory_only.persistence, UrtPersistence::Disabled);
        assert_eq!(
            persistent.urt_directory(),
            persistent.data_root.join("artifacts").join("urt")
        );
    }

    fn assert_no_transaction_residue(directory: &Path) {
        assert!(!directory.join(REGISTRY_JOURNAL_FILE).exists());
        assert!(!directory.join(REGISTRY_COMMIT_FILE).exists());
        assert!(!directory.join(REPORT_JOURNAL_FILE).exists());
        assert!(!directory.join(REPORT_COMMIT_FILE).exists());
        assert!(!fs::read_dir(directory).unwrap().flatten().any(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".staged") || name.ends_with(".backup")
        }));
    }
}
