use std::fs;

use chrono::{TimeZone, Utc};
use hypura::council::{
    AhaEvent, AhaMode, AhaReasonCode, AnswerCouncilResult, CouncilArtifactPolicy, CouncilCandidate,
    CouncilInputKind, CouncilRequestRecord, CouncilStore, CouncilStoreConfig, CouncilStoreError,
    CouncilView, CrossScoreMatrix, DEFAULT_COUNCIL_RETENTION_RECORDS, REDACTED_FINAL_ANSWER,
    TrialityBranchMetrics, TrialityConsensusMetrics,
};
use hypura::scheduler::types::CouncilParallelism;
use hypura::telemetry::metrics::{TelemetryEmitter, TelemetryEvent};

fn request(request_id: &str, timestamp: i64, trace: bool) -> CouncilRequestRecord {
    CouncilRequestRecord {
        request_id: request_id.to_string(),
        created_at: Utc.timestamp_opt(timestamp, 0).single().unwrap(),
        model: Some("tiny-model.gguf".to_string()),
        input_kind: CouncilInputKind::Prompt,
        message_count: None,
        max_tokens: Some(8),
        temperature: Some(0.0),
        seed: Some(7),
        parallelism: CouncilParallelism::Sequential,
        attention_consensus: true,
        cross_score: true,
        synthesis: false,
        aha: true,
        trace,
    }
}

fn result(request_id: &str, secret: &str) -> AnswerCouncilResult {
    let candidates = [
        CouncilCandidate::new(
            CouncilView::Vector,
            format!("{secret}-vector"),
            vec![11, 12],
        ),
        CouncilCandidate::new(
            CouncilView::SpinorPlusProxy,
            format!("{secret}-plus"),
            vec![21, 22],
        ),
        CouncilCandidate::new(
            CouncilView::SpinorMinusProxy,
            format!("{secret}-minus"),
            vec![31, 32],
        ),
    ];
    let matrix = CrossScoreMatrix::try_new(
        [[-1.0, -1.1, -1.2], [-0.5, -0.6, -0.7], [-2.0, -2.1, -2.2]],
        [2, 2, 2],
    )
    .unwrap();
    AnswerCouncilResult {
        request_id: request_id.to_string(),
        candidates,
        cross_scores: matrix,
        candidate_scores: [-1.1, -0.6, -2.1],
        selected_view: CouncilView::SpinorPlusProxy,
        selected_text: format!("{secret}-plus"),
        winner_margin: 0.5,
        agreement: 0.25,
        aha: None,
    }
}

fn store_config(root: &std::path::Path) -> CouncilStoreConfig {
    CouncilStoreConfig::for_data_root(root)
}

#[test]
fn application_data_default_is_bounded_and_outside_the_working_directory() {
    let config = CouncilStoreConfig::app_data_default().unwrap();
    assert!(config.data_root.is_absolute());
    assert_eq!(config.retention_records, DEFAULT_COUNCIL_RETENTION_RECORDS);
    assert_eq!(
        config.artifact_root(),
        config.data_root.join("artifacts").join("triality_council")
    );
    assert!(config.policy.persist_metadata);
    assert!(!config.policy.persist_branch_content);
    assert!(!config.policy.persist_final_answer);
}

#[test]
fn rejects_path_traversal_and_non_portable_request_ids() {
    let temporary = tempfile::tempdir().unwrap();
    let store = CouncilStore::open(store_config(temporary.path())).unwrap();
    for request_id in [
        "",
        ".",
        "..",
        "../escape",
        "..\\escape",
        "/absolute",
        "C:escape",
        "nested/path",
        "unicode-例",
        "CON",
        "nul",
        "Com1",
        "LPT9",
    ] {
        assert!(matches!(
            store.record_directory(request_id),
            Err(CouncilStoreError::InvalidRequestId)
        ));
    }
    assert!(store.record_directory("req_01-safe").is_ok());
}

#[test]
fn trace_false_redacts_branch_content_even_when_policy_allows_it() {
    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.policy = CouncilArtifactPolicy::trace_content();
    let store = CouncilStore::open(config).unwrap();
    let secret = "private-branch-content-9f6c";
    let persisted = store
        .persist(
            &request("req-redacted", 1, false),
            &result("req-redacted", secret),
        )
        .unwrap()
        .unwrap();

    assert!(!persisted.branch_content_persisted);
    assert!(persisted.final_answer_persisted);
    let candidates =
        fs::read_to_string(persisted.directory.join("branch_candidates.json")).unwrap();
    assert!(!candidates.contains(secret));
    assert!(!candidates.contains("token_ids"));
    assert!(persisted.directory.join("request.json").exists());
    assert!(persisted.directory.join("cross_scores.csv").exists());
    assert!(persisted.directory.join("consensus_result.json").exists());
    assert!(persisted.directory.join("aha_event.json").exists());
    assert!(persisted.directory.join("final_answer.txt").exists());
}

#[test]
fn trace_content_requires_both_trace_and_explicit_storage_policy() {
    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.policy = CouncilArtifactPolicy::trace_content();
    let store = CouncilStore::open(config).unwrap();
    let secret = "explicit-trace-content-4c71";
    let persisted = store
        .persist(
            &request("req-traced", 2, true),
            &result("req-traced", secret),
        )
        .unwrap()
        .unwrap();

    assert!(persisted.branch_content_persisted);
    let candidates =
        fs::read_to_string(persisted.directory.join("branch_candidates.json")).unwrap();
    assert!(candidates.contains(secret));
    assert!(candidates.contains("token_ids"));
    assert_eq!(
        fs::read_to_string(persisted.directory.join("final_answer.txt")).unwrap(),
        format!("{secret}-plus")
    );
}

#[test]
fn metadata_only_policy_omits_all_generated_text_artifacts() {
    let temporary = tempfile::tempdir().unwrap();
    let store = CouncilStore::open(store_config(temporary.path())).unwrap();
    let secret = "metadata-only-secret-804d";
    let persisted = store
        .persist(
            &request("req-metadata", 3, true),
            &result("req-metadata", secret),
        )
        .unwrap()
        .unwrap();

    let serialized =
        fs::read_to_string(persisted.directory.join("branch_candidates.json")).unwrap();
    assert!(!serialized.contains(secret));
    assert_eq!(
        fs::read_to_string(persisted.directory.join("final_answer.txt")).unwrap(),
        REDACTED_FINAL_ANSWER
    );
    assert!(!persisted.branch_content_persisted);
    assert!(!persisted.final_answer_persisted);
}

#[test]
fn restart_read_applies_the_current_content_policy() {
    let temporary = tempfile::tempdir().unwrap();
    let secret = "restart-private-content-c273";
    let mut write_config = store_config(temporary.path());
    write_config.policy = CouncilArtifactPolicy::trace_content();
    CouncilStore::open(write_config.clone())
        .unwrap()
        .persist(
            &request("req-restart-read", 31, true),
            &result("req-restart-read", secret),
        )
        .unwrap();

    let metadata_store = CouncilStore::open(store_config(temporary.path())).unwrap();
    let redacted = metadata_store.read("req-restart-read").unwrap().unwrap();
    assert_eq!(redacted.request.request_id, "req-restart-read");
    assert_eq!(
        redacted.consensus.selected_view,
        CouncilView::SpinorPlusProxy
    );
    assert_eq!(redacted.cross_scores.token_counts, [2, 2, 2]);
    assert!(redacted.branch_content_redacted);
    assert!(redacted.final_answer_redacted);
    assert_eq!(redacted.final_answer, REDACTED_FINAL_ANSWER);
    assert!(
        redacted
            .branch_candidates
            .iter()
            .all(|candidate| candidate.text.is_none() && candidate.token_ids.is_none())
    );

    let content_store = CouncilStore::open(write_config).unwrap();
    let restored = content_store.read("req-restart-read").unwrap().unwrap();
    assert!(!restored.branch_content_redacted);
    assert!(!restored.final_answer_redacted);
    assert_eq!(restored.final_answer, format!("{secret}-plus"));
    assert_eq!(
        restored.branch_candidates[0].text.as_deref(),
        Some("restart-private-content-c273-vector")
    );
    assert_eq!(
        restored.branch_candidates[0].token_ids.as_deref(),
        Some(&[11, 12][..])
    );
}

#[test]
fn read_rejects_traversal_and_returns_none_for_a_missing_record() {
    let temporary = tempfile::tempdir().unwrap();
    let store = CouncilStore::open(store_config(temporary.path())).unwrap();
    assert!(store.read("req-missing").unwrap().is_none());
    assert!(matches!(
        store.read("../escape"),
        Err(CouncilStoreError::InvalidRequestId)
    ));
}

#[test]
fn read_rejects_corrupted_csv_and_unexpected_artifacts() {
    let temporary = tempfile::tempdir().unwrap();
    let store = CouncilStore::open(store_config(temporary.path())).unwrap();
    let first = store
        .persist(
            &request("req-corrupt-csv", 32, false),
            &result("req-corrupt-csv", "redacted"),
        )
        .unwrap()
        .unwrap();
    fs::write(
        first.directory.join("cross_scores.csv"),
        b"candidate_view,vector,spinor_plus_proxy,spinor_minus_proxy,token_count\nvector,NaN,-1,-1,2\nspinor_plus_proxy,-1,-1,-1,2\nspinor_minus_proxy,-1,-1,-1,2\n",
    )
    .unwrap();
    assert!(matches!(
        store.read("req-corrupt-csv"),
        Err(CouncilStoreError::InvalidPersistedSchema)
    ));

    let second = store
        .persist(
            &request("req-extra-artifact", 33, false),
            &result("req-extra-artifact", "redacted"),
        )
        .unwrap()
        .unwrap();
    fs::write(second.directory.join("unexpected.txt"), b"unexpected").unwrap();
    assert!(matches!(
        store.read("req-extra-artifact"),
        Err(CouncilStoreError::InvalidPersistedSchema)
    ));
}

#[test]
fn read_rejects_branch_content_when_trace_was_disabled() {
    let temporary = tempfile::tempdir().unwrap();
    let store = CouncilStore::open(store_config(temporary.path())).unwrap();
    let persisted = store
        .persist(
            &request("req-trace-disabled", 34, false),
            &result("req-trace-disabled", "redacted"),
        )
        .unwrap()
        .unwrap();
    let mut candidates: serde_json::Value = serde_json::from_slice(
        &fs::read(persisted.directory.join("branch_candidates.json")).unwrap(),
    )
    .unwrap();
    for candidate in candidates.as_array_mut().unwrap() {
        candidate["text"] = serde_json::Value::String("injected-private-text".to_string());
        candidate["token_ids"] = serde_json::json!([1, 2]);
    }
    fs::write(
        persisted.directory.join("branch_candidates.json"),
        serde_json::to_vec_pretty(&candidates).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        store.read("req-trace-disabled"),
        Err(CouncilStoreError::InvalidPersistedSchema)
    ));
}

#[test]
fn record_publication_is_non_overwriting_and_leaves_no_staging_directory() {
    let temporary = tempfile::tempdir().unwrap();
    let store = CouncilStore::open(store_config(temporary.path())).unwrap();
    let first = result("req-once", "first-value");
    let request = request("req-once", 4, false);
    store.persist(&request, &first).unwrap();

    let second = result("req-once", "second-value");
    assert!(matches!(
        store.persist(&request, &second),
        Err(CouncilStoreError::RecordAlreadyExists(id)) if id == "req-once"
    ));
    let root = store.config().artifact_root();
    assert!(
        !fs::read_dir(root)
            .unwrap()
            .flatten()
            .any(|entry| { entry.file_name().to_string_lossy().starts_with(".pending-") })
    );
}

#[test]
fn retention_removes_oldest_records_and_remains_bounded() {
    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.retention_records = 2;
    let store = CouncilStore::open(config).unwrap();
    for (request_id, timestamp) in [("req-old", 10), ("req-mid", 20), ("req-new", 30)] {
        store
            .persist(
                &request(request_id, timestamp, false),
                &result(request_id, "redacted"),
            )
            .unwrap();
    }

    let root = store.config().artifact_root();
    assert!(!root.join("req-old").exists());
    assert!(root.join("req-mid").exists());
    assert!(root.join("req-new").exists());
    assert_eq!(
        fs::read_dir(root)
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .count(),
        2
    );
}

#[cfg(debug_assertions)]
#[test]
fn failed_publication_restores_quarantined_retention_records_exactly() {
    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.retention_records = 2;
    config.policy = CouncilArtifactPolicy::trace_content();
    let store = CouncilStore::open(config).unwrap();
    for (request_id, timestamp, secret) in [
        ("req-rollback-old", 10, "old-private"),
        ("req-rollback-mid", 20, "mid-private"),
    ] {
        store
            .persist(
                &request(request_id, timestamp, true),
                &result(request_id, secret),
            )
            .unwrap();
    }

    let root = store.config().artifact_root();
    let artifacts = [
        "request.json",
        "branch_candidates.json",
        "cross_scores.csv",
        "consensus_result.json",
        "aha_event.json",
        "final_answer.txt",
    ];
    let before = ["req-rollback-old", "req-rollback-mid"].map(|request_id| {
        (
            request_id,
            artifacts.map(|artifact| fs::read(root.join(request_id).join(artifact)).unwrap()),
        )
    });

    assert!(matches!(
        store.persist_with_injected_publication_failure(
            &request("req-rollback-new", 30, true),
            &result("req-rollback-new", "new-private")
        ),
        Err(CouncilStoreError::PublicationFailureInjected)
    ));

    assert!(!root.join("req-rollback-new").exists());
    for (request_id, expected) in before {
        assert!(root.join(request_id).is_dir());
        for (artifact, expected) in artifacts.into_iter().zip(expected) {
            assert_eq!(
                fs::read(root.join(request_id).join(artifact)).unwrap(),
                expected
            );
        }
    }
    let mut record_names = fs::read_dir(&root)
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    record_names.sort();
    assert_eq!(record_names, ["req-rollback-mid", "req-rollback-old"]);
    assert!(!fs::read_dir(root).unwrap().flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.starts_with(".pending-")
            || name.starts_with(".rollback-")
            || name.starts_with(".purge-")
    }));
}

#[test]
fn retention_refuses_to_delete_an_unowned_nested_tree() {
    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.retention_records = 1;
    let store = CouncilStore::open(config).unwrap();
    let root = store.config().artifact_root();
    let unowned = root.join("req-unowned");
    fs::create_dir_all(unowned.join("nested")).unwrap();
    fs::write(
        unowned.join("request.json"),
        serde_json::to_vec(&request("req-unowned", 1, false)).unwrap(),
    )
    .unwrap();
    fs::write(unowned.join("nested").join("keep.txt"), b"keep").unwrap();

    assert!(matches!(
        store.persist(
            &request("req-owned", 2, false),
            &result("req-owned", "redacted")
        ),
        Err(CouncilStoreError::UnsafeRetentionTarget)
    ));
    assert!(unowned.join("nested").join("keep.txt").exists());
    assert!(!root.join("req-owned").exists());
}

#[test]
fn stale_internal_directories_are_removed_before_publication() {
    let temporary = tempfile::tempdir().unwrap();
    let store = CouncilStore::open(store_config(temporary.path())).unwrap();
    let root = store.config().artifact_root();
    fs::create_dir_all(&root).unwrap();
    let pending = root.join(format!(".pending-{}", uuid::Uuid::new_v4()));
    let purge = root.join(format!(".purge-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&pending).unwrap();
    fs::create_dir(&purge).unwrap();
    fs::write(pending.join("private.txt"), b"pending-secret").unwrap();
    fs::write(purge.join("private.txt"), b"purge-secret").unwrap();

    store
        .persist(
            &request("req-clean", 40, false),
            &result("req-clean", "redacted"),
        )
        .unwrap();
    assert!(!pending.exists());
    assert!(!purge.exists());
}

#[test]
fn stale_uncommitted_retention_quarantine_is_restored_before_next_publication() {
    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.retention_records = 2;
    config.policy = CouncilArtifactPolicy::trace_content();
    let store = CouncilStore::open(config).unwrap();
    let persisted = store
        .persist(
            &request("req-crash-old", 41, true),
            &result("req-crash-old", "crash-private"),
        )
        .unwrap()
        .unwrap();
    let expected = fs::read(persisted.directory.join("final_answer.txt")).unwrap();
    let root = store.config().artifact_root();
    let rollback = root.join(format!(".rollback-{}", uuid::Uuid::new_v4()));
    fs::rename(&persisted.directory, &rollback).unwrap();

    store
        .persist(
            &request("req-crash-new", 42, true),
            &result("req-crash-new", "new-private"),
        )
        .unwrap();

    assert_eq!(
        fs::read(root.join("req-crash-old").join("final_answer.txt")).unwrap(),
        expected
    );
    assert!(root.join("req-crash-new").is_dir());
    assert!(!rollback.exists());
    assert!(!fs::read_dir(root).unwrap().flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.starts_with(".pending-")
            || name.starts_with(".rollback-")
            || name.starts_with(".purge-")
    }));
}

#[test]
fn restart_read_immediately_recovers_crashed_retention_transaction() {
    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.policy = CouncilArtifactPolicy::trace_content();
    let writer = CouncilStore::open(config.clone()).unwrap();
    let persisted = writer
        .persist(
            &request("req-restart-crash", 43, true),
            &result("req-restart-crash", "restart-private"),
        )
        .unwrap()
        .unwrap();
    let artifacts = [
        "request.json",
        "branch_candidates.json",
        "cross_scores.csv",
        "consensus_result.json",
        "aha_event.json",
        "final_answer.txt",
    ];
    let expected = artifacts.map(|artifact| fs::read(persisted.directory.join(artifact)).unwrap());
    let root = writer.config().artifact_root();
    let rollback = root.join(format!(".rollback-{}", uuid::Uuid::new_v4()));
    let pending = root.join(format!(".pending-{}", uuid::Uuid::new_v4()));
    let purge = root.join(format!(".purge-{}", uuid::Uuid::new_v4()));
    fs::rename(&persisted.directory, &rollback).unwrap();
    fs::create_dir(&pending).unwrap();
    fs::create_dir(&purge).unwrap();
    fs::write(pending.join("partial.txt"), b"partial-private").unwrap();
    fs::write(purge.join("retired.txt"), b"retired-private").unwrap();
    drop(writer);

    let restarted = CouncilStore::open(config).unwrap();
    let restored = restarted.read("req-restart-crash").unwrap().unwrap();

    assert_eq!(restored.final_answer, "restart-private-plus");
    for (artifact, expected) in artifacts.into_iter().zip(expected) {
        assert_eq!(
            fs::read(root.join("req-restart-crash").join(artifact)).unwrap(),
            expected
        );
    }
    assert!(!rollback.exists());
    assert!(!pending.exists());
    assert!(!purge.exists());
}

#[test]
fn concurrent_restart_reads_recover_one_crashed_record_without_partial_results() {
    use std::sync::{Arc, Barrier};

    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.policy = CouncilArtifactPolicy::trace_content();
    let writer = CouncilStore::open(config.clone()).unwrap();
    let persisted = writer
        .persist(
            &request("req-concurrent-recovery", 44, true),
            &result("req-concurrent-recovery", "concurrent-private"),
        )
        .unwrap()
        .unwrap();
    let root = writer.config().artifact_root();
    let rollback = root.join(format!(".rollback-{}", uuid::Uuid::new_v4()));
    fs::rename(&persisted.directory, &rollback).unwrap();
    drop(writer);

    let barrier = Arc::new(Barrier::new(8));
    let handles = (0..8)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            let config = config.clone();
            std::thread::spawn(move || {
                barrier.wait();
                CouncilStore::open(config).and_then(|store| store.read("req-concurrent-recovery"))
            })
        })
        .collect::<Vec<_>>();

    for handle in handles {
        let record = handle.join().unwrap().unwrap().unwrap();
        assert_eq!(record.final_answer, "concurrent-private-plus");
        assert_eq!(
            record.branch_candidates[0].text.as_deref(),
            Some("concurrent-private-vector")
        );
    }
    assert!(root.join("req-concurrent-recovery").is_dir());
    assert!(!rollback.exists());
    assert!(!fs::read_dir(root).unwrap().flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.starts_with(".pending-")
            || name.starts_with(".rollback-")
            || name.starts_with(".purge-")
    }));
}

#[test]
fn concurrent_publication_has_one_success_and_no_partial_records() {
    use std::sync::{Arc, Barrier};

    let temporary = tempfile::tempdir().unwrap();
    let store = Arc::new(CouncilStore::open(store_config(temporary.path())).unwrap());
    let barrier = Arc::new(Barrier::new(6));
    let mut handles = Vec::new();
    for index in 0..6 {
        let store = Arc::clone(&store);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            store.persist(
                &request("req-race", 50, false),
                &result("req-race", &format!("candidate-{index}")),
            )
        }));
    }
    let outcomes = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(outcomes.iter().filter(|outcome| outcome.is_ok()).count(), 1);
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, Err(CouncilStoreError::RecordAlreadyExists(id)) if id == "req-race"))
            .count(),
        5
    );
    let directory = store.record_directory("req-race").unwrap();
    for artifact in [
        "request.json",
        "branch_candidates.json",
        "cross_scores.csv",
        "consensus_result.json",
        "aha_event.json",
        "final_answer.txt",
    ] {
        assert!(directory.join(artifact).is_file());
    }
    assert!(
        !fs::read_dir(store.config().artifact_root())
            .unwrap()
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().starts_with(".pending-"))
    );
}

#[test]
fn non_finite_nested_metrics_are_rejected_before_writing() {
    let temporary = tempfile::tempdir().unwrap();
    let store = CouncilStore::open(store_config(temporary.path())).unwrap();

    let mut low_level = result("req-low-level-nan", "redacted");
    low_level.candidates[0].low_level_metrics = Some(TrialityConsensusMetrics {
        branches: [
            TrialityBranchMetrics {
                logit_mean: f64::NAN,
                ..TrialityBranchMetrics::default()
            },
            TrialityBranchMetrics::default(),
            TrialityBranchMetrics::default(),
        ],
        pairwise_js: [0.0; 3],
        mean_pairwise_js: 0.0,
        max_pairwise_js: 0.0,
        numerical_rank: 1.0,
        effective_rank: 1.0,
        ka_fallback_used: false,
        operator_word_hash_128: "00000000000000000000000000000000".to_string(),
    });
    assert!(matches!(
        store.persist(&request("req-low-level-nan", 60, false), &low_level),
        Err(CouncilStoreError::InvalidNumericValue)
    ));

    let mut aha = result("req-aha-nan", "redacted");
    aha.aha = Some(AhaEvent {
        schema_version: 1,
        mode: AhaMode::OnlineObservable,
        reason_code: AhaReasonCode::OnlineObservableResolution,
        selected_view: CouncilView::Vector,
        baseline_view: CouncilView::SpinorPlusProxy,
        pre_consensus_js: f64::NAN,
        post_consensus_js: Some(0.01),
        score_gain: 0.1,
        winner_margin: 0.1,
        urt_pre_error: None,
        urt_post_error: None,
        moment_effective_rank: None,
        message: "observable event".to_string(),
        truth_disclaimer: "not a truth guarantee".to_string(),
    });
    assert!(matches!(
        store.persist(&request("req-aha-nan", 61, false), &aha),
        Err(CouncilStoreError::InvalidNumericValue)
    ));
}

#[cfg(unix)]
#[test]
fn persisted_content_is_user_only_on_unix() {
    use std::os::unix::fs::PermissionsExt;

    let temporary = tempfile::tempdir().unwrap();
    let mut config = store_config(temporary.path());
    config.policy = CouncilArtifactPolicy::trace_content();
    let store = CouncilStore::open(config).unwrap();
    let persisted = store
        .persist(
            &request("req-private-mode", 70, true),
            &result("req-private-mode", "private"),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        fs::metadata(&persisted.directory)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(persisted.directory.join("branch_candidates.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
}

#[tokio::test]
async fn triality_broadcast_events_never_contain_generated_text_or_reasoning() {
    let secret = "broadcast-private-branch-5ba3";
    let emitter = TelemetryEmitter::new(8);
    let mut receiver = emitter.subscribe();
    let events = [
        TelemetryEvent::TrialityBranchCompleted {
            request_id: "req-events".to_string(),
            view: CouncilView::Vector,
            prompt_tokens: 3,
            generated_tokens: 2,
            runtime_ms: 4,
            tok_per_sec: 5.0,
            trace_enabled: true,
            content_persisted: true,
        },
        TelemetryEvent::TrialityConsensusCompleted {
            request_id: "req-events".to_string(),
            selected_view: CouncilView::Vector,
            candidate_scores: [-1.0, -2.0, -3.0],
            winner_margin: 1.0,
            agreement: 0.5,
            result_persisted: true,
        },
        TelemetryEvent::TrialityUrtChecked {
            request_id: "req-events".to_string(),
            comparison_count: 3,
            consistent: Some(true),
            max_absolute_error: Some(0.001),
        },
        TelemetryEvent::TrialityAha {
            request_id: "req-events".to_string(),
            emitted: false,
            mode: None,
            reason_code: None,
        },
    ];
    for event in events {
        emitter.emit(event);
        let received = receiver.recv().await.unwrap();
        let json = serde_json::to_string(&received).unwrap();
        assert!(!json.contains(secret));
        for forbidden in [
            "text",
            "token_ids",
            "reasoning",
            "rationale",
            "analysis",
            "thoughts",
            "chain_of_thought",
        ] {
            assert!(
                !json.contains(forbidden),
                "forbidden key {forbidden}: {json}"
            );
        }
    }
}
