use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::council::{AhaMode, AhaReasonCode, CouncilView};
use crate::scheduler::types::StorageTier;

/// Real-time telemetry events emitted during inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEvent {
    TokenGenerated {
        tok_per_sec: f64,
        token: String,
    },
    PrefetchStatus {
        hit_rate: f64,
        nvme_mbps: f64,
        gpu_slot_hit_rate: f64,
        pinned_slot_hit_rate: f64,
        pageable_fallback_rate: f64,
        h2d_pinned_mbps: f64,
        h2d_pageable_mbps: f64,
        eviction_churn_per_token: f64,
        first_token_stall_ms: f64,
    },
    KvCacheUpdate {
        hot_tokens: u32,
        warm_tokens: u32,
    },
    LayerComputed {
        layer_idx: u32,
        tier: StorageTier,
        duration_us: u64,
    },
    TierRead {
        tier: StorageTier,
        bytes: u64,
        latency_us: u64,
    },
    TrialityBranchCompleted {
        request_id: String,
        view: CouncilView,
        prompt_tokens: u32,
        generated_tokens: u32,
        runtime_ms: u64,
        tok_per_sec: f64,
        trace_enabled: bool,
        content_persisted: bool,
    },
    TrialityConsensusCompleted {
        request_id: String,
        selected_view: CouncilView,
        candidate_scores: [f64; 3],
        winner_margin: f64,
        agreement: f64,
        result_persisted: bool,
    },
    TrialityUrtChecked {
        request_id: String,
        comparison_count: u32,
        consistent: Option<bool>,
        max_absolute_error: Option<f64>,
    },
    TrialityAha {
        request_id: String,
        emitted: bool,
        mode: Option<AhaMode>,
        reason_code: Option<AhaReasonCode>,
    },
}

/// Emitter for telemetry events. Subscribers receive events via broadcast channel.
pub struct TelemetryEmitter {
    tx: broadcast::Sender<TelemetryEvent>,
}

impl TelemetryEmitter {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn emit(&self, event: TelemetryEvent) {
        // Ignore errors (no subscribers)
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TelemetryEvent> {
        self.tx.subscribe()
    }
}

/// Aggregated telemetry for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub tokens_per_second: f64,
    pub gpu_bandwidth_utilization: f64,
    pub ram_bandwidth_utilization: f64,
    pub nvme_read_mbps: f64,
    pub prefetch_hit_rate: f64,
    pub gpu_slot_hit_rate: f64,
    pub pinned_slot_hit_rate: f64,
    pub pageable_fallback_rate: f64,
    pub h2d_pinned_mbps: f64,
    pub h2d_pageable_mbps: f64,
    pub eviction_churn_per_token: f64,
    pub first_token_stall_ms: f64,
    pub kv_cache_hot_tokens: u32,
    pub kv_cache_warm_tokens: u32,
}
