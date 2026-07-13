use std::collections::HashMap;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Physical residence tier for a tensor or streamed runtime unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorResidence {
    /// GPU-resident weights or hot runtime slots.
    GpuResident,
    /// CUDA-registered / otherwise pinned host memory used for warm staging.
    HostPinned,
    /// Regular pageable host memory or OS page-cache backed warm data.
    HostPageable,
    /// Cold backing storage on NVMe.
    NvmeBacked,
}

#[allow(non_upper_case_globals)]
impl TensorResidence {
    /// Compatibility alias for older code paths.
    pub const Gpu: Self = Self::GpuResident;
    /// Compatibility alias for older code paths.
    pub const Ram: Self = Self::HostPageable;
    /// Compatibility alias for older code paths.
    pub const Nvme: Self = Self::NvmeBacked;

    pub fn label(self) -> &'static str {
        match self {
            Self::GpuResident => "GPU",
            Self::HostPinned => "HOST_PINNED",
            Self::HostPageable => "HOST_PAGEABLE",
            Self::NvmeBacked => "NVME",
        }
    }

    pub fn is_host(self) -> bool {
        matches!(self, Self::HostPinned | Self::HostPageable)
    }
}

/// Backward-compatible alias while the rest of the codebase migrates to
/// `TensorResidence` naming.
pub type StorageTier = TensorResidence;

/// Coarse scheduler shape used when comparing the 4-tier residency model against
/// the previous GPU/RAM/NVMe behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
pub enum ResidencyProfile {
    #[value(name = "legacy-3tier")]
    Legacy3Tier,
    #[value(name = "four-tier")]
    FourTier,
}

impl ResidencyProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Legacy3Tier => "legacy-3tier",
            Self::FourTier => "four-tier",
        }
    }
}

/// Whether the scheduler should use the host pinned tier when it is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
pub enum HostPinnedPolicy {
    #[value(name = "auto")]
    Auto,
    #[value(name = "off")]
    Off,
    #[value(name = "force")]
    Force,
}

impl HostPinnedPolicy {
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Force => "force",
        }
    }
}

/// Operator-facing scheduling knobs for residency experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResidencyPolicyConfig {
    pub residency_profile: ResidencyProfile,
    pub host_pinned_policy: HostPinnedPolicy,
}

impl ResidencyPolicyConfig {
    pub const fn new(
        residency_profile: ResidencyProfile,
        host_pinned_policy: HostPinnedPolicy,
    ) -> Self {
        Self {
            residency_profile,
            host_pinned_policy,
        }
    }

    pub const fn normalized(self) -> Self {
        match self.residency_profile {
            ResidencyProfile::Legacy3Tier => Self {
                residency_profile: self.residency_profile,
                host_pinned_policy: HostPinnedPolicy::Off,
            },
            ResidencyProfile::FourTier => self,
        }
    }
}

impl Default for ResidencyPolicyConfig {
    fn default() -> Self {
        Self::new(ResidencyProfile::FourTier, HostPinnedPolicy::Auto)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
pub enum CouncilExecutionMode {
    #[value(name = "off")]
    Off,
    #[value(name = "attention")]
    Attention,
    #[value(name = "answer")]
    Answer,
    #[value(name = "hybrid")]
    Hybrid,
}

impl Default for CouncilExecutionMode {
    fn default() -> Self {
        Self::Off
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
pub enum CouncilParallelism {
    #[value(name = "sequential")]
    Sequential,
    #[value(name = "parallel")]
    Parallel,
    #[value(name = "auto")]
    Auto,
}

impl Default for CouncilParallelism {
    fn default() -> Self {
        Self::Sequential
    }
}

pub const TRIALITY_RESIDUAL_PAYLOAD_BITS_PER_CHANNEL: f64 = 5.0;
pub const TRIALITY_RESIDUAL_CONTROLLER_BYTES_PER_LAYER: u64 = 51;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CouncilKCacheLayout {
    Single {
        bits_per_channel: f64,
    },
    ResidualParity,
    FullTriple {
        branch_bits_per_layer: Vec<[f64; 3]>,
    },
}

impl CouncilKCacheLayout {
    pub fn bits_per_layer(&self, num_layers: u32) -> Result<Vec<f64>, CouncilMemoryRefusal> {
        match self {
            Self::Single { bits_per_channel } => {
                validate_positive_finite(*bits_per_channel, "single-view K bit budget")?;
                Ok(vec![*bits_per_channel; num_layers as usize])
            }
            Self::ResidualParity => Ok(vec![
                TRIALITY_RESIDUAL_PAYLOAD_BITS_PER_CHANNEL;
                num_layers as usize
            ]),
            Self::FullTriple {
                branch_bits_per_layer,
            } => {
                if branch_bits_per_layer.len() != num_layers as usize {
                    return Err(CouncilMemoryRefusal::invalid(format!(
                        "full-triple K budget has {} layer row(s), expected {num_layers}",
                        branch_bits_per_layer.len()
                    )));
                }
                branch_bits_per_layer
                    .iter()
                    .enumerate()
                    .map(|(layer, row)| {
                        let mut total = 0.0;
                        for (branch, bits) in row.iter().copied().enumerate() {
                            validate_positive_finite(
                                bits,
                                &format!(
                                    "full-triple K bit budget for layer {layer}, branch {branch}"
                                ),
                            )?;
                            total += bits;
                        }
                        validate_positive_finite(
                            total,
                            &format!("full-triple aggregate K bit budget for layer {layer}"),
                        )?;
                        Ok(total)
                    })
                    .collect()
            }
        }
    }

    pub const fn controller_bytes_per_layer(&self) -> u64 {
        match self {
            Self::ResidualParity => TRIALITY_RESIDUAL_CONTROLLER_BYTES_PER_LAYER,
            Self::Single { .. } | Self::FullTriple { .. } => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouncilMemoryRequest {
    pub execution: CouncilExecutionMode,
    pub parallelism: CouncilParallelism,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u64,
    pub context_length: u32,
    pub k_layout: CouncilKCacheLayout,
    pub v_bits_per_channel: f64,
    #[serde(default)]
    pub additional_controller_bytes: u64,
    #[serde(default)]
    pub host_pageable_bytes_per_context: u64,
    #[serde(default)]
    pub host_pinned_bytes_per_context: u64,
}

impl CouncilMemoryRequest {
    pub fn residual_parity(
        execution: CouncilExecutionMode,
        parallelism: CouncilParallelism,
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u64,
        context_length: u32,
        v_bits_per_channel: f64,
    ) -> Self {
        Self {
            execution,
            parallelism,
            num_layers,
            num_kv_heads,
            head_dim,
            context_length,
            k_layout: CouncilKCacheLayout::ResidualParity,
            v_bits_per_channel,
            additional_controller_bytes: 0,
            host_pageable_bytes_per_context: 0,
            host_pinned_bytes_per_context: 0,
        }
    }

    pub fn project(
        &self,
        context_count: u32,
    ) -> Result<CouncilMemoryProjection, CouncilMemoryRefusal> {
        if context_count != 1 && context_count != 3 {
            return Err(CouncilMemoryRefusal::invalid(format!(
                "council context count must be 1 or 3, got {context_count}"
            )));
        }
        project_council_memory(self, context_count)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouncilMemoryRegionBudget {
    pub capacity_bytes: u64,
    pub committed_bytes: u64,
    pub required_headroom_bytes: u64,
}

impl CouncilMemoryRegionBudget {
    pub const fn new(
        capacity_bytes: u64,
        committed_bytes: u64,
        required_headroom_bytes: u64,
    ) -> Self {
        Self {
            capacity_bytes,
            committed_bytes,
            required_headroom_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouncilMemoryResources {
    pub gpu: CouncilMemoryRegionBudget,
    pub host_pageable: CouncilMemoryRegionBudget,
    pub host_pinned: CouncilMemoryRegionBudget,
    pub unified: Option<CouncilMemoryRegionBudget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouncilMemoryHeadroom {
    pub gpu_bytes: u64,
    pub host_pageable_bytes: u64,
    pub host_pinned_bytes: u64,
    pub unified_bytes: u64,
}

impl CouncilMemoryHeadroom {
    pub const fn new(
        gpu_bytes: u64,
        host_pageable_bytes: u64,
        host_pinned_bytes: u64,
        unified_bytes: u64,
    ) -> Self {
        Self {
            gpu_bytes,
            host_pageable_bytes,
            host_pinned_bytes,
            unified_bytes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouncilMemoryResource {
    Gpu,
    HostPageable,
    HostPinned,
    Unified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouncilMemoryRefusalCode {
    InvalidRequest,
    ArithmeticOverflow,
    InsufficientCapacity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouncilMemoryRefusal {
    pub code: CouncilMemoryRefusalCode,
    pub resource: Option<CouncilMemoryResource>,
    pub requested_bytes: Option<u64>,
    pub available_after_headroom_bytes: Option<u64>,
    pub reason: String,
}

impl CouncilMemoryRefusal {
    fn invalid(reason: String) -> Self {
        Self {
            code: CouncilMemoryRefusalCode::InvalidRequest,
            resource: None,
            requested_bytes: None,
            available_after_headroom_bytes: None,
            reason,
        }
    }

    fn overflow(reason: impl Into<String>) -> Self {
        Self {
            code: CouncilMemoryRefusalCode::ArithmeticOverflow,
            resource: None,
            requested_bytes: None,
            available_after_headroom_bytes: None,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouncilMemoryProjection {
    pub context_count: u32,
    pub kv_bytes: u64,
    pub controller_bytes: u64,
    pub gpu_bytes: u64,
    pub host_pageable_bytes: u64,
    pub host_pinned_bytes: u64,
    pub unified_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouncilMemoryAdmission {
    pub budget: CouncilMemoryBudget,
    pub projection: CouncilMemoryProjection,
    pub refusal: Option<CouncilMemoryRefusal>,
    pub parallel_refusal: Option<CouncilMemoryRefusal>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouncilMemoryBudget {
    pub execution: CouncilExecutionMode,
    pub parallelism: CouncilParallelism,
    pub context_count: u32,
    pub estimated_kv_bytes: u64,
    pub estimated_controller_bytes: u64,
    pub admitted: bool,
    pub reason: String,
}

impl CouncilMemoryBudget {
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        execution: CouncilExecutionMode,
        requested_parallelism: CouncilParallelism,
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u64,
        context_length: u32,
        k_bits_per_channel: f64,
        v_bits_per_channel: f64,
        controller_bytes: u64,
        available_bytes: u64,
        required_headroom_bytes: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            k_bits_per_channel.is_finite() && k_bits_per_channel > 0.0,
            "K bits per channel must be finite and positive"
        );
        anyhow::ensure!(
            v_bits_per_channel.is_finite() && v_bits_per_channel > 0.0,
            "V bits per channel must be finite and positive"
        );

        if execution == CouncilExecutionMode::Off {
            return Ok(Self {
                execution,
                parallelism: CouncilParallelism::Sequential,
                context_count: 0,
                estimated_kv_bytes: 0,
                estimated_controller_bytes: 0,
                admitted: true,
                reason: "council execution is disabled".to_string(),
            });
        }

        let channels = u64::from(num_layers)
            .checked_mul(u64::from(num_kv_heads))
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(u64::from(context_length)))
            .ok_or_else(|| anyhow::anyhow!("council KV channel count overflow"))?;
        let bits_per_context = (channels as f64) * (k_bits_per_channel + v_bits_per_channel);
        anyhow::ensure!(
            bits_per_context.is_finite() && bits_per_context <= u64::MAX as f64 * 8.0,
            "council KV byte estimate overflow"
        );
        let bytes_per_context = (bits_per_context / 8.0).ceil() as u64;

        let supports_three_contexts = matches!(
            execution,
            CouncilExecutionMode::Answer | CouncilExecutionMode::Hybrid
        );
        let parallel_peak = bytes_per_context
            .checked_mul(3)
            .and_then(|value| value.checked_add(controller_bytes))
            .and_then(|value| value.checked_add(required_headroom_bytes))
            .ok_or_else(|| anyhow::anyhow!("parallel council memory estimate overflow"))?;

        let parallelism = match requested_parallelism {
            CouncilParallelism::Auto if supports_three_contexts => {
                if parallel_peak <= available_bytes {
                    CouncilParallelism::Parallel
                } else {
                    CouncilParallelism::Sequential
                }
            }
            CouncilParallelism::Parallel if !supports_three_contexts => {
                CouncilParallelism::Sequential
            }
            other => other,
        };
        let context_count =
            if supports_three_contexts && parallelism == CouncilParallelism::Parallel {
                3
            } else {
                1
            };
        let estimated_kv_bytes = bytes_per_context
            .checked_mul(u64::from(context_count))
            .ok_or_else(|| anyhow::anyhow!("council KV byte estimate overflow"))?;
        let projected_peak = estimated_kv_bytes
            .checked_add(controller_bytes)
            .and_then(|value| value.checked_add(required_headroom_bytes))
            .ok_or_else(|| anyhow::anyhow!("council peak memory estimate overflow"))?;
        let admitted = projected_peak <= available_bytes;
        let reason = if admitted {
            format!(
                "admitted {context_count} context(s); projected peak {projected_peak} bytes within {available_bytes} bytes"
            )
        } else {
            format!(
                "rejected {context_count} context(s); projected peak {projected_peak} bytes exceeds {available_bytes} bytes"
            )
        };

        Ok(Self {
            execution,
            parallelism,
            context_count,
            estimated_kv_bytes,
            estimated_controller_bytes: controller_bytes,
            admitted,
            reason,
        })
    }

    pub fn admit(
        request: &CouncilMemoryRequest,
        resources: &CouncilMemoryResources,
    ) -> CouncilMemoryAdmission {
        if request.execution == CouncilExecutionMode::Off {
            return CouncilMemoryAdmission::admitted(
                request.execution,
                CouncilParallelism::Sequential,
                CouncilMemoryProjection::zero(),
                None,
            );
        }

        let supports_parallel = matches!(
            request.execution,
            CouncilExecutionMode::Answer | CouncilExecutionMode::Hybrid
        );
        match request.parallelism {
            CouncilParallelism::Auto if supports_parallel => {
                let parallel = evaluate_council_candidate(request, resources, 3);
                if parallel.refusal.is_none() {
                    return CouncilMemoryAdmission::admitted(
                        request.execution,
                        CouncilParallelism::Parallel,
                        parallel.projection,
                        None,
                    );
                }

                let parallel_refusal = parallel.refusal;
                let sequential = evaluate_council_candidate(request, resources, 1);
                match sequential.refusal {
                    None => CouncilMemoryAdmission::admitted(
                        request.execution,
                        CouncilParallelism::Sequential,
                        sequential.projection,
                        parallel_refusal,
                    ),
                    refusal => CouncilMemoryAdmission::rejected(
                        request.execution,
                        CouncilParallelism::Sequential,
                        sequential.projection,
                        refusal.expect("rejected candidate must contain a refusal"),
                        parallel_refusal,
                    ),
                }
            }
            CouncilParallelism::Parallel if supports_parallel => {
                let candidate = evaluate_council_candidate(request, resources, 3);
                CouncilMemoryAdmission::from_candidate(
                    request.execution,
                    CouncilParallelism::Parallel,
                    candidate,
                    None,
                )
            }
            CouncilParallelism::Sequential
            | CouncilParallelism::Auto
            | CouncilParallelism::Parallel => {
                let candidate = evaluate_council_candidate(request, resources, 1);
                CouncilMemoryAdmission::from_candidate(
                    request.execution,
                    CouncilParallelism::Sequential,
                    candidate,
                    None,
                )
            }
        }
    }
}

impl CouncilMemoryProjection {
    const fn zero() -> Self {
        Self {
            context_count: 0,
            kv_bytes: 0,
            controller_bytes: 0,
            gpu_bytes: 0,
            host_pageable_bytes: 0,
            host_pinned_bytes: 0,
            unified_bytes: 0,
        }
    }
}

impl CouncilMemoryAdmission {
    pub fn peak_utilization_ratio(
        &self,
        resources: &CouncilMemoryResources,
    ) -> Result<f32, CouncilMemoryRefusal> {
        if let Some(refusal) = &self.refusal {
            return Err(refusal.clone());
        }
        if !self.budget.admitted {
            return Err(CouncilMemoryRefusal::invalid(
                "council memory utilization requires an admitted budget".to_string(),
            ));
        }
        let mut peak = None;
        for (resource, budget, requested_bytes) in [
            (
                CouncilMemoryResource::Gpu,
                resources.gpu,
                self.projection.gpu_bytes,
            ),
            (
                CouncilMemoryResource::HostPageable,
                resources.host_pageable,
                self.projection.host_pageable_bytes,
            ),
            (
                CouncilMemoryResource::HostPinned,
                resources.host_pinned,
                self.projection.host_pinned_bytes,
            ),
        ] {
            update_peak_utilization(&mut peak, resource, budget, requested_bytes)?;
        }
        if let Some(unified) = resources.unified {
            update_peak_utilization(
                &mut peak,
                CouncilMemoryResource::Unified,
                unified,
                self.projection.unified_bytes,
            )?;
        }
        let peak = peak.ok_or_else(|| {
            CouncilMemoryRefusal::invalid(
                "council memory utilization requires a nonzero projected region".to_string(),
            )
        })?;
        if peak.is_finite() && (0.0..=1.0).contains(&peak) {
            Ok(peak as f32)
        } else {
            Err(CouncilMemoryRefusal::overflow(
                "council memory utilization is non-finite",
            ))
        }
    }

    fn admitted(
        execution: CouncilExecutionMode,
        parallelism: CouncilParallelism,
        projection: CouncilMemoryProjection,
        parallel_refusal: Option<CouncilMemoryRefusal>,
    ) -> Self {
        let reason = match parallel_refusal.as_ref() {
            Some(refusal) => format!(
                "admitted sequential execution with retained headroom; parallel candidate refused: {}",
                refusal.reason
            ),
            None if execution == CouncilExecutionMode::Off => {
                "council execution is disabled".to_string()
            }
            None => format!(
                "admitted {} context(s) with GPU, host pageable, host pinned, and unified headroom retained",
                projection.context_count
            ),
        };
        Self {
            budget: CouncilMemoryBudget {
                execution,
                parallelism,
                context_count: projection.context_count,
                estimated_kv_bytes: projection.kv_bytes,
                estimated_controller_bytes: projection.controller_bytes,
                admitted: true,
                reason,
            },
            projection,
            refusal: None,
            parallel_refusal,
        }
    }

    fn rejected(
        execution: CouncilExecutionMode,
        parallelism: CouncilParallelism,
        projection: CouncilMemoryProjection,
        refusal: CouncilMemoryRefusal,
        parallel_refusal: Option<CouncilMemoryRefusal>,
    ) -> Self {
        Self {
            budget: CouncilMemoryBudget {
                execution,
                parallelism,
                context_count: projection.context_count,
                estimated_kv_bytes: projection.kv_bytes,
                estimated_controller_bytes: projection.controller_bytes,
                admitted: false,
                reason: refusal.reason.clone(),
            },
            projection,
            refusal: Some(refusal),
            parallel_refusal,
        }
    }

    fn from_candidate(
        execution: CouncilExecutionMode,
        parallelism: CouncilParallelism,
        candidate: CouncilCandidateEvaluation,
        parallel_refusal: Option<CouncilMemoryRefusal>,
    ) -> Self {
        match candidate.refusal {
            Some(refusal) => Self::rejected(
                execution,
                parallelism,
                candidate.projection,
                refusal,
                parallel_refusal,
            ),
            None => Self::admitted(
                execution,
                parallelism,
                candidate.projection,
                parallel_refusal,
            ),
        }
    }
}

fn update_peak_utilization(
    peak: &mut Option<f64>,
    resource: CouncilMemoryResource,
    budget: CouncilMemoryRegionBudget,
    requested_bytes: u64,
) -> Result<(), CouncilMemoryRefusal> {
    if requested_bytes == 0 {
        return Ok(());
    }
    let committed_with_headroom = budget
        .committed_bytes
        .checked_add(budget.required_headroom_bytes)
        .ok_or_else(|| {
            CouncilMemoryRefusal::overflow(format!(
                "{resource:?} committed memory and headroom overflow"
            ))
        })?;
    if committed_with_headroom > budget.capacity_bytes {
        return Err(CouncilMemoryRefusal {
            code: CouncilMemoryRefusalCode::InsufficientCapacity,
            resource: Some(resource),
            requested_bytes: Some(requested_bytes),
            available_after_headroom_bytes: Some(0),
            reason: format!("{resource:?} has no capacity after committed memory and headroom"),
        });
    }
    let available_after_headroom = budget.capacity_bytes - committed_with_headroom;
    if available_after_headroom == 0 || requested_bytes > available_after_headroom {
        return Err(CouncilMemoryRefusal {
            code: CouncilMemoryRefusalCode::InsufficientCapacity,
            resource: Some(resource),
            requested_bytes: Some(requested_bytes),
            available_after_headroom_bytes: Some(available_after_headroom),
            reason: format!(
                "{resource:?} projected memory exceeds capacity after committed memory and headroom"
            ),
        });
    }
    let utilization = requested_bytes as f64 / available_after_headroom as f64;
    if !utilization.is_finite() {
        return Err(CouncilMemoryRefusal::overflow(format!(
            "{resource:?} memory utilization is non-finite"
        )));
    }
    *peak = Some(peak.map_or(utilization, |current| current.max(utilization)));
    Ok(())
}

struct CouncilCandidateEvaluation {
    projection: CouncilMemoryProjection,
    refusal: Option<CouncilMemoryRefusal>,
}

fn evaluate_council_candidate(
    request: &CouncilMemoryRequest,
    resources: &CouncilMemoryResources,
    context_count: u32,
) -> CouncilCandidateEvaluation {
    let projection = match project_council_memory(request, context_count) {
        Ok(projection) => projection,
        Err(refusal) => {
            return CouncilCandidateEvaluation {
                projection: CouncilMemoryProjection {
                    context_count,
                    ..CouncilMemoryProjection::zero()
                },
                refusal: Some(refusal),
            };
        }
    };

    let regions = [
        (
            CouncilMemoryResource::Gpu,
            resources.gpu,
            projection.gpu_bytes,
        ),
        (
            CouncilMemoryResource::HostPageable,
            resources.host_pageable,
            projection.host_pageable_bytes,
        ),
        (
            CouncilMemoryResource::HostPinned,
            resources.host_pinned,
            projection.host_pinned_bytes,
        ),
    ];
    for (resource, budget, requested_bytes) in regions {
        if let Some(refusal) = region_refusal(resource, budget, requested_bytes) {
            return CouncilCandidateEvaluation {
                projection,
                refusal: Some(refusal),
            };
        }
    }
    if let Some(unified) = resources.unified {
        if let Some(refusal) = region_refusal(
            CouncilMemoryResource::Unified,
            unified,
            projection.unified_bytes,
        ) {
            return CouncilCandidateEvaluation {
                projection,
                refusal: Some(refusal),
            };
        }
    }

    CouncilCandidateEvaluation {
        projection,
        refusal: None,
    }
}

fn project_council_memory(
    request: &CouncilMemoryRequest,
    context_count: u32,
) -> Result<CouncilMemoryProjection, CouncilMemoryRefusal> {
    if request.num_layers == 0
        || request.num_kv_heads == 0
        || request.head_dim == 0
        || request.context_length == 0
    {
        return Err(CouncilMemoryRefusal::invalid(
            "council KV dimensions must all be positive".to_string(),
        ));
    }
    validate_positive_finite(request.v_bits_per_channel, "V bit budget")?;
    let k_bits_per_layer = request.k_layout.bits_per_layer(request.num_layers)?;
    let channels_per_layer = u64::from(request.num_kv_heads)
        .checked_mul(request.head_dim)
        .and_then(|value| value.checked_mul(u64::from(request.context_length)))
        .ok_or_else(|| CouncilMemoryRefusal::overflow("council KV channel count overflow"))?;
    let bits_per_context = k_bits_per_layer.iter().try_fold(0.0, |total, k_bits| {
        let layer_bits = (channels_per_layer as f64) * (*k_bits + request.v_bits_per_channel);
        let next = total + layer_bits;
        if next.is_finite() {
            Ok(next)
        } else {
            Err(CouncilMemoryRefusal::overflow(
                "council KV bit estimate overflow",
            ))
        }
    })?;
    let bytes_per_context = bits_to_bytes(bits_per_context)?;
    let kv_bytes = bytes_per_context
        .checked_mul(u64::from(context_count))
        .ok_or_else(|| CouncilMemoryRefusal::overflow("council KV byte estimate overflow"))?;
    let kv_controller_bytes = u64::from(request.num_layers)
        .checked_mul(request.k_layout.controller_bytes_per_layer())
        .and_then(|value| value.checked_mul(u64::from(context_count)))
        .ok_or_else(|| {
            CouncilMemoryRefusal::overflow("council controller byte estimate overflow")
        })?;
    let controller_bytes = kv_controller_bytes
        .checked_add(request.additional_controller_bytes)
        .ok_or_else(|| {
            CouncilMemoryRefusal::overflow("council controller byte estimate overflow")
        })?;
    let host_pageable_bytes = request
        .host_pageable_bytes_per_context
        .checked_mul(u64::from(context_count))
        .and_then(|value| value.checked_add(request.additional_controller_bytes))
        .ok_or_else(|| CouncilMemoryRefusal::overflow("council host pageable estimate overflow"))?;
    let host_pinned_bytes = request
        .host_pinned_bytes_per_context
        .checked_mul(u64::from(context_count))
        .ok_or_else(|| CouncilMemoryRefusal::overflow("council host pinned estimate overflow"))?;
    let gpu_bytes = kv_bytes
        .checked_add(kv_controller_bytes)
        .ok_or_else(|| CouncilMemoryRefusal::overflow("council GPU estimate overflow"))?;
    let unified_bytes = gpu_bytes
        .checked_add(host_pageable_bytes)
        .and_then(|value| value.checked_add(host_pinned_bytes))
        .ok_or_else(|| CouncilMemoryRefusal::overflow("council unified estimate overflow"))?;

    Ok(CouncilMemoryProjection {
        context_count,
        kv_bytes,
        controller_bytes,
        gpu_bytes,
        host_pageable_bytes,
        host_pinned_bytes,
        unified_bytes,
    })
}

fn validate_positive_finite(value: f64, label: &str) -> Result<(), CouncilMemoryRefusal> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(CouncilMemoryRefusal::invalid(format!(
            "{label} must be finite and positive"
        )))
    }
}

fn bits_to_bytes(bits: f64) -> Result<u64, CouncilMemoryRefusal> {
    let bytes = (bits / 8.0).ceil();
    if bytes.is_finite() && bytes < u64::MAX as f64 {
        Ok(bytes as u64)
    } else {
        Err(CouncilMemoryRefusal::overflow(
            "council KV byte estimate overflow",
        ))
    }
}

fn region_refusal(
    resource: CouncilMemoryResource,
    budget: CouncilMemoryRegionBudget,
    requested_bytes: u64,
) -> Option<CouncilMemoryRefusal> {
    let committed_with_headroom = match budget
        .committed_bytes
        .checked_add(budget.required_headroom_bytes)
    {
        Some(value) => value,
        None => {
            return Some(CouncilMemoryRefusal::overflow(format!(
                "{resource:?} committed memory and headroom overflow"
            )));
        }
    };
    let available_after_headroom = budget
        .capacity_bytes
        .saturating_sub(committed_with_headroom);
    if committed_with_headroom > budget.capacity_bytes || requested_bytes > available_after_headroom
    {
        Some(CouncilMemoryRefusal {
            code: CouncilMemoryRefusalCode::InsufficientCapacity,
            resource: Some(resource),
            requested_bytes: Some(requested_bytes),
            available_after_headroom_bytes: Some(available_after_headroom),
            reason: format!(
                "council memory refused for {resource:?}: requested {requested_bytes} bytes, only {available_after_headroom} bytes remain after committed memory and configured headroom"
            ),
        })
    } else {
        None
    }
}

/// Where compute is expected to execute for a tensor's consuming operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeTarget {
    Gpu,
    CpuFallback,
}

/// Priority used by runtime prefetch/promotion decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrefetchPriority {
    Critical,
    Warm,
    Opportunistic,
}

/// Complete scheduling intent for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorPlacement {
    pub residence: TensorResidence,
    pub compute_target: ComputeTarget,
    pub prefetch_priority: PrefetchPriority,
}

impl TensorPlacement {
    pub const fn new(
        residence: TensorResidence,
        compute_target: ComputeTarget,
        prefetch_priority: PrefetchPriority,
    ) -> Self {
        Self {
            residence,
            compute_target,
            prefetch_priority,
        }
    }

    pub const fn gpu_resident() -> Self {
        Self::new(
            TensorResidence::GpuResident,
            ComputeTarget::Gpu,
            PrefetchPriority::Critical,
        )
    }

    pub const fn host_pageable(prefetch_priority: PrefetchPriority) -> Self {
        Self::new(
            TensorResidence::HostPageable,
            ComputeTarget::Gpu,
            prefetch_priority,
        )
    }

    pub const fn host_pinned(prefetch_priority: PrefetchPriority) -> Self {
        Self::new(
            TensorResidence::HostPinned,
            ComputeTarget::Gpu,
            prefetch_priority,
        )
    }

    pub const fn nvme_backed(prefetch_priority: PrefetchPriority) -> Self {
        Self::new(
            TensorResidence::NvmeBacked,
            ComputeTarget::Gpu,
            prefetch_priority,
        )
    }

    pub const fn cpu_fallback(residence: TensorResidence) -> Self {
        Self::new(
            residence,
            ComputeTarget::CpuFallback,
            PrefetchPriority::Warm,
        )
    }

    pub fn summary_residence(self) -> TensorResidence {
        self.residence
    }

    pub fn prefers_host_pinned(self) -> bool {
        self.residence == TensorResidence::HostPinned
    }
}

/// User experience classification based on predicted tok/s.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExperienceTier {
    /// >10 tok/s — real-time chat
    Fast,
    /// 2-10 tok/s — usable for coding, Q&A
    Usable,
    /// <2 tok/s — background tasks only
    Slow,
}

impl ExperienceTier {
    pub fn from_tok_per_sec(tps: f64) -> Self {
        if tps > 10.0 {
            Self::Fast
        } else if tps > 2.0 {
            Self::Usable
        } else {
            Self::Slow
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Fast => "FAST",
            Self::Usable => "USABLE",
            Self::Slow => "SLOW",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Fast => "suitable for real-time chat",
            Self::Usable => "suitable for coding, Q&A. Not real-time chat.",
            Self::Slow => "background tasks only",
        }
    }
}

/// Inference mode determined by model size, MoE structure, and hardware capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceMode {
    /// All tensors fit in GPU+RAM. No NVMe I/O.
    FullResident,
    /// NVMe spill is small enough to keep loaded after first pass.
    KeepResident,
    /// MoE model: non-expert tensors resident, experts streamed from NVMe on demand.
    ExpertStreaming,
    /// Dense model: attention + norms resident, FFN tensors streamed from NVMe on demand.
    DenseFfnStreaming,
    /// Ultra-sparse MoE: active working set fits in RAM. Use mmap, let OS page cache
    /// handle sparsity. No pool, no eval callback — faster than expert-streaming when
    /// activation ratio is very low (e.g. 512 experts, 10 active = 2%).
    SparseMoeMmap,
    /// Heavy NVMe spill: all NVMe layers streamed, loaded/released per token.
    FullStreaming,
}

/// Complete placement plan for a model on specific hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub model_id: String,
    pub hardware_profile_hash: String,
    /// tensor_name → placement policy
    pub tensor_placements: HashMap<String, TensorPlacement>,
    pub prefetch_schedule: PrefetchSchedule,
    pub estimated_tok_per_sec: f64,
    pub estimated_time_to_first_token: f64,
    pub kv_cache_plan: KvCachePlan,
    pub experience_tier: ExperienceTier,
    pub inference_mode: InferenceMode,
    pub residency_policy: ResidencyPolicyConfig,
}

impl PlacementPlan {
    pub fn placement_for(&self, tensor_name: &str) -> Option<&TensorPlacement> {
        self.tensor_placements.get(tensor_name)
    }

    pub fn residence_for(&self, tensor_name: &str) -> Option<TensorResidence> {
        self.tensor_placements.get(tensor_name).map(|p| p.residence)
    }
}

/// Schedule for prefetching tensors from slower tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchSchedule {
    /// For each layer, which tensors to begin loading during the previous layer's compute.
    pub layer_prefetches: Vec<Vec<PrefetchOp>>,
}

/// A single prefetch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchOp {
    pub tensor_name: String,
    pub source_residence: TensorResidence,
    pub target_residence: TensorResidence,
    pub size_bytes: u64,
    /// How many layers ahead this prefetch should start.
    pub lead_time_layers: u32,
}

/// KV cache quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvQuantization {
    F16,
    Q8_0,
    Q4_0,
}

impl KvQuantization {
    /// Memory scale factor relative to F16.
    pub fn memory_scale(&self) -> f64 {
        match self {
            Self::F16 => 1.0,
            Self::Q8_0 => 0.53,
            Self::Q4_0 => 0.28,
        }
    }
}

/// Plan for KV cache allocation across tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCachePlan {
    pub hot_window_tokens: u32,
    pub warm_window_tokens: u32,
    pub hot_tier: TensorResidence,
    pub warm_tier: TensorResidence,
    pub hot_bytes: u64,
    pub warm_bytes: u64,
    pub pinned_bytes: u64,
    pub pageable_bytes: u64,
    /// Auto-selected KV quantization (None = F16 default).
    pub kv_quantization: Option<KvQuantization>,
}

/// Summary of placement for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementSummary {
    pub layers_on_gpu: u32,
    pub layers_in_host_pinned: u32,
    pub layers_in_host_pageable: u32,
    pub layers_on_nvme: u32,
    pub total_file_bytes: u64,
    pub total_resident_bytes: u64,
    pub total_staging_bytes: u64,
    pub total_gpu_bytes: u64,
    pub total_host_pinned_bytes: u64,
    pub total_host_pageable_bytes: u64,
    pub total_nvme_bytes: u64,
    pub estimated_dequant_cost_us: f64,
    pub estimated_matmul_cost_us: f64,
    pub host_pinned_active: bool,
    pub host_pinned_budget_bytes: u64,
}

/// Unified memory budget estimation for Apple Silicon.
///
/// Replaces the inconsistent per-site overhead estimates (1 GB / 2 GB / 2.5 GB)
/// with a single computation based on actual model parameters.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub total_ram: u64,
    /// GPU tensor bytes committed in physical memory.
    /// With use_mmap=true: ~60% (demand paging). With use_mmap=false: 100%.
    pub gpu_committed: u64,
    /// KV cache size in bytes (F16 baseline, before quantization).
    pub kv_cache_bytes: u64,
    /// Metal compute buffer overhead (~300 MB measured + framework).
    pub metal_overhead: u64,
    /// OS + page cache reservation.
    pub os_overhead: u64,
    /// Pool buffer size (if any).
    pub pool_bytes: u64,
    /// Resident FFN buffer size (if any).
    pub resident_bytes: u64,
    /// Total committed = gpu + kv + metal + os + pool + resident.
    pub total_committed: u64,
    /// Available for new allocations (total_ram - total_committed).
    pub available: u64,
}

/// Measured Metal compute buffer overhead. The benchmark showed ~273 MB MTL0 +
/// ~33 MB CPU compute = ~306 MB. Round up to 512 MB for safety.
const METAL_COMPUTE_OVERHEAD: u64 = 512 * (1 << 20);

/// OS + page cache reservation. macOS kernel + WindowServer + system daemons
/// typically consume 3-4 GB on Apple Silicon.
const OS_RESERVED: u64 = 3 * (1 << 30);

impl MemoryBudget {
    /// Compute memory budget from model and hardware parameters.
    ///
    /// `gpu_tensor_bytes`: total bytes of tensors on Metal GPU.
    /// `use_mmap`: true if Metal uses mmap (60% commit), false if fread (100%).
    /// `num_layers`, `num_kv_heads`, `head_dim`, `context_length`: for KV cache.
    /// `kv_quant`: optional KV quantization (reduces KV cache size).
    pub fn compute(
        total_ram: u64,
        gpu_tensor_bytes: u64,
        use_mmap: bool,
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u64,
        context_length: u32,
        kv_quant: Option<KvQuantization>,
    ) -> Self {
        let gpu_committed = if use_mmap {
            gpu_tensor_bytes * 60 / 100
        } else {
            gpu_tensor_bytes
        };

        // KV cache: 2 (K+V) × layers × kv_heads × head_dim × 2 (FP16) × context
        let kv_raw =
            2 * num_layers as u64 * num_kv_heads as u64 * head_dim * 2 * context_length as u64;
        let kv_cache_bytes = match kv_quant {
            Some(q) => (kv_raw as f64 * q.memory_scale()) as u64,
            None => kv_raw,
        };

        let total_committed = gpu_committed + kv_cache_bytes + METAL_COMPUTE_OVERHEAD + OS_RESERVED;

        let available = total_ram.saturating_sub(total_committed);

        Self {
            total_ram,
            gpu_committed,
            kv_cache_bytes,
            metal_overhead: METAL_COMPUTE_OVERHEAD,
            os_overhead: OS_RESERVED,
            pool_bytes: 0,
            resident_bytes: 0,
            total_committed,
            available,
        }
    }

    /// Update with pool buffer allocation.
    pub fn with_pool(mut self, pool_bytes: u64) -> Self {
        self.pool_bytes = pool_bytes;
        self.total_committed += pool_bytes;
        self.available = self.total_ram.saturating_sub(self.total_committed);
        self
    }

    /// Update with resident buffer allocation.
    pub fn with_resident(mut self, resident_bytes: u64) -> Self {
        self.resident_bytes = resident_bytes;
        self.total_committed += resident_bytes;
        self.available = self.total_ram.saturating_sub(self.total_committed);
        self
    }

    /// Compute dynamic pool slot count based on available memory.
    /// Returns a slot count clamped between `min_slots` and `max_slots`.
    pub fn pool_slots(&self, slot_size: u64, min_slots: usize, max_slots: usize) -> usize {
        if slot_size == 0 {
            return min_slots;
        }
        let slots_from_memory = (self.available / slot_size) as usize;
        slots_from_memory.clamp(min_slots, max_slots)
    }
}

#[cfg(test)]
mod council_memory_budget_tests {
    use super::{CouncilExecutionMode, CouncilMemoryBudget, CouncilParallelism};

    fn estimate(
        execution: CouncilExecutionMode,
        parallelism: CouncilParallelism,
        available_bytes: u64,
    ) -> CouncilMemoryBudget {
        CouncilMemoryBudget::estimate(
            execution,
            parallelism,
            2,
            4,
            8,
            16,
            16.0,
            16.0,
            256,
            available_bytes,
            128,
        )
        .expect("valid budget")
    }

    #[test]
    fn estimates_fp16_attention_kv_bytes() {
        let budget = estimate(
            CouncilExecutionMode::Attention,
            CouncilParallelism::Parallel,
            4_480,
        );

        assert_eq!(budget.parallelism, CouncilParallelism::Sequential);
        assert_eq!(budget.context_count, 1);
        assert_eq!(budget.estimated_kv_bytes, 4_096);
        assert!(budget.admitted);
    }

    #[test]
    fn auto_parallelism_obeys_peak_memory() {
        let parallel = estimate(
            CouncilExecutionMode::Answer,
            CouncilParallelism::Auto,
            12_672,
        );
        let sequential = estimate(
            CouncilExecutionMode::Answer,
            CouncilParallelism::Auto,
            5_000,
        );

        assert_eq!(parallel.parallelism, CouncilParallelism::Parallel);
        assert_eq!(parallel.context_count, 3);
        assert!(parallel.admitted);
        assert_eq!(sequential.parallelism, CouncilParallelism::Sequential);
        assert_eq!(sequential.context_count, 1);
        assert!(sequential.admitted);
    }

    #[test]
    fn rejects_requested_parallelism_without_headroom() {
        let budget = estimate(
            CouncilExecutionMode::Hybrid,
            CouncilParallelism::Parallel,
            12_000,
        );

        assert!(!budget.admitted);
        assert!(budget.reason.contains("exceeds 12000 bytes"));
    }

    #[test]
    fn rejects_invalid_bit_width() {
        let error = CouncilMemoryBudget::estimate(
            CouncilExecutionMode::Attention,
            CouncilParallelism::Sequential,
            1,
            1,
            1,
            1,
            0.0,
            16.0,
            0,
            1,
            0,
        )
        .expect_err("zero K bit width must fail");

        assert!(error.to_string().contains("K bits per channel"));
    }
}
