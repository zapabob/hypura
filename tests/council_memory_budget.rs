use chrono::Utc;

use hypura::compute::inference::compute_gpu_budget_with_kv_layout;
use hypura::model::metadata::ModelMetadata;
use hypura::profiler::types::{
    CpuProfile, GpuBackend, GpuProfile, HardwareProfile, MemoryProfile, SystemInfo,
};
use hypura::scheduler::placement::council_memory_request_for_metadata;
use hypura::scheduler::types::{
    CouncilExecutionMode, CouncilKCacheLayout, CouncilMemoryBudget, CouncilMemoryRefusalCode,
    CouncilMemoryRegionBudget, CouncilMemoryRequest, CouncilMemoryResource, CouncilMemoryResources,
    CouncilParallelism, TRIALITY_RESIDUAL_CONTROLLER_BYTES_PER_LAYER,
};

fn residual_request(parallelism: CouncilParallelism) -> CouncilMemoryRequest {
    CouncilMemoryRequest::residual_parity(
        CouncilExecutionMode::Answer,
        parallelism,
        2,
        1,
        8,
        4,
        16.0,
    )
}

fn roomy_resources() -> CouncilMemoryResources {
    let region = CouncilMemoryRegionBudget::new(1 << 30, 0, 1 << 20);
    CouncilMemoryResources {
        gpu: region,
        host_pageable: region,
        host_pinned: region,
        unified: None,
    }
}

fn metadata() -> ModelMetadata {
    ModelMetadata {
        architecture: "llama".to_string(),
        parameter_count: 1,
        context_length: 4096,
        embedding_dim: 64,
        num_layers: 2,
        num_heads: 8,
        num_kv_heads: 1,
        vocab_size: 32,
        quantization: None,
        is_moe: false,
        num_experts: None,
        num_experts_used: None,
    }
}

fn hardware(vram_bytes: u64) -> HardwareProfile {
    HardwareProfile {
        timestamp: Utc::now(),
        system: SystemInfo {
            os: "test".to_string(),
            arch: "test".to_string(),
            machine_model: "test".to_string(),
            total_cores: 1,
        },
        memory: MemoryProfile {
            total_bytes: 32 << 30,
            available_bytes: 24 << 30,
            bandwidth_bytes_per_sec: 1,
            h2d_pageable_bandwidth_bytes_per_sec: 1,
            h2d_pinned_bandwidth_bytes_per_sec: 1,
            supports_host_pinning: true,
            pinned_budget_bytes: 2 << 30,
            is_unified: false,
        },
        gpu: Some(GpuProfile {
            name: "test".to_string(),
            vram_bytes,
            bandwidth_bytes_per_sec: 1,
            fp16_tflops: 1.0,
            backend: GpuBackend::Cuda,
        }),
        storage: Vec::new(),
        cpu: CpuProfile {
            model_name: "test".to_string(),
            cores_performance: 1,
            cores_efficiency: 0,
            has_amx: false,
            has_neon: false,
            has_avx512: false,
            has_avx2: true,
            int8_gflops: 1.0,
        },
    }
}

#[test]
fn residual_payload_and_controller_are_accounted_separately() {
    let admission = CouncilMemoryBudget::admit(
        &residual_request(CouncilParallelism::Sequential),
        &roomy_resources(),
    );

    assert!(admission.budget.admitted);
    assert_eq!(admission.projection.context_count, 1);
    assert_eq!(admission.projection.kv_bytes, 168);
    assert_eq!(admission.projection.controller_bytes, 102);
    assert_eq!(
        admission.projection.controller_bytes,
        2 * TRIALITY_RESIDUAL_CONTROLLER_BYTES_PER_LAYER
    );
}

#[test]
fn full_triple_sums_each_layers_branch_bit_budget() {
    let request = CouncilMemoryRequest {
        k_layout: CouncilKCacheLayout::FullTriple {
            branch_bits_per_layer: vec![[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]],
        },
        ..residual_request(CouncilParallelism::Sequential)
    };
    let projection = request.project(1).expect("valid full-triple projection");

    assert_eq!(projection.kv_bytes, 176);
    assert_eq!(projection.controller_bytes, 0);
}

#[test]
fn sequential_peaks_at_one_context_and_parallel_at_three() {
    let sequential = CouncilMemoryBudget::admit(
        &residual_request(CouncilParallelism::Sequential),
        &roomy_resources(),
    );
    let parallel = CouncilMemoryBudget::admit(
        &residual_request(CouncilParallelism::Parallel),
        &roomy_resources(),
    );

    assert_eq!(sequential.budget.context_count, 1);
    assert_eq!(parallel.budget.context_count, 3);
    assert_eq!(
        parallel.projection.kv_bytes,
        sequential.projection.kv_bytes * 3
    );
    assert_eq!(
        parallel.projection.controller_bytes,
        sequential.projection.controller_bytes * 3
    );
    assert_eq!(
        parallel.projection.gpu_bytes,
        parallel.projection.kv_bytes + parallel.projection.controller_bytes
    );
}

#[test]
fn auto_requires_headroom_in_every_resource_region() {
    let request = CouncilMemoryRequest {
        host_pageable_bytes_per_context: 100,
        host_pinned_bytes_per_context: 60,
        ..residual_request(CouncilParallelism::Auto)
    };
    let resources = CouncilMemoryResources {
        gpu: CouncilMemoryRegionBudget::new(10_000, 0, 100),
        host_pageable: CouncilMemoryRegionBudget::new(1_000, 400, 100),
        host_pinned: CouncilMemoryRegionBudget::new(300, 40, 100),
        unified: None,
    };
    let admission = CouncilMemoryBudget::admit(&request, &resources);

    assert!(admission.budget.admitted);
    assert_eq!(admission.budget.parallelism, CouncilParallelism::Sequential);
    assert_eq!(admission.budget.context_count, 1);
    assert_eq!(
        admission.parallel_refusal.as_ref().and_then(|r| r.resource),
        Some(CouncilMemoryResource::HostPinned)
    );
}

#[test]
fn explicit_parallel_returns_structured_refusal() {
    let resources = CouncilMemoryResources {
        gpu: CouncilMemoryRegionBudget::new(600, 0, 100),
        ..roomy_resources()
    };
    let admission =
        CouncilMemoryBudget::admit(&residual_request(CouncilParallelism::Parallel), &resources);

    assert!(!admission.budget.admitted);
    let refusal = admission.refusal.expect("parallel request must be refused");
    assert_eq!(refusal.code, CouncilMemoryRefusalCode::InsufficientCapacity);
    assert_eq!(refusal.resource, Some(CouncilMemoryResource::Gpu));
    assert_eq!(refusal.available_after_headroom_bytes, Some(500));
}

#[test]
fn unified_projection_counts_gpu_host_and_pinned_once() {
    let request = CouncilMemoryRequest {
        host_pageable_bytes_per_context: 10,
        host_pinned_bytes_per_context: 20,
        ..residual_request(CouncilParallelism::Sequential)
    };
    let projected = request.project(1).expect("valid projection");
    assert_eq!(
        projected.unified_bytes,
        projected.gpu_bytes + projected.host_pageable_bytes + projected.host_pinned_bytes
    );
    let resources = CouncilMemoryResources {
        gpu: CouncilMemoryRegionBudget::new(10_000, 0, 0),
        host_pageable: CouncilMemoryRegionBudget::new(10_000, 0, 0),
        host_pinned: CouncilMemoryRegionBudget::new(10_000, 0, 0),
        unified: Some(CouncilMemoryRegionBudget::new(
            projected.unified_bytes + 150,
            100,
            50,
        )),
    };
    assert!(
        CouncilMemoryBudget::admit(&request, &resources)
            .budget
            .admitted
    );

    let insufficient = CouncilMemoryResources {
        unified: Some(CouncilMemoryRegionBudget::new(
            projected.unified_bytes + 149,
            100,
            50,
        )),
        ..resources
    };
    let refusal = CouncilMemoryBudget::admit(&request, &insufficient)
        .refusal
        .expect("one byte below the unified projection must fail");
    assert_eq!(refusal.resource, Some(CouncilMemoryResource::Unified));
}

#[test]
fn current_usage_above_capacity_does_not_saturate_into_admission() {
    let resources = CouncilMemoryResources {
        gpu: CouncilMemoryRegionBudget::new(100, 200, 10),
        ..roomy_resources()
    };
    let admission = CouncilMemoryBudget::admit(
        &residual_request(CouncilParallelism::Sequential),
        &resources,
    );

    assert!(!admission.budget.admitted);
    assert_eq!(
        admission
            .refusal
            .as_ref()
            .and_then(|refusal| refusal.available_after_headroom_bytes),
        Some(0)
    );
}

#[test]
fn admitted_peak_utilization_tracks_low_and_high_pressure() {
    let request = residual_request(CouncilParallelism::Sequential);
    let low_resources = CouncilMemoryResources {
        gpu: CouncilMemoryRegionBudget::new(2_900, 100, 100),
        ..roomy_resources()
    };
    let high_resources = CouncilMemoryResources {
        gpu: CouncilMemoryRegionBudget::new(500, 100, 100),
        ..roomy_resources()
    };
    let low = CouncilMemoryBudget::admit(&request, &low_resources);
    let high = CouncilMemoryBudget::admit(&request, &high_resources);

    assert!((low.peak_utilization_ratio(&low_resources).unwrap() - 0.1).abs() < 1.0e-6);
    assert!((high.peak_utilization_ratio(&high_resources).unwrap() - 0.9).abs() < 1.0e-6);
}

#[test]
fn peak_utilization_fails_closed_on_zero_capacity_and_overflow() {
    let request = residual_request(CouncilParallelism::Sequential);
    let admitted_resources = roomy_resources();
    let admission = CouncilMemoryBudget::admit(&request, &admitted_resources);
    let zero_capacity = CouncilMemoryResources {
        gpu: CouncilMemoryRegionBudget::new(0, 0, 0),
        ..admitted_resources
    };
    assert_eq!(
        admission
            .peak_utilization_ratio(&zero_capacity)
            .unwrap_err()
            .code,
        CouncilMemoryRefusalCode::InsufficientCapacity
    );

    let overflow = CouncilMemoryResources {
        gpu: CouncilMemoryRegionBudget::new(u64::MAX, u64::MAX, 1),
        ..admitted_resources
    };
    assert_eq!(
        admission
            .peak_utilization_ratio(&overflow)
            .unwrap_err()
            .code,
        CouncilMemoryRefusalCode::ArithmeticOverflow
    );
}

#[test]
fn invalid_and_overflowing_requests_fail_closed() {
    for bits in [f64::NAN, f64::INFINITY, -1.0] {
        let request = CouncilMemoryRequest {
            k_layout: CouncilKCacheLayout::FullTriple {
                branch_bits_per_layer: vec![[bits, 1.0, 1.0], [1.0, 1.0, 1.0]],
            },
            ..residual_request(CouncilParallelism::Sequential)
        };
        let refusal = CouncilMemoryBudget::admit(&request, &roomy_resources())
            .refusal
            .expect("invalid branch bits must fail");
        assert_eq!(refusal.code, CouncilMemoryRefusalCode::InvalidRequest);
    }

    let zero_kv_heads = CouncilMemoryRequest {
        num_kv_heads: 0,
        ..residual_request(CouncilParallelism::Sequential)
    };
    assert_eq!(
        CouncilMemoryBudget::admit(&zero_kv_heads, &roomy_resources())
            .refusal
            .expect("zero KV heads must fail")
            .code,
        CouncilMemoryRefusalCode::InvalidRequest
    );
    let zero_context = CouncilMemoryRequest {
        context_length: 0,
        ..residual_request(CouncilParallelism::Sequential)
    };
    assert_eq!(
        CouncilMemoryBudget::admit(&zero_context, &roomy_resources())
            .refusal
            .expect("zero context must fail")
            .code,
        CouncilMemoryRefusalCode::InvalidRequest
    );
    let overflow = CouncilMemoryRequest {
        num_kv_heads: u32::MAX,
        head_dim: u64::MAX,
        context_length: u32::MAX,
        ..residual_request(CouncilParallelism::Sequential)
    };
    assert_eq!(
        CouncilMemoryBudget::admit(&overflow, &roomy_resources())
            .refusal
            .expect("overflow must fail")
            .code,
        CouncilMemoryRefusalCode::ArithmeticOverflow
    );
}

#[test]
fn metadata_requires_integral_head_dimension() {
    let mut invalid = metadata();
    invalid.embedding_dim = 65;
    assert!(
        council_memory_request_for_metadata(
            &invalid,
            4,
            CouncilExecutionMode::Answer,
            CouncilParallelism::Sequential,
            CouncilKCacheLayout::ResidualParity,
            16.0,
        )
        .is_err()
    );
}

#[test]
fn gpu_budget_uses_selected_k_layout_and_context_peak() {
    let hw = hardware(3 << 30);
    let single = compute_gpu_budget_with_kv_layout(
        &hw,
        &metadata(),
        4,
        CouncilKCacheLayout::ResidualParity,
        16.0,
        1,
    )
    .expect("valid residual budget");
    let parallel = compute_gpu_budget_with_kv_layout(
        &hw,
        &metadata(),
        4,
        CouncilKCacheLayout::ResidualParity,
        16.0,
        3,
    )
    .expect("valid parallel residual budget");

    assert_eq!(single - parallel, (168 + 102) * 2);
}
