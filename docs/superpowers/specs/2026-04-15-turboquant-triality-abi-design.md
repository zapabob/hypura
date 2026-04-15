# TurboQuant / Triality Phase A-B Design

Date: 2026-04-15

## Scope

This design covers the first implementation slice for:

- `zapabob/Turboquant-CUDA`
- `zapabob/llama.cpp`

The implementation scope is intentionally limited to **Phase A + Phase B** from the instruction set:

- Phase A: naming cleanup, metadata/docs drift cleanup, schema version introduction
- Phase B: ABI fixation for mixed-bit metadata, export/import contract alignment, parity-test scaffolding

This slice does **not** implement:

- adaptive block view selection
- true `Spin(8)` triality runtime
- key-only CUDA kernel optimization
- PPL / NIAH end-to-end evaluation wiring

Those remain follow-up phases and must build on the ABI introduced here.

## Goals

1. Make the current production path explicitly identifiable as a **proxy** path rather than a mathematically strict Spin(8) triality implementation.
2. Define a shared artifact contract so PyTorch export and `llama.cpp` import can reconstruct the same TurboQuant configuration without relying on implicit `floor(bits)` behavior.
3. Prevent silent or ambiguous decode in `llama.cpp` when required TurboQuant metadata is missing.
4. Prepare a stable foundation for later work on adaptive views, true `Spin(8)`, sign bitpacking, and runtime optimization.

## Non-Goals

- Replacing the existing production proxy path
- Shipping true `Spin(8)` into production
- Reworking the whole runtime around a new artifact format in one pass
- Treating attention replay alone as sufficient production validation

## Current State Summary

### Turboquant-CUDA

- `turboquant/research_extension/triality_proxy.py` already states that the current adapters are empirical proxies.
- `turboquant/research_extension/k_triality.py` still exports production-facing mode names such as:
  - `key_only_block_so8_triality_vector`
  - `key_only_block_so8_triality_plus`
  - `key_only_block_so8_triality_minus`
- The current artifact and reporting surface still risks conflating proxy views with mathematically strict triality.
- Mixed-bit handling depends on split behavior between Stage 1 allocation and Stage 2 QJL bits, but this decomposition is not yet formalized as a strict storage contract.

### llama.cpp side

- GGUF metadata currently writes:
  - `hypura.turboquant.mode`
  - `hypura.turboquant.rotation_policy`
  - `hypura.turboquant.rotation_seed`
  - `hypura.turboquant.triality_view`
  - `hypura.turboquant.triality_mix`
  - `hypura.turboquant.artifact`
- The exposed rotation-policy labels still use names such as:
  - `triality_vector`
  - `triality_spinor_plus`
  - `triality_spinor_minus`
- `src/model/turboquant_sidecar.rs` reconstructs runtime config from these metadata keys, but currently accepts sparse metadata and cannot enforce the full mixed-bit decomposition described in the new requirements.

## Recommended Migration Strategy

Use a **staged compatibility migration**:

1. Define the new canonical names and schema in Python first.
2. Teach `llama.cpp` to read both legacy names and canonical names during the transition.
3. Write only canonical names for newly exported artifacts.
4. Refuse decode when required new-schema fields are missing for a declared new-schema artifact.
5. Keep legacy compatibility constrained to explicit legacy reads, never as a silent fallback from incomplete new metadata.

This avoids breaking existing research outputs while preventing the ABI from staying ambiguous.

## Design

### 1. Naming model: proxy versus true triality

The naming surface is split into two families:

- **proxy family**
  - `triality_proxy_vector`
  - `triality_proxy_spinor_plus`
  - `triality_proxy_spinor_minus`
- **true Spin(8) family** reserved for later
  - `triality_spin8_vector`
  - `triality_spin8_spinor_plus`
  - `triality_spin8_spinor_minus`

Production-facing documentation must describe the current path as **proxy**. The existing names without `proxy` are treated as legacy aliases only.

#### Python-side mode mapping

`Turboquant-CUDA` mode identifiers for the current production path become:

- `key_only_block_so8_triality_proxy_vector`
- `key_only_block_so8_triality_proxy_plus`
- `key_only_block_so8_triality_proxy_minus`

Legacy identifiers remain readable in reports or loaders only when explicitly mapped forward.

#### GGUF / runtime naming

`llama.cpp` canonical metadata values become:

- `triality_proxy_vector`
- `triality_proxy_spinor_plus`
- `triality_proxy_spinor_minus`

The future true-triality family will use separate values and separate enum variants rather than overloading the proxy enum.

### 2. Metadata / artifact ABI contract

Every newly exported TurboQuant artifact in Phase B must declare:

- `tq_schema_version`
- `tq_total_bits`
- `tq_stage1_effective_bits`
- `tq_qjl_bits`
- `tq_qjl_dim`
- `tq_rotation_policy`
- `tq_rotation_seed`
- `tq_qjl_seed`
- `tq_triality_mode`
- `tq_triality_view`
- `tq_stage1_allocation_scheme`
- `tq_stage1_bitwidth_payload_dtype`
- `tq_norm_dtype`
- `tq_sign_pack_format`

Where relevant, the artifact may also include compatibility / display helpers such as:

- `tq_nominal_bits`
- `tq_value_bits`
- `tq_runtime_avg_bits_per_channel`

but those do not replace the required fields above.

### 3. Mixed-bit semantics

Mixed-bit settings like `2.5` or `3.5` must no longer be interpreted through implicit floor behavior. Instead:

- `tq_total_bits` records the nominal total budget label used by the experiment or runtime contract.
- `tq_stage1_effective_bits` records the effective Stage 1 channel allocation target.
- `tq_qjl_bits` records the Stage 2 QJL bit contribution.
- `tq_stage1_allocation_scheme` records how Stage 1 allocation was assigned.

This allows both PyTorch and `llama.cpp` to reproduce the same configuration from explicit fields rather than by re-deriving intent from a single float.

### 4. Schema version policy

Introduce `tq_schema_version` as a strict discriminator.

- Legacy artifacts without `tq_schema_version` are treated as **legacy TurboQuant artifacts** and parsed only through the legacy compatibility path.
- New artifacts with `tq_schema_version >= 1` must satisfy the Phase B required-field set.
- `llama.cpp` must reject new-schema artifacts when any required field is missing.
- New-schema artifacts must never silently degrade to old behavior.

### 5. `llama.cpp` loader behavior

`src/model/turboquant_sidecar.rs` and GGUF readers will adopt this rule set:

1. If no TurboQuant metadata exists, behavior stays unchanged.
2. If legacy metadata exists, load it through an explicit legacy alias mapper.
3. If `tq_schema_version` exists, enforce the new required-field contract.
4. If the artifact declares a new schema but required data is missing, return an error instead of exact fallback.

This preserves safety and prevents silently wrong decode.

### 6. Documentation layers

`Turboquant-CUDA` documentation is reorganized into three explicit layers:

- **production**
  - what the current runtime path is
  - what is default
  - what artifact contract downstream must obey
- **paper-faithful**
  - what reproduces the paper baseline
  - where the baseline diverges from the production proxy path
- **research**
  - proxy ablations
  - future true Spin(8)
  - adaptive-view experiments

This is a documentation boundary only in Phase A-B; the implementation remains mostly where it is.

### 7. Parity-test scaffolding

Phase B adds a cross-repo parity scaffold rather than the full final benchmark matrix.

The first golden set should compare:

- quantized indices
- Stage 1 bitwidth payload
- norms
- QJL sign payload
- pairwise estimated logits

The parity harness should be designed to grow later into:

- adaptive-view tag checks
- value-path comparisons
- generation-oriented metrics

But the initial scope is artifact and decode parity, not end-to-end generation quality.

## Repository-Level Changes

### Turboquant-CUDA

Planned Phase A-B changes:

- README wording update to consistently mark the current production path as proxy
- `implementation_plan.md` scope reconciliation with README
- Python version wording unification
- `reference_paper` correction to the TurboQuant paper
- manifest / CSV / JSON output schema update
- legacy-to-canonical mode alias mapping
- parity export helpers and golden artifact generation hooks

### llama.cpp

Planned Phase A-B changes:

- canonical proxy rotation-policy enum values
- legacy alias support for current `triality_*` metadata names
- `tq_schema_version` recognition
- stricter GGUF / sidecar metadata validation
- required-field enforcement for new-schema artifacts
- parity-test input loader support

## Data Flow

### Export path

1. PyTorch quantization run produces encoded tensors and metadata.
2. Export layer writes canonical proxy naming plus required ABI fields.
3. Optional GGUF metadata mirrors the same canonical naming and schema version.
4. Golden parity artifacts are written from the same canonical export path.

### Import path

1. `llama.cpp` inspects GGUF metadata or sidecar.
2. Legacy artifacts are routed through explicit alias normalization.
3. New-schema artifacts are validated against the required ABI set.
4. Runtime config is constructed only after successful validation.

## Error Handling

- Missing required fields on a declared new-schema artifact is a hard error.
- Unsupported `tq_triality_view` or `tq_rotation_policy` is a hard error.
- Legacy alias use may emit a warning, but must still normalize deterministically.
- Exact fallback remains allowed only for the old "no research sidecar / no metadata" case, not for malformed new-schema artifacts.

## Testing

Phase A-B testing will require:

- unit tests for alias normalization
- unit tests for schema required-field validation
- unit tests for mixed-bit metadata decomposition
- parity-test scaffold that compares Python export against `llama.cpp` import expectations
- regression check that old proxy artifacts still load through the legacy path

Generation metrics, NIAH, and runtime benchmarking are explicitly deferred to later phases.

## Risks

1. Legacy artifacts may encode behavior that was never fully written down; alias mapping may need a narrow compatibility table rather than one generic rule.
2. A partial schema rollout could create artifacts that mix canonical and legacy keys; loaders must reject ambiguous combinations.
3. If naming cleanup is done in docs but not in manifests or GGUF metadata, the confusion remains operationally intact.

## Acceptance Criteria

Phase A-B is complete when:

- production docs consistently describe the current path as **proxy**
- new artifact exports write canonical proxy naming
- `tq_schema_version` exists and gates the new required-field contract
- mixed-bit exports no longer require implicit `floor(bits)` behavior to be reconstructed
- `llama.cpp` can normalize legacy names but rejects malformed new-schema artifacts
- a parity scaffold exists for Python export versus `llama.cpp` import

## Deferred Work

The following are intentionally deferred:

- `triality_spin8.py`
- adaptive block-level view selection
- sign-bit packing optimization implementation
- norm storage compression implementation
- fast Hadamard runtime de-materialization
- key-only CUDA kernel optimization
- PPL / NIAH / generation benchmark wiring

These tasks depend on the ABI and naming cleanup from this design.
