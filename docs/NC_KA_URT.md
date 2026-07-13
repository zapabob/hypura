# NC-KA and URT Contracts

This document describes the boundary between the Turboquant schema producer, llama.cpp inference primitives, and Hypura request-level policy.

## Ownership

`zapabob/Turboquant-CUDA` produces and verifies schema-v2 metadata and tensors. `zapabob/llama.cpp` parses the low-level contract and manifest fields, owns context configuration, validates rotation-tensor shapes and SO(8) values, and exposes finite numerical telemetry. Hypura verifies the embedded controller's physical-byte SHA-256, evaluates request-level NC-KA, performs tier-aware admission and selection, and persists privacy-safe Council and URT reports.

Schema v1 remains the legacy single-view path. Schema v2 is fail closed: missing keys, contradictory values, unexpected tensors, invalid dimensions, non-finite values, and hash mismatches are rejected.

## NC-KA input boundary

The NC-KA controller accepts finite scalar observables only. Its canonical coordinates include branch entropy, rotation validation error, expected quantization error, pairwise Jensen-Shannon divergence, candidate cross-score moments, winner margin, latency ratio, and memory ratio.

The controller does not consume prompt text, candidate text, hidden chain-of-thought, or arbitrary model activations. Coordinates are normalized with the hashed schema-v2 normalization contract before interpolation.

`finite_moment_ka_v1` fixes protocol version 1 and the 24-coordinate order in `NCKA_COORDINATE_NAMES`. The schema hash binds that name order, the `[0,1]` tensor range, and clamping. Hypura applies the following protocol transform before interpolation; a future formula change requires a new controller kind or protocol version.

| Coordinates | Protocol transform |
| --- | --- |
| Branch entropy, orthogonality error, determinant error, expected quantization error | Require a finite nonnegative value and map `x` to `x / (1 + x)` |
| Pairwise Jensen-Shannon divergence | Require `0 <= x <= ln(2)` within numerical tolerance, clamp to that interval, and divide by `ln(2)` |
| Candidate cross-score mean log-likelihood | Require a nonpositive finite value and map `x` to `exp(x)`, the geometric mean token probability |
| Candidate cross-score variance and winner margin | Require a finite nonnegative value and map `x` to `x / (1 + x)` |
| Latency multiplier | Require `x >= 1` within numerical tolerance and map it to `(max(1, x) - 1) / max(1, x)` |
| Memory ratio | Require the admitted peak utilization to be within `[0,1]` and clamp only numerical boundary error |

The memory ratio is the maximum, over every projected nonzero GPU, host-pageable, host-pinned, and unified region, of `projected bytes / capacity remaining after committed bytes and required headroom`. Missing capacity, arithmetic overflow, a rejected admission, or a non-finite derived value fails closed. These transforms are monotone, preserve simultaneous S3 permutation, and prevent ordinary raw log-likelihood and latency values from collapsing onto tensor boundaries.

The finite-moment gate checks matrix rank and conditioning before controller evaluation. For the 3 by 3 branch moment matrix, numerical and effective rank above 3 are invalid. Rank deficiency, unsupported controller type, unavailable tensors, or invalid numerical evidence selects the declared static fallback when the controller is optional. A required unsupported controller is rejected.

Simultaneous permutation of the three views must permute the controller result in the same way. Tests cover the canonical S3 substitutions and deterministic fallback.

## URT representation identity

URT compares observations only when their model hash, state identifier, layer, operator-word hash, and observable agree. A concrete representation identifier contains:

```text
kind
model_hash
artifact_hash
backend
precision
view
```

The registered representation kinds are `python_quantised_reference`, `llama_cpu_gguf`, `llama_cuda_gguf`, `hypura_native`, and `hypura_kobold_worker`. Equal abstract labels do not collapse distinct concrete representations.

Each comparison reports absolute error, declared tolerance, and consistency. Cross-model and cross-manifest comparisons are refused instead of being mixed into one result.

## URT persistence

Persistent URT data is stored under a configured data root. The HTTP service selects its application-data root; the CLI derives the root from the parent of its explicit `--output-dir`, whose documented default is `artifacts/triality_council`. Request correlation identifiers are SHA-256 transformed before persistence. Writes use a synchronized temporary file and replacement, and failed writes remove the temporary file.

The registry can run in memory-only mode. Persistent mode writes the representation registry, observation stream, consistency summary, and failure records. Retention and text policy are controlled separately by the Council store.

## Aha boundary

An Aha event describes resolved cross-view disagreement and improved observable support. It is not a guarantee of factual truth.

Event emission requires all of the following evidence:

1. A named safety comparator whose post-selection penalty does not worsen.
2. A versioned labeled calibration record with a nonzero sample count.
3. An observed false-positive rate no greater than 5 percent.
4. Either offline reference gain or online reduction in observable disagreement.
5. The configured score, margin, and optional URT improvement thresholds.

Invalid evidence is rejected. When required evidence is absent or valid evidence remains below threshold, the response records a disabled reason and does not manufacture an Aha event.

## Release evidence

The stable release record must identify the exact Hypura, llama.cpp, Turboquant-CUDA, and nested llama.cpp commits. It must also state CPU/CUDA tolerance, memory and latency impact, residual-parity bit accounting, Aha calibration dataset identity, and every unsupported layout or fallback.

When labeled Aha data is unavailable, the release must report Aha as uncalibrated and disabled. That is a valid safe state, but it is not evidence that the false-positive target was measured.
