# Hypura v1.0.0 Triality / NC-KA / URT implementation report

Date: 2026-07-14
Release target: `v1.0.0` stable and `main-v1.0.0` main channel

## Source authority and immutable pins

This implementation uses `zapabob/llama.cpp` and `zapabob/Turboquant-CUDA` as the canonical upstreams requested for the release. Hypura pins llama.cpp at `81b08be15c554918a8cfe20d5f0846038082430f`, Turboquant-CUDA at `d2e992e54a557990f9d2e38834987dbeb28d8ec0`, and Turboquant-CUDA's nested llama.cpp at the same exact llama.cpp commit. The Hypura feature implementation commit is `ece7acaed4c35ace5f9a0ffeeaa7b3bdcb92c469`.

## Implemented behavior

The release adds the schema-v2 Triality identity and verification path, three-view context lifecycle, Attention Council, Answer Council, deterministic best-of-views selection, optional cross-view proxy scoring, NC-KA diagnostics, URT diagnostics, and public telemetry without persisting private chain-of-thought. Existing schema-v1 reading remains supported. The prior vector path remains the default so an upgrade does not silently change inference behavior.

Cross-score is explicitly a proxy score. It is suitable for deterministic selection diagnostics but is not presented as a calibrated probability or a model-quality guarantee. NC-KA rank and URT error remain nullable when the model or fixture does not expose the required independent low-level observations.

The Aha gate is disabled in the release evidence because no labeled evaluation suite was supplied. Precision and activation frequency are therefore unavailable, and this report makes no Aha quality-improvement claim.

## CPU and workspace verification

`cargo test --workspace --locked -j1` completed with no failures. The root library reported 158 passed and one ignored test, while the CLI unit suite reported 11 passed; the remaining workspace integration and documentation suites also passed. The complete log is `H:\hypura-release-artifacts\hypura-cpu-workspace-v1.0.0.log`, SHA-256 `DA1753B7CAAF9B6BB3B01C25B3E70AAD10F978D99DE962002D2413D14B4BDF86`.

The ignored schema-v2 library fixture intentionally refuses identity-view fallback without a developer override. The live wiring integration was run explicitly against `stories260K-Q4_0-triality-v2-s260k.gguf`, SHA-256 `5033056f4adb74a545126c9e44bc5a4c9b12253c88070154636dae381cfdaee7`, with the fixture-only developer override and passed. Production defaults were not weakened to make that fixture pass.

Turboquant-CUDA passed 60 of 60 schema-v2 export, verification, round-trip, and negative tests. Its S3 controller hash is `7a1bdd43cdc7e105076d8171b1e29436dbb5367b7616330feb4fa9581bbffab9`; the verified oracle weights are `[0.32970839831029586, 0.34043480116596675, 0.3298568005237375]`.

The focused llama.cpp CPU suite passed 13 of 13 tests. The broader inherited llama.cpp CPU suite remains 63 of 64 because the unchanged `test-quantize-fns` q2_0 threshold case fails in the canonical baseline. This is recorded as a baseline exception rather than being hidden or rewritten for the release.

Advisory `cargo clippy --workspace --all-targets --locked -j1` exits successfully. Strict `-D warnings` remains a baseline exception because Rust 1.94 reports workspace-wide pre-existing style warnings. One substantive raw-pointer callback issue found during the advisory run was corrected by making the callback explicitly unsafe, documenting its safety contract, and containing the call in an explicit unsafe block.

## Live Triality behavior

The live fixture verified deterministic sequential, parallel, best-of-views, and attention modes. Sequential execution peaked at one active context and took 89 ms; parallel execution peaked at three active contexts and took 69 ms. Candidate latencies were 2, 2, and 1 ms. The recorded cross scores were all `-0.558008879874067`, so vector was selected with a zero margin. A tied proxy score is not evidence of semantic superiority.

NC-KA rank is unavailable for this identity-view fixture because it exposes no independent low-level rank observation. URT error is likewise unavailable because it exposes no independent cross-representation observation. Both values are stored as `null` with reasons in the machine-readable result.

## CUDA 12.8 and RTX 50 verification

CUDA operator tests ran on an NVIDIA GeForce RTX 5060 Ti with CUDA compiler 12.8.61. Five focused CUDA tests and three live tests comprising 41 assertions passed. `cuobjdump` found 179 embedded cubins and every one was `sm_120`. Dependency inspection found CUDA 12 runtime libraries and no CUDA 13 runtime dependency.

The maximum attention-output Frobenius relative error was `3.06652582e-07`, including the direct f16 d=8 case. Prefill latency was 0.32916 ms mean, 0.2828 ms p50, and 0.5055 ms p95. Decode latency was 0.040475 ms mean, 0.0358 ms p50, and 0.0652 ms p95.

Peak VRAM is unavailable because the standalone operator backend does not expose per-operation peak allocation. Next-logit KL and hidden-state cosine are unavailable because this operator test emits neither full-model logits nor hidden states. These fields are `null`; no replacement estimate is used.

## GitHub CI baseline audit

The exact llama.cpp main commit has a successful Build Actions Cache run, `29281696076`. Its CANN run, `29280284880`, is an inherited upstream baseline exception: `build-cann.yml` defines no executable jobs, so GitHub rejects it before creating a job or log. The workflow blob is unchanged from `ggml-org/llama.cpp` master and was not touched by the Triality implementation. Rewriting an unrelated canonical upstream workflow solely to remove a red badge was rejected in favor of recording the limitation.

Turboquant-CUDA and Hypura did not contain repository-native GitHub workflows before this release. Their published verification therefore relies on the recorded local schema-v2, workspace, live-model, CUDA, CLI, API, and Desktop gates rather than claiming unavailable GitHub CI coverage.

## Compatibility and rollback

Schema-v1 GGUF reading, schema-v2 round-trip verification, the unchanged native API surface, and KoboldCpp-compatible API behavior are release gates. The new Council surface is additive. Operational rollback is available by using the default vector path or disabling Council behavior; binary rollback remains the preceding signed or checksummed release artifact. The identity-view fallback switches are restricted to developer fixtures and must not be used for production quality claims.

## Evidence index

The machine-readable companion is `benchmarks/results/2026-07-14-triality-consensus-v1.0.0.json`. CUDA JSONL evidence is stored at `H:\hypura-release-artifacts\tq-triality-consensus-sm120-gpu-final.jsonl`; llama.cpp CPU evidence is stored at `H:\hypura-release-artifacts\llama-cpu-postcommit-focused-ctest.log`; dependency evidence is stored at `H:\hypura-release-artifacts\llama-cuda-dll-dependents.log`. Final CLI, API, Desktop, installer, overwrite-install, GitHub branch, tag, release, and downloaded-asset checks are appended to the release evidence set during publication.
