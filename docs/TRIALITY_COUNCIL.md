# Triality Council

Hypura v1.0.0 adds a dedicated Council execution surface without changing the default behavior of the existing OpenAI, Ollama, or Kobold-compatible routes.

## Execution surfaces

The dedicated Council route and `hypura council` command implement Answer Council: three independent contexts over one loaded model followed by deterministic cross-scoring. They reject `attention_consensus=true` and `synthesis=true` because v1.0.0 does not combine those operations with answer selection.

The ordinary single-context `run`, `serve`, and compatible worker paths expose native per-layer Triality execution separately through the explicit TurboQuant context configuration. Those paths may request single-view, best-per-layer, attention-logit consensus, or residual parity when the loaded model and pinned llama.cpp runtime advertise the corresponding capability.

The three canonical views are `vector`, `spinor_plus_proxy`, and `spinor_minus_proxy`. The latter two are proxy views; they are not claims of a complete mathematical spinor representation.

Attention and residual-parity modes are capability negotiated. If the pinned llama.cpp build does not advertise the requested production capability, a required request fails closed. Optional operation may use the explicitly reported fallback policy. Storage-changing modes are reserved during context creation and cannot be enabled after successful encode or decode begins.

## Answer Council

Answer Council uses one loaded `llama_model` and three independent `llama_context` instances. Each context owns its Triality configuration. No process-global environment variable is used to select a view.

Sequential execution is the default. `auto` chooses parallel execution only after the scheduler admits three contexts while preserving configured memory headroom. An explicit parallel request returns an admission error when the estimate is unsafe.

Each branch generates a candidate under the same normalized request parameters. Hypura then teacher-forces every candidate through every evaluator context. This produces a deterministic 3 by 3 mean log-likelihood matrix for a fixed model, prompt, seed, and runtime configuration. Candidate selection uses the public score aggregates and canonical view order for tie-breaking. Private reasoning is neither requested nor returned.

## HTTP API

The dedicated routes are:

```text
POST /api/extra/triality/council
POST /v1/triality/council
GET  /api/extra/triality/council/:id
GET  /api/extra/triality/events
```

A request accepts either `prompt` or `messages`, but not both. Common controls include `max_tokens`, `temperature`, `seed`, `parallelism`, `cross_score`, `aha`, `trace`, and `stream`. The accepted compatibility fields `attention_consensus` and `synthesis` must remain false for the v1.0.0 Answer Council surface; a true value returns a structured unsupported-operation error.

For streaming requests, provisional branch answers are never emitted through the standard OpenAI stream. Hypura completes selection first and streams only the chosen final answer. Progress events belong to the dedicated events route.

The existing chat routes retain their current behavior unless the optional extension is present:

```json
{
  "hypura": {
    "triality_council": true,
    "trace": false
  }
}
```

## CLI

```powershell
hypura council .\model.gguf `
  --prompt 'Explain the result.' `
  --max-tokens 256 `
  --parallelism sequential `
  --cross-score `
  --aha `
  --output-dir artifacts\triality_council
```

Dry run reports the resolved GGUF schema, llama.cpp capabilities, selected Council mode, context count, estimated KV memory, residency decision, NC-KA status, URT status, and Aha calibration state without generating text.

## Persistence and privacy

HTTP Council summaries use the service application-data root and retain a bounded number of request records. The CLI writes below the parent of its explicit `--output-dir`; the documented default is `artifacts/triality_council`. Raw prompt text, provisional candidate text, and final text are excluded unless trace was explicitly requested and the configured storage policy permits text persistence. Request identifiers are validated before they are used as path components.

The event stream contains aggregate progress and numerical diagnostics. It does not contain chain-of-thought or private reasoning fields.

## Determinism and limitations

Fixed-seed selection is deterministic only when the selected backend and kernels are deterministic. CPU and CUDA paths are compared within the tolerance published by the release validation record. The schema fixture proves contract serialization and parser behavior; live generation requires a runnable model.

Aha is disabled unless a versioned labeled calibration record and safety comparator are supplied. Source code and an unlabeled smoke test cannot establish the false-positive target.
