# EasyNovelAssistant Backend Selection Design

## Overview

This design adds a first-class backend switch to EasyNovelAssistant so the user can choose either KoboldCpp or Hypura while keeping the existing novel-generation UX intact.

The selected backend must support the current EasyNovelAssistant generation loop, partial-output polling, and abort flow without forcing the rest of the UI to understand backend-specific protocol differences.

## Goal

Allow EasyNovelAssistant to run with either:

- KoboldCpp launched and controlled by EasyNovelAssistant
- Hypura launched in KoboldCpp compatibility mode and controlled by EasyNovelAssistant

The same EasyNovelAssistant UI, generation loop, and stop behavior should work in both cases.

## Requirements

1. Preserve the existing EasyNovelAssistant workflow for users who keep `llm_backend = "koboldcpp"`.
2. Make `llm_backend = "hypura"` launch Hypura in KoboldCpp-compatible mode instead of requiring a separate manual proxy flow.
3. Reuse the current Kobold-compatible API calls for `model`, `generate`, `check`, and `abort`.
4. Keep backend-specific logic out of `Generator` and the rest of the UI wherever possible.
5. Persist backend selection in `config.json`.
6. Show enough UI context that the user can tell which backend is active.
7. Fail clearly when the selected backend executable is missing or its API is unreachable.

## Non-Goals

- Adding a direct EasyNovelAssistant client for Hypura native `/api/generate` or `/api/chat`
- Replacing the current EasyNovelAssistant generation UX with a streaming-only architecture
- Changing Hypura's native server defaults outside the compat profile required for this integration
- Refactoring unrelated EasyNovelAssistant menu or prompt behavior

## Current State

### EasyNovelAssistant

EasyNovelAssistant currently assumes a KoboldCpp-shaped backend in:

- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\kobold_cpp.py`
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\generator.py`
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\form.py`
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\menu\model_menu.py`
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\menu\setting_menu.py`

The UI already stores `llm_backend`, but the current code path is still centered around a single `KoboldCpp` class.

### Hypura

Hypura already exposes the required KoboldCpp-compatible surface in compat mode, including:

- `/api/v1/model`
- `/api/v1/generate`
- `/api/extra/generate/check`
- `/api/extra/abort`

The entry point for this mode is the `hypura koboldcpp <model>` CLI flow in:

- `C:\Users\downl\Desktop\hypura-main\hypura-main\src\main.rs`
- `C:\Users\downl\Desktop\hypura-main\hypura-main\src\cli\koboldcpp.rs`

## Recommended Architecture

### Decision

Use a backend-selection facade in EasyNovelAssistant and keep both backends speaking the Kobold-compatible HTTP protocol.

### Why

This gives the smallest safe change:

- `Generator` can stay backend-agnostic
- polling and abort semantics remain the same
- the EasyNovelAssistant UI does not need a second inference protocol
- Hypura is integrated through its already-shipped compatibility surface rather than a new native API client

### Structure

EasyNovelAssistant will expose one runtime object through `ctx.kobold_cpp`, but that object becomes a facade that delegates to one of two concrete backends:

- `KoboldCppBackend`
- `HypuraBackend`

Both concrete backends implement the same behavior:

- `get_model()`
- `launch_server()`
- `generate()`
- `check()`
- `abort()`
- `generate_stream()` if retained

The facade keeps the rest of the application stable while allowing backend-specific launch logic.

## Launch and Runtime Flow

### Common flow

1. EasyNovelAssistant reads `llm_backend` from `config.json`.
2. `easy_novel_assistant.py` constructs the backend facade.
3. `Generator.initial_launch()` calls `get_model()`.
4. If a compat API is already available on `koboldcpp_host` and `koboldcpp_port`, EasyNovelAssistant reuses it.
5. If not available, `launch_server()` starts the selected backend.
6. Generation, partial output polling, and abort continue to use Kobold-compatible routes on the configured host and port.

### KoboldCpp mode

KoboldCpp mode keeps existing behavior:

- launch `koboldcpp.exe`
- pass the selected model path
- pass GPU layers and context size
- use the configured host and port

### Hypura mode

Hypura mode launches:

`hypura koboldcpp <model> --host <host> --port <port> --context <context>`

The first implementation keeps the command line intentionally minimal and limits launch flags to the ones EasyNovelAssistant already controls directly.

## Configuration Design

### Persisted config

`config.json` remains the persistence point.

Required persisted values:

- `llm_backend`: `"koboldcpp"` or `"hypura"`

New optional persisted value:

- `hypura_path`: optional explicit path to the Hypura executable; if omitted, EasyNovelAssistant resolves `hypura` from `PATH`

Existing values reused by both backends:

- `koboldcpp_host`
- `koboldcpp_port`
- `llm_name`
- `llm_context_size`
- `llm_gpu_layer`

### Path resolution

Hypura backend resolution order:

1. `hypura_path` if non-empty
2. `hypura` on `PATH`

KoboldCpp backend resolution stays:

- `KoboldCpp\koboldcpp.exe` on Windows
- existing Linux path handling where already present

## UI Design

### Backend selection

Add a visible backend selector in EasyNovelAssistant settings.

The selector uses two radio-style options:

- `KoboldCpp`
- `Hypura`

Changing the selection updates `ctx["llm_backend"]` and persists on normal app shutdown.

### Window title

Extend the title to include the active backend label:

- `EasyNovelAssistant - <model> [KoboldCpp]`
- `EasyNovelAssistant - <model> [Hypura]`

### Model selection

Model selection remains shared:

- the same menu-driven model choice feeds both backends
- the same direct GGUF selection can be used for both

Hypura launch should accept direct GGUF absolute paths without requiring a forced copy into `KoboldCpp\`.

## Error Handling

### KoboldCpp errors

Keep current behavior for missing executable, download failure, or unreachable API.

### Hypura errors

Return explicit messages for:

- Hypura executable not found
- Hypura process launch failure
- compat API still unreachable after launch
- selected GGUF path missing

Error messages must name Hypura directly so the user knows which backend failed.

### Failure observability

Do not silently swallow backend-launch failures. Existing `print`-based logging is not ideal, but the first implementation must at least preserve actionable console messages and attach backend identity to those messages.

## File Plan

### EasyNovelAssistant files expected to change

- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\kobold_cpp.py`
  - split or reorganize into backend facade plus concrete backend launch logic
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\easy_novel_assistant.py`
  - initialize the facade cleanly
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\form.py`
  - include backend label in title
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\src\menu\setting_menu.py`
  - add backend radio selection and a Hypura path entry so explicit executable resolution is available without depending on `PATH`
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\EasyNovelAssistant\setup\res\default_config.json`
  - add `hypura_path` default as an empty string
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\Run-EasyNovelAssistant-Hypura.bat`
  - update launcher guidance to match native backend selection
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\Run-EasyNovelAssistant-KoboldCpp.bat`
  - keep launcher compatibility aligned with backend selection wording
- `C:\Users\downl\Desktop\EasyNovelAssistant\EasyNovelAssistant\README.md`
  - document the new backend selector and launch expectations

### Hypura files that remain untouched unless a CLI gap is discovered during implementation

- `C:\Users\downl\Desktop\hypura-main\hypura-main\src\main.rs`
- `C:\Users\downl\Desktop\hypura-main\hypura-main\src\cli\koboldcpp.rs`

These files should only be touched if the current `hypura koboldcpp` CLI cannot support the launch shape EasyNovelAssistant needs.

## Verification Plan

### EasyNovelAssistant verification

1. Select `koboldcpp` and confirm existing launch flow still works.
2. Select `hypura` and confirm EasyNovelAssistant launches Hypura compat mode.
3. Confirm `/api/v1/model` returns after launch in both modes.
4. Confirm generation produces final output in both modes.
5. Confirm polling updates the partial generation area in both modes.
6. Confirm stop triggers `/api/extra/abort` in both modes.

### Hypura verification

1. Confirm `hypura koboldcpp <model>` still binds to the requested host and port.
2. Confirm compat routes return the expected KoboldCpp-shaped JSON.
3. Confirm no native `serve` behavior regresses if Hypura files are touched.

## Risks and Mitigations

### Risk: EasyNovelAssistant repo is already dirty

Observed uncommitted changes exist in the EasyNovelAssistant repository, including target files such as `src/kobold_cpp.py` and `src/generator.py`.

Mitigation:

- read current file contents before editing
- avoid overwriting unrelated local changes
- keep the implementation narrow and explicit

### Risk: Existing experimental Hypura logic in EasyNovelAssistant

There are signs of prior local experimentation around Hypura behavior inside `kobold_cpp.py`.

Mitigation:

- do not discard those edits blindly
- normalize them into the final facade design if they align with the approved architecture
- stop and reassess only if those local edits conflict materially with the approved flow

### Risk: No automated test harness in EasyNovelAssistant

The Python application appears to rely mainly on runtime behavior rather than a strong automated test suite.

Mitigation:

- add focused automated coverage where practical for backend command construction and backend selection logic
- complement with explicit manual smoke checks
- record any skipped automation and residual risk in the implementation log

## Acceptance Criteria

- User can choose `KoboldCpp` or `Hypura` from EasyNovelAssistant settings.
- The choice persists across restarts.
- `KoboldCpp` mode preserves current behavior.
- `Hypura` mode launches Hypura compat mode without an external proxy requirement.
- Generation, partial-output polling, and abort work in both modes through the same EasyNovelAssistant UI flow.
- Errors clearly identify which backend failed and why.

## Implementation Notes

- Prefer the smallest safe change over a broad refactor.
- Keep `Generator` unchanged unless a backend-agnostic bug forces a change.
- If backend-specific code grows beyond a tidy amount inside `kobold_cpp.py`, split it into focused helper modules rather than building one oversized file.
