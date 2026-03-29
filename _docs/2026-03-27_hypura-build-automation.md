# Hypura Build Automation

- Date: 2026-03-27
- Repository: `hypura-main`
- Purpose: keep heavy Cargo build artifacts on `F:` and provide a repeatable compile / warning gate for the full workspace

## Why `F:`

The current Windows environment has intermittent build-script execution and cache issues when heavy Cargo output stays on the default drive. The build helper moves the expensive writable paths to `F:\hypura-build` so the repo can keep a stable local entrypoint for compile checks without baking machine-specific state into tracked source files.

## Script

Use:

```powershell
pwsh -File .\scripts\hypura-build-check.ps1 -Mode check
pwsh -File .\scripts\hypura-build-check.ps1 -Mode full
```

Modes:

- `check`: workspace `cargo check` with warning counting
- `full`: workspace `cargo check` plus `cargo test --workspace --lib`

## Environment

The script fixes these variables per run:

- `CARGO_TARGET_DIR=F:\hypura-build\target`
- `CARGO_HOME=F:\hypura-build\cargo-home`
- `RUSTC_WRAPPER=` to avoid stale `sccache` interference

`RUSTUP_HOME` is not forced by default. If you want to relocate the toolchain cache as well, set `HYPURA_RUSTUP_HOME` before invoking the script.

## Output

The script writes:

- `logs/build-check.log`
- `logs/build-check-summary.json`

The JSON summary includes:

- success / failure
- warning count
- error count
- failing command
- first warning
- first error

## Warning-zero definition

`warning 0` means:

- workspace `cargo check` succeeds
- if `Mode=full`, workspace `cargo test --workspace --lib` succeeds
- the collected output contains zero lines matching `warning:`

This gate is intentionally strict and applies to the full workspace:

- `hypura`
- `hypura-sys`
