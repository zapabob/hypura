use std::process::Command;

fn hypura() -> Command {
    Command::new(env!("CARGO_BIN_EXE_hypura"))
}

#[test]
fn council_help_exposes_operator_workflow() {
    let output = hypura().args(["council", "--help"]).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--prompt",
        "--max-tokens",
        "--parallelism",
        "--cross-score",
        "--aha",
        "--output-dir",
        "--dry-run",
    ] {
        assert!(stdout.contains(flag), "missing {flag} in {stdout}");
    }
}

#[test]
fn b4_triality_flags_are_available_on_every_required_command() {
    for command in ["run", "serve", "koboldcpp", "bench"] {
        let output = hypura().args([command, "--help"]).output().unwrap();
        assert!(output.status.success(), "{command} --help failed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        for flag in [
            "--tq-triality-execution",
            "--tq-triality-weights",
            "--tq-triality-trace",
            "--tq-ncka-required",
            "--tq-urt",
        ] {
            assert!(stdout.contains(flag), "{command} is missing {flag}");
        }
    }
}

#[test]
fn malformed_triality_weights_fail_before_model_access() {
    let output = hypura()
        .args([
            "run",
            "missing.gguf",
            "--tq-triality-weights",
            "0.2,0.2,0.2",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("must sum to one"), "{stderr}");
}
