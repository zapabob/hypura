use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{anyhow, bail, Context};

use crate::compute::ffi::PerfData;
use crate::compute::inference::GenerationResult;

#[derive(Debug, Clone)]
pub struct MultimodalInvocation {
    pub model_path: PathBuf,
    pub mmproj_path: PathBuf,
    pub prompt: String,
    pub image_paths: Vec<PathBuf>,
    pub audio_paths: Vec<PathBuf>,
    pub context: u32,
    pub max_tokens: u32,
    pub n_gpu_layers: i32,
}

#[derive(Debug, Clone)]
pub struct MultimodalBridgeResponse {
    pub generated_text: String,
    pub result: GenerationResult,
}

pub fn find_mtmd_cli(explicit_llama_root: Option<&Path>) -> anyhow::Result<PathBuf> {
    if let Ok(explicit_path) = std::env::var("HYPURA_MTMD_CLI_PATH") {
        let candidate = PathBuf::from(explicit_path.trim());
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    let mut roots = Vec::new();
    if let Some(root) = explicit_llama_root {
        roots.push(root.to_path_buf());
    }
    if let Ok(root) = std::env::var("HYPURA_LLAMA_CPP_PATH") {
        let candidate = PathBuf::from(root.trim());
        if !candidate.as_os_str().is_empty() {
            roots.push(candidate);
        }
    }

    for root in roots {
        if !root.exists() {
            continue;
        }
        for executable_name in executable_candidates() {
            if let Some(found) = find_named_file(&root, executable_name)? {
                return Ok(found);
            }
        }
    }

    Err(anyhow!(
        "Could not find built llama-mtmd-cli/mtmd-cli. Set HYPURA_MTMD_CLI_PATH or HYPURA_LLAMA_CPP_PATH to a llama.cpp build tree."
    ))
}

pub fn run_mtmd_single_turn(
    invocation: &MultimodalInvocation,
    explicit_llama_root: Option<&Path>,
) -> anyhow::Result<MultimodalBridgeResponse> {
    if invocation.image_paths.is_empty() && invocation.audio_paths.is_empty() {
        bail!("multimodal bridge requires at least one image or audio input");
    }
    if !invocation.model_path.is_file() {
        bail!("model path does not exist: {}", invocation.model_path.display());
    }
    if !invocation.mmproj_path.is_file() {
        bail!("mmproj path does not exist: {}", invocation.mmproj_path.display());
    }

    let mtmd_cli = find_mtmd_cli(explicit_llama_root)?;
    let started = Instant::now();

    let mut command = Command::new(&mtmd_cli);
    command
        .arg("-m")
        .arg(&invocation.model_path)
        .arg("--mmproj")
        .arg(&invocation.mmproj_path)
        .arg("-p")
        .arg(&invocation.prompt)
        .arg("-n")
        .arg(invocation.max_tokens.to_string())
        .arg("-c")
        .arg(invocation.context.to_string())
        .arg("-ngl")
        .arg(invocation.n_gpu_layers.max(1).to_string())
        .arg("--simple-io")
        .arg("--no-warmup")
        .arg("--no-conversation")
        .arg("--no-display-prompt")
        .arg("--reasoning-format")
        .arg("none")
        .arg("--log-disable");

    for image_path in &invocation.image_paths {
        command.arg("--image").arg(image_path);
    }
    for audio_path in &invocation.audio_paths {
        command.arg("--audio").arg(audio_path);
    }

    let output = command
        .output()
        .with_context(|| format!("failed to launch {}", mtmd_cli.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        bail!(
            "mtmd-cli failed with status {:?}: stdout={} stderr={}",
            output.status.code(),
            stdout,
            stderr
        );
    }

    let generated_text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if generated_text.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!("mtmd-cli produced no output. stderr={stderr}");
    }

    let completion_tokens = generated_text.split_whitespace().count().max(1) as u32;
    let elapsed = started.elapsed().as_secs_f64().max(1e-6);
    let prompt_tokens = invocation.prompt.split_whitespace().count() as u32;
    let result = GenerationResult {
        text: generated_text.clone(),
        tokens_generated: completion_tokens,
        prompt_tokens,
        tok_per_sec_avg: completion_tokens as f64 / elapsed,
        prompt_eval_ms: 0.0,
        perf: PerfData::default(),
        context_state: None,
    };

    Ok(MultimodalBridgeResponse {
        generated_text,
        result,
    })
}

fn find_named_file(root: &Path, name: &str) -> anyhow::Result<Option<PathBuf>> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|value| value.eq_ignore_ascii_case(name))
                .unwrap_or(false)
            {
                return Ok(Some(path));
            }
        }
    }
    Ok(None)
}

fn executable_candidates() -> &'static [&'static str] {
    if cfg!(windows) {
        &["llama-mtmd-cli.exe", "mtmd-cli.exe"]
    } else {
        &["llama-mtmd-cli", "mtmd-cli"]
    }
}
