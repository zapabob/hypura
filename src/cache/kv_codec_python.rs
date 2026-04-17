use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const TURBOQUANT_PYTHON_PATH: &str = "vendor/turboquant-cuda";

#[derive(Clone)]
#[allow(dead_code)]
pub struct TurboQuantCodec {
    python_path: PathBuf,
    config_path: Option<PathBuf>,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
    mode: String,
    rotation_policy: String,
    rotation_seed: u32,
    triality_view: Option<String>,
    cachedir: PathBuf,
}

impl TurboQuantCodec {
    pub fn new(
        mode: String,
        config_path: Option<&str>,
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rotation_policy: Option<&str>,
        triality_view: Option<&str>,
        rotation_seed: Option<u32>,
    ) -> anyhow::Result<Self> {
        let vendor_path = resolve_turboquant_python_path();

        let cachedir = std::env::temp_dir().join("hypura_turboquant");
        std::fs::create_dir_all(&cachedir)?;

        let (rot_policy, triality) = parse_rotation_policy(rotation_policy, triality_view);

        Ok(TurboQuantCodec {
            python_path: vendor_path,
            config_path: config_path.map(PathBuf::from),
            num_layers,
            num_kv_heads,
            head_dim,
            mode,
            rotation_policy: rot_policy,
            rotation_seed: rotation_seed.unwrap_or(0),
            triality_view: triality,
            cachedir,
        })
    }

    fn python_executable(&self) -> PathBuf {
        if let Ok(explicit) = std::env::var("HYPURA_TURBOQUANT_PYTHON") {
            let path = PathBuf::from(explicit);
            if path.exists() {
                return path;
            }
        }

        if let Ok(virtual_env) = std::env::var("VIRTUAL_ENV") {
            let path = if cfg!(windows) {
                PathBuf::from(&virtual_env).join("Scripts").join("python.exe")
            } else {
                PathBuf::from(&virtual_env).join("bin").join("python")
            };
            if path.exists() {
                return path;
            }
        }

        PathBuf::from("python")
    }

    fn pythonpath_env(&self) -> String {
        let mut paths = vec![self.python_path.to_string_lossy().to_string()];
        if let Ok(existing) = std::env::var("PYTHONPATH") {
            if !existing.trim().is_empty() {
                paths.push(existing);
            }
        }
        let separator = if cfg!(windows) { ";" } else { ":" };
        paths.join(separator)
    }

    fn workspace_dir(&self) -> &Path {
        self.python_path.parent().unwrap_or(self.python_path.as_path())
    }

    fn prepare_script(&self, script_name: &str, script_body: &str) -> anyhow::Result<PathBuf> {
        let script_path = self.cachedir.join(script_name);
        let bootstrap = format!(
            "import sys\nsys.path.insert(0, r\"{}\")\n",
            self.python_path.display()
        );
        std::fs::write(&script_path, format!("{bootstrap}{script_body}"))?;
        Ok(script_path)
    }

    fn run_python_script(
        &self,
        script_file: &Path,
        input_file: &Path,
        output_file: &Path,
    ) -> anyhow::Result<std::process::Output> {
        let output = Command::new(self.python_executable())
            .args([
                script_file.to_str().unwrap(),
                input_file.to_str().unwrap(),
                output_file.to_str().unwrap(),
            ])
            .env("PYTHONPATH", self.pythonpath_env())
            .current_dir(self.workspace_dir())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;

        Ok(output)
    }

    pub fn compress_k(&self, layer: u32, head: u32, data: &[f32]) -> anyhow::Result<Vec<f32>> {
        let input_file = self
            .cachedir
            .join(format!("k_{}_{}_{}.json", layer, head, data.len()));
        let output_file =
            self.cachedir
                .join(format!("k_out_{}_{}_{}.json", layer, head, data.len()));

        let config = serde_json::json!({
            "layer": layer,
            "head": head,
            "data": data,
            "rotation_policy": self.rotation_policy,
            "rotation_seed": self.rotation_seed,
            "triality_view": self.triality_view,
        });
        std::fs::write(&input_file, serde_json::to_string(&config)?)?;

        let script = r#"
import sys
import json

import torch
from turboquant.rotation import rotation_from_policy, block_so8_rotation
from turboquant.research_extension.triality_proxy import apply_triality_proxy_view

with open(sys.argv[1]) as f:
    d = json.load(f)

dim = len(d['data'])
dtype = torch.float32
device = torch.device('cpu')
vector = torch.tensor(d['data'], device=device, dtype=dtype)

if d.get('triality_view'):
    vector = apply_triality_proxy_view(vector, d['triality_view'])

# Setup rotation based on policy
if d.get('rotation_policy') in ('block_so8_learned', 'block_so8_static'):
    rotation = block_so8_rotation(dim=dim, seed=d.get('rotation_seed', 0), device=device, dtype=dtype)
else:
    rotation = rotation_from_policy(dim=dim, seed=d.get('rotation_seed', 0), policy='random_haar', device=device, dtype=dtype)

# Apply rotation and simulate compression
rotated = rotation @ vector
# Simple quantization simulation (in production, uses full TurboQuant)
result = rotated.cpu().tolist()

with open(sys.argv[2], 'w') as f:
    json.dump(result, f)
"#;

        let script_file = self.prepare_script("compress_k.py", script)?;
        let output = self.run_python_script(&script_file, &input_file, &output_file)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Python compress_k failed: {}", stderr);
        }

        let result_json: Vec<f32> = serde_json::from_str(&std::fs::read_to_string(&output_file)?)?;
        Ok(result_json)
    }

    pub fn compress_v(&self, layer: u32, head: u32, data: &[f32]) -> anyhow::Result<Vec<f32>> {
        let input_file = self
            .cachedir
            .join(format!("v_{}_{}_{}.json", layer, head, data.len()));
        let output_file =
            self.cachedir
                .join(format!("v_out_{}_{}_{}.json", layer, head, data.len()));

        let config = serde_json::json!({
            "layer": layer,
            "head": head,
            "data": data,
            "rotation_policy": self.rotation_policy,
            "rotation_seed": self.rotation_seed,
        });
        std::fs::write(&input_file, serde_json::to_string(&config)?)?;

        let script = r#"
import sys
import json
import torch

with open(sys.argv[1]) as f:
    d = json.load(f)

# For V compression, pass through (exact values in paper-key-only mode)
result = d['data']

with open(sys.argv[2], 'w') as f:
    json.dump(result, f)
"#;

        let script_file = self.prepare_script("compress_v.py", script)?;
        let output = self.run_python_script(&script_file, &input_file, &output_file)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Python compress_v failed: {}", stderr);
        }

        let result_json: Vec<f32> = serde_json::from_str(&std::fs::read_to_string(&output_file)?)?;
        Ok(result_json)
    }

    pub fn score_k(
        &self,
        layer: u32,
        head: u32,
        query: &[f32],
        token_start: u32,
        token_end: u32,
    ) -> anyhow::Result<Vec<f32>> {
        let input_file = self.cachedir.join(format!("score_{}_{}.json", layer, head));
        let output_file = self
            .cachedir
            .join(format!("score_out_{}_{}.json", layer, head));

        let config = serde_json::json!({
            "layer": layer,
            "head": head,
            "query": query,
            "token_range": [token_start, token_end],
        });
        std::fs::write(&input_file, serde_json::to_string(&config)?)?;

        let script = r#"
import sys
import json

with open(sys.argv[1]) as f:
    d = json.load(f)

# Score estimation - returns dummy scores for PoC
# In production: uses TurboQuant inner-product estimator
n_tokens = d['token_range'][1] - d['token_range'][0]
scores = [1.0] * n_tokens

with open(sys.argv[2], 'w') as f:
    json.dump(scores, f)
"#;

        let script_file = self.prepare_script("score_k.py", script)?;
        let output = self.run_python_script(&script_file, &input_file, &output_file)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Python score_k failed: {}", stderr);
        }

        let result_json: Vec<f32> = serde_json::from_str(&std::fs::read_to_string(&output_file)?)?;
        Ok(result_json)
    }

    pub fn read_v(
        &self,
        layer: u32,
        head: u32,
        token_start: u32,
        token_end: u32,
    ) -> anyhow::Result<Vec<f32>> {
        let input_file = self
            .cachedir
            .join(format!("read_v_{}_{}.json", layer, head));
        let output_file = self
            .cachedir
            .join(format!("read_v_out_{}_{}.json", layer, head));

        let config = serde_json::json!({
            "layer": layer,
            "head": head,
            "token_range": [token_start, token_end],
        });
        std::fs::write(&input_file, serde_json::to_string(&config)?)?;

        let script = r#"
import sys
import json

with open(sys.argv[1]) as f:
    d = json.load(f)

# Read V values (pass-through for paper-key-only)
n_tokens = d['token_range'][1] - d['token_range'][0]
values = [0.0] * n_tokens * 128  # Simplified

with open(sys.argv[2], 'w') as f:
    json.dump(values, f)
"#;

        let script_file = self.prepare_script("read_v.py", script)?;
        let output = self.run_python_script(&script_file, &input_file, &output_file)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Python read_v failed: {}", stderr);
        }

        let result_json: Vec<f32> = serde_json::from_str(&std::fs::read_to_string(&output_file)?)?;
        Ok(result_json)
    }
}

fn resolve_turboquant_python_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("HYPURA_TURBOQUANT_PYTHON_PATH") {
        let path = PathBuf::from(explicit);
        if path.exists() {
            return path;
        }
    }

    if let Ok(existing) = std::env::var("PYTHONPATH") {
        let separator = if cfg!(windows) { ';' } else { ':' };
        if let Some(path) = existing
            .split(separator)
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(PathBuf::from)
            .find(|path| path.exists())
        {
            return path;
        }
    }

    let manifest_candidate = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Turboquant-CUDA");
    if manifest_candidate.exists() {
        return manifest_candidate;
    }

    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join(TURBOQUANT_PYTHON_PATH)
}

fn parse_rotation_policy(policy: Option<&str>, triality_view: Option<&str>) -> (String, Option<String>) {
    if let Some(view) = triality_view {
        let normalized_view = match view {
            "vector" => Some("vector".to_string()),
            "spinor_plus_proxy" => Some("spinor_plus_proxy".to_string()),
            "spinor_minus_proxy" => Some("spinor_minus_proxy".to_string()),
            "none" => None,
            _ => None,
        };
        if let Some(view) = normalized_view {
            let rotation = match policy {
                Some("block_so8_static") => "block_so8_static",
                Some("random_haar") => "random_haar",
                _ => "block_so8_learned",
            };
            return (rotation.to_string(), Some(view));
        }
    }
    match policy {
        Some("block_so8_learned") => ("block_so8_learned".to_string(), None),
        Some("block_so8_static") => ("block_so8_static".to_string(), None),
        Some("triality_vector") => ("block_so8_learned".to_string(), Some("vector".to_string())),
        Some("triality_spinor_plus") => (
            "block_so8_learned".to_string(),
            Some("spinor_plus_proxy".to_string()),
        ),
        Some("triality_spinor_minus") => (
            "block_so8_learned".to_string(),
            Some("spinor_minus_proxy".to_string()),
        ),
        _ => ("random_haar".to_string(), None),
    }
}
