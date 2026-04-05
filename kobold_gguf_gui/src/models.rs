//! GGUF listing and path allowlist (Hypura-compatible API).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::settings::KoboldGuiSettings;

#[derive(Serialize, Deserialize)]
pub struct AvailableModelItem {
    pub name: String,
    pub path: String,
    pub selected: bool,
}

#[derive(Serialize, Deserialize)]
pub struct AvailableModelsResponse {
    pub models: Vec<AvailableModelItem>,
    pub active_model_path: String,
}

pub fn model_scan_roots(settings: &KoboldGuiSettings) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for part in settings
        .model_scan_dirs
        .split(&[';', ','][..])
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let p = PathBuf::from(part);
        let key = p.to_string_lossy().to_lowercase();
        if seen.insert(key) {
            out.push(p);
        }
    }

    if out.is_empty() {
        let g = settings.gguf_path.trim();
        if !g.is_empty() {
            if let Some(parent) = Path::new(g).parent() {
                let p = parent.to_path_buf();
                let key = p.to_string_lossy().to_lowercase();
                if seen.insert(key) {
                    out.push(p);
                }
            }
        }
    }

    out
}

pub fn collect_gguf_models(settings: &KoboldGuiSettings, active_path: &str) -> anyhow::Result<Vec<AvailableModelItem>> {
    let roots = model_scan_roots(settings);
    let mut by_path: HashMap<String, AvailableModelItem> = HashMap::new();

    for dir in roots {
        if !dir.is_dir() {
            continue;
        }
        let rd = match fs::read_dir(&dir) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("kobold_gguf_gui: model dir unreadable {}: {e}", dir.display());
                continue;
            }
        };
        for entry in rd {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let is_gguf = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);
            if !is_gguf {
                continue;
            }
            let full = path.to_string_lossy().to_string();
            if by_path.contains_key(&full) {
                continue;
            }
            let name = path
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| full.clone());
            by_path.insert(
                full.clone(),
                AvailableModelItem {
                    name,
                    selected: paths_equal_ci(&full, active_path),
                    path: full,
                },
            );
        }
    }

    let mut models: Vec<AvailableModelItem> = by_path.into_values().collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

fn paths_equal_ci(a: &str, b: &str) -> bool {
    if cfg!(windows) {
        a.eq_ignore_ascii_case(b)
    } else {
        a == b
    }
}

pub fn is_model_path_allowed(path: &Path, settings: &KoboldGuiSettings) -> bool {
    if !path.is_file() {
        return false;
    }
    let ext_ok = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false);
    if !ext_ok {
        return false;
    }

    let Ok(canonical) = dunce::canonicalize(path) else {
        return false;
    };

    for root in model_scan_roots(settings) {
        if !root.is_dir() {
            continue;
        }
        let root_canon = dunce::canonicalize(&root).unwrap_or(root);
        if canonical.starts_with(&root_canon) {
            return true;
        }
    }
    false
}
