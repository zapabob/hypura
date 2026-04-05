//! Resolve GGUF model scan directories from CLI, env, and initial model path.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

const ENV_MODEL_DIRS: &str = "HYPURA_MODEL_DIRS";

/// Split user-provided path lists (`;` or `,`).
pub fn split_path_list(raw: &str) -> Vec<String> {
    raw.split(&[';', ','][..])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn dedupe_key(p: &Path) -> String {
    if p.is_dir() {
        if let Ok(c) = dunce::canonicalize(p) {
            return path_key(&c);
        }
    }
    path_key(p)
}

fn path_key(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    #[cfg(windows)]
    {
        s.to_lowercase()
    }
    #[cfg(not(windows))]
    {
        s
    }
}

fn push_unique_path(ordered: &mut Vec<PathBuf>, seen: &mut HashSet<String>, p: PathBuf) {
    let key = dedupe_key(&p);
    if seen.insert(key) {
        ordered.push(p);
    }
}

/// Merge model directory roots: `HYPURA_MODEL_DIRS`, then CLI string (may contain `;`/`,`),
/// then default to parent of `model_path`. Deduplicate by normalized path.
pub fn resolve_model_directories(model_path: &Path, cli_model_dir: Option<&str>) -> Vec<PathBuf> {
    let mut ordered: Vec<PathBuf> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    if let Ok(env) = std::env::var(ENV_MODEL_DIRS) {
        for s in split_path_list(&env) {
            push_unique_path(&mut ordered, &mut seen, PathBuf::from(s));
        }
    }

    if let Some(cli) = cli_model_dir {
        let cli = cli.trim();
        if !cli.is_empty() {
            for s in split_path_list(cli) {
                push_unique_path(&mut ordered, &mut seen, PathBuf::from(s));
            }
        }
    }

    if ordered.is_empty() {
        if let Some(parent) = model_path.parent() {
            push_unique_path(&mut ordered, &mut seen, parent.to_path_buf());
        } else {
            push_unique_path(&mut ordered, &mut seen, PathBuf::from("."));
        }
    }

    ordered
}

/// True if `path` is a `.gguf` file under one of `roots` (after canonicalization when possible).
pub fn is_model_path_allowed(path: &Path, roots: &[PathBuf]) -> bool {
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

    for root in roots {
        if !root.is_dir() {
            continue;
        }
        let root_canon = match dunce::canonicalize(root) {
            Ok(r) => r,
            Err(_) => root.clone(),
        };
        if canonical.starts_with(&root_canon) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn split_semicolon_comma() {
        let v = split_path_list("a;b,c ");
        assert_eq!(v, vec!["a", "b", "c"]);
    }

    #[test]
    fn default_uses_parent() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf = tmp.path().join("m.gguf");
        fs::write(&gguf, b"x").unwrap();
        let dirs = resolve_model_directories(&gguf, None);
        assert_eq!(dirs.len(), 1);
        assert_eq!(dirs[0], tmp.path());
    }
}
