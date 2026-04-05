//! Hypura on-disk data location: `HYPURA_DATA_DIR`, or `HYPURA_CACHE_ROOTS` (SSD-picked), or default.

use std::path::{Path, PathBuf};

use anyhow::Context;
use sysinfo::DiskKind;
use sysinfo::Disks;

const ENV_DATA_DIR: &str = "HYPURA_DATA_DIR";
const ENV_CACHE_ROOTS: &str = "HYPURA_CACHE_ROOTS";
const ENV_ALLOW_HDD_CACHE: &str = "HYPURA_ALLOW_HDD_CACHE";

/// Minimum free bytes required on a cache root candidate (512 MiB).
const MIN_FREE_BYTES: u64 = 512 * 1024 * 1024;

fn env_flag_true(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

fn allow_hdd_cache() -> bool {
    env_flag_true(ENV_ALLOW_HDD_CACHE)
}

/// Resolved Hypura data directory (profile JSON, etc.).
pub fn resolve_data_dir() -> anyhow::Result<PathBuf> {
    if let Ok(s) = std::env::var(ENV_DATA_DIR) {
        let t = s.trim();
        if !t.is_empty() {
            let p = PathBuf::from(t);
            std::fs::create_dir_all(&p).with_context(|| format!("create {ENV_DATA_DIR}"))?;
            return Ok(p);
        }
    }

    if let Ok(roots) = std::env::var(ENV_CACHE_ROOTS) {
        let base = pick_cache_root(&roots, allow_hdd_cache())?;
        let hypura_home = base.join("Hypura");
        std::fs::create_dir_all(&hypura_home)
            .with_context(|| format!("create cache data dir {}", hypura_home.display()))?;
        return Ok(hypura_home);
    }

    Ok(default_platform_data_dir())
}

pub fn default_platform_data_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA")
            .unwrap_or_else(|_| "C:\\Users\\Default\\AppData\\Roaming".into());
        PathBuf::from(appdata).join("Hypura")
    }
    #[cfg(not(target_os = "windows"))]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".hypura")
    }
}

fn split_roots(raw: &str) -> Vec<PathBuf> {
    raw.split(&[';', ','][..])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn pick_cache_root(raw: &str, allow_hdd: bool) -> anyhow::Result<PathBuf> {
    let roots = split_roots(raw);
    anyhow::ensure!(!roots.is_empty(), "{ENV_CACHE_ROOTS} is empty");

    let disks = Disks::new_with_refreshed_list();

    let mut candidates: Vec<(PathBuf, u64)> = Vec::new();
    for root in roots {
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create cache root {}", root.display()))?;
        if !root.is_dir() {
            continue;
        }

        if !allow_hdd {
            if let Some(kind) = disk_kind_for_path(&root, &disks) {
                if kind == DiskKind::HDD {
                    tracing::warn!(
                        "skipping HDD cache root {} (set {}=1 or CLI --allow-hdd-cache)",
                        root.display(),
                        ENV_ALLOW_HDD_CACHE
                    );
                    continue;
                }
            }
        }

        let free = fs_free_bytes(&root).unwrap_or(0);
        if free < MIN_FREE_BYTES {
            tracing::warn!(
                "skipping cache root {}: free space {} < minimum {}",
                root.display(),
                free,
                MIN_FREE_BYTES
            );
            continue;
        }
        candidates.push((root, free));
    }

    candidates
        .into_iter()
        .max_by_key(|(_, f)| *f)
        .map(|(p, _)| p)
        .context(format!(
            "no usable SSD cache root in {ENV_CACHE_ROOTS} (need free≥{MIN_FREE_BYTES} bytes; HDD excluded unless allowed)"
        ))
}

fn fs_free_bytes(path: &Path) -> Option<u64> {
    let disks = Disks::new_with_refreshed_list();
    let path_str = dunce::canonicalize(path)
        .ok()?
        .to_string_lossy()
        .to_string();
    let path_lower = path_str.to_lowercase();

    let mut best_free: Option<u64> = None;
    let mut best_len = 0usize;
    for disk in disks.list() {
        let mount = disk.mount_point().to_string_lossy().to_lowercase();
        if path_lower.starts_with(&mount) && mount.len() >= best_len {
            best_len = mount.len();
            best_free = Some(disk.available_space());
        }
    }
    best_free
}

fn disk_kind_for_path(path: &Path, disks: &Disks) -> Option<DiskKind> {
    let path_str = dunce::canonicalize(path)
        .ok()?
        .to_string_lossy()
        .to_string();
    let path_lower = path_str.to_lowercase();

    let mut kind: Option<DiskKind> = None;
    let mut best_len = 0usize;
    for disk in disks.list() {
        let mount = disk.mount_point().to_string_lossy().to_lowercase();
        if path_lower.starts_with(&mount) && mount.len() >= best_len {
            best_len = mount.len();
            kind = Some(disk.kind());
        }
    }
    kind
}
