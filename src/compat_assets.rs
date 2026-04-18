use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KoboldcppAssetManifest {
    pub manifest_version: String,
    pub release_tag: String,
    #[serde(default)]
    pub assets: Vec<KoboldcppAssetEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KoboldcppAssetEntry {
    pub id: String,
    pub kind: String,
    pub target_rel_path: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub download_url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub bundled_source_rel_path: Option<String>,
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssetBootstrapReport {
    pub asset_root: PathBuf,
    pub ready: HashMap<String, PathBuf>,
    pub pending: Vec<KoboldcppAssetEntry>,
    pub downloaded: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TreeAssetMarker {
    source: String,
    file_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct HuggingFaceTreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<HuggingFaceLfs>,
}

#[derive(Debug, Clone, Deserialize)]
struct HuggingFaceLfs {
    oid: String,
}

pub fn default_asset_root() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            let trimmed = local_app_data.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed)
                    .join("Hypura")
                    .join("koboldcpp")
                    .join("assets");
            }
        }
        if let Ok(user_profile) = std::env::var("USERPROFILE") {
            let trimmed = user_profile.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed)
                    .join("AppData")
                    .join("Local")
                    .join("Hypura")
                    .join("koboldcpp")
                    .join("assets");
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(home) = std::env::var("HOME") {
            let trimmed = home.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed)
                    .join(".local")
                    .join("share")
                    .join("hypura")
                    .join("koboldcpp")
                    .join("assets");
            }
        }
    }

    PathBuf::from(".hypura")
        .join("koboldcpp")
        .join("assets")
}

pub fn discover_asset_manifest_path() -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(exe_dir) = current_exe.parent() {
            candidates.push(exe_dir.join("koboldcpp-assets.json"));
            candidates.push(exe_dir.join("resources").join("koboldcpp-assets.json"));
        }
    }

    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("docs")
            .join("compat")
            .join("koboldcpp-assets.json"),
    );
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("hypura-desktop")
            .join("src-tauri")
            .join("resources")
            .join("koboldcpp-assets.json"),
    );

    candidates.into_iter().find(|path| path.exists())
}

pub fn load_asset_manifest(path: &Path) -> Result<KoboldcppAssetManifest> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading KoboldCpp asset manifest {}", path.display()))?;
    let manifest: KoboldcppAssetManifest = serde_json::from_str(&text)
        .with_context(|| format!("parsing KoboldCpp asset manifest {}", path.display()))?;
    Ok(manifest)
}

pub fn bootstrap_assets(
    asset_root: &Path,
    manifest_path: Option<&Path>,
) -> Result<AssetBootstrapReport> {
    fs::create_dir_all(asset_root)
        .with_context(|| format!("creating asset root {}", asset_root.display()))?;
    let mut report = AssetBootstrapReport {
        asset_root: asset_root.to_path_buf(),
        ..AssetBootstrapReport::default()
    };

    let Some(manifest_path) = manifest_path else {
        return Ok(report);
    };
    let manifest = load_asset_manifest(manifest_path)?;
    let manifest_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    for entry in manifest.assets {
        let target = asset_root.join(&entry.target_rel_path);
        if verify_asset_entry(&entry, &target)? {
            report.ready.insert(entry.id.clone(), target);
            continue;
        }

        if let Some(bundled_rel) = entry.bundled_source_rel_path.as_deref() {
            let bundled_source = manifest_dir.join(bundled_rel);
            if bundled_source.exists() {
                materialize_asset_entry(&entry, &bundled_source, &target)?;
                report.downloaded.push(entry.id.clone());
                report.ready.insert(entry.id.clone(), target);
                continue;
            }
        }

        report.pending.push(entry);
    }

    Ok(report)
}

pub fn materialize_pending_asset_entry(
    asset_root: &Path,
    entry: &KoboldcppAssetEntry,
) -> Result<PathBuf> {
    let target = asset_root.join(&entry.target_rel_path);
    if is_tree_asset_entry(entry) {
        materialize_tree_asset_entry(entry, &target)?;
        return Ok(target);
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating asset directory {}", parent.display()))?;
    }

    let Some(download_url) = entry.download_url.as_deref() else {
        anyhow::bail!("asset {} has no download_url", entry.id);
    };
    download_file_to_target(
        download_url,
        &target,
        entry.size_bytes,
        entry.sha256.as_deref(),
        &format!("asset {}", entry.id),
    )?;
    if !verify_asset_entry(entry, &target)? {
        anyhow::bail!("downloaded asset {} failed verification", entry.id);
    }
    Ok(target)
}

fn materialize_asset_entry(entry: &KoboldcppAssetEntry, source: &Path, target: &Path) -> Result<()> {
    if is_tree_asset_entry(entry) {
        if !source.is_dir() {
            anyhow::bail!(
                "bundled tree asset {} expected directory source, got {}",
                entry.id,
                source.display()
            );
        }
        copy_dir_recursive(source, target)?;
        write_tree_asset_marker(entry, target, 0)?;
        anyhow::ensure!(
            verify_asset_entry(entry, target)?,
            "bundled asset {} failed verification",
            entry.id
        );
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating asset directory {}", parent.display()))?;
    }
    fs::copy(source, target).with_context(|| {
        format!(
            "copying bundled asset {} to {}",
            source.display(),
            target.display()
        )
    })?;
    anyhow::ensure!(
        verify_asset_entry(entry, target)?,
        "bundled asset {} failed verification",
        entry.id
    );
    Ok(())
}

fn verify_asset_entry(entry: &KoboldcppAssetEntry, target: &Path) -> Result<bool> {
    if is_tree_asset_entry(entry) {
        return verify_tree_asset_entry(entry, target);
    }
    if !target.exists() {
        return Ok(false);
    }
    if let Some(expected_size) = entry.size_bytes {
        let actual_size = fs::metadata(target)?.len();
        if actual_size != expected_size {
            return Ok(false);
        }
    }
    if let Some(expected_sha) = entry.sha256.as_deref() {
        let actual_sha = sha256_hex(target)?;
        if !actual_sha.eq_ignore_ascii_case(expected_sha) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn is_tree_asset_entry(entry: &KoboldcppAssetEntry) -> bool {
    matches!(
        entry.kind.as_str(),
        "tree" | "data_tree" | "model_tree" | "voice_tree" | "runtime_tree"
    )
}

fn tree_asset_marker_path(target: &Path) -> PathBuf {
    target.join(".hypura-asset-tree.json")
}

fn verify_tree_asset_entry(entry: &KoboldcppAssetEntry, target: &Path) -> Result<bool> {
    if !target.is_dir() {
        return Ok(false);
    }
    let marker_path = tree_asset_marker_path(target);
    if !marker_path.is_file() {
        return Ok(false);
    }
    let marker: TreeAssetMarker = serde_json::from_slice(
        &fs::read(&marker_path)
            .with_context(|| format!("reading tree asset marker {}", marker_path.display()))?,
    )
    .with_context(|| format!("parsing tree asset marker {}", marker_path.display()))?;
    Ok(marker.source == entry.download_url.clone().unwrap_or_default())
}

fn write_tree_asset_marker(entry: &KoboldcppAssetEntry, target: &Path, file_count: usize) -> Result<()> {
    fs::create_dir_all(target)
        .with_context(|| format!("creating tree asset directory {}", target.display()))?;
    let marker = TreeAssetMarker {
        source: entry.download_url.clone().unwrap_or_default(),
        file_count,
    };
    let bytes = serde_json::to_vec_pretty(&marker)?;
    fs::write(tree_asset_marker_path(target), bytes)
        .with_context(|| format!("writing tree asset marker {}", target.display()))?;
    Ok(())
}

fn materialize_tree_asset_entry(entry: &KoboldcppAssetEntry, target: &Path) -> Result<()> {
    let Some(download_url) = entry.download_url.as_deref() else {
        anyhow::bail!("tree asset {} has no download_url", entry.id);
    };
    let spec = parse_huggingface_tree_url(download_url)?;
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .user_agent("Hypura-KoboldCpp-AssetBootstrap/0.8.0")
        .build()?;
    let response = client
        .get(download_url)
        .send()
        .with_context(|| format!("downloading tree asset {} from {}", entry.id, download_url))?;
    anyhow::ensure!(
        response.status().is_success(),
        "tree asset {} download failed with status {}",
        entry.id,
        response.status()
    );
    let items: Vec<HuggingFaceTreeEntry> = response
        .json()
        .with_context(|| format!("parsing tree asset listing {}", download_url))?;
    let files: Vec<_> = items
        .into_iter()
        .filter(|item| item.kind == "file")
        .collect();
    anyhow::ensure!(
        !files.is_empty(),
        "tree asset {} listing returned no files",
        entry.id
    );
    if target.exists() {
        fs::remove_dir_all(target)
            .with_context(|| format!("clearing stale tree asset {}", target.display()))?;
    }
    fs::create_dir_all(target)
        .with_context(|| format!("creating tree asset directory {}", target.display()))?;
    for file in &files {
        let relative = file
            .path
            .strip_prefix(&spec.path_prefix)
            .map(|value| value.trim_start_matches('/'))
            .unwrap_or(file.path.as_str());
        anyhow::ensure!(
            !relative.is_empty(),
            "tree asset {} produced empty relative path for {}",
            entry.id,
            file.path
        );
        let direct_url = spec.resolve_file_url(&file.path);
        let destination = target.join(relative.replace('/', std::path::MAIN_SEPARATOR_STR));
        download_file_to_target(
            &direct_url,
            &destination,
            file.size,
            file.lfs.as_ref().map(|value| value.oid.as_str()),
            &format!("tree asset {} file {}", entry.id, file.path),
        )?;
    }
    write_tree_asset_marker(entry, target, files.len())?;
    Ok(())
}

fn download_file_to_target(
    download_url: &str,
    target: &Path,
    expected_size: Option<u64>,
    expected_sha: Option<&str>,
    label: &str,
) -> Result<()> {
    let response = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .user_agent("Hypura-KoboldCpp-AssetBootstrap/0.8.0")
        .build()?
        .get(download_url)
        .send()
        .with_context(|| format!("downloading {label} from {download_url}"))?;
    anyhow::ensure!(
        response.status().is_success(),
        "{label} download failed with status {}",
        response.status()
    );
    let bytes = response.bytes()?;
    if let Some(expected_size) = expected_size {
        anyhow::ensure!(
            bytes.len() as u64 == expected_size,
            "{label} size mismatch: expected {}, got {}",
            expected_size,
            bytes.len()
        );
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(target, &bytes).with_context(|| format!("writing {label} to {}", target.display()))?;
    if let Some(expected_sha) = expected_sha {
        let actual_sha = sha256_hex(target)?;
        anyhow::ensure!(
            actual_sha.eq_ignore_ascii_case(expected_sha),
            "{label} sha256 mismatch: expected {expected_sha}, got {actual_sha}"
        );
    }
    Ok(())
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target)
        .with_context(|| format!("creating recursive copy target {}", target.display()))?;
    for entry in fs::read_dir(source)
        .with_context(|| format!("reading directory {}", source.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &target_path)?;
        } else {
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&source_path, &target_path).with_context(|| {
                format!(
                    "copying bundled tree asset {} to {}",
                    source_path.display(),
                    target_path.display()
                )
            })?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HuggingFaceTreeSpec {
    repo: String,
    revision: String,
    path_prefix: String,
}

impl HuggingFaceTreeSpec {
    fn resolve_file_url(&self, file_path: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, file_path
        )
    }
}

fn parse_huggingface_tree_url(download_url: &str) -> Result<HuggingFaceTreeSpec> {
    let parsed = reqwest::Url::parse(download_url)
        .with_context(|| format!("parsing Hugging Face tree URL {download_url}"))?;
    anyhow::ensure!(
        parsed.domain() == Some("huggingface.co"),
        "unsupported tree asset host in {download_url}"
    );
    let segments: Vec<_> = parsed
        .path_segments()
        .map(|value| value.collect::<Vec<_>>())
        .unwrap_or_default();
    anyhow::ensure!(
        segments.len() >= 6 && segments[0] == "api" && segments[1] == "models",
        "unsupported Hugging Face tree URL shape: {download_url}"
    );
    let Some(tree_index) = segments.iter().position(|segment| *segment == "tree") else {
        anyhow::bail!("unsupported Hugging Face tree URL shape: {download_url}");
    };
    anyhow::ensure!(
        tree_index >= 3 && tree_index + 1 < segments.len(),
        "unsupported Hugging Face tree URL shape: {download_url}"
    );
    let repo = segments[2..tree_index].join("/");
    let revision = segments[tree_index + 1].to_string();
    let path_prefix = if segments.len() > tree_index + 2 {
        segments[tree_index + 2..].join("/")
    } else {
        String::new()
    };
    Ok(HuggingFaceTreeSpec {
        repo,
        revision,
        path_prefix,
    })
}

fn sha256_hex(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("opening asset for sha256 {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_asset_root_looks_hypura_scoped() {
        let root = default_asset_root();
        let rendered = root.to_string_lossy().to_ascii_lowercase();
        assert!(rendered.contains("hypura"));
        assert!(rendered.contains("koboldcpp"));
    }

    #[test]
    fn bootstrap_marks_missing_downloads_as_pending() {
        let temp = tempfile::tempdir().unwrap();
        let manifest_path = temp.path().join("koboldcpp-assets.json");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&KoboldcppAssetManifest {
                manifest_version: "1".to_string(),
                release_tag: "v1.111.2".to_string(),
                assets: vec![KoboldcppAssetEntry {
                    id: "embeddings_model".to_string(),
                    kind: "model".to_string(),
                    target_rel_path: "models/embeddings.gguf".to_string(),
                    version: Some("pending".to_string()),
                    download_url: Some("https://example.invalid/embeddings.gguf".to_string()),
                    sha256: None,
                    size_bytes: None,
                    bundled_source_rel_path: None,
                    optional: true,
                }],
            })
            .unwrap(),
        )
        .unwrap();

        let report = bootstrap_assets(&temp.path().join("assets"), Some(&manifest_path)).unwrap();
        assert!(report.ready.is_empty());
        assert_eq!(report.pending.len(), 1);
        assert_eq!(report.pending[0].id, "embeddings_model");
    }

    #[test]
    fn parse_huggingface_tree_url_extracts_repo_revision_and_prefix() {
        let spec = parse_huggingface_tree_url(
            "https://huggingface.co/api/models/csukuangfj/kitten-nano-en-v0_2-fp16/tree/main/espeak-ng-data?recursive=1",
        )
        .unwrap();
        assert_eq!(spec.repo, "csukuangfj/kitten-nano-en-v0_2-fp16");
        assert_eq!(spec.revision, "main");
        assert_eq!(spec.path_prefix, "espeak-ng-data");
    }
}
