use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFileSnapshot {
    canonical_path: PathBuf,
    size: u64,
    modified: Option<SystemTime>,
    file_id: Option<(u64, u64)>,
    sha256: String,
}

impl ModelFileSnapshot {
    pub fn canonical_path(&self) -> &Path {
        &self.canonical_path
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Debug)]
pub struct GuardedModelFile {
    file: fs::File,
    initial: ModelFileSnapshot,
}

impl GuardedModelFile {
    pub fn acquire(path: &Path) -> anyhow::Result<Self> {
        let canonical_path = fs::canonicalize(path)?;
        let file = open_read_guard(&canonical_path)?;
        let initial = snapshot_from_handle(file.try_clone()?, canonical_path)?;
        Ok(Self { file, initial })
    }

    pub fn canonical_path(&self) -> &Path {
        self.initial.canonical_path()
    }

    pub fn initial_snapshot(&self) -> &ModelFileSnapshot {
        &self.initial
    }

    pub fn verify_unchanged(&self) -> anyhow::Result<()> {
        let final_handle =
            snapshot_from_handle(self.file.try_clone()?, self.initial.canonical_path.clone())?;
        let final_path = snapshot_path(&self.initial.canonical_path)?;
        ensure_unchanged(&self.initial, &final_handle)?;
        ensure_unchanged(&self.initial, &final_path)
    }
}

pub fn open_read_guard(path: &Path) -> anyhow::Result<fs::File> {
    open_model_file_guard(path)
}

pub fn snapshot_path(path: &Path) -> anyhow::Result<ModelFileSnapshot> {
    let canonical_path = fs::canonicalize(path)?;
    snapshot_from_handle(fs::File::open(&canonical_path)?, canonical_path)
}

pub fn ensure_unchanged(
    before: &ModelFileSnapshot,
    after: &ModelFileSnapshot,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        before == after,
        "model file changed while resolving, hashing, or loading; operation refused"
    );
    Ok(())
}

fn snapshot_from_handle(
    mut file: fs::File,
    canonical_path: PathBuf,
) -> anyhow::Result<ModelFileSnapshot> {
    file.seek(SeekFrom::Start(0))?;
    let metadata = file.metadata()?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(ModelFileSnapshot {
        canonical_path,
        size: metadata.len(),
        modified: metadata.modified().ok(),
        file_id: platform_file_id(&metadata),
        sha256: format!("{:x}", digest.finalize()),
    })
}

#[cfg(windows)]
fn platform_file_id(_metadata: &fs::Metadata) -> Option<(u64, u64)> {
    None
}

#[cfg(unix)]
fn platform_file_id(metadata: &fs::Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;

    Some((metadata.dev(), metadata.ino()))
}

#[cfg(not(any(windows, unix)))]
fn platform_file_id(_metadata: &fs::Metadata) -> Option<(u64, u64)> {
    None
}

#[cfg(windows)]
fn open_model_file_guard(path: &Path) -> anyhow::Result<fs::File> {
    use std::os::windows::fs::OpenOptionsExt;

    Ok(fs::OpenOptions::new()
        .read(true)
        .share_mode(0x0000_0001)
        .open(path)?)
}

#[cfg(not(windows))]
fn open_model_file_guard(path: &Path) -> anyhow::Result<fs::File> {
    Ok(fs::File::open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarded_identity_preserves_hash_across_unchanged_load_window() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"model-content").unwrap();
        let guard = GuardedModelFile::acquire(&path).unwrap();
        let initial_hash = guard.initial_snapshot().sha256().to_string();

        guard.verify_unchanged().unwrap();

        assert_eq!(guard.initial_snapshot().sha256(), initial_hash);
    }

    #[test]
    fn rejects_same_size_content_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        std::fs::write(&path, b"abc").unwrap();
        let before = snapshot_path(&path).unwrap();
        assert!(ensure_unchanged(&before, &before).is_ok());

        std::fs::write(&path, b"xyz").unwrap();
        let after = snapshot_path(&path).unwrap();
        assert_eq!(before.size(), after.size());
        assert_ne!(before.sha256(), after.sha256());
        assert!(ensure_unchanged(&before, &after).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_same_content_path_replacement_by_file_identity() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let displaced = directory.path().join("model.old.gguf");
        std::fs::write(&path, b"identical-content").unwrap();
        let guard = GuardedModelFile::acquire(&path).unwrap();

        std::fs::rename(&path, &displaced).unwrap();
        std::fs::write(&path, b"identical-content").unwrap();
        let replacement = snapshot_path(&path).unwrap();
        let mut same_metadata_replacement = replacement.clone();
        same_metadata_replacement.modified = guard.initial_snapshot().modified;

        assert_eq!(guard.initial_snapshot().sha256(), replacement.sha256());
        assert_eq!(
            guard.initial_snapshot().size,
            same_metadata_replacement.size
        );
        assert_eq!(
            guard.initial_snapshot().canonical_path,
            same_metadata_replacement.canonical_path
        );
        assert_ne!(
            guard.initial_snapshot().file_id,
            same_metadata_replacement.file_id
        );
        assert!(ensure_unchanged(guard.initial_snapshot(), &same_metadata_replacement).is_err());
        assert!(guard.verify_unchanged().is_err());
    }
}
