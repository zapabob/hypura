use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::io::aligned_buffer::AlignedBuffer;

/// A request to read a region from disk.
#[derive(Debug, Clone)]
pub struct ReadRequest {
    pub offset: u64,
    pub length: usize,
    pub tag: String,
}

/// Statistics for a completed read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadStats {
    pub bytes_read: u64,
    pub duration_us: u64,
    pub throughput_mbps: f64,
}

/// A completed read with its data.
pub struct ReadResult {
    pub request: ReadRequest,
    pub data: AlignedBuffer,
    pub stats: ReadStats,
}

/// Double-buffered async disk reader with F_NOCACHE.
///
/// Spawns a background task that reads into alternating aligned buffers.
/// Consumer receives completed reads via an mpsc channel.
pub struct NvmePrefetcher {
    _file_path: PathBuf,
    request_tx: mpsc::UnboundedSender<ReadRequest>,
    result_rx: mpsc::UnboundedReceiver<ReadResult>,
    _handle: tokio::task::JoinHandle<()>,
}

impl NvmePrefetcher {
    /// Open a file for prefetching with F_NOCACHE.
    /// `buffer_size` is the size of each double-buffer (e.g., 4 MiB).
    pub fn open(path: impl AsRef<Path>, buffer_size: usize) -> std::io::Result<Self> {
        let file_path = path.as_ref().to_path_buf();
        let (request_tx, mut request_rx) = mpsc::unbounded_channel::<ReadRequest>();
        let (result_tx, result_rx) = mpsc::unbounded_channel::<ReadResult>();

        let path_clone = file_path.clone();

        let handle = tokio::task::spawn_blocking(move || {
            let file = match std::fs::File::open(&path_clone) {
                Ok(f) => f,
                Err(e) => {
                    tracing::error!("NvmePrefetcher: failed to open {}: {e}", path_clone.display());
                    return;
                }
            };

            let fd = file.as_raw_fd();

            // Bypass filesystem cache
            unsafe {
                libc::fcntl(fd, libc::F_NOCACHE, 1);
            }

            while let Some(req) = request_rx.blocking_recv() {
                let mut buf = match AlignedBuffer::new(buffer_size.max(req.length), 4096) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!("NvmePrefetcher: alloc failed: {e}");
                        continue;
                    }
                };

                let start = Instant::now();
                let mut total_read = 0usize;
                let to_read = req.length;

                while total_read < to_read {
                    let n = unsafe {
                        libc::pread(
                            fd,
                            buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                            to_read - total_read,
                            (req.offset + total_read as u64) as libc::off_t,
                        )
                    };
                    if n <= 0 {
                        break;
                    }
                    total_read += n as usize;
                }

                let duration = start.elapsed();
                let duration_us = duration.as_micros() as u64;
                let throughput_mbps = if duration_us > 0 {
                    total_read as f64 / duration.as_secs_f64() / 1e6
                } else {
                    0.0
                };

                let result = ReadResult {
                    request: req,
                    data: buf,
                    stats: ReadStats {
                        bytes_read: total_read as u64,
                        duration_us,
                        throughput_mbps,
                    },
                };

                if result_tx.send(result).is_err() {
                    break; // Consumer dropped
                }
            }
        });

        Ok(Self {
            _file_path: file_path,
            request_tx,
            result_rx,
            _handle: handle,
        })
    }

    /// Enqueue a read request. Returns immediately.
    pub fn submit(&self, request: ReadRequest) -> anyhow::Result<()> {
        self.request_tx
            .send(request)
            .map_err(|_| anyhow::anyhow!("Prefetcher background thread has stopped"))
    }

    /// Receive the next completed read. Blocks until available.
    pub async fn recv(&mut self) -> Option<ReadResult> {
        self.result_rx.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn test_prefetch_read() {
        // Create a temp file with known data
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_data.bin");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            let data = vec![0xA5u8; 1 << 20]; // 1 MiB
            f.write_all(&data).unwrap();
            f.sync_all().unwrap();
        }

        let mut prefetcher = NvmePrefetcher::open(&path, 1 << 20).unwrap();

        prefetcher
            .submit(ReadRequest {
                offset: 0,
                length: 4096,
                tag: "test".into(),
            })
            .unwrap();

        let result = prefetcher.recv().await.unwrap();
        assert_eq!(result.stats.bytes_read, 4096);
        assert_eq!(result.data[0], 0xA5);
        assert!(result.stats.throughput_mbps > 0.0);
    }
}
