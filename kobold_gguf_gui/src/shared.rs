//! Process + settings shared between egui and the HTTP proxy.

use std::process::{Child, Command};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::settings::KoboldGuiSettings;

pub struct SharedState {
    pub settings: RwLock<KoboldGuiSettings>,
    pub llama_child: Mutex<Option<Child>>,
}

impl SharedState {
    pub fn new(settings: KoboldGuiSettings) -> Arc<Self> {
        Arc::new(Self {
            settings: RwLock::new(settings),
            llama_child: Mutex::new(None),
        })
    }

    /// Kill llama-server if running; clear handle.
    pub fn stop_llama(&self) {
        if let Some(mut c) = self.llama_child.lock().unwrap().take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }

    pub fn wait_llama_health(port: u16) -> anyhow::Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()?;
        let url = format!("http://127.0.0.1:{port}/health");
        for i in 0..120 {
            match client.get(&url).send() {
                Ok(r) if r.status().is_success() => return Ok(()),
                _ => {
                    if i % 10 == 0 {
                        std::thread::sleep(Duration::from_millis(200));
                    } else {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        }
        anyhow::bail!("llama-server did not respond on /health (port {port})");
    }

    /// Spawn llama-server from current settings (caller must have stopped any prior child).
    pub fn spawn_llama(&self) -> anyhow::Result<()> {
        let s = self.settings.read().map_err(|e| anyhow::anyhow!("{e}"))?;
        if s.gguf_path.trim().is_empty() || s.llama_server_exe.trim().is_empty() {
            anyhow::bail!("Set GGUF path and llama-server executable.");
        }
        if !std::path::PathBuf::from(s.gguf_path.trim()).exists() {
            anyhow::bail!("GGUF file not found.");
        }
        if !std::path::PathBuf::from(s.llama_server_exe.trim()).exists() {
            anyhow::bail!("llama-server executable not found.");
        }

        let mut cmd = Command::new(s.llama_server_exe.trim());
        cmd.args([
            "-m",
            s.gguf_path.trim(),
            "--port",
            &s.backend_port.to_string(),
            "-ngl",
            &s.ngl.to_string(),
            "-c",
            &s.ctx_len.to_string(),
        ]);
        let child = cmd
            .spawn()
            .map_err(|e| anyhow::anyhow!("spawn llama-server: {e}"))?;
        drop(s);

        *self.llama_child.lock().unwrap() = Some(child);
        let port = self
            .settings
            .read()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .backend_port;
        Self::wait_llama_health(port)?;
        Ok(())
    }
}
