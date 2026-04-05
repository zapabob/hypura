//! Persistent GUI / proxy settings (JSON under OS config dir).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct KoboldGuiSettings {
    pub gguf_path: String,
    pub llama_server_exe: String,
    pub backend_port: u16,
    pub kobold_port: u16,
    pub ngl: u32,
    pub ctx_len: u32,
    #[serde(default)]
    pub recent_ggufs: Vec<String>,
    /// Semicolon- or comma-separated directories for `GET /api/extra/models` (Hypura-compatible).
    #[serde(default)]
    pub model_scan_dirs: String,
}

impl Default for KoboldGuiSettings {
    fn default() -> Self {
        Self {
            gguf_path: String::new(),
            llama_server_exe: String::new(),
            backend_port: 8081,
            kobold_port: 5001,
            ngl: 99,
            ctx_len: 4096,
            recent_ggufs: vec![],
            model_scan_dirs: String::new(),
        }
    }
}

impl KoboldGuiSettings {
    pub fn settings_path() -> Result<PathBuf, String> {
        let base = dirs::data_dir().ok_or_else(|| "no data_dir".to_string())?;
        Ok(base.join("hypura").join("kobold_gguf_gui_settings.json"))
    }

    pub fn load() -> Self {
        match Self::settings_path().and_then(|p| {
            if p.exists() {
                std::fs::read_to_string(&p)
                    .map_err(|e| e.to_string())
                    .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            } else {
                Err("missing".to_string())
            }
        }) {
            Ok(s) => s,
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::settings_path().map_err(|e| anyhow::anyhow!(e))?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn push_recent_gguf(&mut self, path: &str) {
        let p = path.to_string();
        self.recent_ggufs.retain(|x| x != &p);
        self.recent_ggufs.insert(0, p);
        while self.recent_ggufs.len() > 32 {
            self.recent_ggufs.pop();
        }
    }
}
