//! KoboldCpp-compatible GGUF proxy GUI (also launched via `Hypura kobold-gui`).

mod kobold;
mod models;
mod server;
mod settings;
mod shared;
mod stream;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui;
use settings::KoboldGuiSettings;
use shared::SharedState;
use tokio::sync::watch;

struct GuiApp {
    edit: KoboldGuiSettings,
    shared: Arc<SharedState>,
    log: String,
    shutdown_tx: Option<watch::Sender<bool>>,
    proxy_task: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    runtime: Option<tokio::runtime::Runtime>,
}

impl GuiApp {
    fn new() -> Self {
        let edit = KoboldGuiSettings::load();
        Self {
            shared: SharedState::new(edit.clone()),
            edit,
            log: "Stop any running servers on the chosen ports before Start.\n".to_string(),
            shutdown_tx: None,
            proxy_task: None,
            runtime: None,
        }
    }

    fn persist_quiet(&self) {
        if let Err(e) = self.edit.save() {
            eprintln!("kobold_gguf_gui: failed to save settings: {e:#}");
        }
    }

    fn push_edit_to_shared(&self) {
        if let Ok(mut w) = self.shared.settings.write() {
            *w = self.edit.clone();
        }
    }

    fn append_log(&mut self, line: impl AsRef<str>) {
        self.log.push_str(line.as_ref());
        self.log.push('\n');
    }

    fn stop_all(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(true);
        }
        if let Some(t) = self.proxy_task.take() {
            let _ = t.abort();
        }
        self.runtime.take();
        self.shared.stop_llama();
        self.append_log("Stopped.");
    }

    fn start_backend(&mut self) -> anyhow::Result<()> {
        self.stop_all();

        if let Ok(r) = self.shared.settings.read() {
            if !r.gguf_path.trim().is_empty() {
                self.edit.gguf_path = r.gguf_path.clone();
            }
            self.edit.ctx_len = r.ctx_len;
        }

        if self.edit.gguf_path.trim().is_empty() || self.edit.llama_server_exe.trim().is_empty() {
            anyhow::bail!("Set GGUF path and llama-server executable.");
        }
        if !PathBuf::from(self.edit.gguf_path.trim()).exists() {
            anyhow::bail!("GGUF file not found.");
        }
        if !PathBuf::from(self.edit.llama_server_exe.trim()).exists() {
            anyhow::bail!("llama-server executable not found.");
        }
        let gguf_cur = self.edit.gguf_path.trim().to_string();
        self.edit.push_recent_gguf(&gguf_cur);
        self.edit.save()?;
        self.push_edit_to_shared();

        self.shared.spawn_llama()?;

        let kobold_port = self.shared.settings.read().expect("settings").kobold_port;
        self.append_log(format!(
            "llama-server OK. Kobold-compatible proxy: http://127.0.0.1:{kobold_port}/api/v1/generate"
        ));

        let rt = tokio::runtime::Runtime::new()?;
        let (tx, rx) = watch::channel(false);
        self.shutdown_tx = Some(tx);

        let bind: SocketAddr = format!("127.0.0.1:{}", kobold_port)
            .parse()
            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        let backend = self.shared.settings.read().expect("settings").backend_port;
        let gguf = PathBuf::from(
            self.shared
                .settings
                .read()
                .expect("settings")
                .gguf_path
                .trim(),
        );
        let advertised_model = gguf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        let ctx = self.shared.settings.read().expect("settings").ctx_len;
        let meta = std::sync::Arc::new(std::sync::RwLock::new(server::ProxyMeta {
            advertised_model,
            max_length: ctx.min(8192).max(256),
            max_context: ctx,
        }));

        let shared = Arc::clone(&self.shared);
        let gen = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = rt.spawn(async move {
            server::run_proxy_server(bind, backend, rx, meta, shared, gen).await
        });

        self.proxy_task = Some(task);
        self.runtime = Some(rt);

        Ok(())
    }
}

impl eframe::App for GuiApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_all();
        self.persist_quiet();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Kobold-compatible GGUF proxy (llama.cpp backend)");
            ui.label("KoboldCpp互換: /api/v1/*, /api/extra/*, /api/latest/*, /v1/* → llama-server（画像・音声・TTS等は未搭載で503）。");
            ui.label("Hypura互換: GET /api/extra/models, POST /api/extra/model/switch（起動中・llama生存時）。");
            ui.label(format!(
                "設定保存: {:?}",
                KoboldGuiSettings::settings_path().map(|p| p.display().to_string()).unwrap_or_else(|e| e.to_string())
            ));

            ui.horizontal(|ui| {
                ui.label("GGUF:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.edit.gguf_path).desired_width(420.0),
                );
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("GGUF", &["gguf"])
                        .pick_file()
                    {
                        self.edit.gguf_path = p.display().to_string();
                    }
                }
            });

            if !self.edit.recent_ggufs.is_empty() {
                let recent = self.edit.recent_ggufs.clone();
                ui.horizontal(|ui| {
                    ui.label("Recent GGUF:");
                    egui::ComboBox::from_id_salt("recent_gguf")
                        .selected_text("Pick from recent…")
                        .show_ui(ui, |ui| {
                            for r in recent {
                                if ui.selectable_label(self.edit.gguf_path == r, &r).clicked() {
                                    self.edit.gguf_path = r.clone();
                                }
                            }
                        });
                });
            }

            ui.label("Model scan dirs ( ; or , ) for /api/extra/models:");
            ui.add(
                egui::TextEdit::multiline(&mut self.edit.model_scan_dirs)
                    .desired_width(520.0)
                    .desired_rows(2),
            );

            ui.horizontal(|ui| {
                ui.label("llama-server exe:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.edit.llama_server_exe)
                        .desired_width(420.0),
                );
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("exe", &["exe"])
                        .pick_file()
                    {
                        self.edit.llama_server_exe = p.display().to_string();
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Backend port (llama-server):");
                ui.add(
                    egui::DragValue::new(&mut self.edit.backend_port).range(1024_u16..=65500),
                );
                ui.label("Kobold proxy port:");
                ui.add(egui::DragValue::new(&mut self.edit.kobold_port).range(1024_u16..=65500));
            });

            ui.horizontal(|ui| {
                ui.label("GPU layers (-ngl):");
                ui.add(egui::DragValue::new(&mut self.edit.ngl).range(0_u32..=999));
                ui.label("Context (-c):");
                ui.add(egui::DragValue::new(&mut self.edit.ctx_len).range(256_u32..=262144));
            });

            ui.horizontal(|ui| {
                if ui.button("Start").clicked() {
                    match self.start_backend() {
                        Ok(()) => {}
                        Err(e) => {
                            self.append_log(format!("ERROR: {e:#}"));
                            self.stop_all();
                        }
                    }
                }
                if ui.button("Stop").clicked() {
                    self.stop_all();
                }
                if ui.button("Save settings").clicked() {
                    self.push_edit_to_shared();
                    match self.edit.save() {
                        Ok(()) => self.append_log("Settings saved."),
                        Err(e) => self.append_log(format!("Save failed: {e:#}")),
                    }
                }
            });

            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.monospace(&self.log);
            });
        });
    }
}

/// Run the egui window (blocking until closed). Window title shows Hypura for unified branding.
pub fn run_native() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([720.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Hypura Kobold GUI",
        native_options,
        Box::new(|_cc| Ok(Box::new(GuiApp::new()) as Box<dyn eframe::App>)),
    )
}
