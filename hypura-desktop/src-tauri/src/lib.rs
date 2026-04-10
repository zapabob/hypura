use serde::Deserialize;
use serde_json::json;
use std::sync::Mutex;
use tauri::Manager;
use tauri::State;

#[derive(Default)]
struct ServeChildState {
    child: Option<std::process::Child>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServeCmd {
    pub model_path: String,
    pub host: String,
    pub port: u16,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SwitchModelCmd {
    pub path: String,
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[tauri::command]
fn pick_gguf(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let file = app
        .dialog()
        .file()
        .add_filter("GGUF", &["gguf"])
        .blocking_pick_file();
    Ok(file.map(|p| p.to_string()))
}

fn resolve_hypura_exe() -> Result<std::path::PathBuf, String> {
    if let Ok(p) = std::env::var("HYPURA_EXE") {
        let pb = std::path::PathBuf::from(p.trim());
        if pb.exists() {
            return Ok(pb);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let sidecar = dir.join("hypura.exe");
            if sidecar.exists() {
                return Ok(sidecar);
            }
            let sidecar = dir.join("hypura");
            if sidecar.exists() {
                return Ok(sidecar);
            }
        }
    }
    Ok(std::path::PathBuf::from(if cfg!(windows) {
        "hypura.exe"
    } else {
        "hypura"
    }))
}

#[tauri::command]
fn stop_hypura_serve(state: State<'_, Mutex<ServeChildState>>) -> Result<(), String> {
    let mut g = state.lock().map_err(|e| e.to_string())?;
    if let Some(mut c) = g.child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
    Ok(())
}

#[tauri::command]
fn spawn_hypura_serve(
    cmd: ServeCmd,
    state: State<'_, Mutex<ServeChildState>>,
) -> Result<String, String> {
    let mut g = state.lock().map_err(|e| e.to_string())?;
    if let Some(mut c) = g.child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }

    let exe = resolve_hypura_exe()?;
    let mut c = std::process::Command::new(&exe);
    c.arg("serve")
        .arg(&cmd.model_path)
        .arg("--host")
        .arg(&cmd.host)
        .arg("--port")
        .arg(cmd.port.to_string());
    let child = c.spawn().map_err(|e| format!("spawn hypura: {e}"))?;
    g.child = Some(child);
    Ok(format!("http://{}:{}", cmd.host, cmd.port))
}

#[tauri::command]
fn switch_model_http(cmd: SwitchModelCmd) -> Result<String, String> {
    let url = format!(
        "http://{}:{}/api/extra/model/switch",
        cmd.host, cmd.port
    );
    let body = json!({ "path": cmd.path.trim() });

    let resp = match cmd
        .api_key
        .as_ref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        Some(k) => ureq::post(&url)
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {k}"))
            .send_json(body),
        None => ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(body),
    }
    .map_err(|e| format!("HTTP request failed: {e}"))?;

    let status = resp.status();
    let text = resp.into_string().unwrap_or_default();
    if !(200..300).contains(&status) {
        return Err(format!("model switch failed ({status}): {text}"));
    }
    Ok(text)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Mutex::new(ServeChildState::default()))
        .invoke_handler(tauri::generate_handler![
            pick_gguf,
            spawn_hypura_serve,
            stop_hypura_serve,
            switch_model_http
        ])
        .setup(|app| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.set_title("Hypura Desktop");
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Hypura Desktop");
}
