use std::sync::{Arc, Mutex};

use tauri::Manager;

mod packaged;

#[derive(serde::Deserialize)]
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
    Ok(file.map(|value| value.to_string()))
}

#[tauri::command]
fn pick_asset_root(app: tauri::AppHandle) -> Result<Option<String>, String> {
    packaged::pick_asset_root(app)
}

#[tauri::command]
fn get_runtime_status(
    state: tauri::State<'_, Arc<Mutex<packaged::DesktopState>>>,
) -> Result<packaged::RuntimeStatus, String> {
    packaged::get_runtime_status(state)
}

#[tauri::command]
fn stop_managed_runtime(
    state: tauri::State<'_, Arc<Mutex<packaged::DesktopState>>>,
) -> Result<(), String> {
    packaged::stop_managed_runtime(state)
}

#[tauri::command]
fn launch_packaged_koboldcpp(
    cmd: packaged::PackagedLaunchCmd,
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<Mutex<packaged::DesktopState>>>,
) -> Result<String, String> {
    packaged::launch_packaged_koboldcpp(cmd, app, state)
}

#[tauri::command]
fn switch_model_http(cmd: SwitchModelCmd) -> Result<String, String> {
    let url = format!(
        "http://{}:{}/api/extra/model/switch",
        cmd.host, cmd.port
    );
    let body = serde_json::json!({ "path": cmd.path.trim() });
    let response = match cmd
        .api_key
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        Some(key) => ureq::post(&url)
            .set("Content-Type", "application/json")
            .set("Authorization", &format!("Bearer {key}"))
            .send_json(body),
        None => ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(body),
    }
    .map_err(|error| format!("HTTP request failed: {error}"))?;
    let status = response.status();
    let text = response.into_string().unwrap_or_default();
    if !(200..300).contains(&status) {
        return Err(format!("model switch failed ({status}): {text}"));
    }
    Ok(text)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Arc::new(Mutex::new(packaged::DesktopState::default())))
        .invoke_handler(tauri::generate_handler![
            pick_gguf,
            pick_asset_root,
            get_runtime_status,
            stop_managed_runtime,
            launch_packaged_koboldcpp,
            switch_model_http
        ])
        .setup(|app| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_title("Hypura Desktop");
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Hypura Desktop");
}
