use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, ensure};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::ollama_types::{GuiEventItem, GuiHistoryItem, GuiPresetItem, now_rfc3339};

pub const DEFAULT_NET_SAVE_SLOTS: usize = 100;
const PRELOAD_STORY_ID: &str = "preloadstory";
const CURRENT_LAUNCHER_CONFIG_META_KEY: &str = "current_launcher_config";
const UI_THEME_META_KEY: &str = "ui_theme";

#[derive(Debug, Clone)]
pub struct CompatStorageOptions {
    pub canonical_db_path: PathBuf,
    pub savedata_bridge_path: Option<PathBuf>,
    pub preload_story_path: Option<PathBuf>,
    pub admindir: Option<PathBuf>,
    pub migration_dir: Option<PathBuf>,
    pub slot_count: usize,
    pub default_ui_theme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CanonicalStoryBundle {
    pub title: Option<String>,
    pub story_text: String,
    pub memory: String,
    pub authors_note: String,
    pub world_info: Value,
    pub scenario: Value,
    pub sampler_preset: Value,
    pub metadata: Value,
    pub raw_story: Value,
}

impl CanonicalStoryBundle {
    pub fn from_story_json(raw_story: Value) -> Self {
        let title = first_string(
            &raw_story,
            &["title", "name", "story_name", "scenario_title", "chatname"],
        );
        let story_text = first_string(&raw_story, &["prompt", "story", "text"]).unwrap_or_default();
        let memory = first_string(&raw_story, &["memory"]).unwrap_or_default();
        let authors_note = first_string(&raw_story, &["authorsnote", "authornote", "authors_note"])
            .unwrap_or_default();
        let world_info =
            first_value(&raw_story, &["worldinfo", "world_info"]).unwrap_or(Value::Null);
        let scenario =
            first_value(&raw_story, &["scenario", "character", "char"]).unwrap_or(Value::Null);
        let sampler_preset = first_value(&raw_story, &["preset", "sampler", "sampler_settings"])
            .unwrap_or(Value::Null);
        let metadata =
            first_value(&raw_story, &["metadata"]).unwrap_or_else(|| Value::Object(Map::new()));

        Self {
            title,
            story_text,
            memory,
            authors_note,
            world_info,
            scenario,
            sampler_preset,
            metadata,
            raw_story,
        }
    }

    pub fn export_story_json(&self) -> Value {
        self.raw_story.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LauncherProfile {
    pub name: String,
    pub model_param: Option<String>,
    pub preloadstory: Option<String>,
    pub savedatafile: Option<String>,
    pub host: Option<String>,
    pub port_param: Option<u16>,
    pub gendefaults: Option<Value>,
    pub is_template: bool,
    pub raw_config: Value,
}

impl LauncherProfile {
    pub fn from_kcpps_value(name: impl Into<String>, raw_config: Value) -> Result<Self> {
        let name = name.into();
        ensure!(
            raw_config.is_object(),
            "launcher profile must be a JSON object"
        );
        let model_param = first_string(&raw_config, &["model_param", "model"]);
        let preloadstory = first_string(&raw_config, &["preloadstory"]);
        let savedatafile = first_string(&raw_config, &["savedatafile"]);
        let host = first_string(&raw_config, &["host"]);
        let port_param = raw_config
            .get("port_param")
            .and_then(parse_u16_value)
            .or_else(|| raw_config.get("port").and_then(parse_u16_value));
        let gendefaults = raw_config.get("gendefaults").cloned();
        let is_template = raw_config
            .get("istemplate")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        Ok(Self {
            name,
            model_param,
            preloadstory,
            savedatafile,
            host,
            port_param,
            gendefaults,
            is_template,
            raw_config,
        })
    }

    pub fn export_kcpps_value(&self) -> Value {
        self.raw_config.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SaveSlotRecord {
    pub slot: i64,
    pub title: String,
    pub format: String,
    pub data: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeStateSlotMetadata {
    pub slot: i64,
    pub model_path: String,
    pub model_identity: String,
    pub architecture: String,
    pub context_size: u32,
    pub token_count: u32,
    pub byte_size: u64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeStateSlotRecord {
    pub metadata: RuntimeStateSlotMetadata,
    pub token_ids: Vec<i32>,
    pub state_bytes: Vec<u8>,
}

pub struct CompatStorage {
    conn: Mutex<Connection>,
    options: CompatStorageOptions,
}

impl CompatStorage {
    pub fn open(options: CompatStorageOptions) -> Result<Self> {
        if let Some(parent) = options.canonical_db_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating compat storage directory {}", parent.display())
            })?;
        }

        let conn = Connection::open(&options.canonical_db_path).with_context(|| {
            format!(
                "opening compat storage database {}",
                options.canonical_db_path.display()
            )
        })?;
        Self::initialize_schema(&conn)?;

        let storage = Self {
            conn: Mutex::new(conn),
            options,
        };
        storage.bootstrap()?;
        Ok(storage)
    }

    pub fn canonical_db_path(&self) -> &Path {
        &self.options.canonical_db_path
    }

    pub fn savedata_bridge_path(&self) -> Option<&Path> {
        self.options.savedata_bridge_path.as_deref()
    }

    pub fn slot_count(&self) -> usize {
        self.options.slot_count
    }

    pub fn savedata_enabled(&self) -> bool {
        self.options.savedata_bridge_path.is_some()
    }

    pub fn admindir(&self) -> Option<&Path> {
        self.options.admindir.as_deref()
    }

    pub fn get_ui_theme(&self) -> Result<String> {
        let conn = self.conn.lock().unwrap();
        Ok(conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                [UI_THEME_META_KEY],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .unwrap_or_else(|| self.options.default_ui_theme.clone()))
    }

    pub fn set_ui_theme(&self, theme: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![UI_THEME_META_KEY, theme],
        )?;
        Ok(())
    }

    pub fn migrate_gui_state(
        &self,
        presets: &HashMap<String, GuiPresetItem>,
        history: &VecDeque<GuiHistoryItem>,
        events: &VecDeque<GuiEventItem>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        if Self::table_row_count(&conn, "ui_presets")? == 0 {
            for preset in presets.values() {
                Self::save_gui_preset_locked(&conn, preset)?;
            }
        }
        if Self::table_row_count(&conn, "gui_history")? == 0 {
            for item in history.iter().rev() {
                Self::push_history_locked(&conn, item)?;
            }
        }
        if Self::table_row_count(&conn, "gui_events")? == 0 {
            for item in events.iter().rev() {
                Self::push_event_locked(&conn, item)?;
            }
        }
        Ok(())
    }

    pub fn list_gui_presets(&self) -> Result<Vec<GuiPresetItem>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT name, payload_json, updated_at
             FROM ui_presets
             ORDER BY name ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            let payload_text: String = row.get(1)?;
            let payload: Value = serde_json::from_str(&payload_text).map_err(to_sql_error)?;
            Ok(GuiPresetItem {
                name: row.get(0)?,
                payload,
                updated_at: row.get(2)?,
            })
        })?;
        let mut items = Vec::new();
        for row in rows {
            items.push(row?);
        }
        Ok(items)
    }

    pub fn save_gui_preset(&self, preset: &GuiPresetItem) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        Self::save_gui_preset_locked(&conn, preset)
    }

    pub fn delete_gui_preset(&self, name: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM ui_presets WHERE name = ?1", [name])?;
        Ok(())
    }

    pub fn list_gui_history(&self, limit: usize) -> Result<Vec<GuiHistoryItem>> {
        let conn = self.conn.lock().unwrap();
        Self::list_json_rows::<GuiHistoryItem>(&conn, "gui_history", limit)
    }

    pub fn list_gui_events(&self, limit: usize) -> Result<Vec<GuiEventItem>> {
        let conn = self.conn.lock().unwrap();
        Self::list_json_rows::<GuiEventItem>(&conn, "gui_events", limit)
    }

    pub fn push_gui_history(&self, item: &GuiHistoryItem) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        Self::push_history_locked(&conn, item)
    }

    pub fn push_gui_event(&self, item: &GuiEventItem) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        Self::push_event_locked(&conn, item)
    }

    pub fn set_preload_story(&self, story: &CanonicalStoryBundle) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        Self::save_story_bundle_locked(&conn, PRELOAD_STORY_ID, story)
    }

    pub fn get_preload_story(&self) -> Result<Option<CanonicalStoryBundle>> {
        let conn = self.conn.lock().unwrap();
        Self::load_story_bundle_locked(&conn, PRELOAD_STORY_ID)
    }

    pub fn get_preload_story_json(&self) -> Result<Value> {
        Ok(self
            .get_preload_story()?
            .map(|story| story.export_story_json())
            .unwrap_or_else(|| json!({})))
    }

    pub fn import_preload_story_path(&self, path: &Path) -> Result<CanonicalStoryBundle> {
        let raw = load_json_file(path)?;
        let bundle = CanonicalStoryBundle::from_story_json(raw);
        self.set_preload_story(&bundle)?;
        Ok(bundle)
    }

    pub fn list_save_slot_titles(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut titles = vec![String::new(); self.options.slot_count];
        let mut stmt = conn.prepare("SELECT slot, title FROM save_slots ORDER BY slot ASC")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (slot, title) = row?;
            if slot >= 0 && (slot as usize) < titles.len() {
                titles[slot as usize] = title;
            }
        }
        Ok(titles)
    }

    pub fn load_save_slot(&self, slot: i64) -> Result<Option<SaveSlotRecord>> {
        ensure_valid_slot(slot, self.options.slot_count)?;
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT slot, title, format, data, updated_at
             FROM save_slots
             WHERE slot = ?1",
            [slot],
            |row| {
                Ok(SaveSlotRecord {
                    slot: row.get(0)?,
                    title: row.get(1)?,
                    format: row.get(2)?,
                    data: row.get(3)?,
                    updated_at: row.get(4)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    pub fn save_slot(&self, slot: i64, format: &str, title: &str, data: &str) -> Result<()> {
        ensure_valid_slot(slot, self.options.slot_count)?;
        if data.len() > 10 * 1024 * 1024 {
            return Err(anyhow!("story is too long"));
        }

        let conn = self.conn.lock().unwrap();
        if data.is_empty() {
            conn.execute("DELETE FROM save_slots WHERE slot = ?1", [slot])?;
        } else {
            let title = if title.trim().is_empty() {
                "Untitled Save".to_string()
            } else {
                title.trim().to_string()
            };
            conn.execute(
                "INSERT INTO save_slots(slot, title, format, data, updated_at)
                 VALUES(?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(slot) DO UPDATE SET
                    title = excluded.title,
                    format = excluded.format,
                    data = excluded.data,
                    updated_at = excluded.updated_at",
                params![slot, title, format, data, now_rfc3339()],
            )?;
        }
        Self::export_savedata_bridge_locked(&conn, self.options.savedata_bridge_path.as_deref())?;
        Ok(())
    }

    pub fn import_savedata_bridge_path(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, "{}")?;
            return Ok(());
        }

        let text = fs::read_to_string(path)
            .with_context(|| format!("reading savedata bridge {}", path.display()))?;
        if text.trim().is_empty() {
            return Ok(());
        }
        let parsed: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing savedata bridge {}", path.display()))?;
        let Some(entries) = parsed.as_object() else {
            return Err(anyhow!("savedata bridge must be a JSON object"));
        };

        let conn = self.conn.lock().unwrap();
        for (slot_text, entry) in entries {
            let slot = slot_text
                .parse::<i64>()
                .with_context(|| format!("invalid slot id '{slot_text}'"))?;
            ensure_valid_slot(slot, self.options.slot_count)?;
            let Some(entry_obj) = entry.as_object() else {
                return Err(anyhow!("savedata slot '{slot_text}' must be an object"));
            };
            let title = entry_obj
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("Untitled Save");
            let format = entry_obj
                .get("format")
                .and_then(Value::as_str)
                .unwrap_or("");
            let data = entry_obj.get("data").and_then(Value::as_str).unwrap_or("");
            conn.execute(
                "INSERT INTO save_slots(slot, title, format, data, updated_at)
                 VALUES(?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(slot) DO UPDATE SET
                    title = excluded.title,
                    format = excluded.format,
                    data = excluded.data,
                    updated_at = excluded.updated_at",
                params![slot, title, format, data, now_rfc3339()],
            )?;
        }
        Ok(())
    }

    pub fn export_savedata_bridge_path(&self, path: &Path) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        Self::export_savedata_bridge_locked(&conn, Some(path))
    }

    pub fn save_runtime_state_slot(
        &self,
        slot: i64,
        metadata: &RuntimeStateSlotMetadata,
        token_ids: &[i32],
        state_bytes: &[u8],
    ) -> Result<()> {
        ensure_valid_slot(slot, self.options.slot_count)?;
        ensure!(
            metadata.slot == slot,
            "runtime state slot metadata mismatch"
        );

        let state_path = self.runtime_state_file_path(slot);
        if let Some(parent) = state_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&state_path, state_bytes)
            .with_context(|| format!("writing runtime state slot {}", state_path.display()))?;

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO runtime_state_slots(
                slot,
                model_path,
                model_identity,
                architecture,
                context_size,
                token_count,
                byte_size,
                token_ids_json,
                state_path,
                created_at,
                updated_at
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(slot) DO UPDATE SET
                model_path = excluded.model_path,
                model_identity = excluded.model_identity,
                architecture = excluded.architecture,
                context_size = excluded.context_size,
                token_count = excluded.token_count,
                byte_size = excluded.byte_size,
                token_ids_json = excluded.token_ids_json,
                state_path = excluded.state_path,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at",
            params![
                slot,
                metadata.model_path,
                metadata.model_identity,
                metadata.architecture,
                i64::from(metadata.context_size),
                i64::from(metadata.token_count),
                metadata.byte_size as i64,
                serde_json::to_string(token_ids)?,
                state_path.to_string_lossy().to_string(),
                metadata.created_at,
                metadata.updated_at,
            ],
        )?;
        Ok(())
    }

    pub fn load_runtime_state_slot(&self, slot: i64) -> Result<Option<RuntimeStateSlotRecord>> {
        ensure_valid_slot(slot, self.options.slot_count)?;
        let conn = self.conn.lock().unwrap();
        let row: Option<(String, String, String, i64, i64, i64, String, String, String, String)> = conn
            .query_row(
                "SELECT model_path, model_identity, architecture, context_size, token_count, byte_size, token_ids_json, state_path, created_at, updated_at
                 FROM runtime_state_slots
                 WHERE slot = ?1",
                [slot],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                        row.get(8)?,
                        row.get(9)?,
                    ))
                },
            )
            .optional()?;
        drop(conn);

        let Some((
            model_path,
            model_identity,
            architecture,
            context_size,
            token_count,
            byte_size,
            token_ids_json,
            state_path,
            created_at,
            updated_at,
        )) = row
        else {
            return Ok(None);
        };

        let state_path = PathBuf::from(state_path);
        let state_bytes = fs::read(&state_path)
            .with_context(|| format!("reading runtime state slot {}", state_path.display()))?;
        let token_ids: Vec<i32> = serde_json::from_str(&token_ids_json)
            .with_context(|| format!("parsing token ids for runtime state slot {}", slot))?;
        Ok(Some(RuntimeStateSlotRecord {
            metadata: RuntimeStateSlotMetadata {
                slot,
                model_path,
                model_identity,
                architecture,
                context_size: u32::try_from(context_size)
                    .map_err(|_| anyhow!("invalid context_size in runtime state slot"))?,
                token_count: u32::try_from(token_count)
                    .map_err(|_| anyhow!("invalid token_count in runtime state slot"))?,
                byte_size: u64::try_from(byte_size)
                    .map_err(|_| anyhow!("invalid byte_size in runtime state slot"))?,
                created_at,
                updated_at,
            },
            token_ids,
            state_bytes,
        }))
    }

    pub fn list_admin_options(&self) -> Result<Vec<String>> {
        let Some(admindir) = self.options.admindir.as_ref() else {
            return Ok(Vec::new());
        };
        let mut items = Vec::new();
        for entry in fs::read_dir(admindir)
            .with_context(|| format!("reading admindir {}", admindir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            let ext = ext.to_ascii_lowercase();
            if !matches!(ext.as_str(), "kcpps" | "kcppt" | "gguf") {
                continue;
            }
            if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                items.push(name.to_string());
            }
        }
        items.sort_by_key(|item| item.to_ascii_lowercase());
        Ok(items)
    }

    pub fn import_launcher_config_path(&self, path: &Path) -> Result<LauncherProfile> {
        let raw = load_json_file(path)?;
        let name = path
            .file_name()
            .and_then(|file| file.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| "profile.kcpps".to_string());
        let profile = LauncherProfile::from_kcpps_value(name, raw)?;
        self.save_launcher_profile(&profile)?;
        Ok(profile)
    }

    pub fn save_launcher_profile(&self, profile: &LauncherProfile) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let raw_text = serde_json::to_string_pretty(&profile.raw_config)?;
        let gendefaults_text = profile
            .gendefaults
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        conn.execute(
            "INSERT INTO launcher_configs(
                name,
                raw_json,
                model_param,
                preloadstory,
                savedatafile,
                host,
                port_param,
                gendefaults_json,
                is_template,
                updated_at
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(name) DO UPDATE SET
                raw_json = excluded.raw_json,
                model_param = excluded.model_param,
                preloadstory = excluded.preloadstory,
                savedatafile = excluded.savedatafile,
                host = excluded.host,
                port_param = excluded.port_param,
                gendefaults_json = excluded.gendefaults_json,
                is_template = excluded.is_template,
                updated_at = excluded.updated_at",
            params![
                profile.name,
                raw_text,
                profile.model_param,
                profile.preloadstory,
                profile.savedatafile,
                profile.host,
                profile.port_param.map(i64::from),
                gendefaults_text,
                if profile.is_template { 1 } else { 0 },
                now_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn list_launcher_profiles(&self) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT name FROM launcher_configs ORDER BY name ASC")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut items = Vec::new();
        for row in rows {
            items.push(row?);
        }
        Ok(items)
    }

    pub fn get_launcher_profile(&self, name: &str) -> Result<Option<LauncherProfile>> {
        let conn = self.conn.lock().unwrap();
        Self::load_launcher_profile_locked(&conn, name)
    }

    pub fn get_current_launcher_profile(&self) -> Result<Option<LauncherProfile>> {
        let conn = self.conn.lock().unwrap();
        let current_name: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                [CURRENT_LAUNCHER_CONFIG_META_KEY],
                |row| row.get(0),
            )
            .optional()?;
        let Some(name) = current_name else {
            return Ok(None);
        };
        Self::load_launcher_profile_locked(&conn, &name)
    }

    pub fn set_current_launcher_profile(&self, name: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![CURRENT_LAUNCHER_CONFIG_META_KEY, name],
        )?;
        Ok(())
    }

    pub fn export_current_launcher_profile(&self, path: &Path) -> Result<()> {
        let profile = self
            .get_current_launcher_profile()?
            .ok_or_else(|| anyhow!("no current launcher profile has been stored"))?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            path,
            serde_json::to_vec_pretty(&profile.export_kcpps_value())?,
        )?;
        Ok(())
    }

    pub fn reload_admin_profile(
        &self,
        filename: &str,
        baseconfig: Option<&str>,
    ) -> Result<LauncherProfile> {
        let allowed = self.list_admin_options()?;
        ensure!(
            allowed.iter().any(|entry| entry == filename),
            "config '{filename}' is not available in admindir"
        );
        let admindir = self
            .options
            .admindir
            .as_ref()
            .ok_or_else(|| anyhow!("admindir is not configured"))?;
        let target_path = admindir.join(filename);
        let ext = target_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();

        let profile = if matches!(ext.as_str(), "kcpps" | "kcppt") {
            self.import_launcher_config_path(&target_path)?
        } else if ext == "gguf" {
            let mut raw =
                if let Some(baseconfig) = baseconfig.filter(|value| !value.trim().is_empty()) {
                    ensure!(
                        allowed.iter().any(|entry| entry == baseconfig),
                        "baseconfig '{baseconfig}' is not available in admindir"
                    );
                    load_json_file(&admindir.join(baseconfig))?
                } else {
                    json!({})
                };
            let raw_obj = raw
                .as_object_mut()
                .ok_or_else(|| anyhow!("baseconfig must be a JSON object"))?;
            raw_obj.insert(
                "model_param".to_string(),
                Value::String(filename.to_string()),
            );
            let profile = LauncherProfile::from_kcpps_value(filename.to_string(), raw)?;
            self.save_launcher_profile(&profile)?;
            profile
        } else {
            return Err(anyhow!("unsupported admin config type '{}'", ext));
        };

        self.set_current_launcher_profile(&profile.name)?;
        Ok(profile)
    }

    fn bootstrap(&self) -> Result<()> {
        {
            let conn = self.conn.lock().unwrap();
            let has_theme: Option<String> = conn
                .query_row(
                    "SELECT value FROM meta WHERE key = ?1",
                    [UI_THEME_META_KEY],
                    |row| row.get(0),
                )
                .optional()?;
            if has_theme.is_none() {
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES(?1, ?2)",
                    params![UI_THEME_META_KEY, self.options.default_ui_theme],
                )?;
            }
        }

        if let Some(migration_dir) = self.options.migration_dir.as_deref() {
            self.import_migration_dir(migration_dir)?;
        }
        if let Some(savedata_bridge) = self.options.savedata_bridge_path.as_deref() {
            self.import_savedata_bridge_path(savedata_bridge)?;
        }
        if let Some(preload_story) = self.options.preload_story_path.as_deref() {
            self.import_preload_story_path(preload_story)?;
        }
        if let Some(admindir) = self.options.admindir.as_deref() {
            self.import_launcher_configs_from_dir(admindir)?;
        }
        Ok(())
    }

    fn import_launcher_configs_from_dir(&self, admindir: &Path) -> Result<()> {
        for entry in fs::read_dir(admindir)
            .with_context(|| format!("reading admindir {}", admindir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            let ext = ext.to_ascii_lowercase();
            if matches!(ext.as_str(), "kcpps" | "kcppt") {
                self.import_launcher_config_path(&path)?;
            }
        }
        Ok(())
    }

    fn import_migration_dir(&self, migration_dir: &Path) -> Result<()> {
        if !migration_dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(migration_dir)
            .with_context(|| format!("reading migration directory {}", migration_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            match ext.to_ascii_lowercase().as_str() {
                "jsondb" => self.import_savedata_bridge_path(&path)?,
                "kcpps" | "kcppt" => {
                    self.import_launcher_config_path(&path)?;
                }
                "json" => {
                    if self.get_preload_story()?.is_none() {
                        let _ = self.import_preload_story_path(&path);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn initialize_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "BEGIN;
             CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS story_bundles(
                id TEXT PRIMARY KEY,
                title TEXT,
                raw_json TEXT NOT NULL,
                story_text TEXT NOT NULL,
                memory TEXT NOT NULL,
                authors_note TEXT NOT NULL,
                world_info_json TEXT NOT NULL,
                scenario_json TEXT NOT NULL,
                sampler_preset_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS save_slots(
                slot INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                format TEXT NOT NULL,
                data TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS launcher_configs(
                name TEXT PRIMARY KEY,
                raw_json TEXT NOT NULL,
                model_param TEXT,
                preloadstory TEXT,
                savedatafile TEXT,
                host TEXT,
                port_param INTEGER,
                gendefaults_json TEXT,
                is_template INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS runtime_state_slots(
                slot INTEGER PRIMARY KEY,
                model_path TEXT NOT NULL,
                model_identity TEXT NOT NULL,
                architecture TEXT NOT NULL,
                context_size INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                byte_size INTEGER NOT NULL,
                token_ids_json TEXT NOT NULL DEFAULT '[]',
                state_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS ui_presets(
                name TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS gui_history(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                item_json TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS gui_events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                item_json TEXT NOT NULL
             );
             COMMIT;",
        )?;
        Ok(())
    }

    fn runtime_state_file_path(&self, slot: i64) -> PathBuf {
        let base_dir = self
            .options
            .canonical_db_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        base_dir
            .join("runtime-state")
            .join(format!("slot-{slot}.bin"))
    }

    fn table_row_count(conn: &Connection, table: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        Ok(conn.query_row(&sql, [], |row| row.get(0))?)
    }

    fn save_story_bundle_locked(
        conn: &Connection,
        id: &str,
        story: &CanonicalStoryBundle,
    ) -> Result<()> {
        conn.execute(
            "INSERT INTO story_bundles(
                id,
                title,
                raw_json,
                story_text,
                memory,
                authors_note,
                world_info_json,
                scenario_json,
                sampler_preset_json,
                metadata_json,
                updated_at
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                raw_json = excluded.raw_json,
                story_text = excluded.story_text,
                memory = excluded.memory,
                authors_note = excluded.authors_note,
                world_info_json = excluded.world_info_json,
                scenario_json = excluded.scenario_json,
                sampler_preset_json = excluded.sampler_preset_json,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at",
            params![
                id,
                story.title,
                serde_json::to_string_pretty(&story.raw_story)?,
                story.story_text,
                story.memory,
                story.authors_note,
                serde_json::to_string(&story.world_info)?,
                serde_json::to_string(&story.scenario)?,
                serde_json::to_string(&story.sampler_preset)?,
                serde_json::to_string(&story.metadata)?,
                now_rfc3339(),
            ],
        )?;
        Ok(())
    }

    fn load_story_bundle_locked(
        conn: &Connection,
        id: &str,
    ) -> Result<Option<CanonicalStoryBundle>> {
        conn.query_row(
            "SELECT title, raw_json, story_text, memory, authors_note, world_info_json, scenario_json, sampler_preset_json, metadata_json
             FROM story_bundles
             WHERE id = ?1",
            [id],
            |row| {
                let raw_story: Value = serde_json::from_str(&row.get::<_, String>(1)?)
                    .map_err(to_sql_error)?;
                let world_info: Value = serde_json::from_str(&row.get::<_, String>(5)?)
                    .map_err(to_sql_error)?;
                let scenario: Value = serde_json::from_str(&row.get::<_, String>(6)?)
                    .map_err(to_sql_error)?;
                let sampler_preset: Value = serde_json::from_str(&row.get::<_, String>(7)?)
                    .map_err(to_sql_error)?;
                let metadata: Value = serde_json::from_str(&row.get::<_, String>(8)?)
                    .map_err(to_sql_error)?;
                Ok(CanonicalStoryBundle {
                    title: row.get(0)?,
                    raw_story,
                    story_text: row.get(2)?,
                    memory: row.get(3)?,
                    authors_note: row.get(4)?,
                    world_info,
                    scenario,
                    sampler_preset,
                    metadata,
                })
            },
        )
        .optional()
        .map_err(Into::into)
    }

    fn save_gui_preset_locked(conn: &Connection, preset: &GuiPresetItem) -> Result<()> {
        conn.execute(
            "INSERT INTO ui_presets(name, payload_json, updated_at)
             VALUES(?1, ?2, ?3)
             ON CONFLICT(name) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at",
            params![
                preset.name,
                serde_json::to_string(&preset.payload)?,
                preset.updated_at
            ],
        )?;
        Ok(())
    }

    fn list_json_rows<T>(conn: &Connection, table: &str, limit: usize) -> Result<Vec<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let sql = format!(
            "SELECT item_json FROM {table} ORDER BY seq DESC LIMIT {}",
            limit.max(1)
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            let item_json: String = row.get(0)?;
            serde_json::from_str::<T>(&item_json).map_err(to_sql_error)
        })?;
        let mut items = Vec::new();
        for row in rows {
            items.push(row?);
        }
        Ok(items)
    }

    fn push_history_locked(conn: &Connection, item: &GuiHistoryItem) -> Result<()> {
        conn.execute(
            "INSERT INTO gui_history(ts, item_json) VALUES(?1, ?2)",
            params![item.ts, serde_json::to_string(item)?],
        )?;
        Self::trim_json_table_locked(conn, "gui_history", 200)?;
        Ok(())
    }

    fn push_event_locked(conn: &Connection, item: &GuiEventItem) -> Result<()> {
        conn.execute(
            "INSERT INTO gui_events(ts, item_json) VALUES(?1, ?2)",
            params![item.ts, serde_json::to_string(item)?],
        )?;
        Self::trim_json_table_locked(conn, "gui_events", 200)?;
        Ok(())
    }

    fn trim_json_table_locked(conn: &Connection, table: &str, limit: usize) -> Result<()> {
        let sql = format!(
            "DELETE FROM {table}
             WHERE seq NOT IN (
                SELECT seq FROM {table} ORDER BY seq DESC LIMIT {}
             )",
            limit.max(1)
        );
        conn.execute(&sql, [])?;
        Ok(())
    }

    fn load_launcher_profile_locked(
        conn: &Connection,
        name: &str,
    ) -> Result<Option<LauncherProfile>> {
        conn.query_row(
            "SELECT raw_json FROM launcher_configs WHERE name = ?1",
            [name],
            |row| {
                let raw_json: String = row.get(0)?;
                let raw: Value = serde_json::from_str(&raw_json).map_err(to_sql_error)?;
                LauncherProfile::from_kcpps_value(name.to_string(), raw).map_err(to_sql_error)
            },
        )
        .optional()
        .map_err(Into::into)
    }

    fn export_savedata_bridge_locked(conn: &Connection, path: Option<&Path>) -> Result<()> {
        let Some(path) = path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut stmt = conn.prepare(
            "SELECT slot, title, format, data
             FROM save_slots
             ORDER BY slot ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut root = Map::new();
        for row in rows {
            let (slot, title, format, data) = row?;
            root.insert(
                slot.to_string(),
                json!({
                    "title": title,
                    "format": format,
                    "data": data,
                }),
            );
        }
        fs::write(path, serde_json::to_vec_pretty(&Value::Object(root))?)?;
        Ok(())
    }
}

fn ensure_valid_slot(slot: i64, slot_count: usize) -> Result<()> {
    ensure!(slot >= 0, "invalid save slot");
    ensure!((slot as usize) < slot_count, "invalid save slot");
    Ok(())
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        let candidate = value.get(*key)?;
        if let Some(text) = candidate.as_str() {
            return Some(text.to_string());
        }
    }
    None
}

fn first_value(value: &Value, keys: &[&str]) -> Option<Value> {
    keys.iter().find_map(|key| value.get(*key).cloned())
}

fn parse_u16_value(value: &Value) -> Option<u16> {
    value
        .as_u64()
        .and_then(|n| u16::try_from(n).ok())
        .or_else(|| {
            value
                .as_str()
                .and_then(|text| text.trim().parse::<u16>().ok())
        })
}

fn load_json_file(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading JSON file {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing JSON file {}", path.display()))
}

fn to_sql_error<E>(error: E) -> rusqlite::Error
where
    E: std::fmt::Display,
{
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        error.to_string(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_storage(savedata_bridge: Option<&Path>) -> (TempDir, CompatStorage) {
        let temp = TempDir::new().unwrap();
        let storage = CompatStorage::open(CompatStorageOptions {
            canonical_db_path: temp.path().join("compat.sqlite3"),
            savedata_bridge_path: savedata_bridge.map(Path::to_path_buf),
            preload_story_path: None,
            admindir: None,
            migration_dir: None,
            slot_count: 8,
            default_ui_theme: "classic".to_string(),
        })
        .unwrap();
        (temp, storage)
    }

    #[test]
    fn story_json_roundtrip_preserves_unknown_fields() {
        let raw = json!({
            "title": "Demo Story",
            "prompt": "Once upon a time",
            "memory": "remember me",
            "authorsnote": "keep tone warm",
            "worldinfo": [{ "key": "city", "entry": "Neo Kyoto" }],
            "scenario": { "name": "pilot" },
            "preset": { "temperature": 0.7 },
            "metadata": { "version": 2 },
            "custom_field": { "nested": [1, 2, 3] }
        });

        let bundle = CanonicalStoryBundle::from_story_json(raw.clone());

        assert_eq!(bundle.title.as_deref(), Some("Demo Story"));
        assert_eq!(bundle.story_text, "Once upon a time");
        assert_eq!(bundle.memory, "remember me");
        assert_eq!(bundle.export_story_json(), raw);
    }

    #[test]
    fn launcher_profile_roundtrip_preserves_kcpps_payload() {
        let raw = json!({
            "model_param": "model.gguf",
            "preloadstory": "story.json",
            "savedatafile": "remote.jsondb",
            "port_param": 5001,
            "gendefaults": {
                "temperature": 0.7
            },
            "custom_option": true
        });

        let profile = LauncherProfile::from_kcpps_value("demo.kcpps", raw.clone()).unwrap();

        assert_eq!(profile.name, "demo.kcpps");
        assert_eq!(profile.model_param.as_deref(), Some("model.gguf"));
        assert_eq!(profile.port_param, Some(5001));
        assert_eq!(profile.export_kcpps_value(), raw);
    }

    #[test]
    fn savedata_bridge_roundtrip_matches_koboldcpp_shape() {
        let temp = TempDir::new().unwrap();
        let bridge_path = temp.path().join("saves.jsondb");
        let (_, storage) = temp_storage(Some(&bridge_path));

        storage
            .save_slot(2, "kcpp_lzma_b64", "Slot Two", "payload-123")
            .unwrap();

        let exported: Value =
            serde_json::from_str(&fs::read_to_string(&bridge_path).unwrap()).unwrap();
        assert_eq!(
            exported,
            json!({
                "2": {
                    "title": "Slot Two",
                    "format": "kcpp_lzma_b64",
                    "data": "payload-123"
                }
            })
        );

        let imported_path = temp.path().join("import.jsondb");
        fs::write(
            &imported_path,
            serde_json::to_vec_pretty(&json!({
                "1": {
                    "title": "Imported",
                    "format": "kcpp_lzma_b64",
                    "data": "imported-payload"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        storage.import_savedata_bridge_path(&imported_path).unwrap();
        let record = storage.load_save_slot(1).unwrap().unwrap();
        assert_eq!(record.title, "Imported");
        assert_eq!(record.data, "imported-payload");
    }

    #[test]
    fn runtime_state_slot_roundtrip_persists_metadata_and_bytes() {
        let (_temp, storage) = temp_storage(None);
        let metadata = RuntimeStateSlotMetadata {
            slot: 3,
            model_path: "C:/models/demo.gguf".to_string(),
            model_identity: "demo.gguf".to_string(),
            architecture: "llama".to_string(),
            context_size: 4096,
            token_count: 321,
            byte_size: 5,
            created_at: now_rfc3339(),
            updated_at: now_rfc3339(),
        };

        storage
            .save_runtime_state_slot(3, &metadata, &[11, 22, 33], b"abcde")
            .unwrap();

        let loaded = storage.load_runtime_state_slot(3).unwrap().unwrap();
        assert_eq!(loaded.metadata.slot, 3);
        assert_eq!(loaded.metadata.model_identity, "demo.gguf");
        assert_eq!(loaded.metadata.architecture, "llama");
        assert_eq!(loaded.metadata.context_size, 4096);
        assert_eq!(loaded.metadata.token_count, 321);
        assert_eq!(loaded.token_ids, vec![11, 22, 33]);
        assert_eq!(loaded.state_bytes, b"abcde");
    }

    #[test]
    fn runtime_state_slot_overwrite_replaces_previous_payload() {
        let (_temp, storage) = temp_storage(None);
        let mut metadata = RuntimeStateSlotMetadata {
            slot: 1,
            model_path: "C:/models/demo.gguf".to_string(),
            model_identity: "demo.gguf".to_string(),
            architecture: "llama".to_string(),
            context_size: 4096,
            token_count: 12,
            byte_size: 3,
            created_at: now_rfc3339(),
            updated_at: now_rfc3339(),
        };

        storage
            .save_runtime_state_slot(1, &metadata, &[1, 2, 3], b"one")
            .unwrap();

        metadata.token_count = 99;
        metadata.byte_size = 6;
        storage
            .save_runtime_state_slot(1, &metadata, &[4, 5, 6, 7], b"second")
            .unwrap();

        let loaded = storage.load_runtime_state_slot(1).unwrap().unwrap();
        assert_eq!(loaded.metadata.token_count, 99);
        assert_eq!(loaded.token_ids, vec![4, 5, 6, 7]);
        assert_eq!(loaded.state_bytes, b"second");
    }
}
