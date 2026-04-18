use std::path::PathBuf;

use anyhow::{bail, Context};
use base64::Engine;
use tempfile::TempDir;

use crate::server::ollama_types::{ChatMediaAttachment, ChatMediaKind, ChatMessage};

pub struct MaterializedMedia {
    _temp_dir: TempDir,
    pub image_paths: Vec<PathBuf>,
    pub audio_paths: Vec<PathBuf>,
}

pub fn has_chat_media(messages: &[ChatMessage]) -> bool {
    messages
        .iter()
        .any(|message| !message.images.is_empty() || !message.audio.is_empty())
}

pub fn collect_chat_media(messages: &[ChatMessage]) -> anyhow::Result<MaterializedMedia> {
    let temp_dir = tempfile::Builder::new()
        .prefix("hypura-multimodal-")
        .tempdir()
        .context("failed to create temporary media directory")?;

    let mut image_paths = Vec::new();
    let mut audio_paths = Vec::new();

    for (message_index, message) in messages.iter().enumerate() {
        for (media_index, attachment) in message.media_attachments().into_iter().enumerate() {
            let path = materialize_attachment(
                &temp_dir,
                &attachment,
                message_index,
                media_index,
            )?;
            match attachment.kind {
                ChatMediaKind::Image => image_paths.push(path),
                ChatMediaKind::Audio => audio_paths.push(path),
            }
        }
    }

    Ok(MaterializedMedia {
        _temp_dir: temp_dir,
        image_paths,
        audio_paths,
    })
}

fn materialize_attachment(
    temp_dir: &TempDir,
    attachment: &ChatMediaAttachment,
    message_index: usize,
    media_index: usize,
) -> anyhow::Result<PathBuf> {
    let trimmed = attachment.value.trim();
    if trimmed.is_empty() {
        bail!("received an empty multimodal attachment");
    }

    let candidate_path = PathBuf::from(trimmed);
    if candidate_path.is_file() {
        return Ok(candidate_path);
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        bail!("remote multimodal URLs are not supported in this Hypura M1 slice");
    }

    let (decoded, extension) = if let Some(payload) = trimmed.strip_prefix("data:") {
        decode_data_url(payload, attachment.format_hint.as_deref())?
    } else {
        let extension = attachment
            .format_hint
            .as_deref()
            .map(normalize_extension)
            .unwrap_or_else(|| default_extension(attachment.kind));
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(trimmed)
            .with_context(|| "attachment was neither a readable path nor valid base64")?;
        (decoded, extension)
    };

    let file_name = format!(
        "msg-{message_index}-media-{media_index}.{}",
        extension.trim_start_matches('.')
    );
    let path = temp_dir.path().join(file_name);
    std::fs::write(&path, decoded)
        .with_context(|| format!("failed to materialize {}", path.display()))?;
    Ok(path)
}

fn decode_data_url(
    payload: &str,
    explicit_format: Option<&str>,
) -> anyhow::Result<(Vec<u8>, String)> {
    let Some((meta, data)) = payload.split_once(',') else {
        bail!("invalid data URL for multimodal attachment");
    };
    if !meta.contains(";base64") {
        bail!("only base64 data URLs are supported for multimodal attachments");
    }

    let mime = meta.split(';').next().unwrap_or_default();
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .with_context(|| "invalid base64 payload inside data URL")?;
    let extension = explicit_format
        .map(normalize_extension)
        .unwrap_or_else(|| extension_from_mime(mime));
    Ok((decoded, extension))
}

fn normalize_extension(value: &str) -> String {
    let trimmed = value.trim().trim_start_matches('.').to_ascii_lowercase();
    if trimmed.is_empty() {
        return "bin".to_string();
    }
    trimmed
}

fn extension_from_mime(mime: &str) -> String {
    match mime.trim().to_ascii_lowercase().as_str() {
        "image/png" => "png".to_string(),
        "image/jpeg" => "jpg".to_string(),
        "image/webp" => "webp".to_string(),
        "audio/wav" | "audio/x-wav" => "wav".to_string(),
        "audio/mpeg" => "mp3".to_string(),
        "audio/flac" => "flac".to_string(),
        _ => "bin".to_string(),
    }
}

fn default_extension(kind: ChatMediaKind) -> String {
    match kind {
        ChatMediaKind::Image => "bin".to_string(),
        ChatMediaKind::Audio => "bin".to_string(),
    }
}
