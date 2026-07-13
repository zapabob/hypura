use std::io::{IsTerminal, stderr};

use hypura::model::elt_loop::{EltLoopMetadata, elt_loop_runtime_supported_from_env};

/// Whether to show indicatif spinners/bars (stderr TTY, not CI, NO_COLOR unset).
pub fn cli_progress_enabled() -> bool {
    stderr().is_terminal()
        && std::env::var("CI").map_or(true, |v| v.trim().is_empty())
        && std::env::var("NO_COLOR").map_or(true, |v| v.trim().is_empty())
}

pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / (1u64 << 10) as f64)
    } else {
        format!("{bytes} B")
    }
}

pub fn format_params(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1e6)
    } else {
        format!("{count}")
    }
}

pub fn format_bandwidth(bytes_per_sec: u64) -> String {
    let gb_s = bytes_per_sec as f64 / 1e9;
    if gb_s >= 1.0 {
        format!("{gb_s:.1} GB/s")
    } else {
        format!("{:.0} MB/s", bytes_per_sec as f64 / 1e6)
    }
}

pub fn print_elt_loop_status(elt_loop: Option<&EltLoopMetadata>, prefix: &str) {
    let Some(elt_loop) = elt_loop else {
        return;
    };
    let runtime_supported = elt_loop_runtime_supported_from_env();
    println!(
        "{prefix}ELT loop: enabled={}, required={}, L_min={}, L_default={}, L_max={}, unit={}, family={}, runtime_status={}, gate={}",
        elt_loop.enabled,
        elt_loop.required,
        elt_loop.l_min,
        elt_loop.l_default,
        elt_loop.l_max,
        elt_loop.loop_unit_label(),
        elt_loop.family_label(),
        elt_loop.gguf_runtime_status.as_deref().unwrap_or("unknown"),
        elt_loop.runtime_gate_label(runtime_supported),
    );
}
