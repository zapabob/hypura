use std::path::Path;

use hypura::model::{
    gguf::GgufFile, metadata::ModelMetadata, tensor_role::TensorRole,
    turboquant_sidecar::read_gguf_turboquant_config,
};

use super::fmt_util::{format_bytes, format_params};

pub fn run(model_path: &str, show_tensors: bool) -> anyhow::Result<()> {
    let path = Path::new(model_path);
    anyhow::ensure!(path.exists(), "Model file not found: {model_path}");

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "gguf" => inspect_gguf(path, show_tensors),
        "safetensors" => anyhow::bail!("Safetensors inspect not yet implemented"),
        _ => anyhow::bail!("Unsupported model format: .{ext}"),
    }
}

fn inspect_gguf(path: &Path, show_tensors: bool) -> anyhow::Result<()> {
    let gguf = GgufFile::open(path)?;
    let metadata = ModelMetadata::from_gguf(&gguf)?;

    println!("Model: {}", path.display());
    println!("  Format: GGUF v{}", gguf.version);
    println!("  Architecture: {}", metadata.architecture);
    println!("  Parameters: {}", format_params(metadata.parameter_count));
    println!("  Layers: {}", metadata.num_layers);
    println!("  Embedding dim: {}", metadata.embedding_dim);
    println!("  Attention heads: {}", metadata.num_heads);
    println!("  KV heads: {}", metadata.num_kv_heads);
    println!("  Vocab size: {}", metadata.vocab_size);
    println!("  Context length: {}", metadata.context_length);
    if let Some(ref q) = metadata.quantization {
        println!("  Quantization: {q}");
    }
    if metadata.is_moe {
        println!(
            "  MoE: {} experts, {} active per token",
            metadata.num_experts.unwrap_or(0),
            metadata.num_experts_used.unwrap_or(0)
        );
    }
    println!(
        "  Total size: {:.1} GB",
        gguf.total_tensor_bytes() as f64 / (1u64 << 30) as f64
    );
    println!("  Tensors: {}", gguf.tensors.len());

    if let Some(tq) = read_gguf_turboquant_config(&gguf, &metadata)? {
        println!("  Triality/TurboQuant:");
        println!("    Source: {}", tq.source_label());
        println!("    Public mode: {}", tq.public_mode_label);
        println!("    Runtime mode: {}", tq.runtime_mode);
        println!("    Schema version: {}", tq.schema_version);
        println!(
            "    Rotation: {} (seed={})",
            tq.rotation_policy
                .map(|policy| policy.as_str().to_string())
                .unwrap_or_else(|| "none".to_string()),
            tq.rotation_seed
        );
        println!(
            "    Triality view: {} (mix={:.3})",
            tq.triality_view.as_deref().unwrap_or("none"),
            tq.triality_mix.unwrap_or(0.0)
        );
        println!(
            "    Fidelity: paper_fidelity={} k_bits={:.3} v_bits={:.3}",
            tq.paper_fidelity, tq.k_bits, tq.v_bits
        );
        println!(
            "    Payload: format={} bytes={} inline_json={}",
            tq.payload_format.as_deref().unwrap_or("none"),
            tq.payload_bytes,
            if tq.payload_json.is_some() {
                "yes"
            } else {
                "no"
            }
        );
    }

    if show_tensors {
        println!();
        println!(
            "  {:50} {:>12} {:>8} {:>10}",
            "Name", "Size", "Type", "Role"
        );
        println!("  {}", "-".repeat(84));
        for t in &gguf.tensors {
            let role = TensorRole::from_name(&t.name);
            println!(
                "  {:50} {:>12} {:>8} {:>10}",
                t.name,
                format_bytes(t.size_bytes),
                format!("{:?}", t.dtype),
                format!("{role:?}"),
            );
        }
    }

    Ok(())
}
