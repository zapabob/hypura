use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::compute::ffi::{LlamaBackend, LlamaContext, LlamaModel};

use super::ollama_types::{OpenAiEmbeddingsRequest, OpenAiPromptInput};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EmbeddingItem {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OpenAiEmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingItem>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

pub struct EmbeddingsRuntime {
    _backend: LlamaBackend,
    model: LlamaModel,
    model_name: String,
    embedding_dim: usize,
    max_input_tokens: usize,
    n_threads: i32,
}

// SAFETY: the runtime is always accessed under a Mutex when used from async code.
unsafe impl Send for EmbeddingsRuntime {}

impl EmbeddingsRuntime {
    pub fn load(model_path: &Path) -> Result<Self> {
        let backend = LlamaBackend::init();
        let model = LlamaModel::load(model_path, 0, true)
            .with_context(|| format!("loading embeddings model {}", model_path.display()))?;
        let model_name = model_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| "kcpp-embeddings".to_string());
        let embedding_dim = model.n_embd_out().max(1) as usize;
        let max_input_tokens = model.n_ctx_train().max(1) as usize;
        let n_threads = std::thread::available_parallelism()
            .map(|count| count.get().clamp(1, 16) as i32)
            .unwrap_or(4);
        Ok(Self {
            _backend: backend,
            model,
            model_name,
            embedding_dim,
            max_input_tokens,
            n_threads,
        })
    }

    pub fn max_input_tokens(&self) -> usize {
        self.max_input_tokens
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn embed_request(&mut self, req: &OpenAiEmbeddingsRequest) -> Result<OpenAiEmbeddingsResponse> {
        let inputs = match &req.input {
            OpenAiPromptInput::Single(text) => vec![text.clone()],
            OpenAiPromptInput::Many(items) => items.clone(),
        };

        let mut data = Vec::with_capacity(inputs.len());
        let mut total_tokens = 0u32;
        for (index, input) in inputs.iter().enumerate() {
            let mut tokens = self.model.tokenize(input, true, true);
            if tokens.len() > self.max_input_tokens {
                if req.truncate {
                    tokens.truncate(self.max_input_tokens);
                } else {
                    anyhow::bail!(
                        "input exceeds embedding limit of {} tokens; set truncate=true to allow truncation",
                        self.max_input_tokens
                    );
                }
            }
            total_tokens += tokens.len() as u32;
            let mut ctx = LlamaContext::new_for_embeddings(
                &self.model,
                self.max_input_tokens as u32,
                self.max_input_tokens.min(tokens.len()).max(1) as u32,
                self.n_threads,
            )?;
            ctx.decode(&tokens)?;
            ctx.synchronize();
            let embedding = ctx.seq_embeddings(0, self.embedding_dim)?;
            data.push(EmbeddingItem {
                object: "embedding".to_string(),
                index,
                embedding,
            });
        }

        Ok(OpenAiEmbeddingsResponse {
            object: "list".to_string(),
            data,
            model: req
                .model
                .clone()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| self.model_name.clone()),
            usage: EmbeddingUsage {
                prompt_tokens: total_tokens,
                total_tokens,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_request_supports_string_and_array_inputs() {
        let single = serde_json::from_str::<OpenAiEmbeddingsRequest>(
            r#"{"model":"kcpp","input":"hello","truncate":true}"#,
        )
        .unwrap();
        assert!(matches!(single.input, OpenAiPromptInput::Single(_)));
        assert!(single.truncate);

        let many = serde_json::from_str::<OpenAiEmbeddingsRequest>(
            r#"{"input":["hello","world"]}"#,
        )
        .unwrap();
        assert!(matches!(many.input, OpenAiPromptInput::Many(_)));
        assert!(!many.truncate);
    }
}
