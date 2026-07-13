use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use serde::{Deserialize, Serialize};

use crate::council::{TrialityBranchMetrics, TrialityConsensusMetrics};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialityExecution {
    SingleView,
    BestPerLayer,
    AttentionLogitConsensus,
    ResidualParity,
}

impl TrialityExecution {
    fn to_native(self) -> hypura_sys::llama_tq_execution {
        match self {
            Self::SingleView => hypura_sys::llama_tq_execution_LLAMA_TQ_EXEC_SINGLE_VIEW,
            Self::BestPerLayer => hypura_sys::llama_tq_execution_LLAMA_TQ_EXEC_BEST_PER_LAYER,
            Self::AttentionLogitConsensus => {
                hypura_sys::llama_tq_execution_LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS
            }
            Self::ResidualParity => hypura_sys::llama_tq_execution_LLAMA_TQ_EXEC_RESIDUAL_PARITY,
        }
    }

    fn from_native(value: hypura_sys::llama_tq_execution) -> anyhow::Result<Self> {
        match value {
            hypura_sys::llama_tq_execution_LLAMA_TQ_EXEC_SINGLE_VIEW => Ok(Self::SingleView),
            hypura_sys::llama_tq_execution_LLAMA_TQ_EXEC_BEST_PER_LAYER => Ok(Self::BestPerLayer),
            hypura_sys::llama_tq_execution_LLAMA_TQ_EXEC_ATTENTION_LOGIT_CONSENSUS => {
                Ok(Self::AttentionLogitConsensus)
            }
            hypura_sys::llama_tq_execution_LLAMA_TQ_EXEC_RESIDUAL_PARITY => {
                Ok(Self::ResidualParity)
            }
            other => anyhow::bail!("llama.cpp returned unknown Triality execution value {other}"),
        }
    }

    pub fn capability_bit(self) -> u32 {
        1_u32 << (self.to_native() as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialityView {
    Vector,
    SpinorPlusProxy,
    SpinorMinusProxy,
}

impl TrialityView {
    fn to_native(self) -> hypura_sys::llama_tq_view {
        match self {
            Self::Vector => hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_VECTOR,
            Self::SpinorPlusProxy => hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_SPINOR_PLUS_PROXY,
            Self::SpinorMinusProxy => hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_SPINOR_MINUS_PROXY,
        }
    }

    fn from_native(value: hypura_sys::llama_tq_view) -> anyhow::Result<Self> {
        match value {
            hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_VECTOR => Ok(Self::Vector),
            hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_SPINOR_PLUS_PROXY => Ok(Self::SpinorPlusProxy),
            hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_SPINOR_MINUS_PROXY => {
                Ok(Self::SpinorMinusProxy)
            }
            other => anyhow::bail!("llama.cpp returned unknown Triality view value {other}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialityBranchConfig {
    pub view: TrialityView,
    pub weight: f32,
    pub bias: f32,
    pub scale: f32,
    pub temperature: f32,
    pub expected_error: f32,
    pub bits_per_channel: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialityLayerConfig {
    pub branches: [TrialityBranchConfig; 3],
    pub active_branch_mask: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialityContextConfig {
    pub schema_version: u32,
    pub execution: TrialityExecution,
    pub layers: Vec<TrialityLayerConfig>,
    pub required: bool,
    pub trace_enabled: bool,
    pub js_fallback_threshold: f32,
    #[serde(default)]
    pub allow_identity_view_fallback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialityModelCapabilities {
    pub metadata_present: bool,
    pub three_view_bundle: bool,
    pub ncka_available: bool,
    pub ncka_static_fallback_selected: bool,
    pub urt_available: bool,
    pub schema_version: u32,
    pub n_layers: u32,
    pub supported_execution_mask: u32,
    pub selected_execution: TrialityExecution,
    pub profile_id: String,
}

impl TrialityModelCapabilities {
    pub fn supports(&self, execution: TrialityExecution) -> bool {
        self.supported_execution_mask & execution.capability_bit() != 0
    }
}

fn fixed_c_string<const N: usize>(bytes: &[std::ffi::c_char; N]) -> String {
    let length = bytes.iter().position(|byte| *byte == 0).unwrap_or(N);
    let raw = bytes[..length]
        .iter()
        .map(|byte| *byte as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&raw).into_owned()
}

fn triality_error(operation: &str, error: &hypura_sys::llama_tq_error) -> anyhow::Error {
    let message = fixed_c_string(&error.message);
    if message.is_empty() {
        anyhow::anyhow!("{operation} failed with Triality error code {}", error.code)
    } else {
        anyhow::anyhow!(
            "{operation} failed with Triality error code {}: {message}",
            error.code
        )
    }
}

fn native_triality_config(
    config: &TrialityContextConfig,
) -> anyhow::Result<(
    Vec<hypura_sys::llama_tq_layer_config>,
    hypura_sys::llama_tq_context_config,
)> {
    anyhow::ensure!(
        config.schema_version > 0,
        "Triality schema version must be positive"
    );
    anyhow::ensure!(
        !config.layers.is_empty(),
        "Triality config must contain layers"
    );
    anyhow::ensure!(
        config.js_fallback_threshold.is_finite() && config.js_fallback_threshold >= 0.0,
        "Triality JS fallback threshold must be finite and non-negative"
    );

    let mut native_layers = Vec::with_capacity(config.layers.len());
    for layer in &config.layers {
        anyhow::ensure!(
            layer.active_branch_mask > 0 && layer.active_branch_mask <= 0b111,
            "Triality active branch mask must use bits 0..2 and select at least one branch"
        );
        let mut branches = [None, None, None];
        let mut active_mask = 0_u8;
        let mut active_count = 0_u32;
        let mut active_weight_sum = 0.0_f32;
        for (source_index, branch) in layer.branches.iter().enumerate() {
            let target_index = branch.view.to_native() as usize;
            anyhow::ensure!(
                branches[target_index].replace(branch).is_none(),
                "Triality layer contains a duplicate view"
            );
            if layer.active_branch_mask & (1_u32 << source_index) != 0 {
                active_mask |= 1_u8 << target_index;
                active_count += 1;
                active_weight_sum += branch.weight;
            }
        }
        anyhow::ensure!(
            branches.iter().all(Option::is_some),
            "Triality layer must contain each canonical view exactly once"
        );
        anyhow::ensure!(
            active_weight_sum.is_finite() && (active_weight_sum - 1.0).abs() <= 1.0e-5,
            "Triality active branch weights must sum to one"
        );
        anyhow::ensure!(
            config.execution != TrialityExecution::SingleView || active_count == 1,
            "single-view Triality execution requires exactly one active branch"
        );
        anyhow::ensure!(
            !matches!(
                config.execution,
                TrialityExecution::AttentionLogitConsensus | TrialityExecution::ResidualParity
            ) || active_count == 3,
            "consensus and residual-parity Triality execution require all three branches"
        );
        let native_branches = branches
            .map(|branch| {
                let branch = branch
                    .ok_or_else(|| anyhow::anyhow!("Triality layer is missing a canonical view"))?;
                anyhow::ensure!(
                    [
                        branch.weight,
                        branch.bias,
                        branch.scale,
                        branch.temperature,
                        branch.expected_error,
                        branch.bits_per_channel,
                    ]
                    .iter()
                    .all(|value| value.is_finite()),
                    "Triality branch values must be finite"
                );
                anyhow::ensure!(
                    branch.weight >= 0.0
                        && branch.scale > 0.0
                        && branch.temperature > 0.0
                        && branch.expected_error >= 0.0
                        && branch.bits_per_channel >= 0.0,
                    "Triality branch weights, errors, and bit budgets must be non-negative and scale/temperature positive"
                );
                let bits_milli = f64::from(branch.bits_per_channel) * 1000.0;
                anyhow::ensure!(
                    bits_milli <= f64::from(u32::MAX),
                    "Triality branch bit budget exceeds ABI range"
                );
                Ok(hypura_sys::llama_tq_branch_config {
                    view: branch.view.to_native(),
                    weight: branch.weight,
                    bias: branch.bias,
                    scale: branch.scale,
                    temperature: branch.temperature,
                    expected_error: branch.expected_error,
                    bits_per_channel_milli: bits_milli.round() as u32,
                })
            })
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?;
        native_layers.push(hypura_sys::llama_tq_layer_config {
            branches: native_branches.try_into().map_err(|_| {
                anyhow::anyhow!("Triality layer must contain exactly three branches")
            })?,
            active_mask,
        });
    }

    let native = hypura_sys::llama_tq_context_config {
        schema_version: config.schema_version,
        execution: config.execution.to_native(),
        layers: native_layers.as_ptr(),
        n_layers: native_layers.len(),
        required: config.required,
        trace_enabled: config.trace_enabled,
        js_fallback_threshold: config.js_fallback_threshold,
        allow_identity_view_fallback: config.allow_identity_view_fallback,
    };
    Ok((native_layers, native))
}

fn rust_triality_config(
    native: &hypura_sys::llama_tq_context_config,
    layers: &[hypura_sys::llama_tq_layer_config],
) -> anyhow::Result<TrialityContextConfig> {
    anyhow::ensure!(
        native.n_layers == layers.len(),
        "llama.cpp returned inconsistent Triality layer storage"
    );
    let layers = layers
        .iter()
        .map(|layer| {
            let branches = layer
                .branches
                .iter()
                .map(|branch| {
                    Ok(TrialityBranchConfig {
                        view: TrialityView::from_native(branch.view)?,
                        weight: branch.weight,
                        bias: branch.bias,
                        scale: branch.scale,
                        temperature: branch.temperature,
                        expected_error: branch.expected_error,
                        bits_per_channel: branch.bits_per_channel_milli as f32 / 1000.0,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .try_into()
                .map_err(|_| anyhow::anyhow!("llama.cpp returned an invalid branch count"))?;
            Ok(TrialityLayerConfig {
                branches,
                active_branch_mask: u32::from(layer.active_mask),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(TrialityContextConfig {
        schema_version: native.schema_version,
        execution: TrialityExecution::from_native(native.execution)?,
        layers,
        required: native.required,
        trace_enabled: native.trace_enabled,
        js_fallback_threshold: native.js_fallback_threshold,
        allow_identity_view_fallback: native.allow_identity_view_fallback,
    })
}

fn rust_triality_metrics(
    native: &hypura_sys::llama_tq_consensus_metrics,
) -> TrialityConsensusMetrics {
    TrialityConsensusMetrics {
        branches: std::array::from_fn(|index| {
            let branch = &native.branches[index];
            TrialityBranchMetrics {
                logit_mean: f64::from(branch.logit_mean),
                logit_variance: f64::from(branch.logit_variance),
                logit_l2: f64::from(branch.logit_l2),
                probability_entropy: f64::from(branch.probability_entropy),
                top1_probability: f64::from(branch.top1_probability),
                orthogonality_error: f64::from(branch.orthogonality_error),
                determinant_error: f64::from(branch.determinant_error),
                expected_quantisation_error: f64::from(branch.expected_quantisation_error),
                bytes_read: branch.bytes_read,
                duration_us: branch.duration_us,
            }
        }),
        pairwise_js: native.pairwise_js.map(f64::from),
        mean_pairwise_js: f64::from(native.mean_pairwise_js),
        max_pairwise_js: f64::from(native.max_pairwise_js),
        numerical_rank: f64::from(native.numerical_rank),
        effective_rank: f64::from(native.effective_rank),
        ka_fallback_used: native.ka_fallback_used,
        operator_word_hash_128: format!(
            "{:016x}{:016x}",
            native.operator_word_hash_hi, native.operator_word_hash_lo
        ),
    }
}

fn validate_triality_metrics(metrics: &TrialityConsensusMetrics) -> anyhow::Result<()> {
    let finite = metrics
        .branches
        .iter()
        .flat_map(|branch| {
            [
                branch.logit_mean,
                branch.logit_variance,
                branch.logit_l2,
                branch.probability_entropy,
                branch.top1_probability,
                branch.orthogonality_error,
                branch.determinant_error,
                branch.expected_quantisation_error,
            ]
        })
        .chain(metrics.pairwise_js)
        .chain([
            metrics.mean_pairwise_js,
            metrics.max_pairwise_js,
            metrics.numerical_rank,
            metrics.effective_rank,
        ])
        .all(f64::is_finite);
    anyhow::ensure!(finite, "llama.cpp returned non-finite Triality metrics");
    anyhow::ensure!(
        metrics
            .branches
            .iter()
            .all(|branch| branch.logit_variance >= 0.0
                && branch.logit_l2 >= 0.0
                && branch.probability_entropy >= 0.0
                && (0.0..=1.0).contains(&branch.top1_probability)
                && branch.orthogonality_error >= 0.0
                && branch.determinant_error >= 0.0
                && branch.expected_quantisation_error >= 0.0)
            && metrics.pairwise_js.iter().all(|value| *value >= 0.0)
            && metrics.mean_pairwise_js >= 0.0
            && metrics.max_pairwise_js >= 0.0
            && metrics.numerical_rank >= 0.0
            && metrics.effective_rank >= 0.0,
        "llama.cpp returned out-of-range Triality metrics"
    );
    Ok(())
}

/// Sampling parameters for token generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_a: f32,
    pub top_p: f32,
    pub tfs: f32,
    pub typical: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u32,
    pub max_tokens: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_a: 0.0,
            top_p: 0.9,
            tfs: 1.0,
            typical: 1.0,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 42,
            max_tokens: 512,
        }
    }
}

/// Performance data from llama.cpp context.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerfData {
    pub t_load_ms: f64,
    pub t_p_eval_ms: f64,
    pub t_eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
}

/// RAII guard for llama.cpp backend initialization.
/// Must be created before any other llama.cpp calls.
pub struct LlamaBackend(());

impl LlamaBackend {
    pub fn init() -> Self {
        unsafe { hypura_sys::llama_backend_init() }
        Self(())
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        unsafe { hypura_sys::llama_backend_free() }
    }
}

/// Safe wrapper around `llama_model *`.
pub struct LlamaModel {
    ptr: *mut hypura_sys::llama_model,
    vocab: *const hypura_sys::llama_vocab,
}

// SAFETY: llama_model is immutable after loading. llama.cpp supports multiple
// independently-owned contexts reading the same model concurrently.
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

impl LlamaModel {
    /// Load a model from a GGUF file.
    pub fn load(path: &Path, n_gpu_layers: i32, use_mmap: bool) -> anyhow::Result<Self> {
        let c_path = CString::new(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid path encoding"))?,
        )?;

        let mut params = unsafe { hypura_sys::llama_model_default_params() };
        params.n_gpu_layers = n_gpu_layers;
        params.use_mmap = use_mmap;

        let ptr = unsafe { hypura_sys::llama_model_load_from_file(c_path.as_ptr(), params) };
        anyhow::ensure!(
            !ptr.is_null(),
            "Failed to load model from {}",
            path.display()
        );

        let vocab = unsafe { hypura_sys::llama_model_get_vocab(ptr) };
        anyhow::ensure!(!vocab.is_null(), "Failed to get vocab from model");

        Ok(Self { ptr, vocab })
    }

    /// Load a model with custom tensor buffer type overrides.
    /// `overrides` is a NULL-terminated array of pattern/buft pairs.
    pub fn load_with_overrides(
        path: &Path,
        n_gpu_layers: i32,
        use_mmap: bool,
        overrides: *const hypura_sys::llama_model_tensor_buft_override,
    ) -> anyhow::Result<Self> {
        let c_path = CString::new(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid path encoding"))?,
        )?;

        let mut params = unsafe { hypura_sys::llama_model_default_params() };
        params.n_gpu_layers = n_gpu_layers;
        params.use_mmap = use_mmap;
        params.tensor_buft_overrides = overrides;

        let ptr = unsafe { hypura_sys::llama_model_load_from_file(c_path.as_ptr(), params) };
        anyhow::ensure!(
            !ptr.is_null(),
            "Failed to load model from {}",
            path.display()
        );

        let vocab = unsafe { hypura_sys::llama_model_get_vocab(ptr) };
        anyhow::ensure!(!vocab.is_null(), "Failed to get vocab from model");

        Ok(Self { ptr, vocab })
    }

    pub fn as_ptr(&self) -> *mut hypura_sys::llama_model {
        self.ptr
    }

    pub fn n_layers(&self) -> i32 {
        unsafe { hypura_sys::llama_model_n_layer(self.ptr) }
    }

    pub fn n_ctx_train(&self) -> i32 {
        unsafe { hypura_sys::llama_model_n_ctx_train(self.ptr) }
    }

    pub fn n_embd_out(&self) -> i32 {
        unsafe { hypura_sys::llama_model_n_embd_out(self.ptr) }
    }

    pub fn vocab_size(&self) -> anyhow::Result<usize> {
        let count = unsafe { hypura_sys::llama_vocab_n_tokens(self.vocab) };
        usize::try_from(count).map_err(|_| anyhow::anyhow!("llama.cpp returned invalid vocab size"))
    }

    pub fn triality_capabilities(&self) -> anyhow::Result<TrialityModelCapabilities> {
        let mut native = hypura_sys::llama_tq_model_capabilities::default();
        let mut error = hypura_sys::llama_tq_error::default();
        let ok = unsafe {
            hypura_sys::llama_tq_model_get_capabilities(self.ptr, &mut native, &mut error)
        };
        if !ok {
            return Err(triality_error("llama_tq_model_get_capabilities", &error));
        }
        Ok(TrialityModelCapabilities {
            metadata_present: native.metadata_present,
            three_view_bundle: native.three_view_bundle,
            ncka_available: native.ncka_available,
            ncka_static_fallback_selected: native.ncka_static_fallback_selected,
            urt_available: native.urt_available,
            schema_version: native.schema_version,
            n_layers: native.n_layers,
            supported_execution_mask: native.supported_execution_mask,
            selected_execution: TrialityExecution::from_native(native.selected_execution)?,
            profile_id: fixed_c_string(&native.profile_id),
        })
    }

    /// Get the model's embedded chat template, if any.
    pub fn chat_template(&self) -> Option<String> {
        let ptr = unsafe { hypura_sys::llama_model_chat_template(self.ptr, ptr::null()) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string())
        }
    }

    /// Check if a token signals end of generation.
    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { hypura_sys::llama_vocab_is_eog(self.vocab, token) }
    }

    /// Tokenize text into token IDs.
    ///
    /// When `parse_special` is true, chat-template markers and other special
    /// tokens are matched against the vocab instead of being tokenized as raw
    /// bytes.
    pub fn tokenize(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<i32> {
        let c_text = CString::new(text).unwrap_or_default();
        let max = (text.len() as i32 + 32).max(64);
        let mut tokens = vec![0i32; max as usize];

        let n = unsafe {
            hypura_sys::llama_tokenize(
                self.vocab,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                max,
                add_bos,
                parse_special,
            )
        };

        if n < 0 {
            // Buffer too small, retry
            tokens.resize((-n) as usize, 0);
            let n2 = unsafe {
                hypura_sys::llama_tokenize(
                    self.vocab,
                    c_text.as_ptr(),
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    -n,
                    add_bos,
                    parse_special,
                )
            };
            tokens.truncate(n2.max(0) as usize);
        } else {
            tokens.truncate(n as usize);
        }
        tokens
    }

    /// Convert a token ID to its text piece.
    pub fn token_to_piece(&self, token: i32) -> String {
        let mut buf = vec![0u8; 128];
        let n = unsafe {
            hypura_sys::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
                0,
                false,
            )
        };

        if n < 0 {
            buf.resize((-n) as usize, 0);
            let n2 = unsafe {
                hypura_sys::llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr() as *mut i8,
                    buf.len() as i32,
                    0,
                    false,
                )
            };
            buf.truncate(n2.max(0) as usize);
        } else {
            buf.truncate(n as usize);
        }
        String::from_utf8_lossy(&buf).to_string()
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hypura_sys::llama_model_free(self.ptr) }
        }
    }
}

fn context_params(
    n_ctx: u32,
    n_batch: u32,
    n_threads: i32,
    kv_quant: Option<crate::scheduler::types::KvQuantization>,
    disable_flash_attn: bool,
) -> hypura_sys::llama_context_params {
    use crate::scheduler::types::KvQuantization;

    let mut params = unsafe { hypura_sys::llama_context_default_params() };
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    params.offload_kqv = true;
    if disable_flash_attn {
        params.flash_attn_type = hypura_sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    if let Some(kv) = kv_quant {
        let ggml_type = match kv {
            KvQuantization::F16 => hypura_sys::ggml_type_GGML_TYPE_F16,
            KvQuantization::Q8_0 => hypura_sys::ggml_type_GGML_TYPE_Q8_0,
            KvQuantization::Q4_0 => hypura_sys::ggml_type_GGML_TYPE_Q4_0,
        };
        params.type_k = ggml_type;
        params.type_v = ggml_type;
        tracing::info!("KV cache quantization: {:?}", kv);
    }
    params
}

/// Safe wrapper around `llama_context *`.
pub struct LlamaContext {
    ptr: *mut hypura_sys::llama_context,
}

impl LlamaContext {
    /// Create a new inference context from a loaded model.
    pub fn new(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
    ) -> anyhow::Result<Self> {
        Self::new_inner(
            model,
            n_ctx,
            n_batch,
            n_threads,
            None,
            std::ptr::null_mut(),
            None,
            false,
            None,
        )
    }

    pub fn new_for_embeddings(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
    ) -> anyhow::Result<Self> {
        let mut params = unsafe { hypura_sys::llama_context_default_params() };
        params.n_ctx = n_ctx;
        params.n_batch = n_batch;
        params.n_ubatch = n_batch;
        params.n_threads = n_threads;
        params.n_threads_batch = n_threads;
        params.pooling_type = hypura_sys::llama_pooling_type_LLAMA_POOLING_TYPE_MEAN;
        params.attention_type = hypura_sys::llama_attention_type_LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        params.embeddings = true;
        params.offload_kqv = true;

        let ptr = unsafe { hypura_sys::llama_init_from_model(model.as_ptr(), params) };
        anyhow::ensure!(!ptr.is_null(), "Failed to create embedding llama context");
        unsafe {
            hypura_sys::llama_set_embeddings(ptr, true);
            hypura_sys::llama_set_causal_attn(ptr, false);
        }
        Ok(Self { ptr })
    }

    pub fn new_with_triality(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        config: &TrialityContextConfig,
    ) -> anyhow::Result<Self> {
        Self::new_inner(
            model,
            n_ctx,
            n_batch,
            n_threads,
            None,
            std::ptr::null_mut(),
            None,
            false,
            Some(config),
        )
    }

    /// Create a context with a cb_eval callback for layer tracking.
    /// `callback_data` must outlive the context.
    pub fn new_with_callback(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
    ) -> anyhow::Result<Self> {
        Self::new_with_callback_and_options(
            model,
            n_ctx,
            n_batch,
            n_threads,
            cb_eval,
            callback_data,
            false,
        )
    }

    pub fn new_with_callback_and_options(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        disable_flash_attn: bool,
    ) -> anyhow::Result<Self> {
        Self::new_inner(
            model,
            n_ctx,
            n_batch,
            n_threads,
            cb_eval,
            callback_data,
            None,
            disable_flash_attn,
            None,
        )
    }

    /// Create a context with callback and KV cache quantization.
    pub fn new_with_callback_and_kv(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        kv_quant: Option<crate::scheduler::types::KvQuantization>,
    ) -> anyhow::Result<Self> {
        Self::new_with_callback_and_kv_and_options(
            model,
            n_ctx,
            n_batch,
            n_threads,
            cb_eval,
            callback_data,
            kv_quant,
            false,
        )
    }

    pub fn new_with_callback_and_kv_and_options(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        kv_quant: Option<crate::scheduler::types::KvQuantization>,
        disable_flash_attn: bool,
    ) -> anyhow::Result<Self> {
        Self::new_inner(
            model,
            n_ctx,
            n_batch,
            n_threads,
            cb_eval,
            callback_data,
            kv_quant,
            disable_flash_attn,
            None,
        )
    }

    pub fn new_with_options_and_triality(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        kv_quant: Option<crate::scheduler::types::KvQuantization>,
        disable_flash_attn: bool,
        triality: Option<&TrialityContextConfig>,
    ) -> anyhow::Result<Self> {
        Self::new_inner(
            model,
            n_ctx,
            n_batch,
            n_threads,
            cb_eval,
            callback_data,
            kv_quant,
            disable_flash_attn,
            triality,
        )
    }

    fn new_inner(
        model: &LlamaModel,
        n_ctx: u32,
        n_batch: u32,
        n_threads: i32,
        cb_eval: hypura_sys::ggml_backend_sched_eval_callback,
        callback_data: *mut std::ffi::c_void,
        kv_quant: Option<crate::scheduler::types::KvQuantization>,
        disable_flash_attn: bool,
        triality: Option<&TrialityContextConfig>,
    ) -> anyhow::Result<Self> {
        let mut params = context_params(n_ctx, n_batch, n_threads, kv_quant, disable_flash_attn);

        if cb_eval.is_some() {
            params.cb_eval = cb_eval;
            params.cb_eval_user_data = callback_data;
        }

        let ptr = if let Some(triality) = triality {
            let (native_layers, native_config) = native_triality_config(triality)?;
            let mut error = hypura_sys::llama_tq_error::default();
            let ptr = unsafe {
                hypura_sys::llama_tq_init_from_model(
                    model.as_ptr(),
                    params,
                    &native_config,
                    &mut error,
                )
            };
            drop(native_layers);
            if ptr.is_null() {
                return Err(triality_error("llama_tq_init_from_model", &error));
            }
            ptr
        } else {
            let ptr = unsafe { hypura_sys::llama_init_from_model(model.as_ptr(), params) };
            anyhow::ensure!(!ptr.is_null(), "Failed to create llama context");
            ptr
        };

        Ok(Self { ptr })
    }

    /// Decode a batch of tokens (prompt processing or single-token step).
    pub fn decode(&mut self, tokens: &[i32]) -> anyhow::Result<()> {
        let batch = unsafe {
            hypura_sys::llama_batch_get_one(tokens.as_ptr() as *mut i32, tokens.len() as i32)
        };
        let ret = unsafe { hypura_sys::llama_decode(self.ptr, batch) };
        match ret {
            0 => Ok(()),
            1 => anyhow::bail!("KV cache full — try reducing context or batch size"),
            _ => anyhow::bail!("llama_decode failed with code {ret}"),
        }
    }

    pub fn configure_triality(&mut self, config: &TrialityContextConfig) -> anyhow::Result<()> {
        let (native_layers, native_config) = native_triality_config(config)?;
        let mut error = hypura_sys::llama_tq_error::default();
        let ok =
            unsafe { hypura_sys::llama_tq_context_configure(self.ptr, &native_config, &mut error) };
        drop(native_layers);
        if ok {
            Ok(())
        } else {
            Err(triality_error("llama_tq_context_configure", &error))
        }
    }

    pub fn triality_config(&self) -> anyhow::Result<TrialityContextConfig> {
        let mut native_out = hypura_sys::llama_tq_context_config::default();
        let mut required = 0_usize;
        let mut error = hypura_sys::llama_tq_error::default();
        let first = unsafe {
            hypura_sys::llama_tq_context_get_config(
                self.ptr,
                &mut native_out,
                ptr::null_mut(),
                0,
                &mut required,
                &mut error,
            )
        };
        let buffer_too_small =
            error.code == hypura_sys::llama_tq_error_code_LLAMA_TQ_ERROR_BUFFER_TOO_SMALL as i32;
        if !first && !buffer_too_small {
            return Err(triality_error("llama_tq_context_get_config", &error));
        }

        let mut layers = vec![hypura_sys::llama_tq_layer_config::default(); required];
        error = hypura_sys::llama_tq_error::default();
        let ok = unsafe {
            hypura_sys::llama_tq_context_get_config(
                self.ptr,
                &mut native_out,
                layers.as_mut_ptr(),
                layers.len(),
                &mut required,
                &mut error,
            )
        };
        if !ok {
            return Err(triality_error("llama_tq_context_get_config", &error));
        }
        anyhow::ensure!(
            required <= layers.len(),
            "llama.cpp returned an oversized Triality layer count"
        );
        layers.truncate(required);
        rust_triality_config(&native_out, &layers)
    }

    pub fn triality_metrics(&self) -> anyhow::Result<TrialityConsensusMetrics> {
        self.triality_metrics_optional()?.ok_or_else(|| {
            anyhow::anyhow!("llama.cpp has no Triality metrics for the current context")
        })
    }

    pub fn triality_metrics_optional(&self) -> anyhow::Result<Option<TrialityConsensusMetrics>> {
        let mut native = hypura_sys::llama_tq_consensus_metrics::default();
        let mut error = hypura_sys::llama_tq_error::default();
        let ok = unsafe {
            hypura_sys::llama_tq_context_get_last_metrics(self.ptr, &mut native, &mut error)
        };
        if !ok {
            if error.code == hypura_sys::llama_tq_error_code_LLAMA_TQ_ERROR_UNAVAILABLE as i32 {
                return Ok(None);
            }
            return Err(triality_error("llama_tq_context_get_last_metrics", &error));
        }
        let metrics = rust_triality_metrics(&native);
        validate_triality_metrics(&metrics)?;
        Ok(Some(metrics))
    }

    pub fn reset_triality_metrics(&mut self) {
        unsafe { hypura_sys::llama_tq_context_reset_metrics(self.ptr) }
    }

    pub fn logits(&self, vocab_size: usize) -> anyhow::Result<&[f32]> {
        let logits = unsafe { hypura_sys::llama_get_logits(self.ptr) };
        anyhow::ensure!(!logits.is_null(), "llama.cpp has no current logits");
        Ok(unsafe { std::slice::from_raw_parts(logits, vocab_size) })
    }

    pub fn logits_ith(&self, index: i32, vocab_size: usize) -> anyhow::Result<&[f32]> {
        let logits = unsafe { hypura_sys::llama_get_logits_ith(self.ptr, index) };
        anyhow::ensure!(
            !logits.is_null(),
            "llama.cpp has no logits for output index {index}"
        );
        Ok(unsafe { std::slice::from_raw_parts(logits, vocab_size) })
    }

    /// Get performance counters.
    pub fn perf(&self) -> PerfData {
        let data = unsafe { hypura_sys::llama_perf_context(self.ptr) };
        PerfData {
            t_load_ms: data.t_load_ms,
            t_p_eval_ms: data.t_p_eval_ms,
            t_eval_ms: data.t_eval_ms,
            n_p_eval: data.n_p_eval,
            n_eval: data.n_eval,
        }
    }

    pub fn state_size(&self) -> usize {
        unsafe { hypura_sys::llama_state_get_size(self.ptr) }
    }

    pub fn save_state_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let size = self.state_size();
        if size == 0 {
            return Ok(Vec::new());
        }
        let mut buffer = vec![0u8; size];
        let written =
            unsafe { hypura_sys::llama_state_get_data(self.ptr, buffer.as_mut_ptr(), size) };
        anyhow::ensure!(
            written <= size,
            "llama_state_get_data returned oversized state"
        );
        buffer.truncate(written);
        Ok(buffer)
    }

    pub fn load_state_bytes(&mut self, state: &[u8]) -> anyhow::Result<usize> {
        if state.is_empty() {
            return Ok(0);
        }
        let consumed =
            unsafe { hypura_sys::llama_state_set_data(self.ptr, state.as_ptr(), state.len()) };
        anyhow::ensure!(consumed > 0, "llama_state_set_data failed");
        anyhow::ensure!(
            consumed <= state.len(),
            "llama_state_set_data over-consumed input"
        );
        Ok(consumed)
    }

    pub fn remove_sequence_tokens(
        &mut self,
        sequence_id: i32,
        position_start: i32,
        position_end: i32,
    ) -> anyhow::Result<()> {
        let memory = unsafe { hypura_sys::llama_get_memory(self.ptr) };
        anyhow::ensure!(
            !memory.is_null(),
            "llama.cpp context has no sequence memory"
        );
        let removed = unsafe {
            hypura_sys::llama_memory_seq_rm(memory, sequence_id, position_start, position_end)
        };
        anyhow::ensure!(
            removed,
            "llama.cpp could not remove the requested token range"
        );
        Ok(())
    }

    pub fn as_ptr(&self) -> *mut hypura_sys::llama_context {
        self.ptr
    }

    pub fn synchronize(&mut self) {
        unsafe { hypura_sys::llama_synchronize(self.ptr) }
    }

    pub fn seq_embeddings(
        &mut self,
        seq_id: i32,
        embedding_dim: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let ptr = unsafe { hypura_sys::llama_get_embeddings_seq(self.ptr, seq_id) };
        let ptr = if ptr.is_null() {
            unsafe { hypura_sys::llama_get_embeddings(self.ptr) }
        } else {
            ptr
        };
        anyhow::ensure!(!ptr.is_null(), "embedding pointer is null");
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, embedding_dim) };
        Ok(slice.to_vec())
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hypura_sys::llama_free(self.ptr) }
        }
    }
}

/// Safe wrapper around a llama sampler chain.
pub struct LlamaSampler {
    ptr: *mut hypura_sys::llama_sampler,
}

impl LlamaSampler {
    /// Build a sampler chain from parameters.
    pub fn new(params: &SamplingParams) -> Self {
        Self::new_with_order(params, None)
    }

    /// Build a sampler chain from parameters with optional Kobold sampler order.
    /// Supported Kobold IDs:
    /// 0=top_k, 1=top_a, 2=top_p, 3=tfs, 4=typical, 5=temperature, 6=repetition_penalty, 7=min_p.
    pub fn new_with_order(params: &SamplingParams, sampler_order: Option<&[i32]>) -> Self {
        let chain_params = hypura_sys::llama_sampler_chain_params { no_perf: false };
        let ptr = unsafe { hypura_sys::llama_sampler_chain_init(chain_params) };

        let mut added_top_k = false;
        let mut added_top_a = false;
        let mut added_top_p = false;
        let mut added_tfs = false;
        let mut added_typical = false;
        let mut added_min_p = false;
        let mut added_temp = false;
        let mut added_pen = false;

        unsafe {
            let apply_id = |id: i32,
                            added_top_k: &mut bool,
                            added_top_a: &mut bool,
                            added_top_p: &mut bool,
                            added_tfs: &mut bool,
                            added_typical: &mut bool,
                            added_min_p: &mut bool,
                            added_temp: &mut bool,
                            added_pen: &mut bool| {
                match id {
                    0 if !*added_top_k => {
                        hypura_sys::llama_sampler_chain_add(
                            ptr,
                            hypura_sys::llama_sampler_init_top_k(params.top_k),
                        );
                        *added_top_k = true;
                    }
                    1 if !*added_top_a => {
                        hypura_sys::llama_sampler_chain_add(
                            ptr,
                            hypura_sys::hypura_sampler_init_top_a(params.top_a, 1),
                        );
                        *added_top_a = true;
                    }
                    2 if !*added_top_p => {
                        hypura_sys::llama_sampler_chain_add(
                            ptr,
                            hypura_sys::llama_sampler_init_top_p(params.top_p, 1),
                        );
                        *added_top_p = true;
                    }
                    3 if !*added_tfs => {
                        hypura_sys::llama_sampler_chain_add(
                            ptr,
                            hypura_sys::hypura_sampler_init_tfs_z(params.tfs, 1),
                        );
                        *added_tfs = true;
                    }
                    4 if !*added_typical => {
                        hypura_sys::llama_sampler_chain_add(
                            ptr,
                            hypura_sys::llama_sampler_init_typical(params.typical, 1),
                        );
                        *added_typical = true;
                    }
                    5 if !*added_temp => {
                        hypura_sys::llama_sampler_chain_add(
                            ptr,
                            hypura_sys::llama_sampler_init_temp(params.temperature),
                        );
                        *added_temp = true;
                    }
                    6 if !*added_pen => {
                        hypura_sys::llama_sampler_chain_add(
                            ptr,
                            hypura_sys::llama_sampler_init_penalties(
                                params.repeat_last_n,
                                params.repeat_penalty,
                                0.0,
                                0.0,
                            ),
                        );
                        *added_pen = true;
                    }
                    7 if !*added_min_p => {
                        hypura_sys::llama_sampler_chain_add(
                            ptr,
                            hypura_sys::llama_sampler_init_min_p(params.min_p, 1),
                        );
                        *added_min_p = true;
                    }
                    _ => {}
                }
            };

            if let Some(order) = sampler_order {
                for id in order {
                    apply_id(
                        *id,
                        &mut added_top_k,
                        &mut added_top_a,
                        &mut added_top_p,
                        &mut added_tfs,
                        &mut added_typical,
                        &mut added_min_p,
                        &mut added_temp,
                        &mut added_pen,
                    );
                }
            }
            apply_id(
                0,
                &mut added_top_k,
                &mut added_top_a,
                &mut added_top_p,
                &mut added_tfs,
                &mut added_typical,
                &mut added_min_p,
                &mut added_temp,
                &mut added_pen,
            );
            apply_id(
                1,
                &mut added_top_k,
                &mut added_top_a,
                &mut added_top_p,
                &mut added_tfs,
                &mut added_typical,
                &mut added_min_p,
                &mut added_temp,
                &mut added_pen,
            );
            apply_id(
                3,
                &mut added_top_k,
                &mut added_top_a,
                &mut added_top_p,
                &mut added_tfs,
                &mut added_typical,
                &mut added_min_p,
                &mut added_temp,
                &mut added_pen,
            );
            apply_id(
                4,
                &mut added_top_k,
                &mut added_top_a,
                &mut added_top_p,
                &mut added_tfs,
                &mut added_typical,
                &mut added_min_p,
                &mut added_temp,
                &mut added_pen,
            );
            apply_id(
                2,
                &mut added_top_k,
                &mut added_top_a,
                &mut added_top_p,
                &mut added_tfs,
                &mut added_typical,
                &mut added_min_p,
                &mut added_temp,
                &mut added_pen,
            );
            apply_id(
                7,
                &mut added_top_k,
                &mut added_top_a,
                &mut added_top_p,
                &mut added_tfs,
                &mut added_typical,
                &mut added_min_p,
                &mut added_temp,
                &mut added_pen,
            );
            apply_id(
                6,
                &mut added_top_k,
                &mut added_top_a,
                &mut added_top_p,
                &mut added_tfs,
                &mut added_typical,
                &mut added_min_p,
                &mut added_temp,
                &mut added_pen,
            );
            apply_id(
                5,
                &mut added_top_k,
                &mut added_top_a,
                &mut added_top_p,
                &mut added_tfs,
                &mut added_typical,
                &mut added_min_p,
                &mut added_temp,
                &mut added_pen,
            );
            hypura_sys::llama_sampler_chain_add(
                ptr,
                hypura_sys::llama_sampler_init_dist(params.seed),
            );
        }

        Self { ptr }
    }

    /// Sample the next token. `idx = -1` means last token in context.
    pub fn sample(&mut self, ctx: &mut LlamaContext, idx: i32) -> i32 {
        unsafe { hypura_sys::llama_sampler_sample(self.ptr, ctx.as_ptr(), idx) }
    }
}

impl Drop for LlamaSampler {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hypura_sys::llama_sampler_free(self.ptr) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_lifecycle() {
        let _backend = LlamaBackend::init();
        // Drop cleans up
    }

    #[test]
    fn test_sampling_params_default() {
        let p = SamplingParams::default();
        assert!(p.temperature > 0.0);
        assert!(p.top_k > 0);
        assert!(p.max_tokens > 0);
    }

    #[test]
    fn triality_config_round_trips_through_c_abi_shape() {
        let branch = |view| TrialityBranchConfig {
            view,
            weight: 1.0 / 3.0,
            bias: 0.0,
            scale: 1.0,
            temperature: 1.0,
            expected_error: 0.125,
            bits_per_channel: 4.125,
        };
        let config = TrialityContextConfig {
            schema_version: 2,
            execution: TrialityExecution::AttentionLogitConsensus,
            layers: vec![TrialityLayerConfig {
                branches: [
                    branch(TrialityView::Vector),
                    branch(TrialityView::SpinorPlusProxy),
                    branch(TrialityView::SpinorMinusProxy),
                ],
                active_branch_mask: 0b111,
            }],
            required: false,
            trace_enabled: true,
            js_fallback_threshold: 0.25,
            allow_identity_view_fallback: false,
        };
        let (layers, native) = native_triality_config(&config).unwrap();
        let restored = rust_triality_config(&native, &layers).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn invalid_triality_branch_numeric_contract_fails_before_ffi() {
        let config = TrialityContextConfig {
            schema_version: 2,
            execution: TrialityExecution::SingleView,
            layers: vec![TrialityLayerConfig {
                branches: [
                    TrialityBranchConfig {
                        view: TrialityView::Vector,
                        weight: 1.0,
                        bias: 0.0,
                        scale: 0.0,
                        temperature: 1.0,
                        expected_error: 0.0,
                        bits_per_channel: 16.0,
                    },
                    TrialityBranchConfig {
                        view: TrialityView::SpinorPlusProxy,
                        weight: 0.0,
                        bias: 0.0,
                        scale: 1.0,
                        temperature: 1.0,
                        expected_error: 0.0,
                        bits_per_channel: 16.0,
                    },
                    TrialityBranchConfig {
                        view: TrialityView::SpinorMinusProxy,
                        weight: 0.0,
                        bias: 0.0,
                        scale: 1.0,
                        temperature: 1.0,
                        expected_error: 0.0,
                        bits_per_channel: 16.0,
                    },
                ],
                active_branch_mask: 1,
            }],
            required: true,
            trace_enabled: false,
            js_fallback_threshold: 0.1,
            allow_identity_view_fallback: false,
        };
        assert!(native_triality_config(&config).is_err());
    }

    #[test]
    fn native_triality_config_canonicalises_branch_slots_and_active_mask() {
        let branch = |view| TrialityBranchConfig {
            view,
            weight: if view == TrialityView::Vector {
                1.0
            } else {
                0.0
            },
            bias: 0.0,
            scale: 1.0,
            temperature: 1.0,
            expected_error: 0.0,
            bits_per_channel: 16.0,
        };
        let config = TrialityContextConfig {
            schema_version: 2,
            execution: TrialityExecution::SingleView,
            layers: vec![TrialityLayerConfig {
                branches: [
                    branch(TrialityView::SpinorMinusProxy),
                    branch(TrialityView::Vector),
                    branch(TrialityView::SpinorPlusProxy),
                ],
                active_branch_mask: 0b010,
            }],
            required: true,
            trace_enabled: false,
            js_fallback_threshold: 0.1,
            allow_identity_view_fallback: false,
        };
        let (layers, _) = native_triality_config(&config).unwrap();
        assert_eq!(
            layers[0].branches[0].view,
            hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_VECTOR
        );
        assert_eq!(
            layers[0].branches[1].view,
            hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_SPINOR_PLUS_PROXY
        );
        assert_eq!(
            layers[0].branches[2].view,
            hypura_sys::llama_tq_view_LLAMA_TQ_VIEW_SPINOR_MINUS_PROXY
        );
        assert_eq!(layers[0].active_mask, 0b001);
    }
}
