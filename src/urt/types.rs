use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationKind {
    PythonQuantisedReference,
    LlamaCpuGguf,
    LlamaCudaGguf,
    HypuraNative,
    HypuraKoboldWorker,
}

impl RepresentationKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PythonQuantisedReference => "python_quantised_reference",
            Self::LlamaCpuGguf => "llama_cpu_gguf",
            Self::LlamaCudaGguf => "llama_cuda_gguf",
            Self::HypuraNative => "hypura_native",
            Self::HypuraKoboldWorker => "hypura_kobold_worker",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RepresentationId {
    pub kind: RepresentationKind,
    pub model_hash: String,
    pub artefact_hash: Option<String>,
    pub backend: String,
    pub precision: String,
    pub view: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UrtObservation {
    pub request_id: String,
    pub representation: RepresentationId,
    pub state_id: String,
    pub layer: Option<u32>,
    pub operator_word: Vec<String>,
    pub operator_word_sha256: String,
    pub observable: String,
    pub value_real: f64,
    pub value_imag: f64,
    pub tolerance: f64,
}
