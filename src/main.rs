mod cli;

use clap::{Parser, Subcommand};
use hypura::model::turboquant_sidecar::{RotationPolicy, TurboQuantMode};
use hypura::scheduler::types::{HostPinnedPolicy, ResidencyProfile};

#[derive(Parser)]
#[command(
    name = "hypura",
    version,
    about = "Storage-tier-aware LLM inference scheduler"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run hardware profiler and save results
    Profile {
        /// Force re-profiling even if a recent profile exists
        #[arg(long)]
        force: bool,
    },
    /// Show performance estimate for a model without loading it
    Estimate {
        /// Path to model file or HuggingFace model ID
        model: String,
    },
    /// Load model with tiered scheduling and run inference
    Run {
        /// Path to model file
        model: String,
        /// Maximum context length
        #[arg(long, default_value = "4096")]
        context: u32,
        /// Single prompt (non-interactive mode)
        #[arg(long)]
        prompt: Option<String>,
        /// Interactive chat mode
        #[arg(long)]
        interactive: bool,
        /// Maximum tokens to generate
        #[arg(long, default_value = "512")]
        max_tokens: u32,
        /// TurboQuant runtime mode
        #[arg(long, value_enum, default_value_t = TurboQuantMode::ResearchKvSplit)]
        turboquant_mode: TurboQuantMode,
        /// Optional TurboQuant sidecar config path
        #[arg(long)]
        turboquant_config: Option<String>,
        /// Rotation policy for TurboQuant (default: Triality + SO(8) vector view)
        #[arg(long, value_enum, default_value_t = RotationPolicy::TrialityVector)]
        rotation_policy: RotationPolicy,
        /// Rotation seed for deterministic rotation
        #[arg(long, default_value = "0")]
        rotation_seed: u32,
        /// Residency comparison profile
        #[arg(long, value_enum, default_value_t = ResidencyProfile::FourTier)]
        residency_profile: ResidencyProfile,
        /// Host pinned tier policy
        #[arg(long, value_enum, default_value_t = HostPinnedPolicy::Auto)]
        host_pinned: HostPinnedPolicy,
    },
    /// Start Ollama-compatible API server
    Serve {
        /// Path to model file
        model: String,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind to
        #[arg(long, default_value = "8080")]
        port: u16,
        /// Maximum context length
        #[arg(long, default_value = "4096")]
        context: u32,
        /// TurboQuant runtime mode
        #[arg(long, value_enum, default_value_t = TurboQuantMode::ResearchKvSplit)]
        turboquant_mode: TurboQuantMode,
        /// Optional TurboQuant sidecar config path
        #[arg(long)]
        turboquant_config: Option<String>,
        /// Rotation policy for TurboQuant (default: Triality + SO(8) vector view)
        #[arg(long, value_enum, default_value_t = RotationPolicy::TrialityVector)]
        rotation_policy: RotationPolicy,
        /// Rotation seed for deterministic rotation
        #[arg(long, default_value = "0")]
        rotation_seed: u32,
        /// Disable SO8 runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_so8_off: bool,
        /// Enable learned SO8 runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_so8_learned: bool,
        /// Disable Triality runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_triality_off: bool,
        /// Triality mix coefficient for TurboQuant env bridge [0,1]
        #[arg(long, default_value = "0.5")]
        tq_triality_mix: f32,
        /// Rotation seed for TurboQuant runtime env bridge
        #[arg(long, default_value = "0")]
        tq_rotation_seed: u32,
        /// Optional TurboQuant artifact path for runtime env bridge
        #[arg(long)]
        tq_artifact: Option<String>,
        /// Optional model directory used by Kobold-lite model selector
        #[arg(long)]
        model_dir: Option<String>,
        /// Optional Kobold-lite theme hint (stored in env for UI)
        #[arg(long, default_value = "classic")]
        ui_theme: String,
        /// Resolve Triality/TurboQuant runtime wiring without loading the model server
        #[arg(long)]
        dry_run: bool,
        /// Residency comparison profile
        #[arg(long, value_enum, default_value_t = ResidencyProfile::FourTier)]
        residency_profile: ResidencyProfile,
        /// Host pinned tier policy
        #[arg(long, value_enum, default_value_t = HostPinnedPolicy::Auto)]
        host_pinned: HostPinnedPolicy,
    },
    /// Start a KoboldCpp-compatible server profile with Kobold-style defaults
    #[command(name = "koboldcpp", visible_alias = "compat")]
    Koboldcpp {
        /// Path to model file
        model: String,
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to bind to
        #[arg(long, default_value = "5001")]
        port: u16,
        /// Maximum context length
        #[arg(long, default_value = "4096")]
        context: u32,
        /// Default max_length reported through Kobold compatibility routes
        #[arg(long, default_value = "256")]
        max_length: u32,
        /// TurboQuant runtime mode
        #[arg(long, value_enum, default_value_t = TurboQuantMode::ResearchKvSplit)]
        turboquant_mode: TurboQuantMode,
        /// Optional TurboQuant sidecar config path
        #[arg(long)]
        turboquant_config: Option<String>,
        /// Rotation policy for TurboQuant (default: Triality + SO(8) vector view)
        #[arg(long, value_enum, default_value_t = RotationPolicy::TrialityVector)]
        rotation_policy: RotationPolicy,
        /// Rotation seed for deterministic rotation
        #[arg(long, default_value = "0")]
        rotation_seed: u32,
        /// Disable SO8 runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_so8_off: bool,
        /// Enable learned SO8 runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_so8_learned: bool,
        /// Disable Triality runtime path for TurboQuant env bridge
        #[arg(long)]
        tq_triality_off: bool,
        /// Triality mix coefficient for TurboQuant env bridge [0,1]
        #[arg(long, default_value = "0.5")]
        tq_triality_mix: f32,
        /// Rotation seed for TurboQuant runtime env bridge
        #[arg(long, default_value = "0")]
        tq_rotation_seed: u32,
        /// Optional TurboQuant artifact path for runtime env bridge
        #[arg(long)]
        tq_artifact: Option<String>,
        /// Optional model directory used by Kobold-lite model selector
        #[arg(long)]
        model_dir: Option<String>,
        /// Optional KoboldCpp-style remote SaveData bridge file (.jsondb)
        #[arg(long)]
        savedatafile: Option<String>,
        /// Optional dedicated embeddings GGUF model used by /v1/embeddings
        #[arg(long)]
        embeddings_model: Option<String>,
        /// Optional Kobold story JSON exposed through /api/extra/preloadstory
        #[arg(long)]
        preloadstory: Option<String>,
        /// Optional directory containing .kcpps/.kcppt/.gguf admin profiles
        #[arg(long)]
        admindir: Option<String>,
        /// Optional .kcpps config imported into the active compat profile
        #[arg(long)]
        config: Option<String>,
        /// Optional output path to export the active compat launcher config
        #[arg(long)]
        exportconfig: Option<String>,
        /// Optional directory used to import existing KoboldCpp assets on startup
        #[arg(long)]
        migration_dir: Option<String>,
        /// Optional asset root override for first-run bootstrap placement
        #[arg(long)]
        asset_root: Option<String>,
        /// Optional Kobold-lite theme hint
        #[arg(long, default_value = "classic")]
        ui_theme: String,
        /// Do not auto-open Kobold-lite in the browser
        #[arg(long)]
        no_show_gui: bool,
        /// Resolve Triality/TurboQuant runtime wiring without loading the model server
        #[arg(long)]
        dry_run: bool,
        /// Residency comparison profile
        #[arg(long, value_enum, default_value_t = ResidencyProfile::FourTier)]
        residency_profile: ResidencyProfile,
        /// Host pinned tier policy
        #[arg(long, value_enum, default_value_t = HostPinnedPolicy::Auto)]
        host_pinned: HostPinnedPolicy,
    },
    /// Internal hidden worker mode for the KoboldCpp supervisor process
    #[command(name = "__koboldcpp_worker", hide = true)]
    KoboldcppWorker {
        /// Path to the serialized worker bootstrap payload
        #[arg(long)]
        bootstrap_file: String,
    },
    /// Benchmark tok/s: Hypura scheduling vs naive mmap
    Bench {
        /// Path to model file
        model: String,
        /// Also benchmark with naive mmap for comparison
        #[arg(long)]
        baseline: bool,
        /// Maximum context length
        #[arg(long, default_value = "2048")]
        context: u32,
        /// Tokens to generate per run
        #[arg(long, default_value = "128")]
        max_tokens: u32,
        /// Prompt text
        #[arg(long)]
        prompt: Option<String>,
        /// Force unsafe operations (e.g. baseline with model larger than RAM)
        #[arg(long)]
        force: bool,
        /// TurboQuant runtime mode for the Hypura run
        #[arg(long, value_enum, default_value_t = TurboQuantMode::ResearchKvSplit)]
        turboquant_mode: TurboQuantMode,
        /// Optional TurboQuant sidecar config path
        #[arg(long)]
        turboquant_config: Option<String>,
        /// Rotation policy for TurboQuant (default: Triality + SO(8) vector view)
        #[arg(long, value_enum, default_value_t = RotationPolicy::TrialityVector)]
        rotation_policy: RotationPolicy,
        /// Rotation seed for deterministic rotation
        #[arg(long, default_value = "0")]
        rotation_seed: u32,
        /// Resolve runtime and print the benchmark plan without loading the model
        #[arg(long)]
        dry_run: bool,
        /// Run a single residency profile instead of the default comparison trio
        #[arg(long, value_enum)]
        residency_profile: Option<ResidencyProfile>,
        /// Override the host pinned policy for the selected benchmark profile
        #[arg(long, value_enum)]
        host_pinned: Option<HostPinnedPolicy>,
    },
    /// Print model metadata, tensor list, and placement plan
    Inspect {
        /// Path to model file
        model: String,
        /// Show individual tensor details
        #[arg(long)]
        tensors: bool,
    },
    /// Low-level NVMe I/O microbenchmark (diagnostic)
    Iobench {
        /// Path to a GGUF model file
        model: String,
        /// Amount of data to read in each test (GiB)
        #[arg(long, default_value = "1.0")]
        read_gb: f64,
    },
    /// (MoE only) Reorganize expert layout on disk for sequential access
    Optimize {
        /// Path to model file
        model: String,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("HYPURA_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Profile { force } => cli::profile::run(force),
        Commands::Estimate { model } => cli::estimate::run(&model),
        Commands::Run {
            model,
            context,
            prompt,
            interactive,
            max_tokens,
            turboquant_mode,
            turboquant_config,
            rotation_policy,
            rotation_seed,
            residency_profile,
            host_pinned,
        } => cli::run::run(
            &model,
            context,
            prompt.as_deref(),
            interactive,
            max_tokens,
            turboquant_mode,
            turboquant_config.as_deref(),
            rotation_policy,
            rotation_seed,
            residency_profile,
            host_pinned,
        ),
        Commands::Serve {
            model,
            host,
            port,
            context,
            turboquant_mode,
            turboquant_config,
            rotation_policy,
            rotation_seed,
            tq_so8_off,
            tq_so8_learned,
            tq_triality_off,
            tq_triality_mix,
            tq_rotation_seed,
            tq_artifact,
            model_dir,
            ui_theme,
            dry_run,
            residency_profile,
            host_pinned,
        } => cli::serve::run(
            &model,
            &host,
            port,
            context,
            turboquant_mode,
            turboquant_config.as_deref(),
            rotation_policy,
            rotation_seed,
            tq_so8_off,
            tq_so8_learned,
            tq_triality_off,
            tq_triality_mix,
            tq_rotation_seed,
            tq_artifact.as_deref(),
            model_dir.as_deref(),
            &ui_theme,
            dry_run,
            residency_profile,
            host_pinned,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        Commands::Koboldcpp {
            model,
            host,
            port,
            context,
            max_length,
            turboquant_mode,
            turboquant_config,
            rotation_policy,
            rotation_seed,
            tq_so8_off,
            tq_so8_learned,
            tq_triality_off,
            tq_triality_mix,
            tq_rotation_seed,
            tq_artifact,
            model_dir,
            savedatafile,
            embeddings_model,
            preloadstory,
            admindir,
            config,
            exportconfig,
            migration_dir,
            asset_root,
            ui_theme,
            no_show_gui,
            dry_run,
            residency_profile,
            host_pinned,
        } => cli::koboldcpp::run(
            &model,
            &host,
            port,
            context,
            max_length,
            turboquant_mode,
            turboquant_config.as_deref(),
            rotation_policy,
            rotation_seed,
            tq_so8_off,
            tq_so8_learned,
            tq_triality_off,
            tq_triality_mix,
            tq_rotation_seed,
            tq_artifact.as_deref(),
            model_dir.as_deref(),
            savedatafile.as_deref(),
            embeddings_model.as_deref(),
            preloadstory.as_deref(),
            admindir.as_deref(),
            config.as_deref(),
            exportconfig.as_deref(),
            migration_dir.as_deref(),
            asset_root.as_deref(),
            &ui_theme,
            no_show_gui,
            dry_run,
            residency_profile,
            host_pinned,
        ),
        Commands::KoboldcppWorker { bootstrap_file } => {
            cli::serve::run_worker_bootstrap(&bootstrap_file)
        }
        Commands::Bench {
            model,
            baseline,
            context,
            max_tokens,
            prompt,
            force,
            turboquant_mode,
            turboquant_config,
            rotation_policy,
            rotation_seed,
            dry_run,
            residency_profile,
            host_pinned,
        } => cli::bench::run(
            &model,
            baseline,
            context,
            max_tokens,
            prompt.as_deref(),
            force,
            turboquant_mode,
            turboquant_config.as_deref(),
            rotation_policy,
            rotation_seed,
            dry_run,
            residency_profile,
            host_pinned,
        ),
        Commands::Inspect { model, tensors } => cli::inspect::run(&model, tensors),
        Commands::Iobench { model, read_gb } => cli::iobench::run(&model, read_gb),
        Commands::Optimize { model } => cli::optimize::run(&model),
    }
}
