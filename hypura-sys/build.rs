use std::env;
use std::path::PathBuf;

fn resolve_llama_dir(manifest_dir: &str) -> PathBuf {
    let explicit = env::var_os("HYPURA_LLAMA_CPP_PATH").map(PathBuf::from);
    let candidate = explicit
        .clone()
        .unwrap_or_else(|| PathBuf::from(manifest_dir).join("../vendor/llama.cpp"));

    dunce::canonicalize(&candidate).unwrap_or_else(|_| {
        if explicit.is_some() {
            panic!(
                "HYPURA_LLAMA_CPP_PATH not found or invalid: {}",
                candidate.display()
            );
        }
        panic!(
            "vendor/llama.cpp not found: {} (run `git submodule update --init --recursive` or set HYPURA_LLAMA_CPP_PATH)",
            candidate.display()
        );
    })
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let llama_dir = resolve_llama_dir(&manifest_dir);

    println!("cargo:rerun-if-env-changed=HYPURA_LLAMA_CPP_PATH");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let use_metal = target_os == "macos";
    let use_cuda = !use_metal && cuda_is_available();

    let mut cmake_config = cmake::Config::new(&llama_dir);
    if target_os == "windows" && env::var("CMAKE_GENERATOR").is_err() {
        cmake_config.generator("Visual Studio 17 2022");
    }
    if target_os == "windows" {
        if let Some(masm) = find_masm() {
            let masm = masm.display().to_string();
            cmake_config.define("CMAKE_ASM_COMPILER", &masm);
            cmake_config.define("CMAKE_ASM_MASM_COMPILER", masm);
        }
    }
    cmake_config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("GGML_CPU", "ON")
        .define("GGML_BLAS", "OFF")
        .define("GGML_CPU_REPACK", "OFF");

    if use_metal {
        cmake_config
            .define("GGML_METAL", "ON")
            .define("GGML_METAL_EMBED_LIBRARY", "ON")
            .define("GGML_CUDA", "OFF")
            .define("GGML_OPENMP", "OFF");
    } else if use_cuda {
        let cuda_arches =
            env::var("HYPURA_CUDA_ARCHITECTURES").unwrap_or_else(|_| "75;86;89;90".to_string());

        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "ON")
            .define("GGML_OPENMP", "ON")
            .define("CMAKE_CUDA_ARCHITECTURES", cuda_arches);

        if let Some(nvcc) = find_nvcc() {
            cmake_config.define("CMAKE_CUDA_COMPILER", nvcc.display().to_string());
        }
        if let Some(cuda_root) = get_cuda_root() {
            cmake_config.define("CUDAToolkit_ROOT", cuda_root.display().to_string());
        }
    } else {
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF")
            .define("GGML_OPENMP", "ON");
    }

    let dst = cmake_config.build();
    let lib_dir = dst.join("lib");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    if use_metal {
        println!("cargo:rustc-link-lib=static=ggml-metal");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    } else if use_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        if let Some(lib_path) = get_cuda_lib_path() {
            println!("cargo:rustc-link-search=native={}", lib_path.display());
        }
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        if target_os == "linux" {
            println!("cargo:rustc-link-lib=stdc++");
        }
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
    }

    if use_metal {
        println!("cargo:rustc-cfg=hypura_metal");
    } else if use_cuda {
        println!("cargo:rustc-cfg=hypura_cuda");
    }

    let src_dir = PathBuf::from(&manifest_dir).join("src");
    let include_ggml_internal = llama_dir.join("ggml/src");

    let mut cc_build = cc::Build::new();
    cc_build
        .file(src_dir.join("hypura_buft.c"))
        .file(src_dir.join("hypura_kv_codec.c"))
        .file(src_dir.join("hypura_sampler_ext.c"))
        .include(llama_dir.join("include"))
        .include(llama_dir.join("ggml/include"))
        .include(&include_ggml_internal)
        .include(&src_dir);

    if target_os != "windows" {
        cc_build.flag("-std=c11");
    }
    cc_build.compile("hypura_buft");

    println!("cargo:rerun-if-changed=src/hypura_buft.c");
    println!("cargo:rerun-if-changed=src/hypura_buft.h");
    println!("cargo:rerun-if-changed=src/hypura_kv_codec.c");
    println!("cargo:rerun-if-changed=src/hypura_kv_codec.h");
    println!("cargo:rerun-if-changed=src/hypura_sampler_ext.c");
    println!("cargo:rerun-if-changed=src/hypura_sampler_ext.h");

    let include_llama = llama_dir.join("include");
    let include_ggml = llama_dir.join("ggml/include");

    let bindings = bindgen::Builder::default()
        .header(
            PathBuf::from(&manifest_dir)
                .join("wrapper.h")
                .to_str()
                .unwrap()
                .to_string(),
        )
        .clang_arg(format!("-I{}", include_llama.display()))
        .clang_arg(format!("-I{}", include_ggml.display()))
        .clang_arg(format!("-I{}", src_dir.display()))
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_function("hypura_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_type("hypura_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .allowlist_var("GGUF_.*")
        .derive_debug(true)
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .expect("Failed to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("include").display()
    );
    println!("cargo:rerun-if-changed={}", llama_dir.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("ggml").display()
    );
}

fn cuda_is_available() -> bool {
    if env::var("HYPURA_NO_CUDA").is_ok() {
        return false;
    }
    if env::var("HYPURA_CUDA").is_ok() {
        return true;
    }
    get_cuda_root().is_some()
}

fn get_cuda_root() -> Option<PathBuf> {
    if let Ok(p) = env::var("CUDA_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }

    for candidate in &["/usr/local/cuda", "/usr/cuda"] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }

    if let Some(nvcc) = find_nvcc() {
        if let Some(bin) = nvcc.parent() {
            if let Some(root) = bin.parent() {
                return Some(root.to_path_buf());
            }
        }
    }

    None
}

fn get_cuda_lib_path() -> Option<PathBuf> {
    let root = get_cuda_root()?;
    for sub in &["lib64", "lib/x64", "lib"] {
        let p = root.join(sub);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn find_nvcc() -> Option<PathBuf> {
    let candidates = ["/usr/local/cuda/bin/nvcc", "/usr/cuda/bin/nvcc"];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }

    let ok = std::process::Command::new("nvcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        return Some(PathBuf::from("nvcc"));
    }

    None
}

fn find_masm() -> Option<PathBuf> {
    if let Ok(dir) = env::var("VCToolsInstallDir") {
        let candidate = PathBuf::from(dir).join("bin/Hostx64/x64/ml64.exe");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let roots = [
        PathBuf::from(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        ),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
    ];

    for root in roots {
        let mut versions: Vec<PathBuf> = match std::fs::read_dir(&root) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .collect(),
            Err(_) => continue,
        };
        versions.sort();
        versions.reverse();

        for version_dir in versions {
            let candidate = version_dir.join("bin/Hostx64/x64/ml64.exe");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}

fn find_masm() -> Option<PathBuf> {
    if let Ok(dir) = env::var("VCToolsInstallDir") {
        let candidate = PathBuf::from(dir).join("bin/Hostx64/x64/ml64.exe");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    let roots = [
        PathBuf::from(
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        ),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
    ];

    for root in roots {
        let mut versions: Vec<PathBuf> = match std::fs::read_dir(&root) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .collect(),
            Err(_) => continue,
        };
        versions.sort();
        versions.reverse();

        for version_dir in versions {
            let candidate = version_dir.join("bin/Hostx64/x64/ml64.exe");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}
