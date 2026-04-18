fn main() {
    let manifest_src = std::path::PathBuf::from("..")
        .join("..")
        .join("docs")
        .join("compat")
        .join("koboldcpp-assets.json");
    let manifest_dst = std::path::PathBuf::from("resources").join("koboldcpp-assets.json");
    println!("cargo:rerun-if-changed={}", manifest_src.display());
    if let Some(parent) = manifest_dst.parent() {
        std::fs::create_dir_all(parent).expect("create Tauri resource directory");
    }
    if manifest_src.exists() {
        std::fs::copy(&manifest_src, &manifest_dst).expect("copy KoboldCpp asset manifest");
    }
    tauri_build::build()
}
