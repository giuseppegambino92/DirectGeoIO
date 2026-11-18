// Builds kernels/cuda/host.cu (which #includes every kernel) with nvcc into a static library.
// Linux only; on other targets the crate is empty.
//
//   CUDA_PATH / CUDA_HOME   toolkit location (default /usr/local/cuda)
//   DGIO_CUDA_ARCHS         e.g. "80;86;89;90;120"   (default: all of these the nvcc supports)
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    println!("cargo:rerun-if-changed={}", root.join("kernels/cuda").display());
    println!("cargo:rerun-if-changed={}", root.join("kernels/dgio_shared.h").display());
    println!("cargo:rerun-if-env-changed=DGIO_CUDA_ARCHS");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let cuda = std::env::var("CUDA_PATH").or_else(|_| std::env::var("CUDA_HOME")).unwrap_or_else(|_| "/usr/local/cuda".into());
    let nvcc = PathBuf::from(&cuda).join("bin/nvcc");
    let nvcc = if nvcc.exists() { nvcc } else { PathBuf::from("nvcc") };
    let supported = Command::new(&nvcc)
        .arg("--list-gpu-arch")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_else(|_| panic!("nvcc not found (set CUDA_PATH); required for --features cuda"));
    let wanted = std::env::var("DGIO_CUDA_ARCHS").unwrap_or_else(|_| "80;86;89;90;100;120".into());
    let mut gencode = Vec::new();
    let mut last = None;
    for a in wanted.split(';').filter(|s| !s.is_empty()) {
        if supported.contains(&format!("compute_{a}")) {
            gencode.push(format!("-gencode=arch=compute_{a},code=sm_{a}"));
            last = Some(a.to_string());
        }
    }
    if let Some(a) = last {
        gencode.push(format!("-gencode=arch=compute_{a},code=compute_{a}")); // PTX for future GPUs
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("dgio_cuda.o");
    let st = Command::new(&nvcc)
        .args(["-O3", "-std=c++17", "--fmad=false", "-Xcompiler", "-fPIC", "-c"])
        .args(&gencode)
        .arg(root.join("kernels/cuda/host.cu"))
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to run nvcc");
    assert!(st.success(), "nvcc failed");
    let lib = out.join("libdgio_cuda.a");
    let st = Command::new("ar").arg("crs").arg(&lib).arg(&obj).status().unwrap();
    assert!(st.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=dgio_cuda");
    for d in ["lib64", "lib", "targets/x86_64-linux/lib", "targets/sbsa-linux/lib"] {
        println!("cargo:rustc-link-search=native={}/{d}", cuda);
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=dl");
}
