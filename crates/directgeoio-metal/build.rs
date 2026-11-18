// Compiles kernels/metal/*.metal -> directgeoio.metallib (embedded with include_bytes!) and the
// Objective-C Metal shim. Fast-math is disabled so the float-float dequantisation is exact.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let kdir = root.join("kernels/metal");
    println!("cargo:rerun-if-changed={}", kdir.display());
    println!("cargo:rerun-if-changed={}", root.join("kernels/dgio_shared.h").display());
    println!("cargo:rerun-if-changed=src/shim.m");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        // Non-Apple targets build an empty crate (see lib.rs cfg gates).
        return;
    }

    let mut airs = Vec::new();
    let mut srcs: Vec<_> = std::fs::read_dir(&kdir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "metal").unwrap_or(false))
        .collect();
    srcs.sort();
    for src in srcs {
        let air = out.join(src.file_stem().unwrap()).with_extension("air");
        let st = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-std=metal3.2", "-fmetal-math-mode=safe", "-ffp-contract=off", "-O3", "-c"])
            .arg(&src)
            .arg("-o")
            .arg(&air)
            .status()
            .expect("xcrun metal not found (install Xcode command line tools + Metal toolchain)");
        assert!(st.success(), "metal compile failed: {}", src.display());
        airs.push(air);
    }
    let lib = out.join("directgeoio.metallib");
    let st = Command::new("xcrun").args(["-sdk", "macosx", "metallib"]).args(&airs).arg("-o").arg(&lib).status().unwrap();
    assert!(st.success(), "metallib link failed");

    cc::Build::new()
        .file("src/shim.m")
        .flag("-fobjc-arc")
        .flag("-mmacosx-version-min=14.0")
        .flag("-Wno-deprecated-declarations")
        .compile("dgio_metal_shim");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
}
