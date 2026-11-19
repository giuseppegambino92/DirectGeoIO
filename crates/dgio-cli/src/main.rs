//! dgio — inspect / decode / benchmark GRIB2 with DirectGeoIO.
//!
//!   dgio info  <file.grib2>
//!   dgio bench <file.grib2>... [--backend metal|cuda|cpu|cpu-mt|all] [--repeat N] [--fast] [--f16]
//!              [--io span|ranges|auto] [--json]

use directgeoio_core::backend::{AecScan, DType, DecodeOptions, GeoComputeBackend, Timings};
use directgeoio_core::cpu::CpuBackend;
use directgeoio_core::plan::IoStrategy;
use directgeoio_core::{scan, Packing};
use std::path::PathBuf;
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage: dgio info <file> | dgio bench <files>... [--backend metal|cuda|cpu|cpu-mt|all] [--repeat N] [--fast] [--f16] [--io span|ranges|auto] [--cold] [--json]");
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("info") => info(&args[1..]),
        Some("bench") => bench(&args[1..]),
        _ => usage(),
    }
}

fn info(args: &[String]) {
    for a in args {
        let f = std::fs::File::open(a).expect("open");
        let t = Instant::now();
        let fields = scan(&f).expect("scan");
        let dt = t.elapsed();
        println!("{a}: {} fields (metadata scan {:.1} µs)", fields.len(), dt.as_secs_f64() * 1e6);
        for (i, x) in fields.iter().enumerate() {
            let p = x.param();
            let pk = match x.packing {
                Packing::Simple => "5.0 simple".to_string(),
                Packing::Complex(c) => format!("5.{} complex sd{} ng={}", if c.sd_order > 0 { 3 } else { 2 }, c.sd_order, c.ngroups),
                Packing::Ccsds(c) => format!("5.42 ccsds bs={} rsi={} flags={}", c.block_size, c.rsi, c.flags),
                Packing::Unsupported(t) => format!("5.{t} (not supported)"),
            };
            println!(
                "  #{i:<3} {:<6} {:<18} {:>8} {:>3}b {:<28} {}x{} {}{}",
                p.short_name,
                x.level_name(),
                x.product.level(),
                x.scale.nbits,
                pk,
                x.grid.ni,
                x.grid.nj,
                if x.bitmap.is_some() { "bitmap " } else { "" },
                p.units
            );
        }
    }
}

struct Row {
    backend: String,
    file: String,
    t: Timings,
    best_total: f64,
    samples: Vec<f64>,
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    // nearest-rank percentile
    let k = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize - 1;
    sorted[k.min(sorted.len() - 1)]
}

/// Drop the file's pages from the OS page cache (Linux: posix_fadvise DONTNEED, no root needed).
#[cfg(target_os = "linux")]
fn evict(path: &std::path::Path) -> Result<(), String> {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn posix_fadvise(fd: i32, off: i64, len: i64, advice: i32) -> i32;
    }
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let rc = unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, 4 /* POSIX_FADV_DONTNEED */) };
    if rc == 0 { Ok(()) } else { Err(format!("posix_fadvise failed: {rc}")) }
}
#[cfg(not(target_os = "linux"))]
fn evict(_: &std::path::Path) -> Result<(), String> {
    Err("--cold needs Linux (posix_fadvise); on macOS page-cache eviction requires `sudo purge`".into())
}

fn run<B: GeoComputeBackend>(be: &B, files: &[PathBuf], opts: &DecodeOptions, repeat: usize, cold: bool, rows: &mut Vec<Row>) {
    for path in files {
        let mut best: Option<Timings> = None;
        let mut samples = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            if cold {
                evict(path).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) });
            }
            let t0 = Instant::now();
            let fields = scan(&std::fs::File::open(path).unwrap()).unwrap();
            let scan_s = t0.elapsed().as_secs_f64();
            let (_tensor, mut t) = be.decode(path, &fields, opts).unwrap();
            t.plan_s += scan_s;
            t.total_s = t0.elapsed().as_secs_f64();
            samples.push(t.total_s);
            if best.map(|b| t.total_s < b.total_s).unwrap_or(true) {
                best = Some(t);
            }
        }
        let b = best.unwrap();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rows.push(Row { backend: be.name(), file: path.file_name().unwrap().to_string_lossy().into(), t: b, best_total: b.total_s, samples });
    }
}

fn bench(args: &[String]) {
    let mut files = Vec::new();
    let mut backend = "all".to_string();
    let mut repeat = 10usize;
    let mut opts = DecodeOptions::default();
    let mut json = false;
    let mut cold = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => { i += 1; backend = args[i].clone() }
            "--repeat" => { i += 1; repeat = args[i].parse().unwrap() }
            "--fast" => opts.exact = false,
            "--f16" => opts.dtype = DType::F16,
            "--json" => json = true,
            "--cold" => cold = true,
            "--aec-scan" => {
                i += 1;
                opts.aec_scan = match args[i].as_str() { "gpu" => AecScan::Gpu, "cpu" => AecScan::Cpu, _ => AecScan::Auto }
            }
            "--io" => {
                i += 1;
                opts.io = match args[i].as_str() { "span" => IoStrategy::Span, "ranges" => IoStrategy::Ranges, _ => IoStrategy::Auto }
            }
            s if s.starts_with("--") => usage(),
            s => files.push(PathBuf::from(s)),
        }
        i += 1;
    }
    if files.is_empty() { usage() }
    let mut rows = Vec::new();
    let want = |b: &str| backend == "all" || backend == b;
    #[cfg(target_os = "macos")]
    if want("metal") {
        let be = directgeoio_metal::MetalBackend::new().unwrap();
        // warm-up: pipeline-state compilation and pooled buffer allocation
        let f0 = scan(&std::fs::File::open(&files[0]).unwrap()).unwrap();
        let _ = cold;
        let _ = be.decode(&files[0], &f0, &opts).unwrap();
        run(&be, &files, &opts, repeat, cold, &mut rows);
    }
    #[cfg(feature = "cuda")]
    if want("cuda") {
        let be = directgeoio_cuda::CudaBackend::new(0).unwrap();
        let f0 = scan(&std::fs::File::open(&files[0]).unwrap()).unwrap();
        let (t, _) = be.decode(&files[0], &f0, &opts).unwrap();
        eprintln!("cuda: {} (storage path: {})", be.device_name(), t.storage_path.name());
        run(&be, &files, &opts, repeat, cold, &mut rows);
    }
    let cpu_opts = DecodeOptions { dtype: DType::F32, ..opts };
    if want("cpu") { run(&CpuBackend::single(), &files, &cpu_opts, repeat.min(5), cold, &mut rows); }
    if want("cpu-mt") { run(&CpuBackend::multi(), &files, &cpu_opts, repeat.min(5), cold, &mut rows); }
    if json {
        println!("[");
        for (k, r) in rows.iter().enumerate() {
            println!(
                "  {{\"backend\":\"{}\",\"file\":\"{}\",\"total_s\":{:.6},\"median_s\":{:.6},\"p10_s\":{:.6},\"p90_s\":{:.6},\"n\":{},\"cold\":{},\"plan_s\":{:.6},\"io_s\":{:.6},\"compute_s\":{:.6},\"bytes\":{},\"values\":{}}}{}",
                r.backend, r.file, r.best_total, pct(&r.samples, 50.0), pct(&r.samples, 10.0), pct(&r.samples, 90.0), r.samples.len(), cold,
                r.t.plan_s, r.t.io_s, r.t.compute_s, r.t.bytes_read, r.t.values,
                if k + 1 < rows.len() { "," } else { "" }
            );
        }
        println!("]");
        return;
    }
    println!("{:<34} {:<26} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}", "backend", "file", "best ms", "median", "p90", "scan+plan", "io ms", "decode ms", "Mvalues/s");
    for r in &rows {
        let med = pct(&r.samples, 50.0);
        println!(
            "{:<34} {:<26} {:>9.2} {:>9.2} {:>9.2} {:>9.3} {:>9.2} {:>9.2} {:>10.0}",
            r.backend, r.file, r.best_total * 1e3, med * 1e3, pct(&r.samples, 90.0) * 1e3, r.t.plan_s * 1e3, r.t.io_s * 1e3,
            r.t.compute_s * 1e3, r.t.values as f64 / med / 1e6
        );
    }
}
