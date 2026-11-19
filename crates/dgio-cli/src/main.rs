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
    eprintln!("usage: dgio info <file> | dgio bench <files>... [--backend metal|cuda|cpu|cpu-mt|all] [--repeat N] [--fast] [--f16] [--io span|ranges|auto] [--json]");
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
}

fn run<B: GeoComputeBackend>(be: &B, files: &[PathBuf], opts: &DecodeOptions, repeat: usize, rows: &mut Vec<Row>) {
    for path in files {
        let mut best: Option<Timings> = None;
        for _ in 0..repeat {
            let t0 = Instant::now();
            let fields = scan(&std::fs::File::open(path).unwrap()).unwrap();
            let scan_s = t0.elapsed().as_secs_f64();
            let (_tensor, mut t) = be.decode(path, &fields, opts).unwrap();
            t.plan_s += scan_s;
            t.total_s = t0.elapsed().as_secs_f64();
            if best.map(|b| t.total_s < b.total_s).unwrap_or(true) {
                best = Some(t);
            }
        }
        let b = best.unwrap();
        rows.push(Row { backend: be.name(), file: path.file_name().unwrap().to_string_lossy().into(), t: b, best_total: b.total_s });
    }
}

fn bench(args: &[String]) {
    let mut files = Vec::new();
    let mut backend = "all".to_string();
    let mut repeat = 10usize;
    let mut opts = DecodeOptions::default();
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => { i += 1; backend = args[i].clone() }
            "--repeat" => { i += 1; repeat = args[i].parse().unwrap() }
            "--fast" => opts.exact = false,
            "--f16" => opts.dtype = DType::F16,
            "--json" => json = true,
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
        let _ = be.decode(&files[0], &f0, &opts).unwrap();
        run(&be, &files, &opts, repeat, &mut rows);
    }
    #[cfg(feature = "cuda")]
    if want("cuda") {
        let be = directgeoio_cuda::CudaBackend::new(0).unwrap();
        let f0 = scan(&std::fs::File::open(&files[0]).unwrap()).unwrap();
        let (t, _) = be.decode(&files[0], &f0, &opts).unwrap();
        eprintln!("cuda: {} (storage path: {})", be.device_name(), t.storage_path.name());
        run(&be, &files, &opts, repeat, &mut rows);
    }
    let cpu_opts = DecodeOptions { dtype: DType::F32, ..opts };
    if want("cpu") { run(&CpuBackend::single(), &files, &cpu_opts, repeat.min(5), &mut rows); }
    if want("cpu-mt") { run(&CpuBackend::multi(), &files, &cpu_opts, repeat.min(5), &mut rows); }
    if json {
        println!("[");
        for (k, r) in rows.iter().enumerate() {
            println!(
                "  {{\"backend\":\"{}\",\"file\":\"{}\",\"total_s\":{:.6},\"plan_s\":{:.6},\"io_s\":{:.6},\"compute_s\":{:.6},\"bytes\":{},\"values\":{}}}{}",
                r.backend, r.file, r.best_total, r.t.plan_s, r.t.io_s, r.t.compute_s, r.t.bytes_read, r.t.values,
                if k + 1 < rows.len() { "," } else { "" }
            );
        }
        println!("]");
        return;
    }
    println!("{:<34} {:<26} {:>9} {:>9} {:>9} {:>9} {:>10} {:>9}", "backend", "file", "total ms", "scan+plan", "io ms", "decode ms", "Mvalues/s", "GB/s in");
    for r in &rows {
        println!(
            "{:<34} {:<26} {:>9.2} {:>9.3} {:>9.2} {:>9.2} {:>10.0} {:>9.2}",
            r.backend, r.file, r.best_total * 1e3, r.t.plan_s * 1e3, r.t.io_s * 1e3, r.t.compute_s * 1e3,
            r.t.values as f64 / r.best_total / 1e6, r.t.bytes_read as f64 / r.best_total / 1e9
        );
    }
}
