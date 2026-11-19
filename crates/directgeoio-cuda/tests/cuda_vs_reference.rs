//! CUDA backend validation: every value must be bit-identical to the CPU reference decoder
//! (itself verified bit-exact against ecCodes). Files: fixtures/*.grib2 plus any paths in
//! DGIO_EXTRA_FILES (colon-separated), e.g. the complete operational files.
#![cfg(target_os = "linux")]

use directgeoio_core::backend::{AecScan, DType, DecodeOptions, GeoComputeBackend};
use directgeoio_core::plan::IoStrategy;
use directgeoio_core::{decode, scan};
use directgeoio_cuda::CudaBackend;
use std::path::PathBuf;

fn files() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let mut v: Vec<_> = std::fs::read_dir(&dir)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().map(|x| x == "grib2").unwrap_or(false)).collect())
        .unwrap_or_default();
    v.sort();
    if let Ok(extra) = std::env::var("DGIO_EXTRA_FILES") {
        v.extend(extra.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    assert!(!v.is_empty(), "no GRIB2 fixtures found");
    v
}

fn check(opts: DecodeOptions, label: &str) {
    let be = CudaBackend::new(0).expect("CUDA device");
    let (mut total, mut bad) = (0usize, 0usize);
    for path in files() {
        let bytes = std::fs::read(&path).unwrap();
        let fields = scan(&bytes[..]).unwrap();
        let (t, tm) = be.decode(&path, &fields, &opts).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let got = t.to_vec().unwrap();
        let mut off = 0;
        for (i, f) in fields.iter().enumerate() {
            let want = decode::decode_f64(f, &bytes).unwrap();
            for (j, w) in want.iter().enumerate() {
                let g = got[off + j];
                let ok = if w.is_nan() { g.is_nan() } else { g.to_bits() == (*w as f32).to_bits() };
                if !ok {
                    if bad < 5 {
                        eprintln!("{}#{i}[{j}]: got {g:e} want {:e}", path.display(), *w as f32);
                    }
                    bad += 1;
                }
            }
            off += want.len();
            total += want.len();
        }
        println!("[{label}] {}: {} fields ok, {:?} io {:.2} ms decode {:.2} ms",
                 path.file_name().unwrap().to_string_lossy(), fields.len(), t.storage_path, tm.io_s * 1e3, tm.compute_s * 1e3);
    }
    println!("[{label}] {total} values, {bad} mismatches");
    assert_eq!(bad, 0);
}

#[test]
fn cuda_bit_exact_cpu_walk() {
    check(DecodeOptions { aec_scan: AecScan::Cpu, ..Default::default() }, "cpu-walk");
}

#[test]
fn cuda_bit_exact_gpu_walk_ranges() {
    check(DecodeOptions { aec_scan: AecScan::Gpu, io: IoStrategy::Ranges, ..Default::default() }, "gpu-walk");
}

#[test]
fn cuda_f16_runs() {
    let be = CudaBackend::new(0).unwrap();
    let path = files().into_iter().find(|p| p.to_string_lossy().contains("ccsds_bitmap")).unwrap();
    let fields = scan(&std::fs::File::open(&path).unwrap()).unwrap();
    let (t, _) = be.decode(&path, &fields, &DecodeOptions { dtype: DType::F16, ..Default::default() }).unwrap();
    assert_eq!(t.len, fields.iter().map(|f| f.n_points() as usize).sum::<usize>());
}
