//! End-to-end GPU validation: MTLIO storage->GPU load + Metal decode, compared bit-for-bit with
//! float32(ecCodes float64) for every fixture message.

use directgeoio_core::backend::{DType, DecodeOptions, GeoComputeBackend};
use directgeoio_core::plan::IoStrategy;
use directgeoio_core::{npy, scan};
use directgeoio_metal::MetalBackend;
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn files() -> Vec<PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(fixtures())
        .expect("fixtures/ missing: run `make fixtures`")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "grib2").unwrap_or(false))
        .collect();
    v.sort();
    v
}

fn check(io: IoStrategy, exact: bool) -> (usize, usize, f64) {
    let be = MetalBackend::new().unwrap();
    let (mut total, mut mismatched, mut max_rel) = (0usize, 0usize, 0f64);
    for path in files() {
        let fields = scan(&std::fs::File::open(&path).unwrap()).unwrap();
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        let opts = DecodeOptions { io, exact, ..Default::default() };
        let (t, _) = be.decode(&path, &fields, &opts).unwrap_or_else(|e| panic!("{name}: {e}"));
        let got = t.as_f32();
        let mut off = 0;
        for (i, f) in fields.iter().enumerate() {
            let want = npy::read_f64(&fixtures().join(format!("ref/{name}.{i}.npy"))).unwrap();
            let n = f.n_points() as usize;
            let mut bad = 0;
            for j in 0..n {
                let g = got[off + j];
                let w = want[j];
                if w.is_nan() {
                    assert!(g.is_nan(), "{name}#{i}[{j}] expected masked NaN, got {g}");
                    continue;
                }
                let w32 = w as f32;
                if g.to_bits() != w32.to_bits() {
                    bad += 1;
                    let rel = ((g as f64 - w) / w.abs().max(1e-30)).abs();
                    max_rel = max_rel.max(rel);
                    if exact && bad <= 3 {
                        eprintln!("{name}#{i}[{j}] got {g:e} ({:#x}) want {w32:e} ({:#x}) f64 {w:e}", g.to_bits(), w32.to_bits());
                    }
                }
            }
            mismatched += bad;
            total += n;
            off += n;
        }
    }
    (total, mismatched, max_rel)
}

#[test]
fn metal_exact_matches_eccodes_bitwise_span_io() {
    let (n, bad, _) = check(IoStrategy::Span, true);
    println!("span io, exact: {n} values, {bad} float32 mismatches");
    assert_eq!(bad, 0);
}

#[test]
fn metal_exact_matches_eccodes_bitwise_range_io() {
    let (n, bad, _) = check(IoStrategy::Ranges, true);
    println!("range io, exact: {n} values, {bad} float32 mismatches");
    assert_eq!(bad, 0);
}

#[test]
fn metal_fast_mode_within_one_ulp_class() {
    let (n, bad, rel) = check(IoStrategy::Auto, false);
    println!("fast dequant: {n} values, {bad} differ from float32(ecCodes), max rel err {rel:e}");
    assert!(rel < 2.5e-7, "fast mode relative error {rel}");
}

#[test]
fn metal_f16_output() {
    let be = MetalBackend::new().unwrap();
    let path = fixtures().join("ecmwf_ccsds_t_pl.grib2");
    let fields = scan(&std::fs::File::open(&path).unwrap()).unwrap();
    let (t32, _) = be.decode(&path, &fields, &DecodeOptions::default()).unwrap();
    let (t16, _) = be.decode(&path, &fields, &DecodeOptions { dtype: DType::F16, ..Default::default() }).unwrap();
    assert_eq!(t16.shape, vec![8, 721, 1440]);
    let a = t32.as_f32();
    let h = t16.as_u16();
    for j in (0..a.len()).step_by(997) {
        let back = f16_to_f32(h[j]);
        assert!(((back - a[j]) / a[j]).abs() < 1e-3, "{j}: {back} vs {}", a[j]);
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 { s } else if e == 31 { s | 0x7f80_0000 | (m << 13) } else { s | ((e + 112) << 23) | (m << 13) };
    f32::from_bits(bits)
}
