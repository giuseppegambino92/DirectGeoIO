#![cfg(target_os = "linux")]
//! Differential fuzzing CPU <-> CUDA on mutated GRIB2 (seeds: fixtures/edge_*.grib2).
//! Every mutated file is decoded by the CPU reference and by the GPU. Required:
//!   * the GPU never hangs or crashes the process (corrupt input must not read/write out of bounds);
//!   * both paths accept or reject the same inputs;
//!   * when both accept, outputs are bit-identical.
//!   DGIO_GPU_FUZZ_ITERS=20000 cargo test --release -p directgeoio-cuda --test gpu_fuzz -- --nocapture

use directgeoio_core::backend::{DecodeOptions, GeoComputeBackend};
use directgeoio_core::testing::{mutate, seeds, Rng};
use directgeoio_core::{decode, scan};
use directgeoio_cuda::CudaBackend;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[test]
fn gpu_matches_cpu_on_mutated_inputs() {
    let iters: usize = std::env::var("DGIO_GPU_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(3000);
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    let seeds = seeds(&fixtures);
    let be = CudaBackend::new(0).unwrap();
    let tmp = std::env::temp_dir().join(format!("dgio_gpu_fuzz_{}.grib2", std::process::id()));
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let (mut both_ok, mut both_err, mut slow) = (0usize, 0usize, 0usize);
    let mut disagree: Vec<String> = Vec::new();
    for it in 0..iters {
        let seed = &seeds[rng.below(seeds.len())];
        let buf = mutate(&mut rng, seed, &seeds);
        let fields = match scan(&buf[..]) {
            Ok(f) if !f.is_empty() && f.iter().all(|f| f.n_points() <= 4_000_000) => f,
            _ => continue,
        };
        let cpu: Result<Vec<f32>, String> = fields
            .iter()
            .map(|f| decode::decode_f64(f, &buf).map(|v| v.into_iter().map(|x| x as f32).collect::<Vec<_>>()))
            .collect::<Result<Vec<_>, _>>()
            .map(|v| v.concat())
            .map_err(|e| e.to_string());
        std::fs::write(&tmp, &buf).unwrap();
        let t = Instant::now();
        let gpu = be.decode(&tmp, &fields, &DecodeOptions::default()).and_then(|(t, _)| t.to_vec()).map_err(|e| e.to_string());
        if t.elapsed() > Duration::from_secs(2) {
            slow += 1;
        }
        match (&cpu, &gpu) {
            (Ok(c), Ok(g)) => {
                let same = c.len() == g.len() && c.iter().zip(g).all(|(a, b)| (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits());
                if same {
                    both_ok += 1;
                } else {
                    let j = c.iter().zip(g).position(|(a, b)| !((a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()));
                    disagree.push(format!("iter {it}: values differ (len {} vs {}, first at {:?}: cpu {:?} gpu {:?})",
                        c.len(), g.len(), j, j.map(|j| c[j]), j.map(|j| g[j])));
                }
            }
            (Err(_), Err(_)) => both_err += 1,
            (Ok(_), Err(e)) => disagree.push(format!("iter {it}: GPU rejected what CPU accepted: {e}")),
            (Err(e), Ok(_)) => disagree.push(format!("iter {it}: GPU accepted what CPU rejected: {e}")),
        }
        if !disagree.is_empty() && (disagree.len() <= 3 || disagree.last().unwrap().contains("values differ")) && disagree.last().map(|d| d.starts_with(&format!("iter {it}:"))).unwrap_or(false) {
            let p = std::env::temp_dir().join(format!("dgio_gpu_fuzz_case_{it}.grib2"));
            std::fs::write(&p, &buf).ok();
            eprintln!("{}  (input saved to {})", disagree.last().unwrap(), p.display());
        }
    }
    std::fs::remove_file(&tmp).ok();
    println!("{iters} mutated inputs: {both_ok} decoded identically, {both_err} rejected by both, {} disagreements, {slow} slow", disagree.len());
    let mut cats: std::collections::BTreeMap<String, usize> = Default::default();
    for d in &disagree {
        let k: String = d.splitn(2, ": ").nth(1).unwrap_or(d).chars().filter(|c| !c.is_ascii_digit()).take(90).collect();
        *cats.entry(k).or_default() += 1;
    }
    for (k, n) in &cats {
        println!("  x{n:<5} {k}");
    }
    assert!(disagree.is_empty(), "CPU and GPU disagree on {} inputs", disagree.len());
    assert_eq!(slow, 0);
}
