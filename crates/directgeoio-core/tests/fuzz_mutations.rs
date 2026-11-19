//! Deterministic mutation fuzzing on stable Rust (cargo-fuzz targets live in `fuzz/` for nightly).
//!
//! Seeds: fixtures/edge_*.grib2 (small ecCodes-encoded messages covering every supported
//! template, with and without bitmaps). Each mutated buffer goes through the metadata scanner,
//! `Field::validate`, the CPU decoders and the GPU planner inside `catch_unwind`. Any panic, or
//! a single decode taking longer than 2 s (a sign of a non-terminating loop), fails the test.
//!
//!   DGIO_FUZZ_ITERS=200000 cargo test --release -p directgeoio-core --test fuzz_mutations -- --nocapture

use directgeoio_core::plan::{self, IoStrategy};
use directgeoio_core::{decode, scan};
use std::collections::BTreeMap;
use std::panic;
use directgeoio_core::testing::{mutate, seeds, Rng};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[test]
fn mutated_grib2_never_panics_or_hangs() {
    let iters: usize = std::env::var("DGIO_FUZZ_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seeds = seeds(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"));
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    panic::set_hook(Box::new(|_| {}));
    let (mut ok, mut rejected, mut slow) = (0usize, 0usize, 0usize);
    let mut panics: BTreeMap<String, (usize, Vec<u8>)> = BTreeMap::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for _ in 0..iters {
        let seed = &seeds[rng.below(seeds.len())];
        let buf = mutate(&mut rng, seed, &seeds);
        let t = Instant::now();
        let r = panic::catch_unwind(|| -> Result<usize, String> {
            let fields = scan(&buf[..]).map_err(|e| e.to_string())?;
            let mut decoded = 0;
            for f in &fields {
                f.validate().map_err(|e| e.to_string())?;
                // harness memory cap: validated-but-huge fields (e.g. a constant field claiming
                // 10^9 points) are legal; decoding them here would only test the allocator
                if f.n_points() > 4_000_000 {
                    continue;
                }
                decode::decode_f64(f, &buf).map_err(|e| e.to_string())?;
                decoded += 1;
            }
            plan::build(&fields, IoStrategy::Ranges).map_err(|e| e.to_string())?;
            Ok(decoded)
        });
        if t.elapsed() > Duration::from_secs(2) {
            slow += 1;
        }
        match r {
            Ok(Ok(_)) => ok += 1,
            Ok(Err(e)) => {
                rejected += 1;
                let key: String = e.splitn(2, ": ").nth(1).unwrap_or(&e).chars().take(60).collect();
                *reasons.entry(key).or_default() += 1;
            }
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                let e = panics.entry(msg.chars().take(120).collect()).or_insert((0, buf.clone()));
                e.0 += 1;
            }
        }
    }
    let _ = panic::take_hook();
    println!("{iters} mutated inputs: {ok} decoded, {rejected} rejected with an error, {} panics, {slow} slow", panics.values().map(|x| x.0).sum::<usize>());
    let mut rs: Vec<_> = reasons.iter().collect();
    rs.sort_by(|a, b| b.1.cmp(a.1));
    for (k, n) in rs.into_iter().take(10) {
        println!("  rejected x{n:<6} {k}");
    }
    let out = std::env::temp_dir().join("dgio_fuzz_crashes");
    for (i, (msg, (n, input))) in panics.iter().enumerate() {
        std::fs::create_dir_all(&out).ok();
        let p = out.join(format!("crash_{i}.grib2"));
        std::fs::write(&p, input).ok();
        println!("  PANIC x{n}: {msg}  (input: {})", p.display());
    }
    assert!(panics.is_empty(), "decoder panicked on malformed input");
    assert_eq!(slow, 0, "a decode took > 2 s (possible non-terminating loop)");
}
