//! libFuzzer target (nightly + cargo-fuzz):
//!   cargo +nightly fuzz run scan_decode fuzz/corpus -- -max_len=65536
//! Seed corpus: cp fixtures/edge_*.grib2 fuzz/corpus/
//! The stable-Rust equivalent runs in CI: crates/directgeoio-core/tests/fuzz_mutations.rs
#![no_main]
use directgeoio_core::plan::{self, IoStrategy};
use directgeoio_core::{decode, scan};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(fields) = scan(data) {
        for f in &fields {
            if f.validate().is_ok() && f.n_points() <= 4_000_000 {
                let _ = decode::decode_f64(f, data);
            }
        }
        let _ = plan::build(&fields, IoStrategy::Ranges);
    }
});
