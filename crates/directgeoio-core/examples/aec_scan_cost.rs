// Measures the cost of the sequential CCSDS boundary scan alone vs the full CPU decode.
use directgeoio_core::aec::{self, AecParams};
use directgeoio_core::{scan, Packing};
use std::time::Instant;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let fields = scan(&bytes[..]).unwrap();
    let (mut ts, mut td, mut ncds) = (0f64, 0f64, 0usize);
    for f in &fields {
        if let Packing::Ccsds(c) = f.packing {
            let p = AecParams::new(&c, f.scale.nbits, f.n_values);
            let d = &bytes[f.data.offset as usize..(f.data.offset + f.data.len) as usize];
            let mut v = Vec::with_capacity(p.n_blocks() as usize + 1);
            let t = Instant::now();
            for _ in 0..20 { v.clear(); aec::scan_cds(d, &p, &mut v).unwrap(); }
            ts += t.elapsed().as_secs_f64() / 20.0;
            ncds += v.len();
            let t = Instant::now();
            for _ in 0..5 { std::hint::black_box(aec::decode(d, &p).unwrap()); }
            td += t.elapsed().as_secs_f64() / 5.0;
        }
    }
    println!("{path}: {} fields, {} CDS, boundary scan {:.2} ms ({:.1} ns/CDS), full decode {:.2} ms", fields.len(), ncds, ts * 1e3, ts * 1e9 / ncds as f64, td * 1e3);
}
