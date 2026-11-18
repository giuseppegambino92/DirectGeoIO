//! Bit-exact validation of the CPU reference decoders against ecCodes ground truth
//! (fixtures/ref/*.npy written by tools/make_fixtures.py).

use directgeoio_core::{decode, npy, scan};
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

#[test]
fn cpu_decoders_match_eccodes_bit_exact() {
    let dir = fixtures();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("fixtures/ missing: run `make fixtures`")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "grib2").unwrap_or(false))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no fixtures: run `make fixtures`");
    let mut total = 0usize;
    for path in files {
        let bytes = std::fs::read(&path).unwrap();
        let fields = scan(&bytes[..]).unwrap();
        let name = path.file_name().unwrap().to_str().unwrap();
        for (i, f) in fields.iter().enumerate() {
            let want = npy::read_f64(&dir.join(format!("ref/{name}.{i}.npy"))).unwrap();
            let got = decode::decode_f64(f, &bytes).unwrap_or_else(|e| panic!("{name}#{i}: {e}"));
            assert_eq!(got.len(), want.len(), "{name}#{i} length");
            let mut bad = 0usize;
            let mut first = None;
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                let same = (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits();
                if !same {
                    bad += 1;
                    first.get_or_insert((j, *g, *w));
                }
            }
            assert_eq!(bad, 0, "{name}#{i} ({:?} {} bits): {bad} mismatches, first {:?}", f.packing, f.scale.nbits, first);
            total += got.len();
        }
        println!("{name}: {} fields bit-exact", fields.len());
    }
    println!("total {total} values bit-exact vs ecCodes");
}
