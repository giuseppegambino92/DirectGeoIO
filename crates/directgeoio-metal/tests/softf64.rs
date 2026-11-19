//! The exact dequantisation slow path emulates IEEE-754 binary64 on the GPU. This test checks
//! the emulated add, mul and conversions bit-for-bit against native CPU f64 on random bit
//! patterns (all exponents: zeros, subnormals, infinities, NaNs) and on round-to-even ties.

use directgeoio_core::testing::Rng;
use directgeoio_metal::MetalBackend;

fn special(rng: &mut Rng) -> u64 {
    let s = (rng.next() & 1) << 63;
    let frac = rng.next() & 0x000F_FFFF_FFFF_FFFF;
    let e: u64 = match rng.below(8) {
        0 => 0,                                // zero / subnormal
        1 => 0x7FF,                            // inf / nan
        2 => 1 + rng.below(60) as u64,          // tiny normals (subnormal results)
        3 => 0x7FE - rng.below(60) as u64,      // huge normals (overflow)
        4 => 1023 + rng.below(40) as u64 - 20,  // ordinary
        _ => rng.below(0x7FF) as u64,
    };
    let frac = match rng.below(4) {
        0 => 0,
        1 => frac & 0xFFFF_FFFF_0000_0000, // few significant bits -> exact ties
        _ => frac,
    };
    s | (e << 52) | frac
}

#[test]
fn emulated_binary64_matches_native() {
    let be = MetalBackend::new().unwrap();
    let n: usize = 1 << 22;
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let a: Vec<u64> = (0..n).map(|_| special(&mut rng)).collect();
    let mut b: Vec<u64> = (0..n).map(|_| special(&mut rng)).collect();
    // half of the pairs: b close to a (cancellation) or a float32 pattern for conversions
    for i in (0..n).step_by(3) {
        b[i] = (f64::from_bits(a[i]) * (1.0 + (rng.below(1000) as f64 - 500.0) * 1e-15)).to_bits() ^ (1 << 63);
    }
    let buf = |v: &[u64]| {
        let m = be.alloc((v.len() * 8) as u64, false).unwrap();
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), m.contents() as *mut u64, v.len()) };
        m
    };
    let (ga, gb) = (buf(&a), buf(&b));
    let add = be.alloc((n * 8) as u64, false).unwrap();
    let mul = be.alloc((n * 8) as u64, false).unwrap();
    let to32 = be.alloc((n * 4) as u64, false).unwrap();
    let from32 = be.alloc((n * 8) as u64, false).unwrap();
    be.dispatch_1d("softf64_selftest", &[&ga, &gb, &add, &mul, &to32, &from32], n as u32).unwrap();
    let rd64 = |m: &directgeoio_metal::MtlBuffer| unsafe { std::slice::from_raw_parts(m.contents() as *const u64, n) }.to_vec();
    let (add, mul, from32) = (rd64(&add), rd64(&mul), rd64(&from32));
    let to32 = unsafe { std::slice::from_raw_parts(to32.contents() as *const u32, n) }.to_vec();
    let same64 = |g: u64, w: f64| (w.is_nan() && f64::from_bits(g).is_nan()) || g == w.to_bits();
    let (mut bad_add, mut bad_mul, mut bad_32, mut bad_from) = (0, 0, 0, 0);
    for i in 0..n {
        let (x, y) = (f64::from_bits(a[i]), f64::from_bits(b[i]));
        if !same64(add[i], x + y) {
            if bad_add < 3 { eprintln!("add {:#018x} + {:#018x}: gpu {:#018x} cpu {:#018x}", a[i], b[i], add[i], (x + y).to_bits()); }
            bad_add += 1;
        }
        if !same64(mul[i], x * y) {
            if bad_mul < 3 { eprintln!("mul {:#018x} * {:#018x}: gpu {:#018x} cpu {:#018x}", a[i], b[i], mul[i], (x * y).to_bits()); }
            bad_mul += 1;
        }
        let w32 = x as f32;
        if !((w32.is_nan() && f32::from_bits(to32[i]).is_nan()) || to32[i] == w32.to_bits()) {
            if bad_32 < 3 { eprintln!("f64->f32 {:#018x}: gpu {:#010x} cpu {:#010x}", a[i], to32[i], w32.to_bits()); }
            bad_32 += 1;
        }
        let f = f32::from_bits(b[i] as u32) as f64;
        if !same64(from32[i], f) {
            bad_from += 1;
        }
    }
    println!("{n} cases: add {bad_add}, mul {bad_mul}, f64->f32 {bad_32}, f32->f64 {bad_from} mismatches");
    assert_eq!((bad_add, bad_mul, bad_32, bad_from), (0, 0, 0, 0));
}
