//! Deterministic GRIB2 mutation helpers shared by the CPU and GPU fuzz tests.
#![doc(hidden)]

use std::path::Path;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next() % n as u64) as usize }
    }
}

/// Seeds: every fixtures/edge_*.grib2 file.
pub fn seeds(fixtures: &Path) -> Vec<Vec<u8>> {
    let dir = fixtures;
    let mut v: Vec<Vec<u8>> = std::fs::read_dir(dir)
        .expect("fixtures/ missing: run `make fixtures`")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().map(|n| n.to_string_lossy().starts_with("edge_")).unwrap_or(false))
        .map(|p| std::fs::read(p).unwrap())
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no edge_*.grib2 seeds: run `make fixtures`");
    v
}

/// Offsets of section headers (length u32 + number u8) of the first message.
pub fn sections(b: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    if b.len() < 16 {
        return out;
    }
    let mut p = 16;
    while p + 5 <= b.len() && &b[p..p + 4] != b"7777" {
        out.push(p);
        let l = u32::from_be_bytes(b[p..p + 4].try_into().unwrap()) as usize;
        if l < 5 {
            break;
        }
        p += l;
    }
    out
}

pub fn mutate(rng: &mut Rng, seed: &[u8], all: &[Vec<u8>]) -> Vec<u8> {
    let mut b = seed.to_vec();
    let secs = sections(&b);
    match rng.below(7) {
        0 => b.truncate(rng.below(b.len())),
        1 => {
            for _ in 0..1 + rng.below(8) {
                let i = rng.below(b.len());
                b[i] ^= 1 << rng.below(8);
            }
        }
        2 if !secs.is_empty() => {
            let p = secs[rng.below(secs.len())];
            let v: u32 = [0, 1, 4, 5, 6, 8, 11, 20, 21, 0xFFFF_FFFF, rng.next() as u32][rng.below(11)];
            b[p..p + 4].copy_from_slice(&v.to_be_bytes());
        }
        3 if !secs.is_empty() => {
            // corrupt Section 5 template parameters (bits, block size, RSI, group descriptors)
            if let Some(&p) = secs.iter().find(|&&p| b[p + 4] == 5) {
                let l = u32::from_be_bytes(b[p..p + 4].try_into().unwrap()) as usize;
                for _ in 0..1 + rng.below(4) {
                    let i = p + 5 + rng.below(l.saturating_sub(5).max(1));
                    if i < b.len() {
                        b[i] = [0u8, 1, 33, 64, 255, rng.next() as u8][rng.below(6)];
                    }
                }
            }
        }
        4 if !secs.is_empty() => {
            // lie about the number of grid points / packed values
            for (num, off) in [(3u8, 6usize), (5, 5)] {
                if let Some(&p) = secs.iter().find(|&&p| b[p + 4] == num) {
                    if rng.below(2) == 0 && p + off + 4 <= b.len() {
                        let v: u32 = [0, 1, 7, 851, 852, 1 << 30, u32::MAX, rng.next() as u32][rng.below(8)];
                        b[p + off..p + off + 4].copy_from_slice(&v.to_be_bytes());
                    }
                }
            }
        }
        5 => {
            // splice: head of this seed + tail of another
            let o = &all[rng.below(all.len())];
            let cut = rng.below(b.len());
            let from = rng.below(o.len());
            b.truncate(cut);
            b.extend_from_slice(&o[from..]);
        }
        _ => {
            // random bytes inside Section 7 (entropy-coded / packed payload)
            if let Some(&p) = secs.iter().find(|&&p| b[p + 4] == 7) {
                for _ in 0..1 + rng.below(16) {
                    let i = p + 5 + rng.below(b.len().saturating_sub(p + 5).max(1));
                    if i < b.len() {
                        b[i] = rng.next() as u8;
                    }
                }
            }
        }
    }
    b
}

