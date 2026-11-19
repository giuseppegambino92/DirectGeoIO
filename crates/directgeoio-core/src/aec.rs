//! CCSDS 121.0-B-3 Adaptive Entropy Coding (libaec / GRIB2 Template 5.42) decoder,
//! organised exactly like the GPU pipeline:
//!
//!  1. `scan_cds`     — boundary scan. Finds the bit offset of every Coded Data Set (CDS)
//!                      without producing samples (popcount-skips FS codes). This is the only
//!                      sequential pass: CCSDS streams carry no block index, and RSIs are not
//!                      byte aligned unless `AEC_PAD_RSI` is set, so block k's start depends on
//!                      the lengths of blocks 0..k-1.
//!  2. `decode_cds`   — embarrassingly parallel: each CDS is decoded independently into mapped
//!                      prediction residuals (or raw samples) at its block position.
//!  3. `postprocess`  — inverse unit-delay predictor + inverse mapping, one independent
//!                      sequential recurrence per Reference Sample Interval.

use crate::bits::BitReader;
use crate::grib2::{Ccsds, AEC_DATA_PREPROCESS, AEC_PAD_RSI, AEC_RESTRICTED};

pub const SE_TABLE_SIZE: u32 = 90;
const ROS: u32 = 5;

#[derive(Clone, Copy, Debug)]
pub struct AecParams {
    pub nbits: u32,
    pub block_size: u32,
    pub rsi: u32,
    pub id_len: u32,
    pub preprocess: bool,
    pub pad_rsi: bool,
    pub n: u32,
}

impl AecParams {
    pub fn new(c: &Ccsds, nbits: u8, n: u32) -> Self {
        let nbits = nbits as u32;
        let id_len = if nbits > 16 {
            5
        } else if nbits > 8 {
            4
        } else if c.flags & AEC_RESTRICTED != 0 && nbits <= 4 {
            if nbits <= 2 { 1 } else { 2 }
        } else {
            3
        };
        AecParams {
            nbits,
            block_size: c.block_size as u32,
            rsi: c.rsi as u32,
            id_len,
            preprocess: c.flags & AEC_DATA_PREPROCESS != 0,
            pad_rsi: c.flags & AEC_PAD_RSI != 0,
            n,
        }
    }
    pub fn n_blocks(&self) -> u32 {
        self.n.div_ceil(self.block_size)
    }
    pub fn n_rsi(&self) -> u32 {
        self.n_blocks().div_ceil(self.rsi)
    }
}

/// One coded data set: its bit offset in the payload and the first block it covers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Cds {
    pub bitpos: u32,
    pub block: u32,
}

#[derive(Debug)]
pub struct AecError(pub String);

/// Pass 1: locate every CDS. Appends a sentinel {bitpos: end, block: n_blocks}.
pub fn scan_cds(data: &[u8], p: &AecParams, out: &mut Vec<Cds>) -> Result<(), AecError> {
    let nblocks = p.n_blocks();
    let bs = p.block_size;
    let max_id = (1u32 << p.id_len) - 1;
    let mut r = BitReader::new(data);
    let mut b = 0u32;
    while b < nblocks {
        let in_rsi = b % p.rsi;
        if in_rsi == 0 && p.pad_rsi && b > 0 {
            r.align_byte();
        }
        if r.bits_left() <= 0 {
            return Err(AecError(format!("stream ends before block {b}/{nblocks}")));
        }
        out.push(Cds { bitpos: r.pos() as u32, block: b });
        let refs = (p.preprocess && in_rsi == 0) as u32;
        let id = r.read(p.id_len);
        if id == 0 {
            let se = r.read(1);
            r.skip((refs * p.nbits) as u64);
            if se == 1 {
                if !r.skip_fs(bs / 2) {
                    return Err(AecError("truncated SE block".into()));
                }
                b += 1;
            } else {
                let fs = r.read_fs().ok_or_else(|| AecError("truncated zero block".into()))?;
                let mut z = fs + 1;
                if z == ROS {
                    z = (p.rsi - in_rsi).min(64 - in_rsi % 64);
                } else if z > ROS {
                    z -= 1;
                }
                b += z;
            }
        } else if id == max_id {
            r.skip((bs * p.nbits) as u64);
            b += 1;
        } else {
            let k = id - 1;
            r.skip((refs * p.nbits) as u64);
            if !r.skip_fs(bs - refs) {
                return Err(AecError("truncated split block".into()));
            }
            r.skip((k * (bs - refs)) as u64);
            b += 1;
        }
    }
    if r.pos() > data.len() as u64 * 8 {
        return Err(AecError("last coded data set runs past the end of Section 7".into()));
    }
    out.push(Cds { bitpos: r.pos() as u32, block: nblocks });
    Ok(())
}

/// Pass 2: decode one CDS into `out` (indexed by sample). `next_block` is the first block
/// of the following CDS. Returns mapped residuals (preprocessing on) or raw samples.
#[inline]
pub fn decode_cds(data: &[u8], p: &AecParams, cds: Cds, next_block: u32, out: &mut [u32]) {
    let bs = p.block_size;
    let max_id = (1u32 << p.id_len) - 1;
    let n = p.n as usize;
    let mut r = BitReader::at(data, cds.bitpos as u64);
    let s0 = (cds.block * bs) as usize;
    let refs = (p.preprocess && cds.block % p.rsi == 0) as u32;
    let put = |out: &mut [u32], i: usize, v: u32| {
        if i < n {
            out[i] = v;
        }
    };
    let id = r.read(p.id_len);
    if id == 0 {
        let se = r.read(1);
        if refs == 1 {
            put(out, s0, r.read(p.nbits));
        }
        if se == 1 {
            let mut i = refs;
            while i < bs {
                let m = r.read_fs().unwrap_or(0).min(SE_TABLE_SIZE);
                // m = beta(beta+1)/2 + d1, with pair (beta - d1, d1)
                let beta = se_beta(m);
                let d1 = m - beta * (beta + 1) / 2;
                if i & 1 == 0 {
                    put(out, s0 + i as usize, beta - d1);
                    i += 1;
                }
                put(out, s0 + i as usize, d1);
                i += 1;
            }
        } else {
            let end = ((next_block * bs) as usize).min(n);
            for i in (s0 + refs as usize)..end {
                out[i] = 0;
            }
        }
    } else if id == max_id {
        for i in 0..bs as usize {
            put(out, s0 + i, r.read(p.nbits));
        }
    } else {
        let k = id - 1;
        if refs == 1 {
            put(out, s0, r.read(p.nbits));
        }
        let m = (bs - refs) as usize;
        let base = s0 + refs as usize;
        let mut fs = [0u32; 256];
        for f in fs.iter_mut().take(m) {
            *f = r.read_fs().unwrap_or(0);
        }
        for (i, f) in fs.iter().take(m).enumerate() {
            put(out, base + i, (f << k) | r.read(k));
        }
    }
}

/// Largest beta with beta(beta+1)/2 <= m.
#[inline]
pub fn se_beta(m: u32) -> u32 {
    let mut b = (((8 * m + 1) as f64).sqrt() as u32).saturating_sub(1) / 2;
    while (b + 1) * (b + 2) / 2 <= m {
        b += 1;
    }
    while b * (b + 1) / 2 > m {
        b -= 1;
    }
    b
}

/// Pass 3: undo the preprocessor for one RSI starting at sample `s0` (unsigned data).
#[inline]
pub fn postprocess_rsi(p: &AecParams, rsi_index: u32, out: &mut [u32]) {
    if !p.preprocess {
        return;
    }
    let len = (p.rsi * p.block_size) as usize;
    let s0 = rsi_index as usize * len;
    let s1 = (s0 + len).min(p.n as usize);
    let xmax: u32 = if p.nbits == 32 { u32::MAX } else { (1u32 << p.nbits) - 1 };
    let med: u32 = xmax / 2 + 1;
    let mut x = out[s0];
    for v in out.iter_mut().take(s1).skip(s0 + 1) {
        let d = *v;
        let half_d = (d >> 1) + (d & 1);
        let mask = if x & med != 0 { xmax } else { 0 };
        if half_d <= (mask ^ x) {
            x = x.wrapping_add((d >> 1) ^ (!((d & 1).wrapping_sub(1))));
        } else {
            x = mask ^ d;
        }
        *v = x;
    }
}

/// Full sequential decode (reference path).
pub fn decode(data: &[u8], p: &AecParams) -> Result<Vec<u32>, AecError> {
    let mut cds = Vec::with_capacity(p.n_blocks() as usize + 1);
    scan_cds(data, p, &mut cds)?;
    let mut out = vec![0u32; p.n as usize];
    for w in cds.windows(2) {
        decode_cds(data, p, w[0], w[1].block, &mut out);
    }
    for r in 0..p.n_rsi() {
        postprocess_rsi(p, r, &mut out);
    }
    Ok(out)
}
