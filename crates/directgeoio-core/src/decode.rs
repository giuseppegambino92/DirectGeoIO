//! CPU reference decoders. These produce the integer codes X for each packed value and then
//! apply WMO dequantisation exactly the way ecCodes does (in float64):
//!
//! `Y = (X * 2^E + R) * 10^-D`
//!
//! with 10^-D computed by ecCodes' `codes_power` (repeated division), so the float64 result is
//! bit-identical to ecCodes and the float32 output is `(Y as f32)`.

use crate::aec::{self, AecParams};
use crate::bits::{sm_n, BitReader};
use crate::grib2::{Complex, Error, Field, Packing, Result, Scale};

fn trunc(what: &str) -> Error {
    Error::Format(format!("{what}: Section 7 truncated"))
}

/// ecCodes `codes_power<double>(s, n)`.
pub fn codes_power(s: i64, n: i64) -> f64 {
    let mut s = s;
    let mut d = 1.0f64;
    if s == 0 {
        return 1.0;
    }
    if s == 1 {
        return n as f64;
    }
    while s < 0 {
        d /= n as f64;
        s += 1;
    }
    while s > 0 {
        d *= n as f64;
        s -= 1;
    }
    d
}

#[derive(Clone, Copy, Debug)]
pub struct Dequant {
    pub reference: f64,
    pub bscale: f64,
    pub dscale: f64,
}
impl Dequant {
    pub fn new(s: &Scale) -> Self {
        Dequant {
            reference: s.reference as f64,
            bscale: codes_power(s.e as i64, 2),
            dscale: codes_power(-(s.d as i64), 10),
        }
    }
    #[inline]
    pub fn apply(&self, x: f64) -> f64 {
        (x * self.bscale + self.reference) * self.dscale
    }
}

/// Template 5.0: `n` values of `nbits` bits each, MSB first.
pub fn simple_codes(data: &[u8], nbits: u32, n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n];
    if nbits == 0 {
        return out;
    }
    let mut r = BitReader::new(data);
    for v in out.iter_mut() {
        *v = r.read(nbits);
    }
    out
}

/// Template 5.2 / 5.3 group decode + spatial-differencing reconstruction.
/// Returns the integer codes of the *present* values (compacted) and, when missing-value
/// management is used, the per-value presence mask (spatial differencing runs over present
/// values only, as in WMO/g2clib/ecCodes).
pub fn complex_codes(data: &[u8], c: &Complex, nbits: u32, n: usize) -> Result<(Vec<i64>, Option<Vec<bool>>)> {
    if c.missing_mgmt > 2 {
        return Err(Error::Unsupported(format!("missing value management {}", c.missing_mgmt)));
    }
    let ng = c.ngroups as usize;
    let nb = c.sd_octets as usize;
    let mut off = 0usize;
    let (mut ival1, mut ival2, mut minsd) = (0i64, 0i64, 0i64);
    if c.sd_order > 0 {
        ival1 = sm_n(data, 0, nb);
        if c.sd_order == 2 {
            ival2 = sm_n(data, nb, nb);
        }
        minsd = sm_n(data, nb * c.sd_order as usize, nb);
        off = nb * (c.sd_order as usize + 1);
    }
    if c.bits_group_width > 32 || c.bits_group_length > 32 || nbits > 32 || (c.sd_order > 0 && !(1..=4).contains(&nb)) {
        return Err(Error::Format("complex packing descriptors out of range".into()));
    }
    if off > data.len() {
        return Err(trunc("complex packing descriptors"));
    }
    let mut r = BitReader::at(data, (off * 8) as u64);
    let refs: Vec<u32> = (0..ng).map(|_| r.read(nbits)).collect();
    r.align_byte();
    let widths: Vec<u32> = (0..ng).map(|_| r.read(c.bits_group_width as u32) + c.ref_group_width as u32).collect();
    if let Some(w) = widths.iter().find(|&&w| w > 32) {
        return Err(Error::Format(format!("complex packing group width {w} > 32 bits")));
    }
    r.align_byte();
    let mut lens: Vec<u32> = Vec::with_capacity(ng);
    let mut sum = 0u64;
    for g in 0..ng {
        let l = r.read(c.bits_group_length as u32) as u64;
        let len = if g + 1 == ng { c.last_group_length as u64 } else { c.ref_group_length as u64 + l * c.length_increment as u64 };
        sum += len;
        if sum > n as u64 {
            return Err(Error::Format(format!("complex packing: group lengths exceed {n} values")));
        }
        lens.push(len as u32);
    }
    if sum != n as u64 {
        return Err(Error::Format(format!("complex packing: groups sum to {sum} values, expected {n}")));
    }
    r.align_byte();
    let value_bits: u64 = widths.iter().zip(&lens).map(|(&w, &l)| w as u64 * l as u64).sum();
    if r.pos() + value_bits > data.len() as u64 * 8 {
        return Err(trunc("complex packing values"));
    }
    let mut out = Vec::with_capacity(n);
    let mm = c.missing_mgmt;
    let mut present: Option<Vec<bool>> = (mm != 0).then(|| Vec::with_capacity(n));
    let mut total = 0usize;
    for g in 0..ng {
        let w = widths[g];
        let all1 = |b: u32| if b >= 32 { u32::MAX } else { (1u32 << b) - 1 };
        for _ in 0..lens[g] {
            let raw = r.read(w);
            let missing = mm != 0
                && if w > 0 {
                    raw == all1(w) || (mm == 2 && raw == all1(w) - 1)
                } else {
                    refs[g] == all1(nbits) || (mm == 2 && refs[g] == all1(nbits) - 1)
                };
            if let Some(p) = present.as_mut() {
                p.push(!missing);
            }
            if !missing {
                out.push(refs[g] as i64 + raw as i64);
            }
            total += 1;
        }
    }
    debug_assert_eq!(total, n);
    let n = out.len();
    match c.sd_order {
        0 => {}
        // modular arithmetic: identical to the GPU's uint32 scans for valid data and panic-free
        // on corrupt descriptors
        _ if n == 0 => {}
        1 => {
            out[0] = ival1;
            for i in 1..n {
                out[i] = out[i].wrapping_add(minsd).wrapping_add(out[i - 1]);
            }
        }
        2 => {
            out[0] = ival1;
            if n > 1 {
                out[1] = ival2;
            }
            for i in 2..n {
                out[i] = out[i].wrapping_add(minsd).wrapping_add(out[i - 1].wrapping_mul(2)).wrapping_sub(out[i - 2]);
            }
        }
        o => return Err(Error::Unsupported(format!("spatial differencing order {o}"))),
    }
    // Reconstructed integers are offsets from the field minimum (>= 0). They are defined modulo
    // 2^32, exactly as the GPU computes them; for valid fields this equals ecCodes' 64-bit result.
    for v in out.iter_mut() {
        *v = (*v as u64 & 0xFFFF_FFFF) as i64;
    }
    Ok((out, present))
}

/// Decoded integer codes for the packed (non-masked) values of a field.
pub enum Codes {
    U32(Vec<u32>),
    /// compacted codes + optional presence mask (complex packing missing-value management)
    I64(Vec<i64>, Option<Vec<bool>>),
}

/// `payload` is the Section 7 payload bytes of `f`.
pub fn field_codes(f: &Field, payload: &[u8]) -> Result<Codes> {
    f.validate()?;
    let n = f.n_values as usize;
    let nbits = f.scale.nbits as u32;
    Ok(match &f.packing {
        Packing::Simple => Codes::U32(simple_codes(payload, nbits, n)),
        Packing::Ccsds(c) => {
            if nbits == 0 {
                Codes::U32(vec![0; n])
            } else {
                let p = AecParams::new(c, f.scale.nbits, f.n_values);
                Codes::U32(aec::decode(payload, &p).map_err(|e| Error::Format(e.0))?)
            }
        }
        Packing::Complex(c) => {
            let (v, m) = complex_codes(payload, c, nbits, n)?;
            Codes::I64(v, m)
        }
        Packing::Unsupported(t) => return Err(Error::Unsupported(format!("data representation template 5.{t}"))),
    })
}

/// Expand packed values onto the full grid using the Section 6 bitmap (MSB-first, 1 = present).
/// Errors if the bitmap's set bits do not match the number of packed values.
pub fn expand<T: Copy>(packed: &[T], bitmap: Option<&[u8]>, n_points: usize, missing: T) -> Result<Vec<T>> {
    match bitmap {
        None if packed.len() == n_points => Ok(packed.to_vec()),
        None => Err(Error::Format(format!("{} values for {n_points} points and no bitmap", packed.len()))),
        Some(bm) => {
            if bm.len() * 8 < n_points {
                return Err(Error::Format("bitmap shorter than the grid".into()));
            }
            let full = n_points / 8;
            let mut ones: usize = bm[..full].iter().map(|b| b.count_ones() as usize).sum();
            if n_points % 8 != 0 {
                ones += (bm[full] >> (8 - n_points % 8)).count_ones() as usize;
            }
            if ones != packed.len() {
                return Err(Error::Format(format!("bitmap marks {ones} points present, Section 5 declares {}", packed.len())));
            }
            let mut out = Vec::with_capacity(n_points);
            let mut k = 0;
            for j in 0..n_points {
                if bm[j >> 3] & (0x80 >> (j & 7)) != 0 {
                    out.push(packed[k]);
                    k += 1;
                } else {
                    out.push(missing);
                }
            }
            Ok(out)
        }
    }
}

/// Decode a field to float64 on the full grid (NaN where masked). `msg` = whole file bytes.
pub fn decode_f64(f: &Field, file: &[u8]) -> Result<Vec<f64>> {
    let range = |r: crate::grib2::ByteRange| -> Result<&[u8]> {
        file.get(r.offset as usize..(r.offset + r.len) as usize).ok_or_else(|| Error::Format("section outside the buffer".into()))
    };
    let payload = range(f.data)?;
    let dq = Dequant::new(&f.scale);
    let packed: Vec<f64> = match field_codes(f, payload)? {
        Codes::U32(v) => v.iter().map(|&x| dq.apply(x as f64)).collect(),
        Codes::I64(v, None) => v.iter().map(|&x| dq.apply(x as f64)).collect(),
        Codes::I64(v, Some(present)) => {
            let mut it = v.iter();
            present.iter().map(|&p| if p { dq.apply(*it.next().unwrap() as f64) } else { f64::NAN }).collect()
        }
    };
    let bm = match f.bitmap {
        Some(b) => Some(range(b)?),
        None => None,
    };
    expand(&packed, bm, f.n_points() as usize, f64::NAN)
}

/// Decode a field to float32 (rounded from the ecCodes-identical float64 value).
pub fn decode_f32(f: &Field, file: &[u8]) -> Result<Vec<f32>> {
    Ok(decode_f64(f, file)?.into_iter().map(|v| v as f32).collect())
}
