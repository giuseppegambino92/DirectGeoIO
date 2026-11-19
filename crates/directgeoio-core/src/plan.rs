//! Device-independent decode plan: turns scanned `Field` metadata into the `DgField`
//! descriptors (kernels/dgio_shared.h), the storage→device DMA ranges, and the sizes of every
//! scratch buffer. Built from metadata only — payload bytes are never touched by the CPU.

use crate::aec::AecParams;
use crate::decode::codes_power;
use crate::grib2::{Error, Field, Packing, Result, AEC_DATA_PREPROCESS, AEC_PAD_RSI};

pub const DG_KIND_SIMPLE: u32 = 0;
pub const DG_KIND_CCSDS: u32 = 1;
pub const DG_KIND_COMPLEX: u32 = 2;
pub const DG_NONE: u32 = 0xFFFF_FFFF;
pub const DG_SCAN_TILE: u32 = 1024;
pub const DG_SCAN_THREADS: u32 = 256;
pub const DG_ERR_TRUNCATED: u32 = 1;
pub const DG_ERR_GROUPS: u32 = 2;
pub const DG_ERR_BITMAP: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DgField {
    pub kind: u32,
    pub nbits: u32,
    pub n_points: u32,
    pub n_values: u32,
    pub data_off: u32,
    pub data_len: u32,
    pub bitmap_off: u32,
    pub rank_off: u32,
    pub out_off: u32,
    pub codes_off: u32,
    pub aux_off: u32,
    pub aux_cap: u32,
    pub ref_value: f32,
    pub e: i32,
    pub dscale_hi: f32,
    pub dscale_lo: f32,
    pub block_size: u32,
    pub rsi: u32,
    pub id_len: u32,
    pub aec_flags: u32,
    pub ngroups: u32,
    pub ref_gw: u32,
    pub bits_gw: u32,
    pub ref_gl: u32,
    pub len_inc: u32,
    pub last_gl: u32,
    pub bits_gl: u32,
    pub sd_order: u32,
    pub sd_octets: u32,
    pub goff_refs: u32,
    pub goff_widths: u32,
    pub goff_lens: u32,
    pub goff_vals: u32,
    pub scan2: u32,
    pub field_index: u32,
    pub missing_mgmt: u32,
    pub mm_off: u32,
    pub tile_off: u32,
    pub bscale_lo: u32,
    pub bscale_hi: u32,
    pub dscale_lo_bits: u32,
    pub dscale_hi_bits: u32,
    pub exact_slow: u32,
    pub pad4: u32,
    pub pad5: u32,
    pub pad6: u32,
}
const _: () = assert!(std::mem::size_of::<DgField>() == 184);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DgParams {
    pub n_fields: u32,
    pub out_half: u32,
    pub exact: u32,
    pub n: u32,
    pub missing: f32,
    pub inclusive: u32,
    pub max_x: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DgStatus {
    pub count: u32,
    pub error: u32,
    pub base1: u32,
    pub pad: u32,
}

/// One storage→device copy: `len` bytes from `file_offset` to `buf_offset` in the raw buffer.
#[derive(Clone, Copy, Debug)]
pub struct IoRange {
    pub file_offset: u64,
    pub buf_offset: u64,
    pub len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoStrategy {
    /// one DMA command for the byte span covering all selected messages
    Span,
    /// one DMA command per payload / bitmap range
    Ranges,
    /// Span when selected payloads cover >= 50% of the span, else Ranges
    Auto,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub fields: Vec<DgField>,
    pub io: Vec<IoRange>,
    pub raw_len: u64,
    /// total uint32 codes; complex fields occupy [0, scan1_len)
    pub codes_len: u32,
    /// elements covered by the 1st inclusive scan (complex order 1 and 2 fields)
    pub scan1_len: u32,
    /// elements covered by the 2nd inclusive scan (complex order 2 fields)
    pub scan2_len: u32,
    pub rank_len: u32,
    pub cds_len: u32,
    pub groups_len: u32,
    /// values of complex fields with missing-value management (xraw / mvalid buffers)
    pub mm_len: u32,
    pub max_mm_values: u32,
    /// complex fields: first group of every 1024-value tile
    pub tiles_len: u32,
    pub out_len: u64,
    pub max_points: u32,
    pub max_values: u32,
    pub max_words: u32,
    pub max_cds: u32,
    pub max_rsi: u32,
    pub max_groups: u32,
    pub n_ccsds: u32,
    pub n_complex: u32,
    pub n_bitmap: u32,
    pub payload_bytes: u64,
}

impl Plan {
    pub fn n_fields(&self) -> usize {
        self.fields.len()
    }
}

fn align(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

/// Bytes of DMA / scratch padding after every range so that 64-bit bit-reader windows never
/// read outside the buffer.
pub const RAW_PAD: u64 = 64;

pub fn build(fields: &[Field], io: IoStrategy) -> Result<Plan> {
    let mut p = Plan::default();
    if fields.is_empty() {
        return Ok(p);
    }
    // ---- storage ranges ---------------------------------------------------------------
    let span_lo = fields.iter().map(|f| f.message.offset).min().unwrap();
    let span_hi = fields.iter().map(|f| f.message.offset + f.message.len).max().unwrap();
    let payload: u64 = fields.iter().map(|f| f.data.len + f.bitmap.map(|b| b.len).unwrap_or(0)).sum();
    p.payload_bytes = payload;
    let strategy = match io {
        IoStrategy::Auto => {
            if payload * 2 >= span_hi - span_lo { IoStrategy::Span } else { IoStrategy::Ranges }
        }
        s => s,
    };
    // map: file range -> raw buffer offset
    let mut data_off = vec![0u64; fields.len()];
    let mut bm_off = vec![u64::MAX; fields.len()];
    match strategy {
        IoStrategy::Span => {
            p.io.push(IoRange { file_offset: span_lo, buf_offset: 0, len: span_hi - span_lo });
            for (i, f) in fields.iter().enumerate() {
                data_off[i] = f.data.offset - span_lo;
                if let Some(b) = f.bitmap {
                    bm_off[i] = b.offset - span_lo;
                }
            }
            p.raw_len = align(span_hi - span_lo + RAW_PAD, 16);
        }
        _ => {
            let mut cur = 0u64;
            // de-duplicate shared bitmaps (Section 6 indicator 254)
            let mut seen: Vec<(u64, u64)> = Vec::new();
            for (i, f) in fields.iter().enumerate() {
                if let Some(b) = f.bitmap {
                    if let Some(&(_, o)) = seen.iter().find(|(fo, _)| *fo == b.offset) {
                        bm_off[i] = o;
                    } else {
                        p.io.push(IoRange { file_offset: b.offset, buf_offset: cur, len: b.len });
                        bm_off[i] = cur;
                        seen.push((b.offset, cur));
                        cur = align(cur + b.len, 16);
                    }
                }
                p.io.push(IoRange { file_offset: f.data.offset, buf_offset: cur, len: f.data.len });
                data_off[i] = cur;
                cur = align(cur + f.data.len, 16);
            }
            p.raw_len = cur + RAW_PAD;
        }
    }
    if p.raw_len > u32::MAX as u64 {
        return Err(Error::Unsupported("batch larger than 4 GiB; split the selection".into()));
    }

    // ---- per-field descriptors ----------------------------------------------------------
    let mut out_off = 0u64;
    let mut rank_off = 0u32;
    let mut cds_off = 0u32;
    let mut grp_off = 0u32;
    let mut descs: Vec<DgField> = Vec::with_capacity(fields.len());
    for (i, f) in fields.iter().enumerate() {
        f.validate().map_err(|e| Error::Format(format!("field {i}: {e}")))?;
        let d = f.scale.d as i64;
        let dscale = codes_power(-d, 10);
        let hi = dscale as f32;
        let lo = (dscale - hi as f64) as f32;
        let bscale = codes_power(f.scale.e as i64, 2);
        // The Metal fast path evaluates in float-float; it is only valid while every intermediate
        // stays in the normal float32 range. Otherwise every value takes the emulated-f64 path.
        let exact_slow = !(-100..=100).contains(&(f.scale.e as i32))
            || !(dscale > 2f64.powi(-100) && dscale < 2f64.powi(100))
            || f.scale.reference.abs() > 1e37
            || (f.scale.reference != 0.0 && f.scale.reference.abs() < 2f32.powi(-100));
        let mut g = DgField {
            kind: DG_KIND_SIMPLE,
            nbits: f.scale.nbits as u32,
            n_points: f.grid.n_points,
            n_values: f.n_values,
            data_off: data_off[i] as u32,
            data_len: f.data.len as u32,
            bitmap_off: DG_NONE,
            rank_off: 0,
            out_off: out_off as u32,
            codes_off: 0,
            aux_off: 0,
            aux_cap: 0,
            ref_value: f.scale.reference,
            e: f.scale.e as i32,
            dscale_hi: hi,
            dscale_lo: lo,
            bscale_lo: bscale.to_bits() as u32,
            bscale_hi: (bscale.to_bits() >> 32) as u32,
            dscale_lo_bits: dscale.to_bits() as u32,
            dscale_hi_bits: (dscale.to_bits() >> 32) as u32,
            exact_slow: exact_slow as u32,
            field_index: i as u32,
            ..Default::default()
        };
        if out_off + f.grid.n_points as u64 > u32::MAX as u64 {
            return Err(Error::Unsupported("output larger than 2^32 elements; split the selection".into()));
        }
        if f.bitmap.is_some() {
            g.bitmap_off = bm_off[i] as u32;
            g.rank_off = rank_off;
            let nw = f.grid.n_points.div_ceil(32);
            rank_off += nw;
            p.max_words = p.max_words.max(nw);
            p.n_bitmap += 1;
        } else if f.n_values != f.grid.n_points {
            return Err(Error::Format(format!("field {i}: {} values for {} points and no bitmap", f.n_values, f.grid.n_points)));
        }
        match &f.packing {
            Packing::Unsupported(_) => unreachable!("rejected by validate"),
            Packing::Simple => {}
            Packing::Ccsds(c) => {
                if g.nbits == 0 {
                    g.kind = DG_KIND_SIMPLE; // constant field: every code is 0
                } else {
                    let ap = AecParams::new(c, f.scale.nbits, f.n_values);
                    g.kind = DG_KIND_CCSDS;
                    g.block_size = ap.block_size;
                    g.rsi = ap.rsi;
                    g.id_len = ap.id_len;
                    g.aec_flags = (c.flags & AEC_DATA_PREPROCESS != 0) as u32 | (((c.flags & AEC_PAD_RSI != 0) as u32) << 1);
                    g.aux_off = cds_off;
                    g.aux_cap = ap.n_blocks() + 1;
                    cds_off += g.aux_cap;
                    p.max_cds = p.max_cds.max(g.aux_cap);
                    p.max_rsi = p.max_rsi.max(ap.n_rsi());
                    p.n_ccsds += 1;
                }
            }
            Packing::Complex(c) => {
                if c.missing_mgmt > 2 {
                    return Err(Error::Unsupported(format!("missing value management {}", c.missing_mgmt)));
                }
                if c.missing_mgmt != 0 {
                    g.missing_mgmt = c.missing_mgmt as u32;
                    g.mm_off = p.mm_len;
                    p.mm_len += f.n_values;
                    p.max_mm_values = p.max_mm_values.max(f.n_values);
                }
                if c.sd_order > 2 {
                    return Err(Error::Unsupported(format!("spatial differencing order {}", c.sd_order)));
                }
                let ng = c.ngroups as u64;
                g.kind = DG_KIND_COMPLEX;
                g.ngroups = c.ngroups;
                g.ref_gw = c.ref_group_width as u32;
                g.bits_gw = c.bits_group_width as u32;
                g.ref_gl = c.ref_group_length;
                g.len_inc = c.length_increment as u32;
                g.last_gl = c.last_group_length;
                g.bits_gl = c.bits_group_length as u32;
                g.sd_order = c.sd_order as u32;
                g.sd_octets = c.sd_octets as u32;
                let refs = if c.sd_order > 0 { c.sd_octets as u64 * (c.sd_order as u64 + 1) } else { 0 };
                let widths = refs + (ng * g.nbits as u64).div_ceil(8);
                let lens = widths + (ng * g.bits_gw as u64).div_ceil(8);
                let vals = lens + (ng * g.bits_gl as u64).div_ceil(8);
                if vals > f.data.len {
                    return Err(Error::Format(format!("field {i}: complex group tables exceed Section 7")));
                }
                g.goff_refs = refs as u32;
                g.goff_widths = widths as u32;
                g.goff_lens = lens as u32;
                g.goff_vals = vals as u32;
                g.aux_off = grp_off;
                g.aux_cap = c.ngroups;
                g.tile_off = p.tiles_len;
                p.tiles_len += f.n_values.div_ceil(1024) + 1;
                grp_off += c.ngroups;
                p.max_groups = p.max_groups.max(c.ngroups);
                p.n_complex += 1;
            }
        }
        p.max_points = p.max_points.max(f.grid.n_points);
        p.max_values = p.max_values.max(f.n_values);
        out_off += f.grid.n_points as u64;
        descs.push(g);
    }
    // ---- codes layout: [complex order 2][complex order 1][complex order 0][ccsds] -------
    let mut codes = 0u32;
    for order in [2u32, 1, 0] {
        for g in descs.iter_mut().filter(|g| g.kind == DG_KIND_COMPLEX && g.sd_order == order) {
            g.codes_off = codes;
            g.scan2 = (order == 2) as u32;
            // tile-aligned so the fused spatial-differencing tiles never straddle two fields
            codes += if order > 0 { g.n_values.div_ceil(1024) * 1024 } else { g.n_values };
        }
        if order == 2 {
            p.scan2_len = codes;
        }
        if order == 1 {
            p.scan1_len = codes;
        }
    }
    for g in descs.iter_mut().filter(|g| g.kind == DG_KIND_CCSDS) {
        g.codes_off = codes;
        codes += g.n_values;
    }
    p.codes_len = codes;
    p.rank_len = rank_off;
    p.cds_len = cds_off;
    p.groups_len = grp_off;
    p.out_len = out_off;
    p.fields = descs;
    Ok(p)
}

/// Scan levels for a multi-level Blelloch scan of `n` elements: element counts per level.
pub fn scan_levels(n: u32) -> Vec<u32> {
    let mut v = vec![n];
    let mut m = n;
    while m > DG_SCAN_TILE {
        m = m.div_ceil(DG_SCAN_TILE);
        v.push(m);
    }
    v
}
