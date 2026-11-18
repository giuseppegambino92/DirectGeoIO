//! WMO GRIB2 (FM 92 edition 2) section scanner.
//!
//! The scanner never touches Section 7 payloads or Section 6 bitmap bits: it records their
//! absolute byte ranges so that a storage backend can DMA them straight into GPU memory.
//! Only Section 0 and the small metadata sections (1, 3, 4, 5, 6-indicator) are read.

use crate::bits::*;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Format(String),
    Unsupported(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(s) => write!(f, "malformed GRIB2: {s}"),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
pub type Result<T> = std::result::Result<T, Error>;

/// Positional reads (implemented for in-memory bytes and for files via pread).
pub trait ReadAt {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()>;
    fn len(&self) -> u64;
}
impl ReadAt for [u8] {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        let o = off as usize;
        if o + buf.len() > <[u8]>::len(self) {
            return Err(Error::Format("read past end".into()));
        }
        buf.copy_from_slice(&self[o..o + buf.len()]);
        Ok(())
    }
    fn len(&self) -> u64 {
        <[u8]>::len(self) as u64
    }
}
impl ReadAt for std::fs::File {
    fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        self.read_exact_at(buf, off)?;
        Ok(())
    }
    fn len(&self) -> u64 {
        self.metadata().map(|m| m.len()).unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub len: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Grid {
    pub template: u16,
    pub n_points: u32,
    pub ni: u32,
    pub nj: u32,
    /// micro-degrees
    pub la1: i32,
    pub lo1: i32,
    pub la2: i32,
    pub lo2: i32,
    pub di: u32,
    pub dj: u32,
    pub scanning_mode: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Product {
    pub template: u16,
    pub category: u8,
    pub number: u8,
    pub forecast_time: u32,
    pub time_unit: u8,
    pub level_type: u8,
    pub level_scale: i8,
    pub level_value: i64,
}
impl Product {
    /// Level in the native unit of the surface type (Pa for isobaric surfaces, m for heights).
    pub fn level(&self) -> f64 {
        if self.level_value == i64::MIN {
            return f64::NAN;
        }
        self.level_value as f64 * 10f64.powi(-(self.level_scale as i32))
    }
}

/// Template 5.x scaling common to simple, complex and CCSDS packing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scale {
    pub reference: f32,
    pub e: i16,
    pub d: i16,
    pub nbits: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Ccsds {
    pub flags: u8,
    pub block_size: u8,
    pub rsi: u16,
}
pub const AEC_DATA_SIGNED: u8 = 1;
pub const AEC_DATA_3BYTE: u8 = 2;
pub const AEC_DATA_MSB: u8 = 4;
pub const AEC_DATA_PREPROCESS: u8 = 8;
pub const AEC_RESTRICTED: u8 = 16;
pub const AEC_PAD_RSI: u8 = 32;

#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub missing_mgmt: u8,
    pub ngroups: u32,
    pub ref_group_width: u8,
    pub bits_group_width: u8,
    pub ref_group_length: u32,
    pub length_increment: u8,
    pub last_group_length: u32,
    pub bits_group_length: u8,
    /// 0 for Template 5.2; 1 or 2 for Template 5.3
    pub sd_order: u8,
    pub sd_octets: u8,
}

#[derive(Clone, Copy, Debug)]
pub enum Packing {
    /// Template 5.0
    Simple,
    /// Template 5.2 / 5.3
    Complex(Complex),
    /// Template 5.42
    Ccsds(Ccsds),
}

#[derive(Clone, Debug)]
pub struct Field {
    pub message_index: u32,
    pub field_in_message: u32,
    pub message: ByteRange,
    pub discipline: u8,
    pub centre: u16,
    /// year, month, day, hour, minute, second
    pub ref_time: [u16; 6],
    pub grid: Grid,
    pub product: Product,
    pub drt: u16,
    pub packing: Packing,
    pub scale: Scale,
    /// number of packed (non-masked) values, Section 5 octets 6-9
    pub n_values: u32,
    /// absolute byte range of the Section 6 bitmap bits (excluding the 6-byte header)
    pub bitmap: Option<ByteRange>,
    /// absolute byte range of the Section 7 payload (excluding the 5-byte header)
    pub data: ByteRange,
}

impl Field {
    pub fn n_points(&self) -> u32 {
        self.grid.n_points
    }
    pub fn param(&self) -> crate::param::Param {
        crate::param::lookup(self.discipline, self.product.category, self.product.number)
    }
    pub fn level_name(&self) -> &'static str {
        crate::param::level_type_name(self.product.level_type)
    }
}

const GRIB: [u8; 4] = *b"GRIB";

/// Scan every GRIB2 message in `src`, returning one `Field` per Section 7.
/// Non-GRIB bytes between messages are skipped (as ecCodes does).
pub fn scan<R: ReadAt + ?Sized>(src: &R) -> Result<Vec<Field>> {
    let total = src.len();
    let mut fields = Vec::new();
    let mut off = 0u64;
    let mut msg_index = 0u32;
    let mut probe = vec![0u8; 64 * 1024];
    while off + 16 <= total {
        // locate next "GRIB"
        let mut s0 = [0u8; 16];
        src.read_at(off, &mut s0)?;
        if s0[..4] != GRIB {
            let n = probe.len().min((total - off) as usize);
            src.read_at(off, &mut probe[..n])?;
            match probe[1..n].windows(4).position(|w| w == GRIB) {
                Some(p) => {
                    off += 1 + p as u64;
                    continue;
                }
                None => {
                    off += (n as u64).saturating_sub(3).max(1);
                    continue;
                }
            }
        }
        if s0[7] != 2 {
            return Err(Error::Unsupported(format!("GRIB edition {} at offset {off}", s0[7])));
        }
        let discipline = s0[6];
        let msg_len = be_u64(&s0, 8);
        if msg_len < 16 + 4 || off + msg_len > total {
            return Err(Error::Format(format!("bad message length {msg_len} at {off}")));
        }
        scan_message(src, off, msg_len, discipline, msg_index, &mut fields)?;
        msg_index += 1;
        off += msg_len;
    }
    Ok(fields)
}

fn scan_message<R: ReadAt + ?Sized>(
    src: &R,
    msg_off: u64,
    msg_len: u64,
    discipline: u8,
    msg_index: u32,
    out: &mut Vec<Field>,
) -> Result<()> {
    let end = msg_off + msg_len;
    let mut p = msg_off + 16;
    let mut centre = 0u16;
    let mut ref_time = [0u16; 6];
    let mut grid = Grid::default();
    let mut product = Product::default();
    let mut drt = 0u16;
    let mut packing = Packing::Simple;
    let mut scale = Scale::default();
    let mut n_values = 0u32;
    let mut bitmap: Option<ByteRange> = None;
    let mut have_bitmap = false;
    let mut prev_bitmap: Option<ByteRange> = None;
    let mut field_in_message = 0u32;
    let mut buf = Vec::new();
    loop {
        if p + 4 > end {
            return Err(Error::Format(format!("message at {msg_off} truncated")));
        }
        let mut h = [0u8; 5];
        src.read_at(p, &mut h[..4])?;
        if &h[..4] == b"7777" {
            break;
        }
        src.read_at(p, &mut h)?;
        let slen = be_u32(&h, 0) as u64;
        let snum = h[4];
        if slen < 5 || p + slen > end {
            return Err(Error::Format(format!("section {snum} length {slen} at {p}")));
        }
        let small = matches!(snum, 1 | 3 | 4 | 5);
        if small {
            buf.resize(slen as usize, 0);
            src.read_at(p, &mut buf)?;
        }
        match snum {
            1 => {
                let b = &buf;
                centre = be_u16(b, 5);
                ref_time = [be_u16(b, 12), b[14] as u16, b[15] as u16, b[16] as u16, b[17] as u16, b[18] as u16];
            }
            2 => {}
            3 => grid = parse_grid(&buf)?,
            4 => product = parse_product(&buf),
            5 => {
                let b = &buf;
                n_values = be_u32(b, 5);
                drt = be_u16(b, 9);
                let (pk, sc) = parse_drs(drt, b)?;
                packing = pk;
                scale = sc;
            }
            6 => {
                let mut ind = [0u8; 1];
                src.read_at(p + 5, &mut ind)?;
                match ind[0] {
                    0 => {
                        let r = ByteRange { offset: p + 6, len: slen - 6 };
                        bitmap = Some(r);
                        prev_bitmap = Some(r);
                    }
                    254 => bitmap = prev_bitmap,
                    255 => bitmap = None,
                    x => return Err(Error::Unsupported(format!("predefined bitmap {x}"))),
                }
                have_bitmap = true;
            }
            7 => {
                if !have_bitmap {
                    bitmap = None;
                }
                out.push(Field {
                    message_index: msg_index,
                    field_in_message,
                    message: ByteRange { offset: msg_off, len: msg_len },
                    discipline,
                    centre,
                    ref_time,
                    grid: grid.clone(),
                    product: product.clone(),
                    drt,
                    packing,
                    scale,
                    n_values,
                    bitmap,
                    data: ByteRange { offset: p + 5, len: slen - 5 },
                });
                field_in_message += 1;
                have_bitmap = false;
            }
            x => return Err(Error::Format(format!("unknown section number {x}"))),
        }
        p += slen;
    }
    Ok(())
}

fn parse_grid(b: &[u8]) -> Result<Grid> {
    let n_points = be_u32(b, 6);
    let template = be_u16(b, 12);
    let mut g = Grid { template, n_points, ..Default::default() };
    if template == 0 && b.len() >= 72 {
        g.ni = be_u32(b, 30);
        g.nj = be_u32(b, 34);
        g.la1 = sm_i32(b, 46);
        g.lo1 = sm_i32(b, 50);
        g.la2 = sm_i32(b, 55);
        g.lo2 = sm_i32(b, 59);
        g.di = be_u32(b, 63);
        g.dj = be_u32(b, 67);
        g.scanning_mode = b[71];
    }
    Ok(g)
}

fn parse_product(b: &[u8]) -> Product {
    let template = be_u16(b, 7);
    let mut p = Product { template, level_value: i64::MIN, ..Default::default() };
    // Templates 4.0, 4.1, 4.8, 4.11 share this layout for the first 28 octets.
    if matches!(template, 0 | 1 | 2 | 8 | 11 | 12) && b.len() >= 28 {
        p.category = b[9];
        p.number = b[10];
        p.time_unit = b[17];
        p.forecast_time = be_u32(b, 18);
        p.level_type = b[22];
        let sv = be_u32(b, 24);
        if b[23] != 0xFF || sv != 0xFFFF_FFFF {
            p.level_scale = b[23] as i8;
            p.level_value = if sv == 0xFFFF_FFFF { 0 } else { sm_i32(b, 24) as i64 };
        }
    }
    p
}

fn parse_drs(drt: u16, b: &[u8]) -> Result<(Packing, Scale)> {
    let need = |n: usize| -> Result<()> {
        if b.len() < n {
            Err(Error::Format(format!("section 5 too short for template 5.{drt}")))
        } else {
            Ok(())
        }
    };
    need(21)?;
    let scale = Scale {
        reference: f32::from_bits(be_u32(b, 11)),
        e: sm_i16(b, 15),
        d: sm_i16(b, 17),
        nbits: b[19],
    };
    if scale.nbits > 32 {
        return Err(Error::Unsupported(format!("bitsPerValue {}", scale.nbits)));
    }
    let pk = match drt {
        0 => Packing::Simple,
        2 | 3 => {
            need(if drt == 3 { 49 } else { 47 })?;
            Packing::Complex(Complex {
                missing_mgmt: b[22],
                ngroups: be_u32(b, 31),
                ref_group_width: b[35],
                bits_group_width: b[36],
                ref_group_length: be_u32(b, 37),
                length_increment: b[41],
                last_group_length: be_u32(b, 42),
                bits_group_length: b[46],
                sd_order: if drt == 3 { b[47] } else { 0 },
                sd_octets: if drt == 3 { b[48] } else { 0 },
            })
        }
        42 => {
            need(25)?;
            Packing::Ccsds(Ccsds { flags: b[21], block_size: b[22], rsi: be_u16(b, 23) })
        }
        x => return Err(Error::Unsupported(format!("data representation template 5.{x}"))),
    };
    Ok((pk, scale))
}
