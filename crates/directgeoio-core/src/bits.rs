//! Big-endian, MSB-first bit reader (the GRIB2 and CCSDS 121.0-B-3 bit order).

#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: u64, // bit position
}

impl<'a> BitReader<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    #[inline]
    pub fn at(data: &'a [u8], bitpos: u64) -> Self {
        Self { data, pos: bitpos }
    }
    #[inline]
    pub fn pos(&self) -> u64 {
        self.pos
    }
    #[inline]
    pub fn set_pos(&mut self, p: u64) {
        self.pos = p;
    }
    #[inline]
    pub fn bits_left(&self) -> i64 {
        (self.data.len() as i64) * 8 - self.pos as i64
    }
    #[inline]
    pub fn align_byte(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    /// 64-bit window whose MSB is the bit at `pos`. At least 57 bits are valid;
    /// bytes beyond the end of the buffer read as zero.
    #[inline]
    pub fn window(&self) -> u64 {
        let byte = (self.pos >> 3) as usize;
        let w = if byte + 8 <= self.data.len() {
            u64::from_be_bytes(self.data[byte..byte + 8].try_into().unwrap())
        } else {
            let mut b = [0u8; 8];
            if byte < self.data.len() {
                let n = self.data.len() - byte;
                b[..n].copy_from_slice(&self.data[byte..]);
            }
            u64::from_be_bytes(b)
        };
        w << (self.pos & 7)
    }

    /// Read `n` bits, 0 <= n <= 32.
    #[inline]
    pub fn read(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let v = (self.window() >> (64 - n)) as u32;
        self.pos += n as u64;
        v
    }

    #[inline]
    pub fn skip(&mut self, n: u64) {
        self.pos += n;
    }

    /// Fundamental-sequence (unary) code: number of 0 bits before the terminating 1.
    /// Returns None if the stream ends first.
    #[inline]
    pub fn read_fs(&mut self) -> Option<u32> {
        let mut acc = 0u32;
        loop {
            let w = self.window();
            let lz = w.leading_zeros();
            if lz < 57 {
                self.pos += lz as u64 + 1;
                return Some(acc + lz);
            }
            acc += 56;
            self.pos += 56;
            if self.bits_left() <= 0 {
                return None;
            }
        }
    }

    /// Skip `m` complete FS codewords (i.e. advance past the m-th 1-bit): popcount 57-bit
    /// windows, then a 6-step binary select inside the final window.
    #[inline]
    pub fn skip_fs(&mut self, mut m: u32) -> bool {
        while m > 0 {
            if self.bits_left() <= 0 {
                return false;
            }
            let w = self.window() & !0x7Fu64; // 57 valid bits
            let c = w.count_ones();
            if c < m {
                m -= c;
                self.pos += 57;
            } else {
                self.pos += select_msb(w, m) as u64 + 1;
                return true;
            }
        }
        true
    }
}

/// Index (from the MSB) of the m-th set bit of `v` (1 <= m <= popcount(v)).
#[inline(always)]
pub fn select_msb(mut v: u64, mut m: u32) -> u32 {
    let mut p = 0u32;
    let mut s = 32u32;
    while s > 0 {
        let c = (v >> (64 - s)).count_ones();
        if c < m {
            m -= c;
            p += s;
            v <<= s;
        }
        s >>= 1;
    }
    p
}

#[inline]
pub fn be_u16(b: &[u8], o: usize) -> u16 {
    u16::from_be_bytes([b[o], b[o + 1]])
}
#[inline]
pub fn be_u32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
pub fn be_u64(b: &[u8], o: usize) -> u64 {
    u64::from_be_bytes(b[o..o + 8].try_into().unwrap())
}
/// GRIB2 signed integers are sign-magnitude (MSB = sign).
#[inline]
pub fn sm_i16(b: &[u8], o: usize) -> i16 {
    let v = be_u16(b, o);
    let m = (v & 0x7FFF) as i16;
    if v & 0x8000 != 0 { -m } else { m }
}
#[inline]
pub fn sm_i32(b: &[u8], o: usize) -> i32 {
    let v = be_u32(b, o);
    let m = (v & 0x7FFF_FFFF) as i32;
    if v & 0x8000_0000 != 0 { -m } else { m }
}
/// Sign-magnitude integer of arbitrary octet width (Template 7.3 extra descriptors).
#[inline]
pub fn sm_n(b: &[u8], o: usize, n: usize) -> i64 {
    let mut v: u64 = 0;
    for i in 0..n {
        v = (v << 8) | b[o + i] as u64;
    }
    let sign = 1u64 << (8 * n - 1);
    if v & sign != 0 { -((v & !sign) as i64) } else { v as i64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn read_cross_boundaries() {
        let data = [0b1010_1100u8, 0b0101_0011, 0xFF, 0x00, 0x12, 0x34, 0x56, 0x78, 0x9A];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(3), 0b101);
        assert_eq!(r.read(7), 0b0110001);
        assert_eq!(r.read(14), 0b01_0011_1111_1111);
        assert_eq!(r.read(32), 0x0012_3456);
        assert_eq!(r.read(32), 0x789A_0000);
    }
    #[test]
    fn fs_codes() {
        // 1 | 001 | 0000000001 | 01
        let data = [0b1001_0000u8, 0b0000_0101, 0];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_fs(), Some(0));
        assert_eq!(r.read_fs(), Some(2));
        assert_eq!(r.read_fs(), Some(9));
        assert_eq!(r.read_fs(), Some(1));
        let mut s = BitReader::new(&data);
        assert!(s.skip_fs(3));
        assert_eq!(s.pos(), r.pos() - 2);
    }
}
