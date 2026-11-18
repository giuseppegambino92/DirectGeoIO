//! Minimal NumPy `.npy` (v1/v2, little-endian float64/float32) reader used by tests/benchmarks.

use std::io::Read;

pub fn read_f64(path: &std::path::Path) -> std::io::Result<Vec<f64>> {
    let mut b = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut b)?;
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {m}", path.display()));
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(bad("not an npy file"));
    }
    let (hlen, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[start..start + hlen]).map_err(|_| bad("header"))?;
    let data = &b[start + hlen..];
    if header.contains("'<f8'") {
        Ok(data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    } else if header.contains("'<f4'") {
        Ok(data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect())
    } else {
        Err(bad("unsupported dtype"))
    }
}
