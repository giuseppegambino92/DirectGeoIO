//! CPU backend (reference + baseline): pread the selected messages, decode with the bit-exact
//! scalar decoders, single-threaded or one field per thread.

use crate::backend::{DType, DecodeOptions, GeoComputeBackend, Timings};
use crate::decode;
use crate::grib2::{Field, Result};
use std::path::Path;
use std::time::Instant;

pub struct CpuBackend {
    pub threads: usize,
}

pub struct CpuTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl CpuBackend {
    pub fn single() -> Self {
        CpuBackend { threads: 1 }
    }
    pub fn multi() -> Self {
        CpuBackend { threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) }
    }
}

impl GeoComputeBackend for CpuBackend {
    type Tensor = CpuTensor;
    fn name(&self) -> String {
        format!("cpu-rust x{}", self.threads)
    }
    fn decode(&self, path: &Path, fields: &[Field], opts: &DecodeOptions) -> Result<(CpuTensor, Timings)> {
        assert_eq!(opts.dtype, DType::F32, "CPU backend emits float32");
        let t0 = Instant::now();
        let bytes = std::fs::read(path)?;
        let io_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let n: usize = fields.iter().map(|f| f.n_points() as usize).sum();
        let mut out = vec![0f32; n];
        let mut slices = Vec::with_capacity(fields.len());
        let mut rest = &mut out[..];
        for f in fields {
            let (a, b) = rest.split_at_mut(f.n_points() as usize);
            slices.push(a);
            rest = b;
        }
        let work = |f: &Field, dst: &mut [f32]| -> Result<()> {
            let v = decode::decode_f64(f, &bytes)?;
            for (d, s) in dst.iter_mut().zip(v) {
                *d = if s.is_nan() { opts.missing } else { s as f32 };
            }
            Ok(())
        };
        if self.threads <= 1 {
            for (f, d) in fields.iter().zip(slices) {
                work(f, d)?;
            }
        } else {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let jobs: Vec<std::sync::Mutex<Option<&mut [f32]>>> = slices.into_iter().map(|s| std::sync::Mutex::new(Some(s))).collect();
            let errs = std::sync::Mutex::new(Vec::new());
            std::thread::scope(|s| {
                for _ in 0..self.threads.min(fields.len()) {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= fields.len() {
                            break;
                        }
                        let d = jobs[i].lock().unwrap().take().unwrap();
                        if let Err(e) = work(&fields[i], d) {
                            errs.lock().unwrap().push(e);
                        }
                    });
                }
            });
            if let Some(e) = errs.into_inner().unwrap().pop() {
                return Err(e);
            }
        }
        let compute_s = t1.elapsed().as_secs_f64();
        Ok((
            CpuTensor { data: out, shape: crate::backend::shape_of(fields) },
            Timings { plan_s: 0.0, io_s, compute_s, total_s: t0.elapsed().as_secs_f64(), bytes_read: bytes.len() as u64, values: n as u64 },
        ))
    }
}
