//! Unified backend interface implemented by the CPU, Metal and CUDA engines.

use crate::grib2::{Field, Result};
use crate::plan::IoStrategy;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
}
impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
        }
    }
}

/// Where the sequential CCSDS block-boundary walk (aec pass 1) runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AecScan {
    /// one GPU thread per field (best throughput for very large batches)
    Gpu,
    /// CPU threads walk block headers in place in unified memory (lowest latency)
    Cpu,
    /// Cpu unless the batch has many more CCSDS fields than CPU cores
    Auto,
}

#[derive(Clone, Copy, Debug)]
pub struct DecodeOptions {
    pub dtype: DType,
    /// Reproduce float32(ecCodes float64) bit-for-bit (float-float arithmetic). When false a
    /// single-FMA float32 evaluation is used (<= ~1 ulp difference).
    pub exact: bool,
    /// Value stored at bitmap-masked grid points.
    pub missing: f32,
    pub io: IoStrategy,
    pub aec_scan: AecScan,
}
impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions { dtype: DType::F32, exact: true, missing: f32::NAN, io: IoStrategy::Auto, aec_scan: AecScan::Auto }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// metadata scan + plan build (CPU)
    pub plan_s: f64,
    /// storage -> device transfer
    pub io_s: f64,
    /// GPU (or CPU) decode kernels
    pub compute_s: f64,
    /// wall-clock end to end
    pub total_s: f64,
    pub bytes_read: u64,
    pub values: u64,
}

/// Tensor shape helper: (n_fields, nj, ni) for regular lat/lon grids.
pub fn shape_of(fields: &[Field]) -> Vec<usize> {
    if fields.is_empty() {
        return vec![0];
    }
    let g = &fields[0].grid;
    let same = fields.iter().all(|f| f.grid.n_points == g.n_points && f.grid.ni == g.ni && f.grid.nj == g.nj);
    if same && g.ni > 0 && g.nj > 0 && (g.ni as u64 * g.nj as u64) == g.n_points as u64 {
        vec![fields.len(), g.nj as usize, g.ni as usize]
    } else if fields.iter().all(|f| f.grid.n_points == g.n_points) {
        vec![fields.len(), g.n_points as usize]
    } else {
        vec![fields.iter().map(|f| f.grid.n_points as usize).sum()]
    }
}

pub trait GeoComputeBackend {
    type Tensor;
    fn name(&self) -> String;
    fn decode(&self, path: &Path, fields: &[Field], opts: &DecodeOptions) -> Result<(Self::Tensor, Timings)>;
}
