//! # directgeoio-cuda
//!
//! NVIDIA backend: GRIB2 payloads are read with GPUDirect Storage (`cuFileRead` straight into
//! device memory) when the nvidia-fs driver is available, otherwise through double-buffered
//! pinned host staging; decoding runs in the CUDA ports of the Metal kernels.
#![cfg(target_os = "linux")]

use directgeoio_core::aec::{self, AecParams, Cds};
use directgeoio_core::backend::{shape_of, AecScan, DType, DecodeOptions, GeoComputeBackend, Timings};
use directgeoio_core::grib2::{Error, Field, Result};
use directgeoio_core::plan::{self, DgField, DgParams, DgStatus, Plan, DG_KIND_CCSDS};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

#[repr(C)]
#[derive(Default)]
struct IoStats {
    seconds: f64,
    used_gds: i32,
    pad: i32,
    bytes: u64,
}

#[repr(C)]
#[derive(Default)]
struct CPlan {
    nf: u32,
    max_points: u32,
    max_words: u32,
    max_cds: u32,
    max_rsi: u32,
    max_groups: u32,
    rank_len: u32,
    groups_len: u32,
    scan1_len: u32,
    scan2_len: u32,
    codes_len: u32,
    cds_len: u32,
    n_bitmap: u32,
    n_ccsds: u32,
    n_complex: u32,
    gpu_aec_scan: u32,
    mm_len: u32,
    max_mm_values: u32,
    max_values: u32,
    tiles_len: u32,
}

extern "C" {
    fn dgc_init(device: i32, name: *mut c_char, n: usize, err: *mut c_char, errlen: usize) -> i32;
    fn dgc_malloc(n: u64) -> *mut c_void;
    fn dgc_free(p: *mut c_void);
    fn dgc_host_alloc(n: u64) -> *mut c_void;
    fn dgc_host_free(p: *mut c_void);
    fn dgc_d2h(dst: *mut c_void, src: *const c_void, n: u64, err: *mut c_char, errlen: usize) -> i32;
    fn dgc_load(
        path: *const c_char,
        dev: *mut c_void,
        file_off: *const u64,
        buf_off: *const u64,
        len: *const u64,
        n: usize,
        force_posix: i32,
        host_mirror: *mut c_void,
        st: *mut IoStats,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn dgc_decode(
        p: *const CPlan,
        fields: *const DgField,
        raw: *const c_void,
        out: *mut c_void,
        params: *const DgParams,
        cds_host: *const Cds,
        status_host: *const DgStatus,
        status_out: *mut DgStatus,
        gpu_s: *mut f64,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
}

fn cerr(b: &[c_char]) -> String {
    unsafe { CStr::from_ptr(b.as_ptr()).to_string_lossy().into_owned() }
}

/// Decoded fields in CUDA device memory.
pub struct CudaTensor {
    ptr: *mut c_void,
    pub len: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// true when the payload went storage -> GPU via GPUDirect Storage
    pub used_gds: bool,
}
unsafe impl Send for CudaTensor {}
unsafe impl Sync for CudaTensor {}
impl CudaTensor {
    pub fn device_ptr(&self) -> u64 {
        self.ptr as u64
    }
    pub fn copy_to_host(&self, dst: &mut [f32]) -> Result<()> {
        assert_eq!(self.dtype, DType::F32);
        let mut err = [0 as c_char; 256];
        let n = dst.len().min(self.len) as u64 * 4;
        if unsafe { dgc_d2h(dst.as_mut_ptr() as *mut c_void, self.ptr, n, err.as_mut_ptr(), err.len()) } != 0 {
            return Err(Error::Unsupported(cerr(&err)));
        }
        Ok(())
    }
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut v = vec![0f32; self.len];
        self.copy_to_host(&mut v)?;
        Ok(v)
    }
}
impl Drop for CudaTensor {
    fn drop(&mut self) {
        unsafe { dgc_free(self.ptr) }
    }
}

struct DevBuf(*mut c_void);
impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { dgc_free(self.0) }
    }
}
struct HostBuf(*mut c_void);
impl Drop for HostBuf {
    fn drop(&mut self) {
        unsafe { dgc_host_free(self.0) }
    }
}

pub struct CudaBackend {
    name: String,
    lock: Mutex<()>,
    /// Never use GPUDirect Storage (POSIX + pinned staging only).
    pub force_posix: bool,
}

impl CudaBackend {
    pub fn new(device: i32) -> Result<Self> {
        let mut name = [0 as c_char; 256];
        let mut err = [0 as c_char; 256];
        if unsafe { dgc_init(device, name.as_mut_ptr(), name.len(), err.as_mut_ptr(), err.len()) } != 0 {
            return Err(Error::Unsupported(format!("CUDA init: {}", cerr(&err))));
        }
        Ok(CudaBackend { name: cerr(&name), lock: Mutex::new(()), force_posix: std::env::var("DGIO_DISABLE_GDS").is_ok() })
    }
    pub fn device_name(&self) -> String {
        self.name.clone()
    }

    fn cpu_scan(plan: &Plan, host: &[u8], cds: &mut [Cds], status: &mut [DgStatus]) -> Result<()> {
        for f in plan.fields.iter().filter(|f| f.kind == DG_KIND_CCSDS) {
            let p = AecParams {
                nbits: f.nbits,
                block_size: f.block_size,
                rsi: f.rsi,
                id_len: f.id_len,
                preprocess: f.aec_flags & 1 != 0,
                pad_rsi: f.aec_flags & 2 != 0,
                n: f.n_values,
            };
            let mut v = Vec::with_capacity(f.aux_cap as usize);
            aec::scan_cds(&host[f.data_off as usize..(f.data_off + f.data_len) as usize], &p, &mut v)
                .map_err(|e| Error::Format(e.0))?;
            cds[f.aux_off as usize..f.aux_off as usize + v.len()].copy_from_slice(&v);
            status[f.field_index as usize].count = v.len() as u32;
        }
        Ok(())
    }
}

impl GeoComputeBackend for CudaBackend {
    type Tensor = CudaTensor;
    fn name(&self) -> String {
        format!("cuda ({})", self.name)
    }
    fn decode(&self, path: &Path, fields: &[Field], opts: &DecodeOptions) -> Result<(CudaTensor, Timings)> {
        let _g = self.lock.lock().unwrap();
        let t0 = Instant::now();
        let plan = plan::build(fields, opts.io)?;
        // CPU boundary walk is only possible when bytes pass through host memory (no GDS)
        let cpu_scan = plan.n_ccsds > 0
            && match opts.aec_scan {
                AecScan::Cpu => true,
                AecScan::Gpu => false,
                AecScan::Auto => plan.n_ccsds <= 64,
            };
        let plan_s = t0.elapsed().as_secs_f64();
        let raw = DevBuf(unsafe { dgc_malloc(plan.raw_len) });
        let out_bytes = plan.out_len * opts.dtype.size() as u64;
        let out = unsafe { dgc_malloc(out_bytes) };
        if raw.0.is_null() || out.is_null() {
            return Err(Error::Unsupported("cudaMalloc failed".into()));
        }
        let mirror = if cpu_scan { Some(HostBuf(unsafe { dgc_host_alloc(plan.raw_len) })) } else { None };
        let fo: Vec<u64> = plan.io.iter().map(|r| r.file_offset).collect();
        let bo: Vec<u64> = plan.io.iter().map(|r| r.buf_offset).collect();
        let ln: Vec<u64> = plan.io.iter().map(|r| r.len).collect();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut st = IoStats::default();
        let mut err = [0 as c_char; 512];
        let rc = unsafe {
            dgc_load(
                cpath.as_ptr(),
                raw.0,
                fo.as_ptr(),
                bo.as_ptr(),
                ln.as_ptr(),
                fo.len(),
                self.force_posix as i32,
                mirror.as_ref().map(|m| m.0).unwrap_or(std::ptr::null_mut()),
                &mut st,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            unsafe { dgc_free(out) };
            return Err(Error::Unsupported(format!("storage load: {}", cerr(&err))));
        }
        let t1 = Instant::now();
        let nf = plan.n_fields();
        let mut status = vec![DgStatus::default(); nf];
        let mut cds_host: Vec<Cds> = Vec::new();
        if let Some(m) = &mirror {
            cds_host = vec![Cds::default(); plan.cds_len as usize];
            let host = unsafe { std::slice::from_raw_parts(m.0 as *const u8, plan.raw_len as usize) };
            Self::cpu_scan(&plan, host, &mut cds_host, &mut status)?;
        }
        let cp = CPlan {
            nf: nf as u32,
            max_points: plan.max_points,
            max_words: plan.max_words,
            max_cds: plan.max_cds,
            max_rsi: plan.max_rsi,
            max_groups: plan.max_groups,
            rank_len: plan.rank_len,
            groups_len: plan.groups_len,
            scan1_len: plan.scan1_len,
            scan2_len: plan.scan2_len,
            codes_len: plan.codes_len,
            cds_len: plan.cds_len,
            n_bitmap: plan.n_bitmap,
            n_ccsds: plan.n_ccsds,
            n_complex: plan.n_complex,
            gpu_aec_scan: (!cpu_scan) as u32,
            mm_len: plan.mm_len,
            max_mm_values: plan.max_mm_values,
            max_values: plan.max_values,
            tiles_len: plan.tiles_len,
        };
        let params = DgParams {
            n_fields: nf as u32,
            out_half: (opts.dtype == DType::F16) as u32,
            exact: opts.exact as u32,
            missing: opts.missing,
            ..Default::default()
        };
        let mut status_out = vec![DgStatus::default(); nf];
        let mut gpu_s = 0.0;
        let rc = unsafe {
            dgc_decode(
                &cp,
                plan.fields.as_ptr(),
                raw.0,
                out,
                &params,
                if cpu_scan { cds_host.as_ptr() } else { std::ptr::null() },
                if cpu_scan { status.as_ptr() } else { std::ptr::null() },
                status_out.as_mut_ptr(),
                &mut gpu_s,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            unsafe { dgc_free(out) };
            return Err(Error::Unsupported(format!("CUDA decode: {}", cerr(&err))));
        }
        if let Some((i, s)) = status_out.iter().enumerate().find(|(_, s)| s.error != 0) {
            unsafe { dgc_free(out) };
            return Err(Error::Format(format!("field {i}: GPU decode error {}", s.error)));
        }
        let compute_s = t1.elapsed().as_secs_f64();
        let t = CudaTensor { ptr: out, len: plan.out_len as usize, dtype: opts.dtype, shape: shape_of(fields), used_gds: st.used_gds != 0 };
        Ok((
            t,
            Timings { plan_s, io_s: st.seconds, compute_s, total_s: t0.elapsed().as_secs_f64(), bytes_read: st.bytes, values: plan.out_len },
        ))
    }
}
