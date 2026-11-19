//! # directgeoio-metal
//!
//! Apple Silicon backend. GRIB2 payload byte ranges are loaded from storage straight into an
//! `MTLBuffer` with `MTLIOCommandQueue` (Metal fast resource loading — no CPU-side staging
//! buffer or memcpy), then decoded entirely by Metal compute kernels into a
//! `storageModeShared` output buffer that is exported zero-copy (unified memory) via DLPack.
#![cfg(target_os = "macos")]

use directgeoio_core::backend::{shape_of, AecScan, DType, DecodeOptions, GeoComputeBackend, Timings};
use directgeoio_core::grib2::{Error, Field, Result};
use directgeoio_core::aec::{self, AecParams, Cds};
use directgeoio_core::plan::{self, DgField, DgParams, DgStatus, Plan, DG_KIND_CCSDS, DG_ERR_BITMAP, DG_ERR_GROUPS, DG_ERR_TRUNCATED, DG_SCAN_THREADS, DG_SCAN_TILE};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/directgeoio.metallib"));

#[allow(dead_code)]
extern "C" {
    fn dg_mtl_create(lib: *const c_void, len: usize, err: *mut c_char, n: usize) -> *mut c_void;
    fn dg_mtl_destroy(ctx: *mut c_void);
    fn dg_mtl_device_name(ctx: *mut c_void, out: *mut c_char, n: usize) -> usize;
    fn dg_mtl_recommended_working_set(ctx: *mut c_void) -> u64;
    fn dg_mtl_buffer_new(ctx: *mut c_void, len: u64, private: i32) -> *mut c_void;
    fn dg_mtl_buffer_free(b: *mut c_void);
    fn dg_mtl_buffer_contents(b: *mut c_void) -> *mut c_void;
    fn dg_mtl_buffer_length(b: *mut c_void) -> u64;
    fn dg_mtl_buffer_gpu_address(b: *mut c_void) -> u64;
    fn dg_mtl_io_open(ctx: *mut c_void, path: *const c_char, err: *mut c_char, n: usize) -> *mut c_void;
    fn dg_mtl_io_close(fh: *mut c_void);
    fn dg_mtl_io_load(
        ctx: *mut c_void,
        fh: *mut c_void,
        dst: *mut c_void,
        file_off: *const u64,
        buf_off: *const u64,
        len: *const u64,
        n: usize,
        max_chunk: u64,
        seconds: *mut f64,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn dg_mtl_cmd_begin(ctx: *mut c_void) -> *mut c_void;
    fn dg_mtl_dispatch(
        cmd: *mut c_void,
        kernel: *const c_char,
        bufs: *const *mut c_void,
        offs: *const u64,
        nbufs: u32,
        bytes: *const c_void,
        bytes_len: u32,
        bytes_index: u32,
        gx: u32,
        gy: u32,
        gz: u32,
        tgx: u32,
        mode: i32,
        err: *mut c_char,
        errlen: usize,
    ) -> i32;
    fn dg_mtl_cmd_end_wait(cmd: *mut c_void, gpu_s: *mut f64, err: *mut c_char, errlen: usize) -> i32;
}

const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
extern "C" {
    fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32;
}

fn err_str(b: &[c_char]) -> String {
    unsafe { CStr::from_ptr(b.as_ptr()).to_string_lossy().into_owned() }
}

/// Owned, reference-counted `id<MTLBuffer>`.
pub struct MtlBuffer {
    raw: *mut c_void,
    len: u64,
}
unsafe impl Send for MtlBuffer {}
unsafe impl Sync for MtlBuffer {}
impl MtlBuffer {
    pub fn contents(&self) -> *mut u8 {
        unsafe { dg_mtl_buffer_contents(self.raw) as *mut u8 }
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The `id<MTLBuffer>` object pointer (DLPack kDLMetal data pointer convention).
    pub fn as_ptr(&self) -> *mut c_void {
        self.raw
    }
    pub fn gpu_address(&self) -> u64 {
        unsafe { dg_mtl_buffer_gpu_address(self.raw) }
    }
}
impl Drop for MtlBuffer {
    fn drop(&mut self) {
        unsafe { dg_mtl_buffer_free(self.raw) }
    }
}

/// Decoded fields resident in a shared-storage MTLBuffer (unified memory).
type Recycler = std::sync::Arc<Mutex<Vec<MtlBuffer>>>;

pub struct MetalTensor {
    buffer: Option<MtlBuffer>,
    recycle: Option<Recycler>,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub len: usize,
}
impl Drop for MetalTensor {
    fn drop(&mut self) {
        // Output buffers are recycled: GPU first-touch page faults on fresh multi-GB shared
        // buffers dominate otherwise. Released only once every DLPack consumer is gone.
        if let (Some(b), Some(r)) = (self.buffer.take(), &self.recycle) {
            let mut v = r.lock().unwrap();
            v.push(b);
            if v.len() > 3 {
                v.remove(0);
            }
        }
    }
}
impl MetalTensor {
    pub fn buffer(&self) -> &MtlBuffer {
        self.buffer.as_ref().unwrap()
    }
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32);
        unsafe { std::slice::from_raw_parts(self.buffer().contents() as *const f32, self.len) }
    }
    pub fn as_u16(&self) -> &[u16] {
        assert_eq!(self.dtype, DType::F16);
        unsafe { std::slice::from_raw_parts(self.buffer().contents() as *const u16, self.len) }
    }
    pub fn data_ptr(&self) -> *mut u8 {
        self.buffer().contents()
    }
}

#[derive(Default)]
struct Pool {
    bufs: Vec<(&'static str, MtlBuffer)>,
}

pub struct MetalBackend {
    ctx: *mut c_void,
    pool: Mutex<Pool>,
    out_pool: Recycler,
    /// split DMA ranges into chunks of this size so the concurrent IO queue can overlap them
    pub io_chunk: u64,
}
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        self.pool.lock().unwrap().bufs.clear();
        self.out_pool.lock().unwrap().clear();
        unsafe { dg_mtl_destroy(self.ctx) }
    }
}

struct Cmd<'a> {
    raw: std::cell::Cell<*mut c_void>,
    be: &'a MetalBackend,
    /// DGIO_PROFILE=1: one command buffer per dispatch, GPU time per kernel
    profile: Option<std::cell::RefCell<Vec<(String, f64)>>>,
}

#[derive(Clone, Copy)]
struct B<'a>(&'a MtlBuffer, u64);

impl<'a> Cmd<'a> {
    #[allow(clippy::too_many_arguments)]
    fn dispatch(&self, kernel: &str, bufs: &[B], params: Option<(&DgParams, u32)>, grid: [u32; 3], tg: u32, groups: bool) -> Result<()> {
        let name = CString::new(kernel).unwrap();
        let ptrs: Vec<*mut c_void> = bufs.iter().map(|b| b.0.raw).collect();
        let offs: Vec<u64> = bufs.iter().map(|b| b.1).collect();
        let mut err = [0 as c_char; 256];
        let (bp, bl, bi) = match params {
            Some((p, i)) => (p as *const DgParams as *const c_void, std::mem::size_of::<DgParams>() as u32, i),
            None => (std::ptr::null(), 0, 0),
        };
        let rc = unsafe {
            dg_mtl_dispatch(
                self.raw.get(),
                name.as_ptr(),
                ptrs.as_ptr(),
                offs.as_ptr(),
                ptrs.len() as u32,
                bp,
                bl,
                bi,
                grid[0],
                grid[1],
                grid[2],
                tg,
                groups as i32,
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if rc != 0 {
            return Err(Error::Unsupported(format!("Metal dispatch {kernel}: {}", err_str(&err))));
        }
        if let Some(prof) = &self.profile {
            let mut gpu = 0.0;
            let rc = unsafe { dg_mtl_cmd_end_wait(self.raw.get(), &mut gpu, err.as_mut_ptr(), err.len()) };
            if rc != 0 {
                return Err(Error::Unsupported(format!("Metal {kernel}: {}", err_str(&err))));
            }
            prof.borrow_mut().push((kernel.to_string(), gpu));
            self.raw.set(unsafe { dg_mtl_cmd_begin(self.be.ctx) });
        }
        Ok(())
    }

    /// Multi-level Blelloch scan of `n` uint32 at `data` (in place). `aux` holds level sums.
    fn scan(&self, data: B, n: u32, inclusive: bool, aux: &MtlBuffer) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let levels = plan::scan_levels(n);
        // aux layout: sums of level l at aoff[l+1]
        let mut aoff = vec![0u64; levels.len() + 1];
        let mut cur = 0u64;
        for l in 0..levels.len() {
            let tiles = levels[l].div_ceil(DG_SCAN_TILE) as u64;
            aoff[l + 1] = cur;
            cur += tiles * 4;
        }
        assert!(cur <= aux.len());
        let at = |l: usize| -> B { if l == 0 { data } else { B(aux, aoff[l]) } };
        for (l, &m) in levels.iter().enumerate() {
            let p = DgParams { n: m, inclusive: (l == 0 && inclusive) as u32, ..Default::default() };
            let tiles = m.div_ceil(DG_SCAN_TILE);
            self.dispatch("scan_tiles", &[at(l), B(aux, aoff[l + 1])], Some((&p, 2)), [tiles, 1, 1], DG_SCAN_THREADS, true)?;
        }
        for l in (0..levels.len().saturating_sub(1)).rev() {
            let p = DgParams { n: levels[l], ..Default::default() };
            self.dispatch("scan_add", &[at(l), B(aux, aoff[l + 1])], Some((&p, 2)), [levels[l], 1, 1], 256, false)?;
        }
        Ok(())
    }
}

fn scan_aux_bytes(n: u32) -> u64 {
    plan::scan_levels(n.max(1)).iter().map(|&m| m.div_ceil(DG_SCAN_TILE) as u64 * 4).sum::<u64>() + 16
}

/// Resolve `AecScan::Auto` for a plan.
pub fn resolve_aec_scan(plan: &Plan, mode: AecScan) -> AecScan {
    match mode {
        AecScan::Auto => {
            let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) as u32;
            if plan.n_ccsds > 16 * cores { AecScan::Gpu } else { AecScan::Cpu }
        }
        m => m,
    }
}

/// CPU boundary walk over payload bytes already resident in a shared MTLBuffer (no copy).
fn cpu_aec_scan(plan: &Plan, raw: &MtlBuffer, cds: &MtlBuffer, status: &MtlBuffer) -> Result<()> {
    let rawb = unsafe { std::slice::from_raw_parts(raw.contents() as *const u8, raw.len() as usize) };
    let cdsb = cds.contents() as usize;
    let stb = status.contents() as usize;
    let jobs: Vec<&DgField> = plan.fields.iter().filter(|f| f.kind == DG_KIND_CCSDS).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let err = Mutex::new(None);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(jobs.len());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                // keep the latency-critical walkers on performance cores
                unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
                let mut v: Vec<Cds> = Vec::new();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let f = jobs[i];
                    let p = AecParams {
                        nbits: f.nbits,
                        block_size: f.block_size,
                        rsi: f.rsi,
                        id_len: f.id_len,
                        preprocess: f.aec_flags & 1 != 0,
                        pad_rsi: f.aec_flags & 2 != 0,
                        n: f.n_values,
                    };
                    let data = &rawb[f.data_off as usize..(f.data_off + f.data_len) as usize];
                    v.clear();
                    if let Err(e) = aec::scan_cds(data, &p, &mut v) {
                        *err.lock().unwrap() = Some(Error::Format(format!("field {}: {}", f.field_index, e.0)));
                        continue;
                    }
                    assert!(v.len() as u32 <= f.aux_cap);
                    // each field owns a disjoint [aux_off, aux_off+aux_cap) slice and status slot
                    unsafe {
                        let dst = (cdsb as *mut Cds).add(f.aux_off as usize);
                        std::ptr::copy_nonoverlapping(v.as_ptr(), dst, v.len());
                        (*(stb as *mut DgStatus).add(f.field_index as usize)).count = v.len() as u32;
                    }
                }
            });
        }
    });
    match err.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        let mut err = [0 as c_char; 512];
        let ctx = unsafe { dg_mtl_create(METALLIB.as_ptr() as *const c_void, METALLIB.len(), err.as_mut_ptr(), err.len()) };
        if ctx.is_null() {
            return Err(Error::Unsupported(format!("Metal init: {}", err_str(&err))));
        }
        Ok(MetalBackend { ctx, pool: Mutex::new(Pool::default()), out_pool: Default::default(), io_chunk: 8 << 20 })
    }

    pub fn device_name(&self) -> String {
        let mut b = [0 as c_char; 256];
        unsafe { dg_mtl_device_name(self.ctx, b.as_mut_ptr(), b.len()) };
        err_str(&b)
    }

    pub fn alloc(&self, len: u64, private: bool) -> Result<MtlBuffer> {
        let raw = unsafe { dg_mtl_buffer_new(self.ctx, len.max(16), private as i32) };
        if raw.is_null() {
            return Err(Error::Unsupported(format!("MTLBuffer allocation of {len} bytes failed")));
        }
        Ok(MtlBuffer { raw, len: unsafe { dg_mtl_buffer_length(raw) } })
    }

    /// Scratch buffers are pooled across decodes (keyed by role, grown on demand).
    fn take(&self, pool: &mut Pool, role: &'static str, len: u64, private: bool) -> Result<MtlBuffer> {
        if let Some(i) = pool.bufs.iter().position(|(r, b)| *r == role && b.len() >= len) {
            return Ok(pool.bufs.swap_remove(i).1);
        }
        pool.bufs.retain(|(r, _)| *r != role);
        self.alloc(len.next_power_of_two().max(4096), private)
    }

    /// Run a 1-D kernel over `n` threads with `bufs` bound at indices 0.. (test/diagnostic use).
    #[doc(hidden)]
    pub fn dispatch_1d(&self, kernel: &str, bufs: &[&MtlBuffer], n: u32) -> Result<f64> {
        let cmd = Cmd { raw: std::cell::Cell::new(unsafe { dg_mtl_cmd_begin(self.ctx) }), be: self, profile: None };
        let b: Vec<B> = bufs.iter().map(|x| B(x, 0)).collect();
        let r = cmd.dispatch(kernel, &b, None, [n, 1, 1], 0, false);
        let mut gpu = 0.0;
        let mut err = [0 as c_char; 256];
        let rc = unsafe { dg_mtl_cmd_end_wait(cmd.raw.get(), &mut gpu, err.as_mut_ptr(), err.len()) };
        r?;
        if rc != 0 {
            return Err(Error::Unsupported(err_str(&err)));
        }
        Ok(gpu)
    }

    /// DMA the plan's byte ranges from `path` into `dst` with MTLIOCommandQueue.
    pub fn load(&self, path: &Path, plan: &Plan, dst: &MtlBuffer) -> Result<f64> {
        let cpath = CString::new(path.as_os_str().to_str().ok_or_else(|| Error::Format("path".into()))?).unwrap();
        let mut err = [0 as c_char; 512];
        let fh = unsafe { dg_mtl_io_open(self.ctx, cpath.as_ptr(), err.as_mut_ptr(), err.len()) };
        if fh.is_null() {
            return Err(Error::Unsupported(format!("MTLIOFileHandle: {}", err_str(&err))));
        }
        let fo: Vec<u64> = plan.io.iter().map(|r| r.file_offset).collect();
        let bo: Vec<u64> = plan.io.iter().map(|r| r.buf_offset).collect();
        let ln: Vec<u64> = plan.io.iter().map(|r| r.len).collect();
        let mut secs = 0.0;
        let rc = unsafe {
            dg_mtl_io_load(self.ctx, fh, dst.raw, fo.as_ptr(), bo.as_ptr(), ln.as_ptr(), fo.len(), self.io_chunk, &mut secs, err.as_mut_ptr(), err.len())
        };
        unsafe { dg_mtl_io_close(fh) };
        if rc != 0 {
            return Err(Error::Unsupported(format!("MTLIO load: {}", err_str(&err))));
        }
        Ok(secs)
    }

    /// Run the decode kernels for `plan` on payload bytes already resident in `raw`.
    /// Returns the tensor and the wall-clock decode time (CPU boundary walk, if any, + GPU).
    pub fn run(&self, plan: &Plan, raw: &MtlBuffer, opts: &DecodeOptions) -> Result<(MetalTensor, f64)> {
        let t_run = Instant::now();
        let nf = plan.n_fields() as u32;
        let mut pool = self.pool.lock().unwrap();
        let fields = self.take(&mut pool, "fields", (plan.fields.len() * std::mem::size_of::<plan::DgField>()) as u64, false)?;
        unsafe {
            std::ptr::copy_nonoverlapping(plan.fields.as_ptr() as *const u8, fields.contents(), plan.fields.len() * std::mem::size_of::<plan::DgField>());
        }
        let status = self.take(&mut pool, "status", nf as u64 * 16, false)?;
        unsafe { std::ptr::write_bytes(status.contents(), 0, nf as usize * 16) };
        let rank = self.take(&mut pool, "rank", plan.rank_len as u64 * 4 + 16, true)?;
        let codes = self.take(&mut pool, "codes", plan.codes_len as u64 * 4 + 16, true)?;
        let scan_mode = resolve_aec_scan(plan, opts.aec_scan);
        let cds = self.take(&mut pool, if scan_mode == AecScan::Cpu { "cds_shared" } else { "cds" }, plan.cds_len as u64 * 8 + 16, scan_mode != AecScan::Cpu)?;
        if plan.n_ccsds > 0 && scan_mode == AecScan::Cpu {
            let t = Instant::now();
            cpu_aec_scan(plan, raw, &cds, &status)?;
            if std::env::var("DGIO_PROFILE").map(|v| v == "1").unwrap_or(false) {
                eprintln!("[dgio-profile] {:<24} {:>8.3} ms  (cpu, {} fields)", "aec_scan (cpu walk)", t.elapsed().as_secs_f64() * 1e3, plan.n_ccsds);
            }
        }
        let gref = self.take(&mut pool, "gref", plan.groups_len as u64 * 4 + 16, true)?;
        let gwid = self.take(&mut pool, "gwid", plan.groups_len as u64 * 4 + 16, true)?;
        let gscan = self.take(&mut pool, "gscan", plan.groups_len as u64 * 8 + 16, true)?;
        let tiles = self.take(&mut pool, "tiles", plan.tiles_len as u64 * 4 + 16, true)?;
        let agg = self.take(&mut pool, "agg", plan.tiles_len as u64 * 8 + 16, true)?;
        let xraw = self.take(&mut pool, "xraw", plan.mm_len as u64 * 4 + 16, true)?;
        let mvalid = self.take(&mut pool, "mvalid", plan.mm_len as u64 * 4 + 16, true)?;
        let aux_n = [plan.rank_len, plan.groups_len * 2, plan.mm_len].into_iter().max().unwrap();
        let aux = self.take(&mut pool, "aux", scan_aux_bytes(aux_n), true)?;
        let out_bytes = plan.out_len * opts.dtype.size() as u64;
        let recycled = {
            let mut v = self.out_pool.lock().unwrap();
            v.iter().position(|b| b.len() >= out_bytes && b.len() <= out_bytes * 2).map(|i| v.remove(i))
        };
        let out = match recycled {
            Some(b) => b,
            None => self.alloc(out_bytes, false)?,
        };

        let profile = std::env::var("DGIO_PROFILE").map(|v| v == "1").unwrap_or(false);
        let cmd = Cmd {
            raw: std::cell::Cell::new(unsafe { dg_mtl_cmd_begin(self.ctx) }),
            be: self,
            profile: profile.then(|| std::cell::RefCell::new(Vec::new())),
        };
        let r = B(raw, 0);
        let fd = B(&fields, 0);
        let p = DgParams {
            n_fields: nf,
            out_half: (opts.dtype == DType::F16) as u32,
            exact: opts.exact as u32,
            missing: opts.missing,
            ..Default::default()
        };
        let res = (|| -> Result<()> {
            if plan.n_bitmap > 0 {
                cmd.dispatch("bitmap_popcount", &[r, fd, B(&rank, 0)], None, [plan.max_words, nf, 1], 0, false)?;
                cmd.scan(B(&rank, 0), plan.rank_len, false, &aux)?;
            }
            if plan.n_ccsds > 0 {
                if scan_mode == AecScan::Gpu {
                    cmd.dispatch("aec_scan", &[r, fd, B(&cds, 0), B(&status, 0)], None, [nf, 1, 1], 1, false)?;
                }
                cmd.dispatch("aec_decode", &[r, fd, B(&cds, 0), B(&status, 0), B(&codes, 0)], None, [plan.max_cds, nf, 1], 0, false)?;
                cmd.dispatch("aec_post", &[fd, B(&codes, 0)], None, [plan.max_rsi, nf, 1], 64, false)?;
            }
            if plan.n_complex > 0 {
                let pg = DgParams { n: plan.groups_len, ..p };
                cmd.dispatch("cplx_groups", &[r, fd, B(&gref, 0), B(&gwid, 0), B(&gscan, 0), B(&aux, 0), B(&status, 0)], Some((&pg, 5)), [plan.max_groups, nf, 1], 0, false)?;
                cmd.scan(B(&gscan, 0), plan.groups_len * 2, false, &aux)?;
                cmd.dispatch("cplx_tile_index", &[fd, B(&gscan, 0), B(&tiles, 0), B(&aux, 0), B(&status, 0)], Some((&pg, 3)), [plan.max_groups, nf, 1], 0, false)?;
                cmd.dispatch(
                    "cplx_unpack",
                    &[r, fd, B(&gref, 0), B(&gwid, 0), B(&gscan, 0), B(&codes, 0), B(&status, 0), B(&aux, 0), B(&xraw, 0), B(&mvalid, 0), B(&tiles, 0)],
                    Some((&pg, 7)),
                    [plan.max_values.div_ceil(1024), nf, 1],
                    256,
                    true,
                )?;
                if plan.mm_len > 0 {
                    // missing-value management: presence ranks by inclusive Blelloch scan, then compaction
                    cmd.scan(B(&mvalid, 0), plan.mm_len, true, &aux)?;
                    cmd.dispatch("cplx_compact", &[r, fd, B(&xraw, 0), B(&mvalid, 0), B(&codes, 0)], None, [plan.max_mm_values, nf, 1], 0, false)?;
                }
                if plan.scan1_len > 0 {
                    // fused segmented spatial-differencing reconstruction (order 1 and 2)
                    let tiles_x = plan.max_values.div_ceil(1024);
                    cmd.dispatch("sd_tile_reduce", &[fd, B(&codes, 0), B(&agg, 0)], None, [tiles_x, nf, 1], 256, true)?;
                    cmd.dispatch("sd_carry", &[fd, B(&agg, 0)], None, [nf, 1, 1], 0, false)?;
                    cmd.dispatch("sd_apply", &[fd, B(&codes, 0), B(&agg, 0)], None, [tiles_x, nf, 1], 256, true)?;
                }
            }
            cmd.dispatch(
                "unpack_dequant_expand",
                &[r, fd, B(&rank, 0), B(&codes, 0), B(&status, 0), B(&out, 0), B(&aux, 0), B(&mvalid, 0)],
                Some((&p, 6)),
                [plan.max_points, nf, 1],
                0,
                false,
            )?;
            Ok(())
        })();
        let mut gpu = 0.0;
        let mut err = [0 as c_char; 512];
        let rc = unsafe { dg_mtl_cmd_end_wait(cmd.raw.get(), &mut gpu, err.as_mut_ptr(), err.len()) };
        if let Some(prof) = &cmd.profile {
            let mut agg: Vec<(String, f64, u32)> = Vec::new();
            for (k, t) in prof.borrow().iter() {
                match agg.iter_mut().find(|a| &a.0 == k) {
                    Some(a) => { a.1 += t; a.2 += 1 }
                    None => agg.push((k.clone(), *t, 1)),
                }
            }
            for (k, t, n) in agg {
                eprintln!("[dgio-profile] {k:<24} {:>8.3} ms  ({n} dispatch{})", t * 1e3, if n > 1 { "es" } else { "" });
            }
        }
        res?;
        if rc != 0 {
            return Err(Error::Unsupported(format!("Metal command buffer: {}", err_str(&err))));
        }
        let st = unsafe { std::slice::from_raw_parts(status.contents() as *const DgStatus, nf as usize) };
        for (i, s) in st.iter().enumerate() {
            match s.error {
                0 => {}
                DG_ERR_TRUNCATED => return Err(Error::Format(format!("field {i}: CCSDS stream truncated"))),
                DG_ERR_GROUPS => return Err(Error::Format(format!("field {i}: complex group table inconsistent"))),
                DG_ERR_BITMAP => return Err(Error::Format(format!("field {i}: bitmap marks more points than packed values"))),
                e => return Err(Error::Format(format!("field {i}: GPU decode error {e}"))),
            }
        }
        for (role, b) in [
            ("fields", fields),
            ("status", status),
            ("rank", rank),
            ("codes", codes),
            (if scan_mode == AecScan::Cpu { "cds_shared" } else { "cds" }, cds),
            ("gref", gref),
            ("gwid", gwid),
            ("gscan", gscan),
            ("aux", aux),
            ("xraw", xraw),
            ("tiles", tiles),
            ("agg", agg),
            ("mvalid", mvalid),
        ] {
            pool.bufs.push((role, b));
        }
        let len = plan.out_len as usize;
        let _ = gpu;
        let t = MetalTensor { buffer: Some(out), recycle: Some(self.out_pool.clone()), dtype: opts.dtype, shape: vec![len], len };
        Ok((t, t_run.elapsed().as_secs_f64()))
    }
}

impl GeoComputeBackend for MetalBackend {
    type Tensor = MetalTensor;
    fn name(&self) -> String {
        format!("metal ({})", self.device_name())
    }
    fn decode(&self, path: &Path, fields: &[Field], opts: &DecodeOptions) -> Result<(MetalTensor, Timings)> {
        let t0 = Instant::now();
        let plan = plan::build(fields, opts.io)?;
        let plan_s = t0.elapsed().as_secs_f64();
        let raw = {
            let mut pool = self.pool.lock().unwrap();
            // shared storage: unified memory lets the CPU boundary walk read the DMA'd bytes in place
            self.take(&mut pool, "raw", plan.raw_len, false)?
        };
        let io_s = self.load(path, &plan, &raw)?;
        let (mut t, compute_s) = self.run(&plan, &raw, opts)?;
        self.pool.lock().unwrap().bufs.push(("raw", raw));
        t.shape = shape_of(fields);
        let bytes_read = plan.io.iter().map(|r| r.len).sum();
        Ok((t, Timings { plan_s, io_s, compute_s, total_s: t0.elapsed().as_secs_f64(), bytes_read, values: plan.out_len }))
    }
}
