//! # directgeoio
//!
//! Public Rust API + C ABI (`include/directgeoio.h`) + DLPack export.

use directgeoio_core::backend::{AecScan, DType, DecodeOptions, GeoComputeBackend, Timings};
use directgeoio_core::cpu::CpuBackend;
use directgeoio_core::grib2::{Error, Field, Result};
use directgeoio_core::plan::IoStrategy;
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub use directgeoio_core as core;

pub const DL_CPU: i32 = 1;
pub const DL_CUDA: i32 = 2;
pub const DL_METAL: i32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum Backend {
    Auto = 0,
    Cpu = 1,
    CpuMt = 2,
    Metal = 3,
    Cuda = 4,
}

pub enum Storage {
    Host(Vec<f32>),
    #[cfg(target_os = "macos")]
    Metal(directgeoio_metal::MetalTensor),
    #[cfg(feature = "cuda")]
    Cuda(directgeoio_cuda::CudaTensor),
}

pub struct Tensor {
    pub storage: Storage,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub timings: Timings,
}
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    /// Native DLPack device type.
    pub fn device_type(&self) -> i32 {
        match &self.storage {
            Storage::Host(_) => DL_CPU,
            #[cfg(target_os = "macos")]
            Storage::Metal(_) => DL_METAL,
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => DL_CUDA,
        }
    }
    /// CPU-addressable pointer (None for discrete-GPU memory).
    pub fn host_ptr(&self) -> Option<*mut u8> {
        match &self.storage {
            Storage::Host(v) => Some(v.as_ptr() as *mut u8),
            #[cfg(target_os = "macos")]
            Storage::Metal(t) => Some(t.data_ptr()),
            #[cfg(feature = "cuda")]
            Storage::Cuda(_) => None,
        }
    }
    pub fn device_handle(&self) -> *mut c_void {
        match &self.storage {
            Storage::Host(v) => v.as_ptr() as *mut c_void,
            #[cfg(target_os = "macos")]
            Storage::Metal(t) => t.buffer().as_ptr(),
            #[cfg(feature = "cuda")]
            Storage::Cuda(t) => t.device_ptr() as *mut c_void,
        }
    }
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != DType::F32 {
            return None;
        }
        self.host_ptr().map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, self.numel()) })
    }
}

pub struct File {
    pub path: PathBuf,
    pub fields: Vec<Field>,
}

impl File {
    pub fn open(path: impl AsRef<Path>) -> Result<File> {
        let path = path.as_ref().to_path_buf();
        let f = std::fs::File::open(&path)?;
        let fields = directgeoio_core::scan(&f)?;
        Ok(File { path, fields })
    }

    pub fn decode(&self, indices: Option<&[usize]>, backend: Backend, opts: &DecodeOptions) -> Result<Tensor> {
        let sel: Vec<Field> = match indices {
            None => self.fields.clone(),
            Some(ix) => ix
                .iter()
                .map(|&i| self.fields.get(i).cloned().ok_or_else(|| Error::Format(format!("field index {i} out of range"))))
                .collect::<Result<_>>()?,
        };
        decode_fields(&self.path, &sel, backend, opts)
    }
}

#[cfg(target_os = "macos")]
fn metal() -> Result<&'static directgeoio_metal::MetalBackend> {
    static M: OnceLock<std::result::Result<directgeoio_metal::MetalBackend, String>> = OnceLock::new();
    M.get_or_init(|| directgeoio_metal::MetalBackend::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| Error::Unsupported(e.clone()))
}

#[cfg(feature = "cuda")]
fn cuda() -> Result<&'static directgeoio_cuda::CudaBackend> {
    static C: OnceLock<std::result::Result<directgeoio_cuda::CudaBackend, String>> = OnceLock::new();
    C.get_or_init(|| directgeoio_cuda::CudaBackend::new(0).map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| Error::Unsupported(e.clone()))
}

pub fn resolve(backend: Backend) -> Backend {
    if backend != Backend::Auto {
        return backend;
    }
    #[cfg(target_os = "macos")]
    if metal().is_ok() {
        return Backend::Metal;
    }
    #[cfg(feature = "cuda")]
    if cuda().is_ok() {
        return Backend::Cuda;
    }
    Backend::CpuMt
}

pub fn device_name(backend: Backend) -> String {
    match resolve(backend) {
        #[cfg(target_os = "macos")]
        Backend::Metal => metal().map(|m| m.device_name()).unwrap_or_default(),
        #[cfg(feature = "cuda")]
        Backend::Cuda => cuda().map(|c| c.device_name()).unwrap_or_default(),
        Backend::Cpu => "cpu (1 thread)".into(),
        _ => format!("cpu ({} threads)", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)),
    }
}

pub fn decode_fields(path: &Path, fields: &[Field], backend: Backend, opts: &DecodeOptions) -> Result<Tensor> {
    match resolve(backend) {
        #[cfg(target_os = "macos")]
        Backend::Metal => {
            let (t, timings) = metal()?.decode(path, fields, opts)?;
            Ok(Tensor { dtype: t.dtype, shape: t.shape.clone(), storage: Storage::Metal(t), timings })
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            let (t, timings) = cuda()?.decode(path, fields, opts)?;
            Ok(Tensor { dtype: t.dtype, shape: t.shape.clone(), storage: Storage::Cuda(t), timings })
        }
        Backend::Cpu | Backend::CpuMt => {
            if opts.dtype != DType::F32 {
                return Err(Error::Unsupported("CPU backend emits float32 only".into()));
            }
            let be = if resolve(backend) == Backend::Cpu { CpuBackend::single() } else { CpuBackend::multi() };
            let (t, timings) = be.decode(path, fields, opts)?;
            Ok(Tensor { dtype: DType::F32, shape: t.shape, storage: Storage::Host(t.data), timings })
        }
        b => Err(Error::Unsupported(format!("backend {b:?} not available in this build"))),
    }
}

// ======================================================================================
// C ABI
// ======================================================================================

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}
fn set_error(msg: impl ToString) {
    let s = CString::new(msg.to_string().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = s);
}
fn guard<T>(default: T, f: impl FnOnce() -> Result<T>) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(e);
            default
        }
        Err(p) => {
            let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()));
            set_error(format!("internal panic: {}", msg.unwrap_or_default()));
            default
        }
    }
}

#[repr(C)]
pub struct DgioOptions {
    pub dtype: c_int,
    pub exact: i32,
    pub missing_value: f32,
    pub io: c_int,
    pub aec_scan: c_int,
}
impl DgioOptions {
    fn to_rust(&self) -> DecodeOptions {
        DecodeOptions {
            dtype: if self.dtype == 1 { DType::F16 } else { DType::F32 },
            exact: self.exact != 0,
            missing: self.missing_value,
            io: match self.io {
                1 => IoStrategy::Span,
                2 => IoStrategy::Ranges,
                _ => IoStrategy::Auto,
            },
            aec_scan: match self.aec_scan {
                1 => AecScan::Gpu,
                2 => AecScan::Cpu,
                _ => AecScan::Auto,
            },
        }
    }
}

#[repr(C)]
pub struct DgioFieldInfo {
    pub short_name: [c_char; 16],
    pub name: [c_char; 64],
    pub units: [c_char; 24],
    pub level_type: [c_char; 24],
    pub level: f64,
    pub ni: u32,
    pub nj: u32,
    pub n_points: u32,
    pub n_values: u32,
    pub data_template: u16,
    pub bits_per_value: u8,
    pub has_bitmap: u8,
    pub discipline: u8,
    pub category: u8,
    pub number: u8,
    pub level_type_code: u8,
    pub forecast_time: u32,
    pub ref_time: [u16; 6],
    pub centre: u16,
    pub supported: u8,
    pub pad: u8,
    pub lat_first: f64,
    pub lon_first: f64,
    pub lat_last: f64,
    pub lon_last: f64,
    pub di: f64,
    pub dj: f64,
    pub message_offset: u64,
    pub message_length: u64,
}

#[repr(C)]
pub struct DgioTimings {
    pub plan_s: f64,
    pub io_s: f64,
    pub compute_s: f64,
    pub total_s: f64,
    pub bytes_read: u64,
    pub values: u64,
}

fn copy_str<const N: usize>(dst: &mut [c_char; N], s: &str) {
    let b = s.as_bytes();
    let n = b.len().min(N - 1);
    for i in 0..n {
        dst[i] = b[i] as c_char;
    }
    dst[n] = 0;
}

#[no_mangle]
pub extern "C" fn dgio_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn dgio_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn dgio_backend_device_name(backend: c_int) -> *const c_char {
    thread_local!(static NAME: RefCell<CString> = RefCell::new(CString::default()));
    let b = backend_from(backend);
    let s = CString::new(device_name(b)).unwrap_or_default();
    NAME.with(|n| {
        *n.borrow_mut() = s;
        n.borrow().as_ptr()
    })
}

fn backend_from(b: c_int) -> Backend {
    match b {
        1 => Backend::Cpu,
        2 => Backend::CpuMt,
        3 => Backend::Metal,
        4 => Backend::Cuda,
        _ => Backend::Auto,
    }
}

/// # Safety
/// `opts` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn dgio_default_options(opts: *mut DgioOptions) {
    if !opts.is_null() {
        *opts = DgioOptions { dtype: 0, exact: 1, missing_value: f32::NAN, io: 0, aec_scan: 0 };
    }
}

/// # Safety
/// `path` must be a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn dgio_open(path: *const c_char) -> *mut File {
    guard(std::ptr::null_mut(), || {
        if path.is_null() {
            return Err(Error::Format("null path".into()));
        }
        let p = CStr::from_ptr(path).to_str().map_err(|_| Error::Format("path is not UTF-8".into()))?;
        Ok(Box::into_raw(Box::new(File::open(p)?)))
    })
}

/// # Safety
/// `f` must come from `dgio_open` (or be NULL).
#[no_mangle]
pub unsafe extern "C" fn dgio_close(f: *mut File) {
    if !f.is_null() {
        drop(Box::from_raw(f));
    }
}

/// # Safety
/// `f` must be a valid file handle.
#[no_mangle]
pub unsafe extern "C" fn dgio_num_fields(f: *const File) -> usize {
    if f.is_null() { 0 } else { (*f).fields.len() }
}

/// # Safety
/// `f` valid, `out` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_get_field_info(f: *const File, index: usize, out: *mut DgioFieldInfo) -> c_int {
    guard(-1, || {
        let file = f.as_ref().ok_or_else(|| Error::Format("null file".into()))?;
        let x = file.fields.get(index).ok_or_else(|| Error::Format(format!("field index {index} out of range")))?;
        let p = x.param();
        let mut info: DgioFieldInfo = std::mem::zeroed();
        copy_str(&mut info.short_name, p.short_name);
        copy_str(&mut info.name, p.name);
        copy_str(&mut info.units, p.units);
        copy_str(&mut info.level_type, x.level_name());
        info.level = x.product.level();
        info.ni = x.grid.ni;
        info.nj = x.grid.nj;
        info.n_points = x.grid.n_points;
        info.n_values = x.n_values;
        info.data_template = x.drt;
        info.bits_per_value = x.scale.nbits;
        info.has_bitmap = x.bitmap.is_some() as u8;
        info.discipline = x.discipline;
        info.category = x.product.category;
        info.number = x.product.number;
        info.level_type_code = x.product.level_type;
        info.forecast_time = x.product.forecast_time;
        info.ref_time = x.ref_time;
        info.centre = x.centre;
        info.supported = x.is_supported() as u8;
        info.lat_first = x.grid.la1 as f64 * 1e-6;
        info.lon_first = x.grid.lo1 as f64 * 1e-6;
        info.lat_last = x.grid.la2 as f64 * 1e-6;
        info.lon_last = x.grid.lo2 as f64 * 1e-6;
        info.di = x.grid.di as f64 * 1e-6;
        info.dj = x.grid.dj as f64 * 1e-6;
        info.message_offset = x.message.offset;
        info.message_length = x.message.len;
        *out = info;
        Ok(0)
    })
}

/// Opaque C handle: an Arc so DLPack exports can outlive `dgio_tensor_free`.
pub struct CTensor(Arc<Tensor>);

unsafe fn indices<'a>(ix: *const usize, n: usize) -> Option<&'a [usize]> {
    if ix.is_null() || n == 0 { None } else { Some(std::slice::from_raw_parts(ix, n)) }
}

/// # Safety
/// Pointers must be valid; `indices` may be NULL.
#[no_mangle]
pub unsafe extern "C" fn dgio_decode(f: *mut File, ix: *const usize, n: usize, backend: c_int, opts: *const DgioOptions) -> *mut CTensor {
    guard(std::ptr::null_mut(), || {
        let file = f.as_ref().ok_or_else(|| Error::Format("null file".into()))?;
        let o = opts.as_ref().map(|o| o.to_rust()).unwrap_or_default();
        let t = file.decode(indices(ix, n), backend_from(backend), &o)?;
        Ok(Box::into_raw(Box::new(CTensor(Arc::new(t)))))
    })
}

/// # Safety
/// Pointers must be valid; `dst` must hold `dst_len` floats.
#[no_mangle]
pub unsafe extern "C" fn dgio_decode_into(
    f: *mut File,
    ix: *const usize,
    n: usize,
    backend: c_int,
    opts: *const DgioOptions,
    dst: *mut f32,
    dst_len: usize,
) -> c_int {
    guard(-1, || {
        let file = f.as_ref().ok_or_else(|| Error::Format("null file".into()))?;
        let mut o = opts.as_ref().map(|o| o.to_rust()).unwrap_or_default();
        o.dtype = DType::F32;
        let t = file.decode(indices(ix, n), backend_from(backend), &o)?;
        let need = t.numel();
        if dst_len < need {
            return Err(Error::Format(format!("dst holds {dst_len} values, need {need}")));
        }
        match t.as_f32() {
            Some(s) => std::ptr::copy_nonoverlapping(s.as_ptr(), dst, need),
            None => {
                #[cfg(feature = "cuda")]
                if let Storage::Cuda(c) = &t.storage {
                    c.copy_to_host(std::slice::from_raw_parts_mut(dst, need))?;
                    return Ok(0);
                }
                return Err(Error::Unsupported("tensor not host-addressable".into()));
            }
        }
        Ok(0)
    })
}

/// # Safety
/// `t` from `dgio_decode` (or NULL).
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_free(t: *mut CTensor) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// # Safety
/// `t` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_data(t: *const CTensor) -> *mut c_void {
    t.as_ref().and_then(|t| t.0.host_ptr()).map(|p| p as *mut c_void).unwrap_or(std::ptr::null_mut())
}
/// # Safety
/// `t` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_device_handle(t: *const CTensor) -> *mut c_void {
    t.as_ref().map(|t| t.0.device_handle()).unwrap_or(std::ptr::null_mut())
}
/// # Safety
/// `t` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_device_type(t: *const CTensor) -> i32 {
    t.as_ref().map(|t| t.0.device_type()).unwrap_or(0)
}
/// # Safety
/// `t` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_dtype(t: *const CTensor) -> c_int {
    t.as_ref().map(|t| (t.0.dtype == DType::F16) as c_int).unwrap_or(0)
}
/// # Safety
/// `t` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_ndim(t: *const CTensor) -> i32 {
    t.as_ref().map(|t| t.0.shape.len() as i32).unwrap_or(0)
}
/// # Safety
/// `shape` must hold ndim entries.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_shape(t: *const CTensor, shape: *mut i64) -> c_int {
    match t.as_ref() {
        Some(t) => {
            for (i, &d) in t.0.shape.iter().enumerate() {
                *shape.add(i) = d as i64;
            }
            0
        }
        None => -1,
    }
}
/// # Safety
/// `t` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_numel(t: *const CTensor) -> usize {
    t.as_ref().map(|t| t.0.numel()).unwrap_or(0)
}
/// # Safety
/// `t`, `out` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_timings(t: *const CTensor, out: *mut DgioTimings) -> c_int {
    match t.as_ref() {
        Some(t) => {
            let x = &t.0.timings;
            *out = DgioTimings { plan_s: x.plan_s, io_s: x.io_s, compute_s: x.compute_s, total_s: x.total_s, bytes_read: x.bytes_read, values: x.values };
            0
        }
        None => -1,
    }
}

// ---- DLPack -------------------------------------------------------------------------

#[repr(C)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}
#[repr(C)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}
#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

struct DlCtx {
    managed: DLManagedTensor,
    shape: Vec<i64>,
    _keep: Arc<Tensor>,
}

unsafe extern "C" fn dl_deleter(m: *mut DLManagedTensor) {
    if !m.is_null() {
        drop(Box::from_raw((*m).manager_ctx as *mut DlCtx));
    }
}

/// Build a DLManagedTensor for `t` on `device_type` (0 = native).
pub fn to_dlpack(t: &Arc<Tensor>, device_type: i32) -> Result<*mut DLManagedTensor> {
    let native = t.device_type();
    let dev = if device_type == 0 { native } else { device_type };
    let data: *mut c_void = match (native, dev) {
        (a, b) if a == b => t.device_handle(),
        (DL_METAL, DL_CPU) => t.host_ptr().unwrap() as *mut c_void, // unified memory view
        (a, b) => return Err(Error::Unsupported(format!("cannot export device {a} tensor as DLPack device {b}"))),
    };
    let shape: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
    let mut ctx = Box::new(DlCtx {
        managed: DLManagedTensor {
            dl_tensor: DLTensor {
                data,
                device: DLDevice { device_type: dev, device_id: 0 },
                ndim: shape.len() as i32,
                dtype: DLDataType { code: 2, bits: (t.dtype.size() * 8) as u8, lanes: 1 },
                shape: std::ptr::null_mut(),
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(dl_deleter),
        },
        shape,
        _keep: t.clone(),
    });
    ctx.managed.dl_tensor.shape = ctx.shape.as_mut_ptr();
    let raw = Box::into_raw(ctx);
    unsafe {
        (*raw).managed.manager_ctx = raw as *mut c_void;
        Ok(&mut (*raw).managed as *mut DLManagedTensor)
    }
}

/// # Safety
/// `t` valid.
#[no_mangle]
pub unsafe extern "C" fn dgio_tensor_to_dlpack(t: *const CTensor, device_type: i32) -> *mut DLManagedTensor {
    guard(std::ptr::null_mut(), || {
        let t = t.as_ref().ok_or_else(|| Error::Format("null tensor".into()))?;
        to_dlpack(&t.0, device_type)
    })
}

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
#[cfg(target_os = "macos")]
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
#[cfg(not(target_os = "macos"))]
const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

/// PyCapsule destructor: if the capsule was never consumed (still named "dltensor"), release
/// the DLManagedTensor. Consumers rename it to "used_dltensor" when they take ownership.
///
/// # Safety
/// Called by CPython with the GIL held.
#[no_mangle]
pub unsafe extern "C" fn dgio_dlpack_capsule_destructor(capsule: *mut c_void) {
    type IsValid = unsafe extern "C" fn(*mut c_void, *const c_char) -> c_int;
    type GetPtr = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
    let is_valid = dlsym(RTLD_DEFAULT, c"PyCapsule_IsValid".as_ptr());
    let get_ptr = dlsym(RTLD_DEFAULT, c"PyCapsule_GetPointer".as_ptr());
    if is_valid.is_null() || get_ptr.is_null() {
        return;
    }
    let is_valid: IsValid = std::mem::transmute(is_valid);
    let get_ptr: GetPtr = std::mem::transmute(get_ptr);
    let name = c"dltensor".as_ptr();
    if is_valid(capsule, name) != 0 {
        let m = get_ptr(capsule, name) as *mut DLManagedTensor;
        if !m.is_null() {
            if let Some(d) = (*m).deleter {
                d(m);
            }
        }
    }
}
