"""DirectGeoIO — zero-copy NVMe-to-GPU GRIB2 decoding with DLPack export.

    import directgeoio as dg, torch, mlx.core as mx, numpy as np
    f = dg.open("ecmwf_0p25.grib2")
    t = f.decode()                     # Metal (Apple) / CUDA (NVIDIA) / CPU
    x = torch.from_dlpack(t)           # MPS tensor, same MTLBuffer — no copy
    m = mx.from_dlpack(t)              # MLX array
    a = np.from_dlpack(t.cpu())        # unified-memory host view — no copy

The Python layer is a thin ctypes binding over the C ABI in include/directgeoio.h, so the same
libdirectgeoio serves C, C++, Fortran and Python.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = ["open", "File", "FieldInfo", "Tensor", "device_name", "version", "DirectGeoIOError"]

kDLCPU, kDLCUDA, kDLMetal = 1, 2, 8
_DEVICE_NAMES = {kDLCPU: "cpu", kDLCUDA: "cuda", kDLMetal: "metal"}
_BACKENDS = {"auto": 0, "cpu": 1, "cpu-mt": 2, "metal": 3, "cuda": 4}


class DirectGeoIOError(RuntimeError):
    pass


def _find_lib() -> str:
    names = ["libdirectgeoio.dylib", "libdirectgeoio.so"]
    cands = []
    if os.environ.get("DGIO_LIB"):
        cands.append(pathlib.Path(os.environ["DGIO_LIB"]))
    here = pathlib.Path(__file__).resolve().parent
    for n in names:
        cands.append(here / n)
        cands.append(here.parent.parent / "target" / "release" / n)
    for c in cands:
        if c.is_file():
            return str(c)
    found = ctypes.util.find_library("directgeoio")
    if found:
        return found
    raise DirectGeoIOError("libdirectgeoio not found; build with `cargo build --release` or set DGIO_LIB")


_lib = ctypes.CDLL(_find_lib())


class _Options(ctypes.Structure):
    _fields_ = [("dtype", ctypes.c_int), ("exact", ctypes.c_int32), ("missing_value", ctypes.c_float),
                ("io", ctypes.c_int), ("aec_scan", ctypes.c_int)]


class _FieldInfo(ctypes.Structure):
    _fields_ = [("short_name", ctypes.c_char * 16), ("name", ctypes.c_char * 64), ("units", ctypes.c_char * 24),
                ("level_type", ctypes.c_char * 24), ("level", ctypes.c_double), ("ni", ctypes.c_uint32),
                ("nj", ctypes.c_uint32), ("n_points", ctypes.c_uint32), ("n_values", ctypes.c_uint32),
                ("data_template", ctypes.c_uint16), ("bits_per_value", ctypes.c_uint8), ("has_bitmap", ctypes.c_uint8),
                ("discipline", ctypes.c_uint8), ("category", ctypes.c_uint8), ("number", ctypes.c_uint8),
                ("level_type_code", ctypes.c_uint8), ("forecast_time", ctypes.c_uint32),
                ("ref_time", ctypes.c_uint16 * 6), ("centre", ctypes.c_uint16), ("supported", ctypes.c_uint8), ("pad", ctypes.c_uint8),
                ("lat_first", ctypes.c_double), ("lon_first", ctypes.c_double), ("lat_last", ctypes.c_double),
                ("lon_last", ctypes.c_double), ("di", ctypes.c_double), ("dj", ctypes.c_double),
                ("message_offset", ctypes.c_uint64), ("message_length", ctypes.c_uint64)]


class _Timings(ctypes.Structure):
    _fields_ = [("plan_s", ctypes.c_double), ("io_s", ctypes.c_double), ("compute_s", ctypes.c_double),
                ("total_s", ctypes.c_double), ("bytes_read", ctypes.c_uint64), ("values", ctypes.c_uint64)]


_vp, _sz = ctypes.c_void_p, ctypes.c_size_t
for name, res, args in [
    ("dgio_version", ctypes.c_char_p, []),
    ("dgio_last_error", ctypes.c_char_p, []),
    ("dgio_backend_device_name", ctypes.c_char_p, [ctypes.c_int]),
    ("dgio_default_options", None, [ctypes.POINTER(_Options)]),
    ("dgio_open", _vp, [ctypes.c_char_p]),
    ("dgio_close", None, [_vp]),
    ("dgio_num_fields", _sz, [_vp]),
    ("dgio_get_field_info", ctypes.c_int, [_vp, _sz, ctypes.POINTER(_FieldInfo)]),
    ("dgio_decode", _vp, [_vp, ctypes.POINTER(_sz), _sz, ctypes.c_int, ctypes.POINTER(_Options)]),
    ("dgio_tensor_free", None, [_vp]),
    ("dgio_tensor_data", _vp, [_vp]),
    ("dgio_tensor_device_handle", _vp, [_vp]),
    ("dgio_tensor_device_type", ctypes.c_int32, [_vp]),
    ("dgio_tensor_dtype", ctypes.c_int, [_vp]),
    ("dgio_tensor_ndim", ctypes.c_int32, [_vp]),
    ("dgio_tensor_shape", ctypes.c_int, [_vp, ctypes.POINTER(ctypes.c_int64)]),
    ("dgio_tensor_numel", _sz, [_vp]),
    ("dgio_tensor_timings", ctypes.c_int, [_vp, ctypes.POINTER(_Timings)]),
    ("dgio_tensor_to_dlpack", _vp, [_vp, ctypes.c_int32]),
]:
    fn = getattr(_lib, name)
    fn.restype, fn.argtypes = res, args

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_CAPSULE_NAME = ctypes.create_string_buffer(b"dltensor")  # must outlive every capsule
_CAPSULE_DTOR = ctypes.cast(_lib.dgio_dlpack_capsule_destructor, ctypes.c_void_p).value


def _err() -> DirectGeoIOError:
    return DirectGeoIOError((_lib.dgio_last_error() or b"").decode())


def version() -> str:
    return _lib.dgio_version().decode()


def device_name(backend: str = "auto") -> str:
    return _lib.dgio_backend_device_name(_BACKENDS[backend]).decode()


@dataclass(frozen=True)
class FieldInfo:
    index: int
    short_name: str
    name: str
    units: str
    level_type: str
    level: float
    ni: int
    nj: int
    n_points: int
    n_values: int
    data_template: int
    bits_per_value: int
    has_bitmap: bool
    supported: bool
    discipline: int
    category: int
    number: int
    forecast_time: int
    ref_time: tuple
    lat_first: float
    lon_first: float
    lat_last: float
    lon_last: float
    di: float
    dj: float


class Tensor:
    """Decoded fields resident on the backend device. Implements the DLPack protocol."""

    def __init__(self, handle: int, export_device: Optional[int] = None):
        self._h = handle
        self._export = export_device or _lib.dgio_tensor_device_type(handle)
        nd = _lib.dgio_tensor_ndim(handle)
        shp = (ctypes.c_int64 * nd)()
        _lib.dgio_tensor_shape(handle, shp)
        self.shape = tuple(shp)
        self.dtype = "float16" if _lib.dgio_tensor_dtype(handle) == 1 else "float32"
        t = _Timings()
        _lib.dgio_tensor_timings(handle, ctypes.byref(t))
        self.timings = {k: getattr(t, k) for k, _ in _Timings._fields_}

    # a view shares the C handle; only the owner frees it
    _owner = True

    def __del__(self):
        h = getattr(self, "_h", None)
        if h and self._owner:
            _lib.dgio_tensor_free(h)
            self._h = None

    @property
    def device(self) -> str:
        return _DEVICE_NAMES.get(self._export, str(self._export))

    @property
    def native_device(self) -> str:
        return _DEVICE_NAMES.get(_lib.dgio_tensor_device_type(self._h), "?")

    @property
    def data_ptr(self) -> int:
        """CPU-addressable pointer (unified memory on Apple Silicon); 0 for discrete GPU memory."""
        return _lib.dgio_tensor_data(self._h) or 0

    @property
    def device_handle(self) -> int:
        """id<MTLBuffer> (Metal), CUdeviceptr (CUDA) or host pointer (CPU)."""
        return _lib.dgio_tensor_device_handle(self._h) or 0

    def cpu(self) -> "Tensor":
        """Zero-copy host view (kDLCPU) of a Metal tensor — valid because Apple GPUs share memory."""
        if self._export == kDLCPU:
            return self
        if _lib.dgio_tensor_device_type(self._h) == kDLCUDA:
            raise DirectGeoIOError("CUDA tensors live in device memory; use torch/cupy .from_dlpack(t)")
        v = Tensor.__new__(Tensor)
        v.__dict__.update(self.__dict__)
        v._export, v._owner, v._parent = kDLCPU, False, self
        return v

    def numpy(self):
        import numpy as np
        return np.from_dlpack(self.cpu())

    # ---- DLPack protocol ----------------------------------------------------------------
    def __dlpack_device__(self):
        return (self._export, 0)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        if copy:
            raise BufferError("DirectGeoIO exports zero-copy views only")
        dev = self._export
        if dl_device is not None and tuple(dl_device)[0] != dev:
            dev = tuple(dl_device)[0]  # e.g. Metal tensor requested as kDLCPU (unified memory)
        p = _lib.dgio_tensor_to_dlpack(self._h, dev)
        if not p:
            raise BufferError(_err())
        return _PyCapsule_New(p, ctypes.cast(_CAPSULE_NAME, ctypes.c_char_p), _CAPSULE_DTOR)

    def __repr__(self):
        return f"directgeoio.Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"


class File:
    """A GRIB2 file whose metadata (not payload) has been scanned."""

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._h = _lib.dgio_open(self.path.encode())
        if not self._h:
            raise _err()
        self.fields = [self._info(i) for i in range(_lib.dgio_num_fields(self._h))]

    def _info(self, i: int) -> FieldInfo:
        x = _FieldInfo()
        if _lib.dgio_get_field_info(self._h, i, ctypes.byref(x)) != 0:
            raise _err()
        s = lambda b: b.decode(errors="replace")
        return FieldInfo(i, s(x.short_name), s(x.name), s(x.units), s(x.level_type), x.level, x.ni, x.nj,
                         x.n_points, x.n_values, x.data_template, x.bits_per_value, bool(x.has_bitmap), bool(x.supported),
                         x.discipline, x.category, x.number, x.forecast_time, tuple(x.ref_time), x.lat_first,
                         x.lon_first, x.lat_last, x.lon_last, x.di, x.dj)

    def __len__(self):
        return len(self.fields)

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.dgio_close(self._h)
            self._h = None

    def select(self, **kw) -> list[int]:
        """Indices of fields whose attributes match, e.g. select(short_name="t", level_type="isobaricInPa")."""
        return [f.index for f in self.fields if all(getattr(f, k) == v for k, v in kw.items())]

    def decode(self, indices: Optional[Sequence[int]] = None, *, backend: str = "auto", dtype: str = "float32",
               exact: bool = True, missing_value: float = math.nan, io: str = "auto", aec_scan: str = "auto") -> Tensor:
        o = _Options()
        _lib.dgio_default_options(ctypes.byref(o))
        o.dtype = {"float32": 0, "float16": 1}[dtype]
        o.exact = int(exact)
        o.missing_value = missing_value
        o.io = {"auto": 0, "span": 1, "ranges": 2}[io]
        o.aec_scan = {"auto": 0, "gpu": 1, "cpu": 2}[aec_scan]
        if indices is None:
            arr, n = None, 0
        else:
            n = len(indices)
            arr = (_sz * n)(*indices)
        h = _lib.dgio_decode(self._h, arr, n, _BACKENDS[backend], ctypes.byref(o))
        if not h:
            raise _err()
        return Tensor(h)

    def to_tensor(self, short_names: Sequence[str], level_type: str, levels: Sequence[float], **kw) -> Tensor:
        """Decode a (variable, level, lat, lon) block in one batch; returned tensor has shape
        (len(short_names) * len(levels), nj, ni) — reshape to 4D in the consumer framework."""
        idx = []
        for v in short_names:
            for lv in levels:
                m = [f.index for f in self.fields if f.short_name == v and f.level_type == level_type and f.level == lv]
                if not m:
                    raise KeyError(f"{v}@{level_type}={lv} not in {self.path}")
                idx.append(m[0])
        return self.decode(idx, **kw)

    def __repr__(self):
        return f"directgeoio.File({self.path!r}, {len(self.fields)} fields)"


def open(path) -> File:  # noqa: A001
    return File(path)
