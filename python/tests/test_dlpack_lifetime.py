"""DLPack lifetime / ownership / synchronisation tests.

The Metal backend recycles output MTLBuffers through a pool once a tensor is released. These
tests make sure a buffer is never handed to a new decode while any DLPack consumer (PyTorch,
MLX, NumPy, a raw capsule) still references it, that exports do not leak, and that the
CPU/GPU synchronisation contract holds.
"""
import ctypes
import gc
import pathlib
import resource
import sys
import threading

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import directgeoio as dg  # noqa: E402

FIX = ROOT / "fixtures"
A = FIX / "ecmwf_ccsds_t_pl.grib2"   # (8, 721, 1440)
B = FIX / "gfs_complex_t_pl.grib2"   # same shape, different values -> same pooled buffer size

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")
mps = torch.backends.mps.is_available()


def fresh(path):
    return dg.open(path).decode()


def churn(n=5):
    """Hold n simultaneous decodes of another file: this drains the output-buffer pool (<= 3
    entries), so any buffer wrongly returned to the pool is certainly reused and overwritten."""
    gc.collect()
    held = [fresh(B) for _ in range(n)]
    for t in held:
        _ = np.from_dlpack(t.cpu())[0, 0, 0]
    del held
    gc.collect()


@pytest.mark.skipif(not mps, reason="MPS not available")
def test_torch_mps_keeps_buffer_alive_after_tensor_released():
    t = fresh(A)
    x = torch.from_dlpack(t)
    want = x.cpu().clone()
    handle = t.device_handle
    del t
    gc.collect()
    churn()
    assert torch.equal(x.cpu(), want), "recycled buffer was overwritten while PyTorch held it"
    assert x.untyped_storage().data_ptr() == handle


def test_mlx_keeps_buffer_alive_after_tensor_released():
    # note: mx.from_dlpack also keeps a reference to the producer object, so MLX is protected
    # even independently of the DLManagedTensor's own reference
    t = fresh(A)
    m = mx.from_dlpack(t)
    want = np.array(m)
    del t
    gc.collect()
    churn()
    assert np.array_equal(np.array(m + 0), want)


def test_numpy_view_keeps_buffer_alive():
    t = fresh(A)
    a = np.from_dlpack(t.cpu())
    want = a.copy()
    del t
    gc.collect()
    churn()
    assert np.array_equal(a, want)


def test_released_buffer_is_reused_only_after_last_consumer():
    t = fresh(A)
    a = np.from_dlpack(t.cpu())
    ptr = t.data_ptr
    del t
    gc.collect()
    t2 = fresh(A)
    assert t2.data_ptr != ptr, "buffer reused while a NumPy view still exists"
    del a, t2
    gc.collect()
    t3 = fresh(A)  # now the original buffer may legitimately come back from the pool
    assert t3.shape == (8, 721, 1440)


@pytest.mark.skipif(not mps, reason="MPS not available")
def test_many_consumers_any_release_order():
    t = fresh(B)
    want = np.from_dlpack(t.cpu()).copy()
    consumers = [torch.from_dlpack(t), mx.from_dlpack(t), np.from_dlpack(t.cpu()), torch.from_dlpack(t.cpu())]
    del t
    for k in [2, 0, 3]:  # release in a scrambled order, churning the pool each time
        consumers[k] = None
        gc.collect()
        churn(2)
    assert np.array_equal(np.array(consumers[1]), want)


def test_repeated_export_does_not_leak():
    t = fresh(A)
    def rss_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # macOS: bytes
    for _ in range(50):  # warm-up
        np.from_dlpack(t.cpu())
    before = rss_mb()
    for _ in range(5000):
        a = np.from_dlpack(t.cpu())
        cap = t.__dlpack__()  # unconsumed capsule -> destructor must release it
        del a, cap
    gc.collect()
    assert rss_mb() - before < 50, "DLPack exports leak memory"


def test_consumed_capsule_is_not_double_freed():
    t = fresh(A)
    cap = t.__dlpack__()
    x = torch.utils.dlpack.from_dlpack(cap)  # consumer renames the capsule to "used_dltensor"
    del cap  # destructor must now do nothing
    gc.collect()
    churn(2)
    assert x.shape == (8, 721, 1440)
    del x, t
    gc.collect()


def test_dlpack_kwargs_and_device_request():
    t = fresh(A)
    assert t.__dlpack_device__() == (8, 0)
    cap = t.__dlpack__(stream=None, max_version=(1, 0), dl_device=(1, 0))  # ask for the host view
    a = np.from_dlpack(t.cpu())
    assert a.shape == (8, 721, 1440)
    with pytest.raises(BufferError):
        t.__dlpack__(copy=True)
    del cap


@pytest.mark.skipif(not mps, reason="MPS not available")
def test_gpu_writes_visible_to_cpu_after_synchronize():
    """Contract: decode() returns after the GPU finished, so the host view is immediately valid;
    writes made by a consumer on the GPU become visible after that consumer synchronises."""
    t = fresh(A)
    a = np.from_dlpack(t.cpu())
    x = torch.from_dlpack(t)
    before = a.copy()
    x.mul_(2.0)
    torch.mps.synchronize()
    assert np.array_equal(a, before * 2), "MPS writes not visible through the unified-memory view"


def test_concurrent_decodes_from_threads():
    want_a = np.from_dlpack(fresh(A).cpu()).copy()
    want_b = np.from_dlpack(fresh(B).cpu()).copy()
    errors = []

    def work(i):
        try:
            for _ in range(5):
                path, want = (A, want_a) if i % 2 else (B, want_b)
                a = np.from_dlpack(fresh(path).cpu())
                if not np.array_equal(a, want, equal_nan=True):
                    errors.append(f"thread {i}: wrong values")
        except Exception as e:  # noqa: BLE001
            errors.append(f"thread {i}: {e}")

    th = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [x.start() for x in th]
    [x.join() for x in th]
    assert not errors, errors


def test_raw_c_dlpack_deleter_keeps_tensor_alive():
    """C ABI: a DLManagedTensor stays valid after dgio_tensor_free until its deleter runs."""
    lib = dg._lib
    f = lib.dgio_open(str(A).encode())
    o = dg._Options()
    lib.dgio_default_options(ctypes.byref(o))
    h = lib.dgio_decode(f, None, 0, 0, ctypes.byref(o))
    p = lib.dgio_tensor_to_dlpack(h, 1)  # kDLCPU view

    class DLTensor(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("dt", ctypes.c_int32), ("did", ctypes.c_int32),
                    ("ndim", ctypes.c_int32), ("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                    ("lanes", ctypes.c_uint16), ("shape", ctypes.POINTER(ctypes.c_int64)),
                    ("strides", ctypes.c_void_p), ("off", ctypes.c_uint64)]

    class Managed(ctypes.Structure):
        _fields_ = [("t", DLTensor), ("ctx", ctypes.c_void_p), ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p))]

    m = Managed.from_address(p)
    n = m.t.shape[0] * m.t.shape[1] * m.t.shape[2]
    first = np.ctypeslib.as_array(ctypes.cast(m.t.data, ctypes.POINTER(ctypes.c_float)), (n,)).copy()
    lib.dgio_tensor_free(h)
    lib.dgio_close(f)
    churn()
    again = np.ctypeslib.as_array(ctypes.cast(m.t.data, ctypes.POINTER(ctypes.c_float)), (n,))
    assert np.array_equal(first, again)
    m.deleter(p)
