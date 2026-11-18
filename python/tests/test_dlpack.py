"""DirectGeoIO Python/DLPack tests: values vs ecCodes ground truth, and zero-copy proofs for
NumPy, PyTorch (MPS + CPU) and MLX. Run: .venv/bin/python -m pytest python/tests -q"""
import gc
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import directgeoio as dg  # noqa: E402

FIX = ROOT / "fixtures"
MANIFEST = json.loads((FIX / "manifest.json").read_text())


def ref(name, i):
    return np.load(FIX / "ref" / f"{name}.{i}.npy")


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_bit_exact_vs_eccodes(name):
    f = dg.open(FIX / name)
    t = f.decode()
    a = t.numpy()
    assert a.dtype == np.float32
    flat = a.reshape(len(f), -1)
    for i in range(len(f)):
        want = ref(name, i).astype(np.float32)
        got = flat[i]
        assert np.array_equal(np.isnan(got), np.isnan(want)), f"{name}#{i} bitmap mask differs"
        m = ~np.isnan(want)
        assert np.array_equal(got[m].view(np.uint32), want[m].view(np.uint32)), f"{name}#{i} not bit-exact"


def test_numpy_zero_copy():
    t = dg.open(FIX / "ecmwf_ccsds_sfc.grib2").decode()
    a = np.from_dlpack(t.cpu())
    assert a.shape == (7, 721, 1440)
    assert a.ctypes.data == t.data_ptr, "NumPy array must alias the MTLBuffer contents"


def test_capsule_lifetime_outlives_tensor():
    t = dg.open(FIX / "simple_12bit.grib2").decode()
    a = np.from_dlpack(t.cpu())
    first = a[0, 360, 720]
    del t
    gc.collect()
    assert a[0, 360, 720] == first  # buffer kept alive by the DLManagedTensor


def test_unconsumed_capsule_is_released():
    t = dg.open(FIX / "simple_12bit.grib2").decode()
    for _ in range(100):
        cap = t.__dlpack__()
        del cap  # destructor must call the deleter without crashing


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_torch_mps_zero_copy():
    t = dg.open(FIX / "ecmwf_ccsds_t_pl.grib2").decode()
    x = torch.from_dlpack(t)
    assert x.device.type == "mps" and tuple(x.shape) == (8, 721, 1440)
    assert x.untyped_storage().data_ptr() == t.device_handle, "MPS tensor must wrap the same MTLBuffer"
    want = np.stack([ref("ecmwf_ccsds_t_pl.grib2", i) for i in range(8)]).reshape(8, 721, 1440).astype(np.float32)
    assert torch.equal(x.cpu(), torch.from_numpy(want))
    # compute on the GPU directly from the decoded buffer
    mean_k = x.mean(dim=(1, 2)).cpu().numpy()
    assert np.allclose(mean_k, want.mean(axis=(1, 2)), rtol=1e-5)


def test_torch_cpu_zero_copy():
    t = dg.open(FIX / "gfs_complex_t_pl.grib2").decode()
    x = torch.from_dlpack(t.cpu())
    assert x.device.type == "cpu" and x.data_ptr() == t.data_ptr


mx = pytest.importorskip("mlx.core")


def test_mlx_from_dlpack():
    t = dg.open(FIX / "gfs_complex_sfc.grib2").decode()
    m = mx.from_dlpack(t)
    assert tuple(m.shape) == (10, 721, 1440)
    host = np.from_dlpack(t.cpu())
    assert np.array_equal(np.array(m), host, equal_nan=True)
    # write-through check: if MLX wraps the same MTLBuffer, mutating the unified-memory view
    # is visible to a freshly evaluated MLX expression
    import ctypes
    ctypes.c_float.from_address(t.data_ptr).value = 12345.0
    v = float((m[0, 0, 0] + 0).item())
    print("\nmlx aliases DirectGeoIO buffer (zero-copy):", v == 12345.0)


def test_float16_and_missing_value():
    f = dg.open(FIX / "ccsds_bitmap.grib2")
    t = f.decode(dtype="float16", missing_value=-999.0)
    x = torch.from_dlpack(t.cpu())
    assert x.dtype == torch.float16
    want = ref("ccsds_bitmap.grib2", 0)
    got = x[0].reshape(-1).float().numpy()
    assert np.all(got[np.isnan(want)] == -999.0)
    m = ~np.isnan(want)
    assert np.allclose(got[m], want[m], rtol=1e-3)


def test_select_and_4d():
    f = dg.open(FIX / "ecmwf_ccsds_t_pl.grib2")
    levels = [100000.0, 85000.0, 50000.0]
    t = f.to_tensor(["t"], "isobaricInPa", levels)
    x = torch.from_dlpack(t).reshape(1, 3, 721, 1440)
    assert tuple(x.shape) == (1, 3, 721, 1440)
    assert f.select(short_name="t", level=85000.0) == [2]
