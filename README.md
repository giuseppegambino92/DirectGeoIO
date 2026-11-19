# DirectGeoIO

[![CI](https://github.com/giuseppegambino92/DirectGeoIO/actions/workflows/ci.yml/badge.svg)](https://github.com/giuseppegambino92/DirectGeoIO/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**GPU GRIB2 decoding straight from storage into GPU-resident tensors — Apple Silicon (Metal +
`MTLIOCommandQueue`) and NVIDIA (CUDA + GPUDirect Storage `cuFile`) — bit-exact with ECMWF ecCodes,
exported zero-copy to PyTorch, MLX, NumPy and C/C++/Fortran.**

```
                 ┌──────────────── CPU: metadata only (≈200 B / message) ───────────────┐
 GRIB2 file ───► │ Section 0/1/3/4/5 scan + Section 6 indicator → decode plan (DgField[]) │
 on NVMe         └───────────────────────────────┬──────────────────────────────────────┘
     │                                           │ byte ranges of Section 6 bitmaps + Section 7 payloads
     │  MTLIOCommandQueue (Apple)  /  cuFileRead GPUDirect Storage (NVIDIA, pinned-staging fallback)
     ▼                                           ▼
 ┌──────────────────────────────── GPU buffer: raw payload bytes ─────────────────────────────────┐
 │ 5.0 simple   │ fused 1–32-bit unpack (branch-free 64-bit window) ───────────────────────┐      │
 │ 5.42 CCSDS   │ CDS boundary walk → per-CDS Rice/SE/zero-block decode → per-RSI inverse  │      │
 │              │ predictor                                                                ├─► WMO dequant      │
 │ 5.2/5.3      │ group table → Blelloch scan (bit + sample offsets) → tile-parallel unpack │   Y=(R+X·2^E)·10^-D │
 │  complex     │ → missing-value compaction (scan) → fused segmented 2nd-order scan       │   float-float exact │
 │ Section 6    │ popcount → Blelloch scan → rank gather (NaN / missing_value)              ┘   → f32 / f16       │
 └────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
            DLPack (kDLMetal = id<MTLBuffer>, kDLCPU = unified-memory view, kDLCUDA) — zero copies
                 torch.from_dlpack · mx.from_dlpack · np.from_dlpack · C ABI (WRF, Fortran)
```

## Results

Complete operational files, end-to-end from disk to a contiguous float32 tensor. **Median of 5–10
runs** (p10–p90 within a few percent on the idle Linux machine). ecCodes MT = best of 16/32 threads.
Full tables: [`rtx5090.md`](bench/results/rtx5090.md), [`rtx5090-cold.md`](bench/results/rtx5090-cold.md),
[`m3max.md`](bench/results/m3max.md).

**NVIDIA RTX 5090 + Ryzen 9 9950X3D** (Ubuntu 25.04, CUDA 12.9, ecCodes 2.50.0)

| file | cache | ecCodes 1T | ecCodes MT | **DirectGeoIO CUDA** | vs 1T | vs MT |
|---|---|---:|---:|---:|---:|---:|
| ECMWF IFS 0.25° 6 h step — 191 fields, all 5.42 CCSDS, 198 M values | warm | 837 ms | 150 ms | **40.5 ms** | 21× | 3.7× |
| | cold¹ | 856 ms | 181 ms | **87.6 ms** | 9.8× | 2.1× |
| NOAA GFS 0.25° f006 — 743 fields, 5.3 (+5.0), 771 M values | warm | 3365 ms | 822 ms | **60.6 ms** | 56× | 13.6× |
| | cold¹ | 3505 ms | 900 ms | **242 ms** | 14.5× | 3.7× |

**Apple M3 Max** (macOS, ecCodes 2.50.1; best of N while unrelated jobs kept the machine at load ≈11; warm cache)

| file | ecCodes 1T | ecCodes MT | **DirectGeoIO Metal** | vs 1T | vs MT |
|---|---:|---:|---:|---:|---:|
| ECMWF IFS 0.25° 6 h step | 1286 ms | 219 ms | **48.9 ms** | 26× | 4.5× |
| NOAA GFS 0.25° f006 | 4152 ms | 895 ms | **147 ms** | 28× | 6.1× |

¹ every run starts after evicting the file from the page cache (`posix_fadvise(DONTNEED)`). Cold, GFS is
bound by DirectGeoIO's single-stream pinned staging (532 MB in 154 ms ≈ 3.5 GB/s, well below the NVMe's
capability) — parallel reads or GPUDirect Storage are the next step. The CUDA tensor ends in device
memory; ecCodes' output would still need a host→device copy (not counted). That machine has no
`nvidia_fs`, so bytes go through pinned staging rather than GPUDirect Storage.

**Correctness.** Every value of both complete files — 969,716,160 grid points, including 25.7 M
bitmap- or missing-value-masked points — is **bit-identical to `float32(ecCodes float64)`** on the
Metal backend (`make verify-full`) and identical to the ecCodes-verified CPU reference on the CUDA
backend (1,040,316,480 values incl. the test corpus, both CCSDS boundary-walk modes). See [Verification and robustness](#verification-and-robustness) for the full test matrix.

## What is (and is not) implemented

| | status |
|---|---|
| GRIB2 Sections 0–8, multi-field messages, Section 6 indicator 0 / 254 / 255 | ✅ |
| Template 5.0 simple packing, 0–32 bits | ✅ CPU + Metal + CUDA |
| Template 5.42 CCSDS 121.0-B-3 / libaec (preprocessing, SE, zero-block + ROS, uncompressed, `AEC_PAD_RSI`, restricted IDs) | ✅ unsigned samples |
| Templates 5.2 / 5.3 complex packing, spatial differencing order 1/2, missing-value management 0/1/2 | ✅ |
| Templates 5.40 (JPEG 2000), 5.41 (PNG), 5.50/5.51 (spectral) | ❌ listed per field (`supported = false`), decoding them returns `Unsupported` |
| float32 output (bit-exact) / fast single-FMA mode (≤1 ulp) / float16 output | ✅ |
| C ABI `include/directgeoio.h`, DLPack, Python package | ✅ |
| macOS `MTLIOCommandQueue` loading | ✅ measured |
| CUDA kernels + pinned-staging I/O | ✅ run and verified on RTX 5090 (CUDA 12.9, sm_120); compiles with CUDA 13.0 for sm_80/90/120 |
| `cuFile` GPUDirect Storage path | ⚠️ compiled, used only when `nvidia_fs` is loaded — **not exercised with real GDS** (see below) |
| NetCDF4/HDF5 | ❌ not started |

### Honest notes on the data path

* **Apple Silicon has one memory.** `MTLIOCommandQueue` loads file ranges into an `MTLBuffer` without a
  user-space staging copy, but there is no discrete VRAM to "bypass" into, and whether Apple's
  implementation uses the unified buffer cache is not documented. The benchmark numbers above are
  **warm page cache**; cold-NVMe numbers (`sudo purge` between runs) are not yet published.
* **CCSDS has one inherently sequential step.** GRIB2 stores no index of CCSDS block offsets and RSIs are
  not byte-aligned, so the start of block *k* depends on every earlier block. DirectGeoIO walks block
  headers only (popcount-skipping the unary codes, ~25 ns per block, no samples produced), then decodes
  every block, every RSI and every value in parallel on the GPU. On Apple Silicon this walk runs by
  default on CPU threads reading the DMA'd buffer *in place* (unified memory, no copy) — it is ~60×
  faster per block than a single GPU thread. A GPU-only walk is available (`aec_scan="gpu"`) and is used
  automatically on the CUDA GPUDirect path, where payload bytes never reach host memory. On the RTX 5090
  the CPU walk also won (8 CCSDS fields: 2.4 vs 28 ms; 191 fields: 41 vs 92 ms).
* **Blelloch scans do the structural work:** bitmap ranks, complex-packing group bit/sample offsets,
  missing-value compaction, and a fused segmented scan for second-order spatial differencing
  (`v = scan(scan(t))` with a per-tile affine carry).
* **Bit-exactness** is achieved by evaluating `(X·2^E + R)·10^-D` in float-float arithmetic
  (TwoSum/FMA, fast-math disabled) with `10^-D` computed exactly as ecCodes' `codes_power` does.

## Verification and robustness

| layer | what is checked | where |
|---|---|---|
| differential vs ecCodes | 99 messages: NOAA GFS, ECMWF IFS, **DWD ICON** (incl. the icosahedral grid, Template 3.101) as published, plus ecCodes re-encodings covering simple packing at 0–32 bits, D ≠ 0, constant fields, CCSDS block sizes 8/16/32/64, RSI 1–4096, no preprocessing, `AEC_PAD_RSI`, 32-bit samples, complex 5.2 and 5.3 (orders 1 and 2), missing-value management, bitmaps for every template | `crates/directgeoio-core/tests/ecc_reference.rs` |
| complete operational files | 969,716,160 values bit-identical to ecCodes (Metal); CUDA identical to the CPU reference on 1,046,906,482 values (corpus + full files, both CCSDS walk modes) | `tools/verify_full.py`, `crates/directgeoio-cuda/tests` |
| unsupported templates | e.g. CMC GDPS (Template 5.40 JPEG 2000): the field is listed with `supported = false`, decoding it returns `Unsupported`, other fields of the file stay usable | `ecc_reference.rs` |
| malformed input (CPU) | deterministic mutation fuzzing (truncation, bit flips, extreme section lengths, corrupt Section 5 parameters, false point counts, splices, random payload bytes): **0 panics, 0 hangs in 2,000,000 inputs**; 200,000 run in CI. A libFuzzer target is in `fuzz/` (nightly) | `tests/fuzz_mutations.rs` |
| malformed input (GPU) | differential fuzzing CPU ↔ GPU: both must accept/reject the same inputs and agree bit-for-bit when they accept — **0 disagreements, 0 hangs in 200,000 inputs on Metal and on CUDA (RTX 5090)**. Kernels bound every read to its section, CCSDS walks stop at the end of the data, and inconsistent bitmaps / group tables raise errors | `directgeoio-metal/tests/gpu_fuzz.rs` (CUDA: `directgeoio-cuda/tests/gpu_fuzz.rs`) |
| exact arithmetic | Metal emulates IEEE binary64 when the float-float fast path cannot prove the float32 rounding (near ties, subnormals, overflow): emulated add/mul/conversions match native f64 on 4,194,304 edge-heavy cases. CUDA uses native binary64 | `directgeoio-metal/tests/softf64.rs` |
| DLPack ownership | pooled output buffers are never reused while PyTorch, MLX, NumPy or a raw `DLManagedTensor` still references them (verified by draining the pool, and by injecting an ownership bug which the tests catch); repeated export does not leak; consumed capsules are not double-freed; GPU writes are visible to the host view after the consumer synchronises; concurrent decodes from 8 threads | `python/tests/test_dlpack_lifetime.py` |

Benchmarks report the median and p10–p90 of N runs (`tools/bench.py`, `dgio bench`). `--cold` evicts each
file from the page cache before every run with `posix_fadvise(DONTNEED)` (Linux, no root needed). On macOS
no reliable unprivileged eviction was found (`msync(MS_INVALIDATE)` and `F_NOCACHE` copies stay cached), so
Apple cold-cache numbers need `sudo purge`. Energy is not reported: CPU counters (RAPL on Linux,
`powermetrics` on macOS) require root, and GPU-only energy would not be comparable with CPU baselines.

## Quick start (macOS, Apple Silicon)

```bash
brew install eccodes rust uv          # ecCodes only needed for ground truth + baselines
make venv build                       # Python 3.12 venv (eccodes, numpy, torch, mlx, pytest)
make fixtures                         # real GFS + ECMWF Open Data messages + ecCodes ground truth
make test                             # Rust + Metal + Python + C ABI
make fixtures-full verify-full bench  # complete operational files, bit-exact check, benchmarks
```

```python
import sys; sys.path.insert(0, "python")
import directgeoio as dg, torch, mlx.core as mx, numpy as np

f = dg.open("fixtures/raw/bench/ecmwf_ifs_0p25_20261117_00z_6h.grib2")
print(f.fields[0])                          # short_name, level, template, bits, grid, ...
t = f.decode(f.select(short_name="t", level_type="isobaricInPa"))
x = torch.from_dlpack(t)                    # MPS tensor wrapping the same MTLBuffer (zero-copy)
m = mx.from_dlpack(t)                       # MLX array, same buffer
a = np.from_dlpack(t.cpu())                 # unified-memory host view, same bytes
t16 = f.decode(dtype="float16", missing_value=-999.0)
```

```c
#include <directgeoio.h>
dgio_file* f = dgio_open("gfs.t00z.pgrb2.0p25.f006");
dgio_options o; dgio_default_options(&o);
dgio_tensor* t = dgio_decode(f, NULL, 0, DGIO_BACKEND_AUTO, &o);
const float* v = dgio_tensor_data(t);            /* or dgio_decode_into(..., dst, n) from Fortran */
DLManagedTensor* dl = dgio_tensor_to_dlpack(t, kDLMetal);
```

CLI: `target/release/dgio info FILE` · `target/release/dgio bench FILE --backend metal|cpu|cpu-mt`
(`DGIO_PROFILE=1` prints per-kernel GPU times).

## Linux / NVIDIA (RTX 5090 and others)

```bash
export CUDA_PATH=/usr/local/cuda            # CUDA >= 12.8 for sm_120 (RTX 5090)
cargo build --release -p directgeoio --features cuda
DGIO_CUDA_ARCHS="120" cargo build ...       # restrict target architectures
DGIO_DISABLE_GDS=1 ...                       # force the pinned-host staging path
make cuda-check                              # compile check inside nvidia/cuda:13.0.2-devel (no GPU needed)
```

`libcufile` is `dlopen`ed at runtime and used only when the `nvidia_fs` kernel module is loaded
(override: `DGIO_FORCE_CUFILE=1`). Otherwise reads use double-buffered `pread` + `cudaMemcpyAsync`
through pinned memory — on the RTX 5090 test machine this beat libcufile's POSIX compat mode
(26 vs 42 ms for 532 MB). Validation on a CUDA machine:

```bash
DGIO_EXTRA_FILES=$(ls -d $PWD/fixtures/raw/bench/*.grib2 | tr '\n' ':') \
  cargo test --release -p directgeoio-cuda -- --test-threads=1 --nocapture
cargo build --release -p dgio-cli --features dgio-cli/cuda && target/release/dgio bench FILE --backend cuda
```

### GPUDirect Storage status (RTX 5090 test machine)

What was checked on Ubuntu 25.04, kernel 6.14, NVIDIA open driver 580, CUDA 12.9 / libcufile 1.14:

* kernel P2PDMA is built in (`CONFIG_PCI_P2PDMA=y`), the GPU and the data NVMe share one AMD root
  complex, and libcufile reports `kernel Support: 1, pci_p2pdma_supported_cuda: 1`;
* enabling cuFile's P2PDMA mode (`"use_pci_p2pdma": true` via `CUFILE_ENV_PATH_JSON`, no root needed)
  fails at `checkIfAllGPUsSupportP2PDMA(): 0` — the driver refuses a P2P device address for the
  GeForce RTX 5090 (`CUDA_ERROR_NOT_SUPPORTED`, 801);
* the classic `nvidia-fs` route (`nvidia-fs-dkms`, needs root) was not attempted. It relies on the same
  GPU-side P2P mapping, so it may hit the same restriction on GeForce boards.

Data-centre and RTX professional GPUs are the boards NVIDIA documents for GDS. On this machine the
pinned-staging path delivers 532 MB from the page cache in 26 ms, so GDS could save at most that much
per file here.

## Layout

```
crates/directgeoio-core    GRIB2 scanner, WMO tables, bit-exact CPU decoders, GPU decode planner
crates/directgeoio-metal   MTLIOCommandQueue + Metal compute backend (ObjC shim, build.rs → .metallib)
crates/directgeoio-cuda    cuFile/GDS + CUDA backend (build.rs → nvcc)
crates/directgeoio         public Rust API, C ABI, DLPack, backend selection  → libdirectgeoio
crates/dgio-cli            dgio info / bench
kernels/dgio_shared.h      device ABI shared by MSL, CUDA and Rust
kernels/metal/*.metal      unpack_simple, decode_ccsds_aec, decode_complex, spatial_diff, bitmap_expand, scan
kernels/cuda/*.cu          1:1 CUDA ports + host.cu runtime
include/directgeoio.h      C header        python/directgeoio   ctypes + DLPack package
tools/                     fixture fetch/build, full-file verification, benchmarks, CUDA container check
bench/eccodes_bench.c      ecCodes C baseline (single / multi-thread)
```

## Data

Test data is downloaded, not redistributed: NOAA GFS (public domain, NOAA Open Data Dissemination on
AWS) and ECMWF Open Data (© ECMWF, CC BY 4.0). See `NOTICE`.

## License

Apache-2.0. See `LICENSE`.
