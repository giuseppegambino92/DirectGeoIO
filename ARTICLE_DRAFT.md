# Bypassing the CPU to Decode Weather Data: NVMe-to-GPU GRIB2 Decompression on Apple Silicon and NVIDIA GPUs

*Draft — measured on an NVIDIA RTX 5090 (Ryzen 9 9950X3D host) and an Apple M3 Max.*

## The problem: GPUs wait for a single CPU thread

Numerical weather prediction and its machine-learning successors (GraphCast-, Pangu- and AIFS-style
models) run on GPUs, and so increasingly do data assimilation, post-processing and verification. Their
input, though, still arrives as WMO GRIB2, and GRIB2 is still decoded the way it was in 2005: one CPU
thread inside ECMWF's ecCodes, one message at a time, into float64, followed by a conversion, a copy to
pinned memory and a PCIe transfer.

How slow is that? A single 6-hour step of the ECMWF IFS 0.25° open-data forecast is a 100 MB file with
191 fields. ecCodes takes **1.29 s** to turn it into float32 on one core of an M3 Max, and 219 ms with
ten threads. The corresponding NOAA GFS file (532 MB, 743 fields, 771 M grid points) takes **4.15 s**
single-threaded and 895 ms on all fourteen cores. A model that consumes dozens of steps per sample
spends its time waiting.

## The idea

Treat a GRIB2 file the way a GPU texture pipeline treats compressed textures: read only the metadata
on the CPU, move the compressed bytes straight from storage into GPU memory, and decompress there.

* On **Apple Silicon**, `MTLIOCommandQueue` (Metal fast resource loading) fills an `MTLBuffer` directly
  from a file handle — no `read()` into a user buffer and no `memcpy`.
* On **NVIDIA**, GPUDirect Storage (`cuFileRead`) DMAs from NVMe into device memory without a bounce
  through host RAM, with a pinned-staging fallback for filesystems without nvidia-fs.

The CPU reads about 200 bytes per message — Sections 0, 1, 3, 4, 5 and the Section 6 indicator — and
builds a plan: one 160-byte descriptor per field. Everything else happens in compute kernels.

## Three compression schemes, three kinds of parallelism

Real operational data forced the design. ECMWF's open data is **100% Template 5.42 (CCSDS/AEC)**. NOAA's
GFS is almost entirely **Template 5.3 (complex packing with second-order spatial differencing)** — not
simple packing, as is often assumed — and 21 of its fields use missing-value management because their
height-above-sea levels lie underground. Template 5.0 simple packing is what most other producers emit.

**Simple packing** is embarrassingly parallel: value *i* lives at bit `i·nbits`. One thread per grid
point loads a 64-bit big-endian window and shifts — identical instructions for every width from 1 to
32 bits, so there is no SIMD divergence. It runs at memory speed.

**Complex packing** splits the field into groups (≈30,000 per GFS field), each with its own reference,
bit width and length, stored as three bit-packed tables before the data. Where a group's values start
is a *prefix sum* of `width × length` over preceding groups, so a Blelloch scan gives every group its
bit offset and first sample. A tiny scatter kernel then records which group covers the start of each
1024-value tile, and a tile-parallel kernel unpacks every value with coalesced stores. The second-order
spatial differencing (`v_i = 2v_{i-1} − v_{i-2} + y_i`) is two nested prefix sums, which I fuse into a
single segmented tile scan. Each tile is summarised by `A1 = Σt_k` and `A2 = Σ(L−k)·t_k` (the sum of the
first scan's partials, in closed form), an affine carry `(C1, C2) → (C1 + A1, C2 + L·C1 + A2)` is
propagated per field, and each tile finishes with two scans in threadgroup memory. Missing-value
management becomes one more scan: values are flagged, an inclusive scan over the flags ranks the
present ones, and differencing runs over the compacted sequence.

**CCSDS 121.0-B-3** (the libaec algorithm behind "SZIP") is the hard case. The stream is a sequence of
32-sample blocks, each Rice-coded with an adaptively chosen option (fundamental sequence, split-k,
second extension, zero-block runs). Every 128 blocks a Reference Sample Interval restarts the predictor.
GRIB2 stores **no index of block or RSI offsets**, and RSIs are not byte-aligned, so where block *k*
starts depends on the lengths of all blocks before it. No prefix sum can remove that dependency: a
block's length isn't known until its header and unary codes have been parsed.

DirectGeoIO therefore splits CCSDS decoding into three passes:

1. **Boundary walk** — parse only block IDs and *count* unary codes, using 57-bit popcount windows and
   a 6-step binary select instead of decoding samples: ~25 ns per block on a CPU core.
2. **Block decode** — one GPU thread per coded data set, ~32,000 per field, fully parallel.
3. **Inverse preprocessing** — one GPU thread per RSI. The unit-delay predictor's inverse mapping is
   nonlinear (it depends on the distance of the previous sample to the range ends), so it is a short
   sequential recurrence over 4,096 samples, with 254 independent RSIs per field.

Where should pass 1 run? I first wrote it as a GPU kernel, one thread per field. It was correct and
slow: 2.4 µs per block, about 90× slower than a CPU core, because a dependent chain of loads is the
worst case for a throughput processor. Register-streaming the bit buffer brought that to ~1.6 µs. On
Apple Silicon the right answer is to let the CPU walk the block headers **in place in the very buffer
the DMA filled**: unified memory makes it free to read. On the NVIDIA GPUDirect path, where bytes never
reach host memory, the GPU walk runs, amortised across hundreds of fields in flight.

## Bit-exact, not "close enough"

Meteorologists compare decoders by `diff`, so DirectGeoIO reproduces ecCodes' float64 arithmetic in
float32 on the GPU. `Y = (X·2^E + R)·10^-D` is evaluated as a float-float expression (TwoSum plus FMA
error terms, fast-math off), with `10^-D` computed by repeated division exactly as ecCodes'
`codes_power` does, split into a hi/lo pair. The result is identical to `float32(ecCodes float64)` for
**all 969,716,160 values** of the two complete operational files, and for a corpus of 68 messages
spanning 1- to 32-bit simple packing, 3- to 24-bit CCSDS, bitmaps and missing-value management. A naive
single-FMA evaluation differs in 7% of values (by one ulp); the exact path costs nothing measurable,
because the kernels are memory-bound.

## Zero-copy all the way to the model

The output tensor lives in a shared-storage `MTLBuffer` (or CUDA device memory) and is exported through
DLPack. `torch.from_dlpack(t)` yields an MPS tensor that wraps the *same* `MTLBuffer`. `mx.from_dlpack(t)`
gives an MLX array over the same memory. `np.from_dlpack(t.cpu())` is a host view of the same bytes,
which is legal only because Apple GPUs share memory. All three are verified by pointer identity in the
test-suite. Fortran and C codes such as WRF use the plain C ABI.

## Results (warm page cache)

| file | machine | ecCodes 1T | ecCodes MT | DirectGeoIO | vs 1T / MT |
|---|---|---:|---:|---:|---:|
| ECMWF IFS 6 h step, 191 CCSDS fields, 198 M values | RTX 5090 | 831 ms | 149 ms | **40.0 ms** | 21× / 3.7× |
| NOAA GFS f006, 743 complex-packed fields, 771 M values | RTX 5090 | 3372 ms | 807 ms | **59.0 ms** | 57× / 13.7× |
| ECMWF IFS 6 h step | M3 Max | 1286 ms | 219 ms | **48.9 ms** | 26× / 4.5× |
| NOAA GFS f006 | M3 Max | 4152 ms | 895 ms | **147 ms** | 28× / 6.1× |

On the RTX 5090, GFS decoding itself takes 30 ms — 26 billion values per second; half of the 59 ms is
moving 532 MB from the page cache into VRAM. That test machine has no `nvidia_fs`, so bytes went through
pinned staging rather than GPUDirect Storage. libcufile's compat mode, which falls back to POSIX
silently, was measurably slower (42 vs 26 ms), so DirectGeoIO only takes the cuFile path when real GDS
is available. The CUDA backend ran correctly the first time it touched NVIDIA hardware: 1.04 billion
values, identical to the CPU reference that is itself bit-exact with ecCodes. The M3 Max numbers were
taken while unrelated jobs loaded the machine.

## What's next

* Measure real GPUDirect Storage (nvidia-fs loaded) against pinned staging.
* Cold-cache NVMe measurements, and overlapping the CCSDS boundary walk of batch *n+1* with GPU decode
  of batch *n*.
* JPEG 2000 (Template 5.40) and NetCDF4/HDF5 chunked reading.

*Code: Apache-2.0. Data: NOAA GFS (public domain), ECMWF Open Data (CC BY 4.0).*
