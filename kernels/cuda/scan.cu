// Work-efficient Blelloch scan (uint32, wrapping) — port of kernels/metal/scan.metal.
#include "common.cuh"

extern "C" __global__ void scan_tiles(uint* data, uint* sums, DgParams P) {
    __shared__ uint s[DG_SCAN_THREADS];
    uint tid = threadIdx.x, tg = blockIdx.x;
    uint base = tg * DG_SCAN_TILE + tid * 4;
    uint v0 = base + 0 < P.n ? data[base + 0] : 0;
    uint v1 = base + 1 < P.n ? data[base + 1] : 0;
    uint v2 = base + 2 < P.n ? data[base + 2] : 0;
    uint v3 = base + 3 < P.n ? data[base + 3] : 0;
    s[tid] = v0 + v1 + v2 + v3;
    for (uint d = 1; d < DG_SCAN_THREADS; d <<= 1) {
        __syncthreads();
        uint i = (tid + 1) * 2 * d - 1;
        if (i < DG_SCAN_THREADS) s[i] += s[i - d];
    }
    __syncthreads();
    if (tid == 0) { sums[tg] = s[DG_SCAN_THREADS - 1]; s[DG_SCAN_THREADS - 1] = 0; }
    for (uint d = DG_SCAN_THREADS >> 1; d > 0; d >>= 1) {
        __syncthreads();
        uint i = (tid + 1) * 2 * d - 1;
        if (i < DG_SCAN_THREADS) { uint t = s[i - d]; s[i - d] = s[i]; s[i] += t; }
    }
    __syncthreads();
    uint e = s[tid], o0, o1, o2, o3;
    if (P.inclusive) { o0 = e + v0; o1 = o0 + v1; o2 = o1 + v2; o3 = o2 + v3; }
    else { o0 = e; o1 = e + v0; o2 = o1 + v1; o3 = o2 + v2; }
    if (base + 0 < P.n) data[base + 0] = o0;
    if (base + 1 < P.n) data[base + 1] = o1;
    if (base + 2 < P.n) data[base + 2] = o2;
    if (base + 3 < P.n) data[base + 3] = o3;
}

extern "C" __global__ void scan_add(uint* data, const uint* sums, DgParams P) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < P.n) data[gid] += sums[gid / DG_SCAN_TILE];
}
