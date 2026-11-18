// Work-efficient Blelloch exclusive/inclusive prefix sum (uint32, wrapping), multi-level.
// Each threadgroup scans a 1024-element tile (256 threads x 4) in threadgroup memory and
// emits its tile total; the host recursively scans tile totals and runs scan_add.
#include "common.h"

kernel void scan_tiles(device uint* data [[buffer(0)]],
                       device uint* sums [[buffer(1)]],
                       constant DgParams& P [[buffer(2)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tg [[threadgroup_position_in_grid]]) {
    threadgroup uint s[DG_SCAN_THREADS];
    uint base = tg * DG_SCAN_TILE + tid * 4;
    uint v0 = base + 0 < P.n ? data[base + 0] : 0;
    uint v1 = base + 1 < P.n ? data[base + 1] : 0;
    uint v2 = base + 2 < P.n ? data[base + 2] : 0;
    uint v3 = base + 3 < P.n ? data[base + 3] : 0;
    s[tid] = v0 + v1 + v2 + v3;
    // up-sweep (reduce)
    for (uint d = 1; d < DG_SCAN_THREADS; d <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint i = (tid + 1) * 2 * d - 1;
        if (i < DG_SCAN_THREADS) s[i] += s[i - d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { sums[tg] = s[DG_SCAN_THREADS - 1]; s[DG_SCAN_THREADS - 1] = 0; }
    // down-sweep
    for (uint d = DG_SCAN_THREADS >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint i = (tid + 1) * 2 * d - 1;
        if (i < DG_SCAN_THREADS) { uint t = s[i - d]; s[i - d] = s[i]; s[i] += t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint e = s[tid];
    uint o0, o1, o2, o3;
    if (P.inclusive) { o0 = e + v0; o1 = o0 + v1; o2 = o1 + v2; o3 = o2 + v3; }
    else { o0 = e; o1 = e + v0; o2 = o1 + v1; o3 = o2 + v2; }
    if (base + 0 < P.n) data[base + 0] = o0;
    if (base + 1 < P.n) data[base + 1] = o1;
    if (base + 2 < P.n) data[base + 2] = o2;
    if (base + 3 < P.n) data[base + 3] = o3;
}

kernel void scan_add(device uint* data [[buffer(0)]],
                     device const uint* sums [[buffer(1)]],
                     constant DgParams& P [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < P.n) data[gid] += sums[gid / DG_SCAN_TILE];
}
