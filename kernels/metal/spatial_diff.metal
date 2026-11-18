// Spatial-differencing reconstruction (GRIB2 Template 5.3, order 1 or 2) as a fused, segmented
// tile scan. With t = differencing terms (t0 = ival1, t1 = ival2 - 2*ival1, ti = Xi + minsd):
//     order 1:  v = scan(t)            order 2:  v = scan(scan(t))
// Each 1024-value tile is summarised by A1 = sum t_k and A2 = sum (L-k) t_k (= sum of the first
// scan's partials), carries (C1, C2) are propagated per field, and every tile is finished with
// two Blelloch scans in threadgroup memory:  v_i = C2 + (i+1)*C1 + S2_i   (order 2).
// Traffic: one read for the reduce, one read + one write for the apply.
#include "common.h"
#define SD_TILE 1024u
#define SD_T 256u

kernel void sd_tile_reduce(constant DgField* fields [[buffer(0)]],
                           device const uint4* codes4 [[buffer(1)]],
                           device uint2* agg [[buffer(2)]],
                           uint2 tgp [[threadgroup_position_in_grid]],
                           uint2 tid2 [[thread_position_in_threadgroup]]) {
    constant DgField& f = fields[tgp.y];
    if (f.kind != DG_KIND_COMPLEX || f.sd_order == 0) return;
    uint t0 = tgp.x * SD_TILE;
    if (t0 >= f.n_values) return;
    uint L = min(SD_TILE, f.n_values - t0);
    uint tid = tid2.x;
    uint k = tid * 4;
    uint4 v = codes4[(f.codes_off + t0) / 4 + tid];
    uint a1 = 0, a2 = 0;
    for (uint j = 0; j < 4; j++) {
        if (k + j < L) { a1 += v[j]; a2 += (L - (k + j)) * v[j]; }
    }
    threadgroup uint s1[SD_T], s2[SD_T];
    s1[tid] = a1; s2[tid] = a2;
    for (uint d = SD_T / 2; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < d) { s1[tid] += s1[tid + d]; s2[tid] += s2[tid + d]; }
    }
    if (tid == 0) agg[f.tile_off + tgp.x] = uint2(s1[0], s2[0]);
}

// one thread per field: exclusive carry propagation across its tiles (in place)
kernel void sd_carry(constant DgField* fields [[buffer(0)]],
                     device uint2* agg [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    constant DgField& f = fields[gid];
    if (f.kind != DG_KIND_COMPLEX || f.sd_order == 0) return;
    uint nt = (f.n_values + SD_TILE - 1) / SD_TILE;
    uint c1 = 0, c2 = 0;
    for (uint t = 0; t < nt; t++) {
        uint2 a = agg[f.tile_off + t];
        uint L = min(SD_TILE, f.n_values - t * SD_TILE);
        agg[f.tile_off + t] = uint2(c1, c2);
        c2 += L * c1 + a.y;
        c1 += a.x;
    }
}

// in-place exclusive Blelloch scan of SD_T elements in threadgroup memory; returns total
inline uint tg_exclusive_scan(threadgroup uint* s, uint tid) {
    for (uint d = 1; d < SD_T; d <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint i = (tid + 1) * 2 * d - 1;
        if (i < SD_T) s[i] += s[i - d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint total = s[SD_T - 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) s[SD_T - 1] = 0;
    for (uint d = SD_T >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint i = (tid + 1) * 2 * d - 1;
        if (i < SD_T) { uint t = s[i - d]; s[i - d] = s[i]; s[i] += t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void sd_apply(constant DgField* fields [[buffer(0)]],
                     device uint4* codes4 [[buffer(1)]],
                     device const uint2* agg [[buffer(2)]],
                     uint2 tgp [[threadgroup_position_in_grid]],
                     uint2 tid2 [[thread_position_in_threadgroup]]) {
    constant DgField& f = fields[tgp.y];
    if (f.kind != DG_KIND_COMPLEX || f.sd_order == 0) return;
    uint t0 = tgp.x * SD_TILE;
    if (t0 >= f.n_values) return;
    uint tid = tid2.x;
    uint idx4 = (f.codes_off + t0) / 4 + tid;
    uint4 v = codes4[idx4];
    threadgroup uint s[SD_T];
    // first scan (inclusive) of t
    uint4 p1;
    p1.x = v.x; p1.y = p1.x + v.y; p1.z = p1.y + v.z; p1.w = p1.z + v.w;
    s[tid] = p1.w;
    tg_exclusive_scan(s, tid);
    uint e1 = s[tid];
    p1 += uint4(e1);
    uint2 c = agg[f.tile_off + tgp.x];
    uint4 out;
    if (f.sd_order == 1) {
        out = p1 + uint4(c.x);
    } else {
        // second scan (inclusive) of the local first-scan partials
        uint4 p2;
        p2.x = p1.x; p2.y = p2.x + p1.y; p2.z = p2.y + p1.z; p2.w = p2.z + p1.w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s[tid] = p2.w;
        tg_exclusive_scan(s, tid);
        p2 += uint4(s[tid]);
        uint i0 = tid * 4 + 1; // (i+1) for the first of this thread's 4 elements
        out = uint4(c.y) + uint4(i0, i0 + 1, i0 + 2, i0 + 3) * c.x + p2;
    }
    codes4[idx4] = out;
}
