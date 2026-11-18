// Fused segmented spatial-differencing reconstruction — port of spatial_diff.metal.
#include "common.cuh"
#define SD_TILE 1024u
#define SD_T 256u

extern "C" __global__ void sd_tile_reduce(const DgField* fields, const uint4* codes4, uint2* agg) {
    const DgField& f = fields[blockIdx.y];
    if (f.kind != DG_KIND_COMPLEX || f.sd_order == 0) return;
    uint t0 = blockIdx.x * SD_TILE;
    if (t0 >= f.n_values) return;
    uint L = min(SD_TILE, f.n_values - t0);
    uint tid = threadIdx.x, k = tid * 4;
    uint4 v = codes4[(f.codes_off + t0) / 4 + tid];
    uint vv[4] = {v.x, v.y, v.z, v.w};
    uint a1 = 0, a2 = 0;
    for (uint j = 0; j < 4; j++)
        if (k + j < L) { a1 += vv[j]; a2 += (L - (k + j)) * vv[j]; }
    __shared__ uint s1[SD_T], s2[SD_T];
    s1[tid] = a1; s2[tid] = a2;
    for (uint d = SD_T / 2; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) { s1[tid] += s1[tid + d]; s2[tid] += s2[tid + d]; }
    }
    if (tid == 0) agg[f.tile_off + blockIdx.x] = make_uint2(s1[0], s2[0]);
}

extern "C" __global__ void sd_carry(const DgField* fields, uint2* agg, uint nf) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nf) return;
    const DgField& f = fields[gid];
    if (f.kind != DG_KIND_COMPLEX || f.sd_order == 0) return;
    uint nt = (f.n_values + SD_TILE - 1) / SD_TILE;
    uint c1 = 0, c2 = 0;
    for (uint t = 0; t < nt; t++) {
        uint2 a = agg[f.tile_off + t];
        uint L = min(SD_TILE, f.n_values - t * SD_TILE);
        agg[f.tile_off + t] = make_uint2(c1, c2);
        c2 += L * c1 + a.y;
        c1 += a.x;
    }
}

__device__ __forceinline__ void tg_exclusive_scan(uint* s, uint tid) {
    for (uint d = 1; d < SD_T; d <<= 1) {
        __syncthreads();
        uint i = (tid + 1) * 2 * d - 1;
        if (i < SD_T) s[i] += s[i - d];
    }
    __syncthreads();
    if (tid == 0) s[SD_T - 1] = 0;
    for (uint d = SD_T >> 1; d > 0; d >>= 1) {
        __syncthreads();
        uint i = (tid + 1) * 2 * d - 1;
        if (i < SD_T) { uint t = s[i - d]; s[i - d] = s[i]; s[i] += t; }
    }
    __syncthreads();
}

extern "C" __global__ void sd_apply(const DgField* fields, uint4* codes4, const uint2* agg) {
    const DgField& f = fields[blockIdx.y];
    if (f.kind != DG_KIND_COMPLEX || f.sd_order == 0) return;
    uint t0 = blockIdx.x * SD_TILE;
    if (t0 >= f.n_values) return;
    uint tid = threadIdx.x;
    uint idx4 = (f.codes_off + t0) / 4 + tid;
    uint4 v = codes4[idx4];
    __shared__ uint s[SD_T];
    uint p1x = v.x, p1y = p1x + v.y, p1z = p1y + v.z, p1w = p1z + v.w;
    s[tid] = p1w;
    tg_exclusive_scan(s, tid);
    uint e1 = s[tid];
    p1x += e1; p1y += e1; p1z += e1; p1w += e1;
    uint2 c = agg[f.tile_off + blockIdx.x];
    uint4 out;
    if (f.sd_order == 1) {
        out = make_uint4(p1x + c.x, p1y + c.x, p1z + c.x, p1w + c.x);
    } else {
        uint p2x = p1x, p2y = p2x + p1y, p2z = p2y + p1z, p2w = p2z + p1w;
        __syncthreads();
        s[tid] = p2w;
        tg_exclusive_scan(s, tid);
        uint e2 = s[tid];
        uint i0 = tid * 4 + 1;
        out = make_uint4(c.y + i0 * c.x + p2x + e2, c.y + (i0 + 1) * c.x + p2y + e2,
                         c.y + (i0 + 2) * c.x + p2z + e2, c.y + (i0 + 3) * c.x + p2w + e2);
    }
    codes4[idx4] = out;
}
