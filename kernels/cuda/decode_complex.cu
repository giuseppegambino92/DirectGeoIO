// GRIB2 Template 5.2 / 5.3 complex packing + spatial differencing — port of decode_complex.metal.
#include "common.cuh"

extern "C" __global__ void cplx_groups(const uint* raw, const DgField* fields, uint* gref, uint* gwidth, uint* gscan, DgParams P) {
    const DgField& f = fields[blockIdx.y];
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (f.kind != DG_KIND_COMPLEX || g >= f.ngroups) return;
    BitR r = bitr(raw, f.data_off, f.data_len, (ulong)f.goff_refs * 8 + (ulong)g * f.nbits);
    uint ref = r.read(f.nbits);
    r = bitr(raw, f.data_off, f.data_len, (ulong)f.goff_widths * 8 + (ulong)g * f.bits_gw);
    uint w = r.read(f.bits_gw) + f.ref_gw;
    uint len;
    if (g + 1 == f.ngroups) len = f.last_gl;
    else {
        r = bitr(raw, f.data_off, f.data_len, (ulong)f.goff_lens * 8 + (ulong)g * f.bits_gl);
        len = f.ref_gl + r.read(f.bits_gl) * f.len_inc;
    }
    uint a = f.aux_off + g;
    gref[a] = ref;
    gwidth[a] = w;
    gscan[a] = w * len;
    gscan[P.n + a] = len;
}

__device__ __forceinline__ int sm_bytes(const unsigned char* p, uint n) {
    uint v = 0;
    for (uint i = 0; i < n; i++) v = (v << 8) | p[i];
    uint sign = 1u << (8 * n - 1);
    return (v & sign) ? -(int)(v & ~sign) : (int)v;
}

// Tile-parallel unpack (see decode_complex.metal).
#define DG_CPLX_TILE 1024u

extern "C" __global__ void cplx_tile_index(const DgField* fields, const uint* gscan, uint* tiles, DgParams P) {
    const DgField& f = fields[blockIdx.y];
    uint g = blockIdx.x * blockDim.x + threadIdx.x;
    if (f.kind != DG_KIND_COMPLEX || g >= f.ngroups) return;
    const uint* lscan = gscan + P.n + f.aux_off;
    uint s = lscan[g] - lscan[0];
    uint e = g + 1 < f.ngroups ? lscan[g + 1] - lscan[0] : f.n_values;
    for (uint t = (s + DG_CPLX_TILE - 1) / DG_CPLX_TILE; t * DG_CPLX_TILE < e; t++) tiles[f.tile_off + t] = g;
}
extern "C" __global__ void cplx_unpack(const uint* raw, const DgField* fields, const uint* gref, const uint* gwidth,
                                       const uint* gscan, uint* codes, DgStatus* status, DgParams P, uint* xraw,
                                       uint* mvalid, const uint* tiles) {
    const DgField& f = fields[blockIdx.y];
    if (f.kind != DG_KIND_COMPLEX) return;
    uint s0 = blockIdx.x * DG_CPLX_TILE;
    if (s0 >= f.n_values) return;
    uint s1 = min(s0 + DG_CPLX_TILE, f.n_values);
    uint tid = threadIdx.x, ntg = blockDim.x;
    __shared__ uint gstart[DG_CPLX_TILE], gbit[DG_CPLX_TILE], gw[DG_CPLX_TILE], gr[DG_CPLX_TILE];
    const uint* lscan = gscan + P.n + f.aux_off;
    const uint* bscan = gscan + f.aux_off;
    uint lbase = lscan[0], bbase = bscan[0];
    uint ng_all = f.ngroups;
    if (tid == 0 && s1 == f.n_values && lscan[ng_all - 1] - lbase + f.last_gl != f.n_values)
        status[f.field_index].error = DG_ERR_GROUPS;
    uint g0 = tiles[f.tile_off + blockIdx.x];
    uint g1 = s1 == f.n_values ? ng_all - 1 : tiles[f.tile_off + blockIdx.x + 1];
    uint ng = min(g1 - g0 + 1, DG_CPLX_TILE);
    for (uint k = tid; k < ng; k += ntg) {
        uint g = g0 + k;
        gstart[k] = lscan[g] - lbase;
        gbit[k] = bscan[g] - bbase;
        gw[k] = gwidth[f.aux_off + g];
        gr[k] = gref[f.aux_off + g];
    }
    __syncthreads();
    int ival1 = 0, ival2 = 0, minsd = 0;
    if (f.sd_order > 0) {
        const unsigned char* p = (const unsigned char*)raw + f.data_off;
        uint nb = f.sd_octets;
        ival1 = sm_bytes(p, nb);
        if (f.sd_order == 2) ival2 = sm_bytes(p + nb, nb);
        minsd = sm_bytes(p + nb * f.sd_order, nb);
    }
    ulong all1n = (1ull << f.nbits) - 1;
    uint* o = codes + f.codes_off;
    for (uint i = s0 + tid; i < s1; i += ntg) {
        uint lo = 0, hi = ng - 1;
        while (lo < hi) { uint mid = (lo + hi + 1) >> 1; if (gstart[mid] <= i) lo = mid; else hi = mid - 1; }
        uint w = gw[lo], ref = gr[lo];
        uint v = w ? (read32_at(raw, f.data_off, (ulong)f.goff_vals * 8 + gbit[lo] + (ulong)(i - gstart[lo]) * w) >> (32 - w)) : 0u;
        if (f.missing_mgmt) {
            ulong all1w = (1ull << w) - 1;
            bool miss = w > 0 ? (v == all1w || (f.missing_mgmt == 2 && v == all1w - 1))
                              : (ref == all1n || (f.missing_mgmt == 2 && ref == all1n - 1));
            xraw[f.mm_off + i] = ref + v;
            mvalid[f.mm_off + i] = miss ? 0u : 1u;
            continue;
        }
        int x = (int)(ref + v);
        int t;
        if (f.sd_order == 0) t = x;
        else if (i == 0) t = ival1;
        else if (i == 1 && f.sd_order == 2) t = ival2 - 2 * ival1;
        else t = x + minsd;
        o[i] = (uint)t;
    }
}


extern "C" __global__ void cplx_compact(const uint* raw, const DgField* fields, const uint* xraw, const uint* mvalid,
                                        uint* codes) {
    const DgField& f = fields[blockIdx.y];
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (f.kind != DG_KIND_COMPLEX || f.missing_mgmt == 0 || i >= f.n_values) return;
    uint base = f.mm_off ? mvalid[f.mm_off - 1] : 0u;
    uint inc = mvalid[f.mm_off + i] - base;
    uint prev = i ? mvalid[f.mm_off + i - 1] - base : 0u;
    if (inc == prev) return;
    uint r = prev;
    int x = (int)xraw[f.mm_off + i];
    int t = x;
    if (f.sd_order > 0) {
        const unsigned char* p = (const unsigned char*)raw + f.data_off;
        uint nb = f.sd_octets;
        int ival1 = sm_bytes(p, nb);
        int ival2 = f.sd_order == 2 ? sm_bytes(p + nb, nb) : 0;
        int minsd = sm_bytes(p + nb * f.sd_order, nb);
        if (r == 0) t = ival1;
        else if (r == 1 && f.sd_order == 2) t = ival2 - 2 * ival1;
        else t = x + minsd;
    }
    codes[f.codes_off + r] = (uint)t;
}
