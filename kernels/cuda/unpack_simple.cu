// Fused bitmap gather + 1..32-bit unpack / code fetch + dequantisation + store.
#include "common.cuh"
#include <cuda_fp16.h>

extern "C" __global__ void unpack_dequant_expand(const uint* raw, const DgField* fields, const uint* rank,
                                                 const uint* codes, DgStatus* status, void* out, DgParams P,
                                                 const uint* mvalid) {
    const DgField& f = fields[blockIdx.y];
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= f.n_points) return;
    uint o = f.out_off + j;
    uint idx = j;
    if (f.bitmap_off != DG_NONE) {
        uint wi = j >> 5;
        uint b = j & 31;
        uint w = read32_at(raw, f.bitmap_off, (ulong)wi * 32);
        uint present = (w >> (31 - b)) & 1u;
        idx = rank[f.rank_off + wi] - rank[f.rank_off] + (b ? __popc(w >> (32 - b)) : 0u); // exclusive rank
        if (j + 1 == f.n_points && idx + present != f.n_values) // bitmap must mark exactly n_values points
            status[f.field_index].error = DG_ERR_BITMAP;
        if (!present) {
            if (P.out_half) ((__half*)out)[o] = __float2half_rn(P.missing);
            else ((float*)out)[o] = P.missing;
            return;
        }
    }
    if (idx >= f.n_values) { // bitmap marks more points present than Section 5 declares
        status[f.field_index].error = DG_ERR_BITMAP;
        if (P.out_half) ((__half*)out)[o] = __float2half_rn(P.missing);
            else ((float*)out)[o] = P.missing;
        return;
    }
    int hi; uint lo;
    if (f.kind == DG_KIND_SIMPLE) {
        uint x = f.nbits ? (read32_at(raw, f.data_off, (ulong)idx * f.nbits) >> (32 - f.nbits)) : 0u;
        hi = (int)(x >> 16); lo = x & 0xFFFFu;
    } else if (f.kind == DG_KIND_CCSDS) {
        uint x = codes[f.codes_off + idx];
        hi = (int)(x >> 16); lo = x & 0xFFFFu;
    } else {
        uint r = idx;
        if (f.missing_mgmt) {
            uint base = f.mm_off ? mvalid[f.mm_off - 1] : 0u;
            uint inc = mvalid[f.mm_off + idx] - base;
            uint prev = idx ? mvalid[f.mm_off + idx - 1] - base : 0u;
            if (inc == prev) {
                if (P.out_half) ((__half*)out)[o] = __float2half_rn(P.missing);
                else ((float*)out)[o] = P.missing;
                return;
            }
            r = prev;
        }
        uint v = codes[f.codes_off + r]; // spatial differencing already resolved (sd_apply)
        // complex-packed integers are offsets from the field minimum: unsigned, modulo 2^32
        hi = (int)(v >> 16); lo = v & 0xFFFFu;
    }
    float y = dequant(hi, lo, f, P.exact != 0);
    if (P.out_half) ((__half*)out)[o] = __float2half_rn(y);
    else ((float*)out)[o] = y;
}
