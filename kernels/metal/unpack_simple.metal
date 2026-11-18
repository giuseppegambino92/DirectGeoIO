// Final fused pass: bitmap gather + arbitrary-width (1..32 bit) unpack for Template 5.0 or
// code fetch for 5.42/5.3 + WMO dequantisation + float32/float16 store. One thread per grid
// point; the 64-bit window read is branch-free for every bit width (no SIMD divergence).
#include "common.h"

kernel void unpack_dequant_expand(device const uint* raw [[buffer(0)]],
                                  constant DgField* fields [[buffer(1)]],
                                  device const uint* rank [[buffer(2)]],
                                  device const uint* codes [[buffer(3)]],
                                  device const DgStatus* status [[buffer(4)]],
                                  device void* out [[buffer(5)]],
                                  constant DgParams& P [[buffer(6)]],
                                  device const uint* mvalid [[buffer(7)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    constant DgField& f = fields[gid.y];
    uint j = gid.x;
    if (j >= f.n_points) return;
    uint o = f.out_off + j;
    uint idx = j;
    if (f.bitmap_off != DG_NONE) {
        uint wi = j >> 5;
        uint b = j & 31;
        uint w = read32_at(raw, f.bitmap_off, ulong(wi) * 32);
        if (((w >> (31 - b)) & 1u) == 0) {
            if (P.out_half) ((device half*)out)[o] = half(P.missing);
            else ((device float*)out)[o] = P.missing;
            return;
        }
        idx = rank[f.rank_off + wi] - rank[f.rank_off] + (b ? popcount(w >> (32 - b)) : 0u);
    }
    int hi; uint lo;
    if (f.kind == DG_KIND_SIMPLE) {
        uint x = 0;
        if (f.nbits) {
            x = read32_at(raw, f.data_off, ulong(idx) * f.nbits) >> (32 - f.nbits);
        }
        hi = int(x >> 16); lo = x & 0xFFFFu;
    } else if (f.kind == DG_KIND_CCSDS) {
        uint x = codes[f.codes_off + idx];
        hi = int(x >> 16); lo = x & 0xFFFFu;
    } else {
        uint r = idx;
        if (f.missing_mgmt) {
            uint base = f.mm_off ? mvalid[f.mm_off - 1] : 0u;
            uint inc = mvalid[f.mm_off + idx] - base;
            uint prev = idx ? mvalid[f.mm_off + idx - 1] - base : 0u;
            if (inc == prev) {
                if (P.out_half) ((device half*)out)[o] = half(P.missing);
                else ((device float*)out)[o] = P.missing;
                return;
            }
            r = prev;
        }
        uint v = codes[f.codes_off + r]; // spatial differencing already resolved (sd_apply)
        int x = int(v);
        hi = x >> 16; lo = uint(x) & 0xFFFFu;
    }
    float y = dequant(hi, lo, f, P.exact != 0);
    if (P.out_half) ((device half*)out)[o] = half(y);
    else ((device float*)out)[o] = y;
}
