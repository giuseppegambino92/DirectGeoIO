// Section 6 bitmap: per-word popcount feeding the Blelloch scan that gives every grid point
// its rank among present points (bitmap expansion is then a pure gather, see unpack_simple).
#include "common.h"

kernel void bitmap_popcount(device const uint* raw [[buffer(0)]],
                            constant DgField* fields [[buffer(1)]],
                            device uint* rank [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    constant DgField& f = fields[gid.y];
    if (f.bitmap_off == DG_NONE) return;
    uint nw = (f.n_points + 31) / 32;
    uint x = gid.x;
    if (x >= nw) return;
    uint w = read32_at(raw, f.bitmap_off, ulong(x) * 32);
    uint tail = f.n_points & 31;
    if (x == nw - 1 && tail) w &= ~0u << (32 - tail);
    rank[f.rank_off + x] = popcount(w);
}
