// Section 6 bitmap popcount (feeds the Blelloch rank scan) — port of bitmap_expand.metal.
#include "common.cuh"

extern "C" __global__ void bitmap_popcount(const uint* raw, const DgField* fields, uint* rank) {
    const DgField& f = fields[blockIdx.y];
    if (f.bitmap_off == DG_NONE) return;
    uint nw = (f.n_points + 31) / 32;
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nw) return;
    uint w = read32_at(raw, f.bitmap_off, (ulong)x * 32);
    uint tail = f.n_points & 31;
    if (x == nw - 1 && tail) w &= ~0u << (32 - tail);
    rank[f.rank_off + x] = __popc(w);
}
