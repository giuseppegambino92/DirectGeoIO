// CCSDS 121.0-B-3 Adaptive Entropy Coding (GRIB2 Template 5.42 / libaec) on the GPU.
//   aec_scan   : boundary scan, one thread per field (CCSDS has no block index and RSIs are
//                not byte-aligned, so CDS k's offset depends on CDS 0..k-1). Skips FS codes
//                with 33-bit popcount windows; produces no samples.
//   aec_decode : one thread per Coded Data Set — Rice/FS split, second extension, zero-block
//                runs (incl. ROS), uncompressed — writing mapped residuals.
//   aec_post   : one thread per Reference Sample Interval — inverse unit-delay predictor and
//                inverse mapping (a nonlinear recurrence, sequential within the RSI).
#include "common.h"

kernel void aec_scan(device const uint* raw [[buffer(0)]],
                     constant DgField* fields [[buffer(1)]],
                     device uint2* cds [[buffer(2)]],
                     device DgStatus* status [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    constant DgField& f = fields[gid];
    if (f.kind != DG_KIND_CCSDS) return;
    uint bs = f.block_size, rsi = f.rsi, nb = f.nbits;
    uint nblocks = (f.n_values + bs - 1) / bs;
    uint max_id = (1u << f.id_len) - 1;
    bool pp = (f.aec_flags & DG_AEC_PP) != 0;
    bool pad = (f.aec_flags & DG_AEC_PAD_RSI) != 0;
    ulong end_bits = ulong(f.data_len) * 8;
    Stream r = stream(raw, f.data_off);
    device uint2* out = cds + f.aux_off;
    uint n = 0, b = 0;
    while (b < nblocks) {
        uint in_rsi = b % rsi;
        if (in_rsi == 0 && pad && b > 0) r.seek((r.pos() + 7) & ~7ul);
        ulong p = r.pos();
        if (p >= end_bits || n + 1 >= f.aux_cap) { status[f.field_index].error = DG_ERR_TRUNCATED; break; }
        out[n++] = uint2(uint(p), b);
        uint refs = (pp && in_rsi == 0) ? 1u : 0u;
        uint id = r.read(f.id_len);
        if (id == 0) {
            uint se = r.read(1);
            if (refs) r.skip(nb);
            if (se) { r.skip_fs(bs / 2); b += 1; }
            else {
                uint z = r.read_fs() + 1;
                if (z == 5) z = min(rsi - in_rsi, 64 - in_rsi % 64);
                else if (z > 5) z -= 1;
                b += z;
            }
        } else if (id == max_id) {
            r.skip(bs * nb); b += 1;
        } else {
            if (refs) r.skip(nb);
            r.skip_fs(bs - refs);
            r.skip((id - 1) * (bs - refs));
            b += 1;
        }
    }
    out[n++] = uint2(uint(r.pos()), nblocks);
    status[f.field_index].count = n;
}

inline uint se_beta(uint m) {
    uint b = 0;
    while ((b + 1) * (b + 2) / 2 <= m) b++;
    return b;
}

kernel void aec_decode(device const uint* raw [[buffer(0)]],
                       constant DgField* fields [[buffer(1)]],
                       device const uint2* cds [[buffer(2)]],
                       device const DgStatus* status [[buffer(3)]],
                       device uint* codes [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    constant DgField& f = fields[gid.y];
    if (f.kind != DG_KIND_CCSDS) return;
    uint i = gid.x;
    if (i + 1 >= status[f.field_index].count) return;
    uint2 c = cds[f.aux_off + i];
    uint next_block = cds[f.aux_off + i + 1].y;
    uint bs = f.block_size, nb = f.nbits, n = f.n_values;
    uint max_id = (1u << f.id_len) - 1;
    uint refs = ((f.aec_flags & DG_AEC_PP) && (c.y % f.rsi) == 0) ? 1u : 0u;
    device uint* o = codes + f.codes_off;
    uint s0 = c.y * bs;
    BitR r = bitr(raw, f.data_off, f.data_len, c.x);
    uint id = r.read(f.id_len);
    if (id == 0) {
        uint se = r.read(1);
        if (refs && s0 < n) o[s0] = r.read(nb);
        if (se) {
            for (uint k = refs; k < bs;) {
                uint m = min(r.read_fs(), 90u);
                uint beta = se_beta(m);
                uint d1 = m - beta * (beta + 1) / 2;
                if ((k & 1) == 0) { if (s0 + k < n) o[s0 + k] = beta - d1; k++; }
                if (s0 + k < n) o[s0 + k] = d1;
                k++;
            }
        } else {
            uint e = min(next_block * bs, n);
            for (uint k = s0 + refs; k < e; k++) o[k] = 0;
        }
    } else if (id == max_id) {
        for (uint k = 0; k < bs; k++) { uint v = r.read(nb); if (s0 + k < n) o[s0 + k] = v; }
    } else {
        uint kk = id - 1;
        if (refs && s0 < n) o[s0] = r.read(nb);
        uint m = bs - refs;
        BitR lsb = r;
        lsb.skip_fs(m);           // LSB block starts after the m FS codewords
        uint base = s0 + refs;
        for (uint k = 0; k < m; k++) {
            uint v = (r.read_fs() << kk) | lsb.read(kk);
            if (base + k < n) o[base + k] = v;
        }
    }
}

kernel void aec_post(constant DgField* fields [[buffer(0)]],
                     device uint* codes [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    constant DgField& f = fields[gid.y];
    if (f.kind != DG_KIND_CCSDS || (f.aec_flags & DG_AEC_PP) == 0) return;
    uint len = f.rsi * f.block_size;
    uint s0 = gid.x * len;
    if (s0 >= f.n_values) return;
    uint s1 = min(s0 + len, f.n_values);
    uint xmax = f.nbits == 32 ? 0xFFFFFFFFu : ((1u << f.nbits) - 1);
    uint med = xmax / 2 + 1;
    device uint* o = codes + f.codes_off;
    uint x = o[s0];
    for (uint k = s0 + 1; k < s1; k++) {
        uint d = o[k];
        uint half_d = (d >> 1) + (d & 1);
        uint mask = (x & med) ? xmax : 0u;
        if (half_d <= (mask ^ x)) x += (d >> 1) ^ (~((d & 1) - 1));
        else x = mask ^ d;
        o[k] = x;
    }
}
