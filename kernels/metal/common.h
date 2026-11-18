// DirectGeoIO Metal common helpers: big-endian bit reader, Blelloch constants, dequantisation.
#pragma once
#include <metal_stdlib>
#include "../dgio_shared.h"
using namespace metal;

inline uint bswap32(uint x) {
    return (x << 24) | ((x & 0xFF00u) << 8) | ((x >> 8) & 0xFF00u) | (x >> 24);
}

// MSB-first bit reader over a 4-byte aligned big-endian stream. window() has >= 33 valid bits.
struct BitR {
    device const uint* w;
    ulong pos;
    ulong limit;
    inline ulong window() const {
        uint i = uint(pos >> 5);
        ulong v = (ulong(bswap32(w[i])) << 32) | ulong(bswap32(w[i + 1]));
        return v << (pos & 31);
    }
    inline uint read(uint n) {
        if (n == 0) return 0;
        uint v = uint(window() >> (64 - n));
        pos += n;
        return v;
    }
    // unary (fundamental sequence) codeword
    inline uint read_fs() {
        uint acc = 0;
        for (;;) {
            ulong v = window();
            uint lz = uint(clz(v));
            if (lz < 33) { pos += lz + 1; return acc + lz; }
            acc += 32; pos += 32;
            if (pos >= limit) return acc;
        }
    }
    // skip m FS codewords using popcount on 33-bit windows
    inline bool skip_fs(uint m) {
        while (m > 0) {
            if (pos >= limit) return false;
            ulong v = window() & 0xFFFFFFFF80000000ul;
            uint c = uint(popcount(v));
            if (c < m) { m -= c; pos += 33; }
            else {
                while (m > 0) { uint lz = uint(clz(v)); pos += lz + 1; v <<= (lz + 1); m--; }
            }
        }
        return true;
    }
};

// Reader positioned `bitpos` bits after byte `byte_off` of raw (any alignment).
inline BitR bitr(device const uint* raw, uint byte_off, uint byte_len, ulong bitpos) {
    BitR r;
    r.w = raw + (byte_off >> 2);
    r.pos = bitpos + (byte_off & 3) * 8;
    r.limit = ulong(byte_len) * 8 + (byte_off & 3) * 8 + 64;
    return r;
}

// 32 bits starting `bit` bits after byte `byte_off` (any alignment).
inline uint read32_at(device const uint* raw, uint byte_off, ulong bit) {
    ulong p = bit + (byte_off & 3) * 8;
    device const uint* w = raw + (byte_off >> 2);
    uint i = uint(p >> 5);
    ulong v = (ulong(bswap32(w[i])) << 32) | ulong(bswap32(w[i + 1]));
    return uint((v << (p & 31)) >> 32);
}

// --- WMO dequantisation  Y = (X * 2^E + R) * 10^-D ---------------------------------------
// exact: float-float evaluation reproducing float32(ecCodes float64 result). X = hi*2^16 + lo.
inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float e = (a - (s - bb)) + (b - bb);
    return float2(s, e);
}
inline float dequant(int hi, uint lo, constant DgField& f, bool exact) {
    if (!exact) {
        float x = float(hi) * 65536.0f + float(lo);
        return fma(x, ldexp(1.0f, f.e), f.ref_value) * f.dscale_hi;
    }
    float a = float(hi) * ldexp(1.0f, f.e + 16);
    float b = float(lo) * ldexp(1.0f, f.e);
    float2 s1 = two_sum(a, b);
    float2 s2 = two_sum(s1.x, f.ref_value);
    float lo2 = s1.y + s2.y;
    float sh = s2.x + lo2;
    float sl = lo2 - (sh - s2.x);
    float p = sh * f.dscale_hi;
    float pe = fma(sh, f.dscale_hi, -p);
    pe = fma(sh, f.dscale_lo, pe);
    pe = fma(sl, f.dscale_hi, pe);
    return p + pe;
}

// Register-streamed MSB-first reader for long sequential walks (one load per 32 bits).
// `buf` holds `have` valid bits MSB-aligned; bits below are zero.
struct Stream {
    device const uint* w;
    uint widx;
    uint have;
    ulong buf;
    uint mis; // bit misalignment of the payload start inside w[0]
    inline void refill() {
        while (have <= 32) { buf |= ulong(bswap32(w[widx++])) << (32 - have); have += 32; }
    }
    inline void consume(uint n) { buf = n >= 64 ? 0 : (buf << n); have -= n; }
    // relative bit position (from payload start)
    inline ulong pos() const { return ulong(widx) * 32 - have - mis; }
    inline void seek(ulong rel) {
        ulong a = rel + mis;
        widx = uint(a >> 5); have = 0; buf = 0;
        refill();
        consume(uint(a & 31));
    }
    inline void skip(uint n) {
        if (n < have) { consume(n); refill(); }
        else seek(pos() + n);
    }
    inline uint read(uint n) {
        if (n == 0) return 0;
        refill();
        uint v = uint(buf >> (64 - n));
        consume(n);
        return v;
    }
    inline uint read_fs() {
        uint acc = 0;
        for (;;) {
            refill();
            if (buf != 0) { uint lz = uint(clz(buf)); consume(lz + 1); return acc + lz; }
            acc += have; consume(have);
        }
    }
    // skip m FS codewords: popcount whole buffers, then a 6-step binary select
    inline void skip_fs(uint m) {
        for (;;) {
            refill();
            uint c = uint(popcount(buf));
            if (c < m) { m -= c; consume(have); continue; }
            ulong v = buf;
            uint p = 0;
            for (uint s = 32; s >= 1; s >>= 1) {
                uint cc = uint(popcount(v >> (64 - s)));
                bool skipit = cc < m;
                m = skipit ? m - cc : m;
                p = skipit ? p + s : p;
                v = skipit ? (v << s) : v;
            }
            consume(p + 1);
            return;
        }
    }
};

inline Stream stream(device const uint* raw, uint byte_off) {
    Stream s;
    s.w = raw + (byte_off >> 2);
    s.mis = (byte_off & 3) * 8;
    s.widx = 0; s.have = 0; s.buf = 0;
    s.refill();
    s.consume(s.mis);
    return s;
}
