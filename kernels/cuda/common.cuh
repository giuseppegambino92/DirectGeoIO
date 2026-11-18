// DirectGeoIO CUDA common helpers — 1:1 port of kernels/metal/common.h.
// Build with --fmad=false and WITHOUT --use_fast_math so dequantisation stays bit-exact.
#pragma once
#include <cstdint>
#include "../dgio_shared.h"

typedef uint32_t uint;
typedef uint64_t ulong;

__device__ __forceinline__ uint bswap32(uint x) { return __byte_perm(x, 0, 0x0123); }

// MSB-first bit reader; window() has >= 33 valid bits.
struct BitR {
    const uint* w;
    ulong pos;
    ulong limit;
    __device__ __forceinline__ ulong window() const {
        uint i = (uint)(pos >> 5);
        ulong v = ((ulong)bswap32(w[i]) << 32) | (ulong)bswap32(w[i + 1]);
        return v << (pos & 31);
    }
    __device__ __forceinline__ uint read(uint n) {
        if (n == 0) return 0;
        uint v = (uint)(window() >> (64 - n));
        pos += n;
        return v;
    }
    __device__ __forceinline__ uint read_fs() {
        uint acc = 0;
        for (;;) {
            ulong v = window();
            uint lz = (uint)__clzll((long long)v);
            if (lz < 33) { pos += lz + 1; return acc + lz; }
            acc += 32; pos += 32;
            if (pos >= limit) return acc;
        }
    }
    __device__ __forceinline__ bool skip_fs(uint m) {
        while (m > 0) {
            if (pos >= limit) return false;
            ulong v = window() & 0xFFFFFFFF80000000ull;
            uint c = (uint)__popcll(v);
            if (c < m) { m -= c; pos += 33; }
            else {
                while (m > 0) { uint lz = (uint)__clzll((long long)v); pos += lz + 1; v <<= (lz + 1); m--; }
            }
        }
        return true;
    }
};

__device__ __forceinline__ BitR bitr(const uint* raw, uint byte_off, uint byte_len, ulong bitpos) {
    BitR r;
    r.w = raw + (byte_off >> 2);
    r.pos = bitpos + (byte_off & 3) * 8;
    r.limit = (ulong)byte_len * 8 + (byte_off & 3) * 8 + 64;
    return r;
}

__device__ __forceinline__ uint read32_at(const uint* raw, uint byte_off, ulong bit) {
    ulong p = bit + (byte_off & 3) * 8;
    const uint* w = raw + (byte_off >> 2);
    uint i = (uint)(p >> 5);
    ulong v = ((ulong)bswap32(w[i]) << 32) | (ulong)bswap32(w[i + 1]);
    return (uint)((v << (p & 31)) >> 32);
}

// Register-streamed reader for the sequential CCSDS boundary walk.
struct Stream {
    const uint* w;
    uint widx, have, mis;
    ulong buf;
    __device__ __forceinline__ void refill() {
        while (have <= 32) { buf |= (ulong)bswap32(w[widx++]) << (32 - have); have += 32; }
    }
    __device__ __forceinline__ void consume(uint n) { buf = n >= 64 ? 0 : (buf << n); have -= n; }
    __device__ __forceinline__ ulong pos() const { return (ulong)widx * 32 - have - mis; }
    __device__ __forceinline__ void seek(ulong rel) {
        ulong a = rel + mis;
        widx = (uint)(a >> 5); have = 0; buf = 0;
        refill();
        consume((uint)(a & 31));
    }
    __device__ __forceinline__ void skip(uint n) {
        if (n < have) { consume(n); refill(); } else seek(pos() + n);
    }
    __device__ __forceinline__ uint read(uint n) {
        if (n == 0) return 0;
        refill();
        uint v = (uint)(buf >> (64 - n));
        consume(n);
        return v;
    }
    __device__ __forceinline__ uint read_fs() {
        uint acc = 0;
        for (;;) {
            refill();
            if (buf != 0) { uint lz = (uint)__clzll((long long)buf); consume(lz + 1); return acc + lz; }
            acc += have; consume(have);
        }
    }
    __device__ __forceinline__ void skip_fs(uint m) {
        for (;;) {
            refill();
            uint c = (uint)__popcll(buf);
            if (c < m) { m -= c; consume(have); continue; }
            ulong v = buf;
            uint p = 0;
            for (uint s = 32; s >= 1; s >>= 1) {
                uint cc = (uint)__popcll(v >> (64 - s));
                bool sk = cc < m;
                m = sk ? m - cc : m;
                p = sk ? p + s : p;
                v = sk ? (v << s) : v;
            }
            consume(p + 1);
            return;
        }
    }
};

__device__ __forceinline__ Stream stream(const uint* raw, uint byte_off) {
    Stream s;
    s.w = raw + (byte_off >> 2);
    s.mis = (byte_off & 3) * 8;
    s.widx = 0; s.have = 0; s.buf = 0;
    s.refill();
    s.consume(s.mis);
    return s;
}

// WMO dequantisation  Y = (X * 2^E + R) * 10^-D, float-float exact path.
__device__ __forceinline__ float2 two_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    float bb = __fsub_rn(s, a);
    float e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
    return make_float2(s, e);
}
__device__ __forceinline__ float dequant(int hi, uint lo, const DgField& f, bool exact) {
    if (!exact) {
        float x = __fadd_rn(__fmul_rn((float)hi, 65536.0f), (float)lo);
        return __fmul_rn(fmaf(x, ldexpf(1.0f, f.e), f.ref_value), f.dscale_hi);
    }
    float a = __fmul_rn((float)hi, ldexpf(1.0f, f.e + 16));
    float b = __fmul_rn((float)lo, ldexpf(1.0f, f.e));
    float2 s1 = two_sum(a, b);
    float2 s2 = two_sum(s1.x, f.ref_value);
    float lo2 = __fadd_rn(s1.y, s2.y);
    float sh = __fadd_rn(s2.x, lo2);
    float sl = __fsub_rn(lo2, __fsub_rn(sh, s2.x));
    float p = __fmul_rn(sh, f.dscale_hi);
    float pe = fmaf(sh, f.dscale_hi, -p);
    pe = fmaf(sh, f.dscale_lo, pe);
    pe = fmaf(sl, f.dscale_hi, pe);
    return __fadd_rn(p, pe);
}
