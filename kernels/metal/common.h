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
        if (pos >= limit) return 0; // never read past the section (+64-bit slack)
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
// ecCodes evaluates fl64(fl64(fl64(X*bscale) + R) * dscale) and the output is its float32
// rounding. Apple GPUs have no float64, so:
//   fast path : float-float evaluation (~2^-46 relative error) plus its exact residual; valid
//               whenever the result is not within that error of a float32 rounding boundary;
//   slow path : IEEE-754 binary64 add/mul emulated on 64-bit integers (round-to-nearest-even,
//               subnormals, infinities), producing the float32 bit pattern directly.
#define F64_SIGN 0x8000000000000000ul
#define F64_FRAC 0x000FFFFFFFFFFFFFul
#define F64_HID  0x0010000000000000ul
#define F64_INF  0x7FF0000000000000ul
#define F64_NAN  0x7FF8000000000000ul

// m has its leading 1 at bit 62; value = m * 2^(e - 1023 - 62)
inline ulong f64_round_pack(ulong s, int e, ulong m) {
    if (e >= 0x7FF) return s | F64_INF;
    if (e < 1) {
        uint sh = uint(1 - e);
        m = sh >= 63 ? ulong(m != 0) : ((m >> sh) | ulong((m << (64 - sh)) != 0));
        e = 0;
    }
    ulong rem = m & 0x3FFul, q = m >> 10;
    if (rem > 0x200ul || (rem == 0x200ul && (q & 1ul))) q++;
    ulong bits = e == 0 ? q : ((ulong(e - 1) << 52) + q);
    if ((bits >> 52) >= 0x7FFul) return s | F64_INF;
    return s | bits;
}
inline ulong f64_mul(ulong a, ulong b) {
    ulong s = (a ^ b) & F64_SIGN;
    int ea = int((a >> 52) & 0x7FF), eb = int((b >> 52) & 0x7FF);
    ulong ma = a & F64_FRAC, mb = b & F64_FRAC;
    if (ea == 0x7FF || eb == 0x7FF) {
        if ((ea == 0x7FF && ma) || (eb == 0x7FF && mb)) return F64_NAN;
        if ((ea == 0 && ma == 0) || (eb == 0 && mb == 0)) return F64_NAN; // inf * 0
        return s | F64_INF;
    }
    if (ea == 0) { if (ma == 0) return s; int sh = int(clz(ma)) - 11; ma <<= sh; ea = 1 - sh; } else ma |= F64_HID;
    if (eb == 0) { if (mb == 0) return s; int sh = int(clz(mb)) - 11; mb <<= sh; eb = 1 - sh; } else mb |= F64_HID;
    ulong lo = ma * mb, hi = mulhi(ma, mb); // 106-bit product, leading bit 104 or 105
    int e = ea + eb - 1023;
    uint sh;
    if ((hi >> 41) & 1ul) { e += 1; sh = 21; } else { sh = 22; }
    ulong m = (hi << sh) | (lo >> (64 - sh));
    m |= ulong((lo << sh) != 0);
    return f64_round_pack(s, e, m);
}
inline ulong f64_add(ulong a, ulong b) {
    int ea = int((a >> 52) & 0x7FF), eb = int((b >> 52) & 0x7FF);
    ulong ma = a & F64_FRAC, mb = b & F64_FRAC;
    if (ea == 0x7FF || eb == 0x7FF) {
        if ((ea == 0x7FF && ma) || (eb == 0x7FF && mb)) return F64_NAN;
        if (ea == 0x7FF && eb == 0x7FF && ((a ^ b) & F64_SIGN)) return F64_NAN;
        return ea == 0x7FF ? a : b;
    }
    if ((a << 1) == 0) return (b << 1) == 0 ? (a & b) : b;
    if ((b << 1) == 0) return a;
    if (ea == 0) ea = 1; else ma |= F64_HID;
    if (eb == 0) eb = 1; else mb |= F64_HID;
    ulong sa = a & F64_SIGN, sb = b & F64_SIGN;
    if (eb > ea || (eb == ea && mb > ma)) { // |a| >= |b|
        int te = ea; ea = eb; eb = te; ulong tm = ma; ma = mb; mb = tm; ulong ts = sa; sa = sb; sb = ts;
    }
    ma <<= 9; mb <<= 9; // leading bit at 61
    uint d = uint(ea - eb);
    mb = d >= 63 ? ulong(mb != 0) : ((mb >> d) | ulong(d && (mb << (64 - d)) != 0));
    ulong m = sa == sb ? ma + mb : ma - mb;
    if (m == 0) return 0; // exact cancellation: +0 in round-to-nearest
    int shift = int(clz(m)) - 1; // bring the leading bit to 62
    if (shift > 0) m <<= shift;
    return f64_round_pack(sa, ea + 1 - shift, m);
}
inline ulong f64_from_int(long x) {
    if (x == 0) return 0;
    ulong s = x < 0 ? F64_SIGN : 0;
    ulong m = ulong(x < 0 ? -x : x);
    uint lz = uint(clz(m));
    return s | (ulong(1023 + 63 - lz) << 52) | ((m << (lz - 11)) & F64_FRAC);
}
inline ulong f32_to_f64(uint r) {
    ulong s = ulong(r >> 31) << 63;
    int e = int((r >> 23) & 0xFF);
    uint f = r & 0x7FFFFF;
    if (e == 0xFF) return s | F64_INF | (f ? (1ul << 51) : 0ul);
    if (e == 0) { if (f == 0) return s; int sh = int(clz(f)) - 8; f = (f << sh) & 0x7FFFFF; e = 1 - sh; }
    return s | (ulong(e - 127 + 1023) << 52) | (ulong(f) << 29);
}
inline uint f64_to_f32(ulong c) {
    uint s = uint(c >> 63) << 31;
    int e = int((c >> 52) & 0x7FF);
    ulong f = c & F64_FRAC;
    if (e == 0x7FF) return s | 0x7F800000u | (f ? 0x400000u : 0u);
    if (e == 0) return s; // |c| < 2^-1022 rounds to zero in float32
    ulong m = f | F64_HID;
    int e32 = e - 1023 + 127;
    uint sh = 29;
    if (e32 < 1) { sh += uint(1 - e32); e32 = 0; }
    if (sh > 54) return s;
    ulong q = m >> sh, rem = m & ((1ul << sh) - 1), hw = 1ul << (sh - 1);
    if (rem > hw || (rem == hw && (q & 1ul))) q++;
    ulong bits = e32 == 0 ? q : ((ulong(e32 - 1) << 23) + q);
    return bits >= 0x7F800000ul ? (s | 0x7F800000u) : (s | uint(bits));
}
inline float dequant_f64(int hi, uint lo, constant DgField& f) {
    ulong bscale = (ulong(f.bscale_hi) << 32) | f.bscale_lo;
    ulong dscale = (ulong(f.dscale_hi_bits) << 32) | f.dscale_lo_bits;
    ulong x = f64_from_int(long(hi) * 65536 + long(lo));
    ulong y = f64_mul(f64_add(f64_mul(x, bscale), f32_to_f64(as_type<uint>(f.ref_value))), dscale);
    return as_type<float>(f64_to_f32(y));
}

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
    if (f.exact_slow) return dequant_f64(hi, lo, f);
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
    float r = p + pe;
    float t = pe - (r - p);                 // exact residual of the final rounding
    float ar = fabs(r);
    // error bound of the float-float evaluation (relative to the un-cancelled magnitudes)
    float bound = (fabs(a) + fabs(b) + fabs(f.ref_value)) * fabs(f.dscale_hi) * 0x1p-40f;
    float half_ulp = as_type<float>(as_type<uint>(ar) & 0x7F800000u) * 0x1p-24f;
    bool safe = isfinite(r) && ar >= 0x1p-100f && fabs(fabs(t) - half_ulp) > bound && fabs(t) < half_ulp;
    // zero test on the bit pattern: GPU float compares flush subnormals to zero
    bool exact_zero = hi == 0 && lo == 0 && (as_type<uint>(f.ref_value) << 1) == 0;
    if (!safe && !exact_zero) return dequant_f64(hi, lo, f);
    return r;
}

// read32_at bounded to a section of `byte_len` bytes (0 beyond it).
inline uint read32_lim(device const uint* raw, uint byte_off, uint byte_len, ulong bit) {
    return bit >= ulong(byte_len) * 8 ? 0u : read32_at(raw, byte_off, bit);
}

// Register-streamed MSB-first reader for long sequential walks (one load per 32 bits).
// `buf` holds `have` valid bits MSB-aligned; bits below are zero.
struct Stream {
    device const uint* w;
    uint widx;
    uint have;
    ulong buf;
    uint mis; // bit misalignment of the payload start inside w[0]
    uint lim; // words that belong to the payload; beyond them the stream reads zeros
    inline bool exhausted() const { return widx > lim + 2; }
    inline void refill() {
        while (have <= 32) {
            uint v = widx < lim ? bswap32(w[widx]) : 0u;
            widx++;
            buf |= ulong(v) << (32 - have);
            have += 32;
        }
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
            if (exhausted()) return acc;
        }
    }
    // skip m FS codewords: popcount whole buffers, then a 6-step binary select
    inline void skip_fs(uint m) {
        for (;;) {
            refill();
            uint c = uint(popcount(buf));
            if (c < m) { m -= c; consume(have); if (exhausted()) return; continue; }
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

inline Stream stream(device const uint* raw, uint byte_off, uint byte_len) {
    Stream s;
    s.w = raw + (byte_off >> 2);
    s.mis = (byte_off & 3) * 8;
    s.lim = ((byte_off & 3) + byte_len + 3) / 4;
    s.widx = 0; s.have = 0; s.buf = 0;
    s.refill();
    s.consume(s.mis);
    return s;
}
