/*
 * DirectGeoIO — device ABI shared by the Metal (MSL) kernels, the CUDA kernels and the Rust
 * host (crates/directgeoio-core/src/plan.rs mirrors these structs with #[repr(C)]).
 *
 * All offsets are relative to the batch buffers built by the planner:
 *   raw    : GRIB2 payload bytes DMA'd from storage (each range 16-byte aligned, padded)
 *   codes  : uint32 integer codes for CCSDS/complex fields (complex fields first)
 *   rank   : uint32 per 32-bit bitmap word (exclusive popcount prefix sums)
 *   aux    : CCSDS CDS table (uint2) or complex-packing group tables
 *   out    : output tensor (float32 or float16), one row of n_points per field
 */
#ifndef DGIO_SHARED_H
#define DGIO_SHARED_H

#if defined(__METAL_VERSION__)
typedef uint dg_u32;
typedef int dg_i32;
#else
#include <stdint.h>
typedef uint32_t dg_u32;
typedef int32_t dg_i32;
#endif

#define DG_KIND_SIMPLE 0u
#define DG_KIND_CCSDS 1u
#define DG_KIND_COMPLEX 2u

#define DG_AEC_PP 1u
#define DG_AEC_PAD_RSI 2u

#define DG_NONE 0xFFFFFFFFu

#define DG_SCAN_TILE 1024u
#define DG_SCAN_THREADS 256u

/* error codes written to DgStatus.error */
#define DG_ERR_TRUNCATED 1u
#define DG_ERR_GROUPS 2u

typedef struct DgField {
    dg_u32 kind, nbits, n_points, n_values;
    dg_u32 data_off, data_len, bitmap_off, rank_off; /* bytes in raw; words in rank */
    dg_u32 out_off, codes_off, aux_off, aux_cap;     /* elements */
    float ref_value;
    dg_i32 e;
    float dscale_hi, dscale_lo; /* 10^-D (ecCodes codes_power) as a float-float pair */
    /* Template 5.42 */
    dg_u32 block_size, rsi, id_len, aec_flags;
    /* Template 5.2 / 5.3 */
    dg_u32 ngroups, ref_gw, bits_gw, ref_gl;
    dg_u32 len_inc, last_gl, bits_gl, sd_order;
    dg_u32 sd_octets, goff_refs, goff_widths, goff_lens; /* byte offsets from data_off */
    dg_u32 goff_vals, scan2, field_index, missing_mgmt;
    dg_u32 mm_off, tile_off, pad2, pad3; /* xraw/mvalid offset; complex tile-index offset */
} DgField; /* 160 bytes */

typedef struct DgParams {
    dg_u32 n_fields;
    dg_u32 out_half; /* 1 = write float16 */
    dg_u32 exact;    /* 1 = float-float dequantisation reproducing ecCodes float64 rounding */
    dg_u32 n;        /* element count for scan kernels */
    float missing;   /* value for bitmap-masked points (NaN by default) */
    dg_u32 inclusive;
    dg_u32 max_x;
    dg_u32 pad1;
} DgParams; /* 32 bytes */

typedef struct DgStatus {
    dg_u32 count; /* CCSDS: number of CDS entries incl. sentinel */
    dg_u32 error;
    dg_u32 base1; /* complex: first-scan prefix before this field's segment */
    dg_u32 pad;
} DgStatus;

#endif
