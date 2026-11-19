// Validation kernel for the emulated binary64 arithmetic used by the exact dequantisation
// slow path (exercised by crates/directgeoio-metal/tests/softf64.rs against native CPU f64).
#include "common.h"

kernel void softf64_selftest(device const ulong* a [[buffer(0)]],
                             device const ulong* b [[buffer(1)]],
                             device ulong* out_add [[buffer(2)]],
                             device ulong* out_mul [[buffer(3)]],
                             device uint* out_f32 [[buffer(4)]],
                             device ulong* out_from_f32 [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
    out_add[i] = f64_add(a[i], b[i]);
    out_mul[i] = f64_mul(a[i], b[i]);
    out_f32[i] = f64_to_f32(a[i]);
    out_from_f32[i] = f32_to_f64(uint(b[i]));
}
