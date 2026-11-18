/* C consumer of include/directgeoio.h: decode every field of a GRIB2 file on the default
 * backend, print field metadata, timings and a checksum. */
#include <directgeoio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.grib2\n", argv[0]); return 2; }
    printf("DirectGeoIO %s on %s\n", dgio_version(), dgio_backend_device_name(DGIO_BACKEND_AUTO));
    dgio_file* f = dgio_open(argv[1]);
    if (!f) { fprintf(stderr, "open: %s\n", dgio_last_error()); return 1; }
    size_t n = dgio_num_fields(f);
    for (size_t i = 0; i < n; i++) {
        dgio_field_info fi;
        dgio_get_field_info(f, i, &fi);
        printf("  #%zu %-6s %-18s %8.0f  5.%-2u %2ub %ux%u%s [%s]\n", i, fi.short_name, fi.level_type, fi.level,
               fi.data_template, fi.bits_per_value, fi.ni, fi.nj, fi.has_bitmap ? " bitmap" : "", fi.units);
    }
    dgio_options o;
    dgio_default_options(&o);
    dgio_tensor* t = dgio_decode(f, NULL, 0, DGIO_BACKEND_AUTO, &o);
    if (!t) { fprintf(stderr, "decode: %s\n", dgio_last_error()); return 1; }
    int64_t shape[4] = {0};
    dgio_tensor_shape(t, shape);
    const float* v = (const float*)dgio_tensor_data(t);
    size_t numel = dgio_tensor_numel(t), nan = 0;
    double sum = 0;
    for (size_t i = 0; i < numel; i++) { if (isnan(v[i])) nan++; else sum += v[i]; }
    dgio_timings tm;
    dgio_tensor_timings(t, &tm);
    printf("tensor ndim=%d shape=(%lld,%lld,%lld) device_type=%d masked=%zu sum=%.6e\n", dgio_tensor_ndim(t),
           (long long)shape[0], (long long)shape[1], (long long)shape[2], dgio_tensor_device_type(t), nan, sum);
    printf("timings: plan %.3f ms, io %.3f ms, decode %.3f ms, total %.3f ms\n", tm.plan_s * 1e3, tm.io_s * 1e3,
           tm.compute_s * 1e3, tm.total_s * 1e3);
    DLManagedTensor* dl = dgio_tensor_to_dlpack(t, kDLMetal);
    if (dl) { printf("dlpack: device %d data %p ndim %d\n", dl->dl_tensor.device.device_type, dl->dl_tensor.data, dl->dl_tensor.ndim); dl->deleter(dl); }
    dgio_tensor_free(t);
    dgio_close(f);
    return 0;
}
