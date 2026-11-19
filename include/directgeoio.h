/*
 * DirectGeoIO — C ABI (libdirectgeoio.dylib / libdirectgeoio.so)
 *
 * Zero-copy GRIB2 decoding to GPU-resident tensors. Suitable for direct use from C, C++ and
 * Fortran (via ISO_C_BINDING) weather/climate codes such as WRF, and from any framework that
 * speaks DLPack.
 *
 *   dgio_file*   f = dgio_open("gfs.t00z.pgrb2.0p25.f006");
 *   dgio_options o; dgio_default_options(&o);
 *   dgio_tensor* t = dgio_decode(f, NULL, 0, DGIO_BACKEND_AUTO, &o);   // all fields
 *   const float* v = (const float*)dgio_tensor_data(t);                // unified memory on Apple
 *   ...
 *   dgio_tensor_free(t); dgio_close(f);
 *
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef DIRECTGEOIO_H
#define DIRECTGEOIO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DGIO_VERSION_MAJOR 0
#define DGIO_VERSION_MINOR 1
#define DGIO_VERSION_PATCH 0

/* ---- DLPack (v0.8 ABI, legacy "dltensor" capsule) ---------------------------------- */
#ifndef DLPACK_VERSION
typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLMetal = 8,
    kDLCUDAManaged = 13,
} DLDeviceType;
typedef struct { int32_t device_type; int32_t device_id; } DLDevice;
typedef enum { kDLInt = 0, kDLUInt = 1, kDLFloat = 2, kDLBfloat = 4 } DLDataTypeCode;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
typedef struct {
    void* data; /* kDLMetal: id<MTLBuffer>; kDLCPU: host/unified pointer; kDLCUDA: device ptr */
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides; /* NULL = compact row-major */
    uint64_t byte_offset;
} DLTensor;
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;
#endif

/* ---- types ------------------------------------------------------------------------- */
typedef struct dgio_file dgio_file;
typedef struct dgio_tensor dgio_tensor;

typedef enum {
    DGIO_BACKEND_AUTO = 0,  /* Metal on Apple Silicon, CUDA when built with it, else CPU */
    DGIO_BACKEND_CPU = 1,   /* single-threaded reference decoder */
    DGIO_BACKEND_CPU_MT = 2,
    DGIO_BACKEND_METAL = 3,
    DGIO_BACKEND_CUDA = 4,
} dgio_backend;

typedef enum { DGIO_F32 = 0, DGIO_F16 = 1 } dgio_dtype;
typedef enum { DGIO_IO_AUTO = 0, DGIO_IO_SPAN = 1, DGIO_IO_RANGES = 2 } dgio_io;
typedef enum { DGIO_AEC_SCAN_AUTO = 0, DGIO_AEC_SCAN_GPU = 1, DGIO_AEC_SCAN_CPU = 2 } dgio_aec_scan;

typedef struct {
    dgio_dtype dtype;
    int32_t exact;         /* 1 (default): bit-exact float32(ecCodes float64) */
    float missing_value;   /* default NaN */
    dgio_io io;
    dgio_aec_scan aec_scan;
} dgio_options;

typedef struct {
    char short_name[16];
    char name[64];
    char units[24];
    char level_type[24];
    double level;              /* native unit of the surface (Pa, m, ...) */
    uint32_t ni, nj, n_points, n_values;
    uint16_t data_template;    /* 0 simple, 2/3 complex, 42 CCSDS */
    uint8_t bits_per_value;
    uint8_t has_bitmap;
    uint8_t discipline, category, number, level_type_code;
    uint32_t forecast_time;
    uint16_t ref_time[6];      /* Y M D h m s */
    uint16_t centre;
    uint8_t supported;         /* 0: data representation template not implemented */
    uint8_t pad;
    double lat_first, lon_first, lat_last, lon_last, di, dj; /* degrees */
    uint64_t message_offset, message_length;
} dgio_field_info;

typedef struct {
    double plan_s;     /* metadata scan + plan (CPU) */
    double io_s;       /* storage -> device */
    double compute_s;  /* decode */
    double total_s;
    uint64_t bytes_read;
    uint64_t values;
} dgio_timings;

/* ---- API --------------------------------------------------------------------------- */
const char* dgio_version(void);
/* Thread-local description of the last error (empty string if none). */
const char* dgio_last_error(void);
/* Name of the device the backend resolves to (e.g. "Apple M3 Max"). */
const char* dgio_backend_device_name(dgio_backend backend);

void dgio_default_options(dgio_options* opts);

/* Scans metadata only (Sections 0,1,3,4,5 and the Section 6 indicator); payloads are not read. */
dgio_file* dgio_open(const char* path);
void dgio_close(dgio_file* f);
size_t dgio_num_fields(const dgio_file* f);
int dgio_get_field_info(const dgio_file* f, size_t index, dgio_field_info* out);

/* Decode `n` fields (indices into the file; NULL/0 = all) into one contiguous tensor of shape
 * (n, nj, ni) when grids agree. Returns NULL on error (see dgio_last_error). */
dgio_tensor* dgio_decode(dgio_file* f, const size_t* indices, size_t n, dgio_backend backend, const dgio_options* opts);
/* Decode into caller-provided host memory (Fortran-friendly). dst_len in elements. */
int dgio_decode_into(dgio_file* f, const size_t* indices, size_t n, dgio_backend backend, const dgio_options* opts,
                     float* dst, size_t dst_len);

void dgio_tensor_free(dgio_tensor* t);
/* CPU-addressable data pointer (unified memory on Apple Silicon; NULL for CUDA device memory). */
void* dgio_tensor_data(const dgio_tensor* t);
/* Native device handle: id<MTLBuffer> (Metal), CUdeviceptr (CUDA), or host pointer (CPU). */
void* dgio_tensor_device_handle(const dgio_tensor* t);
int32_t dgio_tensor_device_type(const dgio_tensor* t); /* DLDeviceType */
dgio_dtype dgio_tensor_dtype(const dgio_tensor* t);
int32_t dgio_tensor_ndim(const dgio_tensor* t);
int dgio_tensor_shape(const dgio_tensor* t, int64_t* shape /* [ndim] */);
size_t dgio_tensor_numel(const dgio_tensor* t);
int dgio_tensor_timings(const dgio_tensor* t, dgio_timings* out);

/* Export as DLPack. device_type: kDLMetal, kDLCPU (Apple unified memory view), kDLCUDA, or 0
 * for the tensor's native device. The tensor stays valid until both it and every exported
 * DLManagedTensor have been released (the consumer calls ->deleter). */
DLManagedTensor* dgio_tensor_to_dlpack(const dgio_tensor* t, int32_t device_type);

/* PyCapsule destructor for "dltensor" capsules (CPython API resolved at runtime; the library
 * does not link against libpython). Pass its address to PyCapsule_New. */
void dgio_dlpack_capsule_destructor(void* capsule);

#ifdef __cplusplus
}
#endif
#endif /* DIRECTGEOIO_H */
