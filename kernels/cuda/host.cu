// DirectGeoIO CUDA host runtime: device management, NVIDIA GPUDirect Storage (cuFile) with
// automatic fallback to pinned-host staging, and the decode pipeline launcher.
//
// Single translation unit: the kernels are #included so no relocatable device code is needed.
// Build: nvcc -O3 --fmad=false -std=c++17 (no --use_fast_math) — see crates/directgeoio-cuda/build.rs

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "scan.cu"
#include "bitmap_expand.cu"
#include "unpack_simple.cu"
#include "decode_ccsds_aec.cu"
#include "decode_complex.cu"
#include "spatial_diff.cu"

#if __has_include(<cufile.h>)
#include <cufile.h>
#define DGIO_HAVE_CUFILE_H 1
#endif

static void set_err(char* err, size_t n, const char* fmt, const char* a = "", const char* b = "") {
    if (err && n) snprintf(err, n, fmt, a, b);
}
#define CK(call)                                                                     \
    do {                                                                             \
        cudaError_t e_ = (call);                                                     \
        if (e_ != cudaSuccess) {                                                     \
            set_err(err, errlen, "%s: %s", #call, cudaGetErrorString(e_));          \
            return -1;                                                               \
        }                                                                            \
    } while (0)

// ---------------------------------------------------------------------------------------
// cuFile (GPUDirect Storage), resolved at runtime so builds never depend on GDS being present.
// ---------------------------------------------------------------------------------------
#ifdef DGIO_HAVE_CUFILE_H
struct CuFileApi {
    bool tried = false, ok = false;
    decltype(&cuFileDriverOpen) DriverOpen = nullptr;
    decltype(&cuFileHandleRegister) HandleRegister = nullptr;
    decltype(&cuFileHandleDeregister) HandleDeregister = nullptr;
    decltype(&cuFileRead) Read = nullptr;
    decltype(&cuFileBufRegister) BufRegister = nullptr;
    decltype(&cuFileBufDeregister) BufDeregister = nullptr;
};
static CuFileApi g_cufile;

static bool cufile_init() {
    if (g_cufile.tried) return g_cufile.ok;
    g_cufile.tried = true;
    if (getenv("DGIO_DISABLE_GDS")) return false;
    // Without the nvidia-fs kernel module libcufile silently falls back to POSIX "compat mode",
    // which measured slower than our own double-buffered pinned staging (RTX 5090: 42 vs 26 ms
    // for 532 MB). Only take the cuFile path when real GPUDirect Storage is possible.
    if (access("/sys/module/nvidia_fs", F_OK) != 0 && !getenv("DGIO_FORCE_CUFILE")) return false;
    void* h = dlopen("libcufile.so.0", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libcufile.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) return false;
#define SYM(field, name) g_cufile.field = (decltype(g_cufile.field))dlsym(h, name)
    SYM(DriverOpen, "cuFileDriverOpen");
    SYM(HandleRegister, "cuFileHandleRegister");
    SYM(HandleDeregister, "cuFileHandleDeregister");
    SYM(Read, "cuFileRead");
    SYM(BufRegister, "cuFileBufRegister");
    SYM(BufDeregister, "cuFileBufDeregister");
#undef SYM
    if (!g_cufile.DriverOpen || !g_cufile.HandleRegister || !g_cufile.Read) return false;
    CUfileError_t st = g_cufile.DriverOpen();
    g_cufile.ok = (st.err == CU_FILE_SUCCESS);
    return g_cufile.ok;
}
#endif

extern "C" {

typedef struct {
    double seconds;
    int32_t used_gds;
    int32_t pad;
    uint64_t bytes;
} dgc_io_stats;

typedef struct {
    uint32_t nf, max_points, max_words, max_cds;
    uint32_t max_rsi, max_groups, rank_len, groups_len;
    uint32_t scan1_len, scan2_len, codes_len, cds_len;
    uint32_t n_bitmap, n_ccsds, n_complex, gpu_aec_scan;
    uint32_t mm_len, max_mm_values, max_values, tiles_len;
} dgc_plan;

static cudaStream_t g_stream = nullptr;


int dgc_init(int device, char* name, size_t n, char* err, size_t errlen) {
    int count = 0;
    CK(cudaGetDeviceCount(&count));
    if (device >= count) { set_err(err, errlen, "CUDA device index out of range"); return -1; }
    CK(cudaSetDevice(device));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, device));
    if (name && n) snprintf(name, n, "%s (sm_%d%d)", p.name, p.major, p.minor);
    if (!g_stream) CK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));

    return 0;
}

void* dgc_malloc(uint64_t n) {
    void* p = nullptr;
    return cudaMalloc(&p, n ? n : 16) == cudaSuccess ? p : nullptr;
}
void dgc_free(void* p) { if (p) cudaFree(p); }
void* dgc_host_alloc(uint64_t n) {
    void* p = nullptr;
    return cudaHostAlloc(&p, n ? n : 16, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
}
void dgc_host_free(void* p) { if (p) cudaFreeHost(p); }
int dgc_d2h(void* dst, const void* src, uint64_t n, char* err, size_t errlen) {
    CK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
    return 0;
}
int dgc_h2d(void* dst, const void* src, uint64_t n, char* err, size_t errlen) {
    CK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
    return 0;
}

// Storage -> device. Tries GPUDirect Storage (O_DIRECT + cuFileRead straight into device memory);
// otherwise pread()s into pinned host memory and overlaps cudaMemcpyAsync on two staging
// buffers. If `host_mirror` (pinned, >= raw size) is given the fallback stages through it and
// leaves a host copy for the CPU CCSDS boundary walk; GDS never touches host memory.
int dgc_load(const char* path, void* dev_dst, const uint64_t* file_off, const uint64_t* buf_off, const uint64_t* len,
             size_t n, int force_posix, void* host_mirror, dgc_io_stats* st, char* err, size_t errlen) {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t total = 0;
    for (size_t i = 0; i < n; i++) total += len[i];
    st->used_gds = 0;
    st->bytes = total;
#ifdef DGIO_HAVE_CUFILE_H
    if (!force_posix && !host_mirror && cufile_init()) {
        int fd = open(path, O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            CUfileDescr_t d;
            memset(&d, 0, sizeof d);
            d.handle.fd = fd;
            d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
            CUfileHandle_t fh;
            CUfileError_t s = g_cufile.HandleRegister(&fh, &d);
            if (s.err == CU_FILE_SUCCESS) {
                bool ok = true;
                for (size_t i = 0; i < n && ok; i++) {
                    uint64_t done = 0;
                    while (done < len[i]) {
                        ssize_t r = g_cufile.Read(fh, dev_dst, len[i] - done, (off_t)(file_off[i] + done), (off_t)(buf_off[i] + done));
                        if (r <= 0) { ok = false; break; }
                        done += (uint64_t)r;
                    }
                }
                g_cufile.HandleDeregister(fh);
                close(fd);
                if (ok) {
                    st->used_gds = 1;
                    st->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    return 0;
                }
            } else {
                close(fd);
            }
        }
    }
#endif
    // ---- fallback: pinned host staging ------------------------------------------------
    int fd = open(path, O_RDONLY);
    if (fd < 0) { set_err(err, errlen, "open(%s) failed", path); return -1; }
    if (host_mirror) {
        for (size_t i = 0; i < n; i++) {
            uint64_t done = 0;
            while (done < len[i]) {
                ssize_t r = pread(fd, (char*)host_mirror + buf_off[i] + done, len[i] - done, (off_t)(file_off[i] + done));
                if (r <= 0) { close(fd); set_err(err, errlen, "pread failed"); return -1; }
                done += (uint64_t)r;
            }
            CK(cudaMemcpyAsync((char*)dev_dst + buf_off[i], (char*)host_mirror + buf_off[i], len[i], cudaMemcpyHostToDevice, g_stream));
        }
    } else {
        const uint64_t CH = 8ull << 20;
        char* stage[2] = {(char*)dgc_host_alloc(CH), (char*)dgc_host_alloc(CH)};
        cudaEvent_t ev[2];
        cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming);
        bool used[2] = {false, false};
        int k = 0;
        for (size_t i = 0; i < n; i++) {
            for (uint64_t done = 0; done < len[i];) {
                uint64_t sz = len[i] - done < CH ? len[i] - done : CH;
                if (used[k]) cudaEventSynchronize(ev[k]);
                uint64_t got = 0;
                while (got < sz) {
                    ssize_t r = pread(fd, stage[k] + got, sz - got, (off_t)(file_off[i] + done + got));
                    if (r <= 0) { close(fd); set_err(err, errlen, "pread failed"); return -1; }
                    got += (uint64_t)r;
                }
                CK(cudaMemcpyAsync((char*)dev_dst + buf_off[i] + done, stage[k], sz, cudaMemcpyHostToDevice, g_stream));
                cudaEventRecord(ev[k], g_stream);
                used[k] = true;
                done += sz;
                k ^= 1;
            }
        }
        cudaStreamSynchronize(g_stream);
        cudaEventDestroy(ev[0]);
        cudaEventDestroy(ev[1]);
        dgc_host_free(stage[0]);
        dgc_host_free(stage[1]);
    }
    close(fd);
    CK(cudaStreamSynchronize(g_stream));
    st->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

static int scan(uint* data, uint n, bool inclusive, uint* aux, char* err, size_t errlen) {
    if (n == 0) return 0;
    std::vector<uint> lv{n};
    for (uint m = n; m > DG_SCAN_TILE;) { m = (m + DG_SCAN_TILE - 1) / DG_SCAN_TILE; lv.push_back(m); }
    std::vector<uint*> lvl(lv.size() + 1);
    lvl[0] = data;
    uint* cur = aux;
    for (size_t l = 0; l < lv.size(); l++) { lvl[l + 1] = cur; cur += (lv[l] + DG_SCAN_TILE - 1) / DG_SCAN_TILE; }
    for (size_t l = 0; l < lv.size(); l++) {
        DgParams p{};
        p.n = lv[l];
        p.inclusive = (l == 0 && inclusive) ? 1u : 0u;
        uint tiles = (lv[l] + DG_SCAN_TILE - 1) / DG_SCAN_TILE;
        scan_tiles<<<tiles, DG_SCAN_THREADS, 0, g_stream>>>(lvl[l], lvl[l + 1], p);
    }
    for (int l = (int)lv.size() - 2; l >= 0; l--) {
        DgParams p{};
        p.n = lv[l];
        scan_add<<<(lv[l] + 255) / 256, 256, 0, g_stream>>>(lvl[l], lvl[l + 1], p);
    }
    CK(cudaGetLastError());
    return 0;
}

static uint64_t scan_aux_elems(uint n) {
    uint64_t t = 16;
    for (uint m = n ? n : 1;;) {
        uint tiles = (m + DG_SCAN_TILE - 1) / DG_SCAN_TILE;
        t += tiles;
        if (m <= DG_SCAN_TILE) break;
        m = tiles;
    }
    return t;
}

// Run the decode pipeline. `cds_host`/`status_host` carry a CPU boundary walk when
// gpu_aec_scan == 0 (uploaded before decoding).
int dgc_decode(const dgc_plan* P, const DgField* fields_host, const void* raw_dev, void* out_dev, const DgParams* params,
               const uint2* cds_host, const DgStatus* status_host, DgStatus* status_out, double* gpu_seconds, char* err,
               size_t errlen) {
    const uint nf = P->nf;
    auto alloc = [&](size_t bytes) -> void* {
        void* p = nullptr;
        return cudaMallocAsync(&p, bytes ? bytes : 16, g_stream) == cudaSuccess ? p : nullptr;
    };
    DgField* fields = (DgField*)alloc(sizeof(DgField) * nf);
    DgStatus* status = (DgStatus*)alloc(sizeof(DgStatus) * nf);
    uint* rank = (uint*)alloc(4ull * P->rank_len + 16);
    uint* codes = (uint*)alloc(4ull * P->codes_len + 16);
    uint2* cds = (uint2*)alloc(8ull * P->cds_len + 16);
    uint* gref = (uint*)alloc(4ull * P->groups_len + 16);
    uint* gwid = (uint*)alloc(4ull * P->groups_len + 16);
    uint* gscan = (uint*)alloc(8ull * P->groups_len + 16);
    uint* tiles = (uint*)alloc(4ull * P->tiles_len + 16);
    uint2* agg = (uint2*)alloc(8ull * P->tiles_len + 16);
    uint* xraw = (uint*)alloc(4ull * P->mm_len + 16);
    uint* mvalid = (uint*)alloc(4ull * P->mm_len + 16);
    uint aux_n = P->rank_len;
    if (P->mm_len > aux_n) aux_n = P->mm_len;
    if (P->groups_len * 2 > aux_n) aux_n = P->groups_len * 2;
    uint* aux = (uint*)alloc(4 * scan_aux_elems(aux_n));
    if (!fields || !status || !rank || !codes || !cds || !gref || !gwid || !gscan || !aux || !xraw || !mvalid || !tiles || !agg) {
        set_err(err, errlen, "cudaMallocAsync failed");
        return -1;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, g_stream);
    CK(cudaMemcpyAsync(fields, fields_host, sizeof(DgField) * nf, cudaMemcpyHostToDevice, g_stream));
    if (status_host) CK(cudaMemcpyAsync(status, status_host, sizeof(DgStatus) * nf, cudaMemcpyHostToDevice, g_stream));
    else CK(cudaMemsetAsync(status, 0, sizeof(DgStatus) * nf, g_stream));
    if (cds_host && P->n_ccsds) CK(cudaMemcpyAsync(cds, cds_host, 8ull * P->cds_len, cudaMemcpyHostToDevice, g_stream));

    const uint* raw = (const uint*)raw_dev;
    const uint T = 256;
    auto g2 = [&](uint x) { return dim3((x + T - 1) / T, nf, 1); };
    if (P->n_bitmap) {
        bitmap_popcount<<<g2(P->max_words), T, 0, g_stream>>>(raw, fields, rank);
        if (scan(rank, P->rank_len, false, aux, err, errlen)) return -1;
    }
    if (P->n_ccsds) {
        if (P->gpu_aec_scan) aec_scan<<<(nf + 31) / 32, 32, 0, g_stream>>>(raw, fields, cds, status, nf);
        aec_decode<<<g2(P->max_cds), T, 0, g_stream>>>(raw, fields, cds, status, codes);
        aec_post<<<dim3((P->max_rsi + 63) / 64, nf, 1), 64, 0, g_stream>>>(fields, codes);
    }
    if (P->n_complex) {
        DgParams pg = *params;
        pg.n = P->groups_len;
        cplx_groups<<<g2(P->max_groups), T, 0, g_stream>>>(raw, fields, gref, gwid, gscan, pg, status);
        if (scan(gscan, P->groups_len * 2, false, aux, err, errlen)) return -1;
        cplx_tile_index<<<g2(P->max_groups), T, 0, g_stream>>>(fields, gscan, tiles, pg, status);
        cplx_unpack<<<dim3((P->max_values + 1023) / 1024, nf, 1), 256, 0, g_stream>>>(raw, fields, gref, gwid, gscan, codes, status, pg, xraw, mvalid, tiles);
        if (P->mm_len) {
            if (scan(mvalid, P->mm_len, true, aux, err, errlen)) return -1;
            cplx_compact<<<g2(P->max_mm_values), T, 0, g_stream>>>(raw, fields, xraw, mvalid, codes);
        }
        if (P->scan1_len) {
            dim3 tg((P->max_values + 1023) / 1024, nf, 1);
            sd_tile_reduce<<<tg, 256, 0, g_stream>>>(fields, (const uint4*)codes, agg);
            sd_carry<<<(nf + 63) / 64, 64, 0, g_stream>>>(fields, agg, nf);
            sd_apply<<<tg, 256, 0, g_stream>>>(fields, (uint4*)codes, agg);
        }
    }
    unpack_dequant_expand<<<g2(P->max_points), T, 0, g_stream>>>(raw, fields, rank, codes, status, out_dev, *params, mvalid);
    CK(cudaGetLastError());
    cudaEventRecord(e1, g_stream);
    CK(cudaMemcpyAsync(status_out, status, sizeof(DgStatus) * nf, cudaMemcpyDeviceToHost, g_stream));
    for (void* p : {(void*)fields, (void*)status, (void*)rank, (void*)codes, (void*)cds, (void*)gref, (void*)gwid,
                    (void*)gscan, (void*)aux, (void*)xraw, (void*)mvalid, (void*)tiles, (void*)agg})
        cudaFreeAsync(p, g_stream);
    CK(cudaStreamSynchronize(g_stream));
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    if (gpu_seconds) *gpu_seconds = ms * 1e-3;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 0;
}

}  // extern "C"
