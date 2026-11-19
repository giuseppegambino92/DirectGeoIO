/* Reference CPU baseline: ECMWF ecCodes (+ libaec) decoding GRIB2 to a contiguous float32 array.
 * End-to-end from the file on disk. Modes:
 *   st      single thread, codes_handle_new_from_file + codes_get_double_array (-> float32)
 *   st-f32  single thread, codes_get_float_array (ecCodes native float decode)
 *   mt N    read file once, N pthreads decode messages via codes_handle_new_from_message
 * Output: one JSON object per run on stdout.  Build: see Makefile (bench target). */
#include <eccodes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static double pct(double* v, int n, double p) {
    int k = (int)(p / 100.0 * n + 0.999999) - 1;
    if (k < 0) k = 0;
    if (k >= n) k = n - 1;
    return v[k];
}
/* drop the file from the page cache (Linux, no root needed) */
static void evict(const char* path) {
#ifdef __linux__
    int fd = open(path, O_RDONLY);
    if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
#else
    (void)path;
    fprintf(stderr, "cold mode needs Linux (posix_fadvise)\n");
    exit(2);
#endif
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct { size_t off, len, out_off, n; } msg_t;

static size_t index_messages(const unsigned char* b, size_t len, msg_t* m, size_t cap) {
    size_t n = 0, p = 0;
    while (p + 16 <= len && n < cap) {
        if (memcmp(b + p, "GRIB", 4) != 0) { p++; continue; }
        size_t L = 0;
        for (int i = 8; i < 16; i++) L = (L << 8) | b[p + i];
        m[n].off = p; m[n].len = L; n++;
        p += L;
    }
    return n;
}

typedef struct { const unsigned char* buf; msg_t* msgs; size_t nmsg; float* out; int* next; pthread_mutex_t* mu; } job_t;

static void* worker(void* a) {
    job_t* j = (job_t*)a;
    double* tmp = NULL; size_t cap = 0;
    for (;;) {
        pthread_mutex_lock(j->mu);
        int i = (*j->next)++;
        pthread_mutex_unlock(j->mu);
        if ((size_t)i >= j->nmsg) break;
        codes_handle* h = codes_handle_new_from_message(NULL, j->buf + j->msgs[i].off, j->msgs[i].len);
        size_t n = 0;
        codes_get_size(h, "values", &n);
        if (n > cap) { free(tmp); tmp = malloc(n * sizeof(double)); cap = n; }
        codes_get_double_array(h, "values", tmp, &n);
        float* o = j->out + j->msgs[i].out_off;
        for (size_t k = 0; k < n; k++) o[k] = (float)tmp[k];
        codes_handle_delete(h);
    }
    free(tmp);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s st|st-f32|mt FILE [threads] [repeat]\n", argv[0]); return 2; }
    const char* mode = argv[1];
    const char* path = argv[2];
    int threads = argc > 3 ? atoi(argv[3]) : 1;
    int repeat = argc > 4 ? atoi(argv[4]) : 3;
    int cold = argc > 5 && !strcmp(argv[5], "cold");
    double best = 1e30;
    double samples[1024];
    int ns = 0;
    size_t values = 0, bytes = 0;
    for (int r = 0; r < repeat; r++) {
        if (cold) evict(path);
        double t0 = now();
        values = 0;
        if (!strcmp(mode, "st") || !strcmp(mode, "st-f32")) {
            int f32 = !strcmp(mode, "st-f32");
            FILE* f = fopen(path, "rb");
            int err = 0;
            codes_handle* h;
            size_t cap = 0; double* tmp = NULL; float* out = NULL; size_t ocap = 0;
            while ((h = codes_handle_new_from_file(NULL, f, PRODUCT_GRIB, &err)) != NULL) {
                size_t n = 0;
                codes_get_size(h, "values", &n);
                if (values + n > ocap) { ocap = (values + n) * 2; out = realloc(out, ocap * sizeof(float)); }
                if (f32) {
                    codes_get_float_array(h, "values", out + values, &n);
                } else {
                    if (n > cap) { free(tmp); tmp = malloc(n * sizeof(double)); cap = n; }
                    codes_get_double_array(h, "values", tmp, &n);
                    for (size_t k = 0; k < n; k++) out[values + k] = (float)tmp[k];
                }
                values += n;
                codes_handle_delete(h);
            }
            fseek(f, 0, SEEK_END); bytes = ftell(f);
            fclose(f); free(tmp); free(out);
        } else {
            FILE* f = fopen(path, "rb");
            fseek(f, 0, SEEK_END); bytes = ftell(f); fseek(f, 0, SEEK_SET);
            unsigned char* buf = malloc(bytes);
            if (fread(buf, 1, bytes, f) != bytes) { perror("fread"); return 1; }
            fclose(f);
            msg_t msgs[4096];
            size_t nmsg = index_messages(buf, bytes, msgs, 4096);
            for (size_t i = 0; i < nmsg; i++) {
                codes_handle* h = codes_handle_new_from_message(NULL, buf + msgs[i].off, msgs[i].len);
                size_t n = 0; codes_get_size(h, "values", &n);
                msgs[i].out_off = values; msgs[i].n = n; values += n;
                codes_handle_delete(h);
            }
            float* out = malloc(values * sizeof(float));
            int next = 0;
            pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
            job_t j = {buf, msgs, nmsg, out, &next, &mu};
            pthread_t th[256];
            for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, &j);
            for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
            free(out); free(buf);
        }
        double dt = now() - t0;
        if (dt < best) best = dt;
        if (ns < 1024) samples[ns++] = dt;
    }
    const char* base = strrchr(path, '/');
    qsort(samples, ns, sizeof(double), cmpd);
    printf("{\"backend\":\"eccodes-%s\",\"file\":\"%s\",\"threads\":%d,\"total_s\":%.6f,\"median_s\":%.6f,"
           "\"p10_s\":%.6f,\"p90_s\":%.6f,\"n\":%d,\"cold\":%s,\"bytes\":%zu,\"values\":%zu}\n",
           mode, base ? base + 1 : path, threads, best, pct(samples, ns, 50), pct(samples, ns, 10), pct(samples, ns, 90), ns,
           cold ? "true" : "false", bytes, values);
    return 0;
}
