// DirectGeoIO Metal shim: a minimal C ABI over MTLDevice, MTLIOCommandQueue (Metal fast
// resource loading: storage -> MTLBuffer without a CPU staging copy) and compute dispatch.
// Compiled with -fobjc-arc by crates/directgeoio-metal/build.rs.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <stdint.h>
#include <string.h>

static void set_err(char* err, size_t n, NSString* msg) {
    if (err && n) {
        const char* s = msg ? [msg UTF8String] : "unknown error";
        strncpy(err, s, n - 1);
        err[n - 1] = 0;
    }
}

@interface DgCtx : NSObject
@property(strong) id<MTLDevice> dev;
@property(strong) id<MTLCommandQueue> queue;
@property(strong) id<MTLIOCommandQueue> ioq;
@property(strong) id<MTLLibrary> lib;
@property(strong) NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pso;
@end
@implementation DgCtx
@end

@interface DgCmd : NSObject
@property(strong) id<MTLCommandBuffer> cb;
@property(strong) id<MTLComputeCommandEncoder> enc;
@property(weak) DgCtx* ctx;
@end
@implementation DgCmd
@end

void* dg_mtl_create(const void* metallib, size_t len, char* err, size_t errlen) {
    @autoreleasepool {
        DgCtx* c = [DgCtx new];
        c.dev = MTLCreateSystemDefaultDevice();
        if (!c.dev) { set_err(err, errlen, @"no Metal device"); return NULL; }
        c.queue = [c.dev newCommandQueue];
        NSError* e = nil;
        dispatch_data_t data = dispatch_data_create(metallib, len, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        c.lib = [c.dev newLibraryWithData:data error:&e];
        if (!c.lib) { set_err(err, errlen, e.localizedDescription); return NULL; }
        MTLIOCommandQueueDescriptor* d = [MTLIOCommandQueueDescriptor new];
        d.type = MTLIOCommandQueueTypeConcurrent;
        d.priority = MTLIOPriorityHigh;
        c.ioq = [c.dev newIOCommandQueueWithDescriptor:d error:&e];
        if (!c.ioq) { set_err(err, errlen, e.localizedDescription); return NULL; }
        c.pso = [NSMutableDictionary new];
        return (__bridge_retained void*)c;
    }
}

void dg_mtl_destroy(void* ctx) {
    if (ctx) { DgCtx* c = (__bridge_transfer DgCtx*)ctx; (void)c; }
}

size_t dg_mtl_device_name(void* ctx, char* out, size_t n) {
    DgCtx* c = (__bridge DgCtx*)ctx;
    const char* s = [c.dev.name UTF8String];
    if (out && n) { strncpy(out, s, n - 1); out[n - 1] = 0; }
    return strlen(s);
}

uint64_t dg_mtl_recommended_working_set(void* ctx) {
    DgCtx* c = (__bridge DgCtx*)ctx;
    return c.dev.recommendedMaxWorkingSetSize;
}

// ---- buffers ---------------------------------------------------------------------------
void* dg_mtl_buffer_new(void* ctx, uint64_t len, int private_storage) {
    DgCtx* c = (__bridge DgCtx*)ctx;
    MTLResourceOptions o = private_storage ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared;
    id<MTLBuffer> b = [c.dev newBufferWithLength:(len ? len : 16) options:o | MTLResourceHazardTrackingModeTracked];
    return b ? (__bridge_retained void*)b : NULL;
}
void dg_mtl_buffer_free(void* buf) {
    if (buf) { id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)buf; (void)b; }
}
void* dg_mtl_buffer_contents(void* buf) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
    return b.contents;
}
uint64_t dg_mtl_buffer_length(void* buf) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
    return b.length;
}
uint64_t dg_mtl_buffer_gpu_address(void* buf) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf;
    return b.gpuAddress;
}

// ---- MTLIO: storage -> MTLBuffer DMA ---------------------------------------------------
void* dg_mtl_io_open(void* ctx, const char* path, char* err, size_t errlen) {
    @autoreleasepool {
        DgCtx* c = (__bridge DgCtx*)ctx;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        NSError* e = nil;
        id<MTLIOFileHandle> h = [c.dev newIOFileHandleWithURL:url error:&e];
        if (!h) { set_err(err, errlen, e.localizedDescription); return NULL; }
        return (__bridge_retained void*)h;
    }
}
void dg_mtl_io_close(void* fh) {
    if (fh) { id<MTLIOFileHandle> h = (__bridge_transfer id<MTLIOFileHandle>)fh; (void)h; }
}

// Load n ranges into dst; blocks until the I/O command buffer completes.
// `max_chunk` splits large ranges into several concurrent load commands (0 = no split).
int dg_mtl_io_load(void* ctx, void* fh, void* dst, const uint64_t* file_off, const uint64_t* buf_off,
                   const uint64_t* len, size_t n, uint64_t max_chunk, double* seconds, char* err, size_t errlen) {
    @autoreleasepool {
        DgCtx* c = (__bridge DgCtx*)ctx;
        id<MTLIOFileHandle> h = (__bridge id<MTLIOFileHandle>)fh;
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)dst;
        id<MTLIOCommandBuffer> cb = [c.ioq commandBuffer];
        for (size_t i = 0; i < n; i++) {
            uint64_t done = 0;
            while (done < len[i]) {
                uint64_t sz = len[i] - done;
                if (max_chunk && sz > max_chunk) sz = max_chunk;
                [cb loadBuffer:b offset:buf_off[i] + done size:sz sourceHandle:h sourceHandleOffset:file_off[i] + done];
                done += sz;
            }
        }
        CFTimeInterval t0 = CACurrentMediaTime();
        [cb commit];
        [cb waitUntilCompleted];
        if (seconds) *seconds = CACurrentMediaTime() - t0;
        if (cb.status != MTLIOStatusComplete) {
            set_err(err, errlen, cb.error ? cb.error.localizedDescription : @"MTLIO load failed");
            return -1;
        }
        return 0;
    }
}

// ---- compute ---------------------------------------------------------------------------
static id<MTLComputePipelineState> pso_for(DgCtx* c, const char* name, char* err, size_t errlen) {
    NSString* k = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> p = c.pso[k];
    if (p) return p;
    id<MTLFunction> fn = [c.lib newFunctionWithName:k];
    if (!fn) { set_err(err, errlen, [NSString stringWithFormat:@"kernel %@ not found", k]); return nil; }
    NSError* e = nil;
    p = [c.dev newComputePipelineStateWithFunction:fn error:&e];
    if (!p) { set_err(err, errlen, e.localizedDescription); return nil; }
    c.pso[k] = p;
    return p;
}

void* dg_mtl_cmd_begin(void* ctx) {
    DgCtx* c = (__bridge DgCtx*)ctx;
    DgCmd* m = [DgCmd new];
    m.ctx = c;
    m.cb = [c.queue commandBuffer];
    m.enc = [m.cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    return (__bridge_retained void*)m;
}

// mode 0: grid = (gx,gy,gz) threads, threadgroup width tgx (0 = auto)
// mode 1: grid = (gx,gy,gz) threadgroups of tgx threads
int dg_mtl_dispatch(void* cmd, const char* kernel, void* const* bufs, const uint64_t* offs, uint32_t nbufs,
                    const void* bytes, uint32_t bytes_len, uint32_t bytes_index, uint32_t gx, uint32_t gy,
                    uint32_t gz, uint32_t tgx, int mode, char* err, size_t errlen) {
    DgCmd* m = (__bridge DgCmd*)cmd;
    id<MTLComputePipelineState> p = pso_for(m.ctx, kernel, err, errlen);
    if (!p) return -1;
    [m.enc setComputePipelineState:p];
    for (uint32_t i = 0; i < nbufs; i++) {
        if (bufs[i]) [m.enc setBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:offs ? offs[i] : 0 atIndex:i];
    }
    if (bytes && bytes_len) [m.enc setBytes:bytes length:bytes_len atIndex:bytes_index];
    if (gx == 0 || gy == 0 || gz == 0) return 0;
    NSUInteger w = tgx ? tgx : MIN(p.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    if (mode == 0) {
        [m.enc dispatchThreads:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    } else {
        [m.enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    }
    return 0;
}

// Commit, wait, and report GPU execution time (GPUEndTime - GPUStartTime).
int dg_mtl_cmd_end_wait(void* cmd, double* gpu_seconds, char* err, size_t errlen) {
    DgCmd* m = (__bridge_transfer DgCmd*)cmd;
    [m.enc endEncoding];
    [m.cb commit];
    [m.cb waitUntilCompleted];
    if (gpu_seconds) *gpu_seconds = m.cb.GPUEndTime - m.cb.GPUStartTime;
    if (m.cb.status != MTLCommandBufferStatusCompleted) {
        set_err(err, errlen, m.cb.error ? m.cb.error.localizedDescription : @"command buffer failed");
        return -1;
    }
    return 0;
}
