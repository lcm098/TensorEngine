#ifndef __CUDA_KERNEL__
#define __CUDA_KERNEL__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024
#define MATMUL_TILE 16

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

void runDoubleOp(void (*kernel)(const double *, const double *, double *, int),
                  const double *h_a, const double *h_b, double *h_c, int n);

__global__ void addKernelD(const double *a, const double *b, double *c, int n);
__global__ void subKernelD(const double *a, const double *b, double *c, int n);
__global__ void mulKernelD(const double *a, const double *b, double *c, int n);
__global__ void divKernelD(const double *a, const double *b, double *c, int n);

// ---------------------------------------------------------------------------
// Broadcast kernel type alias — all broadcast kernels share this signature
// ---------------------------------------------------------------------------
typedef void (*BroadcastKernelFn)(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size);

// ---------------------------------------------------------------------------
// Broadcast kernels
// ---------------------------------------------------------------------------
__global__ void addBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size);

__global__ void subBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size);

__global__ void mulBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size);

__global__ void divBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size);

__global__ void modBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size);

// ---------------------------------------------------------------------------
// Generic broadcast launcher — pass the desired kernel as first argument
// ---------------------------------------------------------------------------
void runBroadcastDoubleOp(
    BroadcastKernelFn kernel,
    const double *h_t1, const double *h_t2, double *h_out,
    int *shape1, size_t ndim1, size_t pad1,
    int *shape2, size_t ndim2, size_t pad2,
    int *out_shape, size_t out_ndim, size_t out_size);



__global__ void matmulKernelD(const double *A, const double *B, double *C,
                               int M, int K, int cols);

void runMatMulOp(const double *h_A, const double *h_B, double *h_C,
                  int M, int K, int cols);


void runTransposeDoubleOp(const double *h_in, double *h_out, size_t rows, size_t cols);
__global__ void transposeKernelD(const double *in, double *out, size_t rows, size_t cols);

void runTransposeNDDoubleOp(
    const double *h_in, double *h_out,
    const int *h_in_shape, const size_t *h_in_strides,
    const int *h_out_shape, const int *h_axes,
    size_t ndim, size_t total_size);
__global__ void transposeNDKernelD(
    const double *in, double *out,
    const int *in_shape, const size_t *in_strides,
    const int *out_shape, const int *axes,
    size_t ndim, size_t total_size);

void runSliceNDDoubleOp(
    const double *h_in, size_t in_size,
    double *h_out, size_t out_size,
    const int *h_starts, const int *h_steps,
    const size_t *h_in_strides,
    const int *h_out_shape,
    size_t ndim, size_t base_offset);
__global__ void sliceNDKernelD(
    const double *in, double *out,
    const int *starts, const int *steps,
    const size_t *in_strides,
    const int *out_shape,
    size_t ndim, size_t out_size, size_t base_offset);

#endif