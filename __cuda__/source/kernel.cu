#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <math.h>          // fmod
#include "../include/kernel.cuh"
#include "cuda_runtime.h"

// ---------------------------------------------------------------------------
// Element-wise kernels
// ---------------------------------------------------------------------------
__global__ void addKernelD(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void subKernelD(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

__global__ void mulKernelD(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void divKernelD(const double *a, const double *b, double *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = (b[idx] != 0.0) ? (a[idx] / b[idx]) : 0.0;
}

// ---------------------------------------------------------------------------
// Shared index-decomposition helper (device-side, inlined by the compiler)
// ---------------------------------------------------------------------------
__device__ __forceinline__
static void decompose(size_t linear, const int *out_shape, size_t out_ndim,
                      size_t *idx)
{
    size_t rem = linear;
    for (size_t d = out_ndim; d-- > 0; ) {
        idx[d] = rem % (size_t)out_shape[d];
        rem    /= (size_t)out_shape[d];
    }
}

__device__ __forceinline__
static size_t broadcast_offset(const size_t *idx,
                                const int    *shape,  size_t ndim,
                                const size_t *strides, size_t pad)
{
    size_t off = 0;
    for (size_t d = 0; d < ndim; d++) {
        size_t use = ((size_t)shape[d] == 1) ? 0 : idx[d + pad];
        off += use * strides[d];
    }
    return off;
}

// ---------------------------------------------------------------------------
// Broadcast kernels — identical structure, one line of arithmetic differs
// ---------------------------------------------------------------------------
#define BROADCAST_PREAMBLE                                                    \
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;           \
    if (linear >= out_size) return;                                           \
    size_t idx[8];                                                            \
    decompose(linear, out_shape, out_ndim, idx);                              \
    size_t off1 = broadcast_offset(idx, shape1, ndim1, strides1, pad1);      \
    size_t off2 = broadcast_offset(idx, shape2, ndim2, strides2, pad2);

__global__ void addBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size)
{
    BROADCAST_PREAMBLE
    out[linear] = t1[off1] + t2[off2];
}

__global__ void subBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size)
{
    BROADCAST_PREAMBLE
    out[linear] = t1[off1] - t2[off2];
}

__global__ void mulBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size)
{
    BROADCAST_PREAMBLE
    out[linear] = t1[off1] * t2[off2];
}

__global__ void divBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size)
{
    BROADCAST_PREAMBLE
    out[linear] = (t2[off2] != 0.0) ? (t1[off1] / t2[off2]) : 0.0;
}

__global__ void modBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim, size_t out_size)
{
    BROADCAST_PREAMBLE
    out[linear] = (t2[off2] != 0.0) ? fmod(t1[off1], t2[off2]) : 0.0;
}

#undef BROADCAST_PREAMBLE

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static size_t* host_strides(int *shape, size_t ndim) {
    size_t *s = (size_t*) malloc(sizeof(size_t) * ndim);
    size_t acc = 1;
    for (size_t i = ndim; i-- > 0; ) {
        s[i] = acc;
        acc *= (size_t)shape[i];
    }
    return s;
}

// ---------------------------------------------------------------------------
// Generic broadcast launcher
// ---------------------------------------------------------------------------
void runBroadcastDoubleOp(
    BroadcastKernelFn kernel,
    const double *h_t1, const double *h_t2, double *h_out,
    int *shape1, size_t ndim1, size_t pad1,
    int *shape2, size_t ndim2, size_t pad2,
    int *out_shape, size_t out_ndim, size_t out_size)
{
    size_t *strides1 = host_strides(shape1, ndim1);
    size_t *strides2 = host_strides(shape2, ndim2);

    size_t t1_size = 1;
    for (size_t i = 0; i < ndim1; i++) t1_size *= (size_t)shape1[i];
    size_t t2_size = 1;
    for (size_t i = 0; i < ndim2; i++) t2_size *= (size_t)shape2[i];

    int    *d_shape1, *d_shape2, *d_out_shape;
    size_t *d_strides1, *d_strides2;
    double *d_t1, *d_t2, *d_out;

    CUDA_CHECK(cudaMalloc(&d_t1,        sizeof(double) * t1_size));
    CUDA_CHECK(cudaMalloc(&d_t2,        sizeof(double) * t2_size));
    CUDA_CHECK(cudaMalloc(&d_shape1,    sizeof(int)    * ndim1));
    CUDA_CHECK(cudaMalloc(&d_shape2,    sizeof(int)    * ndim2));
    CUDA_CHECK(cudaMalloc(&d_out_shape, sizeof(int)    * out_ndim));
    CUDA_CHECK(cudaMalloc(&d_strides1,  sizeof(size_t) * ndim1));
    CUDA_CHECK(cudaMalloc(&d_strides2,  sizeof(size_t) * ndim2));
    CUDA_CHECK(cudaMalloc(&d_out,       sizeof(double) * out_size));

    CUDA_CHECK(cudaMemcpy(d_t1,        h_t1,     sizeof(double) * t1_size,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t2,        h_t2,     sizeof(double) * t2_size,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape1,    shape1,   sizeof(int)    * ndim1,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape2,    shape2,   sizeof(int)    * ndim2,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_shape, out_shape,sizeof(int)    * out_ndim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides1,  strides1, sizeof(size_t) * ndim1,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides2,  strides2, sizeof(size_t) * ndim2,    cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (int)((out_size + threads - 1) / threads);

    kernel<<<blocks, threads>>>(
        d_t1, d_t2, d_out,
        d_shape1, d_strides1, ndim1, pad1,
        d_shape2, d_strides2, ndim2, pad2,
        d_out_shape, out_ndim, out_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(double) * out_size, cudaMemcpyDeviceToHost));

    cudaFree(d_t1);  cudaFree(d_t2);
    cudaFree(d_shape1);  cudaFree(d_shape2);  cudaFree(d_out_shape);
    cudaFree(d_strides1);  cudaFree(d_strides2);
    cudaFree(d_out);
    free(strides1);  free(strides2);
}

// ---------------------------------------------------------------------------
// Non-broadcast launcher (unchanged)
// ---------------------------------------------------------------------------
void runDoubleOp(void (*kernel)(const double *, const double *, double *, int),
                  const double *h_a, const double *h_b, double *h_c, int n) {
    double *d_a, *d_b, *d_c;
    size_t bytes = (size_t)n * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_a);  cudaFree(d_b);  cudaFree(d_c);
}

__global__ void matmulKernelD(const double *A, const double *B, double *C,
                               int M, int K, int cols)
{
    __shared__ double tileA[MATMUL_TILE][MATMUL_TILE];
    __shared__ double tileB[MATMUL_TILE][MATMUL_TILE];

    int row = blockIdx.y * MATMUL_TILE + threadIdx.y;
    int col = blockIdx.x * MATMUL_TILE + threadIdx.x;
    double acc = 0.0;

    for (int t = 0; t < (K + MATMUL_TILE - 1) / MATMUL_TILE; t++) {
        int a_col = t * MATMUL_TILE + threadIdx.x;
        int b_row = t * MATMUL_TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        tileB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < cols) ? B[b_row * cols + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < MATMUL_TILE; k++)
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < cols)
        C[row * cols + col] = acc;
}

void runMatMulOp(const double *h_A, const double *h_B, double *h_C,
                  int M, int K, int cols)
{
    double *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(double) * K * cols));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(double) * M * cols));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(double) * M * K,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(double) * K * cols, cudaMemcpyHostToDevice));

    dim3 threads(MATMUL_TILE, MATMUL_TILE);
    dim3 blocks((cols + MATMUL_TILE - 1) / MATMUL_TILE,
                (M    + MATMUL_TILE - 1) / MATMUL_TILE);

    matmulKernelD<<<blocks, threads>>>(d_A, d_B, d_C, M, K, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(double) * M * cols, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

__global__ void transposeKernelD(const double *in, double *out, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * cols;
    if (idx >= total) return;

    size_t i = idx / cols; // row in input
    size_t j = idx % cols; // col in input

    out[j * rows + i] = in[i * cols + j];
}

void runTransposeDoubleOp(const double *h_in, double *h_out, size_t rows, size_t cols) {
    size_t total = rows * cols;

    double *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(double) * total));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * total));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(double) * total, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    transposeKernelD<<<blocks, threads>>>(d_in, d_out, rows, cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(double) * total, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void transposeNDKernelD(
    const double *in, double *out,
    const int *in_shape, const size_t *in_strides,
    const int *out_shape, const int *axes,
    size_t ndim, size_t total_size)
{
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total_size) return;

    size_t rem = linear;
    size_t in_offset = 0;

    for (size_t d = ndim; d-- > 0; ) {
        size_t coord = rem % (size_t)out_shape[d];
        rem /= (size_t)out_shape[d];
        in_offset += coord * in_strides[axes[d]];
    }

    out[linear] = in[in_offset];
}

void runTransposeNDDoubleOp(
    const double *h_in, double *h_out,
    const int *h_in_shape, const size_t *h_in_strides,
    const int *h_out_shape, const int *h_axes,
    size_t ndim, size_t total_size)
{
    double *d_in, *d_out;
    int *d_in_shape, *d_out_shape, *d_axes;
    size_t *d_in_strides;

    CUDA_CHECK(cudaMalloc(&d_in, sizeof(double) * total_size));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * total_size));
    CUDA_CHECK(cudaMalloc(&d_in_shape, sizeof(int) * ndim));
    CUDA_CHECK(cudaMalloc(&d_out_shape, sizeof(int) * ndim));
    CUDA_CHECK(cudaMalloc(&d_axes, sizeof(int) * ndim));
    CUDA_CHECK(cudaMalloc(&d_in_strides, sizeof(size_t) * ndim));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(double) * total_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_shape, h_in_shape, sizeof(int) * ndim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_shape, h_out_shape, sizeof(int) * ndim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_axes, h_axes, sizeof(int) * ndim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_strides, h_in_strides, sizeof(size_t) * ndim, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (int)((total_size + threads - 1) / threads);

    transposeNDKernelD<<<blocks, threads>>>(d_in, d_out, d_in_shape, d_in_strides, d_out_shape, d_axes, ndim, total_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(double) * total_size, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_in_shape);
    cudaFree(d_out_shape);
    cudaFree(d_axes);
    cudaFree(d_in_strides);
}

__global__ void sliceNDKernelD(
    const double *in, double *out,
    const int *starts, const int *steps,
    const size_t *in_strides,
    const int *out_shape,
    size_t ndim, size_t out_size, size_t base_offset)
{
    size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= out_size) return;

    size_t rem = linear;
    size_t in_offset = base_offset;

    for (size_t d = ndim; d-- > 0; ) {
        size_t coord = rem % (size_t)out_shape[d];
        rem /= (size_t)out_shape[d];
        int in_coord = starts[d] + (int)coord * steps[d];
        in_offset += (size_t)in_coord * in_strides[d];
    }

    out[linear] = in[in_offset];
}

void runSliceNDDoubleOp(
    const double *h_in, size_t in_size,
    double *h_out, size_t out_size,
    const int *h_starts, const int *h_steps,
    const size_t *h_in_strides,
    const int *h_out_shape,
    size_t ndim, size_t base_offset)
{
    double *d_in, *d_out;
    int *d_starts, *d_steps, *d_out_shape;
    size_t *d_in_strides;

    CUDA_CHECK(cudaMalloc(&d_in, sizeof(double) * in_size));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * out_size));
    CUDA_CHECK(cudaMalloc(&d_starts, sizeof(int) * ndim));
    CUDA_CHECK(cudaMalloc(&d_steps, sizeof(int) * ndim));
    CUDA_CHECK(cudaMalloc(&d_out_shape, sizeof(int) * ndim));
    CUDA_CHECK(cudaMalloc(&d_in_strides, sizeof(size_t) * ndim));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(double) * in_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_starts, h_starts, sizeof(int) * ndim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_steps, h_steps, sizeof(int) * ndim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_shape, h_out_shape, sizeof(int) * ndim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_strides, h_in_strides, sizeof(size_t) * ndim, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (int)((out_size + threads - 1) / threads);

    sliceNDKernelD<<<blocks, threads>>>(
        d_in, d_out, d_starts, d_steps, d_in_strides, d_out_shape, ndim, out_size, base_offset
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(double) * out_size, cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_starts);
    cudaFree(d_steps);
    cudaFree(d_out_shape);
    cudaFree(d_in_strides);
}