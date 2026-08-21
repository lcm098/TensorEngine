#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "../include/kernel.cuh"
#include "cuda_runtime.h"


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
    if (idx < n) {
        // Guard against division by zero: result is set to 0.0f in that case.
        c[idx] = (b[idx] != 0.0f) ? (a[idx] / b[idx]) : 0.0f;
    }
}

__global__ void addBroadcastKernelD(
    const double *t1, const double *t2, double *out,
    const int *shape1, const size_t *strides1, size_t ndim1, size_t pad1,
    const int *shape2, const size_t *strides2, size_t ndim2, size_t pad2,
    const int *out_shape, size_t out_ndim,
    size_t out_size)
{
    size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= out_size) return;

    // Decompose linear index into a multi-dim index over out_shape
    size_t remainder = linear;
    size_t idx[8]; // adjust MAX_DIMS if your tensors can exceed 8 dims
    for (size_t d = out_ndim; d-- > 0; ) {
        idx[d] = remainder % (size_t) out_shape[d];
        remainder /= (size_t) out_shape[d];
    }

    size_t off1 = 0;
    for (size_t d = 0; d < ndim1; d++) {
        size_t out_dim_index = idx[d + pad1];
        size_t dim_size = (size_t) shape1[d];
        size_t use_index = (dim_size == 1) ? 0 : out_dim_index;
        off1 += use_index * strides1[d];
    }

    size_t off2 = 0;
    for (size_t d = 0; d < ndim2; d++) {
        size_t out_dim_index = idx[d + pad2];
        size_t dim_size = (size_t) shape2[d];
        size_t use_index = (dim_size == 1) ? 0 : out_dim_index;
        off2 += use_index * strides2[d];
    }

    out[linear] = t1[off1] + t2[off2];
}

static size_t* host_strides(int *shape, size_t ndim) {
    size_t *strides = (size_t*) malloc(sizeof(size_t) * ndim);
    size_t acc = 1;
    for (size_t i = ndim; i-- > 0; ) {
        strides[i] = acc;
        acc *= (size_t) shape[i];
    }
    return strides;
}

void runBroadcastDoubleOp(
    const double *h_t1, const double *h_t2, double *h_out,
    int *shape1, size_t ndim1, size_t pad1,
    int *shape2, size_t ndim2, size_t pad2,
    int *out_shape, size_t out_ndim,
    size_t out_size)
{
    size_t *strides1 = host_strides(shape1, ndim1);
    size_t *strides2 = host_strides(shape2, ndim2);

    // Derive element counts from the shapes so we know how many bytes to copy.
    size_t t1_size = 1;
    for (size_t i = 0; i < ndim1; i++) t1_size *= (size_t)shape1[i];
    size_t t2_size = 1;
    for (size_t i = 0; i < ndim2; i++) t2_size *= (size_t)shape2[i];

    int    *d_shape1, *d_shape2, *d_out_shape;
    size_t *d_strides1, *d_strides2;
    double *d_t1, *d_t2, *d_out;          // ← input tensors on device

    CUDA_CHECK(cudaMalloc(&d_t1,       sizeof(double) * t1_size));
    CUDA_CHECK(cudaMalloc(&d_t2,       sizeof(double) * t2_size));
    CUDA_CHECK(cudaMalloc(&d_shape1,   sizeof(int)    * ndim1));
    CUDA_CHECK(cudaMalloc(&d_shape2,   sizeof(int)    * ndim2));
    CUDA_CHECK(cudaMalloc(&d_out_shape,sizeof(int)    * out_ndim));
    CUDA_CHECK(cudaMalloc(&d_strides1, sizeof(size_t) * ndim1));
    CUDA_CHECK(cudaMalloc(&d_strides2, sizeof(size_t) * ndim2));
    CUDA_CHECK(cudaMalloc(&d_out,      sizeof(double) * out_size));

    // Copy host → device for both input tensors  ← the missing step
    CUDA_CHECK(cudaMemcpy(d_t1,       h_t1,    sizeof(double) * t1_size,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t2,       h_t2,    sizeof(double) * t2_size,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape1,   shape1,  sizeof(int)    * ndim1,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape2,   shape2,  sizeof(int)    * ndim2,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_shape,out_shape,sizeof(int)   * out_ndim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides1, strides1,sizeof(size_t) * ndim1,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strides2, strides2,sizeof(size_t) * ndim2,    cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (int)((out_size + threads - 1) / threads);

    addBroadcastKernelD<<<blocks, threads>>>(
        d_t1, d_t2, d_out,                 // ← device pointers now
        d_shape1,   d_strides1, ndim1, pad1,
        d_shape2,   d_strides2, ndim2, pad2,
        d_out_shape, out_ndim,
        out_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(double) * out_size, cudaMemcpyDeviceToHost));

    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_shape1);
    cudaFree(d_shape2);
    cudaFree(d_out_shape);
    cudaFree(d_strides1);
    cudaFree(d_strides2);
    cudaFree(d_out);

    free(strides1);
    free(strides2);
}

void runDoubleOp(void (*kernel)(const double *, const double *, double *, int),
                  const double *h_a, const double *h_b, double *h_c, int n) {
    double *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(double);

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

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}