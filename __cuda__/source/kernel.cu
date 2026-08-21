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