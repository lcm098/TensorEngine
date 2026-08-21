#ifndef __CUDA_KERNEL__
#define __CUDA_KERNEL__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024

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

#endif