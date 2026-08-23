#ifndef TENSOR_ENGINE_PIPELINE_HPP
#define TENSOR_ENGINE_PIPELINE_HPP

#include "./common.hpp"
#include "./utils.hpp"
#include "../__cuda__/include/kernel.cuh"

typedef struct {
    TensorEngine **layers; // array of layer output tensors, in order
    size_t depth;          // number of layers
} __pipeLine__;

#ifdef __cplusplus
extern "C" {
#endif

// Variadic: pass any number of TensorEngine* (typically Linear_Lay(...) results),
// terminated by a trailing NULL sentinel.

__pipeLine__* build(TensorEngine *first, ...);
void print_pipeline(__pipeLine__ *pipe);
void free_pipeline(__pipeLine__ *p);

#ifdef __cplusplus
}
#endif
#endif