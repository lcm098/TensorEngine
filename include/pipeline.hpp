#ifndef TENSOR_ENGINE_PIPELINE_HPP
#define TENSOR_ENGINE_PIPELINE_HPP

#include "./common.hpp"
#include "./utils.hpp"
#include "./linear.hpp"
#include "../__cuda__/include/kernel.cuh"

typedef struct {
    Layer **layers; // array of layer configs, in order
    size_t depth;   // number of layers
} __pipeLine__;

#ifdef __cplusplus
extern "C" {
#endif

// Variadic: pass any number of Layer* (typically Linear_Lay(...) results),
// terminated by a trailing NULL sentinel.
__pipeLine__* build(Layer *first, ...);
void print_pipeline(__pipeLine__ *pipe);
void free_pipeline(__pipeLine__ *p);

#ifdef __cplusplus
}
#endif
#endif