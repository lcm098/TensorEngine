#ifndef TENSOR_ENGINE_RANDOM_HPP
#define TENSOR_ENGINE_RANDOM_HPP

#include "common.hpp"
#include "utils.hpp"
#include "../__cuda__/include/kernel.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// Seeds the global PRNG used by all random_* / rand_* functions in this file.
// Call once at program start for reproducible runs; if never called, the PRNG
// self-seeds from the current time on first use.
void random_seed(unsigned int seed);

// Shuffles a tensor's elements in place (Fisher-Yates), treating the tensor
// as a flat 1D sequence of t->size elements regardless of its actual shape.
void random_shuffle(TensorEngine *t);

// Returns a new tensor of the given shape, uniformly distributed in [low, high).
TensorEngine* rand_f64(f64 low, f64 high, int shape[], bool __GPU__);

// Returns a new tensor of the given shape, drawn from the standard normal
// distribution N(0, 1).
TensorEngine* random_randn(int shape[], bool __GPU__);

// Randomly selects `n` elements from `src` (treated as flat), with or without
// replacement, returning a new 1D tensor of length n.
TensorEngine* random_choice(TensorEngine *src, int n, bool replace, bool __GPU__);

// Returns a new tensor of the given shape, drawn from N(mean, stddev).
TensorEngine* random_normal(int shape[], f64 mean, f64 stddev, bool __GPU__);

// Returns a new tensor of the given shape, uniformly distributed in [low, high).
// Functionally identical to rand_f64; provided as the conventionally-named
// counterpart to random_normal.
TensorEngine* random_uniform(int shape[], f64 low, f64 high, bool __GPU__);

// Returns a new tensor of the given shape, drawn from Binomial(n_trials, p).
TensorEngine* random_binomial(int shape[], int n_trials, f64 p, bool __GPU__);

// Returns a new tensor of the given shape, drawn from Poisson(lambda).
TensorEngine* random_poisson(int shape[], f64 lambda, bool __GPU__);

#ifdef __cplusplus
}
#endif
#endif